"""
Main Flask application for the Brain Tumor Detection and Medical Decision Support System.
"""

import os
from functools import wraps

from dotenv import load_dotenv
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import database as db
from api import api_bp
from compare_models import (
    BRAIN_CLASS_LABELS,
    PNEUMONIA_CLASS_LABELS,
    ensure_pneumonia_confusion_matrix_plot,
    generate_comparison_chart,
    get_model_metrics_for_dashboard,
    get_pneumonia_metrics_for_dashboard,
    load_comparison,
    load_pneumonia_comparison,
)
from predict import generate_pdf_report, predict_image

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
REPORTS_FOLDER = os.path.join(BASE_DIR, "reports")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_SIZE_MB", 10)) * 1024 * 1024

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-in-production")
app.config["WTF_CSRF_ENABLED"] = True
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["PERMANENT_SESSION_LIFETIME"] = 86400  # 24 hours

csrf = CSRFProtect(app)
app.register_blueprint(api_bp)
csrf.exempt(api_bp)


@app.context_processor
def inject_model_labels():
    """Shared class labels for confusion matrix charts in all templates."""
    return {
        "brain_class_labels": BRAIN_CLASS_LABELS,
        "pneumonia_class_labels": PNEUMONIA_CLASS_LABELS,
    }


# Ensure required directories exist
for folder in [UPLOAD_FOLDER, REPORTS_FOLDER, os.path.join(BASE_DIR, "models"), os.path.join(BASE_DIR, "models", "pneumonia"), os.path.join(BASE_DIR, "static", "gradcam")]:
    os.makedirs(folder, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            flash("Admin access required.", "danger")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)

    return decorated


@app.context_processor
def inject_globals():
    return {"current_user": session.get("user_name"), "current_role": session.get("role")}


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not name or not email or not password:
            flash("All fields are required.", "danger")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
        elif password != confirm:
            flash("Passwords do not match.", "danger")
        else:
            user_id = db.create_user(name, email, generate_password_hash(password), "doctor")
            if user_id:
                flash("Registration successful! Please log in.", "success")
                return redirect(url_for("login"))
            flash("Email already registered.", "danger")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        user = db.get_user_by_email(email)
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            session["role"] = user["role"]
            session.permanent = True
            flash(f"Welcome back, {user['name']}!", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


@app.route("/dashboard")
@login_required
def dashboard():
    stats = db.get_dashboard_stats()
    model_metrics = get_model_metrics_for_dashboard()
    pneumonia_metrics = get_pneumonia_metrics_for_dashboard()
    comparison = load_comparison()
    pneumonia_comparison = load_pneumonia_comparison()
    ensure_pneumonia_confusion_matrix_plot()
    return render_template(
        "dashboard.html",
        stats=stats,
        model_metrics=model_metrics,
        pneumonia_metrics=pneumonia_metrics,
        comparison=comparison,
        pneumonia_comparison=pneumonia_comparison,
        brain_class_labels=BRAIN_CLASS_LABELS,
        pneumonia_class_labels=PNEUMONIA_CLASS_LABELS,
        best_model=comparison.get("best_model") if comparison else "Not trained",
        best_pneumonia_model=(
            pneumonia_comparison.get("best_model") if pneumonia_comparison else "Not trained"
        ),
    )


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload_mri():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file selected.", "danger")
            return redirect(request.url)

        file = request.files["file"]
        if not file.filename:
            flash("No file selected.", "danger")
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash("Invalid file type. Only image formats are accepted (PNG, JPG, JPEG, BMP, TIF).", "danger")
            return redirect(request.url)

        try:
            scan_type = request.form.get("scan_type", "brain_mri")
            if scan_type not in ("brain_mri", "chest_xray"):
                flash("Invalid scan type selected.", "danger")
                return redirect(request.url)

            filename = secure_filename(file.filename)
            base, ext = os.path.splitext(filename)
            counter = 1
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            while os.path.exists(save_path):
                filename = f"{base}_{counter}{ext}"
                save_path = os.path.join(UPLOAD_FOLDER, filename)
                counter += 1

            file.save(save_path)

            result = predict_image(save_path, scan_type=scan_type)
            scan_type = result.get("scan_type", scan_type)
            app.logger.info(
                "Upload prediction scan_type=%s model=%s result=%s user_id=%s file=%s",
                scan_type,
                result["model_used"],
                result["prediction"],
                session.get("user_id"),
                filename,
            )

            prediction_id = db.create_prediction(
                user_id=session["user_id"],
                image_name=filename,
                image_path=save_path,
                prediction=result["prediction"],
                confidence_score=result["confidence_score"],
                model_used=result["model_used"],
                gradcam_path=result["gradcam_path"],
                scan_type=scan_type,
            )

            report_path = generate_pdf_report(
                prediction_data=result,
                doctor_name=session.get("user_name", "Unknown"),
                image_name=filename,
                image_path=save_path,
                prediction_id=prediction_id,
            )
            db.create_report(prediction_id, report_path)

            label = "Brain MRI" if scan_type == "brain_mri" else "Chest X-Ray"
            flash(f"{label} analysis completed successfully.", "success")
            return redirect(url_for("prediction_result", prediction_id=prediction_id))

        except FileNotFoundError as e:
            flash(str(e), "warning")
            return redirect(request.url)
        except Exception as e:
            flash(f"Analysis failed: {str(e)}", "danger")

    return render_template("upload.html")


@app.route("/prediction/<int:prediction_id>")
@login_required
def prediction_result(prediction_id):
    prediction = db.get_prediction_by_id(prediction_id)
    if not prediction:
        flash("Prediction not found.", "danger")
        return redirect(url_for("dashboard"))

    if session.get("role") != "admin" and prediction["user_id"] != session["user_id"]:
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))

    gradcam_filename = os.path.basename(prediction["gradcam_path"]) if prediction.get("gradcam_path") else None
    return render_template("prediction_result.html", prediction=prediction, gradcam_filename=gradcam_filename)


@app.route("/predictions")
@login_required
def predictions_list():
    if session.get("role") == "admin":
        predictions = db.get_all_predictions()
    else:
        predictions = db.get_predictions_by_user(session["user_id"])
    return render_template("predictions.html", predictions=predictions)


@app.route("/reports")
@login_required
def reports_list():
    if session.get("role") == "admin":
        reports = db.get_all_reports()
    else:
        reports = db.get_reports_by_user(session["user_id"])
    return render_template("reports.html", reports=reports)


@app.route("/reports/download/<path:filename>")
@login_required
def download_report(filename):
    safe_name = secure_filename(filename)
    return send_from_directory(REPORTS_FOLDER, safe_name, as_attachment=True)


@app.route("/model-comparison")
@login_required
def model_comparison():
    comparison = load_comparison()
    pneumonia_comparison = load_pneumonia_comparison()
    ensure_pneumonia_confusion_matrix_plot()
    chart_path = generate_comparison_chart()
    model_metrics = get_model_metrics_for_dashboard()
    pneumonia_metrics = get_pneumonia_metrics_for_dashboard()
    return render_template(
        "model_comparison.html",
        comparison=comparison,
        pneumonia_comparison=pneumonia_comparison,
        brain_class_labels=BRAIN_CLASS_LABELS,
        pneumonia_class_labels=PNEUMONIA_CLASS_LABELS,
        model_metrics=model_metrics,
        pneumonia_metrics=pneumonia_metrics,
        chart_available=chart_path is not None,
    )


@app.route("/uploads/<path:filename>")
@login_required
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, secure_filename(filename))


@app.errorhandler(413)
def too_large(e):
    flash(f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024*1024)}MB.", "danger")
    return redirect(url_for("upload_mri"))


@app.route("/models/training_history/<path:filename>")
@login_required
def serve_training_plot(filename):
    history_dir = os.path.join(BASE_DIR, "models", "training_history")
    return send_from_directory(history_dir, secure_filename(filename))


@app.route("/models/pneumonia/training_history/<path:filename>")
@login_required
def serve_pneumonia_training_plot(filename):
    history_dir = os.path.join(BASE_DIR, "models", "pneumonia", "training_history")
    return send_from_directory(history_dir, secure_filename(filename))


@app.errorhandler(404)
def not_found(e):
    return render_template("home.html"), 404


def create_app():
    """Application factory for production deployment."""
    db.init_db()
    db.ensure_default_admin()
    return app


# Initialize database on import
with app.app_context():
    db.init_db()
    db.ensure_default_admin()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "development") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
