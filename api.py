"""
REST API module for the Brain Tumor Detection System.
Provides JSON endpoints for authentication, prediction, and analytics.
"""

import os
from functools import wraps
from typing import Callable

from flask import Blueprint, current_app, jsonify, request, session
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import database as db
from compare_models import get_model_metrics_for_dashboard, get_pneumonia_metrics_for_dashboard, load_comparison
from predict import generate_pdf_report, predict_image

api_bp = Blueprint("api", __name__, url_prefix="/api")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_SIZE_MB", 10)) * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def api_login_required(f: Callable):
    """Decorator for API routes requiring authentication."""

    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"success": False, "error": "Authentication required."}), 401
        return f(*args, **kwargs)

    return decorated


def api_admin_required(f: Callable):
    """Decorator for admin-only API routes."""

    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"success": False, "error": "Authentication required."}), 401
        if session.get("role") != "admin":
            return jsonify({"success": False, "error": "Admin access required."}), 403
        return f(*args, **kwargs)

    return decorated


@api_bp.route("/register", methods=["POST"])
def register():
    """
    POST /api/register
    Body: { "name", "email", "password", "role" (optional) }
    """
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip()
    password = data.get("password") or ""
    role = data.get("role", "doctor")

    if not name or not email or not password:
        return jsonify({"success": False, "error": "Name, email, and password are required."}), 400
    if len(password) < 6:
        return jsonify({"success": False, "error": "Password must be at least 6 characters."}), 400
    if role not in ("doctor", "admin"):
        role = "doctor"

    password_hash = generate_password_hash(password)
    user_id = db.create_user(name, email, password_hash, role)
    if user_id is None:
        return jsonify({"success": False, "error": "Email already registered."}), 409

    return jsonify({"success": True, "message": "Registration successful.", "user_id": user_id}), 201


@api_bp.route("/login", methods=["POST"])
def login():
    """
    POST /api/login
    Body: { "email", "password" }
    """
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"success": False, "error": "Email and password are required."}), 400

    user = db.get_user_by_email(email)
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"success": False, "error": "Invalid email or password."}), 401

    session.clear()
    session["user_id"] = user["id"]
    session["user_name"] = user["name"]
    session["role"] = user["role"]
    session.permanent = True

    return jsonify(
        {
            "success": True,
            "message": "Login successful.",
            "user": {"id": user["id"], "name": user["name"], "email": user["email"], "role": user["role"]},
        }
    )


@api_bp.route("/logout", methods=["POST"])
def logout():
    """POST /api/logout"""
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully."})


@api_bp.route("/upload", methods=["POST"])
@api_login_required
def upload():
    """
    POST /api/upload
    Form data: file (MRI image)
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"success": False, "error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file type. Allowed: PNG, JPG, JPEG, BMP, TIF."}), 400

    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > MAX_UPLOAD_BYTES:
        return jsonify({"success": False, "error": f"File exceeds {MAX_UPLOAD_BYTES // (1024*1024)}MB limit."}), 400

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)

    # Avoid overwriting: append counter if needed
    base, ext = os.path.splitext(filename)
    counter = 1
    while os.path.exists(save_path):
        filename = f"{base}_{counter}{ext}"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        counter += 1

    file.save(save_path)
    return jsonify({"success": True, "filename": filename, "path": save_path})


@api_bp.route("/predict", methods=["POST"])
@api_login_required
def predict():
    """
    POST /api/predict
    Form data: file (MRI image) OR JSON: { "image_path": "..." }
    """
    try:
        image_path = None
        image_name = None

        if "file" in request.files and request.files["file"].filename:
            file = request.files["file"]
            if not allowed_file(file.filename):
                return jsonify({"success": False, "error": "Invalid file type."}), 400
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            image_name = secure_filename(file.filename)
            image_path = os.path.join(UPLOAD_FOLDER, image_name)
            file.save(image_path)
        else:
            data = request.get_json(silent=True) or {}
            image_path = data.get("image_path")
            if image_path and os.path.exists(image_path):
                image_name = os.path.basename(image_path)
            else:
                return jsonify({"success": False, "error": "No valid image provided."}), 400

        json_data = request.get_json(silent=True) or {}
        scan_type = request.form.get("scan_type") or json_data.get("scan_type", "brain_mri")
        if scan_type not in ("brain_mri", "chest_xray"):
            scan_type = "brain_mri"

        result = predict_image(image_path, scan_type=scan_type)
        scan_type = result.get("scan_type", scan_type)
        current_app.logger.info(
            "API prediction scan_type=%s model=%s result=%s user_id=%s file=%s",
            scan_type,
            result["model_used"],
            result["prediction"],
            session.get("user_id"),
            image_name,
        )
        user_id = session["user_id"]

        prediction_id = db.create_prediction(
            user_id=user_id,
            image_name=image_name,
            image_path=image_path,
            prediction=result["prediction"],
            confidence_score=result["confidence_score"],
            model_used=result["model_used"],
            gradcam_path=result["gradcam_path"],
            scan_type=scan_type,
        )

        report_path = generate_pdf_report(
            prediction_data=result,
            doctor_name=session.get("user_name", "Unknown"),
            image_name=image_name,
            image_path=image_path,
            prediction_id=prediction_id,
        )
        db.create_report(prediction_id, report_path)

        result["prediction_id"] = prediction_id
        result["report_file"] = os.path.basename(report_path)
        result["doctor_name"] = session.get("user_name")
        return jsonify({"success": True, "data": result})

    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except Exception as e:
        return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500


@api_bp.route("/predictions", methods=["GET"])
@api_login_required
def get_predictions():
    """GET /api/predictions - List predictions for current user (all for admin)."""
    if session.get("role") == "admin":
        predictions = db.get_all_predictions()
    else:
        predictions = db.get_predictions_by_user(session["user_id"])
    return jsonify({"success": True, "count": len(predictions), "data": predictions})


@api_bp.route("/reports", methods=["GET"])
@api_login_required
def get_reports():
    """GET /api/reports - List generated PDF reports."""
    if session.get("role") == "admin":
        reports = db.get_all_reports()
    else:
        reports = db.get_reports_by_user(session["user_id"])
    return jsonify({"success": True, "count": len(reports), "data": reports})


@api_bp.route("/dashboard", methods=["GET"])
@api_login_required
def dashboard():
    """GET /api/dashboard - Analytics data for dashboard charts."""
    stats = db.get_dashboard_stats()
    model_metrics = get_model_metrics_for_dashboard()
    pneumonia_metrics = get_pneumonia_metrics_for_dashboard()
    comparison = load_comparison()

    return jsonify(
        {
            "success": True,
            "data": {
                "stats": stats,
                "model_metrics": model_metrics,
                "pneumonia_metrics": pneumonia_metrics,
                "best_model": comparison.get("best_model") if comparison else None,
            },
        }
    )


@api_bp.route("/docs", methods=["GET"])
def api_docs():
    """GET /api/docs - API documentation."""
    docs = {
        "title": "Brain Tumor Detection API",
        "version": "1.0.0",
        "endpoints": [
            {"method": "POST", "path": "/api/register", "description": "Register a new doctor/admin", "auth": False},
            {"method": "POST", "path": "/api/login", "description": "Authenticate and create session", "auth": False},
            {"method": "POST", "path": "/api/logout", "description": "End current session", "auth": True},
            {"method": "POST", "path": "/api/upload", "description": "Upload MRI image file", "auth": True},
            {"method": "POST", "path": "/api/predict", "description": "Analyze MRI and generate report", "auth": True},
            {"method": "GET", "path": "/api/predictions", "description": "List prediction history", "auth": True},
            {"method": "GET", "path": "/api/reports", "description": "List generated PDF reports", "auth": True},
            {"method": "GET", "path": "/api/dashboard", "description": "Dashboard analytics data", "auth": True},
        ],
    }
    return jsonify(docs)
