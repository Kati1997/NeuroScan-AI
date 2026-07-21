"""
SQLite database module for the Brain Tumor Detection System.
Handles initialization, user management, predictions, and reports.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

DATABASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database")
DATABASE_PATH = os.path.join(DATABASE_DIR, "brain_tumor.db")
SCHEMA_PATH = os.path.join(DATABASE_DIR, "schema.sql")


def get_connection() -> sqlite3.Connection:
    """Create a database connection with row factory enabled."""
    os.makedirs(DATABASE_DIR, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Initialize database tables from schema file."""
    os.makedirs(DATABASE_DIR, exist_ok=True)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as schema_file:
        schema_sql = schema_file.read()
    with get_db() as conn:
        conn.executescript(schema_sql)
        _migrate_add_scan_type(conn)


def _migrate_add_scan_type(conn: sqlite3.Connection) -> None:
    """Add scan_type column to existing databases without data loss."""
    columns = [row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()]
    if "scan_type" not in columns:
        conn.execute(
            "ALTER TABLE predictions ADD COLUMN scan_type TEXT NOT NULL DEFAULT 'brain_mri'"
        )


def create_user(name: str, email: str, password_hash: str, role: str = "doctor") -> Optional[int]:
    """Register a new user. Returns user id or None if email exists."""
    try:
        with get_db() as conn:
            cursor = conn.execute(
                "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
                (name, email.lower().strip(), password_hash, role),
            )
            return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Fetch user by email address."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
        ).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Fetch user by primary key."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def create_prediction(
    user_id: int,
    image_name: str,
    image_path: str,
    prediction: str,
    confidence_score: float,
    model_used: str,
    gradcam_path: Optional[str] = None,
    scan_type: str = "brain_mri",
) -> int:
    """Store a new prediction record."""
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions
            (user_id, image_name, image_path, scan_type, prediction, confidence_score, model_used, gradcam_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, image_name, image_path, scan_type, prediction, confidence_score, model_used, gradcam_path),
        )
        return cursor.lastrowid


def create_report(prediction_id: int, report_file: str) -> int:
    """Store a generated PDF report reference."""
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO reports (prediction_id, report_file) VALUES (?, ?)",
            (prediction_id, report_file),
        )
        return cursor.lastrowid


def get_prediction_by_id(prediction_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a single prediction with doctor name."""
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT p.*, u.name AS doctor_name, u.email AS doctor_email
            FROM predictions p
            JOIN users u ON p.user_id = u.id
            WHERE p.id = ?
            """,
            (prediction_id,),
        ).fetchone()
        return dict(row) if row else None


def get_predictions_by_user(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch predictions for a specific doctor."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT p.*, u.name AS doctor_name
            FROM predictions p
            JOIN users u ON p.user_id = u.id
            WHERE p.user_id = ?
            ORDER BY p.timestamp DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]


def get_all_predictions(limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch all predictions (admin view)."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT p.*, u.name AS doctor_name
            FROM predictions p
            JOIN users u ON p.user_id = u.id
            ORDER BY p.timestamp DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


def get_reports_by_user(user_id: int) -> List[Dict[str, Any]]:
    """Fetch reports linked to a doctor's predictions."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT r.*, p.image_name, p.scan_type, p.prediction, p.confidence_score, p.model_used,
                   p.timestamp AS prediction_time, u.name AS doctor_name
            FROM reports r
            JOIN predictions p ON r.prediction_id = p.id
            JOIN users u ON p.user_id = u.id
            WHERE p.user_id = ?
            ORDER BY r.timestamp DESC
            """,
            (user_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def get_all_reports() -> List[Dict[str, Any]]:
    """Fetch all reports (admin view)."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT r.*, p.image_name, p.scan_type, p.prediction, p.confidence_score, p.model_used,
                   p.timestamp AS prediction_time, u.name AS doctor_name
            FROM reports r
            JOIN predictions p ON r.prediction_id = p.id
            JOIN users u ON p.user_id = u.id
            ORDER BY r.timestamp DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]


def get_dashboard_stats() -> Dict[str, Any]:
    """Aggregate statistics for the admin dashboard."""
    with get_db() as conn:
        total_scans = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        brain_scans = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE scan_type = 'brain_mri' OR scan_type IS NULL"
        ).fetchone()[0]
        xray_scans = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE scan_type = 'chest_xray'"
        ).fetchone()[0]
        tumor_count = conn.execute(
            """
            SELECT COUNT(*) FROM predictions
            WHERE (scan_type = 'brain_mri' OR scan_type IS NULL)
              AND prediction LIKE '%Tumor%' AND prediction NOT LIKE '%No Tumor%'
            """
        ).fetchone()[0]
        healthy_count = conn.execute(
            """
            SELECT COUNT(*) FROM predictions
            WHERE (scan_type = 'brain_mri' OR scan_type IS NULL)
              AND prediction LIKE '%No Tumor%'
            """
        ).fetchone()[0]
        pneumonia_count = conn.execute(
            """
            SELECT COUNT(*) FROM predictions
            WHERE scan_type = 'chest_xray' AND prediction LIKE '%Pneumonia%'
            """
        ).fetchone()[0]
        xray_normal_count = conn.execute(
            """
            SELECT COUNT(*) FROM predictions
            WHERE scan_type = 'chest_xray' AND prediction = 'Normal'
            """
        ).fetchone()[0]

        daily = conn.execute(
            """
            SELECT DATE(timestamp) AS day, COUNT(*) AS count
            FROM predictions
            GROUP BY DATE(timestamp)
            ORDER BY day DESC
            LIMIT 30
            """
        ).fetchall()

        monthly = conn.execute(
            """
            SELECT strftime('%Y-%m', timestamp) AS month, COUNT(*) AS count
            FROM predictions
            GROUP BY strftime('%Y-%m', timestamp)
            ORDER BY month DESC
            LIMIT 12
            """
        ).fetchall()

        model_usage = conn.execute(
            """
            SELECT model_used, COUNT(*) AS count
            FROM predictions
            GROUP BY model_used
            """
        ).fetchall()

        return {
            "total_scans": total_scans,
            "brain_scans": brain_scans,
            "xray_scans": xray_scans,
            "tumor_detections": tumor_count,
            "healthy_cases": healthy_count,
            "pneumonia_detections": pneumonia_count,
            "xray_normal_cases": xray_normal_count,
            "daily_analyses": [{"day": row[0], "count": row[1]} for row in daily],
            "monthly_analyses": [{"month": row[0], "count": row[1]} for row in monthly],
            "model_usage": [{"model": row[0], "count": row[1]} for row in model_usage],
        }


def ensure_default_admin() -> None:
    """Create default admin account if no users exist."""
    from werkzeug.security import generate_password_hash

    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count == 0:
            conn.execute(
                "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
                (
                    "System Admin",
                    "admin@hospital.com",
                    generate_password_hash("Admin@123"),
                    "admin",
                ),
            )
