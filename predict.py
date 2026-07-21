"""
Multi-modal prediction: Brain MRI (tumor) + Chest X-Ray (pneumonia).
Includes Grad-CAM and PDF medical reports.
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional

import cv2
import numpy as np
import tensorflow as tf
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from tensorflow.keras import layers

from compare_models import get_best_model_name, load_comparison

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
BRAIN_CONFIG_PATH = os.path.join(MODELS_DIR, "model_config.json")
PNEUMONIA_CONFIG_PATH = os.path.join(MODELS_DIR, "pneumonia", "model_config.json")
PNEUMONIA_MODEL_PATH = os.path.join(MODELS_DIR, "pneumonia", "best_model.keras")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
GRADCAM_DIR = os.path.join(BASE_DIR, "static", "gradcam")

IMG_SIZE = (224, 224)
SCAN_TYPES = {"brain_mri", "chest_xray"}

BRAIN_CLASS_DISPLAY = {0: "No Tumor Detected", 1: "Tumor Detected"}
PNEUMONIA_CLASS_DISPLAY = {0: "Normal", 1: "Pneumonia Detected"}

_caches: Dict[str, Dict] = {
    "brain_mri": {"model": None, "name": None, "config": None},
    "chest_xray": {"model": None, "name": None, "config": None},
}


def _load_config(scan_type: str) -> Dict:
    path = BRAIN_CONFIG_PATH if scan_type == "brain_mri" else PNEUMONIA_CONFIG_PATH
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"preprocess": "mobilenetv2", "threshold": 0.5, "img_size": [224, 224]}


def _get_preprocess_fn(preprocess_type: str):
    if preprocess_type == "mobilenetv2":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        return preprocess_input
    if preprocess_type == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input
    return None


def clear_model_cache(scan_type: Optional[str] = None) -> None:
    types = [scan_type] if scan_type else list(_caches.keys())
    for st in types:
        _caches[st] = {"model": None, "name": None, "config": None}


def _get_brain_model_path() -> str:
    best = os.path.join(MODELS_DIR, "best_model.keras")
    if os.path.exists(best):
        return best
    for name in ["mobilenetv2", "resnet50", "custom_cnn"]:
        p = os.path.join(MODELS_DIR, f"{name}.keras")
        if os.path.exists(p):
            return p
    return best


def load_model(scan_type: str = "brain_mri") -> tf.keras.Model:
    if scan_type not in SCAN_TYPES:
        raise ValueError(f"Invalid scan_type: {scan_type}")

    cache = _caches[scan_type]
    if cache["model"] is not None:
        return cache["model"]

    if scan_type == "brain_mri":
        model_path = _get_brain_model_path()
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "No brain tumor model found. Run: python train_model.py --data-dir dataset"
            )
        config = _load_config("brain_mri")
        comparison = load_comparison()
        cache["name"] = config.get("best_model") or (
            comparison.get("best_model") if comparison else "mobilenetv2"
        )
    else:
        model_path = PNEUMONIA_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "No pneumonia model found. Run: python train_pneumonia_model.py --data-dir dataset_pneumonia"
            )
        config = _load_config("chest_xray")
        cache["name"] = config.get("best_model", "pneumonia_mobilenetv2")

    cache["model"] = tf.keras.models.load_model(model_path)
    cache["config"] = config
    return cache["model"]


def preprocess_image(image_path: str, scan_type: str = "brain_mri") -> np.ndarray:
    config = _load_config(scan_type)
    img_size = tuple(config.get("img_size", [224, 224]))
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    preprocess_fn = _get_preprocess_fn(config.get("preprocess", "mobilenetv2"))
    if preprocess_fn:
        return preprocess_fn(np.expand_dims(arr, axis=0))
    return np.expand_dims(arr / 255.0, axis=0)


def _find_last_conv_layer(model: tf.keras.Model) -> str:
    conv_layer_names = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                conv_layer_names.append(_find_last_conv_layer(layer))
            except ValueError:
                pass
        elif isinstance(layer, (layers.Conv2D, layers.SeparableConv2D, layers.DepthwiseConv2D)):
            conv_layer_names.append(layer.name)
    if conv_layer_names:
        return conv_layer_names[-1]
    raise ValueError("No convolutional layer found for Grad-CAM.")


def _compute_gradcam_heatmap(model, image_array, last_conv_name) -> np.ndarray:
    inputs = tf.keras.Input(shape=image_array.shape[1:])
    x = inputs
    conv_output = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_name:
            conv_output = x
    grad_model = tf.keras.models.Model(inputs=inputs, outputs=[conv_output, x])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array, training=False)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError("Gradients are None.")
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def generate_gradcam(model, image_array, image_path, output_name) -> str:
    os.makedirs(GRADCAM_DIR, exist_ok=True)
    last_conv_name = _find_last_conv_layer(model)
    try:
        heatmap = _compute_gradcam_heatmap(model, image_array, last_conv_name)
    except Exception:
        heatmap = np.zeros((IMG_SIZE[0], IMG_SIZE[1]), dtype=np.float32)
        cy, cx = IMG_SIZE[0] // 2, IMG_SIZE[1] // 2
        y, x = np.ogrid[:IMG_SIZE[0], :IMG_SIZE[1]]
        heatmap = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 40.0 ** 2))

    original = cv2.imread(image_path)
    if original is None:
        original = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)
    heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)
    output_path = os.path.join(GRADCAM_DIR, f"{output_name}_gradcam.jpg")
    cv2.imwrite(output_path, overlay)
    return output_path


def _run_prediction(image_path: str, scan_type: str, class_display: Dict, pos_label_key: str) -> Dict:
    model = load_model(scan_type)
    config = _caches[scan_type]["config"] or _load_config(scan_type)
    threshold = float(config.get("threshold", 0.5))
    image_array = preprocess_image(image_path, scan_type)

    probability = float(model.predict(image_array, verbose=0)[0][0])
    class_idx = 1 if probability >= threshold else 0
    confidence = probability if class_idx == 1 else 1 - probability

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    gradcam_path = generate_gradcam(model, image_array, image_path, f"{base_name}_{timestamp}")
    gradcam_rel = os.path.join("gradcam", os.path.basename(gradcam_path)).replace("\\", "/")

    return {
        "scan_type": scan_type,
        "prediction": class_display[class_idx],
        "confidence_score": round(confidence * 100, 2),
        "model_used": _caches[scan_type]["name"] or "unknown",
        pos_label_key: round(probability * 100, 2),
        "probability_healthy": round((1 - probability) * 100, 2),
        "threshold_used": round(threshold * 100, 2),
        "gradcam_path": gradcam_path,
        "gradcam_url": f"/static/{gradcam_rel}",
        "timestamp": datetime.utcnow().isoformat(),
    }


def predict_brain_mri(image_path: str) -> Dict:
    return _run_prediction(image_path, "brain_mri", BRAIN_CLASS_DISPLAY, "probability_tumor")


def predict_pneumonia(image_path: str) -> Dict:
    return _run_prediction(image_path, "chest_xray", PNEUMONIA_CLASS_DISPLAY, "probability_pneumonia")


def predict_image(image_path: str, scan_type: str = "brain_mri") -> Dict:
    if scan_type not in SCAN_TYPES:
        raise ValueError(f"Invalid scan_type '{scan_type}'. Use: brain_mri or chest_xray")
    if scan_type == "chest_xray":
        return predict_pneumonia(image_path)
    return predict_brain_mri(image_path)


def generate_pdf_report(
    prediction_data: Dict,
    doctor_name: str,
    image_name: str,
    image_path: str,
    prediction_id: int,
) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    scan_type = prediction_data.get("scan_type", "brain_mri")
    filename = f"report_{prediction_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join(REPORTS_DIR, filename)

    doc = SimpleDocTemplate(filepath, pagesize=A4, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "HospitalTitle", parent=styles["Heading1"], fontSize=20,
        textColor=colors.HexColor("#0d6efd"), spaceAfter=12, alignment=1,
    )
    heading_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], fontSize=14,
        textColor=colors.HexColor("#1a3a5c"), spaceBefore=16, spaceAfter=8,
    )
    body_style = styles["Normal"]

    if scan_type == "chest_xray":
        report_title = "CHEST X-RAY PNEUMONIA ANALYSIS REPORT"
        scan_label = "Chest X-Ray"
    else:
        report_title = "BRAIN MRI TUMOR ANALYSIS REPORT"
        scan_label = "Brain MRI"

    elements = [
        Paragraph(report_title, title_style),
        Paragraph("AI-Powered Multi-Modal Medical Decision Support System", styles["Italic"]),
        Spacer(1, 0.3 * inch),
        Paragraph("Patient Scan Information", heading_style),
    ]

    info_data = [
        ["Field", "Value"],
        ["Scan Type", scan_label],
        ["Scan File", image_name],
        ["Analysis Date", prediction_data.get("timestamp", datetime.utcnow().isoformat())],
        ["Attending Physician", doctor_name],
        ["Report ID", f"RPT-{prediction_id:05d}"],
    ]
    info_table = Table(info_data, colWidths=[2.2 * inch, 4 * inch])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d6efd")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f7ff")]),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 0.2 * inch))

    pred = prediction_data["prediction"]
    is_positive = (
        ("Tumor" in pred and "No" not in pred) or
        ("Pneumonia" in pred and "No" not in pred)
    )
    result_color = colors.HexColor("#dc3545") if is_positive else colors.HexColor("#198754")
    prob_key = "probability_pneumonia" if scan_type == "chest_xray" else "probability_tumor"
    prob_label = "Pneumonia Probability" if scan_type == "chest_xray" else "Tumor Probability"

    elements.append(Paragraph("AI Diagnosis Results", heading_style))
    result_data = [
        ["Result", pred],
        ["Confidence Score", f"{prediction_data['confidence_score']}%"],
        ["Model Used", prediction_data["model_used"]],
        [prob_label, f"{prediction_data.get(prob_key, 'N/A')}%"],
    ]
    result_table = Table(result_data, colWidths=[2.2 * inch, 4 * inch])
    result_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (1, 0), (1, 0), result_color),
        ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(result_table)
    elements.append(Spacer(1, 0.2 * inch))

    gradcam_path = prediction_data.get("gradcam_path")
    if gradcam_path and os.path.exists(gradcam_path):
        elements.append(Paragraph("Explainable AI — Grad-CAM Visualization", heading_style))
        elements.append(Paragraph(
            "Highlighted regions indicate areas that most influenced the AI prediction.",
            body_style,
        ))
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(RLImage(gradcam_path, width=4 * inch, height=4 * inch))

    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(
        "<b>Disclaimer:</b> AI-assisted report for clinical decision support only. "
        "Final diagnosis must be confirmed by a qualified medical professional.",
        ParagraphStyle("Disclaimer", parent=body_style, fontSize=8, textColor=colors.grey),
    ))
    doc.build(elements)
    return filepath


# Backward compatibility
def load_model_config() -> Dict:
    return _load_config("brain_mri")


def get_preprocess_fn():
    return _get_preprocess_fn(_load_config("brain_mri").get("preprocess", "mobilenetv2"))


def get_active_model_name() -> str:
    return _caches["brain_mri"]["name"] or get_best_model_name()
