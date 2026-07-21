"""
Model comparison dashboard data generator.
Loads training results and produces comparison visualizations.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
COMPARISON_PATH = os.path.join(MODELS_DIR, "model_comparison.json")
PNEUMONIA_COMPARISON_PATH = os.path.join(MODELS_DIR, "pneumonia", "model_comparison.json")
PNEUMONIA_HISTORY_DIR = os.path.join(MODELS_DIR, "pneumonia", "training_history")
PNEUMONIA_CM_PLOT = os.path.join(PNEUMONIA_HISTORY_DIR, "pneumonia_confusion_matrix.png")
COMPARISON_CHART = os.path.join(MODELS_DIR, "model_comparison_chart.png")

PNEUMONIA_CLASS_LABELS = ["Normal", "Pneumonia"]
BRAIN_CLASS_LABELS = ["No Tumor", "Tumor"]


def load_comparison() -> Optional[Dict]:
    """Load model comparison JSON if it exists."""
    if not os.path.exists(COMPARISON_PATH):
        return None
    with open(COMPARISON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pneumonia_comparison() -> Optional[Dict]:
    """Load chest X-ray pneumonia model comparison JSON if it exists."""
    if not os.path.exists(PNEUMONIA_COMPARISON_PATH):
        return None
    with open(PNEUMONIA_COMPARISON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_pneumonia_metrics_for_dashboard() -> List[Dict]:
    """Return simplified pneumonia metrics for Chart.js dashboard."""
    data = load_pneumonia_comparison()
    if not data:
        return _default_pneumonia_metrics()
    return [
        {
            "name": m["model_name"],
            "accuracy": round(m["accuracy"] * 100, 2),
            "precision": round(m["precision"] * 100, 2),
            "recall": round(m["recall"] * 100, 2),
            "f1_score": round(m["f1_score"] * 100, 2),
            "auc": round(m["auc"] * 100, 2),
        }
        for m in data.get("models", [])
    ]


def _default_pneumonia_metrics() -> List[Dict]:
    """Placeholder pneumonia metrics when model hasn't been trained yet."""
    return [
        {"name": "pneumonia_mobilenetv2", "accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0, "auc": 0},
    ]


def get_model_metrics_for_dashboard() -> List[Dict]:
    """Return simplified metrics for Chart.js dashboard."""
    data = load_comparison()
    if not data:
        return _default_metrics()
    return [
        {
            "name": m["model_name"],
            "accuracy": round(m["accuracy"] * 100, 2),
            "precision": round(m["precision"] * 100, 2),
            "recall": round(m["recall"] * 100, 2),
            "f1_score": round(m["f1_score"] * 100, 2),
            "auc": round(m["auc"] * 100, 2),
        }
        for m in data.get("models", [])
    ]


def _default_metrics() -> List[Dict]:
    """Placeholder metrics when models haven't been trained yet."""
    return [
        {"name": "custom_cnn", "accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0, "auc": 0},
        {"name": "resnet50", "accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0, "auc": 0},
        {"name": "mobilenetv2", "accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0, "auc": 0},
    ]


def get_best_model_name() -> str:
    """Return the name of the best performing model."""
    data = load_comparison()
    if data:
        return data.get("best_model", "custom_cnn")
    return "custom_cnn"


def _save_confusion_matrix_plot(
    matrix: List[List[int]],
    class_labels: List[str],
    output_path: str,
    title: str,
    cmap: str = "Blues",
) -> str:
    """Render and save a confusion matrix heatmap PNG."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = np.array(matrix)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        ax=ax,
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    return output_path


def ensure_pneumonia_confusion_matrix_plot() -> None:
    """Regenerate chest X-ray confusion matrix PNGs from JSON (one per model)."""
    data = load_pneumonia_comparison()
    if not data or not data.get("models"):
        return
    os.makedirs(PNEUMONIA_HISTORY_DIR, exist_ok=True)
    for model in data["models"]:
        cm = model.get("confusion_matrix")
        if not cm:
            continue
        model_name = model.get("model_name", "pneumonia_mobilenetv2")
        plot_path = os.path.join(PNEUMONIA_HISTORY_DIR, f"{model_name}_confusion_matrix.png")
        if not os.path.exists(plot_path):
            _save_confusion_matrix_plot(
                cm,
                PNEUMONIA_CLASS_LABELS,
                plot_path,
                f"Confusion Matrix — {model_name.replace('_', ' ')} (Chest X-Ray)",
                cmap="Oranges",
            )
        # Legacy alias for older templates
        if model == data["models"][0] and not os.path.exists(PNEUMONIA_CM_PLOT):
            _save_confusion_matrix_plot(
                cm,
                PNEUMONIA_CLASS_LABELS,
                PNEUMONIA_CM_PLOT,
                "Confusion Matrix — Chest X-Ray (Pneumonia)",
                cmap="Oranges",
            )


def get_pneumonia_confusion_matrix() -> Optional[Tuple[List[List[int]], List[str]]]:
    """Return first pneumonia confusion matrix and class labels (legacy helper)."""
    data = load_pneumonia_comparison()
    if not data or not data.get("models"):
        return None
    cm = data["models"][0].get("confusion_matrix")
    if not cm:
        return None
    return cm, PNEUMONIA_CLASS_LABELS


def generate_comparison_chart() -> Optional[str]:
    """Generate bar chart comparing all models."""
    data = load_comparison()
    if not data or not data.get("models"):
        return None

    models = data["models"]
    names = [m["model_name"] for m in models]
    metrics = ["accuracy", "precision", "recall", "f1_score", "auc"]
    x = range(len(names))
    width = 0.15

    plt.figure(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        values = [m[metric] * 100 for m in models]
        offset = (i - len(metrics) / 2) * width + width / 2
        plt.bar([xi + offset for xi in x], values, width, label=metric.replace("_", " ").title())

    plt.xlabel("Model")
    plt.ylabel("Score (%)")
    plt.title("Model Performance Comparison")
    plt.xticks(list(x), names)
    plt.legend()
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(COMPARISON_CHART, dpi=120)
    plt.close()
    return COMPARISON_CHART


def print_comparison_report() -> None:
    """Print formatted comparison report to console."""
    data = load_comparison()
    if not data:
        print("No comparison data found. Run train_model.py first.")
        return

    print("\n" + "=" * 70)
    print("BRAIN TUMOR DETECTION - MODEL COMPARISON REPORT")
    print("=" * 70)
    print(f"Generated: {data.get('generated_at', 'N/A')}")
    print(f"Best Model: {data.get('best_model', 'N/A')} (by {data.get('selection_criterion', 'f1_score')})")
    print("-" * 70)

    for m in data.get("models", []):
        print(f"\n  Model: {m['model_name']}")
        print(f"    Accuracy:  {m['accuracy']:.4f}")
        print(f"    Precision: {m['precision']:.4f}")
        print(f"    Recall:    {m['recall']:.4f}")
        print(f"    F1 Score:  {m['f1_score']:.4f}")
        print(f"    AUC:       {m['auc']:.4f}")

    chart = generate_comparison_chart()
    if chart:
        print(f"\nComparison chart saved to: {chart}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print_comparison_report()
