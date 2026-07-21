"""
Pneumonia Detection training — Chest X-Ray classification with MobileNetV2.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, f1_score,
    precision_score, recall_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
FINE_TUNE_EPOCHS = 12
RANDOM_SEED = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models", "pneumonia")
HISTORY_DIR = os.path.join(MODELS_DIR, "training_history")
CONFIG_PATH = os.path.join(MODELS_DIR, "model_config.json")
COMPARISON_PATH = os.path.join(MODELS_DIR, "model_comparison.json")

CLASS_LABELS = ["normal", "pneumonia"]
sns.set_theme(style="whitegrid", palette="husl", font_scale=1.05)
PLOT_PALETTE = sns.color_palette("husl", 4)


def set_seeds(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def discover_dataset(data_dir: str) -> Tuple[List[str], List[int]]:
    image_paths, labels = [], []
    class_folders = {"normal": 0, "no": 0, "healthy": 0, "pneumonia": 1, "yes": 1}
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for root in [data_dir, os.path.join(data_dir, "train"), os.path.join(data_dir, "Training")]:
        if not os.path.isdir(root):
            continue
        for folder_name in os.listdir(root):
            folder_path = os.path.join(root, folder_name)
            if not os.path.isdir(folder_path):
                continue
            label = class_folders.get(folder_name.lower())
            if label is None:
                continue
            for fname in os.listdir(folder_path):
                if os.path.splitext(fname)[1].lower() in valid_ext:
                    image_paths.append(os.path.join(folder_path, fname))
                    labels.append(label)

    if not image_paths:
        raise FileNotFoundError(f"No images in {data_dir}. Run: python download_dataset_pneumonia.py")
    return image_paths, labels


def load_raw_images(image_paths: List[str], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    images, valid_labels = [], []
    for path, label in zip(image_paths, labels):
        try:
            img = tf.keras.preprocessing.image.load_img(path, target_size=IMG_SIZE)
            images.append(tf.keras.preprocessing.image.img_to_array(img))
            valid_labels.append(label)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
    return np.array(images, dtype=np.float32), np.array(valid_labels, dtype=np.float32)


def build_mobilenetv2() -> tf.keras.Model:
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = False
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs, name="pneumonia_mobilenetv2")
    model._base_model = base  # noqa: SLF001
    return model


def find_optimal_threshold(y_true, y_prob) -> float:
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.2, 0.81, 0.02):
        f1 = f1_score(y_true, (y_prob >= thresh).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(thresh)
    return best_thresh


def evaluate_model(model, x_test, y_test, threshold=0.5) -> Dict:
    y_prob = model.predict(x_test, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "model_name": "pneumonia_mobilenetv2",
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    metrics["auc"] = float(auc(fpr, tpr))

    os.makedirs(HISTORY_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color=PLOT_PALETTE[0], linewidth=2.5, label=f"AUC = {metrics['auc']:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="#9aa0a6")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Pneumonia Detection", fontweight="bold")
    ax.legend()
    sns.despine(ax=ax)
    plt.tight_layout()
    roc_path = os.path.join(HISTORY_DIR, "pneumonia_roc.png")
    plt.savefig(roc_path, dpi=120, bbox_inches="tight")
    plt.close()

    cm = np.array(metrics["confusion_matrix"])
    model_name = metrics["model_name"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax,
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    ax.set_title(f"Confusion Matrix — {model_name.replace('_', ' ')}", fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(HISTORY_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close()
    metrics["roc_plot"] = roc_path
    metrics["confusion_matrix_plot"] = cm_path
    return metrics


def train_pneumonia_model(data_dir: str) -> Dict:
    set_seeds()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

    image_paths, labels = discover_dataset(data_dir)
    print(f"Found {len(image_paths)} images (Normal: {labels.count(0)}, Pneumonia: {labels.count(1)})")

    x_raw, y_data = load_raw_images(image_paths, labels)
    x_train_raw, x_temp, y_train, y_temp = train_test_split(
        x_raw, y_data, test_size=0.25, random_state=RANDOM_SEED, stratify=y_data
    )
    x_val_raw, x_test_raw, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.4, random_state=RANDOM_SEED, stratify=y_temp
    )

    x_train = mobilenet_preprocess(x_train_raw.copy())
    x_val = mobilenet_preprocess(x_val_raw.copy())
    x_test = mobilenet_preprocess(x_test_raw.copy())

    class_weights_arr = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight = {i: class_weights_arr[i] for i in range(len(class_weights_arr))}

    model = build_mobilenetv2()
    model.compile(optimizer=optimizers.Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])

    datagen = ImageDataGenerator(
        rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=0.15, horizontal_flip=True, fill_mode="nearest",
    )
    steps = max(1, len(x_train) // BATCH_SIZE)
    checkpoint = os.path.join(MODELS_DIR, "best_model.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint(checkpoint, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    print("\n=== Phase 1: Transfer Learning ===")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=steps, validation_data=(x_val, y_val),
        epochs=EPOCHS, class_weight=class_weight, callbacks=callbacks, verbose=1,
    )

    print("\n=== Phase 2: Fine-Tuning ===")
    base = model._base_model
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(optimizer=optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=steps, validation_data=(x_val, y_val),
        epochs=FINE_TUNE_EPOCHS, class_weight=class_weight, callbacks=callbacks, verbose=1,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    epochs_range = range(1, len(history.history["accuracy"]) + 1)
    axes[0].plot(epochs_range, history.history["accuracy"], label="Train", color=PLOT_PALETTE[0])
    axes[0].plot(epochs_range, history.history["val_accuracy"], label="Val", color=PLOT_PALETTE[1])
    axes[0].set_title("Accuracy", fontweight="bold")
    axes[0].legend()
    axes[1].plot(epochs_range, history.history["loss"], label="Train", color=PLOT_PALETTE[2])
    axes[1].plot(epochs_range, history.history["val_loss"], label="Val", color=PLOT_PALETTE[3])
    axes[1].set_title("Loss", fontweight="bold")
    axes[1].legend()
    plt.tight_layout()
    history_plot = os.path.join(HISTORY_DIR, "pneumonia_history.png")
    plt.savefig(history_plot, dpi=120, bbox_inches="tight")
    plt.close()

    val_probs = model.predict(x_val, verbose=0).flatten()
    threshold = find_optimal_threshold(y_val, val_probs)
    print(f"Optimal threshold: {threshold:.2f}")

    metrics = evaluate_model(model, x_test, y_test, threshold)
    model.save(checkpoint)

    config = {
        "best_model": "pneumonia_mobilenetv2",
        "preprocess": "mobilenetv2",
        "threshold": threshold,
        "img_size": list(IMG_SIZE),
        "scan_type": "chest_xray",
        "class_labels": {0: "Normal", 1: "Pneumonia Detected"},
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    comparison = {
        "generated_at": datetime.utcnow().isoformat(),
        "best_model": "pneumonia_mobilenetv2",
        "models": [metrics],
    }
    with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nAccuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | AUC: {metrics['auc']:.4f}")
    print(f"Model saved: {checkpoint}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(BASE_DIR, "dataset_pneumonia"))
    args = parser.parse_args()
    train_pneumonia_model(args.data_dir)
