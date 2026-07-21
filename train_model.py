"""
Deep Learning training module for Brain Tumor Detection.
Trains Custom CNN, ResNet50, and MobileNetV2 with proper preprocessing and fine-tuning.
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
FINE_TUNE_EPOCHS = 15
RANDOM_SEED = 42

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
HISTORY_DIR = os.path.join(MODELS_DIR, "training_history")
COMPARISON_PATH = os.path.join(MODELS_DIR, "model_comparison.json")
CONFIG_PATH = os.path.join(MODELS_DIR, "model_config.json")

CLASS_LABELS = ["no_tumor", "tumor"]


def set_seeds(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def discover_dataset(data_dir: str) -> Tuple[List[str], List[int]]:
    image_paths: List[str] = []
    labels: List[int] = []

    search_roots = [
        data_dir,
        os.path.join(data_dir, "Training"),
        os.path.join(data_dir, "train"),
    ]

    class_folders = {
        "no_tumor": 0,
        "notumor": 0,
        "no": 0,
        "healthy": 0,
        "tumor": 1,
        "yes": 1,
        "glioma": 1,
        "meningioma": 1,
        "pituitary": 1,
        "glioma_tumor": 1,
        "meningioma_tumor": 1,
        "pituitary_tumor": 1,
    }

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for folder_name in os.listdir(root):
            folder_path = os.path.join(root, folder_name)
            if not os.path.isdir(folder_path):
                continue
            label = class_folders.get(folder_name.lower())
            if label is None:
                continue
            for filename in os.listdir(folder_path):
                ext = os.path.splitext(filename)[1].lower()
                if ext in valid_extensions:
                    image_paths.append(os.path.join(folder_path, filename))
                    labels.append(label)

    if not image_paths:
        raise FileNotFoundError(
            f"No training images found in {data_dir}. "
            "Run: python download_dataset.py"
        )

    return image_paths, labels


def load_raw_images(image_paths: List[str], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Load images as uint8 RGB arrays (0-255) before model-specific preprocessing."""
    images, valid_labels = [], []
    for path, label in zip(image_paths, labels):
        try:
            img = tf.keras.preprocessing.image.load_img(path, target_size=IMG_SIZE)
            arr = tf.keras.preprocessing.image.img_to_array(img)
            images.append(arr)
            valid_labels.append(label)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
    return np.array(images, dtype=np.float32), np.array(valid_labels, dtype=np.float32)


def apply_preprocess(images: np.ndarray, preprocess_fn: Optional[Callable]) -> np.ndarray:
    if preprocess_fn is None:
        return images / 255.0
    return preprocess_fn(images.copy())


def build_custom_cnn(input_shape: Tuple[int, int, int] = (224, 224, 3)) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="custom_cnn",
    )
    return model


def build_resnet50(input_shape: Tuple[int, int, int] = (224, 224, 3)) -> tf.keras.Model:
    base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs, name="resnet50")
    model._base_model = base  # noqa: SLF001
    return model


def build_mobilenetv2(input_shape: Tuple[int, int, int] = (224, 224, 3)) -> tf.keras.Model:
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs, name="mobilenetv2")
    model._base_model = base  # noqa: SLF001
    return model


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximizes F1 score on validation data."""
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.2, 0.81, 0.02):
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)
    return best_thresh


def get_callbacks(model_name: str) -> List:
    os.makedirs(HISTORY_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODELS_DIR, f"{model_name}.keras")
    return [
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]


def evaluate_model(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    threshold: float = 0.5,
) -> Dict:
    y_prob = model.predict(x_test, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "model_name": model_name,
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    metrics["auc"] = float(auc(fpr, tpr))

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(HISTORY_DIR, f"{model_name}_roc.png")
    plt.savefig(roc_path)
    plt.close()
    metrics["roc_plot"] = roc_path

    cm = np.array(metrics["confusion_matrix"])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_LABELS))
    plt.xticks(tick_marks, CLASS_LABELS, rotation=45)
    plt.yticks(tick_marks, CLASS_LABELS)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(HISTORY_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    metrics["confusion_matrix_plot"] = cm_path

    return metrics


def train_single_model(
    model_builder,
    model_name: str,
    preprocess_fn: Optional[Callable],
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    x_val_raw: np.ndarray,
    y_val: np.ndarray,
    x_test_raw: np.ndarray,
    y_test: np.ndarray,
    fine_tune: bool = False,
) -> Dict:
    print(f"\n{'='*60}\nTraining {model_name}\n{'='*60}")

    x_train = apply_preprocess(x_train_raw, preprocess_fn)
    x_val = apply_preprocess(x_val_raw, preprocess_fn)
    x_test = apply_preprocess(x_test_raw, preprocess_fn)

    class_weights_arr = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight = {i: class_weights_arr[i] for i in range(len(class_weights_arr))}

    model = model_builder()
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])

    datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    steps = max(1, len(x_train) // BATCH_SIZE)
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=steps,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=get_callbacks(model_name),
        verbose=1,
    )

    # Fine-tune transfer learning models
    if fine_tune and hasattr(model, "_base_model"):
        print(f"Fine-tuning {model_name} base layers...")
        base = model._base_model
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-5), loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(
            datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
            steps_per_epoch=steps,
            validation_data=(x_val, y_val),
            epochs=FINE_TUNE_EPOCHS,
            class_weight=class_weight,
            callbacks=get_callbacks(model_name),
            verbose=1,
        )

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title(f"{model_name} - Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title(f"{model_name} - Loss")
    plt.legend()
    plt.tight_layout()
    history_plot = os.path.join(HISTORY_DIR, f"{model_name}_history.png")
    plt.savefig(history_plot)
    plt.close()

    val_probs = model.predict(x_val, verbose=0).flatten()
    optimal_threshold = find_optimal_threshold(y_val, val_probs)
    print(f"Optimal threshold for {model_name}: {optimal_threshold:.2f}")

    metrics = evaluate_model(model, x_test, y_test, model_name, threshold=optimal_threshold)
    metrics["history_plot"] = history_plot
    metrics["model_path"] = os.path.join(MODELS_DIR, f"{model_name}.keras")
    metrics["preprocess"] = model_name if preprocess_fn else "normalize"
    return metrics


def train_all_models(data_dir: str, models_to_train: Optional[List[str]] = None) -> Dict:
    set_seeds()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

    image_paths, labels = discover_dataset(data_dir)
    print(f"Found {len(image_paths)} images.")
    print(f"  No tumor: {labels.count(0)}")
    print(f"  Tumor:    {labels.count(1)}")

    x_raw, y_data = load_raw_images(image_paths, labels)
    x_train_raw, x_temp, y_train, y_temp = train_test_split(
        x_raw, y_data, test_size=0.25, random_state=RANDOM_SEED, stratify=y_data
    )
    x_val_raw, x_test_raw, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.4, random_state=RANDOM_SEED, stratify=y_temp
    )

    all_builders = [
        (build_custom_cnn, "custom_cnn", None, False),
        (build_resnet50, "resnet50", resnet_preprocess, True),
        (build_mobilenetv2, "mobilenetv2", mobilenet_preprocess, True),
    ]

    if models_to_train:
        all_builders = [b for b in all_builders if b[1] in models_to_train]

    all_metrics = []
    for builder, name, preprocess_fn, fine_tune in all_builders:
        metrics = train_single_model(
            builder, name, preprocess_fn,
            x_train_raw, y_train, x_val_raw, y_val, x_test_raw, y_test,
            fine_tune=fine_tune,
        )
        all_metrics.append(metrics)

    best = max(all_metrics, key=lambda m: m["f1_score"])
    best_model_path = os.path.join(MODELS_DIR, "best_model.keras")
    shutil.copy2(best["model_path"], best_model_path)

    config = {
        "best_model": best["model_name"],
        "preprocess": best.get("preprocess", "normalize"),
        "threshold": best["threshold"],
        "img_size": list(IMG_SIZE),
        "class_labels": {0: "No Tumor Detected", 1: "Tumor Detected"},
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    comparison = {
        "generated_at": datetime.utcnow().isoformat(),
        "best_model": best["model_name"],
        "best_model_path": best_model_path,
        "selection_criterion": "f1_score",
        "models": all_metrics,
    }
    with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nBest model: {best['model_name']} | F1: {best['f1_score']:.4f} | Acc: {best['accuracy']:.4f}")
    print(f"Threshold: {best['threshold']:.2f}")
    print(f"Saved to: {best_model_path}")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train brain tumor detection models")
    parser.add_argument("--data-dir", default=os.path.join(BASE_DIR, "dataset"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mobilenetv2", "resnet50", "custom_cnn"],
        help="Models to train: custom_cnn resnet50 mobilenetv2",
    )
    args = parser.parse_args()
    train_all_models(args.data_dir, models_to_train=args.models)
