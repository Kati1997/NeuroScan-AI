"""
Download Chest X-Ray Pneumonia dataset from Kaggle and prepare binary folders.

Usage:
  python download_dataset_pneumonia.py
"""

import os
import shutil
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_pneumonia")

NORMAL_NAMES = {"normal", "no", "healthy", "no_pneumonia"}
PNEUMONIA_NAMES = {"pneumonia", "bacterial_pneumonia", "viral_pneumonia", "yes"}
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SKIP_DIRS = {"__macosx", "__pycache__", ".git"}


def _download_with_kagglehub() -> str:
    import kagglehub

    print("Downloading Chest X-Ray Pneumonia dataset via kagglehub...")
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print(f"Downloaded to: {path}")
    return path


def _download_with_kaggle_cli() -> str:
    import subprocess
    import zipfile

    zip_path = os.path.join(BASE_DIR, "chest-xray-pneumonia.zip")
    extract_dir = os.path.join(BASE_DIR, "kaggle_raw_pneumonia")
    print("Downloading via Kaggle CLI...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "paultimothymooney/chest-xray-pneumonia", "-p", BASE_DIR],
        check=True,
    )
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def _is_valid_image(path: str) -> bool:
    if os.path.basename(path).startswith("._"):
        return False
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _find_train_roots(source_dir: str) -> list:
    candidates = []
    for root, dirs, _ in os.walk(source_dir):
        dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS and not d.startswith(".")]
        folder = os.path.basename(root).lower()
        parent = os.path.basename(os.path.dirname(root)).lower()
        if parent == "train" and folder in NORMAL_NAMES | PNEUMONIA_NAMES:
            candidates.append(root)
    by_class = {}
    for path in candidates:
        cls = os.path.basename(path).lower()
        if cls not in by_class or len(path) < len(by_class[cls]):
            by_class[cls] = path
    return list(by_class.values())


def prepare_binary_dataset(source_dir: str, target_dir: str = DATASET_DIR) -> None:
    """Copy train-split images into dataset_pneumonia/NORMAL/ and PNEUMONIA/."""
    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir)

    normal_dir = os.path.join(target_dir, "NORMAL")
    pneumonia_dir = os.path.join(target_dir, "PNEUMONIA")
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(pneumonia_dir, exist_ok=True)

    train_roots = _find_train_roots(source_dir)
    if not train_roots:
        raise RuntimeError(f"No train/NORMAL or train/PNEUMONIA folders found under {source_dir}")

    count_normal, count_pneumonia = 0, 0
    seen_names = set()

    for class_root in sorted(train_roots):
        folder = os.path.basename(class_root).lower()
        dest = normal_dir if folder in NORMAL_NAMES else pneumonia_dir
        counter = count_normal if folder in NORMAL_NAMES else count_pneumonia

        for fname in sorted(os.listdir(class_root)):
            if os.path.splitext(fname)[1].lower() not in VALID_EXT:
                continue
            src = os.path.join(class_root, fname)
            if not _is_valid_image(src):
                continue
            out_name = fname if fname not in seen_names else f"{counter:05d}_{fname}"
            seen_names.add(out_name)
            shutil.copy2(src, os.path.join(dest, out_name))
            counter += 1

        if folder in NORMAL_NAMES:
            count_normal = counter
        else:
            count_pneumonia = counter

    print(f"\nPrepared dataset at: {target_dir}")
    print(f"  NORMAL:    {count_normal} images (train split)")
    print(f"  PNEUMONIA: {count_pneumonia} images (train split)")

    if count_normal == 0 or count_pneumonia == 0:
        raise RuntimeError("Dataset preparation failed — check folder structure.")


def main():
    source = None
    try:
        source = _download_with_kagglehub()
    except Exception as e1:
        print(f"kagglehub failed: {e1}")
        try:
            source = _download_with_kaggle_cli()
        except Exception as e2:
            print(f"Kaggle CLI failed: {e2}")
            print("\nManual: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
            sys.exit(1)

    prepare_binary_dataset(source)
    print("\nDone! Run: python train_pneumonia_model.py --data-dir dataset_pneumonia")


if __name__ == "__main__":
    main()
