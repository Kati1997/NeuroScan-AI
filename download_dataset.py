"""
Download Brain Tumor MRI dataset from Kaggle and prepare binary folders.

Requires one of:
  pip install kagglehub
  OR Kaggle CLI configured (kaggle.json in ~/.kaggle/)

Usage:
  python download_dataset.py
"""

import os
import shutil
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

NO_TUMOR_NAMES = {"no_tumor", "notumor", "no", "healthy"}
TUMOR_NAMES = {"tumor", "yes", "glioma", "meningioma", "pituitary", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"}
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _download_with_kagglehub() -> str:
    import kagglehub

    print("Downloading via kagglehub...")
    path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    print(f"Downloaded to: {path}")
    return path


def _download_with_kaggle_cli() -> str:
    import subprocess
    import zipfile

    zip_path = os.path.join(BASE_DIR, "brain-tumor-mri-dataset.zip")
    extract_dir = os.path.join(BASE_DIR, "kaggle_raw")
    print("Downloading via Kaggle CLI...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "masoudnickparvar/brain-tumor-mri-dataset", "-p", BASE_DIR],
        check=True,
    )
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def prepare_binary_dataset(source_dir: str, target_dir: str = DATASET_DIR) -> None:
    """Copy images into dataset/no_tumor and dataset/tumor."""
    no_dir = os.path.join(target_dir, "no_tumor")
    tumor_dir = os.path.join(target_dir, "tumor")
    os.makedirs(no_dir, exist_ok=True)
    os.makedirs(tumor_dir, exist_ok=True)

    count_no, count_tumor = 0, 0
    for root, _, files in os.walk(source_dir):
        folder = os.path.basename(root).lower()
        if folder in NO_TUMOR_NAMES:
            dest = no_dir
            counter = count_no
        elif folder in TUMOR_NAMES:
            dest = tumor_dir
            counter = count_tumor
        else:
            continue

        for fname in files:
            if os.path.splitext(fname)[1].lower() not in VALID_EXT:
                continue
            src = os.path.join(root, fname)
            dst_name = f"{folder}_{counter:05d}{os.path.splitext(fname)[1].lower()}"
            shutil.copy2(src, os.path.join(dest, dst_name))
            counter += 1

        if folder in NO_TUMOR_NAMES:
            count_no = counter
        else:
            count_tumor = counter

    print(f"\nPrepared dataset at: {target_dir}")
    print(f"  no_tumor: {count_no} images")
    print(f"  tumor:    {count_tumor} images")

    if count_no == 0 or count_tumor == 0:
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
            print("\nManual download:")
            print("  1. https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
            print("  2. Extract to dataset/ with no_tumor/ and tumor/ subfolders")
            print("  OR: pip install kagglehub && python download_dataset.py")
            sys.exit(1)

    prepare_binary_dataset(source)
    print("\nDone! Run: python train_model.py --data-dir dataset")


if __name__ == "__main__":
    main()
