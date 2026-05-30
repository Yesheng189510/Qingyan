"""Centralized paths for the Qingyan repo.

Two INDEPENDENT projects share this file but do NOT share image data:

  v1 (classification / LDL / KD)  →  data/ACNE04/Classification/
  v2 (YOLO detection)             →  acne04v2-main/acne_yolo_dataset/
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════
# Project v1 — ACNE04 classification / LDL / KD (5-fold CV)
# ═══════════════════════════════════════════════════════════════════
ACNE04_ROOT = os.path.join(ROOT_DIR, 'data', 'ACNE04')
ACNE04_CLS = os.path.join(ACNE04_ROOT, 'Classification')
DATA_PATH = os.path.join(ACNE04_CLS, 'JPEGImages')

DOCTOR_RESULTS_DIR = os.path.join(ROOT_DIR, 'doctor-results')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
MODULE3_KD_OUTPUT_DIR = os.path.join(ROOT_DIR, 'module3_output_kd')

# ═══════════════════════════════════════════════════════════════════
# Project v2 — ACNE04-v2 YOLO detection (self-contained dataset)
# All images & labels live under acne04v2-main/acne_yolo_dataset/
# ═══════════════════════════════════════════════════════════════════
ACNE04V2_ROOT = os.path.join(ROOT_DIR, 'acne04v2-main')
ACNE04V2_ANNOTATIONS = os.path.join(ACNE04V2_ROOT, 'Acne04-v2_annotations.json')

ACNE04V2_YOLO_ROOT = os.path.join(ACNE04V2_ROOT, 'acne_yolo_dataset')
ACNE04V2_YOLO_YAML = os.path.join(ACNE04V2_YOLO_ROOT, 'acne_detection.yaml')
ACNE04V2_IMAGES_ROOT = os.path.join(ACNE04V2_YOLO_ROOT, 'images')
ACNE04V2_YOLO_TRAIN_IMAGES = os.path.join(ACNE04V2_IMAGES_ROOT, 'train')
ACNE04V2_YOLO_VAL_IMAGES = os.path.join(ACNE04V2_IMAGES_ROOT, 'val')
ACNE04V2_YOLO_TRAIN_LABELS = os.path.join(ACNE04V2_YOLO_ROOT, 'labels', 'train')
ACNE04V2_YOLO_VAL_LABELS = os.path.join(ACNE04V2_YOLO_ROOT, 'labels', 'val')
ACNE04V2_EXAMPLES_DIR = os.path.join(ACNE04V2_ROOT, 'examples_random10')
ACNE04V2_YOLO_RUNS_DIR = os.path.join(ACNE04V2_ROOT, 'runs')
ACNE04V2_YOLO_RUN_NAME = 'acne_yolo_final'
ACNE04V2_YOLO_BEST_WEIGHTS = os.path.join(
    ACNE04V2_YOLO_RUNS_DIR, ACNE04V2_YOLO_RUN_NAME, 'weights', 'best.pt'
)


def setup_project_path() -> str:
    """Ensure project root is on sys.path for shared imports."""
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    return ROOT_DIR


def trainval_path(fold) -> str:
    return os.path.join(ACNE04_CLS, f'NNEW_trainval_{fold}.txt')


def test_path(fold) -> str:
    return os.path.join(ACNE04_CLS, f'NNEW_test_{fold}.txt')


def acne04v2_image_path(file_name: str) -> str | None:
    """Resolve an image file inside the v2 YOLO dataset (train/ then val/)."""
    for folder in (ACNE04V2_YOLO_TRAIN_IMAGES, ACNE04V2_YOLO_VAL_IMAGES):
        path = os.path.join(folder, file_name)
        if os.path.isfile(path):
            return path
    return None


def acne04v2_image_split(file_name: str) -> str | None:
    """Return 'train' or 'val' if the image exists in the v2 dataset."""
    if os.path.isfile(os.path.join(ACNE04V2_YOLO_TRAIN_IMAGES, file_name)):
        return 'train'
    if os.path.isfile(os.path.join(ACNE04V2_YOLO_VAL_IMAGES, file_name)):
        return 'val'
    return None


def get_device():
    """Return GPU index 0 when CUDA is available, otherwise 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except ImportError:
        pass
    return 'cpu'
