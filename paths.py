"""Centralized paths and import bootstrap for the Qingyan project."""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ACNE04_ROOT = os.path.join(ROOT_DIR, 'data', 'ACNE04')
ACNE04_CLS = os.path.join(ACNE04_ROOT, 'Classification')
DATA_PATH = os.path.join(ACNE04_CLS, 'JPEGImages')
DOCTOR_RESULTS_DIR = os.path.join(ROOT_DIR, 'doctor-results')


def setup_project_path() -> str:
    """Ensure project root is on sys.path for shared imports."""
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    return ROOT_DIR


def trainval_path(fold) -> str:
    return os.path.join(ACNE04_CLS, f'NNEW_trainval_{fold}.txt')


def test_path(fold) -> str:
    return os.path.join(ACNE04_CLS, f'NNEW_test_{fold}.txt')
