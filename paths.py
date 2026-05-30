"""Centralized ACNE04 dataset paths for Qingyan project."""
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ACNE04_ROOT = os.path.join(ROOT_DIR, 'data', 'ACNE04')
ACNE04_CLS = os.path.join(ACNE04_ROOT, 'Classification')
DATA_PATH = os.path.join(ACNE04_CLS, 'JPEGImages')


def trainval_path(fold) -> str:
    return os.path.join(ACNE04_CLS, f'NNEW_trainval_{fold}.txt')


def test_path(fold) -> str:
    return os.path.join(ACNE04_CLS, f'NNEW_test_{fold}.txt')
