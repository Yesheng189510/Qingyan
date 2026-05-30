"""Dataset wrapper for classification KD.  Reuses existing DatasetProcessing."""

import sys
from pathlib import Path

# Reuse the existing DatasetProcessing class from the teacher training code
_QINGYAN_PATH = Path(__file__).resolve().parents[1] / 'Qingyan-master' / 'train_dual_sigma'
if str(_QINGYAN_PATH) not in sys.path:
    sys.path.insert(0, str(_QINGYAN_PATH))

from dataset.dataset_processing import DatasetProcessing


__all__ = ['DatasetProcessing']
