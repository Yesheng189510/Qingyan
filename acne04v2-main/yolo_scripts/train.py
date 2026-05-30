"""Backward-compatible wrapper — runs the quick preset on the main trainer."""
import subprocess
import sys
from pathlib import Path

TRAIN_PY = Path(__file__).resolve().parents[1] / "train.py"

if __name__ == "__main__":
    cmd = [sys.executable, str(TRAIN_PY), "--preset", "quick", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))
