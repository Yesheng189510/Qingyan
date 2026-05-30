"""YOLO acne detection training for ACNE04-v2.

Usage:
  python acne04v2-main/train.py              # full (1280px, tuned for ~8GB VRAM)
  python acne04v2-main/train.py --preset quick
  python acne04v2-main/train.py --batch 1    # override batch if OOM persists
"""
import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import setup_project_path, ACNE04V2_YOLO_YAML, ACNE04V2_YOLO_RUNS_DIR, get_device

setup_project_path()

PRESETS = {
    # ~8GB VRAM (e.g. RTX 5060 Laptop): batch=2 @ 1280, no multi_scale
    "full": {
        "imgsz": 1280,
        "epochs": 300,
        "batch": 2,
        "patience": 150,
        "overlap_mask": True,
        "box": 7.5,
        "cls": 0.8,
        "dfl": 1.5,
        "multi_scale": False,
        "hsv_h": 0.015,
        "hsv_s": 0.3,
        "hsv_v": 0.2,
        "name": "acne_yolo_final",
    },
    "quick": {
        "imgsz": 640,
        "epochs": 200,
        "batch": 4,
        "patience": 100,
        "multi_scale": False,
        "name": "acne_yolo_quick",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on ACNE04-v2")
    parser.add_argument(
        "--preset",
        choices=PRESETS.keys(),
        default="full",
        help="full: 1280px / batch=2 / 300 epochs; quick: 640px / 200 epochs",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="override preset batch size (use 1 if full preset still OOMs)",
    )
    return parser.parse_args()


def train(preset: str = "full", batch: int | None = None):
    cfg = PRESETS[preset].copy()
    if batch is not None:
        cfg["batch"] = batch
    device = get_device()
    print(f"Preset: {preset}")
    print(f"Using device: {device}")
    print(f"Dataset yaml: {ACNE04V2_YOLO_YAML}")
    print(f"Training: imgsz={cfg['imgsz']} batch={cfg['batch']} multi_scale={cfg.get('multi_scale', False)}")

    model = YOLO("yolov8m.pt")
    return model.train(
        data=ACNE04V2_YOLO_YAML,
        device=device,
        workers=0,
        amp=True,
        project=ACNE04V2_YOLO_RUNS_DIR,
        **cfg,
    )


if __name__ == "__main__":
    args = parse_args()
    train(args.preset, batch=args.batch)
