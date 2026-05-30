"""Regenerate YOLO labels from Acne04-v2 JSON.

Uses images already present in acne_yolo_dataset/images/{train,val}.
Does NOT read from data/ACNE04 — v2 is a separate self-contained dataset.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import (
    setup_project_path,
    ACNE04V2_ANNOTATIONS,
    ACNE04V2_YOLO_ROOT,
    ACNE04V2_YOLO_YAML,
    ACNE04V2_YOLO_TRAIN_IMAGES,
    ACNE04V2_YOLO_VAL_IMAGES,
    ACNE04V2_YOLO_TRAIN_LABELS,
    ACNE04V2_YOLO_VAL_LABELS,
    acne04v2_image_split,
)

setup_project_path()

SPLIT_DIRS = {
    'train': (ACNE04V2_YOLO_TRAIN_IMAGES, ACNE04V2_YOLO_TRAIN_LABELS),
    'val': (ACNE04V2_YOLO_VAL_IMAGES, ACNE04V2_YOLO_VAL_LABELS),
}


def write_yolo_labels(img_info, anns, label_path):
    width = img_info['width']
    height = img_info['height']

    lines = []
    for ann in anns:
        cx, cy = ann['coordinates']
        radius = ann['radius']

        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(width, cx + radius)
        y2 = min(height, cy + radius)

        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            continue

        x_center = (x1 + x2) / 2 / width
        y_center = (y1 + y2) / 2 / height
        bw_n = bw / width
        bh_n = bh / height

        if any(v < 0 or v > 1 for v in (x_center, y_center, bw_n, bh_n)):
            continue

        lines.append(f"0 {x_center:.6f} {y_center:.6f} {bw_n:.6f} {bh_n:.6f}\n")

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(''.join(lines), encoding='utf-8')


def main():
    with open(ACNE04V2_ANNOTATIONS, 'r', encoding='utf-8') as f:
        data = json.load(f)

    id_to_info = {img['id']: img for img in data['images']}
    ann_dict = {}
    for ann in data['annotations']:
        ann_dict.setdefault(ann['image_id'], []).append(ann)

    print(f"JSON: {ACNE04V2_ANNOTATIONS}")
    print(f"Dataset root: {ACNE04V2_YOLO_ROOT}")

    stats = {'train': 0, 'val': 0, 'skipped': 0}

    for split, (img_dir, label_dir) in SPLIT_DIRS.items():
        img_dir = Path(img_dir)
        label_dir = Path(label_dir)

        if not img_dir.is_dir():
            print(f"Warning: missing {img_dir}")
            continue

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue

            file_name = img_path.name
            split_found = acne04v2_image_split(file_name)
            if split_found != split:
                continue

            img_info = next(
                (info for info in data['images'] if info['file_name'] == file_name),
                None,
            )
            if img_info is None:
                print(f"No JSON entry for {file_name}, skip")
                stats['skipped'] += 1
                continue

            anns = ann_dict.get(img_info['id'], [])
            label_path = label_dir / f"{img_path.stem}.txt"
            write_yolo_labels(img_info, anns, label_path)
            stats[split] += 1

    print(f"Labels written — train: {stats['train']}, val: {stats['val']}, skipped: {stats['skipped']}")

    yaml_text = """# YOLO dataset config — paths relative to this file's directory
path: .

train: images/train
val: images/val

nc: 1
names:
  0: acne
"""
    Path(ACNE04V2_YOLO_YAML).write_text(yaml_text, encoding='utf-8')
    print("YAML saved:", ACNE04V2_YOLO_YAML)


if __name__ == '__main__':
    main()
