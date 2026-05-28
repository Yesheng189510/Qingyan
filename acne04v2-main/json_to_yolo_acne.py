import os
import json
import random
import shutil
from pathlib import Path

import cv2

# =====================================================
# Acne04-v2 JSON -> YOLO Dataset Converter
# =====================================================

# -------------------------
# PATH CONFIG
# -------------------------
json_path = r"C:\Users\28268\Desktop\LDL-master\acne04v2-main\Acne04-v2_annotations.json"

images_path = r"C:\Users\28268\Desktop\LDL-master\code\ACNE04\Classification\JPEGImages"

output_root = r"C:\Users\28268\Desktop\LDL-master\acne_yolo_dataset"

# -------------------------
# TRAIN / VAL SPLIT
# -------------------------
train_ratio = 0.8
random_seed = 42

# =====================================================
# OUTPUT STRUCTURE
# =====================================================

train_img_dir = Path(output_root) / "images" / "train"
val_img_dir = Path(output_root) / "images" / "val"

train_label_dir = Path(output_root) / "labels" / "train"
val_label_dir = Path(output_root) / "labels" / "val"

train_img_dir.mkdir(parents=True, exist_ok=True)
val_img_dir.mkdir(parents=True, exist_ok=True)

train_label_dir.mkdir(parents=True, exist_ok=True)
val_label_dir.mkdir(parents=True, exist_ok=True)

# =====================================================
# LOAD JSON
# =====================================================

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

images_info = data["images"]
annotations = data["annotations"]

print(f"Total images: {len(images_info)}")
print(f"Total annotations: {len(annotations)}")

# =====================================================
# GROUP ANNOTATIONS
# =====================================================

ann_dict = {}

for ann in annotations:

    image_id = ann["image_id"]

    if image_id not in ann_dict:
        ann_dict[image_id] = []

    ann_dict[image_id].append(ann)

# =====================================================
# TRAIN / VAL SPLIT
# =====================================================

random.seed(random_seed)

random.shuffle(images_info)

split_index = int(len(images_info) * train_ratio)

train_images = images_info[:split_index]
val_images = images_info[split_index:]

print(f"Train images: {len(train_images)}")
print(f"Val images: {len(val_images)}")

# =====================================================
# CONVERT FUNCTION
# =====================================================

def process_dataset(image_list, img_out_dir, label_out_dir):

    success_count = 0
    skip_count = 0

    for img_info in image_list:

        image_id = img_info["id"]
        file_name = img_info["file_name"]

        width = img_info["width"]
        height = img_info["height"]

        src_img_path = os.path.join(images_path, file_name)

        # -------------------------
        # image exists?
        # -------------------------
        if not os.path.exists(src_img_path):

            print("Missing image:", src_img_path)
            skip_count += 1
            continue

        # -------------------------
        # copy image
        # -------------------------
        dst_img_path = img_out_dir / file_name

        shutil.copy(src_img_path, dst_img_path)

        # -------------------------
        # label path
        # -------------------------
        label_path = label_out_dir / f"{Path(file_name).stem}.txt"

        # -------------------------
        # get annotations
        # -------------------------
        anns = ann_dict.get(image_id, [])

        # -------------------------
        # write YOLO labels
        # -------------------------
        with open(label_path, "w") as f:

            for ann in anns:

                cx, cy = ann["coordinates"]
                radius = ann["radius"]

                # ---------------------------------
                # convert circle -> bbox
                # ---------------------------------
                x1 = max(0, cx - radius)
                y1 = max(0, cy - radius)

                x2 = min(width, cx + radius)
                y2 = min(height, cy + radius)

                bw = x2 - x1
                bh = y2 - y1

                # ---------------------------------
                # skip invalid boxes
                # ---------------------------------
                if bw <= 0 or bh <= 0:
                    continue

                # ---------------------------------
                # YOLO normalize
                # ---------------------------------
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height

                bw = bw / width
                bh = bh / height

                # ---------------------------------
                # final safety check
                # ---------------------------------
                values = [x_center, y_center, bw, bh]

                if any(v < 0 or v > 1 for v in values):
                    continue

                # class_id = 0
                f.write(
                    f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n"
                )

        success_count += 1

    print("\nFinished.")
    print("Success:", success_count)
    print("Skipped:", skip_count)


# =====================================================
# RUN
# =====================================================

print("\nProcessing TRAIN dataset...")
process_dataset(
    train_images,
    train_img_dir,
    train_label_dir
)

print("\nProcessing VAL dataset...")
process_dataset(
    val_images,
    val_img_dir,
    val_label_dir
)

# =====================================================
# YAML
# =====================================================

yaml_text = f"""
path: {output_root.replace(os.sep, "/")}

train: images/train
val: images/val

nc: 1
names: ["acne"]
"""

yaml_path = Path(output_root) / "acne_detection.yaml"

with open(yaml_path, "w") as f:
    f.write(yaml_text)

print("\nYAML saved:")
print(yaml_path)

print("\nDataset conversion complete!")