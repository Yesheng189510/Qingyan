import os
from pathlib import Path
import cv2

# =====================
# CONFIG
# =====================
dataset_root = r"C:\Users\28268\Desktop\LDL-master\acne_yolo_dataset"

img_dir = Path(dataset_root) / "images" / "train"
label_dir = Path(dataset_root) / "labels" / "train"

# =====================
# stats
# =====================
total_images = 0
missing_images = 0
empty_labels = 0
bad_labels = 0
total_boxes = 0

print("🚀 Starting dataset health check...\n")

# =====================
# scan labels
# =====================
for label_file in label_dir.glob("*.txt"):

    img_file = img_dir / (label_file.stem + ".jpg")

    total_images += 1

    # check image exists
    if not img_file.exists():
        print("❌ Missing image:", img_file)
        missing_images += 1
        continue

    # read label
    with open(label_file, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        print("⚠️ Empty label:", label_file.name)
        empty_labels += 1
        continue

    for i, line in enumerate(lines):

        parts = line.strip().split()

        # format check
        if len(parts) != 5:
            print(f"❌ FORMAT ERROR {label_file.name} line {i}: {line}")
            bad_labels += 1
            continue

        try:
            cls, x, y, w, h = map(float, parts)
        except:
            print(f"❌ TYPE ERROR {label_file.name} line {i}")
            bad_labels += 1
            continue

        # range check
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            print(f"❌ OUT OF RANGE {label_file.name} line {i}: {line.strip()}")
            bad_labels += 1
            continue

        total_boxes += 1


# =====================
# summary
# =====================
print("\n========== DATASET REPORT ==========")
print("Total label files:", total_images)
print("Missing images:", missing_images)
print("Empty labels:", empty_labels)
print("Bad labels:", bad_labels)
print("Total valid boxes:", total_boxes)
print("====================================\n")

if bad_labels == 0 and missing_images == 0:
    print("✅ Dataset is CLEAN and ready for YOLO training!")
else:
    print("⚠️ Dataset has issues. Fix before training.")