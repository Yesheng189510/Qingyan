import os
import cv2
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import setup_project_path, ACNE04V2_ANNOTATIONS, ACNE04V2_EXAMPLES_DIR, acne04v2_image_path

setup_project_path()

json_path = ACNE04V2_ANNOTATIONS
output_path = ACNE04V2_EXAMPLES_DIR

os.makedirs(output_path, exist_ok=True)
random.seed(42)

with open(json_path, 'r', encoding='utf-8') as f:
    labels_dict = json.load(f)

all_images = labels_dict['images']
print("Total images in JSON:", len(all_images))

sample_num = 10
selected_images = random.sample(all_images, sample_num)
print(f"Randomly selected {sample_num} images")

for i, img_dict in enumerate(selected_images):
    img_id = img_dict['id']
    file_name = img_dict['file_name']
    img_file = acne04v2_image_path(file_name)

    if img_file is None:
        print("Not in v2 dataset:", file_name)
        continue

    img = cv2.imread(img_file)
    if img is None:
        print("Failed to read:", img_file)
        continue

    annotations = [
        ann for ann in labels_dict['annotations']
        if ann['image_id'] == img_id
    ]

    for ann in annotations:
        center = tuple(ann['coordinates'])
        radius = int(ann['radius'])
        thickness = 1 + int(max(img.shape[:2]) / 1000)
        cv2.circle(img, center, radius, (255, 0, 0), thickness)

    save_path = os.path.join(output_path, file_name)
    cv2.imwrite(save_path, img)
    print(f"[{i+1}/{sample_num}] Saved:", file_name)

print("Finished!")
