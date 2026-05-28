import os
import cv2
import json
import random

# =========================
# 路径设置
# =========================
images_path = r"C:\Users\28268\Desktop\LDL-master\code\ACNE04\Classification\JPEGImages"

json_path = r"C:\Users\28268\Desktop\LDL-master\acne04v2-main\Acne04-v2_annotations.json"

output_path = r"C:\Users\28268\Desktop\LDL-master\acne04v2-main\examples_random10"

os.makedirs(output_path, exist_ok=True)

# =========================
# 固定随机种子
# =========================
random.seed(42)

# =========================
# 读取 JSON
# =========================
with open(json_path, 'r', encoding='utf-8') as f:
    labels_dict = json.load(f)

all_images = labels_dict['images']

total = len(all_images)

print("Total images:", total)

# =========================
# 随机抽取 10 张
# =========================
sample_num = 10

selected_images = random.sample(all_images, sample_num)

print(f"Randomly selected {sample_num} images")

# =========================
# 开始处理
# =========================
for i, img_dict in enumerate(selected_images):

    img_id = img_dict['id']
    file_name = img_dict['file_name']

    img_file = os.path.join(images_path, file_name)

    img = cv2.imread(img_file)

    if img is None:
        print("Failed:", img_file)
        continue

    # 找到当前图片对应标注
    annotations = [
        ann for ann in labels_dict['annotations']
        if ann['image_id'] == img_id
    ]

    # 画圆
    for ann in annotations:

        center = tuple(ann['coordinates'])

        radius = int(ann['radius'])

        thickness = 1 + int(max(img.shape[:2]) / 1000)

        cv2.circle(
            img,
            center,
            radius,
            (255, 0, 0),   # BGR 蓝色
            thickness
        )

    # 保存
    save_path = os.path.join(output_path, file_name)

    cv2.imwrite(save_path, img)

    print(f"[{i+1}/{sample_num}] Saved:", file_name)

print("Finished!")