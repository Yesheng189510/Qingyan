import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import setup_project_path, ACNE04V2_YOLO_BEST_WEIGHTS, ACNE04V2_YOLO_VAL_IMAGES, get_device

setup_project_path()

MODEL_WEIGHTS = Path(ACNE04V2_YOLO_BEST_WEIGHTS)

model = YOLO(str(MODEL_WEIGHTS))

results = model(
    source=ACNE04V2_YOLO_VAL_IMAGES,
    stream=True,
    imgsz=1280,
    conf=0.0,
    iou=0.6,
    max_det=300,
    device=get_device(),
)


def bbox2circle(bbox):
    x1, y1, x2, y2 = bbox
    x = int((x1 + x2) / 2)
    y = int((y1 + y2) / 2)
    r = int(math.sqrt((x - x1) ** 2 + (y - y1) ** 2))
    return (x, y), r


list_coord = []
paths = []
scores = []

for result in results:
    boxes = result.boxes
    path = result.path

    coord = []
    for box in boxes:
        x_center, y_center = box.xywh[0].cpu().numpy()[:2]
        width, height = box.xywh[0].cpu().numpy()[2:]
        xmin, ymin = x_center - width / 2, y_center - height / 2
        xmax, ymax = x_center + width / 2, y_center + height / 2
        cpf, rpf = bbox2circle([xmin, ymin, xmax, ymax])
        coord.append([cpf, rpf, box.conf.cpu().numpy().item()])

    dict_coord = {'acne': coord}
    score = float(np.mean(boxes.conf.cpu().numpy())) if len(boxes) > 0 else 0.0

    scores.append(score)
    paths.append(path)
    list_coord.append(dict_coord)

df = pd.DataFrame({
    'image name': paths,
    'score': scores,
    'group_acne_dict': list_coord,
})
out_csv = Path(__file__).resolve().parent / 'yolov8_acne04_predictions.csv'
df.to_csv(out_csv, index=False)
print(f'Saved: {out_csv}')
