import sys
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import setup_project_path, ACNE04V2_YOLO_BEST_WEIGHTS, ACNE04V2_YOLO_YAML, get_device

setup_project_path()

MODEL_WEIGHTS = Path(ACNE04V2_YOLO_BEST_WEIGHTS)

trained_model = YOLO(str(MODEL_WEIGHTS))
device = get_device()

for thresh in tqdm(range(10, 55, 5)):
    metrics = trained_model.val(
        data=ACNE04V2_YOLO_YAML,
        imgsz=1280,
        batch=6,
        conf=thresh / 100,
        iou=0.6,
        save_json=True,
        device=device,
    )

    print('conf', thresh / 100)
    print('map50:', metrics.box.map50)
    print('sensitivity', metrics.box.r)
