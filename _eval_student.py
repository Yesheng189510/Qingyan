import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
from paths import setup_project_path, DATA_PATH, test_path

setup_project_path()
from module3_kd.student_model import get_student
from module3_kd.config_kd import CFG
from dataset.dataset_processing import DatasetProcessing

run_dir = root / "module3_output_kd/20260530_172300"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = dict(CFG)
tfm = transforms.Compose([
    transforms.Resize(cfg["resize_test"]),
    transforms.ToTensor(),
    transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
])

print("Device:", device)
print("Student arch:", cfg["student_arch"])
print()
print(f"{'Fold':<6} {'Log best':<12} {'Re-eval':<12} {'Match':<8}")

accs = []
for fold in range(5):
    ckpt = run_dir / f"fold_{fold}_best.pth"
    model = get_student(cfg["student_arch"], cfg["num_classes"])
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device).eval()

    ds = DatasetProcessing(DATA_PATH, test_path(str(fold)), transform=tfm)
    ds.cache_images(resize=cfg["resize_test"])
    loader = DataLoader(
        ds, batch_size=cfg["batch_size_test"], shuffle=False, num_workers=0
    )

    preds, labels = [], []
    with torch.no_grad():
        for img, label, _ in loader:
            out = model(img.to(device)).argmax(1).cpu()
            preds.append(out)
            labels.append(label)
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    acc = accuracy_score(labels, preds)
    accs.append(acc)

    records = [
        json.loads(line)
        for line in open(run_dir / f"fold_{fold}.jsonl", encoding="utf-8")
        if line.strip()
    ]
    summary = next(r for r in records if r.get("type") == "summary")
    log_best = summary["best_acc"]
    match = abs(acc - log_best) < 1e-4
    print(f"{fold:<6} {log_best:<12.4f} {acc:<12.4f} {'OK' if match else 'DIFF'}")

print(f"Mean re-eval acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
