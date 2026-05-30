"""Evaluate trained student model and generate a per-fold report."""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

_PROJECT = Path(__file__).resolve().parents[1]
_QINGYAN = _PROJECT / 'Qingyan-master' / 'train_dual_sigma'
for _p in [str(_PROJECT), str(_QINGYAN)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from module3_kd.student_model import StudentResNet18
from module3_kd.config_kd import CFG as DEFAULT_CFG
from dataset.dataset_processing import DatasetProcessing
from utils.report import report_precision_se_sp_yi, report_mae_mse
from utils.utils import AverageMeter


@torch.no_grad()
def evaluate_fold(model, loader, device, num_classes=4):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for img, label, _ in loader:
        img = img.to(device)
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(label)
        all_probs.append(probs.cpu())

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = torch.cat(all_probs).numpy()

    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    # Per-class metrics
    result, ave_acc, cls_report = report_precision_se_sp_yi(preds, labels)

    # MAE / MSE (use prediction class index as pseudo-count estimate)
    _, mae, mse, _ = report_mae_mse(labels, preds, labels)

    # Per-class accuracy
    per_class_acc = {}
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        if len(idx) > 0:
            per_class_acc[f'class_{c}_acc'] = float(accuracy_score(
                labels[idx], preds[idx]))
        else:
            per_class_acc[f'class_{c}_acc'] = 0.0

    return {
        'accuracy': float(acc),
        'mae': float(mae),
        'mse': float(mse),
        'confusion_matrix': cm.tolist(),
        'per_class': per_class_acc,
        'n_images': len(labels),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate KD student model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to student checkpoint (.pth)')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--predictions', action='store_true',
                        help='Also save per-image predictions CSV')
    args = parser.parse_args()

    cfg = dict(DEFAULT_CFG)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load model
    model = StudentResNet18(num_classes=cfg['num_classes'])
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    print(f'Loaded checkpoint: {args.checkpoint}')

    # Data
    data_root = _PROJECT / 'data' / 'ACNE04' / 'Classification'
    test_file = str(data_root / f'NNEW_test_{args.fold}.txt')
    img_dir   = str(data_root / 'JPEGImages')

    normalize = transforms.Normalize(mean=cfg['mean'], std=cfg['std'])
    test_tfm = transforms.Compose([
        transforms.Resize(cfg['resize_test']),
        transforms.ToTensor(),
        normalize,
    ])

    ds = DatasetProcessing(img_dir, test_file, transform=test_tfm)
    ds.cache_images(resize=cfg['resize_test'])
    loader = DataLoader(ds, batch_size=cfg['batch_size_test'],
                        shuffle=False, num_workers=cfg['num_workers'])
    print(f'Test set: {len(ds)} images')

    # Evaluate
    report = evaluate_fold(model, loader, device, cfg['num_classes'])
    report['fold'] = args.fold
    report['checkpoint'] = args.checkpoint

    # Output
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / f'fold_{args.fold}_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'Saved report: {report_path}')

    # Print summary
    print(f'\n{"="*50}')
    print(f'Fold {args.fold} Report')
    print(f'  Accuracy:     {report["accuracy"]:.4f}')
    print(f'  MAE:          {report["mae"]:.4f}')
    print(f'  MSE:          {report["mse"]:.4f}')
    print(f'  N images:     {report["n_images"]}')
    print(f'  Confusion Matrix:')
    print(np.array(report['confusion_matrix']))
    print(f'{"="*50}')

    # Save per-image predictions
    if args.predictions:
        _, preds, labels = [], [], []
        for img, label, _ in loader:
            img = img.to(device)
            logits = model(img)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            preds.extend(pred.cpu().tolist())
            labels.extend(label.tolist())

        csv_path = out_dir / f'fold_{args.fold}_predictions.csv'
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('filename,true_label,pred_label\n')
            for fname, t, p in zip(ds.img_filename, labels, preds):
                f.write(f'{fname},{t},{p}\n')
        print(f'Saved predictions: {csv_path}')


if __name__ == '__main__':
    main()
