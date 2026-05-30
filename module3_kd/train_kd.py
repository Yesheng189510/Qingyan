"""Train student model via KD (vanilla or DKD) from ResNet-50 teacher.

Usage:
    # Auto-find teacher weights (logs/<latest>/fold_X_best.pth or mode/)
    python module3_kd/train_kd.py --folds "1"

    # Point to a specific teacher training run
    python module3_kd/train_kd.py --folds "1" --teacher_dir logs/20260530_120000

    # Override specific params via CLI
    python module3_kd/train_kd.py --folds "1" --lr 0.0001 --epochs 120
"""

import sys
import json
import time
import argparse
import copy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix

# ── paths ──
_PROJECT = Path(__file__).resolve().parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from paths import setup_project_path, DATA_PATH, trainval_path, test_path
setup_project_path()

from model.resnet50 import resnet50 as ResNet50Teacher
from module3_kd.student_model import get_student
from module3_kd.kd_losses import compute_kd_loss
from module3_kd.config_kd import CFG as DEFAULT_CFG
from dataset.dataset_processing import DatasetProcessing
from transforms.affine_transforms import RandomRotate
from utils.report import report_precision_se_sp_yi, report_mae_mse
from utils.utils import AverageMeter


# ── JSON Lines logger ──────────────────────────────────

class FoldLogger:
    def __init__(self, path: Path):
        self.path = path
        path.write_text('', encoding='utf-8')

    def log(self, record: dict):
        record['time'] = datetime.now().isoformat(timespec='seconds')
        line = json.dumps(record, ensure_ascii=False)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
        print(line)


# ── Helpers ────────────────────────────────────────────

def get_lr(optimizer) -> float:
    return optimizer.param_groups[0]['lr']


# ── Indexed dataset wrapper ──────────────────────────

class IndexedDataset:
    """Wraps a dataset to also return the sample index."""
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label, extra = self.ds[idx]
        return img, label, idx, extra


# ── Inference helpers ──────────────────────────────────

@torch.no_grad()
def teacher_predict(teacher, loader, device):
    teacher.eval()
    all_probs = []
    all_labels = []
    for img, label, _ in loader:
        img = img.to(device)
        cls_prob, _, _ = teacher(img, None)
        all_probs.append(cls_prob.cpu())
        all_labels.append(label)
    return torch.cat(all_probs), torch.cat(all_labels)


@torch.no_grad()
def student_predict(student, loader, device):
    student.eval()
    all_logits = []
    all_labels = []
    for img, label, _ in loader:
        img = img.to(device)
        logits = student(img)
        all_logits.append(logits.cpu())
        all_labels.append(label)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds = logits.argmax(dim=1)
    return logits, preds, labels


# ── MixUp ──────────────────────────────────────────────

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Config loading ─────────────────────────────────────

def deep_update(base: dict, override: dict) -> dict:
    """Recursively update base dict with override values."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config_path: str | Path) -> dict:
    """Load a JSON config file and return as dict."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_teacher_checkpoint(fold_idx: int, cfg: dict, project: Path) -> Path:
    """Locate fold_{fold}_best.pth without requiring a manual copy to mode/.

    Search order:
      1. cfg['teacher_ckpt']  (supports {fold} placeholder)
      2. cfg['teacher_dir'] / fold_{fold}_best.pth
      3. mode/fold_{fold}_best.pth  (legacy layout)
      4. logs/<latest_run>/fold_{fold}_best.pth  (train_pth.py output)
    """
    filename = f'fold_{fold_idx}_best.pth'
    candidates: list[Path] = []

    ckpt = cfg.get('teacher_ckpt')
    if ckpt:
        ckpt_str = str(ckpt)
        p = Path(ckpt_str.format(fold=fold_idx) if '{fold}' in ckpt_str else ckpt_str)
        candidates.append(p if p.is_absolute() else project / p)

    teacher_dir = cfg.get('teacher_dir')
    if teacher_dir:
        d = Path(teacher_dir)
        candidates.append((d if d.is_absolute() else project / d) / filename)

    candidates.append(project / 'mode' / filename)

    logs_root = project / 'logs'
    if logs_root.is_dir():
        matches = list(logs_root.glob(f'*/{filename}'))
        if matches:
            candidates.append(max(matches, key=lambda p: p.stat().st_mtime))

    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path

    tried = '\n  '.join(str(p) for p in candidates)
    raise FileNotFoundError(
        f'Teacher checkpoint not found for fold {fold_idx} ({filename}).\n'
        f'Searched:\n  {tried}\n'
        'Train the teacher first:\n'
        '  python train_dual_sigma/train_pth.py\n'
        'Or point to an existing run:\n'
        '  python module3_kd/train_kd.py --teacher_dir logs/<run_id> --folds "1"'
    )


# ── Main train + eval ──────────────────────────────────

def train_one_fold(fold_idx: int, cfg: dict, out_dir: Path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    logger = FoldLogger(out_dir / f'fold_{fold_idx}.jsonl')

    # ── Data files ──
    train_file = trainval_path(fold_idx)
    test_file  = test_path(fold_idx)
    img_dir    = DATA_PATH

    # ── Log meta ──
    logger.log({
        'type': 'meta', 'fold': fold_idx,
        'train_file': train_file, 'test_file': test_file,
        **{k: v for k, v in cfg.items() if k not in ('mean', 'std')},
    })

    # ── Transforms ──
    normalize = transforms.Normalize(mean=cfg['mean'], std=cfg['std'])

    train_tfm_list = [
        transforms.Resize(cfg['resize_train']),
        transforms.RandomCrop(cfg['crop_size']),
    ]
    if cfg.get('use_random_horizontal_flip', True):
        train_tfm_list.append(transforms.RandomHorizontalFlip())
    train_tfm_list.append(transforms.ToTensor())
    train_tfm_list.append(RandomRotate(cfg.get('rotate_degrees', 20)))
    train_tfm_list.append(normalize)
    train_tfm = transforms.Compose(train_tfm_list)

    test_tfm = transforms.Compose([
        transforms.Resize(cfg['resize_test']),
        transforms.ToTensor(),
        normalize,
    ])

    # ── Datasets ──
    ds_train = DatasetProcessing(img_dir, train_file, transform=train_tfm)
    ds_test  = DatasetProcessing(img_dir, test_file,  transform=test_tfm)
    ds_train.cache_images(resize=cfg['resize_train'])
    ds_test.cache_images(resize=cfg['resize_test'])

    # Check for weighted sampling
    use_weighted = cfg.get('use_weighted_sampler', False)
    if use_weighted:
        all_labels = []
        for i in range(len(ds_train)):
            all_labels.append(int(ds_train.labels[i]))
        class_counts = np.bincount(all_labels)
        sample_weights = [1.0 / class_counts[l] for l in all_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                        replacement=True)
        ds_train_indexed = IndexedDataset(ds_train)
        loader_train = DataLoader(ds_train_indexed, batch_size=cfg['batch_size'],
                                  sampler=sampler, num_workers=cfg['num_workers'])
        print(f'[Fold {fold_idx}] Using WeightedRandomSampler. '
              f'Class counts: {class_counts.tolist()}')
    else:
        loader_train = DataLoader(ds_train, batch_size=cfg['batch_size'],
                                  shuffle=True, num_workers=cfg['num_workers'])

    loader_test = DataLoader(ds_test, batch_size=cfg['batch_size_test'],
                              shuffle=False, num_workers=cfg['num_workers'])

    print(f'[Fold {fold_idx}] Train: {len(ds_train)} images, '
          f'Test: {len(ds_test)} images')

    # ── Load teacher (same fold, validation-best checkpoint — not averaged) ──
    teacher = ResNet50Teacher()
    ckpt_path = resolve_teacher_checkpoint(fold_idx, cfg, _PROJECT)
    print(f'[Fold {fold_idx}] Loading teacher from: {ckpt_path}')
    teacher.load_state_dict(torch.load(ckpt_path, map_location=device,
                                       weights_only=True))
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Pre-compute teacher soft labels (sequential, non-weighted)
    loader_train_sequential = DataLoader(ds_train, batch_size=cfg['batch_size'],
                                         shuffle=False, num_workers=cfg['num_workers'])
    print('Computing teacher soft labels on train set...')
    teacher_train_probs, teacher_train_labels = teacher_predict(
        teacher, loader_train_sequential, device)

    # ── Student ──
    student = get_student(cfg.get('student_arch', 'resnet18'),
                          num_classes=cfg['num_classes'])
    student.to(device)
    print(f'Student architecture: {cfg.get("student_arch", "resnet18")}, '
          f'params: {sum(p.numel() for p in student.parameters()) / 1e6:.1f}M')

    # ── Optimizer ──
    opt_name = cfg.get('optimizer', 'sgd').lower()
    if opt_name == 'adamw':
        optimizer = optim.AdamW(
            student.parameters(),
            lr=cfg['lr'],
            weight_decay=cfg['weight_decay'],
        )
    else:
        optimizer = optim.SGD(
            student.parameters(),
            lr=cfg['lr'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'],
        )

    # ── LR Scheduler ──
    if cfg.get('lr_scheduler', 'step') == 'cosine':
        eta_min = cfg.get('cosine_eta_min', 0.0)
        if eta_min <= 0.0:
            eta_min = cfg['lr'] * 1e-4
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['epochs'], eta_min=eta_min,
        )
        print(f'Using CosineAnnealingLR, T_max={cfg["epochs"]}, '
              f'eta_min={eta_min:.2e}')
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg['lr_steps'], gamma=cfg['lr_decay'],
        )

    # ── MixUp ──
    use_mixup = cfg.get('use_mixup', False)
    mixup_alpha = cfg.get('mixup_alpha', 0.2)
    if use_mixup:
        print(f'Using MixUp with alpha={mixup_alpha}')

    best_acc = 0.0
    best_epoch = 0
    start_time = time.time()

    # ── Training loop ──
    for epoch in range(cfg['epochs']):
        student.train()
        loss_m     = AverageMeter()
        loss_kd_m  = AverageMeter()
        loss_ce_m  = AverageMeter()

        for batch_idx, batch in enumerate(loader_train):
            if len(batch) == 4:
                # Weighted sampler: IndexedDataset returns (img, label, idx, extra)
                img, label, sample_idx, _ = batch
                t_probs = teacher_train_probs[sample_idx].to(device)
            else:
                # Sequential: DatasetProcessing returns (img, label, extra)
                img, label, _ = batch
                idx_start = batch_idx * cfg['batch_size']
                idx_end = idx_start + img.size(0)
                t_probs = teacher_train_probs[idx_start:idx_end].to(device)

            B = img.size(0)
            img = img.to(device)
            label = label.to(device)

            # MixUp (only when not using weighted sampler — incompatible)
            if use_mixup and not use_weighted:
                img, label_a, label_b, lam = mixup_data(img, label, mixup_alpha)

            # Student forward
            s_logits = student(img)

            # KD loss
            if use_mixup and not use_weighted:
                # MixUp: interpolate KD loss between two targets
                loss_a, kd_a, ce_a = compute_kd_loss(
                    s_logits, t_probs, label_a,
                    method=cfg.get('kd_method', 'dkd'),
                    T=cfg.get('temperature', 3.0),
                    alpha_kd=cfg.get('alpha_kd', 0.7),
                    alpha_dkd=cfg.get('alpha_dkd', 0.5),
                )
                loss_b, kd_b, ce_b = compute_kd_loss(
                    s_logits, t_probs, label_b,
                    method=cfg.get('kd_method', 'dkd'),
                    T=cfg.get('temperature', 3.0),
                    alpha_kd=cfg.get('alpha_kd', 0.7),
                    alpha_dkd=cfg.get('alpha_dkd', 0.5),
                )
                loss = lam * loss_a + (1 - lam) * loss_b
                kd   = lam * kd_a   + (1 - lam) * kd_b
                ce   = lam * ce_a   + (1 - lam) * ce_b
            else:
                loss, kd, ce = compute_kd_loss(
                    s_logits, t_probs, label,
                    method=cfg.get('kd_method', 'dkd'),
                    T=cfg.get('temperature', 3.0),
                    alpha_kd=cfg.get('alpha_kd', 0.7),
                    alpha_dkd=cfg.get('alpha_dkd', 0.5),
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_m.update(loss.item(), B)
            loss_kd_m.update(kd.item() if isinstance(kd, torch.Tensor) else kd, B)
            loss_ce_m.update(ce.item() if isinstance(ce, torch.Tensor) else ce, B)

        scheduler.step()
        elapsed = (time.time() - start_time) / 60.0

        # ── Log training ──
        logger.log({
            'type': 'train', 'fold': fold_idx, 'epoch': epoch,
            'lr': get_lr(optimizer),
            'loss': round(loss_m.avg, 6),
            'loss_kd': round(loss_kd_m.avg, 6),
            'loss_ce': round(loss_ce_m.avg, 6),
            'elapsed_min': round(elapsed, 3),
        })

        # ── Evaluate ──
        eval_start = cfg.get('eval_start_epoch', 5)
        if epoch >= eval_start or epoch == cfg['epochs'] - 1:
            _, preds, labels = student_predict(student, loader_test, device)
            acc = accuracy_score(labels, preds)

            result, ave_acc, report_str = report_precision_se_sp_yi(
                preds.numpy(), labels.numpy(),
            )
            cm = confusion_matrix(labels, preds, labels=list(range(cfg['num_classes'])))

            is_best = acc > best_acc
            if is_best:
                best_acc = acc
                best_epoch = epoch
                torch.save(student.state_dict(), out_dir / f'fold_{fold_idx}_best.pth')

            logger.log({
                'type': 'test', 'fold': fold_idx, 'epoch': epoch,
                'cls_acc': round(float(acc), 4),
                'best_acc': round(float(best_acc), 4),
                'best_epoch': best_epoch,
                'is_best': is_best,
                'confusion_matrix': cm.tolist(),
            })

    # ── Summary ──
    total_min = (time.time() - start_time) / 60.0
    logger.log({
        'type': 'summary', 'fold': fold_idx,
        'best_acc': round(float(best_acc), 4),
        'best_epoch': best_epoch,
        'total_min': round(total_min, 2),
    })
    print(f'[Fold {fold_idx}] Best Accuracy: {best_acc:.4f} at epoch {best_epoch}')

    return best_acc


# ── CLI ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Classification KD Training — hyperparameter tuning ready')

    # Config file (new)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file. Overrides default config_kd.py values.')

    # Quick CLI overrides (keep existing ones for backward compatibility)
    parser.add_argument('--folds', type=str, default='0',
                        help='Comma-separated fold indices, e.g. "0,1,2,3,4"')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--kd_method', type=str, default=None,
                        choices=['vanilla', 'dkd'])
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--alpha_kd', type=float, default=None)
    parser.add_argument('--alpha_dkd', type=float, default=None)
    parser.add_argument('--student_arch', type=str, default=None,
                        choices=['resnet18', 'resnet34'])
    parser.add_argument('--use_weighted_sampler', type=lambda x: x.lower() == 'true',
                        default=None)
    parser.add_argument('--use_mixup', type=lambda x: x.lower() == 'true',
                        default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--teacher_dir', type=str, default=None,
                        help='Directory containing fold_X_best.pth teacher weights '
                             '(default: auto-search mode/ then logs/<latest>/)')
    parser.add_argument('--teacher_ckpt', type=str, default=None,
                        help='Explicit teacher checkpoint path; use {fold} for per-fold file')
    args = parser.parse_args()

    # ── Build config: DEFAULT → JSON file → CLI overrides ──
    cfg = copy.deepcopy(DEFAULT_CFG)

    # Layer 1: JSON config file
    if args.config:
        file_cfg = load_config(args.config)
        cfg = deep_update(cfg, file_cfg)
        print(f'Loaded config from: {args.config}')

    # Layer 2: CLI arguments (highest priority)
    cli_overrides = {}
    cli_keys = set(cfg) | {'teacher_dir', 'teacher_ckpt'}
    for k, v in vars(args).items():
        if v is not None and k in cli_keys:
            cli_overrides[k] = v
    cfg.update(cli_overrides)

    if cli_overrides:
        print(f'CLI overrides: {list(cli_overrides.keys())}')

    fold_indices = [int(x.strip()) for x in args.folds.split(',')]

    # Output dir
    if args.out_dir:
        out_root = Path(args.out_dir)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_root = _PROJECT / 'module3_output_kd' / ts
    out_root.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_root / 'run_config.json', 'w', encoding='utf-8') as f:
        json.dump({**cfg, 'out_dir': str(out_root), 'folds': fold_indices},
                  f, indent=2, ensure_ascii=False)

    print(f'Output: {out_root}')
    print(f'Training folds: {fold_indices}')
    print(f'Student: {cfg["student_arch"]} | KD: {cfg["kd_method"]} | '
          f'Epochs: {cfg["epochs"]} | LR: {cfg["lr"]}')

    accs = []
    for fi in fold_indices:
        print(f'\n{"="*60}\n  Fold {fi}\n{"="*60}')
        fold_dir = out_root / f'fold_{fi}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        acc = train_one_fold(fi, cfg, out_root)
        accs.append(acc)

    print(f'\n{"="*60}')
    print('All folds completed.')
    for fi, acc in zip(fold_indices, accs):
        print(f'  fold_{fi}: best Accuracy = {acc:.4f}')
    if len(accs) > 1:
        print(f'  Mean Accuracy: {np.mean(accs):.4f} +- {np.std(accs):.4f}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
