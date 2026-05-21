# ──────────────────────────────────────────────────────────────────────────────
# plot.py  —  Read fold_*.jsonl logs produced by train.py → publication plots
#
# Usage:
#   python plot.py                            # reads ./logs, saves to ./plots
#   python plot.py --log_dir ./logs --plot_dir ./plots
#
# Output (all in plots/):
#   train_loss_mean.png          mean ± std total training loss, 5-fold
#   train_loss_folds.png         individual fold losses
#   train_loss_components.png    cls / cou / cou2cls / total (5-fold mean)
#   lr_schedule.png              learning rate vs epoch
#   cls_fusion_acc.png           CLS vs Fusion accuracy, mean ± std
#   acc_gain.png                 Fusion − CLS accuracy gain
#   mae_mse.png                  MAE & MSE, mean ± std
#   acc_vs_mae.png               dual-axis: fusion_acc (L) vs MAE (R)
#   per_class_se.png             per-class SE at best epoch
#   confusion_matrix.png         5-fold avg normalised confusion matrix
#   roc_curves.png               per-class ROC (OvR) from best_predictions.npz
#   summary.txt                  numerical summary table
#   config.txt                   experiment config recovered from meta records
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', required=True, type=str)
parser.add_argument('--plot_dir', default='./plots', type=str)
args = parser.parse_args()
print(args)
LOG_DIR  = Path(args.log_dir)
RUN_NAME = LOG_DIR.name
PLOT_DIR = Path(args.plot_dir) / RUN_NAME
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────
SMOOTH_SIGMA = 1.2
DPI          = 300
PALETTE      = ['#2c7bb6', '#d7191c', '#1a9641', '#fdae61', '#8856a7']
GREY         = '#777777'
GRADE_LABELS = ['Grade 0\n(≤5)', 'Grade 1\n(6–20)', 'Grade 2\n(21–50)', 'Grade 3\n(>50)']
GRADE_LABELS_SHORT = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3']

matplotlib.rcParams.update({
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linestyle':   '--',
})


# ─────────────────────────────────────────────
# 1. Load all folds into a single DataFrame
# ─────────────────────────────────────────────
def load_logs(log_dir: Path) -> pd.DataFrame:
    files = sorted(log_dir.glob('fold_*.jsonl'))
    if not files:
        raise FileNotFoundError(
            f'No fold_*.jsonl found in {log_dir}.\nRun train.py first.'
        )
    records = []
    for f in files:
        with open(f, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 2. Helpers
# ─────────────────────────────────────────────
def smooth(arr: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(arr)
    out  = arr.copy()
    if mask.sum() > 2:
        out[mask] = gaussian_filter1d(arr[mask], SMOOTH_SIGMA)
    return out


def fold_matrix(df_type: pd.DataFrame, col: str):
    """Returns (epochs array, matrix shape n_folds × n_epochs)."""
    epochs = np.array(sorted(df_type['epoch'].unique()))
    folds  = sorted(df_type['fold'].unique())
    mat    = np.full((len(folds), len(epochs)), np.nan)
    for fi, fold in enumerate(folds):
        sub = df_type[df_type['fold'] == fold].set_index('epoch')[col]
        for ei, e in enumerate(epochs):
            mat[fi, ei] = sub.get(e, np.nan)
    return epochs, mat


def band(ax, x, mean, std, color, label, lw=2.2, alpha=0.18):
    ax.plot(x, smooth(mean), color=color, lw=lw, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha)


def save(fig, name: str):
    p = PLOT_DIR / name
    fig.savefig(p, dpi=DPI, bbox_inches='tight')
    print(f'  → {p}')
    plt.close(fig)


# ─────────────────────────────────────────────
# 3. Load & split
# ─────────────────────────────────────────────
print(f'Loading logs from {LOG_DIR} …')
df    = load_logs(LOG_DIR)
meta  = df[df['type'] == 'meta'].copy()
train = df[df['type'] == 'train'].copy()
test  = df[df['type'] == 'test'].copy()
print(f'  {len(train)} train rows, {len(test)} test rows '
      f'across {df["fold"].nunique()} folds.\n')


# ─────────────────────────────────────────────
# 4. Save experiment config
# ─────────────────────────────────────────────
def write_config():
    if meta.empty:
        return
    cfg_path = PLOT_DIR / 'config.txt'
    with open(cfg_path, 'w', encoding='utf-8') as f:
        f.write('===== Experiment Configuration =====\n\n')
        # use fold 0 as representative
        row = meta[meta['fold'] == meta['fold'].min()].iloc[0].dropna()
        for k, v in row.items():
            if k not in ('type', 'time', 'train_file', 'test_file'):
                f.write(f'{k:<20} {v}\n')
        # include file paths separately
        for k in ('train_file', 'test_file'):
            if k in row:
                f.write(f'{k:<20} {row[k]}\n')
    print(f'  → {cfg_path}')

write_config()


# ─────────────────────────────────────────────
# 5. Training loss — mean ± std
# ─────────────────────────────────────────────
def plot_train_loss_mean():
    epochs, mat = fold_matrix(train, 'loss')
    mean, std   = np.nanmean(mat, 0), np.nanstd(mat, 0)
    fig, ax = plt.subplots(figsize=(9, 5))
    band(ax, epochs, mean, std, PALETTE[0], 'Mean train loss')
    ax.set(xlabel='Epoch', ylabel='Total Loss',
           title='Average Training Loss — 5-fold CV')
    ax.title.set_fontweight('bold'); ax.legend()
    save(fig, 'train_loss_mean.png')

plot_train_loss_mean()


# ─────────────────────────────────────────────
# 6. Training loss — individual folds
# ─────────────────────────────────────────────
def plot_train_loss_folds():
    epochs, mat = fold_matrix(train, 'loss')
    fig, ax = plt.subplots(figsize=(9, 5))
    for fi in range(mat.shape[0]):
        ax.plot(epochs, smooth(mat[fi]), color=PALETTE[fi % len(PALETTE)],
                lw=1.8, alpha=0.85, label=f'Fold {fi}')
    ax.set(xlabel='Epoch', ylabel='Total Loss', title='Training Loss by Fold')
    ax.title.set_fontweight('bold'); ax.legend()
    save(fig, 'train_loss_folds.png')

plot_train_loss_folds()


# ─────────────────────────────────────────────
# 7. Loss components
# ─────────────────────────────────────────────
def plot_loss_components():
    components = [
        ('loss_cls',    'CLS loss',      PALETTE[0], '-'),
        ('loss_cou',    'Counting loss', PALETTE[1], '-'),
        ('loss_cou2cls','Cou→CLS loss',  PALETTE[2], '-'),
        ('loss',        'Total loss',    '#333333',  '--'),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    for col, label, color, ls in components:
        if col not in train.columns:
            continue
        epochs, mat = fold_matrix(train, col)
        ax.plot(epochs, smooth(np.nanmean(mat, 0)),
                color=color, lw=2.2, ls=ls, label=label)
    ax.set(xlabel='Epoch', ylabel='Loss',
           title='Loss Components — 5-fold Mean')
    ax.title.set_fontweight('bold'); ax.legend()
    save(fig, 'train_loss_components.png')

plot_loss_components()


# ─────────────────────────────────────────────
# 8. LR schedule
# ─────────────────────────────────────────────
def plot_lr_schedule():
    if 'lr' not in train.columns:
        return
    fold0 = train[train['fold'] == train['fold'].min()].sort_values('epoch')
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.step(fold0['epoch'], fold0['lr'], where='post', color=PALETTE[0], lw=2)
    ax.set(xlabel='Epoch', ylabel='Learning Rate',
           title='Learning Rate Schedule (Fold 0, representative)')
    ax.title.set_fontweight('bold')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    save(fig, 'lr_schedule.png')

plot_lr_schedule()


# ─────────────────────────────────────────────
# 9. CLS vs Fusion accuracy
# ─────────────────────────────────────────────
def plot_cls_fusion_acc():
    epochs_c, mat_c = fold_matrix(test, 'cls_acc')
    epochs_f, mat_f = fold_matrix(test, 'fusion_acc')
    epochs = epochs_f

    mean_c, std_c = np.nanmean(mat_c, 0), np.nanstd(mat_c, 0)
    mean_f, std_f = np.nanmean(mat_f, 0), np.nanstd(mat_f, 0)

    fig, ax = plt.subplots(figsize=(9, 5))
    band(ax, epochs, mean_c, std_c, PALETTE[0], 'CLS branch')
    band(ax, epochs, mean_f, std_f, PALETTE[1], 'Fusion branch')

    best_idx = int(np.nanargmax(mean_f))
    best_ep  = int(epochs[best_idx])
    ax.axvline(best_ep, color=GREY, lw=1, ls=':', label=f'Best epoch {best_ep}')

    ax.set(xlabel='Epoch', ylabel='Accuracy',
           title='Classification Accuracy — CLS vs Fusion (5-fold)')
    ax.title.set_fontweight('bold')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    ax.legend()
    save(fig, 'cls_fusion_acc.png')
    return epochs, mean_c, mean_f, best_ep

epochs_t, mean_cls, mean_fusion, best_epoch = plot_cls_fusion_acc()
print(f'  Best epoch (fusion): {best_epoch}\n')


# ─────────────────────────────────────────────
# 10. Accuracy gain
# ─────────────────────────────────────────────
def plot_acc_gain():
    gap = mean_fusion - mean_cls
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs_t, smooth(gap), color=PALETTE[2], lw=2.2, label='Fusion − CLS')
    ax.axhline(0, color=GREY, lw=0.8, ls='--')
    ax.set(xlabel='Epoch', ylabel='Accuracy gain',
           title='Fusion Accuracy Gain over CLS Branch')
    ax.title.set_fontweight('bold')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
    ax.legend()
    save(fig, 'acc_gain.png')

plot_acc_gain()


# ─────────────────────────────────────────────
# 11. MAE & MSE
# ─────────────────────────────────────────────
def plot_mae_mse():
    epochs_a, mat_a = fold_matrix(test, 'MAE')
    epochs_m, mat_m = fold_matrix(test, 'MSE')
    mean_a, std_a   = np.nanmean(mat_a, 0), np.nanstd(mat_a, 0)
    mean_m, std_m   = np.nanmean(mat_m, 0), np.nanstd(mat_m, 0)

    fig, ax = plt.subplots(figsize=(9, 5))
    band(ax, epochs_a, mean_a, std_a, PALETTE[0], 'MAE')
    band(ax, epochs_m, mean_m, std_m, PALETTE[1], 'MSE')
    ax.set(xlabel='Epoch', ylabel='Error (lesion count)',
           title='Regression Error — MAE & MSE (5-fold)')
    ax.title.set_fontweight('bold'); ax.legend()
    save(fig, 'mae_mse.png')
    return epochs_a, mean_a

epochs_reg, mean_mae = plot_mae_mse()


# ─────────────────────────────────────────────
# 12. Dual-axis: Fusion Acc vs MAE
# ─────────────────────────────────────────────
def plot_acc_vs_mae():
    common  = np.intersect1d(epochs_t, epochs_reg)
    acc_aln = mean_fusion[np.isin(epochs_t,   common)]
    mae_aln = mean_mae   [np.isin(epochs_reg, common)]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    ax2.spines['right'].set_visible(True)
    ax2.spines['top'].set_visible(False)

    l1, = ax1.plot(common, smooth(acc_aln), color=PALETTE[0], lw=2.2, label='Fusion Acc')
    l2, = ax2.plot(common, smooth(mae_aln), color=PALETTE[1], lw=2.2,
                   ls='--', label='MAE')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Fusion Accuracy', color=PALETTE[0])
    ax2.set_ylabel('MAE (lesion count)', color=PALETTE[1])
    ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    ax1.tick_params(axis='y', colors=PALETTE[0])
    ax2.tick_params(axis='y', colors=PALETTE[1])
    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()], loc='center right')
    ax1.set_title('Multi-task Consistency: Fusion Acc vs MAE', fontweight='bold')
    ax1.grid(True, alpha=0.3, ls='--')
    save(fig, 'acc_vs_mae.png')

plot_acc_vs_mae()


# ─────────────────────────────────────────────
# 13. Per-class Sensitivity
# ─────────────────────────────────────────────
def plot_per_class_se():
    best_rows = test[test['epoch'] == best_epoch]
    if 'fusion_se' not in best_rows.columns or best_rows.empty:
        return

    n = 4
    se_mat = np.full((len(best_rows), n), np.nan)
    for i, (_, row) in enumerate(best_rows.iterrows()):
        vals = row['fusion_se']
        if isinstance(vals, list) and len(vals) >= n:
            se_mat[i] = vals[:n]

    se_mean = np.nanmean(se_mat, 0)
    se_std  = np.nanstd(se_mat,  0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(n), se_mean, width=0.55,
                  color=PALETTE[:n], alpha=0.85, zorder=3)
    ax.errorbar(range(n), se_mean, yerr=se_std, fmt='none',
                color='#333333', capsize=5, lw=1.5, zorder=4)
    for bar, v in zip(bars, se_mean):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_xticks(range(n)); ax.set_xticklabels(GRADE_LABELS)
    ax.set_ylabel('Sensitivity (SE)'); ax.set_ylim(0, 1.12)
    ax.set_title(f'Per-class Sensitivity at Epoch {best_epoch} — Fusion Branch',
                 fontweight='bold')
    save(fig, 'per_class_se.png')

plot_per_class_se()


# ─────────────────────────────────────────────
# 14. Confusion matrix
# ─────────────────────────────────────────────
def plot_confusion_matrix():
    best_rows = test[test['epoch'] == best_epoch]
    if 'confusion_matrix' not in best_rows.columns or best_rows.empty:
        return

    n = 4
    cm_sum = np.zeros((n, n))
    count  = 0
    for _, row in best_rows.iterrows():
        cm = row['confusion_matrix']
        if isinstance(cm, list):
            cm_sum += np.array(cm, dtype=float); count += 1
    if count == 0:
        return

    cm_norm = cm_sum / cm_sum.sum(axis=1, keepdims=True)
    cmap    = LinearSegmentedColormap.from_list('blues', ['#ffffff', '#2c7bb6'])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Proportion')
    ax.set_xticks(range(n)); ax.set_xticklabels(GRADE_LABELS_SHORT, rotation=30, ha='right')
    ax.set_yticks(range(n)); ax.set_yticklabels(GRADE_LABELS_SHORT)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix (normalised, 5-fold avg, epoch {best_epoch})',
                 fontweight='bold')
    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=11,
                    fontweight='bold', color='white' if v > 0.55 else 'black')
    fig.tight_layout()
    save(fig, 'confusion_matrix.png')

plot_confusion_matrix()


# ─────────────────────────────────────────────
# 15. ROC curves from best_predictions.npz
# ─────────────────────────────────────────────
def plot_roc_curves():
    n = 4
    npz_files = sorted(LOG_DIR.glob('fold_*_best_predictions.npz'))
    if not npz_files:
        print('  Skipping roc_curves.png — no .npz files found.')
        return

    # accumulate all folds' predictions
    all_y_true    = []
    all_prob_fusion = []
    for f in npz_files:
        data = np.load(f)
        all_y_true.append(data['y_true'])
        all_prob_fusion.append(data['prob_fusion'])

    y_true      = np.concatenate(all_y_true)
    prob_fusion = np.concatenate(all_prob_fusion)

    y_bin = label_binarize(y_true, classes=list(range(n)))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n):
        fpr, tpr, _ = roc_curve(y_bin[:, i], prob_fusion[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=PALETTE[i], lw=2,
                label=f'{GRADE_LABELS_SHORT[i]} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls='--', label='Random')
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
           title='ROC Curves — Fusion Branch (all folds pooled, best epoch)',
           xlim=[0, 1], ylim=[0, 1.02])
    ax.title.set_fontweight('bold')
    ax.legend(loc='lower right')
    save(fig, 'roc_curves.png')

plot_roc_curves()


# ─────────────────────────────────────────────
# 16. Numerical summary
# ─────────────────────────────────────────────
def write_summary():
    best_rows = test[test['epoch'] == best_epoch]
    metrics   = ['cls_acc', 'fusion_acc', 'MAE', 'MSE']
    labels    = ['CLS Acc', 'Fusion Acc', 'MAE', 'MSE']

    lines = [
        f'===== 5-fold Summary at best epoch {best_epoch} =====',
        f'{"Metric":<14} {"Mean":>8} {"Std":>8} {"Min":>8} {"Max":>8}',
        '-' * 50,
    ]
    for col, label in zip(metrics, labels):
        vals = pd.to_numeric(best_rows[col], errors='coerce').dropna().values
        if len(vals) == 0:
            continue
        lines.append(f'{label:<14} {vals.mean():>8.4f} {vals.std():>8.4f}'
                     f' {vals.min():>8.4f} {vals.max():>8.4f}')

    lines += ['', 'Per-fold values:',
              f'{"Fold":<6} ' + ' '.join(f'{l:>11}' for l in labels)]
    for _, row in best_rows.sort_values('fold').iterrows():
        vals_str = ' '.join(
            f'{float(row[c]):>11.4f}' if c in row and pd.notna(row[c]) else f'{"N/A":>11}'
            for c in metrics
        )
        lines.append(f'Fold {int(row["fold"])}  {vals_str}')

    out = PLOT_DIR / 'summary.txt'
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'  → {out}')
    print('\n'.join(lines))

write_summary()

print(f'\nAll done. Outputs saved to {PLOT_DIR.resolve()}')