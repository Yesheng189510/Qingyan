# ──────────────────────────────────────────────────────────────────────────────
# train.py  —  LDL Acne Grading
# ──────────────────────────────────────────────────────────────────────────────
import os, json, math, warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from dataset import dataset_processing
from utils.report import report_precision_se_sp_yi, report_mae_mse
from utils.utils import AverageMeter
from model.resnet50 import resnet50
import torch.backends.cudnn as cudnn
from transforms.affine_transforms import *

# ─────────────────────────────────────────────
# Device & AMP
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
AMP_ENABLED = torch.cuda.is_available()
scaler      = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

# ─────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────
CFG = dict(
    model           = "resnet50",
    batch_size      = 32,
    batch_size_test = 64,
    lr              = 0.001,
    momentum        = 0.9,
    weight_decay    = 5e-4,
    num_workers     = 0,
    num_classes     = 4,
    epochs          = 60,
    lr_steps        = [30, 60, 90, 120],
    lr_decay        = 0.5,
    seed            = 42,
    amp             = AMP_ENABLED,
)

# 每次运行生成独立文件夹，不会覆盖历史日志
RUN_ID  = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = Path('./logs') / RUN_ID
LOG_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = 'C:/Users/28268/Desktop/LDL-master/LDL-master/code/ACNE04/Classification/JPEGImages'

np.random.seed(CFG['seed'])

# 修复①：KLDivLoss 改用 batchmean（数学上更正确，PyTorch 推荐）
EPS = 1e-8   # 修复⑤：防止 log(0) 导致 NaN


# ─────────────────────────────────────────────
# JSON Lines logger
# ─────────────────────────────────────────────
class FoldLogger:
    def __init__(self, path: Path):
        self.path = path
        path.write_text('')

    def log(self, record: dict):
        record['time'] = datetime.now().isoformat(timespec='seconds')
        line = json.dumps(record, ensure_ascii=False)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
        print(line)


# ─────────────────────────────────────────────
# GPU GenLD
# ─────────────────────────────────────────────
class GenLD:
    def __init__(self, class_num: int, device: torch.device):
        # sigma 不再固定在 __init__，每次 __call__ 传入，支持动态退火
        label_set      = torch.arange(class_num, dtype=torch.float32, device=device)
        self.label_set = label_set.unsqueeze(1)
        self.device    = device

    @torch.no_grad()
    def __call__(self, label: torch.Tensor, sigma: float):
        inv_denom  = 1.0 / (2.0 * sigma ** 2)
        norm_const = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
        label_f = label.float().to(self.device)
        dif  = self.label_set - label_f.unsqueeze(0)
        ld   = norm_const * torch.exp(-dif ** 2 * inv_denom)
        ld   = ld / ld.sum(dim=0, keepdim=True)
        ld   = ld.T
        ld_4 = torch.stack([
            ld[:, :5].sum(1),
            ld[:, 5:20].sum(1),
            ld[:, 20:50].sum(1),
            ld[:, 50:].sum(1),
        ], dim=1)
        return ld, ld_4


# ─────────────────────────────────────────────
# Parse report → flat per-class lists
# ─────────────────────────────────────────────
def parse_cls_report(report_obj, n_classes: int = 4):
    pre = [float('nan')] * n_classes
    se  = [float('nan')] * n_classes
    sp  = [float('nan')] * n_classes
    yi  = [float('nan')] * n_classes

    if isinstance(report_obj, dict):
        for i in range(n_classes):
            vals = report_obj.get(f'class{i}', [])
            if len(vals) >= 4:
                pre[i], se[i], sp[i], yi[i] = vals[:4]
        return pre, se, sp, yi

    for line in str(report_obj).strip().splitlines():
        parts = line.split()
        try:
            if parts and parts[0].startswith('class') and parts[0][5:].isdigit():
                i = int(parts[0][5:])
                if i < n_classes:
                    pre[i], se[i], sp[i], yi[i] = (
                        float(parts[1]), float(parts[2]),
                        float(parts[3]), float(parts[4])
                    )
        except (IndexError, ValueError):
            continue
    return pre, se, sp, yi


def get_lr(optimizer) -> float:
    return optimizer.param_groups[0]['lr']


# ─────────────────────────────────────────────
# Main train / test loop
# ─────────────────────────────────────────────
def trainval_test(fold_idx: int, cross_val_index: str, lam: float):

    logger   = FoldLogger(LOG_DIR / f'fold_{fold_idx}.jsonl')
    npz_path = LOG_DIR / f'fold_{fold_idx}_best_predictions.npz'

    TRAIN_FILE = (
        'C:/Users/28268/Desktop/LDL-master/LDL-master/code/'
        f'ACNE04/Classification/NNEW_trainval_{cross_val_index}.txt'
    )
    TEST_FILE = (
        'C:/Users/28268/Desktop/LDL-master/LDL-master/code/'
        f'ACNE04/Classification/NNEW_test_{cross_val_index}.txt'
    )

    logger.log({
        "type":              "meta",
        "fold":              fold_idx,
        "split":             cross_val_index,
        "cls_sigma_start":   6.0,   # cls/cou2cls分支：退火起点
        "cls_sigma_end":     1.5,   # cls/cou2cls分支：退火终点
        "cls_sigma_k":       0.1,   # cls/cou2cls分支：衰减速度
        "cou_sigma":         3.0,   # cou分支：固定sigma，保持counting精度
        "lam":               lam,
        "train_file":        TRAIN_FILE,
        "test_file":         TEST_FILE,
        **CFG,
    })

    normalize = transforms.Normalize(
        mean=[0.45815152, 0.361242,  0.29348266],
        std= [0.2814769,  0.226306,  0.20132513],
    )
    dset_train = dataset_processing.DatasetProcessing(
        DATA_PATH, TRAIN_FILE, transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            RandomRotate(rotation_range=20),
            normalize,
        ]))
    dset_test = dataset_processing.DatasetProcessing(
        DATA_PATH, TEST_FILE, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]))

    print(f'[Fold {fold_idx}] Caching datasets to RAM...')
    dset_train.cache_images(resize=256)
    dset_test.cache_images(resize=224)
    print(f'[Fold {fold_idx}] Cache done.')

    train_loader = DataLoader(
        dset_train, batch_size=CFG['batch_size'],
        shuffle=True,  num_workers=CFG['num_workers'], pin_memory=True,
    )
    test_loader = DataLoader(
        dset_test, batch_size=CFG['batch_size_test'],
        shuffle=False, num_workers=CFG['num_workers'], pin_memory=True,
    )

    cnn = resnet50().to(device)
    cudnn.benchmark = True

    params = [{'params': [v], 'lr': CFG['lr'], 'weight_decay': CFG['weight_decay']}
              for v in cnn.parameters() if v.requires_grad]
    optimizer = torch.optim.SGD(params, momentum=CFG['momentum'])


    # 改回这样
    kl_loss_1 = nn.KLDivLoss()
    kl_loss_2 = nn.KLDivLoss()
    kl_loss_3 = nn.KLDivLoss()

    gen_ld = GenLD(class_num=65, device=device)

    # ── 双sigma配置 ───────────────────────────────────────────────────────────
    # 方向B核心：cls/cou2cls 和 cou 分支用不同sigma，互不干扰
    #
    # cls_sigma（退火）：从大到小，让分类分布先平滑后锐化
    #   → 提升分类精度
    #
    # cou_sigma（固定）：保持3.0，counting任务依赖宽分布估计lesion数量
    #   → 保持MAE/MSE不退步
    #
    CLS_SIGMA_START = 6.0
    CLS_SIGMA_END   = 1.5
    CLS_K           = 0.1
    COU_SIGMA       = 3.0   # 固定，不退火

    def get_cls_sigma(epoch: int) -> float:
        return CLS_SIGMA_END + (CLS_SIGMA_START - CLS_SIGMA_END) * math.exp(-CLS_K * epoch)

    def adjust_lr(optimizer, decay=CFG['lr_decay']):
        for pg in optimizer.param_groups:
            pg['lr'] *= decay

    start           = timer()
    best_fusion_acc = 0.0

    for epoch in range(CFG['epochs']):
        if epoch in CFG['lr_steps']:
            adjust_lr(optimizer)

        # ── Train ─────────────────────────────────────────────────────────────
        cnn.train()
        m_cls = AverageMeter(); m_cou = AverageMeter()
        m_c2c = AverageMeter(); m_tot = AverageMeter()

        for b_x, b_y, b_l in train_loader:
            b_x = b_x.to(device, non_blocking=True)
            b_l = (b_l - 1).long()

            # ── 双sigma：cls分支用退火sigma，cou分支用固定sigma
            cls_sigma_now      = get_cls_sigma(epoch)
            ld_cou,  _         = gen_ld(b_l, COU_SIGMA)       # cou用固定sigma
            _,       ld_4_cls  = gen_ld(b_l, cls_sigma_now)   # cls用退火sigma

            with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                cls, cou, cou2cls = cnn(b_x, None)
                loss_cls     = kl_loss_1(torch.log(cls.clamp_min(EPS)),     ld_4_cls) * 4.0
                loss_cou     = kl_loss_2(torch.log(cou.clamp_min(EPS)),     ld_cou  ) * 65.0
                loss_cls_cou = kl_loss_3(torch.log(cou2cls.clamp_min(EPS)), ld_4_cls) * 4.0
                loss = (loss_cls + loss_cls_cou) * 0.5 * lam + loss_cou * (1.0 - lam)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            n = b_x.size(0)
            m_cls.update(loss_cls.item(),     n)
            m_cou.update(loss_cou.item(),     n)
            m_c2c.update(loss_cls_cou.item(), n)
            m_tot.update(loss.item(),         n)

        logger.log({
            "type":            "train",
            "fold":            fold_idx,
            "epoch":           epoch,
            "cls_sigma":       round(get_cls_sigma(epoch), 4),
            "cou_sigma":       COU_SIGMA,
            "lr":              round(get_lr(optimizer), 8),
            "loss_cls":        round(m_cls.avg, 6),
            "loss_cou":        round(m_cou.avg, 6),
            "loss_cou2cls":    round(m_c2c.avg, 6),
            "loss":            round(m_tot.avg, 6),
            "elapsed_total_min": round((timer() - start) / 60, 3),
        })

        # ── Test (epoch >= 9) ──────────────────────────────────────────────────
        if epoch < 9:
            continue

        cnn.eval()
        with torch.no_grad():
            test_loss = 0.0
            y_true = np.array([]); y_pred = np.array([]); y_pred_m = np.array([])
            l_true = np.array([]); l_pred = np.array([])
            prob_cls_list    = []
            prob_fusion_list = []

            for test_x, test_y, test_l in test_loader:
                test_x = test_x.to(device, non_blocking=True)
                test_y = test_y.to(device).long()

                y_true = np.hstack((y_true, test_y.cpu().numpy()))
                l_true = np.hstack((l_true, test_l.cpu().numpy()))

                cls, cou, cou2cls = cnn(test_x, None)

                # 修复②：cls/cou2cls 已经是 softmax 概率，用 nll_loss 才正确
                # nll_loss 要求输入是 log-probability
                test_loss += F.nll_loss(
                    torch.log(cou2cls.clamp_min(EPS)), test_y
                ).item()

                # cls branch
                _, preds = torch.max(cls, 1)
                y_pred   = np.hstack((y_pred, preds.cpu().numpy()))
                prob_cls_list.append(cls.cpu().numpy())   # cls 已是概率，直接存


                # softmax(prob + prob) 数学上不对；归一化才是正确操作
                # 修复④：log space fusion（product of experts）
                # log(p1) + log(p2) = log(p1*p2)，再 softmax 归一化
                # 比线性叠加概率数学上更严格
                fusion_prob = torch.log(cls.clamp_min(EPS)) + torch.log(cou2cls.clamp_min(EPS))
                fusion_prob = torch.softmax(fusion_prob, dim=1)

                _, preds_m = torch.max(fusion_prob, 1)
                y_pred_m   = np.hstack((y_pred_m, preds_m.cpu().numpy()))
                prob_fusion_list.append(fusion_prob.cpu().numpy())

                _, preds_l = torch.max(cou, 1)
                l_pred = np.hstack((l_pred, (preds_l + 1).cpu().numpy()))

            test_loss  /= len(test_loader)
            cls_acc     = float((y_pred   == y_true).mean())
            fusion_acc  = float((y_pred_m == y_true).mean())

            is_best = fusion_acc > best_fusion_acc
            if is_best:
                best_fusion_acc = fusion_acc

            _, _, cls_rpt    = report_precision_se_sp_yi(y_pred,   y_true)
            _, _, fusion_rpt = report_precision_se_sp_yi(y_pred_m, y_true)
            _, MAE, MSE, _   = report_mae_mse(l_true, l_pred, y_true)

            cls_pre,    cls_se,    cls_sp,    cls_yi    = parse_cls_report(cls_rpt)
            fusion_pre, fusion_se, fusion_sp, fusion_yi = parse_cls_report(fusion_rpt)

            cm = confusion_matrix(
                y_true.astype(int), y_pred_m.astype(int),
                labels=list(range(CFG['num_classes']))
            ).tolist()

            logger.log({
                "type":       "test",
                "fold":       fold_idx,
                "epoch":      epoch,
                "loss":       round(test_loss,  6),
                "cls_acc":    round(cls_acc,    4),
                "fusion_acc": round(fusion_acc, 4),
                "cls_pre":    [round(v, 4) for v in cls_pre],
                "cls_se":     [round(v, 4) for v in cls_se],
                "cls_sp":     [round(v, 4) for v in cls_sp],
                "cls_yi":     [round(v, 4) for v in cls_yi],
                "fusion_pre": [round(v, 4) for v in fusion_pre],
                "fusion_se":  [round(v, 4) for v in fusion_se],
                "fusion_sp":  [round(v, 4) for v in fusion_sp],
                "fusion_yi":  [round(v, 4) for v in fusion_yi],
                "MAE":        round(float(MAE), 4),
                "MSE":        round(float(MSE), 4),
                "confusion_matrix": cm,
                "is_best":    is_best,
            })

            if is_best:
                prob_cls    = np.vstack(prob_cls_list)
                prob_fusion = np.vstack(prob_fusion_list)

                logger.log({
                    "type":          "best_predictions",
                    "fold":          fold_idx,
                    "epoch":         epoch,
                    "fusion_acc":    round(fusion_acc, 4),
                    "n_samples":     int(len(y_true)),
                    "y_true":        y_true.astype(int).tolist(),
                    "y_pred_cls":    y_pred.astype(int).tolist(),
                    "y_pred_fusion": y_pred_m.astype(int).tolist(),
                })

                np.savez_compressed(
                    npz_path,
                    y_true        = y_true.astype(np.int32),
                    y_pred_cls    = y_pred.astype(np.int32),
                    y_pred_fusion = y_pred_m.astype(np.int32),
                    prob_cls      = prob_cls.astype(np.float32),
                    prob_fusion   = prob_fusion.astype(np.float32),
                )
                print(f'[Fold {fold_idx}] Best predictions saved → {npz_path}')


def main():
    cross_val_lists = ['0', '1', '2', '3', '4']
    for fold_idx, split in enumerate(cross_val_lists):
        print(f'\n\n========== Fold {fold_idx} (split {split}) ==========\n')
        trainval_test(fold_idx, split, lam=6 * 0.1)


if __name__ == "__main__":
    main()