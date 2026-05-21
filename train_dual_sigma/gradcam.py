"""
gradcam.py — Grad-CAM 热力图 + Pseudo Mask 生成脚本
=====================================================
输出三类文件（对应模块三需要的硬知识+软知识）：
  - *_cam.png      彩色热力图叠加原图（软知识，给ADKD蒸馏用）
  - *_cam.npy      热力图原始float32数值矩阵（模块三直接np.load读取）
  - *_mask.png     二值化Pseudo Mask（硬知识，给分割Loss用）
  - *_mask.npy     Mask原始bool矩阵

使用方法：
  # 随机抽样10张看效果（默认）
  python gradcam.py

  # 指定fold、pth路径、图片目录
  python gradcam.py ^
      --pth_dir   C:/path/to/logs/20260516_230852 ^
      --data_path C:/path/to/JPEGImages ^
      --test_file C:/path/to/NNEW_test_0.txt ^
      --fold      0 ^
      --n_samples 10 ^
      --threshold 0.5 ^
      --out_dir   ./gradcam_output
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ── 把项目根目录加入path，确保能import model
import sys
sys.path.insert(0, str(Path(__file__).parent))
from model.resnet50 import resnet50

# ─────────────────────────────────────────────
# 默认路径（直接运行时用，也可通过命令行覆盖）
# ─────────────────────────────────────────────
DEFAULTS = dict(
    pth_dir   = r'C:\Users\28268\Desktop\LDL-master\LDL-master\train_dual_sigma\logs\20260516_230852',
    data_path = r'C:\Users\28268\Desktop\LDL-master\LDL-master\code\ACNE04\Classification\JPEGImages',
    test_file = r'C:\Users\28268\Desktop\LDL-master\LDL-master\code\ACNE04\Classification\NNEW_test_0.txt',
    fold      = 0,
    n_samples = 10,       # 随机抽几张；-1表示全部
    threshold = 0.5,      # 二值化阈值（热力图归一化到0~1后，>threshold的区域为前景）
    out_dir   = r'.\gradcam_output',
)

CLASS_NAMES = ['Grade0 (≤5)', 'Grade1 (6-20)', 'Grade2 (21-50)', 'Grade3 (>50)']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────
# 图像预处理（和train.py的test transform一致）
# ─────────────────────────────────────────────
normalize = transforms.Normalize(
    mean=[0.45815152, 0.361242,  0.29348266],
    std= [0.2814769,  0.226306,  0.20132513],
)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])


# ─────────────────────────────────────────────
# Grad-CAM 实现
# 原理：对目标类别的logit对layer4的feature map求梯度，
#       梯度全局平均池化得到每个channel的权重，
#       加权求和后ReLU得到热力图
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model: torch.nn.Module):
        self.model     = model
        self.gradients = None   # 保存layer4的梯度
        self.features  = None   # 保存layer4的特征图

        # 在layer4上挂钩子
        self.model.layer4.register_forward_hook(self._save_features)
        self.model.layer4.register_full_backward_hook(self._save_gradients)

    def _save_features(self, module, input, output):
        self.features = output   # (1, 2048, 7, 7)

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]   # (1, 2048, 7, 7)

    def __call__(self, img_tensor: torch.Tensor, target_class: int = None):
        """
        img_tensor: (1, 3, 224, 224)，已归一化
        target_class: None则取预测类别

        返回：
            cam        (224, 224) float32，已归一化到 [0, 1]
            pred_class int
            pred_prob  float
        """
        self.model.eval()
        img_tensor = img_tensor.to(device)

        # 前向传播
        self.model.zero_grad()
        cls, cou, cou2cls = self.model(img_tensor, None)

        # cls分支作为Grad-CAM目标（分类头）
        if target_class is None:
            target_class = cls.argmax(dim=1).item()
        pred_prob = cls[0, target_class].item()

        # 对目标类别的score反向传播
        score = cls[0, target_class]
        score.backward()

        # 梯度全局平均池化 → 每个channel的权重
        # gradients: (1, 2048, 7, 7) → weights: (2048,)
        weights = self.gradients.mean(dim=[0, 2, 3])   # (2048,)

        # 特征图加权求和
        # features: (1, 2048, 7, 7)
        cam = (weights[:, None, None] * self.features[0]).sum(dim=0)   # (7, 7)
        cam = F.relu(cam)   # 只保留正激活区域

        # 上采样到224×224
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)

        # 归一化到 [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32), target_class, pred_prob


# ─────────────────────────────────────────────
# 可视化工具
# ─────────────────────────────────────────────
def cam_to_heatmap_overlay(cam: np.ndarray, orig_img: np.ndarray) -> np.ndarray:
    """
    cam:      (224, 224) float32 in [0,1]
    orig_img: (224, 224, 3) uint8 RGB

    返回：(224, 224, 3) uint8，热力图叠加在原图上
    """
    # 热力图着色（JET colormap）
    heatmap = cv2.applyColorMap(
        (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
    )   # BGR
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)   # 转RGB

    # 叠加（alpha=0.4热力图，0.6原图）
    orig_bgr = orig_img[:, :, ::-1].copy()   # RGB→BGR for blending
    overlay  = cv2.addWeighted(
        orig_bgr, 0.6,
        cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR), 0.4,
        0
    )
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


# 在 cam_to_mask 函数里改一行
def cam_to_mask(cam: np.ndarray, threshold: float) -> np.ndarray:
    cam_uint8 = (cam * 255).astype(np.uint8)
    # Otsu自动找最优分割阈值
    _, mask = cv2.threshold(cam_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask.astype(bool)


# ─────────────────────────────────────────────
# 读取测试集文件列表
# ─────────────────────────────────────────────
def load_test_list(test_file: str):
    filenames, labels = [], []
    with open(test_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                filenames.append(parts[0])
                labels.append(int(parts[1]))
    return filenames, labels


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────
def run(args):
    out_dir = Path(args.out_dir) / f'fold_{args.fold}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载模型
    pth_path = Path(args.pth_dir) / f'fold_{args.fold}_best.pth'
    print(f'Loading model: {pth_path}')
    model = resnet50().to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()

    grad_cam = GradCAM(model)

    # ── 加载测试集
    filenames, labels = load_test_list(args.test_file)
    print(f'Test set: {len(filenames)} images')

    # ── 随机抽样
    if args.n_samples > 0 and args.n_samples < len(filenames):
        idxs = random.sample(range(len(filenames)), args.n_samples)
    else:
        idxs = list(range(len(filenames)))
    print(f'Processing {len(idxs)} images...\n')

    # ── 逐张处理
    results_summary = []
    for i, idx in enumerate(idxs):
        fname = filenames[idx]
        true_label = labels[idx]
        img_path = Path(args.data_path) / fname

        # 读原图
        orig_pil = Image.open(img_path).convert('RGB')
        orig_224 = orig_pil.resize((224, 224), Image.BILINEAR)
        orig_np  = np.array(orig_224)   # (224, 224, 3) uint8

        # 预处理
        img_tensor = preprocess(orig_pil).unsqueeze(0)   # (1, 3, 224, 224)

        # Grad-CAM
        cam, pred_class, pred_prob = grad_cam(img_tensor)

        # 生成overlay和mask
        overlay = cam_to_heatmap_overlay(cam, orig_np)
        mask    = cam_to_mask(cam, args.threshold)

        # 文件名（去掉路径和扩展名）
        stem = Path(fname).stem

        # 保存彩色热力图叠加图（软知识可视化）
        cam_png_path = out_dir / f'{stem}_cam.png'
        Image.fromarray(overlay).save(cam_png_path)

        # 保存热力图数值（软知识，模块三读npy）
        cam_npy_path = out_dir / f'{stem}_cam.npy'
        np.save(cam_npy_path, cam)

        # 保存二值mask图（硬知识可视化）
        mask_png_path = out_dir / f'{stem}_mask.png'
        mask_vis = (mask.astype(np.uint8) * 255)
        Image.fromarray(mask_vis, mode='L').save(mask_png_path)

        # 保存mask数值（硬知识，模块三读npy）
        mask_npy_path = out_dir / f'{stem}_mask.npy'
        np.save(mask_npy_path, mask)

        results_summary.append({
            'file':       fname,
            'true':       CLASS_NAMES[true_label],
            'pred':       CLASS_NAMES[pred_class],
            'pred_prob':  round(pred_prob, 4),
            'correct':    pred_class == true_label,
        })

        print(f'[{i+1}/{len(idxs)}] {fname}')
        print(f'         True: {CLASS_NAMES[true_label]}  |  '
              f'Pred: {CLASS_NAMES[pred_class]} ({pred_prob:.3f})')
        print(f'         cam  → {cam_png_path.name}')
        print(f'         mask → {mask_png_path.name}\n')

    # ── 汇总
    n_correct = sum(r['correct'] for r in results_summary)
    print('=' * 50)
    print(f'Done. {len(idxs)} images processed.')
    print(f'Accuracy on sampled images: {n_correct}/{len(idxs)} = '
          f'{n_correct/len(idxs):.1%}')
    print(f'Output directory: {out_dir}')
    print('=' * 50)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_dir',   default=DEFAULTS['pth_dir'])
    parser.add_argument('--data_path', default=DEFAULTS['data_path'])
    parser.add_argument('--test_file', default=DEFAULTS['test_file'])
    parser.add_argument('--fold',      default=DEFAULTS['fold'],      type=int)
    parser.add_argument('--n_samples', default=DEFAULTS['n_samples'], type=int,
                        help='-1 表示处理全部测试图片')
    parser.add_argument('--threshold', default=DEFAULTS['threshold'], type=float,
                        help='热力图二值化阈值，0~1，越小mask越大')
    parser.add_argument('--out_dir',   default=DEFAULTS['out_dir'])
    args = parser.parse_args()

    run(args)
