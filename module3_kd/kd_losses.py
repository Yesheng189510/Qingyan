"""KD loss functions: Vanilla KD and DKD (Decoupled Knowledge Distillation).

Reference:
  - Hinton et al., "Distilling the Knowledge in a Neural Network", 2015
  - Zhao et al., "Decoupled Knowledge Distillation", CVPR 2022
"""

import torch
import torch.nn.functional as F


# ── Vanilla KD ──────────────────────────────────────────

def vanilla_kd_loss(student_logits, teacher_probs, labels, T=3.0, alpha=0.7):
    """Hinton's original knowledge distillation loss.

    Args:
        student_logits: (B, C) raw logits
        teacher_probs:  (B, C) softmax probabilities from teacher
        labels:         (B,) ground-truth class indices
        T:              temperature
        alpha:          KD weight (vs cross-entropy)

    Returns:
        total_loss, loss_kd, loss_ce
    """
    soft_teacher = teacher_probs ** (1.0 / T)
    soft_teacher = soft_teacher / soft_teacher.sum(dim=1, keepdim=True)

    loss_kd = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        soft_teacher,
        reduction='batchmean',
    ) * (T * T)

    loss_ce = F.cross_entropy(student_logits, labels)

    return alpha * loss_kd + (1 - alpha) * loss_ce, loss_kd, loss_ce


# ── DKD ─────────────────────────────────────────────────

def dkd_loss(student_logits, teacher_probs, labels,
             T=3.0, alpha_kd=0.7, alpha_dkd=0.5):
    """Decoupled Knowledge Distillation (Zhao et al., CVPR 2022).

    Splits KD into:
      - TCKD: binary distillation on the target class
      - NCKD: distillation on non-target classes only

    Args:
        student_logits: (B, C) raw logits
        teacher_probs:  (B, C) softmax probabilities from teacher
        labels:         (B,) ground-truth class indices
        T:              temperature
        alpha_kd:       overall KD loss weight (vs CE)
        alpha_dkd:      weight of TCKD within the KD term (vs NCKD)

    Returns:
        total_loss, loss_kd, loss_ce
    """
    B, C = student_logits.shape

    # Teacher / student soft probabilities
    t_soft = teacher_probs ** (1.0 / T)
    t_soft = t_soft / t_soft.sum(dim=1, keepdim=True)

    s_soft = F.softmax(student_logits / T, dim=1)

    # ── TCKD: binary distillation on the target class ──
    t_target = t_soft[range(B), labels]
    s_target = s_soft[range(B), labels]

    loss_tckd = F.binary_cross_entropy(s_target, t_target, reduction='mean')

    # ── NCKD: KL on non-target classes ──
    # Build masks to zero out target-class probabilities
    mask = torch.ones_like(t_soft).scatter_(1, labels.unsqueeze(1), 0.0)
    t_non = t_soft * mask
    s_non = s_soft * mask

    # Renormalize
    t_non = t_non / t_non.sum(dim=1, keepdim=True).clamp_min(1e-8)
    s_non = s_non / s_non.sum(dim=1, keepdim=True).clamp_min(1e-8)

    loss_nckd = F.kl_div(
        torch.log(s_non.clamp_min(1e-8)),
        t_non,
        reduction='batchmean',
    )

    loss_kd = alpha_dkd * loss_tckd + (1 - alpha_dkd) * loss_nckd
    loss_ce = F.cross_entropy(student_logits, labels)

    total = alpha_kd * loss_kd + (1 - alpha_kd) * loss_ce
    return total, loss_kd, loss_ce


# ── Dispatcher ──────────────────────────────────────────

def compute_kd_loss(student_logits, teacher_probs, labels,
                    method='dkd', T=3.0, alpha_kd=0.7, alpha_dkd=0.5):
    """Unified interface for KD loss computation."""
    if method == 'dkd':
        return dkd_loss(student_logits, teacher_probs, labels, T, alpha_kd, alpha_dkd)
    else:
        return vanilla_kd_loss(student_logits, teacher_probs, labels, T, alpha_kd)
