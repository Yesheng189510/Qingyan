"""Baseline configuration for classification knowledge distillation.

This file is the SINGLE SOURCE OF TRUTH for all tunable hyperparameters.
To run an experiment, copy this dict into a JSON file, modify the values,
then pass it via:

    python train_kd.py --config my_experiment.json

Every parameter is documented with its meaning, typical range, and tuning priority.
"""

CFG = dict(
    # ═══════════════════════════════════════════════════════════════
    # 1. STUDENT MODEL
    # ═══════════════════════════════════════════════════════════════
    student_arch="resnet34",        # "resnet18" | "resnet34" | "resnet50"
    num_classes=4,                  # fixed — do not change

    # ═══════════════════════════════════════════════════════════════
    # 2. OPTIMIZER
    # ═══════════════════════════════════════════════════════════════
    lr=0.001,                       # ★★★  learning rate  [1e-4, 1e-2]
    momentum=0.9,                   # SGD momentum      [0.8, 0.99]
    weight_decay=5e-4,              # ★★   L2 penalty   [1e-5, 1e-2]
    optimizer="sgd",                # "sgd" | "adamw"

    # ═══════════════════════════════════════════════════════════════
    # 3. LEARNING RATE SCHEDULE
    # ═══════════════════════════════════════════════════════════════
    lr_scheduler="cosine",          # "step" | "cosine"
    # --- when lr_scheduler = "step" ---
    lr_steps=[30, 60],              # ★★   milestones (in epochs)
    lr_decay=0.5,                   # ★★   multiplicative factor  [0.1, 0.5]
    # --- when lr_scheduler = "cosine" ---
    cosine_eta_min=0.0,             # minimum lr (default: lr * 1e-4 if 0)

    # ═══════════════════════════════════════════════════════════════
    # 4. TRAINING LOOP
    # ═══════════════════════════════════════════════════════════════
    epochs=90,                      # ★★★  total epochs  [30, 150]
    batch_size=64,                  # ★★   per GPU/CPU   [16, 128]
    batch_size_test=128,            # test batch size
    num_workers=0,                  # DataLoader workers (0 = main process)
    seed=42,                        # ★     random seed (change for multi-run averaging)
    eval_start_epoch=5,             # start test evaluation from this epoch

    # ═══════════════════════════════════════════════════════════════
    # 5. KNOWLEDGE DISTILLATION
    # ═══════════════════════════════════════════════════════════════
    kd_method="dkd",                # ★★★  "vanilla" | "dkd"
    temperature=3.0,                # ★★★  soften teacher distribution  [1.0, 10.0]
    alpha_kd=0.7,                   # ★★★  KD loss weight (vs CE)       [0.3, 0.95]
    # --- DKD-specific ---
    alpha_dkd=0.5,                  # ★★   TCKD weight within KD term   [0.1, 0.9]
    #       alpha_dkd close to 0 → NCKD dominates (inter-class relations)
    #       alpha_dkd close to 1 → TCKD dominates (target-class only)

    # ═══════════════════════════════════════════════════════════════
    # 6. DATA AUGMENTATION
    # ═══════════════════════════════════════════════════════════════
    use_weighted_sampler=True,      # ★     balance class sampling
    use_mixup=False,                # ★★    MixUp augmentation
    mixup_alpha=0.2,                # Beta distribution alpha for MixUp [0.1, 0.4]
    rotate_degrees=20,              # random rotation range (±degrees)
    use_random_horizontal_flip=True,# horizontal flip

    # ═══════════════════════════════════════════════════════════════
    # 7. NORMALIZATION (pre-computed from training data — do NOT tune)
    # ═══════════════════════════════════════════════════════════════
    mean=[0.45815152, 0.361242, 0.29348266],
    std=[0.2814769, 0.226306, 0.20132513],

    # ═══════════════════════════════════════════════════════════════
    # 8. IMAGE SIZES (usually fixed)
    # ═══════════════════════════════════════════════════════════════
    resize_train=256,
    resize_test=224,
    crop_size=224,
)

# ═══════════════════════════════════════════════════════════════════
# TUNING PRIORITY GUIDE
# ═══════════════════════════════════════════════════════════════════
#
# ★★★  First round — biggest impact:
#        lr, epochs, kd_method, temperature, alpha_kd
#
# ★★   Second round — refinement:
#        student_arch, weight_decay, lr_steps, lr_decay, alpha_dkd,
#        use_mixup, mixup_alpha, batch_size
#
# ★    Stability / ablation:
#        seed, use_weighted_sampler, cosine_eta_min
#
# Suggested workflow:
#   1. Copy this dict to experiment.json
#   2. Change the parameters you want to test
#   3. Run: python train_kd.py --config experiment.json
#   4. Results automatically saved to module3_output_kd/<timestamp>/
#      with the full config recorded in run_config.json
# ═══════════════════════════════════════════════════════════════════
