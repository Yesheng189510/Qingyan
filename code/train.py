# library
# standard library
import os, sys

# Add current directory to path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# third-party library
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.dataset_processing import DatasetProcessing
from timeit import default_timer as timer
from utils.report import report_precision_se_sp_yi, report_mae_mse
from utils.utils import Logger, AverageMeter, time_to_str, weights_init
from utils.genLD import genLD
from model.resnet50 import resnet50
import torch.backends.cudnn as cudnn
from transforms.affine_transforms import *
import time
import warnings

warnings.filterwarnings("ignore")

# ==================== 【GPU 核心设置】====================
# 自动判断是否有 GPU，有就用，没有就用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

# Hyper Parameters
BATCH_SIZE = 32
BATCH_SIZE_TEST = 20
LR = 0.001  # learning rate
NUM_WORKERS = 0  # Windows 先保持 0，不报错
NUM_CLASSES = 4
LOG_FILE_NAME = './logs/log_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log'
lr_steps = [30, 60, 90, 120]

np.random.seed(42)

# Use relative path instead of hard-coded absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'ACNE04', 'Classification', 'JPEGImages')

os.makedirs('./logs', exist_ok=True)
log = Logger()
log.open(LOG_FILE_NAME, mode="a")


def criterion(lesions_num):
    if lesions_num <= 5:
        return 0
    elif lesions_num <= 20:
        return 1
    elif lesions_num <= 50:
        return 2
    else:
        return 3


def trainval_test(cross_val_index, sigma, lam):
    TRAIN_FILE = os.path.join(BASE_DIR, 'ACNE04', 'Classification', 'NNEW_trainval_' + cross_val_index + '.txt')
    TEST_FILE = os.path.join(BASE_DIR, 'ACNE04', 'Classification', 'NNEW_test_' + cross_val_index + '.txt')

    normalize = transforms.Normalize(mean=[0.45815152, 0.361242, 0.29348266],
                                     std=[0.2814769, 0.226306, 0.20132513])

    dset_train = DatasetProcessing(
        DATA_PATH, TRAIN_FILE, transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            RandomRotate(rotation_range=20),  # 我帮你修正了增强顺序！
            transforms.ToTensor(),
            normalize,
        ]))

    dset_test = DatasetProcessing(
        DATA_PATH, TEST_FILE, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = DataLoader(dset_train,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=True)

    test_loader = DataLoader(dset_test,
                             batch_size=BATCH_SIZE_TEST,
                             shuffle=False,
                             num_workers=NUM_WORKERS,
                             pin_memory=True)

    # ==================== 【模型放到 GPU】====================
    cnn = resnet50().to(device)
    cudnn.benchmark = True

    params = []
    new_param_names = ['fc', 'counting']
    for key, value in dict(cnn.named_parameters()).items():
        if value.requires_grad:
            if any(i in key for i in new_param_names):
                params += [{'params': [value], 'lr': LR * 1.0, 'weight_decay': 5e-4}]
            else:
                params += [{'params': [value], 'lr': LR * 1.0, 'weight_decay': 5e-4}]

    optimizer = torch.optim.SGD(params, momentum=0.9)

    # Use PyTorch's built-in lr scheduler instead of manual adjustment
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=0.5)

    loss_func = nn.CrossEntropyLoss()
    kl_loss_1 = nn.KLDivLoss()
    kl_loss_2 = nn.KLDivLoss()
    kl_loss_3 = nn.KLDivLoss()

    start = timer()
    test_acc_his = 0.7
    test_mae_his = 8
    test_mse_his = 18

    for epoch in range(30):
        # Learning rate scheduling is handled by the scheduler
        scheduler.step()

        losses_cls = AverageMeter()
        losses_cou = AverageMeter()
        losses_cou2cls = AverageMeter()
        losses = AverageMeter()

        cnn.train()
        for step, (b_x, b_y, b_l) in enumerate(train_loader):
            # ==================== 【数据送到 GPU】====================
            b_x = b_x.to(device)
            b_l = b_l.numpy()

            b_l = b_l - 1
            ld = genLD(b_l, sigma, 'klloss', 65)
            ld_4 = np.vstack((np.sum(ld[:, :5], 1), np.sum(ld[:, 5:20], 1), np.sum(ld[:, 20:50], 1),
                              np.sum(ld[:, 50:], 1))).transpose()

            # ==================== 【标签送到 GPU】====================
            ld = torch.from_numpy(ld).float().to(device)
            ld_4 = torch.from_numpy(ld_4).float().to(device)

            cnn.train()
            cls, cou, cou2cls = cnn(b_x, None)
            loss_cls = kl_loss_1(torch.log(cls), ld_4) * 4.0
            loss_cou = kl_loss_2(torch.log(cou), ld) * 65.0
            loss_cls_cou = kl_loss_3(torch.log(cou2cls), ld_4) * 4.0
            loss = (loss_cls + loss_cls_cou) * 0.5 * lam + loss_cou * (1.0 - lam)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_cls.update(loss_cls.item(), b_x.size(0))
            losses_cou.update(loss_cou.item(), b_x.size(0))
            losses_cou2cls.update(loss_cls_cou.item(), b_x.size(0))
            losses.update(loss.item(), b_x.size(0))

        message = '%s %6.0f | %0.3f | %0.3f | %0.3f | %0.3f | %s\n' % (
            "train", epoch,
            losses_cls.avg,
            losses_cou.avg,
            losses_cou2cls.avg,
            losses.avg,
            time_to_str((timer() - start), 'min'))
        log.write(message)

        if epoch >= 9:
            with torch.no_grad():
                test_loss = 0
                test_corrects = 0
                y_true = np.array([])
                y_pred = np.array([])
                y_pred_m = np.array([])
                l_true = np.array([])
                l_pred = np.array([])
                cnn.eval()

                for step, (test_x, test_y, test_l) in enumerate(test_loader):
                    # ==================== 【测试数据送到 GPU】====================
                    test_x = test_x.to(device)
                    test_y = test_y.long().to(device)

                    y_true = np.hstack((y_true, test_y.data.cpu().numpy()))
                    l_true = np.hstack((l_true, test_l.data.cpu().numpy()))

                    cls, cou, cou2cls = cnn(test_x, None)

                    loss = loss_func(cou2cls, test_y)
                    test_loss += loss.data

                    _, preds_m = torch.max(cls + cou2cls, 1)
                    _, preds = torch.max(cls, 1)
                    y_pred = np.hstack((y_pred, preds.data.cpu().numpy()))
                    y_pred_m = np.hstack((y_pred_m, preds_m.data.cpu().numpy()))

                    _, preds_l = torch.max(cou, 1)
                    preds_l = (preds_l + 1).data.cpu().numpy()
                    l_pred = np.hstack((l_pred, preds_l))

                    batch_corrects = torch.sum((preds == test_y)).data.cpu().numpy()
                    test_corrects += batch_corrects

                test_loss = test_loss.float() / len(test_loader)
                test_acc = test_corrects / len(test_loader.dataset)
                message = '%s %6.1f | %0.3f | %0.3f\n' % (
                    "test ", epoch,
                    test_loss.data,
                    test_acc)

                _, _, pre_se_sp_yi_report = report_precision_se_sp_yi(y_pred, y_true)
                _, _, pre_se_sp_yi_report_m = report_precision_se_sp_yi(y_pred_m, y_true)
                _, MAE, MSE, mae_mse_report = report_mae_mse(l_true, l_pred, y_true)

                log.write(str(pre_se_sp_yi_report) + '\n')
                log.write(str(pre_se_sp_yi_report_m) + '\n')
                log.write(str(mae_mse_report) + '\n')


# ==================== 【Windows 必须加！防止多进程报错】====================
if __name__ == '__main__':
    cross_val_lists = ['0', '1', '2', '3', '4']
    for cross_val_index in cross_val_lists:
        log.write('\n\ncross_val_index: ' + cross_val_index + '\n\n')
        trainval_test(cross_val_index, sigma=30 * 0.1, lam=6 * 0.1)