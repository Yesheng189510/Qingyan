"""Student classification models with ImageNet pretrained weights."""

import torch.nn as nn
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights


def _freeze_bn(backbone):
    def set_bn_fix(m):
        if m.__class__.__name__.find('BatchNorm') != -1:
            for p in m.parameters():
                p.requires_grad = False
    backbone.apply(set_bn_fix)


def _make_student(backbone, num_classes):
    _freeze_bn(backbone)
    return backbone


class StudentResNet18(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        _make_student(self.backbone, num_classes)

    def forward(self, x):
        return self.backbone(x)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            def set_bn_eval(m):
                if m.__class__.__name__.find('BatchNorm') != -1:
                    m.eval()
            self.backbone.apply(set_bn_eval)


class StudentResNet34(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        _make_student(self.backbone, num_classes)

    def forward(self, x):
        return self.backbone(x)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            def set_bn_eval(m):
                if m.__class__.__name__.find('BatchNorm') != -1:
                    m.eval()
            self.backbone.apply(set_bn_eval)


def get_student(arch: str, num_classes: int = 4):
    if arch == 'resnet18':
        return StudentResNet18(num_classes)
    elif arch == 'resnet34':
        return StudentResNet34(num_classes)
    raise ValueError(f'Unknown student architecture: {arch}')
