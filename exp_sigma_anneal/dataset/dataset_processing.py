import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np


class DatasetProcessing(Dataset):
    def __init__(self, data_path, img_filename, transform=None):
        self.img_path  = data_path
        self.transform = transform

        fp = open(img_filename, 'r')
        self.img_filename = []
        self.labels   = []
        self.lesions  = []
        for line in fp.readlines():
            filename, label, lesion = line.split()
            self.img_filename.append(filename)
            self.labels.append(int(label))
            self.lesions.append(int(lesion))
        fp.close()

        self.img_filename = np.array(self.img_filename)
        self.labels       = np.array(self.labels)
        self.lesions      = np.array(self.lesions)

        if 'NNEW_trainval' in img_filename:
            ratio = 1.0
            import random
            random.seed(42)
            indexes = []
            for i in range(4):
                index = random.sample(
                    list(np.where(self.labels == i)[0]),
                    int(len(np.where(self.labels == i)[0]) * ratio)
                )
                indexes.extend(index)
            self.img_filename = self.img_filename[indexes]
            self.labels       = self.labels[indexes]
            self.lesions      = self.lesions[indexes]

        # ── PIL缓存，初始为None，调用 cache_images() 后填充
        # 存的是原始PIL图（未经任何transform），这样RandomCrop/Flip每次仍随机，数据增强不受影响
        self._cache = [None] * len(self.img_filename)
        self._cached = False

    def cache_images(self, resize=256):
        if self._cached:
            return
        for i, fname in enumerate(self.img_filename):
            img = Image.open(os.path.join(self.img_path, fname)).convert('RGB')
            img = img.resize((resize, resize), Image.BILINEAR)  # ← 关键：先压缩再存
            self._cache[i] = img
        self._cached = True

    def __getitem__(self, index):
        if self._cached:
            img = self._cache[index]          # 直接从内存取，无磁盘IO
        else:
            img = Image.open(
                os.path.join(self.img_path, self.img_filename[index])
            ).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)         # transform每次仍随机执行

        label  = torch.from_numpy(np.array(self.labels[index]))
        lesion = torch.from_numpy(np.array(self.lesions[index]))
        return img, label, lesion

    def __len__(self):
        return len(self.img_filename)