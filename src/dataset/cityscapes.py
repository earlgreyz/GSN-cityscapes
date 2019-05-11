import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def image_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return np.array(img.convert('RGB'))


class CityscapesDataset(Dataset):
    def __init__(self, root, classes):
        super().__init__()
        self.samples = glob.glob(os.path.join(root, '*.png'))
        self.classes = classes
        self.transform = transforms.ToTensor()

    def __getitem__(self, item):
        img = image_loader(self.samples[item])
        h, w, d = img.shape
        n = w // 2

        label = np.zeros((h, n))
        for y in range(h):
            for x in range(n):
                index = np.where((self.classes == img[y, n + x, :]).all(axis=1))
                assert len(index) == 1
                label[y, x] = index[0]

        sample = img[:, :n, :]
        target = torch.tensor(label, dtype=torch.long)
        return self.transform(sample), target

    def __len__(self):
        return len(self.samples)
