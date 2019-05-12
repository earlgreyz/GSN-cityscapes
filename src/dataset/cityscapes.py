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
        return img.convert('RGB')


class CityscapesDataset(Dataset):
    def __init__(self, root, classes):
        super().__init__()
        self.samples = glob.glob(os.path.join(root, '*.png'))
        self.classes = classes
        self.transform = transforms.ToTensor()

    def __getitem__(self, item):
        img = image_loader(self.samples[item])
        w, h = img.size
        n = w // 2

        sample = np.array(img)[:, :n, :]
        target = np.array([[self.classes[img.getpixel((n + x, y))] for x in range(n)] for y in range(h)])

        return self.transform(sample.copy()), torch.tensor(target.copy(), dtype=torch.long)

    def __len__(self):
        return len(self.samples)
