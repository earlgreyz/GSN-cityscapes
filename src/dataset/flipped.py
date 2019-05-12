import torch
from torch.utils.data import Dataset


class FlippedDataset(Dataset):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    def __getitem__(self, item):
        sample, target = self.parent[item]
        sample = torch.flip(sample, dims=(2,))
        target = torch.flip(target, dims=(1,))
        return sample, target

    def __len__(self):
        return len(self.parent)
