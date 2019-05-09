import numpy as np
from torch.utils.data import Dataset


class SampledDataset(Dataset):
    def __init__(self, parent, mapping):
        super().__init__()
        self.parent = parent
        self.mapping = mapping

    def __getitem__(self, item):
        return self.parent[self.mapping[item]]

    def __len__(self):
        return len(self.mapping)


def split_dataset(dataset: Dataset, sample_rate: float) -> (SampledDataset, SampledDataset):
    size = len(dataset)
    items = list(range(size))
    np.random.shuffle(items)
    n = int(size * sample_rate)
    return SampledDataset(dataset, items[:n]), SampledDataset(dataset, items[n:])
