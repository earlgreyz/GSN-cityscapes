import numpy as np
from torch.utils.data import Dataset

from dataset.sampled import SampledDataset


def split_dataset(dataset: Dataset, sample_rate: float, seed: None) -> (SampledDataset, SampledDataset):
    size = len(dataset)
    items = list(range(size))
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(items)
    n = int(size * sample_rate)
    return SampledDataset(dataset, items[n:]), SampledDataset(dataset, items[:n])
