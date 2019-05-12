import numpy as np
from torch.utils.data import Dataset

from dataset.sampled import SampledDataset


def split_dataset(dataset: Dataset, sample_rate: float) -> (SampledDataset, SampledDataset):
    size = len(dataset)
    items = list(range(size))
    np.random.shuffle(items)
    n = int(size * sample_rate)
    return SampledDataset(dataset, items[n:]), SampledDataset(dataset, items[:n])
