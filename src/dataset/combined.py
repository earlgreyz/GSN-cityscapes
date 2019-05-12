from torch.utils.data import Dataset


class CombinedDataset(Dataset):
    def __init__(self, *args):
        self.ds = args
        self.n = sum([len(d) for d in self.ds])

    def __getitem__(self, item):
        for d in self.ds:
            if item < len(d):
                return d[item]
            item -= len(d)
        raise KeyError

    def __len__(self):
        return self.n
