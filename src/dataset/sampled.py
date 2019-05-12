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
