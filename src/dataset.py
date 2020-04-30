import os, pickle, random
import numpy as np

from torch.utils.data import Dataset


class CBowDataset(Dataset):

    def __init__(self, data_end, ws=None):
        data = data_end
        if ws is not None:
            def f(_, w, __):
                return random.random() > ws[w]
            self.data = filter(f, data)
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iword, owords = self.data[idx]
        return iword, np.array(owords)
