import random

import numpy as np
from torch.utils.data import Dataset


class CBowDataset(Dataset):

    def __init__(self, data_end, ws=None):
        self.data_end = data_end
        if ws is not None:
            def f(w):  # w is of (fun_id, center_word, ctx_words)
                return random.random() < ws[w[1]]
            self.data_end.data = list(filter(f, data_end.data))

    def __len__(self):
        return len(self.data_end.data)

    def __getitem__(self, idx):
        fun_id, c_word, ctx_words = self.data_end.data[idx]
        return fun_id, c_word, np.array(ctx_words)


class NMTInspiredDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        # data should be consistant with label
        assert len(self.data) == len(self.label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # lx, rx = self.data[idx]
        # y = self.label[idx]
        # return (lx, rx), y
        return self.data[idx], self.label[idx]

    def get_x(self):
        return self.data

    def get_y(self):
        return self.label
