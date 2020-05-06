import random

import numpy as np
from torch.utils.data import Dataset, DataLoader


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


def get_data_loaders(data, label, n_batch):
    X_train, X_valid, X_test = \
        data[:90000], data[90000:100821], data[:100821]
    Y_train, Y_valid, Y_test = \
        label[:90000], label[90000:100821], label[:100821]

    def create(x, y, batch_size, shuffle=True):
        return DataLoader(NMTInspiredDataset(x, y),
                          batch_size=batch_size, shuffle=shuffle)

    train_ds = create(X_train, Y_train, n_batch)
    valid_ds = create(X_valid, Y_valid, n_batch)
    test_ds = create(X_test, Y_test, n_batch)

    return train_ds, valid_ds, test_ds
