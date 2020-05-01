import random

import numpy as np
from torch.utils.data import Dataset


class CBowDataset(Dataset):

    def __init__(self, data_end, ws=None):
        self.data_end = data_end
        if ws is not None:
            def f(w):  # w is of (fun_id, center_word, ctx_words)
                return random.random() > ws[w[1]]
            self.data_end.data = list(filter(f, data_end.data))

    def __len__(self):
        return len(self.data_end.data)

    def __getitem__(self, idx):
        fun_id, c_word, ctx_words = self.data_end.data[idx]
        return fun_id, c_word, np.array(ctx_words)
