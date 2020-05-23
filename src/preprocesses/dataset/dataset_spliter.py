import logging
from functools import lru_cache

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np


_n_batch = 64


class DatasetSpliter:

    def __init__(self, dataset, test_train_split=0.2, val_train_split=0.1,
                 shuffle=False):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))

        if shuffle:
            np.random.shuffle(self.indices)

        self.test_indices, self.val_train_indices = _split(
            self.indices, test_train_split
        )
        self.val_indices, self.train_indices = _split(
            self.val_train_indices, val_train_split / (1 - test_train_split)
        )

        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.test_sampler = SubsetRandomSampler(self.test_indices)
        self.val_sampler = SubsetRandomSampler(self.val_indices)

    def get_train_split_point(self):
        return len(self.train_indices) + len(self.val_indices)

    def get_validation_split_point(self):
        return len(self.train_indices)

    @lru_cache(maxsize=4)
    def split(self, batch_size=_n_batch, num_workers=1):
        logging.debug('Initializing train-validation-test dataloaders')
        train_loader = self.get_train_loader(
            batch_size=batch_size, num_workers=num_workers)
        val_loader = self.get_validation_loader(
            batch_size=batch_size, num_workers=num_workers)
        test_loader = self.get_test_loader(
            batch_size=batch_size, num_workers=num_workers)
        return train_loader, val_loader, test_loader

    @lru_cache(maxsize=4)
    def get_train_loader(self, batch_size=_n_batch, num_workers=1):
        logging.debug('Initializing train dataloader')
        return torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, sampler=self.train_sampler,
            shuffle=False, num_workers=num_workers)

    @lru_cache(maxsize=4)
    def get_validation_loader(self, batch_size=_n_batch, num_workers=1):
        logging.debug('Initializing validation dataloader')
        return torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, sampler=self.val_sampler,
            shuffle=False, num_workers=num_workers)

    @lru_cache(maxsize=4)
    def get_test_loader(self, batch_size=_n_batch, num_workers=1):
        logging.debug('Initializing test dataloader')
        return torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, sampler=self.test_sampler,
            shuffle=False, num_workers=num_workers)


def _split(indices, split_ratio):
    size = len(indices)
    split_point = int(np.floor(split_ratio * size))
    return indices[:split_point], indices[split_point:]
