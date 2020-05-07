from torch.utils.data import Dataset, DataLoader


class UnSupervisedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class SupervisedDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

        # data should be consistant with label
        if len(self.data) == len(self.label):
            raise ValueError(f'Inconsistant data and label! '
                             f'The length of data is {len(self.data)}, while '
                             f'the length of lable is {len(self.label)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


def get_data_loaders(data, label, n_batch):
    X_train, X_valid, X_test = \
        data[:90000], data[90000:100821], data[:100821]
    Y_train, Y_valid, Y_test = \
        label[:90000], label[90000:100821], label[:100821]

    def create(x, y, batch_size, shuffle=True):
        return DataLoader(SupervisedDataset(x, y),
                          batch_size=batch_size, shuffle=shuffle)

    train_ds = create(X_train, Y_train, n_batch)
    valid_ds = create(X_valid, Y_valid, n_batch)
    test_ds = create(X_test, Y_test, n_batch)

    return train_ds, valid_ds, test_ds
