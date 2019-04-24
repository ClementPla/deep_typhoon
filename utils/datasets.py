import numpy as np
from torch.utils.data import Dataset


def split_dataframe(df, test_set_year=2011, validation_ratio=0.2, seed=1234):
    """
    Splits the dataframe in training, validation and test set. The test set is defined by a year (all sequences coming
    after this year are in the test set). The validation test is defined as a fraction of the training set (randomized)
    :param df:
    :param test_set_year:
    :param validation_ratio:
    :return: A dictionary containing three keys (train, validation, test) and their respective dataframe
    """

    sequences = df.index.get_level_values(0)
    prev2011 = np.unique([_ for _ in sequences if int(_[:4]) <= test_set_year])
    post2011 = np.unique([_ for _ in sequences if int(_[:4]) > test_set_year])
    np.random.seed(seed)
    indexes = np.arange(len(prev2011))
    np.random.shuffle(indexes)
    test = df.loc[post2011]
    if validation_ratio:
        validation_indexes = indexes[:int(len(prev2011) * validation_ratio)]
        train_indexes = indexes[int(len(prev2011) * validation_ratio):]
        train = df.loc[prev2011[train_indexes]]
        validation = df.loc[prev2011[validation_indexes]]
        return dict(train=train, validation=validation, test=test)
    else:
        train_indexes = indexes[int(len(prev2011) * validation_ratio):]
        train = df.loc[prev2011[train_indexes]]
        return dict(train=train, test=test)




class TyphoonSequencesDataset(Dataset):
    def __init__(self, df, max_length, columns='z_space', mask_columns=False):
        if not isinstance(columns, list):
            columns = [columns]
        self.sequences = np.unique(df.index.get_level_values(0))
        self.df = df
        self.max_length = max_length
        self.columns = columns
        self.mask_columns = mask_columns

    def __len__(self):
        return len(self.sequences)

    def pad_seq(self, array):
        shape = array.shape
        pad = self.max_length - shape[0]
        padding = [(0, pad)] + [(0, 0) for _ in shape[1:]]
        padded_array = np.pad(array, padding, mode='constant', constant_values=0)
        return padded_array.astype(np.float32)

    def get_element(self, idx):
        seq = self.sequences[idx]
        seq_size = len(self.df.loc[seq][self.columns[0]])
        results = [self.pad_seq(np.vstack(self.df.loc[seq][col])) for col in self.columns]
        if len(results)==1:
            results = results[0]
        if not self.mask_columns:
            return tuple(results) + (seq_size,)
        else:
            mask = np.zeros((self.max_length), dtype=np.float32)
            mask[:seq_size] = 1

            return (results, mask)+(seq_size, )

    def __getitem__(self, idx):
        return self.get_element(idx)