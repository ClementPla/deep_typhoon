import numpy as np
from torch.utils.data import Dataset


def avance_time(df, column, delay):
    if not isinstance(column, list):
        column = [column]
    sequences = np.unique(df.index.get_level_values(0))
    for seq in sequences:
        df_seq = df.loc[seq]
        for col in column:
            df.loc[(seq, col)] = np.roll(df_seq[col], -delay)
        seq_length = len(df_seq)
        indexes = np.arange(0, seq_length)[::-1]
        df.loc[(seq, 'temp_index')] = indexes  # Create temporary indexes to indicate which values to delete
    df = df.set_index('temp_index', append=True)
    df.drop(np.arange(0, delay), level=2, inplace=True)  # Drop those values
    df = df.reset_index(level=2, drop=True)  # Remove temporary indexes
    return df


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
    def __init__(self, df, max_length, columns='z_space', column_mask=False):
        if not isinstance(columns, list):
            columns = [columns]
        self.sequences = np.unique(df.index.get_level_values(0))
        self.df = df
        self.max_length = max_length
        self.columns = columns
        self.column_mask = column_mask

    def __len__(self):
        return len(self.sequences)

    def pad_seq(self, array):
        shape = array.shape
        dtype = array.dtype
        pad = self.max_length - shape[0]
        padding = [(0, pad)] + [(0, 0) for _ in shape[1:]]
        padded_array = np.pad(array, padding, mode='constant', constant_values=0)

        if dtype==int:
            return padded_array.astype(dtype)
        else:
            return padded_array.astype(np.float32)

    def get_element(self, idx):
        seq = self.sequences[idx]
        seq_size = len(self.df.loc[seq][self.columns[0]])
        results = [self.pad_seq(np.vstack(self.df.loc[seq][col])) for col in self.columns]

        if not self.column_mask:
            return tuple(results) + (seq_size,)
        else:
            mask = np.zeros((self.max_length), dtype=np.float32)
            mask[:seq_size] = 1.
            return tuple(results) + (mask,) + (seq_size,)

    def __getitem__(self, idx):
        return self.get_element(idx)
