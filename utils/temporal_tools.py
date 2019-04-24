import numpy as np
import pandas as pd
from scripts.interpolation import linear_interpolation


def add_time_interval(df, data_col='z_space'):
    """
    This function is called to prepare the dataset for missing values (deal with NaN rows).
    :param df: Multi-index dataframes containing a column of input data (data_col).
    :param data_col: Name of the column containing the input data
    :return: Modify in place the dataframe to add a mask columns ``m``, a time interval columns ``l`` and for each row
    that contains an NaN value in data_col, the NaN is replaced by the previous valid cell.
    """
    df['m'] = ~pd.isna(df['z_space'])
    sequences = np.unique(df.index.get_level_values(0))
    l_full = []
    z_space_full = []
    for seq in sequences:
        l = []
        m_seq = np.asarray(df.loc[seq]['m'])
        z_space_seq = list(df.loc[seq][data_col])

        for i in np.arange(len(m_seq)):
            if m_seq[i]:
                prev_z = z_space_seq[i]
            else:
                z_space_seq[i] = prev_z

            if i == 0:
                l.append(0)
                continue
            if m_seq[i - 1]:
                l.append(1)
            else:
                l.append(1 + l[i - 1])
        l_full += l
        z_space_full += z_space_seq

    df['l'] = l_full
    new_values = np.vstack(z_space_full)
    df[data_col] = new_values.tolist()


def get_sequence_max_length(df):
    sequences = np.unique(df.index.get_level_values(0))
    max_length = 0
    for seq in sequences:
        seq_length = len(df.loc[seq])
        if seq_length>max_length:
            max_length = seq_length
    return max_length


def interpolate_nan_values(df, data_col='z_space'):
    """
    Interpolate NaN cells with previous and next neighbors
    :param df:
    :param data_col:
    :return:
    """
    pd.options.mode.chained_assignment = None
    df['m'] = ~pd.isna(df['z_space'])
    sequences = np.unique(df.index.get_level_values(0))
    new_data_space_full = []
    for seq in sequences:
        m_seq = np.asarray(df.loc[seq]['m'])
        contains_NaN = np.any(~m_seq)
        z_space_seq = list(df.loc[seq][data_col])
        if contains_NaN:
            for i in np.arange(len(m_seq)):
                if isinstance(z_space_seq[i], float):
                    prev = z_space_seq[i - 1]
                    found_next = False
                    j = 0
                    while not found_next:
                        j += 1
                        found_next = not isinstance(z_space_seq[i + j], float)

                    next_ = z_space_seq[i + j]
                    interpolated = linear_interpolation(prev, next_, nb_frames=j)
                    for h in range(j):
                        z_space_seq[i + h] = interpolated[h]
        new_data_space_full += z_space_seq

    df[data_col] = new_data_space_full
    pd.options.mode.chained_assignment = 'warn'
    return df


