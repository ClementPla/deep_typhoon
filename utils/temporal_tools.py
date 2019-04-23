import numpy as np
import pandas as pd


def add_time_interval(df, data_col='z_space'):
    """
    This function is called to prepare the dataset for missing values (deal with NaN rows).
    :param df: Multi-index dataframes containing a column of input data (data_col).
    :param data_col: Name of the column containing the input data
    :return: Modify in place the dataframe to add a mask columns ``m``, a time interval columns ``l`` and for each row
    that contains an NaN value in data_col, a new cell containing the previous valid cell.
    """
    df['m'] = ~pd.isna(df['z_space'])
    sequences = np.unique(df.index.get_level_values(0))
    l_full = []
    previous_z_full = []
    for seq in sequences:
        l = []
        previous_z = []
        m_seq = np.asarray(df.loc[seq]['m'])
        contains_NaN = np.any(m_seq)
        if contains_NaN:
            z_space_seq = np.asarray(df.loc[seq][data_col])
        else:
            previous_z = [-1] * len(m_seq)

        for i in np.arange(len(m_seq)):
            if contains_NaN:
                if m_seq[i]:
                    prev_z = z_space_seq[i]
                    previous_z.append(-1)
                else:
                    previous_z.append(prev_z)

            if i == 0:
                l.append(0)
                continue
            if m_seq[i - 1]:
                l.append(1)
            else:
                l.append(1 + l[i - 1])
        l_full += l
        previous_z_full += previous_z

    df['l'] = l_full
    df['previous_' + data_col] = previous_z_full


def get_sequence_max_length(df):
    sequences = np.unique(df.index.get_level_values(0))
    max_length = 0
    for seq in sequences:
        seq_length = len(df.loc[seq])
        if seq_length>max_length:
            max_length = seq_length
    return max_length
