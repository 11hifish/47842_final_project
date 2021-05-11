import numpy as np


def even_split_data(X, s=2):
    """
        Row-wise split data matrix across s machines evenly.
        :param X: n x d data matrix
        :param s: split across s machines
        :return: A list of splitted data matrices.
    """
    num_rows = X.shape[0]
    batch_size = num_rows // s
    P = []
    start_idx = 0
    while start_idx < num_rows:
        end_idx = start_idx + batch_size
        if num_rows - end_idx < batch_size:
            P.append(X[start_idx:, :])
            start_idx = num_rows
        else:
            P.append(X[start_idx:end_idx, :])
            start_idx = end_idx
    return P
