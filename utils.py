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


def get_possible_opt_doubles(min_dist, max_dist, gamma):
    """
    Generate a list of possible 2 opts.
    :param min_dist: minimum dist between data points
    :param max_dist: maximum dist between data points
    :param gamma: approx ratio
    :return: 2 * (1 + gamma)^{i} | min_dist <= (1 + gamma)^{i} <= max_dist
    """
    base = 1 + gamma
    min_power = int(np.floor(np.log(min_dist, 1 + gamma)))
    max_power = int(np.ceil(np.log(max_dist, 1 + gamma)))
    powers = np.arange(min_power, max_power + 1, 1)
    opt_vals = np.array([(1 + gamma) ** i for i in powers])
    return 2 * opt_vals
