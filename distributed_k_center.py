"""
    Sample and prune distributed k center.
"""
import numpy as np
from greedy_k_center import greedy_k_center, distance_to_center
from utils import get_possible_opt_doubles
import ray


def construct_tester(X, k, eps=1/2):
    """
    Construct a tester in sample and prune.
    :param X: n x d, input data
    :param eps: n^{eps}, the number of points to be sampled, default = 1/2
    :param k: number of centers
    :return: A tester set of points
    """
    sample_size = int(X.shape[0] ** eps)
    sampled_idx = np.random.choice(np.arange(X.shape[0]), size=sample_size, replace=False)
    S = X[sampled_idx]
    T, index, distance = greedy_k_center(S, k)
    return T


@ray.remote
def prune(X_local, T, all_opt_doubles):
    """
    Prune data points on a local machine
    :param X_local: local samples, n x d
    :param T: tester
    :return: A list of set of remaining data points, and a list of opt doubles
    """
    all_dist = np.array([distance_to_center(x, T)[0] for x in X_local])
    all_R = []
    for opt_double in all_opt_doubles:
        R_idx = np.where(all_dist > opt_double)[0]
        R = X_local[R_idx]
        all_R.append(R)
    return all_R

