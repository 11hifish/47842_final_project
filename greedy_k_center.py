import numpy as np
from scipy.spatial import distance_matrix


def distance_to_center(x, C):
    """
    :param x: dim d, single data point
    :param C: c x d, a set of picked centers
    :return: distance to the closest center, and the corresponding center
    """
    x = np.array([x])
    dist_to_centers = distance_matrix(x, C)  # c x 1
    min_dist = np.min(dist_to_centers.ravel())
    min_c = C[np.argmin(dist_to_centers.ravel())]
    return min_dist, min_c


def greedy_k_center(X, k):
    """
    :param X: n x d, input data
    :param k: number of centers
    :return: a set of picked center idx, a set of picked centers
    """
    c1_idx = np.random.choice(X.shape[0])
    C = np.array([X[c1_idx]])
    C_idx = [c1_idx]
    num_centers = 1
    while num_centers < k:
        all_dist = np.array([distance_to_center(x, C)[0] for x in X])
        next_center_idx = np.argmax(all_dist.ravel())
        next_center = X[next_center_idx]
        C = np.vstack((C, next_center))
        C_idx.append(next_center_idx)
        num_centers += 1
    dist_to_picked_center = np.array([distance_to_center(x, C)[0] for x in X])
    return C, np.array(C_idx), dist_to_picked_center


if __name__ == '__main__':
    X = np.array([[1,2,3], [0,0,1], [4,5,6], [10,0,3]])
    k = 2
    C, idx, all_dist = greedy_k_center(X, k)
    print(C)
    print(idx)
    print(all_dist)
