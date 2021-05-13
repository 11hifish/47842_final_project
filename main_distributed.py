import ray
ray.init(address='auto', _redis_password='5241590000000000')
from distributed_k_center import construct_tester, prune
import numpy as np
from utils import *
from greedy_k_center import greedy_k_center, distance_to_center
from scipy.spatial import distance_matrix
import pickle
import os


num_machines = 5
k = 3
eps = 1/2
gamma = 1/2
data_name = 'SCADI'

# get data
# X = np.random.random(size=(100, 3))
with open(os.path.join('data', data_name + '.pkl'), 'rb') as f:
    X, y = pickle.load(f)
# partition data
P = even_split_data(X, num_machines)

# get min dist, max dist
D = distance_matrix(X, X)
max_dist = np.max(D)
D = D + np.eye(X.shape[0]) * 100000
min_dist = np.min(D)
print(max_dist, min_dist)
# get all possible 2OPTs
all_opt_doubles = get_possible_opt_doubles(min_dist, max_dist, gamma)
print('all possible 2OPTs: ', all_opt_doubles)

# greedy baseline and get 2 OPT
C_baseline, C_baseline_idx, dist_to_C = greedy_k_center(X, k)
opt_double = np.max(dist_to_C)

# construct tester
T = construct_tester(X, k, eps)

# prune
futures = [prune.remote(X_local, T, all_opt_doubles) for X_local in P]
R_lists = []
while len(futures) > 0:
    finished, rest = ray.wait(futures)
    R_l = ray.get(finished[0])
    R_lists.append(R_l)
    futures = rest

print('length of all R: {}'.format(len(R_lists)))  # should be # machines

min_opt = None
C_distri_opt = None
C_distri_idx_opt = None
dist_to_C_distri_opt = None
for opt_double_idx in range(len(all_opt_doubles)):
    R_opt_double = []
    for R_m in R_lists:
        R_opt_double.append(R_m[opt_double_idx])
    R_opt_double = np.vstack(R_opt_double)
    C_distri, C_distri_idx, dist_to_C_distri = greedy_k_center(np.vstack((R_opt_double, T)), k)
    opt_all_dist = np.array([distance_to_center(x, C_distri)[0] for x in X])
    opt_dist = np.max(opt_all_dist)
    print('opt_dist: ', opt_dist)
    if min_opt is None or opt_dist < min_opt:
        min_opt = opt_dist
        C_distri_opt = C_distri
        C_distri_idx_opt = C_distri_idx
        dist_to_C_distri_opt = dist_to_C_distri

# print results
print('baseline dist: {}'.format(opt_double))
print('opt dist: {}'.format(min_opt))

# save results
with open('result_{}_k_{}_m_{}.pkl'.format(data_name, k, num_machines), 'wb') as f:
    pickle.dump((C_baseline, C_baseline_idx, dist_to_C,
                 C_distri_opt, C_distri_idx_opt, dist_to_C_distri_opt), f)
