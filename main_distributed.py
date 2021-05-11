import ray
ray.init(address='auto', _redis_password='5241590000000000')
from distributed_k_center import construct_tester, prune
import numpy as np
from utils import *
from greedy_k_center import greedy_k_center
import pickle


num_machines = 5
k = 3
eps = 1/2

# get data
X = np.random.random(size=(100, 3))
# partition data
P = even_split_data(X, num_machines)

# greedy baseline and get 2 OPT
C_baseline, C_baseline_idx, dist_to_C = greedy_k_center(X, k)
opt_double = np.max(dist_to_C)

# construct tester
T = construct_tester(X, k, eps)

# prune
futures = [prune.remote(X_local, T, opt_double) for X_local in P]
R_list = []
while len(futures) > 0:
    finished, rest = ray.wait(futures)
    R = ray.get(finished[0])
    R_list.append(R)
    futures = rest

all_R = np.vstack(R_list)
print('length of all R: {}'.format(all_R.shape[0]))

C_distri, C_distri_idx, dist_to_C_distri = greedy_k_center(np.vstack(all_R, T), k)
opt_dist = np.max(dist_to_C_distri)
# print results
print('baseline dist: {}'.format(opt_double))
print('opt dist: {}'.format(opt_dist))

# save results
with open('result_k_{}_m_{}.pkl'.format(k, num_machines), 'wb') as f:
    pickle.dump((C_baseline, C_baseline_idx, dist_to_C,
                 C_distri, C_distri_idx, dist_to_C_distri), f)
