import nndescent
from urllib.request import urlretrieve
from sklearn.neighbors import KDTree

import time
import numpy as np
import random

import os
import h5py

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(rc={"figure.figsize": (10, 6)})

print(nndescent.version())

class Timer:
    def __init__(self):
        self.time0 = time.time()
    def start(self):
        self.time0 = time.time()
    def stop(self, text=None):
        delta_time = 1000*(time.time() - self.time0)
        comment = "" if text is None else "(" + text + ")"
        print("Time passed:", delta_time, "ms", comment)
        self.time0 = time.time()

timer = Timer()

# Data
pnts = np.array(
    [
        [1, 10],
        [17, 5],
        [59, 5],
        [60, 5],
        [9, 13],
        [60, 13],
        [17, 19],
        [54, 19],
        [52, 20],
        [54, 22],
        [9, 25],
        [70, 28],
        [31, 31],
        [9, 32],
        [52, 32],
        [67, 33],
    ],
    dtype=np.float32,
)

pnts2 = np.array(
    [[random.randint(0, 100) for dim in range(10)] for size in range(300)],
    dtype=np.float32,
)

pnts3 = np.array(np.random.randint(2, size=(1300, 500)), dtype=np.float32)
# pnts4 = np.array(np.random.randint(2, size=(30_000, 5_000)))
pnts5 = np.array(np.random.randint(2, size=(208, 3339)), dtype=np.float32)
# t=time.time(); x=nndescent.test(pnts4, 30); print(time.time()-t)
data_NR = np.loadtxt(
    os.path.expanduser(
            "~/Dropbox/WorkHome/programming/nnd/data/NR208x3339.csv"
    ),
    delimiter=",",
    dtype=np.float32,
    # dtype=str,
)


def get_ann_benchmark_data(dataset_name):
    data_path = os.path.expanduser(f"~/Downloads/{dataset_name}.hdf5")
    if not os.path.exists(data_path):
        print(f"Dataset {dataset_name} is not cached; downloading now ...")
        urlretrieve(
            f"http://ann-benchmarks.com/{dataset_name}.hdf5",
            data_path,
        )
    hdf5_file = h5py.File(data_path, "r")
    return (
        np.array(hdf5_file["train"]),
        np.array(hdf5_file["test"]),
        hdf5_file.attrs["distance"],
    )


def accuracy(approx_neighbors, true_neighbors):
    result = np.zeros(approx_neighbors.shape[0])
    for i in range(approx_neighbors.shape[0]):
        n_correct = np.intersect1d(
            approx_neighbors[i], true_neighbors[i]
        ).shape[0]
        result[i] = n_correct / true_neighbors.shape[1]
    return result


fmnist_train, fmnist_test, _ = get_ann_benchmark_data(
    "fashion-mnist-784-euclidean"
)


k = 4
data = data_NR
data = pnts
# data = np.double(fmnist_train)
timer.start(); bf = nndescent.bfnn(data, k); timer.stop()
timer.start(); nnd = nndescent.nnd(data, k); timer.stop()
# t = nndescent.test(pnts, k)
tree_index = KDTree(data)
kdt = tree_index.query(data, k=k)[1]

accuracy_stats = accuracy(nnd, kdt)
print(f"Average accuracy of {np.mean(accuracy_stats)}")
sns.histplot(accuracy_stats, kde=False)
plt.title("Distribution of accuracy per query point")
plt.xlabel("Accuracy")
plt.show()


# import pynndescent
# index = pynndescent.NNDescent(data, verbose=True)
# timer.start(); index = pynndescent.NNDescent(data, leaf_size=k, verbose=True); timer.stop()
# timer.start(); index = pynndescent.NNDescent(data, leaf_size=k, n_trees=1, n_iters=0, verbose=True); timer.stop()
# nnd = index.neighbor_graph[0]

# index.prepare()
# neighbors = index.query(pnts, k=k)


# import numpy as np
# import random
# import pynndescent
# data = np.double(
#     [[random.randint(0, 100) for dim in range(10)] for size in range(300)]
# )
# self = pynndescent.NNDescent(data, verbose=True)
# from pynndescent.pynndescent_ import *
# from pynndescent.utils import *
# from pynndescent.rp_trees import *
# metric="euclidean"
# metric_kwds=None
# n_neighbors=30
# n_trees=None
# leaf_size=None
# pruning_degree_multiplier=1.5
# diversify_prob=1.0
# n_search_trees=1
# tree_init=True
# init_graph=None
# init_dist=None
# random_state=None
# low_memory=True
# max_candidates=None
# n_iters=None
# delta=0.001
# n_jobs=None
# compressed=False
# parallel_batch_queries=False
# verbose=True




# timer.start(); self._rp_forest = make_forest( data, n_neighbors, n_trees, leaf_size, self.rng_state, current_random_state, self.n_jobs, self._angular_trees,); timer.stop("make forest")

# timer.start(); self._rp_forest = make_forest( data, n_neighbors, n_trees, leaf_size, self.rng_state, current_random_state, self.n_jobs, self._angular_trees,); timer.stop()


# data = check_array(data, dtype=np.float32, accept_sparse="csr", order="C")
# indices = np.arange(data.shape[0]).astype(np.int32)
# current_random_state = check_random_state(self.random_state)
# rng_state = current_random_state.randint(INT32_MIN, INT32_MAX, 3).astype(
    # np.int64
# )

# timer.start(); euclidean_random_projection_split(data, indices, rng_state); timer.stop()


# timer.start(); make_euclidean_tree( data, indices, hyperplanes, offsets, children, point_indices, rng_state, leaf_size,); timer.stop()


import numba
@numba.njit
def dot():
    dim = 1e7
    hyperplane_offset = 0.0
    v0 = np.arange(0, dim, 1,  dtype=np.float32)
    v1 = np.arange(1, dim+1, 1, dtype=np.float32)
    cumsum=0
    for d in range(dim):
        cumsum += v0[d] * v1[d]
    return cumsum

timer.start(); dot(); timer.stop("dot")

@numba.njit
def dota(data):
    size = data.shape[0]
    dim = data.shape[1]
    cumsum=0
    for i in range(size):
        for d in range(dim):
            cumsum += data[0,d] * data[i,d]
    return cumsum

timer.start(); dota(data); timer.stop("dot")
