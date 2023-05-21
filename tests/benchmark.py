"""
Comparison with pynndescent. Please first fun make_test_data.py.
"""


import nndescent
import pynndescent
from urllib.request import urlretrieve
from sklearn.neighbors import KDTree

import time
import numpy as np
import random

import os
import h5py

import seaborn as sns
import matplotlib.pyplot as plt

DATA_PATH = os.path.expanduser("~/Downloads/nndescent_test_data")
os.makedirs(DATA_PATH, exist_ok=True)

sns.set(rc={"figure.figsize": (10, 6)})


class Timer:
    """Measures the time elapsed in milliseconds."""

    def __init__(self):
        self.time0 = time.time()

    def start(self):
        self.time0 = time.time()

    def stop(self, text=None):
        delta_time = 1000 * (time.time() - self.time0)
        comment = "" if text is None else "(" + text + ")"
        print("Time passed:", delta_time, "ms", comment)
        self.time0 = time.time()
        return delta_time


timer = Timer()

def ann_benchmark_data(dataset_name):
    """Downloads data if necessary and returns as arrays."""
    data_path = os.path.join(DATA_PATH, f"{dataset_name}.hdf5")
    if not os.path.exists(data_path):
        print(f"Dataset {dataset_name} is not cached; downloading now ...")
        urlretrieve(
            f"http://ann-benchmarks.com/{dataset_name}.hdf5",
            data_path,
        )
    hdf5_file = h5py.File(data_path, "r")
    nn_ect = np.array(hdf5_file["neighbors"])
    nn_ect = np.c_[range(nn_ect.shape[0]), nn_ect]
    return (
        np.array(hdf5_file["train"]),
        np.array(hdf5_file["test"]),
        hdf5_file.attrs["distance"],
        nn_ect,
    )


def accuracy(approx_neighbors, true_neighbors):
    result = np.zeros(approx_neighbors.shape[0])
    for i in range(approx_neighbors.shape[0]):
        n_correct = np.intersect1d(
            approx_neighbors[i], true_neighbors[i]
        ).shape[0]
        result[i] = n_correct / true_neighbors.shape[1]
    print(f"Average accuracy of {np.mean(result)}\n")
    return result


def benchmark_pynndescent(**kwargs):
    timer.start()
    index = pynndescent.NNDescent(**kwargs)
    indices = index.neighbor_graph[0]
    dim = kwargs["data"].shape
    time = timer.stop(f"pynndescent dim={dim}")
    return indices, time


def benchmark_nndescent(**kwargs):
    timer.start()
    index = nndescent.NNDescent(**kwargs)
    indices = index.neighbor_graph[0]
    dim = kwargs["data"].shape
    time = timer.stop(f"nndescent dim={dim}")
    return indices, time


def benchmark_bf(**kwargs):
    timer.start()
    index = nndescent.NNDescent(**kwargs, algorithm="bf")
    indices = index.neighbor_graph[0]
    dim = kwargs["data"].shape
    time = timer.stop(f"nndescent brute force dim={dim}")
    return indices, time


def benchmark_kdtree(**kwargs):
    timer.start()
    tree_index = KDTree(kwargs["data"])
    indices = tree_index.query(
        kwargs["data"], k=kwargs["n_neighbors"]
    )[1]
    dim = kwargs["data"].shape
    time = timer.stop(f"KDTree dim={dim}")
    return indices, time



# Detailed example
############################################################

print(f"Benchmarking coil20 ...")
print(f"------------------------\n")

# Small dataset useful for debugging
coil20 = np.loadtxt(
    os.path.join(DATA_PATH, "coil20.csv"), delimiter=",", dtype=np.float32
)

# train, test, dist, nn_train = ann_benchmark_data("mnist-784-euclidean")
# train = np.array(
#     [[random.randint(0, 100) for dim in range(3000)] for size in range(3000)]
# )

# Parameters in used algorithm
kwargs = {
    "data": coil20,
    # "data": train,
    "n_neighbors": 30,
    "verbose": True,
    "metric": "euclidean",
    # "metric": "dot",
}

# Ignore first pynndescent run (slow due to numba compilation)
_ , _ = benchmark_pynndescent(**kwargs)
nn_pynnd, t_pynnd_coil20 = benchmark_pynndescent(**kwargs)
nn_nnd, t_nnd_coil20 = benchmark_nndescent(**kwargs)
nn_bf, _ = benchmark_bf(**kwargs)

# pynndescent accuracy
print("\ncoil20: Accuracy pynndescent vs bf")
accuracy_stats = accuracy(nn_pynnd, nn_bf)

# Accuracy with plots
print("\ncoil20: Accuracy nndescent vs bf")
accuracy_stats = accuracy(nn_nnd, nn_bf)

sns.histplot(accuracy_stats, kde=False)
plt.title("Distribution of accuracy per query point")
plt.xlabel("Accuracy")
plt.show()

# Brute force vs kdtrees (must be almost 1.0)
nn_kdtree, _ = benchmark_kdtree(**kwargs)
print("\ncoil20: Accuracy bf vs kdtree")
accuracy_stats = accuracy(nn_bf, nn_kdtree)

# Accuracy relative to pynndescent
print("\ncoil20: Accuracy nndescent vs pynndescent")
accuracy_stats = accuracy(nn_nnd, nn_pynnd)

# Ad results to summary
summary = [
    [
        "coil20",
        t_pynnd_coil20,
        t_nnd_coil20,
        t_nnd_coil20/t_pynnd_coil20,
        np.mean(accuracy_stats)
    ],
]

# ANN Benchmark tests
############################################################

# ANN Benchmark datasets. Comment out accordingly (depending on
# hardware some data sets are too big and calculation will be
# killed by os).
# https://github.com/erikbern/ann-benchmarks
ANNBEN = {
    # "deep": "deep-image-96-angular", # huge
    "fmnist": "fashion-mnist-784-euclidean",
    # "gist": "gist-960-euclidean", # huge
    # "glove25": "glove-25-angular",
    # "glove50": "glove-50-angular",
    # "glove100": "glove-100-angular",
    # "glove200": "glove-200-angular",
    # "kosark": "kosarak-jaccard", # url not working
    # "mnist": "mnist-784-euclidean",
    # "movielens": "movielens10m-jaccard", # url not working
    # "nytimes": "nytimes-256-angular",
    # "sift": "sift-128-euclidean",
    # "lastfm": "lastfm-64-dot", # url not working
}

distance_translation = {
    "euclidean": "euclidean",
    "jaccard": "jaccard",
    "angular": "dot",
}

for name, url_name in ANNBEN.items():
    print(f"Benchmarking {name} ...")
    print(f"------------------------\n")
    train, test, dist, nn_train = ann_benchmark_data(url_name)
    kwargs = {
        "data": train,
        "n_neighbors": 30,
        "verbose": True,
        "metric": distance_translation[dist],
    }
    # Ignore first pynndescent run (slow due to numba compilation)
    _, _ = benchmark_pynndescent(**kwargs)
    nn_pynnd, t_pynnd = benchmark_pynndescent(**kwargs)
    nn_nnd, t_nnd = benchmark_nndescent(**kwargs)
    # Accuracy relative to pynndescent
    print(f"\n{name}: Accuracy nndescent vs pynndescent")
    accuracy_stats = accuracy(nn_nnd, nn_pynnd)
    summary.append(
        [name, t_pynnd, t_nnd, t_nnd / t_pynnd, np.mean(accuracy_stats)]
    )

# Summary
print(f"# Benchmark test pynndescent vs nndescent")
print(f"Data set  | pynndescent [ms] | nndescent [ms] | ratio | accuracy")
print(f"----------|------------------|----------------|-------|---------")
for nm, t_py, t_cpp, rat, acc in summary:
    print(
        f"{nm:<9} | {t_py: 16.1f} | {t_cpp: 14.1f} | {rat:5.3f} | {acc: 5.3f}"
    )
