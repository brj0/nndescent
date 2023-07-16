"""
Comparison with pynndescent.
"""

from urllib.request import urlretrieve
import os
import time

from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree, NearestNeighbors
import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pynndescent
import nndescent

DATA_PATH = os.path.expanduser("~/Downloads/nndescent_test_data")
os.makedirs(DATA_PATH, exist_ok=True)

sns.set(rc={"figure.figsize": (10, 6)})


class Timer:
    """Measures the time elapsed in milliseconds."""

    def __init__(self):
        self.time0 = time.time()

    def start(self):
        """Resets timer."""
        self.time0 = time.time()

    def stop(self, text=None):
        """Resets timer and return elapsed time."""
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


def exact_nn(train, test, n_neighbors):
    """Calculates exact nearest neighbors (of test in train)."""
    a, b = train.shape
    c, d = test.shape
    fname_nn = f"20newsgroups-{a}x{b}-{c}x{d}_nn.npy"
    fname_dist = f"20newsgroups-{a}x{b}-{c}x{d}_dist.npy"
    fpath_nn = os.path.join(DATA_PATH, fname_nn)
    fpath_dist = os.path.join(DATA_PATH, fname_dist)
    if not (os.path.exists(fpath_nn) and os.path.exists(fpath_dist)):
        print("Exact k-NN for comparison not cached; calculating now...")
        nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        nn_model.fit(train)
        knn = nn_model.kneighbors(test)
        np.save(fpath_nn, knn[1])
        np.save(fpath_dist, knn[0])
        print("Calculation done. Result cached.")
    nn = np.load(fpath_nn)
    dist = np.load(fpath_dist)
    return nn, dist


def accuracy(approx_neighbors, true_neighbors):
    """Returns accuracy of algorithm when compared with exact values."""
    result = np.zeros(approx_neighbors.shape[0])
    for i in range(approx_neighbors.shape[0]):
        n_correct = np.intersect1d(
            approx_neighbors[i], true_neighbors[i]
        ).shape[0]
        result[i] = n_correct / true_neighbors.shape[1]
    print(f"Average accuracy of {np.mean(result)}\n")
    return result


def benchmark_pynndescent(**parms):
    """Runs pynndescent on data and query data. Returns NN-indices
    and elapsed time.
    """
    query_data = parms["query_data"]
    k = parms["k"]
    dim = parms["data"].shape
    del parms["query_data"]
    del parms["k"]
    timer.start()
    index = pynndescent.NNDescent(**parms, random_state=1234)
    indices_train = index.neighbor_graph[0]
    time_train = timer.stop(f"pynndescent train, dim={dim}")
    indices_test = index.query(query_data, k)[0]
    time_test = timer.stop(f"pynndescent test, dim={dim}")
    return indices_train, indices_test, time_train, time_test


def benchmark_nndescent(**parms):
    """Runs nndescent on data and query data. Returns NN-indices
    and elapsed time.
    """
    query_data = parms["query_data"]
    k = parms["k"]
    dim = parms["data"].shape
    del parms["query_data"]
    del parms["k"]
    timer.start()
    index = nndescent.NNDescent(**parms, seed=1234)
    indices_train = index.neighbor_graph[0]
    time_train = timer.stop(f"nndescent train, dim={dim}")
    indices_test = index.query(query_data, k)[0]
    time_test = timer.stop(f"nndescent test, dim={dim}")
    return indices_train, indices_test, time_train, time_test


def benchmark_bf(**parms):
    """Runs nndescent using brute force on data and query data.
    Returns NN-indices and elapsed time.
    """
    query_data = parms["query_data"]
    k = parms["k"]
    dim = parms["data"].shape
    del parms["query_data"]
    del parms["k"]
    timer.start()
    index = nndescent.NNDescent(**parms, algorithm="bf")
    indices_train = index.neighbor_graph[0]
    time_train = timer.stop(f"brute force train, dim={dim}")
    indices_test = index.query(query_data, k)[0]
    time_test = timer.stop(f"brute force test, dim={dim}")
    return indices_train, indices_test, time_train, time_test


def benchmark_kdtree(**parms):
    """Runs kdtree on data and query data. Returns NN-indices (only for test)
    and elapsed time.
    """
    dim = parms["data"].shape
    timer.start()
    index = KDTree(parms["data"])
    time_train = timer.stop(f"KDTree train, dim={dim}")
    indices_test = index.query(parms["query_data"], k=parms["k"])[1]
    time_test = timer.stop(f"KDTree test, dim={dim}")
    return None, indices_test, time_train, time_test


# Detailed example
############################################################

print("Benchmarking olivetty faces ...")
print("-------------------------------\n")

# Small data set, useful to quickly check correctness of the algorithm.
olivetti_faces = fetch_olivetti_faces(data_home=DATA_PATH).data
train, test = train_test_split(
    olivetti_faces, test_size=0.2, random_state=1234
)

# Parameters in used algorithm
kwargs = {
    "data": train,
    "n_neighbors": 30,
    "verbose": True,
    "metric": "euclidean",
    "query_data": test,
    "k": 10,
}

# Ignore first pynndescent run (slow due to numba compilation)
_ = benchmark_pynndescent(**kwargs)
nn_train_py, nn_test_py, t_train_py, t_test_py = benchmark_pynndescent(
    **kwargs
)

nn_train_c, nn_test_c, t_train_c, t_test_c = benchmark_nndescent(**kwargs)
nn_train_bf, nn_test_bf, t_train_bf, t_test_bf = benchmark_bf(**kwargs)


# pynndescent accuracy
print("\nfaces_train: Accuracy pynndescent vs exact values")
accuracy(nn_train_py, nn_train_bf)
print("\nfaces_test: Accuracy pynndescent vs exact values")
accuracy(nn_test_py, nn_test_bf)

# Accuracy with plots
print("\nfaces_train: Accuracy nndescent vs exact values")
accuracy(nn_train_c, nn_train_bf)
print("\nfaces_test: Accuracy nndescent vs exact values")
accuracy_stats = accuracy(nn_test_c, nn_test_bf)

sns.histplot(accuracy_stats, kde=False)
plt.title("Distribution of accuracy per query point")
plt.xlabel("Accuracy")
plt.show()

# Brute force vs kdtrees (must be almost 1.0)
_, nn_test_kdt, t_train_kdt, t_test_kdt = benchmark_kdtree(**kwargs)
print("\nfaces_test: Accuracy bf vs kdtree")
accuracy(nn_test_bf, nn_test_kdt)

# Accuracy relative to pynndescent
print("\nfaces_train: Accuracy nndescent vs pynndescent")
acc_train = accuracy(nn_train_c, nn_train_py)

# Accuracy
print("\nfaces_test: Accuracy pynndescent vs exact values")
acc_test_py = accuracy(nn_test_py, nn_test_kdt)
print("\nfaces_test: Accuracy nndescent vs exact values")
acc_test_c = accuracy(nn_test_c, nn_test_kdt)

# Add results to summary
summary = [
    [
        "faces",
        t_train_py,
        t_train_c,
        t_train_c / t_train_py,
        np.mean(acc_train),
        t_test_py,
        t_test_c,
        t_test_c / t_test_py,
        np.mean(acc_test_py),
        np.mean(acc_test_c),
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
    "mnist": "mnist-784-euclidean",
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
    print(f"\nBenchmarking {name} ...")
    print("-------------------------------\n")
    train, test, dist, nn_test_ect = ann_benchmark_data(url_name)
    kwargs = {
        "data": train,
        "n_neighbors": 30,
        "verbose": True,
        "metric": distance_translation[dist],
        "query_data": test,
        "k": 10,
    }
    # Ignore first pynndescent run (slow due to numba compilation)
    _ = benchmark_pynndescent(**kwargs)
    nn_train_py, nn_test_py, t_train_py, t_test_py = benchmark_pynndescent(
        **kwargs
    )
    nn_train_c, nn_test_c, t_train_c, t_test_c = benchmark_nndescent(**kwargs)

    # Accuracy relative to pynndescent
    print(f"\n{name}_train: Accuracy nndescent vs pynndescent")
    acc_train = accuracy(nn_train_c, nn_train_py)
    # Accuracy relative to exact value
    nn_test_ect = nn_test_ect[:, 1 : (kwargs["k"] + 1)]
    print(f"\n{name}_test: Accuracy pynndescent vs exact values")
    acc_test_py = accuracy(nn_test_py, nn_test_ect)
    print(f"\n{name}_test: Accuracy nndescent vs exact values")
    acc_test_c = accuracy(nn_test_c, nn_test_ect)
    summary.append(
        [
            name,
            t_train_py,
            t_train_c,
            t_train_c / t_train_py,
            np.mean(acc_train),
            t_test_py,
            t_test_c,
            t_test_c / t_test_py,
            np.mean(acc_test_py),
            np.mean(acc_test_c),
        ],
    )

# Print Summary
print(
    "# Preliminary results: Benchmark test pynndescent (py) vs nndescent (c)"
)
print(
    "Data set  | py train [ms] | c train [ms] | ratio | py vs c match"
    " | py test [ms] | c test [ms] | ratio | py accuracy | c accuracy"
)
print(
    "----------|---------------|--------------|-------|--------------"
    "-|--------------|-------------|-------|-------------|-----------"
)
for nm, tr_py, tr_c, tr_r, tr_a, te_py, te_c, te_r, te_a_py, te_a_c in summary:
    print(
        f"{nm:<9} | {tr_py: 13.1f} | {tr_c: 12.1f} | {tr_r:5.3f} |"
        f" {tr_a: 13.3f} | {te_py: 12.1f} | {te_c: 11.1f} | {te_r:5.3f} |"
        f" {te_a_py: 11.3f} | {te_a_c: 10.3f}"
    )


# Test sparse matrix

train = fetch_20newsgroups_vectorized(data_home=DATA_PATH, subset="train").data
test = fetch_20newsgroups_vectorized(data_home=DATA_PATH, subset="test").data

kwargs = {
    "data": train,
    "n_neighbors": 30,
    "verbose": True,
    "metric": "cosine",
    "query_data": test,
    "k": 10,
}

nn_train_ect, _ = exact_nn(train, train, kwargs["n_neighbors"])
nn_test_ect, _ = exact_nn(train, test, kwargs["k"])

_ = benchmark_pynndescent(**kwargs)
nn_train_py, nn_test_py, t_train_py, t_test_py = benchmark_pynndescent(
    **kwargs
)

nn_train_c, nn_test_c, t_train_c, t_test_c = benchmark_nndescent(**kwargs)

acc_train = accuracy(nn_train_c, nn_train_py)
acc_test_py = accuracy(nn_test_py, nn_test_ect)
acc_test_c = accuracy(nn_test_c, nn_test_ect)

summary.append(
    [
        "20newsgroups",
        t_train_py,
        t_train_c,
        t_train_c / t_train_py,
        np.mean(acc_train),
        t_test_py,
        t_test_c,
        t_test_c / t_test_py,
        np.mean(acc_test_py),
        np.mean(acc_test_c),
    ],
)


# Print Summary
print("# Benchmark test pynndescent (py) vs nndescent (c)")
print(
    "Data set     | py train [ms] | c train [ms] | ratio | py vs c match"
    " | py test [ms] | c test [ms] | ratio | py accuracy | c accuracy"
)
print(
    "-------------|---------------|--------------|-------|--------------"
    "-|--------------|-------------|-------|-------------|-----------"
)
for nm, tr_py, tr_c, tr_r, tr_a, te_py, te_c, te_r, te_a_py, te_a_c in summary:
    print(
        f"{nm:<12} | {tr_py: 13.1f} | {tr_c: 12.1f} | {tr_r:5.3f} |"
        f" {tr_a: 13.3f} | {te_py: 12.1f} | {te_c: 11.1f} | {te_r:5.3f} |"
        f" {te_a_py: 11.3f} | {te_a_c: 10.3f}"
    )
