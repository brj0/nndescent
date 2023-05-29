"""
ANN Benchmark example of dimension 60'000x784 (train) and 10'000x784 (test)
"""

import os
from urllib.request import urlretrieve

import h5py
import nndescent
import numpy as np

DATA_PATH = os.path.expanduser("~/Downloads/nndescent_test_data")


def accuracy(approx_neighbors, true_neighbors):
    """Returns accuracy of algorithm when compared with exact values."""
    result = np.zeros(approx_neighbors.shape[0])
    for i in range(approx_neighbors.shape[0]):
        n_correct = np.intersect1d(
            approx_neighbors[i], true_neighbors[i]
        ).shape[0]
        result[i] = n_correct / true_neighbors.shape[1]
    print(f"Average accuracy of {np.mean(result)}\n")


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


# Data
data, query_data, _, query_result = ann_benchmark_data(
    "fashion-mnist-784-euclidean"
)

# NEAREST NEIGHBORS - TRAINING

# Run NND algorithm.
n_neighbors = 30
nnd = nndescent.NNDescent(data, n_neighbors=n_neighbors)

# Get result
nn_indices, nn_distances = nnd.neighbor_graph

print("\nInput data\n", data)
print("\nApproximate nearest neighbors\n", nn_indices)
print("\nApproximate nearest neighbors distances\n", nn_distances)


# NEAREST NEIGHBORS - TESTING

# Calculate nearest neighbors for each query point.
k_query = 10
nn_query_indices, nn_query_distances = nnd.query(query_data, k=k_query)

# Get precomputed exact values of k_query-NN

nn_query_indices_ect = query_result[:, 1 : (k_query + 1)]


print("\nInput query data\n", query_data)
print("\nApproximate nearest neighbors of query points\n", nn_query_indices)
print("\nExact nearest neighbors of query points\n", nn_query_indices_ect)
print("\nAccuracy of nndescent algorithm compared with exact values (test):")
accuracy(nn_query_indices, nn_query_indices_ect)
