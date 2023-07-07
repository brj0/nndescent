"""
20newsgroups is a sparse dataset of dimension 11'314x130'107 (train) and
7'532x130'107 (test)
"""

import os

from sklearn.datasets import fetch_20newsgroups_vectorized
import nndescent
import numpy as np
import scipy.sparse
from sklearn.neighbors import NearestNeighbors


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


def exact_nn(train, test, n_neighbors):
    """Calculates exact nearest neighbors (of test in train)."""
    a, b = train.shape
    c, d = test.shape
    fname_nn = f"20newsgroups-{a}x{b}-{c}x{d}_nn.npy"
    fname_dist = f"20newsgroups-{a}x{b}-{c}x{d}_dist.npy"
    fpath_nn = os.path.join(DATA_PATH, fname_nn)
    fpath_dist = os.path.join(DATA_PATH, fname_dist)
    if not (os.path.exists(fpath_nn) and os.path.exists(fpath_dist)):
        print(f"Exact k-NN for comparison not cached; calculating now...")
        nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        nn_model.fit(train)
        knn = nn_model.kneighbors(test)
        np.save(fpath_nn, knn[1])
        np.save(fpath_dist, knn[0])
        print(f"Calculation done. Result cached.")
    nn = np.load(fpath_nn)
    dist = np.load(fpath_dist)
    return nn, dist


# Data (as CSR Matrix)
csr_data = fetch_20newsgroups_vectorized(
    data_home=DATA_PATH, subset="train"
).data
csr_query_data = fetch_20newsgroups_vectorized(
    data_home=DATA_PATH, subset="test"
).data


# NEAREST NEIGHBORS - TRAINING

# Run NND algorithm.
n_neighbors = 30
nnd = nndescent.NNDescent(csr_data, n_neighbors=n_neighbors, metric="cosine")

# Get result
nn_indices, nn_distances = nnd.neighbor_graph

# Calculate exact nearest neighbors for comparison.
nn_indices_ect, nn_distances_ect = exact_nn(csr_data, csr_data, n_neighbors)

print("\nInput csr_data\n", csr_data)
print("\nApproximate nearest neighbors\n", nn_indices)
print("\nExact nearest neighbors\n", nn_indices_ect)
print("\nApproximate nearest neighbors distances\n", nn_distances)
print("\nExact nearest neighbors distances\n", nn_distances_ect)
print("\nAccuracy of nndescent algorithm compared with exact values (train):")
accuracy(nn_indices, nn_indices_ect)


# NEAREST NEIGHBORS - TESTING

# Calculate nearest neighbors for each query point.
k_query = 10
nn_query_indices, nn_query_distances = nnd.query(csr_query_data, k=k_query)

# Calculate exact nearest neighbors for comparison.
nn_query_indices_ect, nn_query_distances_ect = exact_nn(
    csr_data, csr_query_data, k_query
)

print("\nInput query csr_data\n", csr_data)
print("\nApproximate nearest neighbors of query points\n", nn_query_indices)
print("\nExact nearest neighbors of query points\n", nn_query_indices_ect)
print("\nApproximate nearest neighbors query distances\n", nn_query_distances)
print("\nExact nearest neighbors query distances\n", nn_query_distances_ect)
print("\nAccuracy of nndescent algorithm compared with exact values (test):")
accuracy(nn_query_indices, nn_query_indices_ect)
