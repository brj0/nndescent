"""
Olivetty faces is a 400x4096 image dataset useful for quick checking
and debugging.
"""

import os

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import nndescent
import numpy as np
import scipy.sparse

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


def nn_brute_force(train, test, n_neighbors):
    """Calculates nearest neighbors (of test in train) using brute force."""
    nn_indices_test = []
    nn_distances_test = []
    for pnt_test in test:
        distances = [
            [i, np.linalg.norm(pnt_test - pnt_train)]
            for i, pnt_train in enumerate(train)
        ]
        distances.sort(key=lambda x: x[1])
        nn_indices_test.append([x for x, _ in distances[:n_neighbors]])
        nn_distances_test.append([y for _, y in distances[:n_neighbors]])
    return np.array(nn_indices_test), np.array(nn_distances_test)


# Data
olivetti_faces = fetch_olivetti_faces(data_home=DATA_PATH).data
data, query_data = train_test_split(
    olivetti_faces, test_size=0.2, random_state=1234
)

# Convert to sparse format
csr_data = scipy.sparse.csr_matrix(data)
csr_query_data = scipy.sparse.csr_matrix(query_data)

# NEAREST NEIGHBORS - TRAINING

# Run NND algorithm.
n_neighbors = 30
nnd = nndescent.NNDescent(csr_data, n_neighbors=n_neighbors)

# Get result
nn_indices, nn_distances = nnd.neighbor_graph

# Calculate exact nearest neighbors for comparison.
nn_indices_ect, nn_distances_ect = nn_brute_force(data, data, n_neighbors)

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
nn_query_indices_ect, nn_query_distances_ect = nn_brute_force(data, query_data, k_query)


print("\nInput query csr_data\n", csr_data)
print("\nApproximate nearest neighbors of query points\n", nn_query_indices)
print("\nExact nearest neighbors of query points\n", nn_query_indices_ect)
print("\nApproximate nearest neighbors query distances\n", nn_query_distances)
print("\nExact nearest neighbors query distances\n", nn_query_distances_ect)
print("\nAccuracy of nndescent algorithm compared with exact values (test):")
accuracy(nn_query_indices, nn_query_indices_ect)
