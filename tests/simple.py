"""
Very simple dataset to illustrate the functionality of the library.
"""

import numpy as np
import nndescent


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
data = np.array(
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


# Run NND algorithm
n_neighbors = 4
nnd = nndescent.NNDescent(data, n_neighbors=n_neighbors)

# Get result
nn_indices, nn_distances = nnd.neighbor_graph

# Calculate exact nearest neighbors for comparison.
nn_indices_ect, nn_distances_ect = nn_brute_force(data, data, n_neighbors)


print("\nInput data\n", data)
print("\nApproximate nearest neighbors\n", nn_indices)
print("\nExact nearest neighbors\n", nn_indices_ect)
print("\nApproximate nearest neighbors distances\n", nn_distances)
print("\nExact nearest neighbors distances\n", nn_distances_ect)

# Query data
query_data = np.array(
    [
        [5, 34],
        [65, 12],
        [44, 0],
        [18, 16],
        [52, 19],
        [35, 9],
        [1, 9],
    ],
    dtype=np.float32,
)

# Calculate nearest neighbors for each query point
k_query = 6
nn_query_indices, nn_query_distances = nnd.query(query_data, k=k_query)

# Calculate exact nearest neighbors for comparison.
nn_query_indices_ect, nn_query_distances_ect = nn_brute_force(data, query_data, k_query)

print("\nInput query data\n", query_data)
print("\nApproximate nearest neighbors of query points\n", nn_query_indices)
print("\nExact nearest neighbors of query points\n", nn_query_indices_ect)
print("\nApproximate nearest neighbors query distances\n", nn_query_distances)
print("\nExact nearest neighbors query distances\n", nn_query_distances_ect)
