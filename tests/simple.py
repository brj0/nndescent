"""
Simple toy dataset.
"""

import nndescent

# data
data = [
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
]


# Run NND algorithm.
nnd = nndescent.NNDescent(data, n_neighbors=5)

# Get result
nn_graph_indices, nn_graph_distances = nnd.neighbor_graph

print("\nInput data\n", data)
print("\nNearest neighbor graph indices\n", nn_graph_indices)
print("\nNearest neighbor graph distances\n", nn_graph_distances)
