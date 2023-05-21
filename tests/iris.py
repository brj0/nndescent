"""
Iris dataset.
"""

from sklearn.datasets import load_iris
import nndescent

# data
data = load_iris().data

# Run NND algorithm.
nnd = nndescent.NNDescent(data, n_neighbors=5)

# Get result
nn_graph_indices, nn_graph_distances = nnd.neighbor_graph

print("\nInput data\n", data)
print("\nNearest neighbor graph indices\n", nn_graph_indices)
print("\nNearest neighbor graph distances\n", nn_graph_distances)
