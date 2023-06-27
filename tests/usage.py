import numpy as np
import nndescent

# Data must be a 2D numpy array of dtype 'float32'.
data = np.random.randint(50, size=(20, 3)).astype(np.float32)

# Run NND algorithm
nnd = nndescent.NNDescent(data, n_neighbors=4)

# Get result
nn_indices, nn_distances = nnd.neighbor_graph

# Query data must be a 2D numpy array of dtype 'float32'.
query_data = np.random.randint(50, size=(5, 3)).astype(np.float32)

# Calculate nearest neighbors for each query point
nn_query_indices, nn_query_distances = nnd.query(query_data, k=6)
