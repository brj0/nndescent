# Nearest Neighbor Descent (nndescent)

Nearest Neighbor Descent (nndescent) is a C++ implementation of the nearest neighbor descent algorithm, designed for efficient and accurate approximate nearest neighbor search. With seamless integration into Python, it offers a powerful solution for constructing k-nearest neighbor graphs. This algorithm is based on the pynndescent library, originally written by Leland McInnes.


## Features

- Seamless integration into Python and effortless installation using `pip`.
- The handling of nndescent is very similar to that of pynndescent.
- Pure C++11 implementation utilizing OpenMP for parallel computation. No other libraries are needed.
- Currently tested only on Linux.
- Both dense and sparse matrices are supported.
- Implementation of multiple distance functions, i.e.
    - euclidean
    - manhattan
    - chebyshev
    - canberra
    - braycurtis
    - seuclidean
    - cosine
    - correlation
    - haversine
    - hamming
    - hellinger

Please note that not all distances have undergone thorough testing. Therefore, it is advised to use them with caution and at your own discretion.


## Installation

1. Clone the repository:

```sh
git clone https://github.com/brj0/nndescent.git
cd nndescent
```

2. The project can by build with:

```sh
pip install .
```

If you want to run the examples in `tests`, additional packages are needed. You can install them manually or install nndescent with the full option:

```sh
pip install .[full]
```

3. To run the examples in `tests` you must first download the datasets:

```sh
python tests/make_test_data.py
```


## Usage

In Python you can utilize the nndescent library in the following way:

```python
import numpy as np
import nndescent

# Data must be a 2D numpy array of dtype 'float32'.
data = np.random.randint(50, size=(20,3)).astype(np.float32)

# Run NND algorithm
nnd = nndescent.NNDescent(data, n_neighbors=4)

# Get result
nn_indices, nn_distances = nnd.neighbor_graph

# Query data must be a 2D numpy array of dtype 'float32'.
query_data = np.random.randint(50, size=(5,3)).astype(np.float32)

# Calculate nearest neighbors for each query point
nn_query_indices, nn_query_distances = nnd.query(query_data, k=6)
```

To compile and run the C++ examples use the following commands within the project folder:

```sh
mkdir build
cd build
cmake ..
make
./simple
```

For detailed usage in C++ and for further Python/C++ examples please refer to the examples provided in the `tests` directory of the repository and the code documentation.


## Performance

On my computer, the training phase of nndescent is approximately 10-15% faster than pynndescent. Furthermore, the search query phase shows a significant improvement, with >70% faster execution time. Below is the output obtained from running `tests/benchmark.py`, an ad hoc benchmark test that is not representative of all scenarios (low accuracy in the angular case for pynndescent). In this test, both nndescent and pynndescent were executed with the same parameters using either 'euclidean' or 'dot' as metric:


### Benchmark test pynndescent (py) vs nndescent (c)
Data set  | py train [ms] | c train [ms] | ratio | py vs c match | py test [ms] | c test [ms] | ratio | py accuracy | c accuracy
----------|---------------|--------------|-------|---------------|--------------|-------------|-------|-------------|-----------
faces     |         159.6 |        175.2 | 1.098 |         1.000 |       1636.5 |        17.5 | 0.011 |       1.000 |      0.999
fmnist    |       12281.2 |      10830.2 | 0.882 |         0.997 |       6001.6 |      1283.0 | 0.214 |       0.978 |      0.978
glove25   |      166062.1 |     102499.3 | 0.617 |         0.020 |     121915.6 |      9967.7 | 0.082 |       0.030 |      0.808
glove50   |      191170.0 |     138956.4 | 0.727 |         0.088 |     115711.5 |     10896.3 | 0.094 |       0.028 |      0.743
glove100  |      215343.9 |     181356.0 | 0.842 |         0.176 |     112383.3 |     12555.7 | 0.112 |       0.042 |      0.731
glove200  |      294725.6 |     244945.1 | 0.831 |         0.300 |     121324.7 |     18224.6 | 0.150 |       0.067 |      0.773
mnist     |       11778.2 |      10364.7 | 0.880 |         0.997 |       5798.9 |      1279.1 | 0.221 |       0.969 |      0.968
nytimes   |       67046.5 |      55050.4 | 0.821 |         0.729 |      23828.5 |      7320.4 | 0.307 |       0.546 |      0.810
sift      |      138194.7 |     109386.4 | 0.792 |         0.974 |      82608.6 |      8178.0 | 0.099 |       0.838 |      0.839

The compilation time and the lengthy numba loading time during runtime and import for 'pynndescent' are not considered in this ad hoc benchmark test. An [Ann-Benchmarks](https://github.com/erikbern/ann-benchmarks/tree/main) wrapper is planned for the future.


## Background

The theoretical background of NND is based on the following paper:

- Dong, Wei, Charikar Moses, and Kai Li. ["Efficient k-nearest neighbor graph construction for generic similarity measures."](https://www.cs.princeton.edu/cass/papers/www11.pdf) Proceedings of the 20th International Conference on World Wide Web. 2011.

In addition, the algorithm utilizes random projection trees for initializing
the nearest neighbor graph. The nndescent algorithm constructs a tree by
randomly selecting two points and splitting the data along a hyperplane passing
through their midpoint. For a more theoretical background, please refer to:

- DASGUPTA, Sanjoy; FREUND, Yoav. [Random projection trees and low dimensional manifolds](https://cseweb.ucsd.edu/~dasgupta/papers/rptree-stoc.pdf). In: Proceedings of the Fortieth Annual ACM Symposium on Theory of Computing. 2008.


## Contributing

Contributions are welcome! If you have any bug reports, feature requests, or suggestions, please open an issue or submit a pull request.


## License

This project is licensed under the [BSD-2-Clause license](LICENSE).


## Acknowledgements

This implementation is based on the original pynndescent library by Leland McInnes. I would like to express my gratitude for his work.

For more information, visit the [pynndescent GitHub repository](https://github.com/lmcinnes/pynndescent).

