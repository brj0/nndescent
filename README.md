# Nearest Neighbor Descent (nndescent)

Nearest Neighbor Descent (nndescent) is a C++ implementation of the nearest neighbor descent algorithm, designed for efficient and accurate approximate nearest neighbor search. With seamless integration into Python, it offers a powerful solution for constructing k-nearest neighbor graphs. This algorithm is based on the [pynndescent library](https://github.com/lmcinnes/pynndescent).


## Features

- Seamless integration into Python and effortless installation using `pip`.
- The handling of nndescent is very similar to that of pynndescent.
- Pure C++11 implementation utilizing OpenMP for parallel computation. No other external libraries are needed.
- Currently tested only on Linux.
- Both dense and sparse matrices are supported.
- Implementation of multiple distance functions, i.e.
    - Bray-Curtis
    - Canberra
    - Chebyshev
    - Circular Kantorovich (no sparse verion)
    - Correlation
    - Cosine
    - Dice
    - Dot
    - Euclidean
    - Hamming
    - Haversine
    - Hellinger
    - Hellinger
    - Jaccard
    - Jensen-Shannon
    - Kulsinski
    - Manhattan
    - Matching
    - Minkowski
    - Rogers-Tanimoto
    - Russell-Rao
    - Sokal-Michener
    - Sokal-Sneath
    - Spearman's Rank Correlation (no sparse version)
    - Symmetric KL Divergence
    - TSSS
    - True Angular
    - Wasserstein 1D (no sparse version)
    - Yule

Please note that not all distances have undergone thorough testing. Therefore, it is advised to use them with caution and at your own discretion.


## Installation

### From PyPI

You can install nndescent directly from PyPI using pip:

```sh
pip install nndescent
```

If you want to run the examples in `tests`, additional packages are needed. You can install them manually or install nndescent with the full option:

```sh
pip install nndescent[full]
```

### From Source

1. Clone the repository:

```sh
git clone https://github.com/brj0/nndescent.git
cd nndescent
```

2. Build and install the package:

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

On my computer, the training phase of nndescent is approximately 15% faster than pynndescent for dense matrices, and 75% faster for sparse matrices. Furthermore, the search query phase shows a significant improvement, with >70% faster execution time. Below is the output obtained from running `tests/benchmark.py`, an ad hoc benchmark test. In this test, both nndescent and pynndescent were executed with the same parameters using either 'euclidean' or 'dot' as metric:

# Benchmark test pynndescent (py) vs nndescent (c)
Data set     | py train [ms] | c train [ms] | ratio | py vs c match | py test [ms] | c test [ms] | ratio | py accuracy | c accuracy
-------------|---------------|--------------|-------|---------------|--------------|-------------|-------|-------------|-----------
faces        |         149.8 |        145.9 | 0.974 |         1.000 |       1663.7 |        18.4 | 0.011 |       1.000 |      0.999
fmnist       |       11959.2 |      10768.7 | 0.900 |         0.997 |       5754.8 |      1334.1 | 0.232 |       0.978 |      0.978
glove25      |      149754.2 |     101864.0 | 0.680 |         0.964 |      98740.6 |      9907.4 | 0.100 |       0.796 |      0.808
glove50      |      192965.8 |     137171.8 | 0.711 |         0.885 |      99750.8 |     10647.7 | 0.107 |       0.705 |      0.743
glove100     |      218202.9 |     180088.4 | 0.825 |         0.815 |      98770.2 |     12080.4 | 0.122 |       0.651 |      0.731
glove200     |      287206.6 |     243466.6 | 0.848 |         0.772 |     101639.4 |     17615.6 | 0.173 |       0.622 |      0.773
mnist        |       11319.7 |      10188.1 | 0.900 |         0.997 |       5725.9 |      1273.8 | 0.222 |       0.969 |      0.968
nytimes      |       63323.8 |      55638.1 | 0.879 |         0.814 |      23632.1 |      7108.9 | 0.301 |       0.614 |      0.811
sift         |      131711.4 |     105826.0 | 0.803 |         0.974 |      82503.7 |      7957.9 | 0.096 |       0.838 |      0.839
20newsgroups |      107339.0 |      28339.7 | 0.264 |         0.922 |      67518.6 |     22513.1 | 0.333 |       0.858 |      0.929

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

This implementation is based on the original pynndescent library by Leland McInnes. I would like to acknowledge and appreciate his work as a source of inspiration for this project.

For more information, visit the [pynndescent GitHub repository](https://github.com/lmcinnes/pynndescent).
