# Nearest Neighbor Descent (nndescent)

Nearest Neighbor Descent (nndescent) is a C++ implementation of the pynndescent library, originally written by Leland McInnes, which performs approximate nearest neighbor search. The goal of this algorithm is to construct a k-nearest neighbor graph quickly and accurately.

## Background

The theoretical background of NND is based on the following paper:
- Dong, Wei, Charikar Moses, and Kai Li. "Efficient k-nearest neighbor graph construction for generic similarity measures." Proceedings of the 20th International Conference on World Wide Web. 2011.

In addition, the algorithm utilizes random projection trees for initializing the nearest neighbor graph, based on the following paper:
- DASGUPTA, Sanjoy; FREUND, Yoav. Random projection trees and low dimensional manifolds. In: Proceedings of the Fortieth Annual ACM Symposium on Theory of Computing. 2008.

## Features

- C++ implementation utilizing OpenMP for efficient computation
- Support for dense matrices
- Implementation of a subset of distance functions

## Installation

1. Clone the repository:

```sh
git clone https://github.com/brj0/nndescent.git
cd nndescent
```

2. Build the project:

```sh
pip install .
```

3. Run the examples in `tests`. To build the dataset you should first run `make_test_data.py`

## Performance

On my computer, the training phase of nndescent is approximately 5-10% faster than pynndescent. Additionally, the search query phase is approximately 75% faster. Below is the output obtained from running tests/benchmark.py:

### Benchmark test pynndescent vs nndescent
Data set  | py train [ms] | c train [ms] | ratio | py vs c match | py test [ms] | c test [ms] | ratio | py accuracy | c accuracy
----------|---------------|--------------|-------|---------------|--------------|-------------|-------|-------------|-----------
faces     |         191.8 |        190.0 | 0.991 |         1.000 |       1631.6 |        20.5 | 0.013 |       1.000 |      0.999
fmnist    |       13587.5 |      12935.1 | 0.952 |         0.997 |       6751.2 |      1757.2 | 0.260 |       0.978 |      0.978
mnist     |       14187.2 |      12712.9 | 0.896 |         0.997 |       6664.2 |      1665.1 | 0.250 |       0.969 |      0.968

The compilation time and the long numba loading time during import in Python for pynndescent are not taken into account here and are not required in nndescent.

## Usage

Please refer to the examples provided in the repository for instructions on how to use the NND library in your projects.

## Contributing

Contributions are welcome! If you have any bug reports, feature requests, or suggestions, please open an issue or submit a pull request.

## License

This project is licensed under the [BSD-2-Clause license](LICENSE).

## Acknowledgements

This implementation is based on the original pynndescent library by Leland McInnes. I would like to express my gratitude for his work.

For more information, visit the [pynndescent GitHub repository](https://github.com/lmcinnes/pynndescent).

