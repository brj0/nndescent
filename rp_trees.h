/*
 * DASGUPTA, Sanjoy; FREUND, Yoav. Random projection trees and low dimensional
 * manifolds. In: Proceedings of the fortieth annual ACM symposium on Theory of
 * computing. 2008. S. 537-546.
 *
 * https://dl.acm.org/doi/pdf/10.1145/1374376.1374452
 * https://cseweb.ucsd.edu/~dasgupta/papers/rptree-stoc.pdf
 */

#pragma once


#include "dtypes.h"


std::vector<IntMatrix> make_forest
(
    const Matrix<float> &data,
    int n_trees,
    int leaf_size,
    RandomState &rng_state
);

