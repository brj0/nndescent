
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


IntMatrix make_rp_tree
(
    const Matrix<float> &data,
    unsigned int leaf_size,
    RandomState &rng_state
);

void rand_tree_split
(
    const Matrix<float> &data,
    IntVec &parent,
    IntVec &child0,
    IntVec &child1
);

