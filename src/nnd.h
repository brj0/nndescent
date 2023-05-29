/**
 * @file nnd.h
 *
 * @brief Implements the Nearest Neighbor Descent algorithm for approximate
 * nearest neighbor search.
 *
 * This file contains the C++ implementation of the pynndescent library,
 * originally written by Leland McInnes, which performs approximate nearest
 * neighbor search.
 *
 * @see https://github.com/lmcinnes/pynndescent
 *
 * The algorithm is based on the following paper:
 *
 * Dong, Wei, Charikar Moses, and Kai Li. "Efficient k-nearest neighbor graph
 * construction for generic similarity measures." Proceedings of the 20th
 * International Conference on World Wide Web. 2011.
 *
 * @see https://dl.acm.org/doi/pdf/10.1145/1963405.1963487
 * @see https://www.cs.princeton.edu/cass/papers/www11.pdf
 *
 * Furthermore the algorithm utilizes random projection trees for initializing
 * the nearest neighbor graph, which is based on the following paper:
 *
 * DASGUPTA, Sanjoy; FREUND, Yoav. Random projection trees and low dimensional
 * manifolds. In: Proceedings of the Fortieth Annual ACM Symposium on Theory of
 * Computing. 2008. pp. 537-546.
 *
 * @see https://dl.acm.org/doi/pdf/10.1145/1374376.1374452
 * @see https://cseweb.ucsd.edu/~dasgupta/papers/rptree-stoc.pdf
 *
 * This implementation utilizes C++ and OpenMP for efficient computation. It
 * currently supports dense matrices and provides implementations for a subset
 * of distance functions. The main goal is to construct a k-nearest neighbor
 * graph quickly and accurately.
 */


#pragma once

#include <functional>

#include "distances.h"
#include "dtypes.h"
#include "utils.h"
#include "rp_trees.h"


namespace nndescent {

// Constants
const std::string PROJECT_VERSION = "0.0.0";
const char OLD = '0';
const char NEW = '1';
const int MAX_INT = std::numeric_limits<int>::max();
const int DEFAULT_K = 10;
const float DEFAULT_EPSILON = 0.1f;

using It = float*;
// using Metric = std::function<float(It, It, It)>;
using Metric = float (*)(It, It, It);
using Function1d = float (*)(float);

void nn_descent
(
    const Matrix<float> &data,
    HeapList<float> &current_graph,
    int n_neighbors,
    RandomState &rng_state,
    int max_candidates,
    int n_iters,
    float delta,
    bool verbose
);

float recall_accuracy(Matrix<int> apx, Matrix<int> ect);


struct Parms
{
    std::string metric="euclidean";
    int n_neighbors=30;
    int n_trees=NONE;
    int leaf_size=NONE;
    float pruning_degree_multiplier=1.5;
    float diversify_prob=1.0;
    bool tree_init=true;
    int seed=NONE;
    bool low_memory=true;
    int max_candidates=NONE;
    int n_iters=NONE;
    float delta=0.001;
    int n_threads=NONE;
    bool compressed=false;
    bool parallel_batch_queries=false;
    bool verbose=false;

    std::string algorithm="nnd";
};


class NNDescent
{
private:
    std::vector<RPTree> forest;
    Matrix<float> data;
    RandomState rng_state;
    void get_distance_function();
    Metric dist;
    Function1d distance_correction=NULL;
    bool angular_trees;
    bool search_function_prepared=false;

public:
    RPTree search_tree;
    HeapList<float> search_graph;
    void prepare();
    Matrix<int> query_indices;
    Matrix<float> query_distances;
    void query(
        const Matrix<float> &query_data,
        int k=DEFAULT_K,
        float epsilon=DEFAULT_EPSILON
    );
    std::string metric;
    int n_neighbors;
    int n_trees;
    int leaf_size;
    float pruning_degree_multiplier;
    float diversify_prob;
    bool tree_init;
    int seed;
    bool low_memory;
    int max_candidates;
    int n_iters;
    float delta;
    int n_threads;
    bool compressed;
    bool parallel_batch_queries;
    bool verbose;
    std::string algorithm;

    HeapList<float> current_graph;

    NNDescent() {}
    NNDescent(Matrix<float> &input_data, Parms &parms);
    NNDescent(Matrix<float> &input_data, int n_neighbors);
    int data_size() {return data.nrows();}
    Matrix<int> neighbor_indices;
    Matrix<float> neighbor_distances;
    Matrix<int> brute_force();
    void query_brute_force(const Matrix<float> &query_data, int k);
    void start();
    void set_parameters(Parms &parms);
    friend std::ostream& operator<<(std::ostream &out, const NNDescent &nnd);
};

} // namespace nndescent
