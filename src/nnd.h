#pragma once


#include <functional>

#include "distances.h"
#include "dtypes.h"
#include "utils.h"

const std::string PROJECT_VERSION = "0.0.0";

const char OLD = '0';
const char NEW = '1';

const int MAX_INT = std::numeric_limits<int>::max();

// using It = std::vector<float>::const_iterator;
using It = float*;
// using Metric = std::function<float(It, It, It)>;
using Metric = float (*)(It, It, It);
using Function1d = float (*)(float);

void nn_descent
(
    const Matrix<float> &data,
    HeapList<float> &current_graph,
    int n_neighbors,
    RandomState rng_state,
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
    const Matrix<float> data;
    RandomState rng_state;
    void get_distance_function();
    Metric dist;
    Function1d distance_correction=NULL;
    bool angular_trees;
    HeapList<float> search_graph;

public:
    void init_search_graph();
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

    NNDescent(Matrix<float> &input_data, Parms &parms);
    NNDescent(Matrix<float> &input_data, int n_neighbors);
    int data_size() {return data.nrows();}
    Matrix<int> neighbor_indices;
    Matrix<float> neighbor_distances;
    Matrix<int> brute_force();
    void start();
    void set_parameters(Parms &parms);
    friend std::ostream& operator<<(std::ostream &out, const NNDescent &nnd);
};
