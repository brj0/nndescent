#pragma once


#include "dtypes.h"
#include "utils.h"

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

inline float dist
(
    const Matrix<float> &data,
    size_t row0,
    size_t row1
);

struct Parms
{
    Matrix<float> data;
    std::string metric="euclidean";
    // metric_kwds=NULL;
    int n_neighbors=30;
    int n_trees=NONE;
    int leaf_size=NONE;
    float pruning_degree_multiplier=1.5;
    float diversify_prob=1.0;
    int n_search_trees=1;
    bool tree_init=true;
    // init_graph=NULL;
    // init_dist=NULL;
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
        std::string metric="euclidean";
        // metric_kwds=NULL;
        int n_neighbors=30;
        int n_trees=NONE;
        int leaf_size=NONE;
        float pruning_degree_multiplier=1.5;
        float diversify_prob=1.0;
        int n_search_trees=1;
        bool tree_init=true;
        // init_graph=NULL;
        // init_dist=NULL;
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
        RandomState rng_state;

    public:
        HeapList<float> current_graph;

        NNDescent(Matrix<float> input_data, int k);
        NNDescent(Parms parms);
        int data_size() {return data.nrows();}
        void print();
        Matrix<int> neighbor_graph;
        Matrix<int> bf_graph;
        Matrix<int> brute_force();
        void start();
        friend std::ostream& operator<<(std::ostream &out, const NNDescent &nnd);
};
