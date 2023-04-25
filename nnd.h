#pragma once


#include "dtypes.h"
#include "utils.h"

void nn_descent(
    SlowMatrix &data,
    std::vector<NNHeap> &current_graph,
    int n_neighbors,
    int max_candidates=50,
    int n_iters=10,
    float delta=0.001f,
    bool rp_tree_init=true,
    bool verbose=true
);
IntMatrix nn_brute_force(SlowMatrix data, int k);
double recall_accuracy(IntMatrix apx, IntMatrix ect);
inline double dist(
    const std::vector<double> &v0,
    const std::vector<double> &v1
);

struct Parms
{
    SlowMatrix data;
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
    int n_jobs=NONE;
    bool compressed=false;
    bool parallel_batch_queries=false;
    bool verbose=false;
};


class NNDescent
{
    private:
        const SlowMatrix data;
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
        int n_jobs=NONE;
        bool compressed=false;
        bool parallel_batch_queries=false;
        bool verbose=false;

        std::vector<NNHeap> current_graph;
        RandomState rng_state;

    public:
        NNDescent(SlowMatrix input_data, int k);
        NNDescent(Parms parms);
        int data_size() {return this->data.size();}
        void print();
        IntMatrix neighbor_graph;
        void start();
};
