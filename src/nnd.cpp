/**
 * @file nnd.cpp
 *
 * @brief Implementaton of nearest neighbor descent.
 */


#include <assert.h>
#include <iomanip>
#include <iostream>
#include <thread>
#include <map>
#include <vector>
#include <stdexcept>

#include "nnd.h"

namespace nndescent
{


/**
 * @brief Initializes the nearest neighbor graph with random neighbors for
 * missing nodes.
 *
 * @param data The input data matrix.
 * @param current_graph The current nearest neighbor graph to initialize.
 * @param n_neighbors The number of neighbors.
 * @param dist The distance metric used for neighbor selection.
 * @param rng_state The random state used for randomization.
 */
template<class MatrixType>
void init_random
(
    const MatrixType &data,
    HeapList<float> &current_graph,
    size_t n_neighbors,
    const DistanceFunction &dist,
    RandomState &rng_state
)
{
    for (size_t idx0 = 0; idx0 < current_graph.nheaps(); ++idx0)
    {
        int missing = n_neighbors - current_graph.size(idx0);

        // Sample nodes
        for (int j = 0; j < missing; ++j)
        {
            int idx1 = rand_int(rng_state) % current_graph.nheaps();
            float d = dist(data, idx0, idx1);
            current_graph.checked_push(idx0, idx1, d, NEW);
        }
    }
}


/**
 * @brief Adds every node to its own neighborhod.
 *
 * @param current_graph The current nearest neighbor graph.
 */
void add_zero_node
(
    HeapList<float> &current_graph
)
{
    for (size_t idx0 = 0; idx0 < current_graph.nheaps(); ++idx0)
    {
        current_graph.checked_push(idx0, idx0, 0.0f, NEW);
    }
}


void correct_distances
(
    Function1d distance_correction,
    Matrix<float> &in,
    Matrix<float> &out
)
{
    out.resize(in.nrows(), in.ncols());
    for (size_t i = 0; i < in.nrows(); ++i)
    {
        for (size_t j = 0; j < in.ncols(); ++j)
        {
            out(i, j) = distance_correction(in(i,j));
        }
    }
    return;
}


 /**
 * @brief Updates the nearest neighbor graph using leaves constructed from
 * random projection trees.
 *
 * This function updates the nearest neighbor graph by incorporating the
 * information from the leaves constructed from random projection trees.
 *
 * @param data The input data matrix.
 * @param current_graph The current nearest neighbor graph.
 * @param leaf_array The matrix of leaf indices.
 * @param dist The distance metric used for nearest neighbor calculations.
 * @param n_threads The number of threads to use for parallelization.
 */
template<class MatrixType>
void update_by_leaves
(
    const MatrixType &data,
    HeapList<float> &current_graph,
    Matrix<int> &leaf_array,
    const DistanceFunction &dist,
    int n_threads
)
{
    int n_leaves = leaf_array.nrows();
    int leaf_size = leaf_array.ncols();
    int block_size = n_leaves / n_threads;

    std::vector<std::vector<NNUpdate>> updates(n_threads);

    // Generate leaf updates
    #pragma omp parallel for
    for (int thread = 0; thread < n_threads; ++thread)
    {
        int block_start  = thread * block_size;
        int block_end = (thread + 1) * block_size;
        block_end = (thread == n_threads) ? n_leaves : block_end;

        for (int i = block_start; i < block_end; ++i)
        {
            for (int j = 0; j < leaf_size; ++j)
            {
                int idx0 = leaf_array(i, j);
                if (idx0 == NONE)
                {
                    break;
                }
                for (int k = j + 1; k < leaf_size; ++k)
                {
                    int idx1 = leaf_array(i, k);
                    if (idx1 == NONE)
                    {
                        break;
                    }
                    float d = dist(data, idx0, idx1);
                    if (
                        (d < current_graph.max(idx0)) ||
                        (d < current_graph.max(idx1))
                    )
                    {
                        NNUpdate update = {idx0, idx1, d};
                        updates[thread].push_back(update);
                    }
                }
            }
        }
    }

    // Apply updates
    #pragma omp parallel for
    for (int thread = 0; thread < n_threads; ++thread)
    {
        for (const auto& updates_vec : updates)
        {
            for (const auto& update : updates_vec)
            {
                int idx0 = update.idx0;
                int idx1 = update.idx1;
                assert(idx0 >= 0);
                assert(idx1 >= 0);
                float d = update.key;
                if (idx0 % n_threads == thread)
                {
                    current_graph.checked_push(idx0, idx1, d, NEW);
                }
                if (idx1 % n_threads == thread)
                {
                    current_graph.checked_push(idx1, idx0, d, NEW);
                }
            }
        }
    }
}


/*
 * @brief Builds a heap of candidate neighbors for nearest neighbor descent.
 *
 * For each vertex, the candidate neighbors include any current neighbors and
 * any vertices that have the vertex as one of their nearest neighbors.
 *
 * @param current_graph The current nearest neighbor graph.
 * @param new_candidates The empty heap of new candidate neighbors.
 * @param old_candidates The empty heap of old candidate neighbors.
 * @param rng_state The random state used for randomization.
 * @param n_threads The number of threads to use for parallelization.
 */
void sample_candidates
(
    HeapList<float> &current_graph,
    HeapList<int> &new_candidates,
    HeapList<int> &old_candidates,
    RandomState rng_state,
    int n_threads
)
{
    #pragma omp parallel for
    for (int thread = 0; thread < n_threads; ++thread)
    {
        RandomState local_rng_state;
        for (int state = 0; state < STATE_SIZE; ++state)
        {
            local_rng_state[state] = rng_state[state] + thread + 1;
        }
        for (int idx0 = 0; idx0 < (int)current_graph.nheaps(); ++idx0)
        {
            for (int j = 0; j < (int)current_graph.nnodes(); ++j)
            {
                int idx1 = current_graph.indices(idx0, j);
                char flag = current_graph.flags(idx0, j);

                if (idx1 == NONE)
                {
                    continue;
                }

                // Setting a random priority results in a random sampling
                // of at most 'max_candidates' candidates.
                int priority = rand_int(local_rng_state);

                if (flag == NEW)
                {
                    if (idx0 % n_threads == thread)
                    {
                        new_candidates.checked_push(idx0, idx1, priority);
                    }
                    // Reverse nearest neighbours.
                    if (idx1 % n_threads == thread)
                    {
                        new_candidates.checked_push(idx1, idx0, priority);
                    }
                }
                else
                {
                    if (idx0 % n_threads == thread)
                    {
                        old_candidates.checked_push(idx0, idx1, priority);
                    }
                    // Reverse nearest neighbours.
                    if (idx1 % n_threads == thread)
                    {
                        old_candidates.checked_push(idx1, idx0, priority);
                    }
                }
            }
        }
    }
    // Mark sampled nodes in current_graph as old.
    for (size_t idx0 = 0; idx0 < current_graph.nheaps(); ++idx0)
    {
        for (size_t j = 0; j < current_graph.nnodes(); ++j)
        {
            int idx1 = current_graph.indices(idx0, j);
            for (size_t k = 0; k < new_candidates.nnodes(); ++k)
            {
                if (new_candidates.indices(idx0, k) == idx1)
                {
                    current_graph.flags(idx0, j) = OLD;
                    break;
                }
            }
        }
    }
}


/*
 * @brief Generates potential nearest neighbor updates.
 *
 * This function generates potential nearest neighbor updates, which are
 * objects containing two identifiers that identify nodes and their
 * corresponding distance.
 *
 * @param data The input data matrix.
 * @param current_graph The current nearest neighbor graph.
 * @param new_candidate_neighbors The heap of new candidate neighbors.
 * @param old_candidate_neighbors The heap of old candidate neighbors.
 * @param dist The distance metric used for calculating distances.
 * @param n_threads The number of threads to use for parallelization.
 * @param verbose The verbosity flag indicating whether to enable logging
 * messages.
 * @return A vector of vectors of NNUpdate objects representing the nearest
 * neighbor updates.
 */
template<class MatrixType>
std::vector<std::vector<NNUpdate>> generate_graph_updates
(
    const MatrixType &data,
    HeapList<float> &current_graph,
    HeapList<int> &new_candidate_neighbors,
    HeapList<int> &old_candidate_neighbors,
    const DistanceFunction &dist,
    int n_threads
)
{
    assert(data.nrows() == new_candidate_neighbors.nheaps());
    assert(data.nrows() == old_candidate_neighbors.nheaps());
    std::vector<std::vector<NNUpdate>> updates(n_threads);
    int size_new = new_candidate_neighbors.nheaps();
    int block_size = size_new / 4 + 1;
    #pragma omp parallel for
    for (int thread = 0; thread < n_threads; ++thread)
    {
        size_t block_start = thread * block_size;
        size_t block_end = std::min((thread + 1) * block_size, size_new);
        for (size_t i = block_start; i < block_end; ++i)
        {
            for (size_t j = 0; j < new_candidate_neighbors.nnodes(); ++j)
            {
                int idx0 = new_candidate_neighbors.indices(i, j);
                if (idx0 == NONE)
                {
                    continue;
                }
                for
                (
                    size_t k = j + 1; k < new_candidate_neighbors.nnodes(); ++k
                )
                {
                    int idx1 = new_candidate_neighbors.indices(i, k);
                    if (idx1 == NONE)
                    {
                        continue;
                    }
                    float d = dist(data, idx0, idx1);
                    if (
                        (d < current_graph.max(idx0)) ||
                        (d < current_graph.max(idx1))
                    )
                    {
                        NNUpdate update = {idx0, idx1, d};
                        updates[thread].push_back(update);
                    }

                }
                for (size_t k = 0; k < old_candidate_neighbors.nnodes(); ++k)
                {
                    int idx1 = old_candidate_neighbors.indices(i, k);
                    if (idx1 == NONE)
                    {
                        continue;
                    }
                    float d = dist(data, idx0, idx1);
                    if (
                        (d < current_graph.max(idx0)) ||
                        (d < current_graph.max(idx1))
                    )
                    {
                        NNUpdate update = {idx0, idx1, d};
                        updates[thread].push_back(update);
                    }
                }
            }
        }
    }

    return updates;
}


/*
 * @brief Applies graph updates to the current nearest neighbor graph.
 *
 * @param current_graph The current nearest neighbor graph.
 * @param updates A vector of vectors of NNUpdate objects representing the
 * potential graph updates.
 * @param n_threads The number of threads to use for parallelization.
 * @return The number of updates applied to the graph.
 */
int apply_graph_updates(
    HeapList<float>& current_graph,
    std::vector<std::vector<NNUpdate>>& updates,
    int n_threads
);

int apply_graph_updates
(
    HeapList<float> &current_graph,
    std::vector<std::vector<NNUpdate>> &updates,
    int n_threads
)
{
    int n_changes = 0;

    #pragma omp parallel for
    for (int thread = 0; thread < n_threads; ++thread)
    {
        for (const auto& updates_vec : updates)
        {
            for (const auto& update : updates_vec)
            {
                int idx0 = update.idx0;
                int idx1 = update.idx1;
                assert(idx0 >= 0);
                assert(idx1 >= 0);
                float d = update.key;
                if (idx0 % n_threads == thread)
                {
                    n_changes += current_graph.checked_push(idx0, idx1, d, NEW);
                }
                if (idx1 % n_threads == thread)
                {
                    n_changes += current_graph.checked_push(idx1, idx0, d, NEW);
                }
            }
        }
    }

    return n_changes;
}


template<class MatrixType>
void nn_descent
(
    const MatrixType &data,
    HeapList<float> &current_graph,
    int n_neighbors,
    RandomState &rng_state,
    int max_candidates,
    const DistanceFunction &dist,
    int n_iters,
    float delta,
    int n_threads,
    bool verbose
)
{
    assert(current_graph.nheaps() == data.nrows());

    log("NN descent for " + std::to_string(n_iters) + " iterations", verbose);

    for (int iter = 0; iter < n_iters; ++iter)
    {
        log(
            (
                "\t" + std::to_string(iter + 1) + "  /  "
                + std::to_string(n_iters)
            ),
            verbose
        );

        HeapList<int> new_candidates(data.nrows(), max_candidates, MAX_INT);
        HeapList<int> old_candidates(data.nrows(), max_candidates, MAX_INT);

        sample_candidates(
            current_graph,
            new_candidates,
            old_candidates,
            rng_state,
            n_threads
        );

        std::vector<std::vector<NNUpdate>> updates = generate_graph_updates(
            data,
            current_graph,
            new_candidates,
            old_candidates,
            dist,
            n_threads
        );

        int cnt = apply_graph_updates(
            current_graph,
            updates,
            n_threads
        );
        log("\t\t" + std::to_string(cnt) + " updates applied", verbose);

        if (cnt < delta * data.nrows() * n_neighbors)
        {
            log(
                "Stopping threshold met -- exiting after "
                    + std::to_string(iter + 1) + " iterations",
                verbose
            );
            break;
        }
    }
    log("NN descent done.", verbose);
}


float recall_accuracy(Matrix<int> apx, Matrix<int> ect)
{
    assert(apx.nrows() == ect.nrows());
    assert(apx.ncols() == ect.ncols());
    int hits = 0;
    int nrows = apx.nrows();
    int ncols = apx.ncols();

    for (size_t i = 0; i < apx.nrows(); i++)
    {
        for (size_t j = 0; j < apx.ncols(); ++j)
        {
            for (size_t k = 0; k < apx.ncols(); ++k)
            {
                if (apx(i, j) == ect(i, k) && apx(i,j) != NONE)
                {
                    ++hits;
                    break;
                }
            }
        }
    }
    std::cout << "Recall accuracy: " << (1.0*hits) / (nrows*ncols)
        << " (" << hits << "/" << nrows*ncols << ")\n";
    return (1.0*hits) / (nrows*ncols);
}


void NNDescent::set_parameters(Parms &parms)
{
    metric = parms.metric;
    p_metric = parms.p_metric;
    n_neighbors = parms.n_neighbors;
    n_trees = parms.n_trees;
    leaf_size = parms.leaf_size;
    pruning_degree_multiplier = parms.pruning_degree_multiplier;
    pruning_prob = parms.pruning_prob;
    tree_init = parms.tree_init;
    seed = parms.seed;
    max_candidates = parms.max_candidates;
    n_iters = parms.n_iters;
    delta = parms.delta;
    n_threads = parms.n_threads;
    verbose = parms.verbose;
    algorithm = parms.algorithm;

    if (leaf_size == NONE)
    {
        leaf_size = std::max(10, n_neighbors);
    }
    if (n_trees == NONE)
    {
        n_trees = 5 + (int)std::round(std::pow(data_size, 0.25));

        // Only so many trees are useful
        n_trees = std::min(32, n_trees);
    }
    if (n_trees == 0)
    {
        tree_init = false;
    }
    if (max_candidates == NONE)
    {
        max_candidates = std::min(60, n_neighbors);
    }
    if (n_iters == NONE)
    {
        n_iters = std::max(5, (int)std::round(std::log2(data_size)));
    }
    if (n_threads == NONE || n_threads == 0)
    {
        int n_processors = std::thread::hardware_concurrency();
        n_threads = n_processors == 0 ? 4 : n_processors;
    }
    if
    (
           (metric == "correlation")
        || (metric == "cosine")
        || (metric == "dice")
        || (metric == "dot")
        || (metric == "hamming")
        || (metric == "hellinger")
        || (metric == "jaccard")
    )
    {
        angular_trees = true;
    }
    else
    {
        angular_trees = false;
    }
    if (metric == "dot")
    {
        // Make shure original data cannot be modified.
        data.deep_copy();
        data.normalize();
    }
    seed_state(rng_state, seed);
    this->get_distance_function();
    if (verbose)
    {
        std::cout << *this;
    }
}


NNDescent::NNDescent(Matrix<float> &input_data, Parms &parms)
    : data(input_data)
    , data_size(data.nrows())
    , data_dim(data.ncols())
    , current_graph(input_data.nrows(), parms.n_neighbors, FLOAT_MAX, NEW)
{
    is_sparse = false;
    this->set_parameters(parms);
    this->start();
}


NNDescent::NNDescent(CSRMatrix<float> &input_data, Parms &parms)
    : csr_data(input_data)
    , data_size(csr_data.nrows())
    , data_dim(csr_data.ncols())
    , current_graph(input_data.nrows(), parms.n_neighbors, FLOAT_MAX, NEW)
{
    is_sparse = true;
    this->set_parameters(parms);
    this->start_sparse();
}



void NNDescent::start()
{
    if (algorithm == "bf")
    {
        this->start_brute_force(data);
        return;
    }

    if (tree_init)
    {
        log(
            "Building RP forest with " + std::to_string(n_trees) + " trees",
            verbose
        );

        forest = make_forest(
            data, n_trees, leaf_size, rng_state
        );

        log("Update Graph by  RP forest", verbose);

        Matrix<int> leaf_array = get_leaves_from_forest(forest);
        update_by_leaves(
            data, current_graph, leaf_array, dist.get_fct(), n_threads
            // data, current_graph, leaf_array, *dist_fct, n_threads
        );
    }

    init_random(
        data, current_graph, n_neighbors, dist.get_fct(), rng_state
        // data, current_graph, n_neighbors, *dist_fct, rng_state
    );


    nn_descent(
        data,
        current_graph,
        n_neighbors,
        rng_state,
        max_candidates,
        // *dist_fct,
        dist.get_fct(),
        n_iters,
        delta,
        n_threads,
        verbose
    );

    // Make shure every nodes neighborhod contains the node itself.
    add_zero_node(current_graph);

    current_graph.heapsort();

    correct_distances(
        distance_correction, current_graph.keys, neighbor_distances
    );

    neighbor_indices = current_graph.indices;
}


void NNDescent::start_sparse()
{
    if (algorithm == "bf")
    {
        this->start_brute_force(csr_data);
        return;
    }

    if (tree_init)
    {
        log(
            "Building RP forest with " + std::to_string(n_trees) + " trees",
            verbose
        );

        forest = make_forest(
            csr_data, n_trees, leaf_size, rng_state
        );

        log("Update Graph by  RP forest", verbose);

        Matrix<int> leaf_array = get_leaves_from_forest(forest);
        update_by_leaves(
            csr_data, current_graph, leaf_array, dist.get_fct(), n_threads
            // csr_data, current_graph, leaf_array, *dist_fct, n_threads
        );
    }

    init_random(
        csr_data, current_graph, n_neighbors, dist.get_fct(), rng_state
        // csr_data, current_graph, n_neighbors, *dist_fct, rng_state
    );

    nn_descent(
        csr_data,
        current_graph,
        n_neighbors,
        rng_state,
        max_candidates,
        dist.get_fct(),
        // *dist_fct,
        n_iters,
        delta,
        n_threads,
        verbose
    );

    // Make shure every nodes neighborhod contains the node itself.
    add_zero_node(current_graph);

    current_graph.heapsort();

    correct_distances(
        distance_correction, current_graph.keys, neighbor_distances
    );

    neighbor_indices = current_graph.indices;
}


/*
 * @brief Prune long edges in the graph.
 *
 * This function prunes long edges in the graph, which are edges that are
 * closer to a node's neighbor than to the node itself. It helps to improve
 * the efficiency of the k-nearest neighbor graph query search.
 *
 * @param data The input data matrix.
 * @param graph The current k-nearest neighbor graph.
 * @param rng_state Random number generator state.
 * @param dist The distance metric used for pruning.
 * @param n_threads The number of threads to use for parallelization.
 * @param verbose Flag indicating whether to print verbose output.
 * @param pruning_prob The probability of pruning a long edge (default:
 * 1.0).
 */
template<class MatrixType>
void prune_long_edges
(
    const MatrixType &data,
    HeapList<float> &graph,
    RandomState &rng_state,
    const DistanceFunction &dist,
    int n_threads,
    float pruning_prob
)
{
    #pragma omp parallel for num_threads(n_threads)
    for (size_t i = 0; i < graph.nheaps(); ++i)
    {
        std::vector<int> new_indices;
        std::vector<float> new_keys;
        // First element is node itself and can be pruned.
        for (size_t j = 1; j < graph.nnodes(); ++j)
        {
            int idx = graph.indices(i, j);
            float key = graph.keys(i, j);
            if  (idx == NONE)
            {
                continue;
            }

            bool add_node = true;

            for (size_t k = 0; k < new_indices.size(); ++k)
            {
                int new_idx = new_indices[k];
                float new_key = new_keys[k];
                float d = dist(data, idx, new_idx);
                if (new_key > FLOAT_EPS && d < key)
                {
                    // idx is closer to a node in the neighborhood than
                    // to the central node i, i.e. it is a long edge.
                    if (rand_float(rng_state) < pruning_prob)
                    {
                        add_node = false;
                        break;
                    }


                }
            }
            if (add_node)
            {
                new_indices.push_back(idx);
                new_keys.push_back(key);
            }
        }
        for (size_t j = 0; j < graph.nnodes(); ++j)
        {
            if  (j < new_indices.size())
            {
                graph.indices(i, j) = new_indices[j];
                graph.keys(i, j) = new_keys[j];
            }
            else
            {
                graph.indices(i, j) = NONE;
                graph.keys(i, j) = FLOAT_MAX;
            }
        }
    }
}


void NNDescent::prepare()
{
    // Make a search tree if necessary.
    if (forest.size() == 0)
    {
        if (is_sparse)
        {
            forest = make_forest(
                csr_data, 1, leaf_size, rng_state
            );
        }
        else
        {
            forest = make_forest(
                data, 1, leaf_size, rng_state
            );
        }
    }
    // The trees are very close in their performance, so the first is selected.
    search_tree = forest[0];

    HeapList<float> forward_graph = current_graph;

    if (is_sparse)
    {
        prune_long_edges
        (
            csr_data,
            forward_graph,
            rng_state,
            dist.get_fct(),
            // *dist_fct,
            n_threads,
            pruning_prob
        );
    }
    else
    {
        prune_long_edges
        (
            data,
            forward_graph,
            rng_state,
            dist.get_fct(),
            // *dist_fct,
            n_threads,
            pruning_prob
        );
    }

    if (verbose)
    {
        size_t edges_cnt_before = forward_graph.indices.nrows()
            * forward_graph.indices.ncols();
        size_t edges_cnt_after = forward_graph.indices.non_none_cnt();
        log(
            "Forward graph pruning reduced edges from "
                + std::to_string(edges_cnt_before)
                + " to "
                + std::to_string(edges_cnt_after)
        );
    }

    size_t n_seach_cols = std::round(n_neighbors * pruning_degree_multiplier);
    search_graph = HeapList<float>(data_size, n_seach_cols, FLOAT_MAX);

    for (size_t i = 0; i < forward_graph.nheaps(); ++i)
    {
        for (size_t j = 0; j < forward_graph.nnodes(); ++j)
        {
            int idx = forward_graph.indices(i, j);
            if (idx != NONE)
            {
                float d = forward_graph.keys(i, j);
                search_graph.checked_push(i, idx, d);
                search_graph.checked_push(idx, i, d);
            }
        }
    }
    search_graph.heapsort();

    if (verbose)
    {
        log(
            "Merging pruned graph with its transpose results in "
                + std::to_string(search_graph.indices.non_none_cnt())
                + " edges for the search graph."
        );
    }
}

template<class MatrixType>
void NNDescent::start_brute_force(const MatrixType &mtx_data)
{
    ProgressBar bar(mtx_data.nrows(), verbose);
    #pragma omp parallel for num_threads(n_threads)
    for (size_t idx0 = 0; idx0 < mtx_data.nrows(); ++idx0)
    {
        bar.show();
        for (size_t idx1 = 0; idx1 < mtx_data.nrows(); ++idx1)
        {
            float d = (dist.get_fct())(mtx_data, idx0, idx1);
            // float d = (*dist_fct)(mtx_data, idx0, idx1);
            current_graph.checked_push(idx0, idx1, d);
        }
    }
    current_graph.heapsort();
    neighbor_indices = current_graph.indices;
    correct_distances(
        distance_correction, current_graph.keys, neighbor_distances
    );
}


template<>
Matrix<float>* NNDescent::get_data()
{
    if (is_sparse)
    {
        throw std::runtime_error(
            "The model was trained using a sparse matrix. Applications using "
            "a dense matrix are not supported."
        );
    }
    return &data;
}


template<>
CSRMatrix<float>* NNDescent::get_data()
{
    if (!is_sparse)
    {
        throw std::runtime_error(
            "The model was trained using a dense matrix. Applications using "
            "a sparse matrix are not supported."
        );
    }
    return &csr_data;
}


std::ostream& operator<<(std::ostream &out, const NNDescent &nnd)
{
    out << "NNDescent(\n\t"
        << "data=Matrix<float>(n_rows=" <<  nnd.data.nrows() << ", n_cols="
        <<  nnd.data.ncols() << "),\n\t"
        << "csr_data=CSRMatrix<float>(n_rows=" <<  nnd.csr_data.nrows()
        << ", n_cols=" <<  nnd.csr_data.ncols() << "),\n\t"
        << "metric=" << nnd.metric << ",\n\t"
        << "p_metric=" << nnd.p_metric << ",\n\t"
        << "n_neighbors=" << nnd.n_neighbors << ",\n\t"
        << "n_trees=" << nnd.n_trees << ",\n\t"
        << "leaf_size=" << nnd.leaf_size << ",\n\t"
        << "pruning_degree_multiplier=" << nnd.pruning_degree_multiplier << ",\n\t"
        << "pruning_prob=" << nnd.pruning_prob << ",\n\t"
        << "tree_init=" << nnd.tree_init  << ",\n\t"
        << "seed=" << nnd.seed  << ",\n\t"
        << "max_candidates=" << nnd.max_candidates  << ",\n\t"
        << "n_iters=" << nnd.n_iters  << ",\n\t"
        << "delta=" << nnd.delta  << ",\n\t"
        << "n_threads=" << nnd.n_threads  << ",\n\t"
        << "verbose=" << nnd.verbose  << ",\n\t"
        << "algorithm=" << nnd.algorithm  << ",\n\t"
        << "angular_trees=" << nnd.angular_trees  << ",\n\t"
        << "is_sparse=" << nnd.is_sparse  << ",\n"
        << ")\n";
    return out;
}


/**
 * @brief Get the distance function based on the selected metric.
 *
 * This function retrieves the appropriate distance function based on the
 * selected metric. It assigns the distance function to the variable 'dist'
 * and the distance correction function to the variable 'distance_correction'.
 * The distance function is used to compute the distance between two points,
 * while the distance correction function is used to correct the distances at
 * the end of the calculation if necessary.
 */
void NNDescent::get_distance_function()
{
    using DF = DistanceFunction;
    using DFp = DistanceFunction_p;
    using DF_p = DistanceFunction__p;

    std::map<std::string, DF> DF_map
    {
        { "alternative_cosine", DF(alternative_cosine, sparse_alternative_cosine)},
        { "alternative_dot", DF(alternative_dot, sparse_alternative_dot) },
        { "braycurtis", DF(bray_curtis, sparse_bray_curtis) },
        { "canberra", DF(canberra, sparse_canberra) },
        { "chebyshev", DF(chebyshev, sparse_chebyshev) },
        { "cosine", DF(cosine, sparse_cosine) },
        { "dice", DF(dice, sparse_dice) },
        { "dot", DF(dot, sparse_dot) },
        { "euclidean", DF(squared_euclidean, sparse_squared_euclidean, std::sqrt) },
        { "hamming", DF(hamming, sparse_hamming) },
        { "haversine", DF(haversine, nullptr) },
        { "hellinger", DF(hellinger, sparse_hellinger) },
        { "jaccard", DF(jaccard, sparse_jaccard) },
        { "manhattan", DF(manhattan, sparse_manhattan) },
        { "matching", DF(matching, sparse_matching) },
        { "sokalsneath", DF(sokal_sneath, sparse_sokal_sneath) },
        { "spearmanr", DF(spearmanr, nullptr) },
        { "sqeuclidean", DF(squared_euclidean, sparse_squared_euclidean) },
        { "true_angular", DF(true_angular, sparse_true_angular) },
        { "tsss", DF(tsss, sparse_tsss) },
    };

    std::map<std::string, DFp> DFp_map
    {
        { "circular_kantorovich", DFp(circular_kantorovich, nullptr, p_metric) },
        { "minkowski", DFp(minkowski, sparse_minkowski, p_metric) },
        { "wasserstein_1d", DFp(wasserstein_1d, nullptr, p_metric) },
    };

    std::map<std::string, DF_p> DF_p_map
    {
        { "correlation", DF_p(correlation, sparse_correlation, data_dim) },
        { "jensen_shannon", DF_p(jensen_shannon_divergence,
            sparse_jensen_shannon_divergence, data_dim) },
        { "kulsinski", DF_p(kulsinski, sparse_kulsinski, data_dim) },
        { "rogerstanimoto", DF_p(rogers_tanimoto, sparse_rogers_tanimoto, data_dim) },
        { "russellrao", DF_p(russellrao, sparse_rogers_tanimoto, data_dim) },
        { "sokalmichener", DF_p(sokal_michener, sparse_sokal_michener, data_dim) },
        { "symmetric_kl", DF_p(symmetric_kl_divergence,
            sparse_symmetric_kl_divergence, data_dim) },
        { "yule", DF_p(yule, sparse_yule, data_dim) }
    };

    if (DF_map.count(metric) > 0)
    {
        // _dist_DF = DF_map[metric];
        // dist_fct = &_dist_DF;
        dist = Distance(DF_map[metric]);
    }
    else if (DFp_map.count(metric) > 0)
    {
        // _dist_DFp = DFp_map[metric];
        // dist_fct = &_dist_DFp;
        dist = Distance(DFp_map[metric]);
    }
    else if (DF_p_map.count(metric) > 0)
    {
        // _dist_DF_p = DF_p_map[metric];
        // dist_fct = &_dist_DF_p;
        dist = Distance(DF_p_map[metric]);
    }
    else
    {
        throw std::invalid_argument("Invalid metric");
    }
    distance_correction = dist.get_fct().correction;
    // distance_correction = dist_fct->correction;

    if
    (
        is_sparse &&
        !dist.get_fct().sparse_metric &&
        !dist.get_fct().sparse_metric_p
        // !dist_fct->sparse_metric &&
        // !dist_fct->sparse_metric_p
    )
    {
        throw std::invalid_argument(
            "Sparse variant of '" + std::string(metric) + "' not implemented."
        );

    }
    if (metric == "haversine")
    {
        if (data_dim != 2)
        {
            throw std::invalid_argument(
                "haversine is only defined for 2 dimensional graph_data"
            );
        }
    }
}


} // namespace nndescent
