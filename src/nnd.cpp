/**
 * @file nnd.cpp
 *
 * @brief Implementaton of nearest neighbor descent.
 */


#include <assert.h>
#include <iostream>
#include <thread>
#include <vector>
#include <stdexcept>

#include "nnd.h"
#include "distances.h"

namespace nndescent
{


void throw_exception_if_sparse(std::string metric, bool is_sparse)
{
    if (is_sparse)
    {
        throw std::invalid_argument(
            "Sparse variant of '" + std::string(metric) + "' not implemented."
        );
    }
}


/*
 * @brief Initializes the nearest neighbor graph with random neighbors for
 * missing nodes.
 *
 * @param data The input data matrix.
 * @param current_graph The current nearest neighbor graph to initialize.
 * @param n_neighbors The number of neighbors.
 * @param dist The distance metric used for neighbor selection.
 * @param rng_state The random state used for randomization.
 */
template<class MatrixType, class DistType>
void init_random(
    const MatrixType &data,
    HeapList<float> &current_graph,
    size_t n_neighbors,
    const DistType &dist,
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


/*
 * @brief Adds every node to its own neighborhod.
 */
void add_zero_node(
    HeapList<float> &current_graph
)
{
    for (size_t idx0 = 0; idx0 < current_graph.nheaps(); ++idx0)
    {
        current_graph.checked_push(idx0, idx0, 0.0f, NEW);
    }
}


/*
 * @brief Updates the nearest neighbor graph using leaves constructed from
 * random projection trees.
 *
 * This function updates the nearest neighbor graph by incorporating the
 * information from the leaves constructed from random projection trees.
 *
 * @param data The input data matrix.
 * @param current_graph The current nearest neighbor graph.
 * @param leaf_array The matrix of leaf indices.
 * @param dist The distance function used for nearest neighbor calculations.
 * @param n_threads The number of threads to use for parallelization.
 */
template<class MatrixType, class DistType>
void update_by_leaves(
    const MatrixType &data,
    HeapList<float> &current_graph,
    Matrix<int> &leaf_array,
    const DistType &dist,
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
void sample_candidates(
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
    // Mark sampled nodes in current_graph as 'OLD'.
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
 *
 * @return A vector of vectors of NNUpdate objects representing the nearest
 * neighbor updates.
 */
template<class MatrixType, class DistType>
std::vector<std::vector<NNUpdate>> generate_graph_updates(
    const MatrixType &data,
    HeapList<float> &current_graph,
    HeapList<int> &new_candidate_neighbors,
    HeapList<int> &old_candidate_neighbors,
    const DistType &dist,
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
                for (
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
 *
 * @return The number of updates applied to the graph.
 */
int apply_graph_updates(
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


 /*
  * @brief Performs the NN-descent algorithm for approximate nearest neighbor
  * search.
  *
  * This function applies the NN-descent algorithm to construct an approximate
  * nearest neighbor graph. It iteratively refines the graph by exploring
  * neighbor candidates and updating the graph connections based on the
  * distances between nodes. The algorithm aims to find a graph that represents
  * the nearest neighbor relationships in the data.
  *
  * @tparam MatrixType The type of the input data matrix (e.g. Matrix or
  * CSRMatrix).
  * @tparam DistType The type of the distance metric.
  *
  * @param data The input data matrix.
  * @param current_graph The initial nearest neighbor graph. The resulting
  * nearest neighbor graph will be stored in this variable
  * @param n_neighbors The desired number of neighbors for each node.
  * @param rng_state The random state used for randomization.
  * @param max_candidates The maximum number of candidate neighbors to consider
  * during exploration.
  * @param dist The metric used for distance computation.
  * @param n_iters The number of iterations to perform.
  * @param delta The value controlling the early abort.
  * @param n_threads The number of threads to use for parallelization.
  * @param verbose Flag indicating whether to print progress and diagnostic
  * messages.
  */
template<class MatrixType, class DistType>
void nn_descent(
    const MatrixType &data,
    HeapList<float> &current_graph,
    int n_neighbors,
    RandomState &rng_state,
    int max_candidates,
    const DistType &dist,
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
    if (
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
    if (verbose)
    {
        std::cout << *this;
    }
}


NNDescent::NNDescent(Matrix<float> &train_data, Parms &parms)
    : data(train_data)
    , data_size(data.nrows())
    , data_dim(data.ncols())
    , current_graph(train_data.nrows(), parms.n_neighbors, FLOAT_MAX, NEW)
{
    is_sparse = false;
    this->set_parameters(parms);
    this->set_dist_and_start_nn<Matrix<float>>();
}


NNDescent::NNDescent(CSRMatrix<float> &train_data, Parms &parms)
    : csr_data(train_data)
    , data_size(csr_data.nrows())
    , data_dim(csr_data.ncols())
    , current_graph(train_data.nrows(), parms.n_neighbors, FLOAT_MAX, NEW)
{
    is_sparse = true;
    this->set_parameters(parms);
    this->set_dist_and_start_nn<CSRMatrix<float>>();
}


template<class MatrixType, class DistType>
void NNDescent::run_nn_descent(
    const MatrixType &train_data,
    const DistType &dist
)
{
    if (algorithm == "bf")
    {
        this->start_brute_force(train_data, dist);
        return;
    }

    if (tree_init)
    {
        log(
            "Building RP forest with " + std::to_string(n_trees) + " trees",
            verbose
        );

        forest = make_forest(
            train_data, n_trees, leaf_size, rng_state
        );

        log("Update Graph by  RP forest", verbose);

        Matrix<int> leaf_array = get_leaves_from_forest(forest);
        update_by_leaves(
            train_data, current_graph, leaf_array, dist, n_threads
        );
    }

    init_random(
        train_data, current_graph, n_neighbors, dist, rng_state
    );

    nn_descent(
        train_data,
        current_graph,
        n_neighbors,
        rng_state,
        max_candidates,
        dist,
        n_iters,
        delta,
        n_threads,
        verbose
    );

    // Make shure every nodes neighborhod contains the node itself.
    add_zero_node(current_graph);

    current_graph.heapsort();

    correct_distances(
        dist, current_graph.keys, neighbor_distances
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
 * @param pruning_prob The probability of pruning a long edge.
 */
template<class MatrixType, class DistType>
void prune_long_edges(
    const MatrixType &data,
    HeapList<float> &graph,
    RandomState &rng_state,
    const DistType &dist,
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


template<class DistType>
void NNDescent::prepare(const DistType &dist)
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
        prune_long_edges(
            csr_data,
            forward_graph,
            rng_state,
            dist,
            n_threads,
            pruning_prob
        );
    }
    else
    {
        prune_long_edges(
            data,
            forward_graph,
            rng_state,
            dist,
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


template<class MatrixType, class DistType>
void NNDescent::start_brute_force(
    const MatrixType &train_data, const DistType &dist
)
{
    ProgressBar bar(train_data.nrows(), verbose);
    #pragma omp parallel for num_threads(n_threads)
    for (size_t idx0 = 0; idx0 < train_data.nrows(); ++idx0)
    {
        bar.show();
        for (size_t idx1 = 0; idx1 < train_data.nrows(); ++idx1)
        {
            float d = dist(train_data, idx0, idx1);
            current_graph.checked_push(idx0, idx1, d);
        }
    }
    current_graph.heapsort();
    neighbor_indices = current_graph.indices;
    correct_distances(
        dist, current_graph.keys, neighbor_distances
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


} // namespace nndescent
