/*
 * Dong, Wei, Charikar Moses, and Kai Li. "Efficient k-nearest neighbor graph
 * construction for generic similarity measures." Proceedings of the 20th
 * international conference on World wide web. 2011.
 *
 * https://dl.acm.org/doi/pdf/10.1145/1963405.1963487
 * https://www.cs.princeton.edu/cass/papers/www11.pdf
 */

#include <assert.h>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>
#include <stdexcept>

#include "dtypes.h"
#include "nnd.h"
#include "utils.h"
#include "rp_trees.h"
#include "distances.h"

// Global timer for debugging
Timer timer;

// Initializes heaps by choosing nodes randomly.
void init_random
(
    const Matrix<float> &data,
    HeapList<float> &current_graph,
    size_t n_neighbors,
    const Metric &dist,
    RandomState &rng_state
)
{
    // TODO assert must not be triggered
    assert(n_neighbors <= current_graph.nheaps());

    // TODO parallel
    for (size_t idx0 = 0; idx0 < current_graph.nheaps(); ++idx0)
    {
        int missing = n_neighbors - current_graph.size(idx0);
        // Sample nodes
        for (int j = 0; j < missing; ++j)
        {
            int idx1 = rand_int(rng_state) % current_graph.nheaps();
            float d = dist(data.begin(idx0), data.end(idx0), data.begin(idx1));
            current_graph.checked_push(idx0, idx1, d, FALSE);
        }
    }
}



void update_by_leaves(
    const Matrix<float> &data,
    HeapList<float> &current_graph,
    Matrix<int> &leaf_array,
    const Metric &dist,
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
                // std::cout << "i =" << i << " j=" << j << " leafsize=" << leaf_size
                    // << " blockst=" << block_start << " blocke=" << block_end << " blos="
                    // << block_size << " nleav=" << n_leaves  << " idx0=" << idx0
                    // << "\n";
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
                    float d = dist(
                        data.begin(idx0), data.end(idx0), data.begin(idx1)
                    );
                    if (
                        (d < current_graph.max(idx0)) ||
                        (d < current_graph.max(idx1))
                    )
                    {
                        NNUpdate update = {idx0, idx1, d};
                        updates[thread].push_back(update);
                        // std::cout << update <<" i="<<i<<" j="<<j<<" k="<<k<<"\n";
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
                    current_graph.checked_push(idx0, idx1, d, FALSE);
                }
                if (idx1 % n_threads == thread)
                {
                    current_graph.checked_push(idx1, idx0, d, FALSE);
                }
            }
        }
    }
}

void update_by_rp_forest(
    const Matrix<float> &data,
    HeapList<float> &current_graph,
    std::vector<IntMatrix> &forest,
    const Metric &dist
)
{
    for (const auto& tree : forest)
    {
        // std::cout << "tree=" << tree << "\n" << "graph=" << current_graph;
        for (const auto& leaf : tree)
        {
            for (const int& idx0 : leaf)
            {
                for (const int& idx1 : leaf)
                {
                    if (idx0 >= idx1)
                    {
                        continue;
                    }
                    float d = dist(
                        data.begin(idx0), data.end(idx0), data.begin(idx1)
                    );
                    current_graph.checked_push(idx0, idx1, d, FALSE);
                    current_graph.checked_push(idx1, idx0, d, FALSE);

                }
            }
        }
    }
}


// Build a heap of candidate neighbors for nearest neighbor descent. For
// each vertex the candidate neighbors are any current neighbors, and any
// vertices that have the vertex as one of their nearest neighbors.
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

                int priority = rand_int(local_rng_state);

                if (flag == FALSE)
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
    // Mark sampled nodes in current_graph as flag.
    for (size_t idx0 = 0; idx0 < current_graph.nheaps(); ++idx0)
    {
        for (size_t j = 0; j < current_graph.nnodes(); ++j)
        {
            int idx1 = current_graph.indices(idx0, j);
            for (size_t k = 0; k < new_candidates.nnodes(); ++k)
            {
                if (new_candidates.indices(idx0, k) == idx1)
                {
                    current_graph.flags(idx0, j) = TRUE;
                    break;
                }
            }
        }
    }
}

std::vector<std::vector<NNUpdate>> generate_graph_updates
(
    const Matrix<float> &data,
    HeapList<float> &current_graph,
    HeapList<int> &new_candidate_neighbors,
    HeapList<int> &old_candidate_neighbors,
    const Metric &dist,
    int n_threads,
    int verbose
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
                for (size_t k = j + 1; k < new_candidate_neighbors.nnodes(); ++k)
                {
                    int idx1 = new_candidate_neighbors.indices(i, k);
                    if (idx1 == NONE)
                    {
                        continue;
                    }
                    float d = dist(
                        data.begin(idx0), data.end(idx0), data.begin(idx1)
                    );
                    if (
                        (d < current_graph.max(idx0)) ||
                        (d < current_graph.max(idx1))
                    )
                    {
                        NNUpdate update = {idx0, idx1, d};
                        updates[thread].push_back(update);
                        // std::cout << "new " << update <<" i="<<i<<" j="<<j<<" k="<<k<<"\n";
                    }

                }
                for (size_t k = 0; k < old_candidate_neighbors.nnodes(); ++k)
                {
                    int idx1 = old_candidate_neighbors.indices(i, k);
                    if (idx1 == NONE)
                    {
                        continue;
                    }
                    float d = dist(
                        data.begin(idx0), data.end(idx0), data.begin(idx1)
                    );
                    if (
                        (d < current_graph.max(idx0)) ||
                        (d < current_graph.max(idx1))
                    )
                    {
                        NNUpdate update = {idx0, idx1, d};
                        updates[thread].push_back(update);
                        // std::cout << "old " << update <<" i="<<i<<" j="<<j<<" k="<<k<<"\n";
                    }
                }
            }
            // log(
                // "\t\tGenerate updates " + std::to_string(i + 1) + "/"
                    // + std::to_string(new_candidate_neighbors.nheaps()),
                // (verbose && (i % (new_candidate_neighbors.nheaps() / 4) == 0))
            // );
        }
    }

    return updates;
}


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
                    n_changes += current_graph.checked_push(idx0, idx1, d, TRUE);
                }
                if (idx1 % n_threads == thread)
                {
                    n_changes += current_graph.checked_push(idx1, idx0, d, TRUE);
                }
            }
        }
    }

    return n_changes;
}


void nn_descent
(
    const Matrix<float> &data,
    HeapList<float> &current_graph,
    int n_neighbors,
    RandomState rng_state,
    int max_candidates,
    const Metric &dist,
    int n_iters,
    float delta,
    int n_threads,
    bool verbose
)
{
    timer.start();
    assert(current_graph.nheaps() == data.nrows());

    log("NN descent for " + std::to_string(n_iters) + " iterations", verbose);

    timer.stop("nn descent: init");

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

        timer.start();

        sample_candidates(
            current_graph,
            new_candidates,
            old_candidates,
            rng_state,
            n_threads
        );
        // std::cout << "NEW=" << new_candidates << "\n";
        // std::cout << "OLD=" << old_candidates << "\n";
        timer.stop("sample_candidates");
        // std::cout << "current_graph with flags=" << current_graph;

        std::vector<std::vector<NNUpdate>> updates = generate_graph_updates(
            data,
            current_graph,
            new_candidates,
            old_candidates,
            dist,
            n_threads,
            verbose
        );
        // std::cout << "updates=" << updates;
        timer.stop("generate_graph_updates");

        int cnt = apply_graph_updates(
            current_graph,
            updates,
            n_threads
        );
        // std::cout << "current_graph updated=" << current_graph;
        log("\t\t" + std::to_string(cnt) + " updates applied", verbose);
        timer.stop("apply apply_graph_updates updates");

        if (cnt < delta * data.nrows() * n_neighbors)
        {
            log(
                "Stopping threshold met -- exiting after "
                    + std::to_string(iter) + " iterations",
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
                if (apx(i, j) == ect(i, k))
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
    n_neighbors = parms.n_neighbors;
    n_trees = parms.n_trees;
    leaf_size = parms.leaf_size;
    pruning_degree_multiplier = parms.pruning_degree_multiplier;
    diversify_prob = parms.diversify_prob;
    tree_init = parms.tree_init;
    seed = parms.seed;
    low_memory = parms.low_memory;
    max_candidates = parms.max_candidates;
    n_iters = parms.n_iters;
    delta = parms.delta;
    n_threads = parms.n_threads;
    compressed = parms.compressed;
    parallel_batch_queries = parms.parallel_batch_queries;
    verbose = parms.verbose;
    algorithm = parms.algorithm;

    if (leaf_size == NONE)
    {
        leaf_size = std::max(10, n_neighbors);
    }
    if (n_trees == NONE)
    {
        n_trees = 5 + (int)std::round(std::pow(data.nrows(), 0.25));
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
        n_iters = std::max(5, (int)std::round(std::log2(data.nrows())));
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
        _angular_trees = true;
    }
    else
    {
        _angular_trees = false;
    }
    seed_state(rng_state, seed);
    this->get_distance_function();
}

NNDescent::NNDescent(Matrix<float> &input_data, Parms &parms)
    : data(input_data)
    , current_graph(input_data.nrows(), parms.n_neighbors, FLOAT_MAX, FALSE)
{
    timer.start();
    this->set_parameters(parms);
    std::cout << *this;
    this->start();
}

NNDescent::NNDescent(Matrix<float> &input_data, int n_neighbors)
    : data(input_data)
    , current_graph(input_data.nrows(), n_neighbors, FLOAT_MAX, FALSE)
{
    timer.start();
}

void NNDescent::start()
{
    if (algorithm == "bf")
    {
        this->brute_force();
        return;
    }

    timer.stop("Constructor");

    if (tree_init)
    {
        log(
            "Building RP forest with " + std::to_string(n_trees) + " trees",
            verbose
        );
        std::vector<IntMatrix> forest = make_forest(
            data, n_trees, leaf_size, rng_state
        );
        // std::cout << "forest=" << forest;
        timer.stop("make forest");

        log("Update Graph by  RP forest", verbose);

        Matrix<int> leaf_array = get_leaves_from_forest(forest, leaf_size);
        // std::cout << "leaf_array=" << leaf_array;
        timer.stop("make leaf array");
        update_by_leaves(data, current_graph, leaf_array, dist, n_threads);
        timer.stop("update by leaf array");


        // update_by_rp_forest(data, current_graph, forest, dist);
        // std::cout << current_graph;
        timer.stop("update graph by rp-tree forest");
    }

    init_random(data, current_graph, n_neighbors, dist, rng_state);
    // std::cout << "curent graph 0=" << current_graph;
    timer.stop("random init neighbours");

    nn_descent(
        data,
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
    timer.stop("nn_descent");
    current_graph.heapsort();
    timer.stop("heapsort");
    neighbor_graph = current_graph.indices;
    // std::cout << "end current graph=" << current_graph;
    // std::cout << neighbor_graph;
    // std::cout << *this;
}

Matrix<int> NNDescent::brute_force()
{
    for (size_t idx0 = 0; idx0 < data.nrows(); ++idx0)
    {
        for (size_t idx1 = 0; idx1 < data.nrows(); ++idx1)
        {
            float d = dist(
                data.begin(idx0), data.end(idx0), data.begin(idx1)
            );
            current_graph.checked_push(idx0, idx1, d);
        }
    }
    current_graph.heapsort();
    neighbor_graph = current_graph.indices;
    return neighbor_graph;
}

std::ostream& operator<<(std::ostream &out, const NNDescent &nnd)
{
    out << "NNDescent(\n\t"
        << "data=Matrix<float>(n_rows= " <<  nnd.data.nrows() << ", n_cols="
        <<  nnd.data.ncols() << "),\n\t"
        << "metric=" << nnd.metric << ",\n\t"
        << "n_neighbors=" << nnd.n_neighbors << ",\n\t"
        << "n_trees=" << nnd.n_trees << ",\n\t"
        << "leaf_size=" << nnd.leaf_size << ",\n\t"
        << "pruning_degree_multiplier=" << nnd.pruning_degree_multiplier << ",\n\t"
        << "diversify_prob=" << nnd.diversify_prob << ",\n\t"
        << "tree_init=" << nnd.tree_init  << ",\n\t"
        << "seed=" << nnd.seed  << ",\n\t"
        << "low_memory=" << nnd.low_memory  << ",\n\t"
        << "max_candidates=" << nnd.max_candidates  << ",\n\t"
        << "n_iters=" << nnd.n_iters  << ",\n\t"
        << "delta=" << nnd.delta  << ",\n\t"
        << "n_threads=" << nnd.n_threads  << ",\n\t"
        << "compressed=" << nnd.compressed  << ",\n\t"
        << "parallel_batch_queries=" << nnd.parallel_batch_queries  << ",\n\t"
        << "verbose=" << nnd.verbose  << ",\n\t"
        << "algorithm=" << nnd.algorithm  << ",\n"
        << "\n\t"
        << "_angular_trees=" << nnd._angular_trees  << ",\n"
        << ")\n";
    return out;
}

void NNDescent::get_distance_function()
{

    if (metric == "euclidean")
    {
        dist = euclidean<It, It>;
    }
    else if (metric == "sqeuclidean")
    {
        dist = squared_euclidean<It, It>;
    }
    else if (metric == "manhattan")
    {
        dist = manhattan<It, It>;
    }
    else if (metric == "chebyshev")
    {
        dist = chebyshev<It, It>;
    }
    else if (metric == "minkowski")
    {
        // float p = 2.0f;
    }
    else if (metric == "standardised_euclidean")
    {
        // dist = standardised_euclidean<It, It>;
    }
    else if (metric == "weighted_minkowski")
    {
    }
    else if (metric == "mahalanobis")
    {
        // dist = mahalanobis<It, It>;
    }
    else if (metric == "canberra")
    {
        dist = canberra<It, It>;
    }
    else if (metric == "cosine")
    {
        dist = cosine<It, It>;
    }
    else if (metric == "dot")
    {
        dist = dot<It, It>;
    }
    else if (metric == "correlation")
    {
        dist = correlation<It, It>;
    }
    else if (metric == "haversine")
    {
        if (data.ncols() != 2)
        {
            throw std::invalid_argument(
                "haversine is only defined for 2 dimensional graph_data"
            );
        }
    }
    else if (metric == "braycurtis")
    {
        dist = bray_curtis<It, It>;
    }
    else if (metric == "spearmanr")
    {
        dist = spearmanr<It, It>;
    }
    else if (metric == "tsss")
    {
        dist = tsss<It, It>;
    }
    else if (metric == "true_angular")
    {
        dist = true_angular<It, It>;
    }
    else if (metric == "hellinger")
    {
        dist = hellinger<It, It>;
    }
    else if (metric == "wasserstein_1d")
    {
        // dist = wasserstein_1d<It, It>;
    }
    else if (metric == "circular_kantorovich")
    {
        // dist = circular_kantorovich<It, It>;
    }
    else if (metric == "jensen_shannon")
    {
        dist = jensen_shannon_divergence<It, It>;
    }
    else if (metric == "symmetric_kl")
    {
        dist = symmetric_kl_divergence<It, It>;
    }
    else if (metric == "hamming")
    {
        dist = hamming<It, It>;
    }
    else if (metric == "jaccard")
    {
        dist = alternative_jaccard<It, It>;
    }
    else if (metric == "dice")
    {
        dist = dice<It, It>;
    }
    else if (metric == "matching")
    {
        dist = matching<It, It>;
    }
    else if (metric == "kulsinski")
    {
        dist = kulsinski<It, It>;
    }
    else if (metric == "rogerstanimoto")
    {
        dist = rogers_tanimoto<It, It>;
    }
    else if (metric == "russellrao")
    {
        dist = russellrao<It, It>;
    }
    else if (metric == "sokalsneath")
    {
        dist = sokal_sneath<It, It>;
    }
    else if (metric == "sokalmichener")
    {
        dist = sokal_michener<It, It>;
    }
    else if (metric == "yule")
    {
        dist = yule<It, It>;
    }
    else
    {
        throw std::invalid_argument("invalid metric");
    }
}

