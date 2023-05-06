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
            float d = dist(data, idx0, idx1);
            current_graph.checked_push(idx0, idx1, d, FALSE);
        }
    }
}


void update_graph_by_rp_forest(
    const Matrix<float> &data,
    HeapList<float> &current_graph,
    std::vector<IntMatrix> &forest
)
{
    for (IntMatrix tree : forest)
    {
        // std::cout << "tree=" << tree << "\n" << "graph=" << current_graph;
        for (IntVec leaf : tree)
        {
            for (int idx0 : leaf)
            {
                for (int idx1 : leaf)
                {
                    if (idx0 >= idx1)
                    {
                        continue;
                    }
                    float d = dist(data, idx0, idx1);
                    // float d = squared_euclidean(
                        // data.begin(idx0),
                        // data.end(idx0),
                        // data.begin(idx1)
                    // );
                    current_graph.checked_push(idx0, idx1, d, FALSE);
                    current_graph.checked_push(idx1, idx0, d, FALSE);

                }
            }
        }
    }
}


void log(std::string text, bool verbose=true)
{
    if (!verbose)
    {
        return;
    }
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::cout << std::put_time(localtime(&time), "%F %T ") << text << "\n";
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

                if (idx1 < 0)
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


typedef struct
{
    int idx0;
    int idx1;
    float dist;
} NNUpdate;

std::ostream& operator<<(std::ostream &out, NNUpdate &update)
{
    out << "(idx0=" << update.idx0
        << ", idx1=" << update.idx1
        << ", dist=" << update.dist
        << ")";
    return out;
}

std::ostream& operator<<(std::ostream &out, std::vector<NNUpdate> &updates)
{
    out << "[";
    for (size_t i = 0; i < updates.size(); ++i)
    {
        if (i > 0)
        {
            out << " ";
        }
        out << updates[i];
        if (i + 1 != updates.size())
        {
            out << ",\n";
        }
    }
    out << "]\n";
    return out;
}


std::vector<NNUpdate> generate_graph_updates
(
    const Matrix<float> &data,
    HeapList<float> &current_graph,
    HeapList<int> &new_candidate_neighbors,
    HeapList<int> &old_candidate_neighbors,
    int n_threads,
    int verbose
)
{
    assert(data.nrows() == new_candidate_neighbors.nheaps());
    assert(data.nrows() == old_candidate_neighbors.nheaps());
    std::vector<NNUpdate> updates;
    // #pragma omp parallel for num_threads(4)
    for (size_t i = 0; i < new_candidate_neighbors.nheaps(); ++i)
    {
        for (size_t j = 0; j < new_candidate_neighbors.nnodes(); ++j)
        {
            int idx0 = new_candidate_neighbors.indices(i, j);
            if (idx0 < 0)
            {
                continue;
            }
            for (size_t k = j + 1; k < new_candidate_neighbors.nnodes(); ++k)
            {
                int idx1 = new_candidate_neighbors.indices(i, k);
                if (idx1 < 0)
                {
                    continue;
                }
                float d = dist(data, idx0, idx1);
                // std::cout << "new idx0=" << idx0 << " idx1="<< idx1 << " d="
                    // << d << " max0=" << current_graph.max(idx0)<< " max1="
                    // << current_graph.max(idx1) << " i="<<i<<" j="<<j<<" k="<<k<<"\n";
                if (
                    (d < current_graph.max(idx0)) ||
                    (d < current_graph.max(idx1))
                )
                {
                    NNUpdate update = {idx0, idx1, d};
                    updates.push_back(update);
                    // std::cout << "new " << update <<" i="<<i<<" j="<<j<<" k="<<k<<"\n";
                }

            }
            for (size_t k = 0; k < old_candidate_neighbors.nnodes(); ++k)
            {
                int idx1 = old_candidate_neighbors.indices(i, k);
                if (idx1 < 0)
                {
                    continue;
                }
                float d = dist(data, idx0, idx1);
                // std::cout << "new idx0=" << idx0 << " idx1="<< idx1 << " d="
                    // << d << " max0=" << current_graph.max(idx0)<< " max1="
                    // << current_graph.max(idx1) << " i="<<i<<" j="<<j<<" k="<<k<<"\n";
                if (
                    (d < current_graph.max(idx0)) ||
                    (d < current_graph.max(idx1))
                )
                {
                    NNUpdate update = {idx0, idx1, d};
                    updates.push_back(update);
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

    return updates;
}


int apply_graph_updates
(
    HeapList<float> &current_graph,
    std::vector<NNUpdate> &updates,
    int n_threads
)
{
    int n_changes = 0;

    #pragma omp parallel for
    for (int thread = 0; thread < n_threads; ++thread)
    {
        for (NNUpdate update : updates)
        {
            int idx0 = update.idx0;
            int idx1 = update.idx1;
            assert(idx0 >= 0);
            assert(idx1 >= 0);
            float d = update.dist;
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

    return n_changes;
}



// TODO do parallel
void nn_descent
(
    const Matrix<float> &data,
    HeapList<float> &current_graph,
    int n_neighbors,
    RandomState rng_state,
    int max_candidates,
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

        std::vector<NNUpdate> updates = generate_graph_updates(
            data,
            current_graph,
            new_candidates,
            old_candidates,
            n_threads,
            verbose
        );
        // std::cout << "updates=" << updates;
        timer.stop("generate graph updates");

        int cnt = apply_graph_updates(
            current_graph,
            updates,
            n_threads
        );
        // std::cout << "current_graph updated=" << current_graph;
        log("\t\t" + std::to_string(cnt) + " updates applied", verbose);
        timer.stop("apply graph updates");

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


NNDescent::NNDescent(Parms parms)
    : data(parms.data)
    , metric(parms.metric)
    , n_neighbors(parms.n_neighbors)
    , n_trees(parms.n_trees)
    , leaf_size(parms.leaf_size)
    , pruning_degree_multiplier(parms.pruning_degree_multiplier)
    , diversify_prob(parms.diversify_prob)
    , tree_init(parms.tree_init)
    , seed(parms.seed)
    , low_memory(parms.low_memory)
    , max_candidates(parms.max_candidates)
    , n_iters(parms.n_iters)
    , delta(parms.delta)
    , n_threads(parms.n_threads)
    , compressed(parms.compressed)
    , parallel_batch_queries(parms.parallel_batch_queries)
    , verbose(parms.verbose)
    , algorithm(parms.algorithm)
    , current_graph(parms.data.nrows(), parms.n_neighbors, MAX_FLOAT, FALSE)
{
    timer.start();

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
    seed_state(rng_state, seed);

    std::cout << *this;

    if (algorithm == "bf")
    {
        this->brute_force();
        return;
    }
    if (algorithm == "nnd")
    {
        this->start();
    }
}


void NNDescent::start()
{
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
        // std::cout << forest;
        timer.stop("make forest");

        log("Update Graph by  RP forest", verbose);

        update_graph_by_rp_forest(data, current_graph, forest);
        // std::cout << current_graph;
        timer.stop("update graph by rp-tree forest");
    }

    init_random(data, current_graph, n_neighbors, rng_state);
    // std::cout << "curent graph 0=" << current_graph;
    timer.stop("random init neighbours");

    nn_descent(
        data,
        current_graph,
        n_neighbors,
        rng_state,
        max_candidates,
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
    for (size_t i = 0; i < data.nrows(); ++i)
    {
        for (size_t j = 0; j < data.nrows(); ++j)
        {
            float d = dist(data, i, j);
            current_graph.checked_push(i, j, d);
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
        << ")\n";
    return out;
}

void NNDescent::get_distance_function()
{
    using It = std::vector<float>::const_iterator;
    std::function<float(It, It, It)> distance_function;

    if (metric == "euclidean")
    {
        distance_function = euclidean<It,It>;
    }
    else if (metric == "l2")
    {
    }
    else if (metric == "sqeuclidean")
    {
        distance_function = squared_euclidean<It,It>;
    }
    else if (metric == "manhattan")
    {
        distance_function = manhattan<It,It>;
    }
    else if (metric == "taxicab")
    {
    }
    else if (metric == "l1")
    {
    }
    else if (metric == "chebyshev")
    {
        distance_function = chebyshev<It,It>;
    }
    else if (metric == "linfinity")
    {
    }
    else if (metric == "linfty")
    {
    }
    else if (metric == "linf")
    {
    }
    else if (metric == "minkowski")
    {
    }
    else if (metric == "seuclidean")
    {
    }
    else if (metric == "standardised_euclidean")
    {
    }
    else if (metric == "wminkowski")
    {
    }
    else if (metric == "weighted_minkowski")
    {
    }
    else if (metric == "mahalanobis")
    {
    }
    else if (metric == "canberra")
    {
        distance_function = canberra<It,It>;
    }
    else if (metric == "cosine")
    {
    }
    else if (metric == "dot")
    {
    }
    else if (metric == "correlation")
    {
    }
    else if (metric == "haversine")
    {
    }
    else if (metric == "braycurtis")
    {
        distance_function = bray_curtis<It,It>;
    }
    else if (metric == "spearmanr")
    {
    }
    else if (metric == "tsss")
    {
    }
    else if (metric == "true_angular")
    {
    }
    else if (metric == "hellinger")
    {
    }
    else if (metric == "kantorovich")
    {
    }
    else if (metric == "wasserstein")
    {
    }
    else if (metric == "wasserstein_1d")
    {
    }
    else if (metric == "wasserstein-1d")
    {
    }
    else if (metric == "kantorovich-1d")
    {
    }
    else if (metric == "kantorovich_1d")
    {
    }
    else if (metric == "circular_kantorovich")
    {
    }
    else if (metric == "circular_wasserstein")
    {
    }
    else if (metric == "sinkhorn")
    {
    }
    else if (metric == "jensen-shannon")
    {
    }
    else if (metric == "jensen_shannon")
    {
    }
    else if (metric == "symmetric-kl")
    {
    }
    else if (metric == "symmetric_kl")
    {
    }
    else if (metric == "symmetric_kullback_liebler")
    {
    }
    else if (metric == "hamming")
    {
        distance_function = hamming<It,It>;
    }
    else if (metric == "jaccard")
    {
        distance_function = jaccard<It,It>;
    }
    else if (metric == "dice")
    {
    }
    else if (metric == "matching")
    {
    }
    else if (metric == "kulsinski")
    {
    }
    else if (metric == "rogerstanimoto")
    {
    }
    else if (metric == "russellrao")
    {
    }
    else if (metric == "sokalsneath")
    {
    }
    else if (metric == "sokalmichener")
    {
    }
    else if (metric == "yule")
    {
    }
}

