/*
 * Dong, Wei, Charikar Moses, and Kai Li. "Efficient k-nearest neighbor graph
 * construction for generic similarity measures." Proceedings of the 20th
 * international conference on World wide web. 2011.
 *
 * https://dl.acm.org/doi/pdf/10.1145/1963405.1963487
 * https://www.cs.princeton.edu/cass/papers/www11.pdf
 */

#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <chrono>
#include <assert.h>
#include <ctime>
#include <iomanip>
#include <experimental/algorithm>

#include "dtypes.h"
#include "nnd.h"
#include "utils.h"

// Global timer for debugging
Timer timer;
const double MAX_DOUBLE = std::numeric_limits<double>::max();

inline double euclid_sqr
(
    const std::vector<double> &v0,
    const std::vector<double> &v1
)
{
    double sum = 0.0;
    for (size_t i = 0; i < v0.size(); ++i)
    {
        sum += (v0[i] - v1[i])*(v0[i] - v1[i]);
    }
    return sum;
}

inline double dist
(
    const std::vector<double> &v0,
    const std::vector<double> &v1
)
{
    return euclid_sqr(v0, v1);
}

// Calculates reverse nearest neighbours.
void reverse_neighbours(std::vector<IntVec> &dst, std::vector<IntVec> &src)
{
    for (size_t u = 0; u < src.size(); u++)
    {
        for (int v : src[u])
        {
            // std::cout << "u=" << u << " v=" << v << " dst[v]=";
            // print(dst[v]);
            dst[v].push_back(u);
        }
    }
}

// Initializes heaps by choosing nodes randomly.
std::vector<NNHeap> empty_graph
(
    unsigned int n_heaps,
    unsigned int n_nodes
)
{
    NNNode nullnode = {
        .idx=-1,
        .key=MAX_DOUBLE,
        .visited=false
    };
    NNHeap nullheap(n_nodes, nullnode);
    std::vector<NNHeap> heaplist(n_heaps, nullheap);
    return heaplist;
}

// Initializes heaps by choosing nodes randomly.
void init_random
(
    const SlowMatrix &data,
    std::vector<NNHeap> &current_graph,
    size_t k,
    RandomState &rng_state
)
{
    // std::cout << "\n\nPrint heaps:\n";
    // TODO assert must not be triggered
    assert(k < current_graph.size());

    for (size_t i = 0; i < current_graph.size(); ++i)
    {
        int missing = k - current_graph[i].valid_idx_size();
        // Sample nodes
        for (int j = 0; j < missing; ++j)
        {
            int rand_idx = rand_int(rng_state) % current_graph.size();
            double d = dist(data[i], data[rand_idx]);
            NNNode node = {
                .idx=rand_idx,
                .key=d,
                .visited=false
            };
            current_graph[i].update_max(node);
        }
        // std::cout << "\nIndex: "<< i << "\n";
    }
}

// Improves nearest neighbors heaps by a single random projection tree.
void update_nn_graph_by_rp_tree(
    const SlowMatrix &data, std::vector<NNHeap> &current_graph, unsigned int leaf_size
)
{
    // timer.start();
    RandomState rng_state;
    IntMatrix rp_tree = make_rp_tree(data, leaf_size, rng_state);
    // timer.stop("make_rp_tree");
    // std::cout << "\nRPTree="; print(rp_tree);
    for (IntVec leaf : rp_tree)
    {
        for (int p0 : leaf)
        {
            for (int p1 : leaf)
            {
                if (p0 >= p1)
                {
                    continue;
                }
                double l = dist(data[p0], data[p1]);
                NNNode node0 = {.idx=p0, .key=l, .visited=false};
                NNNode node1 = {.idx=p1, .key=l, .visited=false};
                current_graph[p0].update_max(node1);
                current_graph[p1].update_max(node0);
            }
        }
    }
    // timer.stop("heap updates");
}

// Improves nearest neighbors heaps by multiple random projection trees.
void _update_nn_graph_by_rp_forest(
    const SlowMatrix &data,
    std::vector<NNHeap> &current_graph,
    unsigned int forest_size,
    unsigned int leaf_size
)
{
    for (size_t i = 0; i < forest_size; ++i)
    {
        update_nn_graph_by_rp_tree(data, current_graph, leaf_size);
    }
}

void update_nn_graph_by_rp_forest(
    const SlowMatrix &data,
    std::vector<NNHeap> &current_graph,
    std::vector<IntMatrix> &forest
)
{
    // std::cout << "forestsize=" << forest.size() << "\n";
    for (IntMatrix tree : forest)
    {
        // std::cout << "\ttreesize=" << tree.size() << "\n";
        for (IntVec leaf : tree)
        {
            // std::cout << "\t\tleafsize=" << leaf.size() << "\n";
            for (int idx0 : leaf)
            {
                for (int idx1 : leaf)
                {
                    if (idx0 >= idx1)
                    {
                        continue;
                    }
                    double l = dist(data[idx0], data[idx1]);
                    NNNode node0 = {.idx=idx0, .key=l, .visited=false};
                    NNNode node1 = {.idx=idx1, .key=l, .visited=false};
                    current_graph[idx0].update_max(node1);
                    current_graph[idx1].update_max(node0);
                }
            }
        }
    }
}


void sample_neighbors(
    IntVec &dst,
    NNHeap &src,
    IntVec &indices,
    int sample_size
)
{
    IntVec samples;

    // Sample without replacement
    // std::sample(
    std::experimental::sample(
        indices.begin(),
        indices.end(),
        std::back_inserter(samples),
        sample_size,
        RandNum::mersenne
    );

    // std::cout << "popsize=" << indices.size()
              // << " smpsize=" << sample_size
              // << "\nsamples=";
    // print(samples);

    for (size_t i = 0; i < samples.size(); ++i)
    {
        // print(indices);
        // print(samples);
        // std::cout << "sample_size=" << sample_size << ",";
        // std::cout << "XXXXXXXXXX\n\n";

        int index = samples[i];
        NNNode node = src[index];
        dst.push_back(node.idx);
        node.visited = true;
    }
    // print(src);
    // print(dst);
}

void set_new_and_old(
    IntVec &idx_new_nodes,
    IntVec &idx_old_nodes,
    NNHeap &all_nodes,
    int k_part
)
{
    IntVec indices_new;
    for (size_t i = 0; i < all_nodes.size(); i++)
    {
        NNNode node = all_nodes[i];
        if (!node.visited)
        {
            indices_new.push_back(i);
        }
        else
        {
            idx_old_nodes.push_back(node.idx);
        }
    }

    // print(idx_new_nodes);
    // print(idx_old_nodes);
    // print(all_nodes);

    sample_neighbors(
        idx_new_nodes,
        all_nodes,
        indices_new,
        k_part
    );
}

int local_join(
    const SlowMatrix &data,
    IntVec &idx_new_nodes,
    IntVec &idx_old_nodes,
    IntVec &idx_new_nodes_r,
    IntVec &idx_old_nodes_r,
    std::vector<NNHeap> &current_graph,
    int k_part
)
{
    IntVec buf_new;
    IntVec buf_old;

    // // Sample without replacement
    // std::sample(
    std::experimental::sample(
        idx_new_nodes_r.begin(),
        idx_new_nodes_r.end(),
        std::back_inserter(buf_new),
        k_part,
        RandNum::mersenne
    );
    // std::sample(
    std::experimental::sample(
        idx_old_nodes_r.begin(),
        idx_old_nodes_r.end(),
        std::back_inserter(buf_old),
        k_part,
        RandNum::mersenne
    );

    // buf_new = idx_new_nodes_r;
    // buf_old = idx_old_nodes_r;
    // std::cout << "size new=" << buf_new.size() << "\told=" << buf_old.size() << "\n";

    // Extend vectors by buf
    idx_new_nodes.insert(idx_new_nodes.end(), buf_new.begin(), buf_new.end());
    idx_old_nodes.insert(idx_old_nodes.end(), buf_old.begin(), buf_old.end());

    int cnt = 0;

    for (size_t i = 0; i < idx_new_nodes.size(); ++i)
    {
        int idx0 = idx_new_nodes[i];
        for (size_t j = 0; j < idx_new_nodes.size(); ++j)
        {
            int idx1 = idx_new_nodes[j];

            if (idx0 >= idx1)
            {
                continue;
            }
            double l = dist(data[idx0], data[idx1]);
            NNNode u0 = {.idx=idx0, .key=l, .visited=true};
            NNNode u1 = {.idx=idx1, .key=l, .visited=true};

            cnt += current_graph[u0.idx].update_max(u1);
            cnt += current_graph[u1.idx].update_max(u0);
        }
        for (size_t j = 0; j < idx_old_nodes.size(); ++j)
        {
            int idx1 = idx_old_nodes[j];

            double l = dist(data[idx0], data[idx1]);
            NNNode u0 = {.idx=idx0, .key=l, .visited=true};
            NNNode u1 = {.idx=idx1, .key=l, .visited=true};

            cnt += current_graph[u0.idx].update_max(u1);
            cnt += current_graph[u1.idx].update_max(u0);
        }
    }
    return cnt;
}

// void print_heap_lists(
    // NNHeap* current_graph,
    // IntVec* old,
    // IntVec* old_r,
    // IntVec* new,
    // IntVec* new_r,
    // int size)
// {
    // for (int v = 0; v < size; v++)
    // {
        // print_heap(current_graph[v]);

        // printf("old[%d]=", v);
        // print(old[v]);

        // printf("old_r[%d]=", v);
        // print(old_r[v]);

        // printf("new[%d]=", v);
        // print(new[v]);

        // printf("new_r[%d]=", v);
        // print(new_r[v]);

        // printf("\n");
    // }
// }

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
    std::vector<NNHeap> &current_graph,
    std::vector<RandHeap> &new_candidates,
    std::vector<RandHeap> &old_candidates,
    int max_candidates,
    RandomState rng_state,
    int n_threads
)
{
    // print(current_graph);
    timer.stop("sample_candidates: start^");
    new_candidates.resize(current_graph.size());
    old_candidates.resize(current_graph.size());

    for (int thread = 0; thread < n_threads; ++thread)
    {
        RandomState local_rng_state;
        for (int state = 0; state < STATE_SIZE; ++state)
        {
            local_rng_state[state] = rng_state[state] + thread + 1;
        }
        for (int i = 0; i < (int)current_graph.size(); ++i)
        {
            for (int j = 0; j < (int)current_graph[i].size(); ++j)
            {
                int idx = current_graph[i][j].idx;
                bool visited = current_graph[i][j].visited;

                if (idx < 0)
                {
                    continue;
                }

                uint64_t priority = rand_int(local_rng_state);

                if (!visited)
                {
                    if (i % n_threads == thread)
                    {
                        RandNode node = {.idx=idx, .key=priority};
                        new_candidates[i].update_max(node, max_candidates);
                    }
                    if (idx % n_threads == thread)
                    {
                        RandNode node = {.idx=i, .key=priority};
                        new_candidates[idx].update_max(node, max_candidates);
                    }
                }
                else
                {
                    if (i % n_threads == thread)
                    {
                        RandNode node = {.idx=idx, .key=priority};
                        old_candidates[i].update_max(node, max_candidates);
                    }
                    if (idx % n_threads == thread)
                    {
                        RandNode node = {.idx=i, .key=priority};
                        old_candidates[idx].update_max(node, max_candidates);
                    }
                }
            }
        }
    }
    // Mark sampled nodes in current_graph as visited.
    for (size_t i = 0; i < current_graph.size(); ++i)
    {
        for (size_t j = 0; j < current_graph[i].size(); ++j)
        {
            int idx = current_graph[i][j].idx;
            for (size_t k = 0; k < new_candidates[i].size(); ++k)
            {
                if (new_candidates[i][j].idx == idx)
                {
                    current_graph[i][j].visited = true;
                    break;
                }
            }
        }
    }

    // print(new_candidates);
    // print(old_candidates);
    // timer.stop("sample_candidates: stop^");
}


typedef struct
{
    int idx0;
    int idx1;
    double dist;
} NNUpdate;

void print(NNUpdate update)
{
    std::cout << "(idx0=" << update.idx0
              << ", idx1=" << update.idx1
              << ", dist=" << update.dist
              << ")";
}

void print(std::vector<NNUpdate> updates)
{
    std::cout << "[";
    for (size_t i = 0; i < updates.size(); ++i)
    {
        if (i > 0)
        {
            std::cout << " ";
        }
        print(updates[i]);
        if (i + 1 != updates.size())
        {
            std::cout << ",\n";
        }
    }
    std::cout << "]\n";
}


std::vector<NNUpdate> generate_graph_updates(
    const SlowMatrix &data,
    std::vector<NNHeap> &current_graph,
    std::vector<RandHeap> &new_candidate_neighbors,
    std::vector<RandHeap> &old_candidate_neighbors,
    int verbose
)
{
    assert(data.size() == new_candidate_neighbors.size());
    assert(data.size() == old_candidate_neighbors.size());
    std::vector<NNUpdate> updates;
    for (size_t i = 0; i < new_candidate_neighbors.size(); ++i)
    {
        for (size_t j = 0; j < new_candidate_neighbors[i].size(); ++j)
        {
            int idx0 = new_candidate_neighbors[i][j].idx;
            assert(idx0 >= 0);
            for (size_t k = j + 1; k < new_candidate_neighbors[i].size(); ++k)
            {
                int idx1 = new_candidate_neighbors[i][k].idx;
                assert(idx1 >= 0);
                double d = dist(data[idx0], data[idx1]);
                if (
                    (d < current_graph[idx0].max()) ||
                    (d < current_graph[idx1].max())
                )
                {
                    NNUpdate update = {idx0, idx1, d};
                    updates.push_back(update);
                    // print(update); std::cout << "i="<<i<<" j="<<j<<" k="<<k<<"\n";
                }

            }
            for (size_t k = 0; k < old_candidate_neighbors[i].size(); ++k)
            {
                int idx1 = old_candidate_neighbors[i][k].idx;
                assert(idx1 >= 0);
                double d = dist(data[idx0], data[idx1]);
                if (
                    (d < current_graph[idx0][0].key) ||
                    (d < current_graph[idx1][0].key)
                )
                {
                    NNUpdate update = {idx0, idx1, d};
                    updates.push_back(update);
                    // print(update); std::cout << "i="<<i<<" j="<<j<<" k="<<k<<"\n";
                }
            }
        }
        log(
            "\t\tGenerate updates" + std::to_string(i + 1) + "/"
                + std::to_string(new_candidate_neighbors.size()),
            (verbose && (i % (new_candidate_neighbors.size() / 4) == 0))
        );
    }

    return updates;
}


int apply_graph_updates(
    std::vector<NNHeap> &current_graph,
    std::vector<NNUpdate> updates,
    int n_threads
)
{
    int n_changes = 0;

    for (int thread = 0; thread < n_threads; ++thread)
    {
        for (NNUpdate update : updates)
        {
            int idx0 = update.idx0;
            int idx1 = update.idx1;
            assert(idx0 >= 0);
            assert(idx1 >= 0);
            double d = update.dist;
            if (idx0 % n_threads == 0)
            {
                NNNode u0 = {.idx=idx0, .key=d, .visited=true};
                NNNode u1 = {.idx=idx1, .key=d, .visited=true};
                n_changes += current_graph[u0.idx].update_max(u1);
            }
            if (idx1 % n_threads == 0)
            {
                NNNode u0 = {.idx=idx0, .key=d, .visited=true};
                NNNode u1 = {.idx=idx1, .key=d, .visited=true};
                n_changes += current_graph[u1.idx].update_max(u0);
            }
        }
    }

    return n_changes;
}



// TODO do parallel
void nn_descent(
    const SlowMatrix &data,
    std::vector<NNHeap> &current_graph,
    int n_neighbors,
    RandomState rng_state,
    int max_candidates,
    int n_iters,
    float delta,
    bool verbose
)
{
    timer.start();
    assert(current_graph.size() == data.size());

    log("NN descent for " + std::to_string(n_iters) + " iterations", verbose);

    timer.stop("init0");

    for (int iter = 0; iter < n_iters; ++iter)
    {
        log(
            (
                "\t" + std::to_string(iter + 1) + "  /  "
                + std::to_string(n_iters)
            ),
            verbose
        );

        std::vector<RandHeap> new_candidates;
        std::vector<RandHeap> old_candidates;
        int n_threads = 1;

        timer.start();

        sample_candidates(
            current_graph,
            new_candidates,
            old_candidates,
            max_candidates,
            rng_state,
            n_threads
        );
        timer.stop("sample_candidates");

        std::vector<NNUpdate> updates = generate_graph_updates(
            data,
            current_graph,
            new_candidates,
            old_candidates,
            verbose
        );
        // print(updates);
        timer.stop("generate graph updates");

        int cnt = 0;
        cnt += apply_graph_updates(
            current_graph,
            updates,
            n_threads
        );
        log("\t\t" + std::to_string(cnt) + " updates applied", verbose);
        timer.stop("apply graph updates");

        if (cnt < delta * data.size() * n_neighbors)
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

IntMatrix nn_brute_force(const SlowMatrix data, int n_neighbors)
{
    std::vector<NNHeap> current_graph(data.size());
    for (size_t i = 0; i < data.size(); ++i)
    {
        for (size_t j = 0; j < data.size(); ++j)
        {
            double l = dist(data[i], data[j]);
            NNNode u = {.idx=(int) j, .key=l, .visited=true};
            current_graph[i].update_max(u, n_neighbors);
        }
    }
    return get_index_matrix(current_graph);
}

double recall_accuracy(IntMatrix apx, IntMatrix ect)
{
    assert(apx.size() == ect.size());
    int hits = 0;
    int nrows = apx.size();
    int ncols = apx[0].size();

    for (size_t i = 0; i < apx.size(); i++)
    {
        for (size_t j = 0; j < apx[i].size(); ++j)
        {
            for (size_t k = 0; k < ect[i].size(); ++k)
            {
                if (apx[i][j] == ect[i][k])
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


NNDescent::NNDescent(Parms parms):
    data(parms.data),
    metric(parms.metric),
    n_neighbors(parms.n_neighbors),
    n_trees(parms.n_trees),
    leaf_size(parms.leaf_size),
    pruning_degree_multiplier(parms.pruning_degree_multiplier),
    diversify_prob(parms.diversify_prob),
    n_search_trees(parms.n_search_trees),
    tree_init(parms.tree_init),
    seed(parms.seed),
    low_memory(parms.low_memory),
    max_candidates(parms.max_candidates),
    n_iters(parms.n_iters),
    delta(parms.delta),
    n_jobs(parms.n_jobs),
    compressed(parms.compressed),
    parallel_batch_queries(parms.parallel_batch_queries),
    verbose(parms.verbose)
{
    timer.start();

    if (leaf_size == NONE)
    {
        leaf_size = std::max(10, n_neighbors);
    }
    if (n_trees == NONE)
    {
        n_trees = 5 + (int)std::round(std::pow(data.size(), 0.25));
        // Only so many trees are useful
        n_trees = std::min(32, n_trees);
    }
    if (max_candidates == NONE)
    {
        max_candidates = std::min(60, n_neighbors);
    }
    if (n_iters == NONE)
    {
        n_iters = std::max(5, (int)std::round(std::log2(data.size())));
    }
    seed_state(rng_state, seed);
    this->start();
}


void NNDescent::start()
{
    current_graph = empty_graph(data.size(), n_neighbors);
    timer.stop("empty graph");

    init_random(data, current_graph, n_neighbors, rng_state);
    // ::print(current_graph);
    // timer.stop("random init neighbours");

    if (verbose)
    {
        log("Building RP forest with " + std::to_string(n_trees) + " trees");
    }
    std::vector<IntMatrix> forest = make_forest(
        data, n_trees, leaf_size, rng_state
    );
    // ::print(forest);
    timer.stop("make forest");

    if (verbose)
    {
        log("Update Graph by  RP forest");
    }
    update_nn_graph_by_rp_forest(data, current_graph, forest);
    // ::print(current_graph);
    timer.stop("update graph by rp-tree forest");

    nn_descent(
        data,
        current_graph,
        n_neighbors,
        rng_state,
        max_candidates,
        n_iters,
        delta,
        verbose
    );
    neighbor_graph = get_index_matrix(current_graph);
    // ::print(current_graph);
    this->print();
    // ::print(neighbor_graph);
}
void NNDescent::print()
{
   std::cout << "\nNNDescent parameters\n********************\n"
        << "Data dimension: " <<  data.size() << "x" <<  data[0].size() << "\n"
        << "metric=" << metric << "\n"
        << "n_neighbors=" << n_neighbors << "\n"
        << "n_trees=" << n_trees << "\n"
        << "leaf_size=" << leaf_size << "\n"
        << "pruning_degree_multiplier=" << pruning_degree_multiplier << "\n"
        << "diversify_prob=" << diversify_prob << "\n"
        << "n_search_trees=" << n_search_trees  << "\n"
        << "tree_init=" << tree_init  << "\n"
        << "seed=" << seed  << "\n"
        << "low_memory=" << low_memory  << "\n"
        << "max_candidates=" << max_candidates  << "\n"
        << "n_iters=" << n_iters  << "\n"
        << "delta=" << delta  << "\n"
        << "n_jobs=" << n_jobs  << "\n"
        << "compressed=" << compressed  << "\n"
        << "parallel_batch_queries=" << parallel_batch_queries  << "\n"
        << "verbose=" << verbose  << "\n"
        << "\n";
}
