#include <chrono>
#include <iostream>
#include <limits>
#include <random>
#include <string>

#include "dtypes.h"

namespace RandNum
{
    // int seed = 1234;
    // int seed = time(NULL);
    uint64_t seed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
    std::mt19937 mersenne(seed);
}

const double EPS = 1e-8;
Timer timer_dtyp;
Timer timer_dtyp2;

// Converts heap list to Matrix.
IntMatrix get_index_matrix(std::vector<NNHeap> &graph)
{
    size_t nrows = graph.size();
    if (nrows == 0)
    {
        return IntMatrix(0, IntVec(0));
    }
    size_t ncols = graph[0].size();
    IntMatrix matrix (nrows, IntVec(ncols));
    for (size_t i = 0; i < nrows; ++i)
    {
        for (size_t j = 0; j < graph[i].size(); ++j)
        {
            matrix[i][j] = graph[i][j].idx;
        }
        // TODO del after changes to algorithm
        for (size_t j = graph[i].size(); j < ncols; ++j)
        {
            matrix[i][j] = -1;
        }
    }
    return matrix;
}

void print(std::vector<NNHeap> &graph)
{
    for (size_t i = 0; i < graph.size(); i++)
    {
        std::cout << i << ": ";
        for (size_t j = 0; j < graph[i].size(); ++j)
        {
            std::cout << " " << graph[i][j].idx;
        }
        std::cout << "\n";
    }
}

void print(IntMatrix &matrix)
{
    for (size_t i = 0; i < matrix.size(); i++)
    {
        std::cout << i << ": ";
        for (size_t j = 0; j < matrix[i].size(); ++j)
        {
            std::cout << " " << matrix[i][j];
        }
        std::cout << "\n";
    }
}

void print(std::vector<IntMatrix> &array)
{
    for (size_t i = 0; i < array.size(); i++)
    {
        std::cout << "[" <<i << "]\n";
        print(array[i]);
        std::cout << "\n";
    }
}

std::vector<double> midpoint(
    std::vector<double> &vec0,
    std::vector<double> &vec1
)
{
    std::vector<double> midpnt (vec0.size());
    for (size_t i = 0; i < vec0.size(); ++i)
    {
        midpnt[i] = (vec0[i] + vec1[i]) / 2;
    }
    return midpnt;
}

// Performs a random projection tree split on the data in 'parent'
// by selecting two points in random and splitting along the connecting
// line.
void euclidean_random_projection_split
(
    const SlowMatrix &data,
    IntVec &parent,
    IntVec &child0,
    IntVec &child1,
    RandomState &rng_state
)
{
    int rand0 = rand_int(rng_state) % parent.size();
    int rand1 = rand_int(rng_state) % parent.size();

    if (rand0 == rand1)
    {
        // Leads to non-uniform sampling but is simple and avoids looping.
        rand0 = (rand1 + 1) % parent.size();
    }

    std::vector<double> vec0 = data[parent[rand0]];
    std::vector<double> vec1 = data[parent[rand1]];
    std::vector<double> midpnt = midpoint(vec0, vec1);

    std::vector<double> norm (vec0.size());
    for (size_t i = 0; i < vec0.size(); ++i)
    {
        norm[i] = vec1[i] - vec0[i];
    }

    double affine_const = dot_product(midpnt, norm);

    std::vector<bool> side(parent.size());
    int cnt0 = 0;
    int cnt1 = 0;

    const int parent_size = parent.size();

    for (int i = 0; i < parent_size; ++i)
    {
        double dot_prod = dot_product(norm, data[parent[i]]);

        if (dot_prod - affine_const < -EPS)
        {
            side[i] = 0;
            ++cnt0;
        }
        else if (dot_prod - affine_const > EPS)
        {
            side[i] = 1;
            ++cnt1;
        }
        else if (rand_int(rng_state) % 2 == 0)
        {
            side[i] = 0;
            ++cnt0;
        }
        else
        {
            side[i] = 1;
            ++cnt1;
        }
    }
    // If all points end up on one side, something went wrong numerically
    // In this case, assign points randomly; they are likely very close anyway
    if (cnt0 == 0 || cnt1 == 0)
    {
        cnt0 = 0;
        cnt1 = 0;
        for (size_t i = 0; i < parent.size(); ++i)
        {
            side[i] = rand_int(rng_state) % 2;
            if (side[i] == 0)
            {
                ++cnt0;
            }
            else
            {
                ++cnt1;
            }
        }
        std::cout << "random tree split failed\n";
    }
    child0.resize(cnt0);
    child1.resize(cnt1);
    cnt0 = 0;
    cnt1 = 0;
    for (size_t i = 0; i < parent.size(); ++i)
    {
        if (side[i] == 0)
        {
            child0[cnt0] = parent[i];
            ++cnt0;
        }
        else
        {
            child1[cnt1] = parent[i];
            ++cnt1;
        }
    }
}



// // Performs a random projection tree split on the data in 'parent'
// // by selecting two points in random and splitting along the connecting
// // line.
// void euclidean_random_projection_split
// (
    // const SlowMatrix &data,
    // IntVec &parent,
    // IntVec &child0,
    // IntVec &child1,
    // RandomState &rng_state
// )
// {
    // // timer_dtyp.start();
    // // timer_dtyp.stop("----S T A R T----");
    // int rand0 = rand_int(rng_state) % parent.size();
    // int rand1 = rand_int(rng_state) % parent.size();
    // // timer_dtyp.stop("rng");

    // if (rand0 == rand1)
    // {
        // // Leads to non-uniform sampling but is simple and avoids looping.
        // rand0 = (rand1 + 1) % parent.size();
    // }

    // // timer_dtyp.stop("r0==r1");
    // std::vector<double> vec0 = data[parent[rand0]];
    // // timer_dtyp.stop("vec0");
    // std::vector<double> vec1 = data[parent[rand1]];
    // // timer_dtyp.stop("vec1");
    // std::vector<double> midpnt = midpoint(vec0, vec1);
    // // timer_dtyp.stop("midpnt");

    // std::vector<double> norm (vec0.size());
    // // timer_dtyp.stop("normalloc");
    // for (size_t i = 0; i < vec0.size(); ++i)
    // {
        // norm[i] = vec1[i] - vec0[i];
    // }
    // // timer_dtyp.stop("norm calc");

    // double affine_const = std::inner_product(
        // midpnt.begin(), midpnt.end(), norm.begin(), 0.0
    // );
    // // timer_dtyp.stop("affine");

    // IntVec side(parent.size());
    // int cnt0 = 0;
    // int cnt1 = 0;

    // // timer_dtyp.stop("----");
    // for (size_t i = 0; i < parent.size(); ++i)
    // {
        // double dot_prod = std::inner_product(
            // norm.begin(), norm.end(), data[parent[i]].begin(), 0.0
        // );
        // // double dot_prod = dot_product(norm, data[parent[i]]);
        // // double dot_prod = (double)(rand_int(rng_state) % 10) - 5.0;

        // if (dot_prod - affine_const < -EPS)
        // {
            // side[i] = 0;
            // ++cnt0;
        // }
        // else if (dot_prod - affine_const > EPS)
        // {
            // side[i] = 1;
            // ++cnt1;
        // }
        // else if (rand_int(rng_state) % 2 == 0)
        // {
            // side[i] = 0;
            // ++cnt0;
        // }
        // else
        // {
            // side[i] = 1;
            // ++cnt1;
        // }
    // }
    // // timer_dtyp.stop("first loop");

    // // // TODO del for testing only
    // // std::vector<float> f1 (10000000);
    // // // f1 = [0,1,2,...]
    // // std::iota(f1.begin(), f1.end(), 0);
    // // // timer_dtyp.start();
    // // float dotf = std::inner_product(
        // // f1.begin(), f1.end(), f1.begin(), 0.0f
    // // );
    // // // timer_dtyp.stop("float dot product");
    // // std::cout << "float dot=" << dotf << "\n";
    // // std::vector<double> d1 (10000000);
    // // // d1 = [0,1,2,...]
    // // std::iota(d1.begin(), d1.end(), 0);
    // // // timer_dtyp.start();
    // // double dotd = std::inner_product(
        // // d1.begin(), d1.end(), d1.begin(), 0.0
    // // );
    // // // timer_dtyp.stop("double dot product");
    // // std::cout << "double dot=" << dotd << "\n";

    // // If all points end up on one side, something went wrong numerically
    // // In this case, assign points randomly; they are likely very close anyway
    // if (cnt0 == 0 || cnt1 == 0)
    // {
        // cnt0 = 0;
        // cnt1 = 0;
        // for (size_t i = 0; i < parent.size(); ++i)
        // {
            // side[i] = rand_int(rng_state) % 2;
            // if (side[i] == 0)
            // {
                // ++cnt0;
            // }
            // else
            // {
                // ++cnt1;
            // }
        // }
        // std::cout << "random tree split failed\n";
    // }
    // // timer_dtyp.stop("womething wrong");

    // child0.resize(cnt0);
    // child1.resize(cnt1);

    // // timer_dtyp.stop("resize");
    // cnt0 = 0;
    // cnt1 = 0;

    // for (size_t i = 0; i < parent.size(); ++i)
    // {
        // if (side[i] == 0)
        // {
            // child0[cnt0] = parent[i];
            // ++cnt0;
        // }
        // else
        // {
            // child1[cnt1] = parent[i];
            // ++cnt1;
        // }
    // }
    // // timer_dtyp.stop("last loop");
// }

// Builds a random projection tree by recursively splitting.
void build_rp_tree
(
    IntMatrix &rp_tree,
    const SlowMatrix &data,
    IntVec parent,
    unsigned int leaf_size,
    RandomState &rng_state,
    // int max_depth=2 //TODO del
    int max_depth=100

)
{
    if (parent.size() <= leaf_size)
    {
        rp_tree.push_back(parent);
        return;
    }
    if (max_depth <= 0)
    {
        std::cout << "tree depth limit reached\n";
        int parent_size = std::min(leaf_size, (unsigned int)parent.size());
        parent.resize(parent_size);
        rp_tree.push_back(parent);
        return;
    }

    IntVec child0;
    IntVec child1;

    timer_dtyp2.start();
    euclidean_random_projection_split
    (
        data,
        parent,
        child0,
        child1,
        rng_state
    );
    if (max_depth >98){timer_dtyp2.stop("euclidean_random_projection_split");}
    build_rp_tree
    (
        rp_tree,
        data,
        child0,
        leaf_size,
        rng_state,
        max_depth - 1
    );
    build_rp_tree
    (
        rp_tree,
        data,
        child1,
        leaf_size,
        rng_state,
        max_depth - 1
    );
}

// Builds a random projection tree.
IntMatrix make_rp_tree
(
    const SlowMatrix &data,
    unsigned int leaf_size,
    RandomState &rng_state
)
{
    // IntMatrix rp_tree(1 + data.size()/leaf_size);
    IntMatrix rp_tree;

    IntVec all_points (data.size());
    // all_points = [0,1,2,...]
    std::iota(all_points.begin(), all_points.end(), 0);
    build_rp_tree
    (
        rp_tree,
        data,
        all_points,
        leaf_size,
        rng_state
    );

    return rp_tree;
}

std::vector<IntMatrix> make_forest
(
    const SlowMatrix &data,
    int n_trees,
    int leaf_size,
    RandomState &rng_state
)
{
    std::vector<IntMatrix> forest(n_trees);
    for (int i = 0; i < n_trees; ++i)
    {
        RandomState local_rng_state;
        for (int state = 0; state < STATE_SIZE; ++state)
        {
            local_rng_state[state] = rng_state[state] + i + 1;
        }
        IntMatrix tree = make_rp_tree(data, leaf_size, local_rng_state);
        forest[i] = tree;
    }
    return forest;
}

void print(SlowMatrix matrix)
{
    std::cout << "[";

    for(size_t i = 0; i < matrix.size(); ++i)
    {
        if (i > 0)
        {
            std::cout << " ";
        }
        std::cout << "[";
        for(size_t j = 0; j < matrix[i].size(); j++)
        {
            std::cout << matrix[i][j];
            if (j + 1 != matrix[i].size())
            {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (i + 1 != matrix.size())
        {
            std::cout << ",\n";
        }
    }

    std::cout << "]\n";
}

// Print the data as 2d map.
void print_map(SlowMatrix matrix)
{
    // Calculate maximum coordinates.
    int x_max = 0;
    int y_max = 0;
    for (size_t j = 0; j < matrix.size(); ++j)
    {
        int x_cur = matrix[j][0];
        int y_cur = matrix[j][1];
        x_max = (x_cur > x_max) ? x_cur : x_max;
        y_max = (y_cur > y_max) ? y_cur : y_max;
    }

    // Initialize 2d map to be printed to console.
    std::vector<std::vector<char>> map(
        y_max + 1, std::vector<char>(x_max+ 4)
    );
    for (int i = 0; i <= y_max; ++i)
    {
        for (int j = 0; j <= x_max; ++j)
        {
            map[i][j] = ' ';
        }
        map[i][x_max + 1] = '|';
        map[i][x_max + 2] = '\n';
        map[i][x_max + 3] = '\0';
    }

    // Draw data as single digits.
    for (size_t j = 0; j < matrix.size(); ++j)
    {
        int x_cur = matrix[j][0];
        int y_cur = matrix[j][1];
        map[y_cur][x_cur] = '0' + j % 10;
    }

    // Print map.
    for (int i = 0; i <= y_max; ++i)
    {
        std::string row(map[i].begin(), map[i].end());
        std::cout << row;
    }
}

void print(IntVec vec)
{
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i)
    {
        std::cout << vec[i];
        if (i + 1 != vec.size())
        {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}

void print(NNNode nd)
{
    std::cout << "(idx=" << nd.idx
              << " dist=" << nd.key
              << " visited=" << nd.visited
              << ")\n";
}

void print(RandNode nd)
{
    std::cout << "(idx=" << nd.idx
              << " priority=" << nd.key
              << ")\n";
}

void print(NNHeap heap)
{
    print<NNHeap>(heap);
}

void print(RandHeap heap)
{
    print<RandHeap>(heap);
}

void print(std::vector<NNHeap> graph)
{
    print<NNHeap>(graph);
}

void print(std::vector<RandHeap> graph)
{
    print<RandHeap>(graph);
}
