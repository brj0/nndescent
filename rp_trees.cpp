#include <cstring>

#include "dtypes.h"
#include "rp_trees.h"
#include "utils.h"


const float EPS = 1e-8;

Timer timer_dtyp;

// Dummy DistanceType classes for templates
class Euclidean {};
class Angular {};


/**
 * @brief Performs a random projection tree split on the data in 'parent'
 *
 * Performs a random projection tree split on the data in 'parent'
 * by selecting two points in random and splitting along the connecting
 * line.
 */
template<class DistanceType>
void random_projection_split(
    const Matrix<float> &data,
    IntVec &parent,
    IntVec &child0,
    IntVec &child1,
    RandomState &rng_state
);

template<>
void random_projection_split<Euclidean>
(
    const Matrix<float> &data,
    IntVec &parent,
    IntVec &child0,
    IntVec &child1,
    RandomState &rng_state
)
{
    // timer_dtyp.start();
    size_t dim = data.ncols();
    size_t size = parent.size();

    size_t rand0 = rand_int(rng_state) % size;
    size_t rand1 = rand_int(rng_state) % size;

    if (rand0 == rand1)
    {
        // Leads to non-uniform sampling but is simple and avoids looping.
        rand0 = (rand1 + 1) % size;
    }
    std::vector<float> midpnt(dim);
    std::vector<float> hyperplane_vector(dim);
    for (size_t i = 0; i < dim; ++i)
    {
        midpnt[i] = (data(parent[rand0], i) + data(parent[rand1], i)) / 2;
        hyperplane_vector[i] = data(parent[rand0], i) - data(parent[rand1], i);
    }

    // float affine_const = 0.0f;
    // for (size_t j = 0; j < dim; ++j)
    // {
        // affine_const += hyperplane_vector[j] * midpnt[j];
    // }
    const float affine_const = std::inner_product(
        hyperplane_vector.begin(),
        hyperplane_vector.end(),
        midpnt.begin(),
        0.0f
    );

    std::vector<bool> side(size);
    int cnt0 = 0;
    int cnt1 = 0;

    // timer_dtyp.stop("first part");
    for (size_t i = 0; i < size; ++i)
    {
        // float margin = -affine_const;

        // for (size_t j = 0; j < dim; ++j)
        // {
            // margin += hyperplane_vector[j] * data(parent[i], j);
        // }
        float margin = std::inner_product(
            hyperplane_vector.begin(),
            hyperplane_vector.end(),
            data.begin(parent[i]),
            -affine_const
        );

        if (margin < -EPS)
        {
            side[i] = 0;
            ++cnt0;
        }
        else if (margin > EPS)
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
    // timer_dtyp.stop("loop");

    // If all points end up on one side, something went wrong numerically
    // In this case, assign points randomly; they are likely very close anyway
    if (cnt0 == 0 || cnt1 == 0)
    {
        cnt0 = 0;
        cnt1 = 0;
        for (size_t i = 0; i < size; ++i)
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
    for (size_t i = 0; i < size; ++i)
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
    // timer_dtyp.stop("end");
}


template<>
void random_projection_split<Angular>
(
    const Matrix<float> &data,
    IntVec &parent,
    IntVec &child0,
    IntVec &child1,
    RandomState &rng_state
)
{
    // timer_dtyp.start();
    size_t dim = data.ncols();
    size_t size = parent.size();

    size_t rand0 = rand_int(rng_state) % size;
    size_t rand1 = rand_int(rng_state) % size;

    if (rand0 == rand1)
    {
        // Leads to non-uniform sampling but is simple and avoids looping.
        rand0 = (rand1 + 1) % size;
    }
    float norm0 = std::sqrt(
        std::inner_product(
            data.begin(rand0), data.end(rand0), data.begin(rand0), 0.0f
        )
    );
    float norm1 = std::sqrt(
        std::inner_product(
            data.begin(rand1), data.end(rand1), data.begin(rand1), 0.0f
        )
    );

    if (std::abs(norm0) < EPS)
    {
        norm0 = 1.0f;
    }
    if (std::abs(norm1) < EPS)
    {
        norm1 = 1.0f;
    }

    // Compute the normal vector to the hyperplane (the vector between
    // the two normalized points)
    std::vector<float> hyperplane_vector(dim);
    for (size_t i = 0; i < dim; ++i)
    {
        hyperplane_vector[i] = data(parent[rand0], i) / norm0
            - data(parent[rand1], i) / norm1;
    }

    float hyperplane_norm = std::sqrt(
        std::inner_product(
            hyperplane_vector.begin(),
            hyperplane_vector.end(),
            hyperplane_vector.begin(),
            0.0f
        )
    );
    if (std::abs(hyperplane_norm) < EPS)
    {
        norm1 = 1.0f;
    }
    for (size_t i = 0; i < dim; ++i)
    {
        hyperplane_vector[i] = hyperplane_vector[i] / hyperplane_norm;
    }

    // For each point compute the margin (project into normal vector)
    // If we are on lower side of the hyperplane put in one pile, otherwise
    // put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    std::vector<bool> side(size);
    int cnt0 = 0;
    int cnt1 = 0;

    for (size_t i = 0; i < size; ++i)
    {
        float margin = std::inner_product(
            hyperplane_vector.begin(),
            hyperplane_vector.end(),
            data.begin(parent[i]),
            0.0f
        );

        if (margin < -EPS)
        {
            side[i] = 0;
            ++cnt0;
        }
        else if (margin > EPS)
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

    // If all points end up on one side, something went wrong numerically.
    // In this case, assign points randomly; they are likely very close anyway
    if (cnt0 == 0 || cnt1 == 0)
    {
        cnt0 = 0;
        cnt1 = 0;
        for (size_t i = 0; i < size; ++i)
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

    // Now that we have the counts allocate arrays
    child0.resize(cnt0);
    child1.resize(cnt1);
    cnt0 = 0;
    cnt1 = 0;
    // Populate the arrays with graph_indices according to which
    // side they fell on.
    for (size_t i = 0; i < size; ++i)
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


// Builds a random projection tree by recursively splitting.
template<class DistanceType>
void make_sparse_tree
(
    IntMatrix &rp_tree,
    const Matrix<float> &data,
    IntVec parent,
    unsigned int leaf_size,
    RandomState &rng_state,
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

    random_projection_split<DistanceType>
    (
        data,
        parent,
        child0,
        child1,
        rng_state
    );

    make_sparse_tree<DistanceType>
    (
        rp_tree,
        data,
        child0,
        leaf_size,
        rng_state,
        max_depth - 1
    );
    make_sparse_tree<DistanceType>
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
IntMatrix make_sparse_tree
(
    const Matrix<float> &data,
    unsigned int leaf_size,
    RandomState &rng_state,
    bool angular=false
)
{
    // IntMatrix rp_tree(1 + data.nrows()/leaf_size);
    IntMatrix rp_tree;

    IntVec all_points (data.nrows());
    // all_points = [0,1,2,...]
    std::iota(all_points.begin(), all_points.end(), 0);
    if (angular)
    {
        make_sparse_tree<Angular>(
            rp_tree,
            data,
            all_points,
            leaf_size,
            rng_state
        );
    }
    else
    {
        make_sparse_tree<Euclidean>(
            rp_tree,
            data,
            all_points,
            leaf_size,
            rng_state
        );
    }

    return rp_tree;
}

std::vector<IntMatrix> make_forest
(
    const Matrix<float> &data,
    int n_trees,
    int leaf_size,
    RandomState &rng_state
)
{
    std::vector<IntMatrix> forest(n_trees);
    #pragma omp parallel for //shared(forest) //num_threads(1)
    for (int i = 0; i < n_trees; ++i)
    {
        RandomState local_rng_state;
        for (int state = 0; state < STATE_SIZE; ++state)
        {
            local_rng_state[state] = rng_state[state] + i + 1;
        }
        IntMatrix tree = make_sparse_tree(data, leaf_size, local_rng_state);
        forest[i] = tree;
    }
    return forest;
}

Matrix<int> get_leaves_from_forest
(
    std::vector<IntMatrix> &forest,
    int leaf_size
)
{
    size_t n_rows = 0;
    for (const auto& tree : forest)
    {
        n_rows += tree.size();
    }
    Matrix<int> leaf_matrix(n_rows, leaf_size, -1);

    int current_row = 0;
    for (size_t i = 0; i < forest.size(); ++i)
    {
        for (size_t j = 0; j < forest[i].size(); ++j)
        {
            for (size_t k = 0; k < forest[i][j].size(); ++k)
            {
                leaf_matrix(current_row, k) = forest[i][j][k];
            }
            ++current_row;
        }
    }
    return leaf_matrix;
}

