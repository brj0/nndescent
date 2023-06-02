/**
 * @file rp_trees.cpp
 *
 * @brief Contains implementation of random projection trees.
 */


#include <cstring>
#include <tuple>

#include "dtypes.h"
#include "rp_trees.h"
#include "utils.h"


namespace nndescent
{


// Types
typedef std::vector<int> IntVec;
typedef std::vector<IntVec> IntMatrix;

// Dummy DistanceType classes for templates
class Euclidean {};
class Angular {};


/**
 * @brief Performs a random projection tree split.
 *
 * This function performs a random projection tree split on the given data
 * indices by selecting two points at random and splitting along the hyperplane
 * that goes through the midpoint between the two points and is normal to their
 * difference.
 *
 * @tparam DistanceType The distance type to use for the split, which can be
 * either Euclidean or Angular.
 *
 * @param data The input data matrix.
 *
 * @param indices The indices of the data points to perform the split on.
 *
 * @param rng_state The random state used for generating random numbers.
 *
 * @return A tuple containing the left child indices, right child indices,
 * hyperplane normal vector, and hyperplane offset.
 */
template<class DistanceType>
std::tuple<std::vector<int>, std::vector<int>, std::vector<float>, float>
random_projection_split
(
    const Matrix<float> &data,
    IntVec &indices,
    RandomState &rng_state
);


template<>
std::tuple<std::vector<int>, std::vector<int>, std::vector<float>, float>
random_projection_split<Euclidean>
(
    const Matrix<float> &data,
    IntVec &indices,
    RandomState &rng_state
)
{
    size_t dim = data.ncols();
    size_t size = indices.size();

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
        midpnt[i] = (data(indices[rand0], i) + data(indices[rand1], i)) / 2;
        hyperplane_vector[i] = data(indices[rand0], i) - data(indices[rand1], i);
    }

    const float hyperplane_offset = std::inner_product(
        hyperplane_vector.begin(),
        hyperplane_vector.end(),
        midpnt.begin(),
        0.0f
    );

    std::vector<bool> side(size);
    int cnt0 = 0;
    int cnt1 = 0;

    for (size_t i = 0; i < size; ++i)
    {
        // Time consuming operation.
        float margin = std::inner_product(
            hyperplane_vector.begin(),
            hyperplane_vector.end(),
            data.begin(indices[i]),
            -hyperplane_offset
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

    // If all points end up on one side, something went wrong numerically.  In
    // this case, assign points randomly; they are likely very close anyway.
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
    }
    std::vector<int> left_indices(cnt0);
    std::vector<int> right_indices(cnt1);
    cnt0 = 0;
    cnt1 = 0;
    for (size_t i = 0; i < size; ++i)
    {
        if (side[i] == 0)
        {
            left_indices[cnt0] = indices[i];
            ++cnt0;
        }
        else
        {
            right_indices[cnt1] = indices[i];
            ++cnt1;
        }
    }
    return std::make_tuple(
        left_indices, right_indices, hyperplane_vector, hyperplane_offset
    );
}


template<>
std::tuple<std::vector<int>, std::vector<int>, std::vector<float>, float>
random_projection_split<Angular>
(
    const Matrix<float> &data,
    IntVec &indices,
    RandomState &rng_state
)
{
    size_t dim = data.ncols();
    size_t size = indices.size();

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
        hyperplane_vector[i] = data(indices[rand0], i) / norm0
            - data(indices[rand1], i) / norm1;
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

    // For each point compute the margin (project into normal vector) If we are
    // on lower side of the hyperplane put in one pile, otherwise put it in the
    // other pile (if we hit hyperplane on the nose, flip a coin)
    std::vector<bool> side(size);
    int cnt0 = 0;
    int cnt1 = 0;

    for (size_t i = 0; i < size; ++i)
    {
        // Time consuming operation.
        float margin = std::inner_product(
            hyperplane_vector.begin(),
            hyperplane_vector.end(),
            data.begin(indices[i]),
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

    // If all points end up on one side, something went wrong numerically. In
    // this case, assign points randomly; they are likely very close anyway.
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
    }

    // Now that we have the counts allocate arrays
    std::vector<int> left_indices(cnt0);
    std::vector<int> right_indices(cnt1);
    cnt0 = 0;
    cnt1 = 0;

    // Populate the arrays with graph_indices according to which
    // side they fell on.
    for (size_t i = 0; i < size; ++i)
    {
        if (side[i] == 0)
        {
            left_indices[cnt0] = indices[i];
            ++cnt0;
        }
        else
        {
            right_indices[cnt1] = indices[i];
            ++cnt1;
        }
    }
    return std::make_tuple(left_indices, right_indices, hyperplane_vector, 0.0f);
}


/**
 * @brief Builds a random projection tree by recursively splitting.
 *
 * This function builds a random projection tree by recursively splitting the
 * data using random projection.
 *
 * @tparam DistanceType The distance type to use for the split.
 * @param rp_tree The random projection tree object to build.
 * @param data The input data matrix.
 * @param indices The indices of the data points to consider for splitting.
 * @param leaf_size The maximum number of points in a leaf node.
 * @param rng_state The random state used for generating random numbers.
 * @param max_depth The maximum depth of the tree (default: 100).
 */
template<class DistanceType>
void make_sparse_tree
(
    RPTree &rp_tree,
    const Matrix<float> &data,
    IntVec indices,
    unsigned int leaf_size,
    RandomState &rng_state,
    int max_depth=100
)
{
    if (indices.size() <= leaf_size)
    {
        rp_tree.add_leaf(indices);
        return;
    }
    if (max_depth <= 0)
    {
        // Prune leaf to 'leaf_size'.
        int parent_size = std::min(leaf_size, (unsigned int)indices.size());
        indices.resize(parent_size);
        rp_tree.add_leaf(indices);
        return;
    }

    std::vector<int> left_indices;
    std::vector<int> right_indices;
    std::vector<float> hyperplane;
    float offset;

    std::tie
    (
        left_indices, right_indices, hyperplane, offset
    ) = random_projection_split<DistanceType>
    (
        data,
        indices,
        rng_state
    );

    make_sparse_tree<DistanceType>
    (
        rp_tree,
        data,
        left_indices,
        leaf_size,
        rng_state,
        max_depth - 1
    );

    size_t left_subtree = rp_tree.get_index();

    make_sparse_tree<DistanceType>
    (
        rp_tree,
        data,
        right_indices,
        leaf_size,
        rng_state,
        max_depth - 1
    );

    size_t right_subtree = rp_tree.get_index();
    rp_tree.add_node(left_subtree, right_subtree, offset, hyperplane);
}


/**
 * @brief Builds a random projection tree.
 *
 * This function builds a random projection tree. The tree partitions the data
 * points into leaves using random projections.
 *
 * @param data The input data matrix.
 * @param leaf_size The maximum number of points in a leaf.
 * @param rng_state The random state used for generating random numbers.
 * @param angular Specifies whether to use angular distance (default: false).
 * @return The constructed random projection tree.
 */
RPTree make_sparse_tree
(
    const Matrix<float> &data,
    unsigned int leaf_size,
    RandomState &rng_state,
    bool angular=false
)
{
    RPTree rp_tree(leaf_size);

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


std::vector<RPTree> make_forest
(
    const Matrix<float> &data,
    int n_trees,
    int leaf_size,
    RandomState &rng_state
)
{
    std::vector<RPTree> forest(n_trees);
    #pragma omp parallel for
    for (int i = 0; i < n_trees; ++i)
    {
        RandomState local_rng_state;
        for (int state = 0; state < STATE_SIZE; ++state)
        {
            local_rng_state[state] = rng_state[state] + i + 1;
        }
        RPTree tree = make_sparse_tree(data, leaf_size, local_rng_state);
        forest[i] = tree;
    }
    return forest;
}


Matrix<int> get_leaves_from_forest
(
    std::vector<RPTree> &forest
)
{
    size_t leaf_size = forest[0].leaf_size;
    size_t n_leaves = 0;
    for (const auto& tree : forest)
    {
        n_leaves += tree.n_leaves;
    }
    Matrix<int> leaf_matrix(n_leaves, leaf_size, NONE);

    int row = 0;
    for (const auto& tree : forest)
    {
        for (const auto &node : tree.nodes)
        {
            if (node.indices.size() > 0)
            {
                for (size_t j = 0; j < node.indices.size(); ++j)
                {
                    leaf_matrix(row, j) = node.indices[j];
                }
                ++row;
            }
        }
    }
    return leaf_matrix;
}


std::ostream& operator<<(std::ostream &out, RPTNode &node)
{
    out << "[of=" << node.offset << ", hp="
        << node.hyperplane << ", id=" << node.indices
        << ", lr=" << node.left << "," << node.right << "]";
    return out;
}


/*
 * This is an auxiliary function used for recursively printing a random
 * projection tree.  It adds the tree representation to the output stream
 * 'out', with a specified 'prefix', starting from a given node 'from'. The
 * 'is_left' parameter indicates whether the node is the left child of its
 * parent.
 */
void _add_tree_from_to_stream
(
    std::ostream &out,
    std::string prefix,
    RPTree tree,
    int from,
    char is_left
)
{
    if (from < 0)
    {
        return;
    }

    out << prefix;
    out << (is_left && (from > 0) ? "├──" : "└──");

    // Print current node
    RPTNode current_node = tree.nodes[from];
    out << current_node
        << "\n";
    std::string prefix_children = prefix + (is_left ? "│   " : "    ");

    // Add children of current node.
    int left = current_node.left;
    int right = current_node.right;
    _add_tree_from_to_stream(out, prefix_children, tree, left, true);
    _add_tree_from_to_stream(out, prefix_children, tree, right, false);
}


std::ostream& operator<<(std::ostream &out, RPTree &tree)
{
    out << "Tree(leaf_size=" << tree.leaf_size << ", n_leaves="
        << tree.n_leaves << ", n_nodes=" << tree.nodes.size() << ",\n";
    int start = tree.nodes.size() - 1;
    _add_tree_from_to_stream(out, "", tree, start, false);
    out << ")\n";
    return out;
}


} // namespace nndescent
