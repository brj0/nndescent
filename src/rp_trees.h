/**
 * @file rp_trees.h
 *
 * @brief Contains implementation of random projection trees.
 */


#pragma once

#include <tuple>

#include "utils.h"
#include "dtypes.h"
#include "distances.h"


namespace nndescent
{


// Dummy classes for templates
class EuclideanSplit {};
class AngularSplit {};


// Epsilon value used for random projection split.
const float EPS = 1e-8;


/*
 * @brief Counts the number of common elements between two ranges.
 */
template<class Iter0, class Iter1>
int arr_intersect(Iter0 first0, Iter0 last0, Iter1 first1, Iter1 last1)
{
    std::vector<int> arr0;
    std::vector<int> arr1;
    std::vector<int> intersection;

    std::copy(first0, last0, std::back_inserter(arr0));
    std::copy(first1, last1, std::back_inserter(arr1));

    std::sort(arr0.begin(), arr0.end());
    std::sort(arr1.begin(), arr1.end());

    std::set_intersection(
        arr0.begin(),
        arr0.end(),
        arr1.begin(),
        arr1.end(),
        std::back_inserter(intersection)
    );

    return intersection.size();
}


/*
 * @brief Represents a node in the Random Projection Tree.
 *
 * This struct represents a node in the Random Projection Tree. It stores the
 * left and right child indices, the offset value, the hyperplane coefficients,
 * and the indices associated with the leaves.
 */
struct RPTNode
{

    /*
     * Index of the left child node.
     */
    int left;

    /*
     * Index of the right child node.
     */
    int right;

    /*
     * Hyperplane offset.
     */
    float offset;

    /*
     * Indices of hyperplane used for sparse matrices.
     */
    std::vector<size_t> hyperplane_ind;

    /*
     * Hyperplane for random projecton split at current node.
     */
    std::vector<float> hyperplane;

    /*
     * Contains the indices of a leaf or is empty for non-leave nodes.
     */
    std::vector<int> indices;

    /*
     * @brief Constructs an RPTNode object for dense input data.
     */
    RPTNode(
        int left,
        int right,
        float offset,
        const std::vector<float> &hyperplane,
        const std::vector<int> &indices
    )
        : left(left)
        , right(right)
        , offset(offset)
        , hyperplane(hyperplane)
        , indices(indices)
    {
    }

    /*
     * @brief Constructs an RPTNode object for sparse input data.
     */
    RPTNode(
        int left,
        int right,
        float offset,
        const std::vector<size_t> &hyperplane_ind,
        const std::vector<float> &hyperplane,
        const std::vector<int> &indices
    )
        : left(left)
        , right(right)
        , offset(offset)
        , hyperplane_ind(hyperplane_ind)
        , hyperplane(hyperplane)
        , indices(indices)
    {
    }
};


/*
 * @brief Random Projection Tree class.
 */
class RPTree
{

private:

    /*
     * @brief Perform a nearest neighbor query on the tree.
     *
     * @param query_data Pointer to the beginning of the query point.
     * @param rng_state The random state used for generating random numbers.
     *
     * @return The indices of the nearest neighbors.
     */
    std::vector<int> get_leaf(
        float *query_data,
        RandomState &rng_state
    ) const
    {
        size_t index = get_index();
        while (nodes[index].left != NONE)
        {
            std::vector<float> hyperplane_vector = nodes[index].hyperplane;
            float hyperplane_offset = nodes[index].offset;

            float margin = std::inner_product(
                hyperplane_vector.begin(),
                hyperplane_vector.end(),
                query_data,
                -hyperplane_offset
            );

            if (margin < -EPS)
            {
                index = nodes[index].left;
            }
            else if (margin > EPS)
            {
                index = nodes[index].right;
            }
            else if (rand_int(rng_state) % 2 == 0)
            {
                index = nodes[index].left;
            }
            else
            {
                index = nodes[index].right;
            }
        }
        return nodes[index].indices;
    }

    /*
     * @brief Perform a nearest neighbor query on the tree (sparse version).
     *
     * @param query_first_ind Pointer to the beginning of the array containing
     * the CSR indices of the query point.
     * @param query_last_ind Pointer to the end of the array containing the CSR
     * indices of the query points.
     * @param query_data Pointer to the beginning of the array containing the
     * CSR data of the query point.
     * query point data.
     * @param rng_state The random state used for generating random numbers.
     *
     * @return The indices of the nearest neighbors.
     */
    std::vector<int> get_leaf(
        size_t *query_first_ind,
        size_t *query_last_ind,
        float *query_data,
        RandomState &rng_state
    ) const
    {
        size_t index = get_index();
        while (nodes[index].left != NONE)
        {
            std::vector<size_t> hyperplane_ind = nodes[index].hyperplane_ind;
            std::vector<float> hyperplane_data = nodes[index].hyperplane;
            float hyperplane_offset = nodes[index].offset;

            float margin = -hyperplane_offset;
            margin += sparse_inner_product(
                hyperplane_ind.begin(),
                hyperplane_ind.end(),
                hyperplane_data.begin(),
                query_first_ind,
                query_last_ind,
                query_data
            );

            if (margin < -EPS)
            {
                index = nodes[index].left;
            }
            else if (margin > EPS)
            {
                index = nodes[index].right;
            }
            else if (rand_int(rng_state) % 2 == 0)
            {
                index = nodes[index].left;
            }
            else
            {
                index = nodes[index].right;
            }
        }
        return nodes[index].indices;
    }

public:

    /*
     * The maximum number of points in a leaf node.
     */
    size_t leaf_size;

    /*
     * The number of leaf nodes in the tree.
     */
    size_t n_leaves;

    /*
     * The nodes of the tree.
     */
    std::vector<RPTNode> nodes;

    /*
     * @brief Default constructor builds an empty tree.
     */
    RPTree() {}

    /*
     * @brief Constructs an rp tree with fixed leaf size.
     */
    explicit RPTree(size_t leaf_size)
        : leaf_size(leaf_size)
        , n_leaves(0)
    {
    }

    /*
     * @brief Add a leaf node to the tree.
     * @param indices The indices of the data points in the leaf node.
     */
    void add_leaf(std::vector<int> indices)
    {
        RPTNode node(NONE, NONE, FLOAT_MAX, {}, indices);
        nodes.push_back(node);
        ++n_leaves;
    }

    /*
     * @brief Get the index of the last added node.
     *
     * @return The index of the last added node.
     */
    size_t get_index() const
    {
        return nodes.size() - 1;
    }

    /*
     * @brief Add an internal node to the tree, containing no indices.
     */
    void add_node(
        size_t left_subtree,
        size_t right_subtree,
        float offset,
        std::vector<float> &hyperplane
    )
    {
        RPTNode node(left_subtree, right_subtree, offset, hyperplane, {});
        nodes.push_back(node);
    }

    /*
     * @brief Add an internal node to the tree, containing no indices (sparse
     * case).
     */
    void add_node(
        size_t left_subtree,
        size_t right_subtree,
        float offset,
        std::vector<size_t> &hyperplane_ind,
        std::vector<float> &hyperplane
    )
    {
        RPTNode node(
            left_subtree,
            right_subtree,
            offset,
            hyperplane_ind,
            hyperplane,
            {}
        );
        nodes.push_back(node);
    }

    /*
     * @brief Perform a nearest neighbor query on the tree.
     *
     * @param query_data The query data containing the query point.
     * @param row The index of the query point in the query data.
     * @param rng_state The random state used for generating random numbers.
     *
     * @return The indices of the nearest neighbors.
     */
    template<class MatrixType>
    std::vector<int> get_leaf(
        const MatrixType &query_data,
        size_t row,
        RandomState &rng_state
    ) const;

    /*
     * @brief Calculate the score of the tree in comparison to the nearest
     * neighbor graph.
     *
     * This function calculates the score of the tree by comparing it to a
     * nearest neighbor graph. Note that this function is currently not used as
     * the scores in a forest are very close to each other, and calculating
     * scores is time-consuming.
     */
    float score(Matrix<int> &nn_graph) const
    {
        int hits = 0;

        for (const auto &node : nodes)
        {
            for (const int &idx : node.indices)
            {
                int intersection = arr_intersect(
                    node.indices.begin(),
                    node.indices.end(),
                    nn_graph.begin(idx),
                    nn_graph.end(idx)
                );
                if (intersection > 1)
                {
                    ++hits;
                }
            }
        }
        return ((float) hits) / nn_graph.nrows();
    }

};


/*
 * @brief Performs a random projection tree split.
 *
 * This function performs a random projection tree split on the given data
 * indices by selecting two points at random and splitting along the hyperplane
 * that goes through the midpoint between the two points and is normal to their
 * difference.
 *
 * @tparam SplitType The split type to use for the split, which can be either
 * EuclideanSplit or AngularSplit.
 *
 * @param data The input data matrix.
 * @param indices The indices of the data points to perform the split on.
 * @param rng_state The random state used for generating random numbers.
 *
 * @return A tuple containing the left child indices, right child indices,
 * hyperplane normal vector, and hyperplane offset.
 */
template<class SplitType>
std::tuple<std::vector<int>, std::vector<int>, std::vector<float>, float>
random_projection_split(
    const Matrix<float> &data,
    std::vector<int> &indices,
    RandomState &rng_state
);


/*
 * @brief Performs a random projection tree split (sparse version).
 *
 * This function performs a random projection tree split on the given data
 * indices by selecting two points at random and splitting along the hyperplane
 * that goes through the midpoint between the two points and is normal to their
 * difference.
 *
 * @tparam SplitType The split type to use for the split, which can be either
 * EuclideanSplit or AngularSplit.
 *
 * @param data The input sparse CSR matrix.
 * @param indices The indices of the data points to perform the split on.
 * @param rng_state The random state used for generating random numbers.
 *
 * @return A tuple containing the left child indices, right child indices,
 * hyperplane normal vector CSR-indices, hyperplane normal vector CSR-data, and
 * hyperplane offset.
 */
template<class SplitType>
std::tuple<
    std::vector<int>,
    std::vector<int>,
    std::vector<size_t>,
    std::vector<float>,
    float
>
sparse_random_projection_split(
    const CSRMatrix<float> &data,
    std::vector<int> &indices,
    RandomState &rng_state
);


/*
 * @brief Builds a random projection tree by recursively splitting.
 *
 * This function builds a random projection tree by recursively splitting the
 * data using random projection.
 *
 * @tparam SplitType The split type to use for the split.
 * @tparam MatrixType The matrix type (either dense or sparse).
 *
 * @param rp_tree The random projection tree object to build.
 * @param data The input data matrix.
 * @param indices The indices of the data points to consider for splitting.
 * @param leaf_size The maximum number of points in a leaf node.
 * @param rng_state The random state used for generating random numbers.
 * @param max_depth The maximum depth of the tree (default: 100).
 */
template<class SplitType, class MatrixType>
void build_rp_tree_recursively(
    RPTree &rp_tree,
    const MatrixType &data,
    std::vector<int> indices,
    unsigned int leaf_size,
    RandomState &rng_state,
    int max_depth=100
);


template<class SplitType>
void build_rp_tree_recursively(
    RPTree &rp_tree,
    const Matrix<float> &data,
    std::vector<int> indices,
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

    std::tie(
        left_indices, right_indices, hyperplane, offset
    ) = random_projection_split<SplitType>(
        data,
        indices,
        rng_state
    );

    build_rp_tree_recursively<SplitType>(
        rp_tree,
        data,
        left_indices,
        leaf_size,
        rng_state,
        max_depth - 1
    );

    size_t left_subtree = rp_tree.get_index();

    build_rp_tree_recursively<SplitType>(
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


template<class SplitType>
void build_rp_tree_recursively(
    RPTree &rp_tree,
    const CSRMatrix<float> &data,
    std::vector<int> indices,
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
    std::vector<size_t> hyperplane_ind;
    std::vector<float> hyperplane_data;
    float offset;

    std::tie(
        left_indices, right_indices, hyperplane_ind, hyperplane_data, offset
    ) = sparse_random_projection_split<SplitType>(
        data,
        indices,
        rng_state
    );

    build_rp_tree_recursively<SplitType>(
        rp_tree,
        data,
        left_indices,
        leaf_size,
        rng_state,
        max_depth - 1
    );

    size_t left_subtree = rp_tree.get_index();

    build_rp_tree_recursively<SplitType>(
        rp_tree,
        data,
        right_indices,
        leaf_size,
        rng_state,
        max_depth - 1
    );

    size_t right_subtree = rp_tree.get_index();
    rp_tree.add_node(
        left_subtree, right_subtree, offset, hyperplane_ind, hyperplane_data
    );
}


/*
 * @brief Builds a random projection tree.
 *
 * This function builds a random projection tree. The tree partitions the data
 * points into leaves using random projections.
 *
 * @tparam MatrixType The matrix type (either dense or sparse).
 *
 * @param data The input data matrix.
 * @param leaf_size The maximum number of points in a leaf.
 * @param rng_state The random state used for generating random numbers.
 * @param angular Specifies whether to use angular split (default: false).
 *
 * @return The constructed random projection tree.
 */
template<class MatrixType>
RPTree build_rp_tree(
    const MatrixType &data,
    unsigned int leaf_size,
    RandomState &rng_state,
    bool angular=false
)
{
    RPTree rp_tree(leaf_size);

    std::vector<int> all_points (data.nrows());
    // all_points = [0,1,2,...]
    std::iota(all_points.begin(), all_points.end(), 0);
    if (angular)
    {
        build_rp_tree_recursively<AngularSplit>(
            rp_tree,
            data,
            all_points,
            leaf_size,
            rng_state
        );
    }
    else
    {
        build_rp_tree_recursively<EuclideanSplit>(
            rp_tree,
            data,
            all_points,
            leaf_size,
            rng_state
        );
    }

    return rp_tree;
}


/*
 * @brief Builds a forest of random projection trees.
 *
 * This function builds a vector of random projection trees. The forest
 * consists of multiple random projection trees, each constructed
 * independently.
 *
 * @tparam MatrixType The matrix type (either dense or sparse).
 *
 * @param data The input data matrix.
 * @param n_trees The number of trees in the forest.
 * @param leaf_size The maximum number of points in a leaf node for each tree.
 * @param rng_state The random state used for generating random numbers.
 *
 * @return The constructed forest of random projection trees.
 */
template<class MatrixType>
std::vector<RPTree> make_forest(
    const MatrixType &data,
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
        RPTree tree = build_rp_tree(data, leaf_size, local_rng_state);
        forest[i] = tree;
    }
    return forest;
}


/*
 * @brief Extracts the leaves from a random projection tree forest.
 *
 * This function extracts the 'leaves', which are sets of indices of maximal
 * size 'leaf_size', from a random projection tree forest and puts them into a
 * more cache-friendly data structure. The resulting matrix has dimensions
 * 'n_leaves' x 'leaf_size'. If a leaf has fewer than 'leaf_size' elements, the
 * remaining elements in the row are filled with the special value 'NONE'.
 *
 * @param forest The random projection tree forest.
 *
 * @return A matrix containing the extracted leaves.
 */
Matrix<int> get_leaves_from_forest( std::vector<RPTree> &forest);


/*
 * @brief Prints a RPTNode object to an output stream.
 */
std::ostream& operator<<(std::ostream &out, const RPTNode &node);


/*
 * @brief Prints a RPTree object to an output stream.
 */
std::ostream& operator<<(std::ostream &out, const RPTree &tree);


} // namespace nndescent
