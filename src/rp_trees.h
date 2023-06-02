/**
 * @file rp_trees.h
 *
 * @brief Contains implementation of random projection trees.
 */


#pragma once


#include "dtypes.h"
#include "distances.h"

namespace nndescent
{


// Epsilon value used for random projection split.
const float EPS = 1e-8;


/**
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


/**
 * @brief Represents a node in the Random Projection Tree.
 *
 * This struct represents a node in the Random Projection Tree. It stores the
 * left and right child indices, the offset value, the hyperplane coefficients,
 * and the indices associated with the leaves.
 */
struct RPTNode
{
    /**
     * Index of the left child node.
     */
    int left;


    /**
     * Index of the right child node.
     */
    int right;


    /**
     * Hyperplane offset.
     */
    float offset;


    /**
     * Hyperplane for random projecton split at current node.
     */
    std::vector<float> hyperplane;


    /**
     * Contains the indices of a leaf or is empty for non-leave nodes.
     */
    std::vector<int> indices;


    /**
     * @brief Constructs an RPTNode object.
     *
     * @param left Index of the left child node.
     * @param right Index of the right child node.
     * @param offset Hyperplane offset value for splitting the node.
     * @param hyperplane Hyperplane for splitting the node.
     * @param indices Indices associated with the node.
     */
    RPTNode(
        int left,
        int right,
        float offset,
        std::vector<float> hyperplane,
        std::vector<int> indices
    )
    : left(left)
    , right(right)
    , offset(offset)
    , hyperplane(hyperplane)
    , indices(indices)
    {}
};


/**
 * @brief Random Projection Tree class.
 */
class RPTree
{
private:
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


    /**
     * @brief Default constructor builds an empty tree.
     */
    RPTree() {}


    /**
     * @brief Constructor an rp tree with fixed leaf size.
     */
    RPTree(size_t leaf_size)
        : leaf_size(leaf_size)
        , n_leaves(0)
        {}


    /**
     * @brief Add a leaf node to the tree.
     * @param indices The indices of the data points in the leaf node.
     */
    void add_leaf(std::vector<int> indices)
    {
        RPTNode node(NONE, NONE, FLOAT_MAX, {}, indices);
        nodes.push_back(node);
        ++n_leaves;
    }


    /**
     * @brief Get the index of the last added node.
     * @return The index of the last added node.
     */
    size_t get_index() const
    {
        return nodes.size() - 1;
    }


    /**
     * @brief Add an internal node to the tree, containing no indices.
     *
     * @param left_subtree The index of the left subtree.
     * @param right_subtree The index of the right subtree.
     * @param offset The offset value of the hyperplane.
     * @param hyperplane The hyperplane vector.
     */
    void add_node
    (
        size_t left_subtree,
        size_t right_subtree,
        float offset,
        std::vector<float> hyperplane
    )
    {
        RPTNode node(left_subtree, right_subtree, offset, hyperplane, {});
        nodes.push_back(node);
    }


    /**
     * @brief Perform a nearest neighbor query on the tree.
     *
     * @param query_point The query point.
     * @param rng_state The random state used for generating random numbers.
     * @return The indices of the nearest neighbors.
     */
    std::vector<int> get_leaf
    (
        float *query_point,
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
                query_point,
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


     /**
     * @brief Calculate the score of the tree in comparison to the nearest
     * neighbor graph.
     *
     * This function calculates the score of the tree by comparing it to a
     * nearest neighbor graph. Note that this function is currently not used as
     * the scores in a forest are very close to each other, and calculating
     * scores can be time-consuming.
     */
    float score(Matrix<int> &nn_graph) const
    {
        int hits = 0;

        for (const auto &node : nodes)
        {
            for (const int &idx : node.indices)
            {
                int intersection = arr_intersect
                (
                    node.indices.begin(),
                    node.indices.end(),
                    nn_graph.begin(idx),
                    nn_graph.end(idx)
                );
                std::vector<int>i=node.indices;
                std::vector<int>n(nn_graph.begin(idx), nn_graph.end(idx));
                if (intersection > 1)
                {
                    ++hits;
                }
            }
        }
        return ((float) hits) / nn_graph.nrows();
    }
};


/**
 * @brief Builds a forest of random projection trees.
 *
 * This function builds a vector of random projection trees.  The forest
 * consists of multiple random projection trees, each constructed
 * independently.
 *
 * @param data The input data matrix.
 * @param n_trees The number of trees in the forest.
 * @param leaf_size The maximum number of points in a leaf node for each tree.
 * @param rng_state The random state used for generating random numbers.
 * @return The constructed forest of random projection trees.
 */
std::vector<RPTree> make_forest
(
    const Matrix<float> &data,
    int n_trees,
    int leaf_size,
    RandomState &rng_state
);


/**
 * @brief Extracts the leaves from a random projection tree forest.
 *
 * This function extracts the 'leaves', which are sets of indices of maximal
 * size 'leaf_size', from a random projection tree forest and puts them into a
 * more cache-friendly data structure.  The resulting matrix has dimensions
 * 'n_leaves' x 'leaf_size'. If a leaf has fewer than 'leaf_size' elements, the
 * remaining elements in the row are filled with a special value (e.g., NONE).
 *
 * @param forest The random projection tree forest.
 * @return A matrix containing the extracted leaves.
 */
Matrix<int> get_leaves_from_forest
(
    std::vector<RPTree> &forest
);


/*
 * @brief Prints a RPTNode object to an output stream.
 */
std::ostream& operator<<(std::ostream &out, RPTNode &node);


/*
 * @brief Prints a RPTree object to an output stream.
 */
std::ostream& operator<<(std::ostream &out, RPTree &tree);


} // namespace nndescent
