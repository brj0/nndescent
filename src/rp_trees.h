/*
 * DASGUPTA, Sanjoy; FREUND, Yoav. Random projection trees and low dimensional
 * manifolds. In: Proceedings of the fortieth annual ACM symposium on Theory of
 * computing. 2008. S. 537-546.
 *
 * https://dl.acm.org/doi/pdf/10.1145/1374376.1374452
 * https://cseweb.ucsd.edu/~dasgupta/papers/rptree-stoc.pdf
 */

#pragma once


#include "dtypes.h"
#include "distances.h"

const float EPS = 1e-8;

struct RPTNode
{
    int left;
    int right;
    float offset;
    std::vector<float> hyperplane;
    std::vector<int> indices;

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

class RPTree
{
private:
public:
    size_t leaf_size;
    size_t n_leaves;
    std::vector<RPTNode> nodes;
    RPTree() {}
    RPTree(size_t leaf_size)
        : leaf_size(leaf_size)
        , n_leaves(0)
        {}
    void add_leaf(std::vector<int> indices)
    {
        RPTNode node(NONE, NONE, FLOAT_MAX, {}, indices);
        nodes.push_back(node);
        ++n_leaves;
    }
    size_t get_index()
    {
        return nodes.size() - 1;
    }
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
    std::vector<int> get_leaf
    (
        std::vector<float> query_point, RandomState &rng_state
    ) const
    {
        size_t index = n_leaves;
        while (nodes[index].left != NONE)
        {
            std::vector<float> hyperplane_vector = nodes[index].hyperplane;
            float hyperplane_offset = nodes[index].offset;

            float margin = std::inner_product(
                hyperplane_vector.begin(),
                hyperplane_vector.end(),
                query_point.begin(),
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
};


std::vector<RPTree> make_forest
(
    const Matrix<float> &data,
    int n_trees,
    int leaf_size,
    RandomState &rng_state
);

Matrix<int> get_leaves_from_forest
(
    std::vector<RPTree> &forest
);


std::ostream& operator<<(std::ostream &out, RPTNode &node);
std::ostream& operator<<(std::ostream &out, RPTree &tree);
