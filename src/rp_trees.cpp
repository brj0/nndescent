/**
 * @file rp_trees.cpp
 *
 * @brief Contains implementation of random projection trees.
 */


#include "rp_trees.h"


namespace nndescent
{


template<>
std::tuple<std::vector<int>, std::vector<int>, std::vector<float>, float>
random_projection_split<EuclideanSplit>(
    const Matrix<float> &data,
    std::vector<int> &indices,
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

    size_t idx0 = indices[rand0];
    size_t idx1 = indices[rand1];

    std::vector<float> midpnt(dim);
    std::vector<float> hyperplane_vector(dim);
    for (size_t i = 0; i < dim; ++i)
    {
        midpnt[i] = (data(idx0, i) + data(idx1, i)) / 2;
        hyperplane_vector[i] = data(idx0, i) - data(idx1, i);
    }

    const float hyperplane_offset = std::inner_product(
        hyperplane_vector.begin(),
        hyperplane_vector.end(),
        midpnt.begin(),
        0.0f
    );

    // For each point compute the margin (project into normal vector). If we
    // are on lower side of the hyperplane put in one pile, otherwise put it in
    // the other pile (if we hit hyperplane on the nose, flip a coin)
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
random_projection_split<AngularSplit>(
    const Matrix<float> &data,
    std::vector<int> &indices,
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

    size_t idx0 = indices[rand0];
    size_t idx1 = indices[rand1];

    float norm0 = std::sqrt(
        std::inner_product(
            data.begin(idx0), data.end(idx0), data.begin(idx0), 0.0f
        )
    );
    float norm1 = std::sqrt(
        std::inner_product(
            data.begin(idx1), data.end(idx1), data.begin(idx1), 0.0f
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
        hyperplane_vector[i] = data(idx0, i) / norm0
            - data(idx1, i) / norm1;
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
        hyperplane_norm = 1.0f;
    }
    for (size_t i = 0; i < dim; ++i)
    {
        hyperplane_vector[i] = hyperplane_vector[i] / hyperplane_norm;
    }

    // For each point compute the margin (project into normal vector). If we
    // are on lower side of the hyperplane put in one pile, otherwise put it in
    // the other pile (if we hit hyperplane on the nose, flip a coin)
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


template<>
std::tuple<
    std::vector<int>,
    std::vector<int>,
    std::vector<size_t>,
    std::vector<float>,
    float
>
sparse_random_projection_split<EuclideanSplit>(
    const CSRMatrix<float> &data,
    std::vector<int> &indices,
    RandomState &rng_state
)
{
    size_t size = indices.size();

    size_t rand0 = rand_int(rng_state) % size;
    size_t rand1 = rand_int(rng_state) % size;

    if (rand0 == rand1)
    {
        // Leads to non-uniform sampling but is simple and avoids looping.
        rand0 = (rand1 + 1) % size;
    }

    size_t idx0 = indices[rand0];
    size_t idx1 = indices[rand1];

    std::vector<size_t> midpnt_col_ind;
    std::vector<float> midpnt_data;

    std::tie(midpnt_col_ind, midpnt_data) = sparse_sum(
        data.begin_col(idx0), data.end_col(idx0), data.begin_data(idx0),
        data.begin_col(idx1), data.end_col(idx1), data.begin_data(idx1)
    );

    for (float& element : midpnt_data)
    {
        element /= 2.0f;
    }

    std::vector<size_t> hyperplane_ind;
    std::vector<float> hyperplane_data;

    std::tie(hyperplane_ind, hyperplane_data) = sparse_diff(
        data.begin_col(idx0), data.end_col(idx0), data.begin_data(idx0),
        data.begin_col(idx1), data.end_col(idx1), data.begin_data(idx1)
    );

    const float hyperplane_offset = sparse_inner_product(
        hyperplane_ind.begin(),
        hyperplane_ind.end(),
        hyperplane_data.begin(),
        midpnt_col_ind.begin(),
        midpnt_col_ind.end(),
        midpnt_data.begin()
    );

    // For each point compute the margin (project into normal vector). If we
    // are on lower side of the hyperplane put in one pile, otherwise put it in
    // the other pile (if we hit hyperplane on the nose, flip a coin)
    std::vector<bool> side(size);
    int cnt0 = 0;
    int cnt1 = 0;

    for (size_t i = 0; i < size; ++i)
    {
        // Time consuming operation.
        float margin = -hyperplane_offset;
        margin += sparse_inner_product(
            hyperplane_ind.begin(),
            hyperplane_ind.end(),
            hyperplane_data.begin(),

            data.begin_col(indices[i]),
            data.end_col(indices[i]),
            data.begin_data(indices[i])
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
        left_indices,
        right_indices,
        hyperplane_ind,
        hyperplane_data,
        hyperplane_offset
    );
}


template<>
std::tuple<
    std::vector<int>,
    std::vector<int>,
    std::vector<size_t>,
    std::vector<float>,
    float
>
sparse_random_projection_split<AngularSplit>(
    const CSRMatrix<float> &data,
    std::vector<int> &indices,
    RandomState &rng_state
)
{
    size_t size = indices.size();

    size_t rand0 = rand_int(rng_state) % size;
    size_t rand1 = rand_int(rng_state) % size;

    if (rand0 == rand1)
    {
        // Leads to non-uniform sampling but is simple and avoids looping.
        rand0 = (rand1 + 1) % size;
    }

    size_t idx0 = indices[rand0];
    size_t idx1 = indices[rand1];

    float norm0 = std::sqrt(
        sparse_inner_product(
            data.begin_col(rand0),
            data.end_col(rand0),
            data.begin_data(rand0),
            data.begin_col(rand0),
            data.end_col(rand0),
            data.begin_data(rand0)
        )
    );
    float norm1 = std::sqrt(
        sparse_inner_product(
            data.begin_col(idx1),
            data.end_col(idx1),
            data.begin_data(idx1),
            data.begin_col(idx1),
            data.end_col(idx1),
            data.begin_data(idx1)
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
    std::vector<size_t> hyperplane_ind;
    std::vector<float> hyperplane_data;

    std::tie(hyperplane_ind, hyperplane_data) = sparse_weighted_diff(
        data.begin_col(idx0),
        data.end_col(idx0),
        data.begin_data(idx0),
        norm0,
        data.begin_col(idx1),
        data.end_col(idx1),
        data.begin_data(idx1),
        norm1
    );

    float hyperplane_norm = std::sqrt(
        sparse_inner_product(
            hyperplane_ind.begin(),
            hyperplane_ind.end(),
            hyperplane_data.begin(),
            hyperplane_ind.begin(),
            hyperplane_ind.end(),
            hyperplane_data.begin()
        )
    );
    if (std::abs(hyperplane_norm) < EPS)
    {
        hyperplane_norm = 1.0f;
    }
    for (float& element : hyperplane_data)
    {
        element /= hyperplane_norm;
    }

    // For each point compute the margin (project into normal vector). If we
    // are on lower side of the hyperplane put in one pile, otherwise put it in
    // the other pile (if we hit hyperplane on the nose, flip a coin)
    std::vector<bool> side(size);
    int cnt0 = 0;
    int cnt1 = 0;

    for (size_t i = 0; i < size; ++i)
    {
        // Time consuming operation.
        float margin = 0.0f;
        margin += sparse_inner_product(
            hyperplane_ind.begin(),
            hyperplane_ind.end(),
            hyperplane_data.begin(),

            data.begin_col(indices[i]),
            data.end_col(indices[i]),
            data.begin_data(indices[i])
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
        left_indices,
        right_indices,
        hyperplane_ind,
        hyperplane_data,
        0.0f
    );
}


Matrix<int> get_leaves_from_forest(
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


template<>
std::vector<int> RPTree::get_leaf(
    const Matrix<float> &query_data,
    size_t row,
    RandomState &rng_state
) const
{
    return this->get_leaf(query_data.begin(row), rng_state);
}


template<>
std::vector<int> RPTree::get_leaf(
    const CSRMatrix<float> &query_data,
    size_t row,
    RandomState &rng_state
) const
{
    return this->get_leaf(
        query_data.begin_col(row),
        query_data.end_col(row),
        query_data.begin_data(row),
        rng_state
    );
}


std::ostream& operator<<(std::ostream &out, const RPTNode &node)
{
    out << "[of=" << node.offset
        << ", hp=" << node.hyperplane
        << ", hpi=" << node.hyperplane_ind
        << ", id=" << node.indices
        << ", lr=" << node.left << "," << node.right
        << "]";
    return out;
}


/*
 * This is an auxiliary function used for recursively printing a random
 * projection tree. It adds the tree representation to the output stream
 * 'out', with a specified 'prefix', starting from a given node 'from'. The
 * 'is_left' parameter indicates whether the node is the left child of its
 * parent.
 */
void _add_tree_from_to_stream(
    std::ostream &out,
    const std::string &prefix,
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


std::ostream& operator<<(std::ostream &out, const RPTree &tree)
{
    out << "Tree(leaf_size=" << tree.leaf_size << ", n_leaves="
        << tree.n_leaves << ", n_nodes=" << tree.nodes.size() << ",\n";
    int start = tree.nodes.size() - 1;
    _add_tree_from_to_stream(out, "", tree, start, false);
    out << ")\n";
    return out;
}


} // namespace nndescent
