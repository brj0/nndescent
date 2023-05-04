#include "dtypes.h"
#include "rp_trees.h"

// Performs a random projection tree split on the data in 'parent'
// by selecting two points in random and splitting along the connecting
// line.
void euclidean_random_projection_split
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
    std::vector<float> norm(dim);
    for (size_t i = 0; i < dim; ++i)
    {
        midpnt[i] = (data(parent[rand0], i) + data(parent[rand1], i)) / 2;
        norm[i] = data(parent[rand0], i) - data(parent[rand1], i);
    }

    float affine_const = 0.0f;
    for (size_t j = 0; j < dim; ++j)
    {
        affine_const += norm[j] * midpnt[j];
    }

    std::vector<bool> side(size);
    int cnt0 = 0;
    int cnt1 = 0;

    // timer_dtyp.stop("first part");
    for (size_t i = 0; i < size; ++i)
    {
        float margin = -affine_const;

        for (size_t j = 0; j < dim; ++j)
        {
            margin += norm[j] * data(parent[i], j);
        }

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


// Builds a random projection tree by recursively splitting.
void build_rp_tree
(
    IntMatrix &rp_tree,
    const Matrix<float> &data,
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

    // std::vector<size_t> parent_(parent.size());
    // std::vector<size_t> child0_(child0.size());
    // std::vector<size_t> child1_(child1.size());
    // for (size_t i = 0; i < parent_.size(); ++i) { parent_.at(i) = parent.at(i); }
    // for (size_t i = 0; i < child0_.size(); ++i) { child0_.at(i) = child0.at(i); }
    // for (size_t i = 0; i < child1_.size(); ++i) { child1_.at(i) = child1.at(i); }
    // timer_dtyp2.start();
    euclidean_random_projection_split
    (
        data,
        parent,
        child0,
        child1,
        rng_state
    );
    // if (max_depth >98){timer_dtyp2.stop("euclidean_random_projection_split");}
    // parent.resize(parent_.size());
    // child0.resize(child0_.size());
    // child1.resize(child1_.size());
    // for (size_t i = 0; i < parent_.size(); ++i) { parent.at(i) = parent_.at(i); }
    // for (size_t i = 0; i < child0_.size(); ++i) { child0.at(i) = child0_.at(i); }
    // for (size_t i = 0; i < child1_.size(); ++i) { child1.at(i) = child1_.at(i); }

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
    const Matrix<float> &data,
    unsigned int leaf_size,
    RandomState &rng_state
)
{
    // IntMatrix rp_tree(1 + data.nrows()/leaf_size);
    IntMatrix rp_tree;

    IntVec all_points (data.nrows());
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
    const Matrix<float> &data,
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

