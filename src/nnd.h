/**
 * @file nnd.h
 *
 * @author Jon Brugger
 *
 * @brief Implements the Nearest Neighbor Descent algorithm for approximate
 * nearest neighbor search.
 *
 * This file contains a C++ implementation of the pynndescent library,
 * originally written by Leland McInnes, which performs approximate nearest
 * neighbor search. The main goal is to construct a k-nearest neighbor graph
 * quickly and accurately.
 *
 * @see https://github.com/lmcinnes/pynndescent
 *
 * The algorithm is based on the following paper:
 *
 * Dong, Wei, Charikar Moses, and Kai Li. "Efficient k-nearest neighbor graph
 * construction for generic similarity measures." Proceedings of the 20th
 * International Conference on World Wide Web. 2011.
 *
 * @see https://dl.acm.org/doi/pdf/10.1145/1963405.1963487
 * @see https://www.cs.princeton.edu/cass/papers/www11.pdf
 *
 * Furthermore the algorithm utilizes random projection trees for initializing
 * the nearest neighbor graph. The nndescent algorithm constructs a tree by
 * randomly selecting two points and splitting the data along a hyperplane
 * passing through their midpoint. For a more theoretical background, please
 * refer to:
 *
 * DASGUPTA, Sanjoy; FREUND, Yoav. Random projection trees and low dimensional
 * manifolds. In: Proceedings of the Fortieth Annual ACM Symposium on Theory of
 * Computing. 2008. pp. 537-546.
 *
 * @see https://dl.acm.org/doi/pdf/10.1145/1374376.1374452
 * @see https://cseweb.ucsd.edu/~dasgupta/papers/rptree-stoc.pdf
 *
 * This implementation utilizes C++ and OpenMP for efficient computation. It
 * supports dense and sparse matrices and provides implementations of several
 * distance functions.
 */


#pragma once

#include "utils.h"
#include "dtypes.h"
#include "distances.h"
#include "rp_trees.h"


namespace nndescent
{


/**
 * @version 1.0.3
 */
const std::string PROJECT_VERSION = "1.0.3";


// Constants
const char OLD = '0';
const char NEW = '1';
const int MAX_INT = std::numeric_limits<int>::max();
const int DEFAULT_K = 10;
const float DEFAULT_EPSILON = 0.1f;


/*
 * Throws an exception if no sparse metric is implemented.
 */
void throw_exception_if_sparse(std::string metric, bool is_sparse);


/*
 * @brief Corrects distances using a distance correction function.
 *
 * Some metrics have alternative forms that allow for faster calculations.
 * However, these alternative forms may produce slightly different distances
 * compared to the original metric. This function applies a distance correction
 * by using the provided distance correction function to adjust the distances
 * calculated using the alternative form.
 *
 * @param dist The distance function containing a correction function.
 * @param in The input matrix of distances to be corrected.
 * @param out The output matrix of corrected distances.
 */
template<class DistType>
void correct_distances(
    DistType &dist,
    Matrix<float> &in,
    Matrix<float> &out
)
{
    out.resize(in.nrows(), in.ncols());
    for (size_t i = 0; i < in.nrows(); ++i)
    {
        for (size_t j = 0; j < in.ncols(); ++j)
        {
            out(i, j) = dist.correction(in(i,j));
        }
    }
    return;
}


/**
 * @brief Calculates the recall accuracy between two k-NN matrices.
 *
 * This function calculates the recall accuracy between two matrices, 'apx' and
 * 'ect'.  The matrices represent approximate and exact nearest neighbors,
 * respectively.  The recall accuracy is defined as the ratio of the number of
 * common neighbors in the matrices to the total number of neighbors in the
 * exact matrix.
 *
 * @param apx The matrix representing approximate nearest neighbors.
 * @param ect The matrix representing exact nearest neighbors.
 *
 * @return The recall accuracy between the matrices.
 */
float recall_accuracy(Matrix<int> apx, Matrix<int> ect);


/**
 * @brief Structure representing the parameters for NNDescent.
 *
 * This structure holds the parameters with default values for configuring the
 * NNDescent algorithm. See NNDescent class for a description of the
 * parameters.
 */
struct Parms
{
    std::string metric="euclidean";
    float p_metric=1.0f;
    int n_neighbors=30;
    int n_trees=NONE;
    int leaf_size=NONE;
    float pruning_degree_multiplier=1.5;
    float pruning_prob=1.0f;
    bool tree_init=true;
    int seed=NONE;
    int max_candidates=NONE;
    int n_iters=NONE;
    float delta=0.001;
    int n_threads=NONE;
    bool verbose=false;
    std::string algorithm="nnd";
};


/**
 * @brief NNDescent class for fast approximate nearest neighbor queries.
 *
 * NNDescent is a flexible algorithm that supports a wide variety of distances,
 * including non-metric distances. It scales well against high-dimensional data
 * in many cases. This implementation provides a straightforward interface with
 * access to tuning parameters.
 */
class NNDescent
{
private:

    /*
     * The training data matrix.
     */
    Matrix<float> data;

    /*
     * The training data as CSR matrix.
     */
    CSRMatrix<float> csr_data;

    /*
     * The size of the training data matrix.
     */
    size_t data_size;

    /*
     * The dimension of the training data matrix.
     */
    size_t data_dim;

    /*
     * The random state used for generating random numbers.
     */
    RandomState rng_state;

    /*
     * The collection of random projection trees.
     */
    std::vector<RPTree> forest;

    /*
     * The search tree used for nearest neighbor queries.
     */
    RPTree search_tree;

    /*
     * The search graph used for nearest neighbor queries.
     */
    HeapList<float> search_graph;

    /*
     * Flag indicating whether angular trees are used.
     */
    bool angular_trees;

    /*
     * Flag indicating whether the training data is sparse or dense.
     */
    bool is_sparse;

    /*
     * @brief Set the parameters for NNDescent.
     */
    void set_parameters(Parms &parms);

    /*
     * @brief Sets the distance template and performs either NN algorithm
     * indexing/training or a query.
     *
     * Sets the distance template based on the specified metric and performs
     * either NN algorithm indexing or starts a query, depending on the
     * 'perform_query' flag.
     */
    template<class MatrixType>
    void set_dist_and_start_nn(
        bool perform_query=false,
        const MatrixType &query_data=MatrixType(),
        int query_k=0,
        float query_epsilon=0
    );

    /*
     * @brief Starts the nearest neighbor search algorithm.
     *
     * This function starts the nearest neighbor search algorithm using the
     * specified distance metric and performs either indexing/training or a
     * query based on the 'perform_query' flag.
     */
    template<class MatrixType, class DistType>
    void start_nn(
        DistType &dist,
        bool perform_query,
        const MatrixType &query_data,
        int query_k,
        float query_epsilon
    );

    /*
     * @brief Performs the NN-d algorithm for nearest neighbor search on the
     * training data.
     */
    template<class MatrixType, class DistType>
    void run_nn_descent(const MatrixType &train_data, const DistType &dist);

    /*
     * @brief Perform k-nearest neighbors search using brute force, used for
     * debugging purposes.
     */
    template<class MatrixType, class DistType>
    void start_brute_force(const MatrixType &train_data, const DistType &dist);

    /*
     * @brief Prepare the NNDescent object for querying.
     *
     * This function is invoked the first time 'query' is called to construct
     * a 'search_tree' and a 'pruned search_graph'.
     */
    template<class DistType>
    void prepare(const DistType &dist);

    /*
     * @brief Query the training data for the k nearest neighbors.
     *
     * This function is called by the main 'query' function with the appropriate
     * types for MatrixType and DistType. It performs the actual querying of the
     * training data for the k nearest neighbors of the provided query points.
     * The indices and distances of the nearest neighbors are saved in the
     * respective member variables: query_indices and query_distances.
     *
     * @param train_data Matrix of training data points.
     * @param query_data Matrix of query points.
     * @param dist Distance metric object.
     * @param k Number of nearest neighbors to return.
     * @param epsilon Controls the trade-off between accuracy and search cost.
     * Larger values produce more accurate results at larger computational
     * cost. Values should be in the range 0.0 to 0.5, but typically not exceed
     * 0.3 without good reason.
     */
    template<class MatrixType, class DistType>
    void query(
        const MatrixType &train_data,
        const MatrixType &query_data,
        DistType &dist,
        int k,
        float epsilon
    );

    /*
     * @brief Perform k-nearest neighbors search using brute force for query
     * data.
     */
    template<class MatrixType, class DistType>
    void query_brute_force(
        const MatrixType &train_data,
        const MatrixType &query_data,
        const DistType &dist,
        int k
    );

    /*
     * @brief Get the data matrix.
     *
     * This function returns a pointer to the data matrix. The type of the matrix
     * returned depends on whether the data is dense or sparse. If the data is dense,
     * the function returns a pointer to the dense data matrix. If the data is sparse,
     * the function returns a pointer to the compressed sparse row (CSR) data matrix.
     *
     * @tparam MatrixType The type of the data matrix (dense or sparse).
     *
     * @return A pointer to the data matrix.
     */
    template<class MatrixType>
    MatrixType* get_data();

public:

    /**
     * The metric to use for computing nearest neighbors. Default is
     * 'euclidean'. Supported metrics include:
     *     - 'euclidean'
     *     - 'manhattan'
     *     - 'chebyshev'
     *     - 'canberra'
     *     - 'braycurtis'
     *     - 'seuclidean'
     *     - 'cosine'
     *     - 'correlation'
     *     - 'haversine'
     *     - 'hamming'
     *     - 'hellinger'
     *
     * Implemented but with limited functionality (currently only float input
     * data is possible; CAVE float comparison).
     *     - 'dice'
     *     - 'jaccard'
     *     - 'kulsinski'
     *     - 'rogerstanimoto'
     *     - 'russelrao'
     *     - 'matching'
     *     - 'sokalmichener'
     *     - 'sokalsneath'
     *     - 'yule'
     */
    std::string metric;

    /**
     * Argument to pass on to the metric such as the 'p' value for Minkowski
     * distance.
     */
    float p_metric;

    /**
     * The number of neighbors to use in the k-neighbor graph data structure
     * used for fast approximate nearest neighbor search. Larger values will
     * result in more accurate search results at the cost of computation time.
     */
    int n_neighbors;

    /**
     * This implementation uses random projection forests for initializing the
     * index build process. This parameter controls the number of trees in that
     * forest. A larger number will result in more accurate neighbor
     * computation at the cost of performance. The default of NONE means a
     * value will be chosen based on the size of the graph_data.
     */
    int n_trees;

    /**
     * The maximum number of points in a leaf for the random projection trees.
     * The default of NONE means a value will be chosen based on 'n_neighbors'.
     */
    int leaf_size;

    /**
     * This parameter determines how aggressively the search graph is pruned.
     * Since the search graph is the result of merging the nearest neighbors
     * with the reverse nearest neighbors, vertices can have a very high
     * degree. The graph will be pruned such that no vertex has a degree
     * greater than 'pruning_degree_multiplier * n_neighbors'.
     */
    float pruning_degree_multiplier;

    /**
     *  The search graph gets pruned by removing potentially unnecessary edges.
     *  This parameter controls the volume of edges removed. A value of 0.0
     *  ensures that no edges get removed, and larger values result in
     *  significantly more aggressive edge removal. A value of 1.0 will prune
     *  all edges that it can. Default is 1.0.
     */
    float pruning_prob;

    /**
     * Whether to use random projection trees for initialization.  Default is
     * true.
     */
    bool tree_init;

    /**
     * The random seed. Default is NONE.
     */
    int seed;

    /**
     * Internally each "self-join" keeps a maximum number of candidates
     * (nearest neighbors and reverse nearest neighbors) to be considered. This
     * value controls this aspect of the algorithm. Larger values will provide
     * more accurate search results later, but potentially at non-negligible
     * computation cost in building the index. Don't tweak this value unless
     * you know what you're doing. Default is NONE.
     */
    int max_candidates;

    /**
     * The maximum number of NN-descent iterations to perform. The NN-descent
     * algorithm can abort early if limited progress is being made, so this
     * only controls the worst case. Don't tweak this value unless you know
     * what you're doing. The default of NONE means a value will be chosen
     * based on the size of the graph_data.
     */
    int n_iters;

    /**
     * Controls the early abort due to limited progress. Larger values will
     * result in earlier aborts, providing less accurate indexes, and less
     * accurate searching. Don't tweak this value unless you know what you're
     * doing. Default is 0.001.
     */
    float delta;

    /**
     * The number of parallel threads to use. Default of NONE means that the
     * number will be determined depending on the number or cores.
     */
    int n_threads;

    /**
     * Whether to print status updates during computation. Default
     * is false.
    */
    bool verbose;

    /**
     * The algorithm to use for construction of the k-nearest-neighbors graph
     * used as a search index. Available options are 'bf' (brute force) and
     * 'nnd' (nearest neighbor descent). Default is 'nnd'.
     */
    std::string algorithm;

    /**
     * The current nearest neighbor graph.
     */
    HeapList<float> current_graph;

    /**
     * The indices of the nearest neighbors for each data entry.
     */
    Matrix<int> neighbor_indices;

    /**
     * The distances to the nearest neighbors for each data entry.
     */
    Matrix<float> neighbor_distances;

    /**
     * The indices of the nearest neighbors for each query.
     */
    Matrix<int> query_indices;

    /**
     * The distances to the nearest neighbors for each query.
     */
    Matrix<float> query_distances;

    /**
     * Default constructor. Creates an empty object.
     */
    NNDescent() {}

    /**
     * @brief Construct an instance of NNDescent.
     *
     * This constructor initializes and starts an instance of the NNDescent
     * algorithm with the given input data and parameters.
     *
     * @param train_data The input data matrix.
     * @param parms The parameters for the NNDescent algorithm.
     */
    NNDescent(Matrix<float> &train_data, Parms &parms);

    /**
     * @brief Construct an instance of NNDescent.
     *
     * This constructor initializes and starts an instance of the NNDescent
     * algorithm with the given input data and parameters.
     *
     * @param train_data The input data matrix.
     * @param parms The parameters for the NNDescent algorithm.
     */
    NNDescent(CSRMatrix<float> &train_data, Parms &parms);

    /**
     * @brief Query the training data for the k nearest neighbors.
     *
     * This function queries the training data for the k nearest neighbors of
     * the provided query points and saves the indices and distances in the
     * respective member variables 'query_indices' and 'query_distances'.
     *
     * @param query_data A matrix of points to query. It should have the
     * the same number of columns as 'self.data'.
     *
     * @param k The number of nearest neighbors to return (default: 10).
     * @param epsilon Controls the trade-off between accuracy and search cost.
     * Larger values produce more accurate results at larger computational
     * cost.  Values should be in the range 0.0 to 0.5, but typically not
     * exceed 0.3 without good reason (default: 0.1).
     */
    template<class MatrixType>
    void query(
        const MatrixType &query_data,
        int k=DEFAULT_K,
        float epsilon=DEFAULT_EPSILON
    );

    /*
     * @brief Prints a the parameters of an NNDescent object to an output
     * stream.
     */
    friend std::ostream& operator<<(std::ostream &out, const NNDescent &nnd);

};


template<class MatrixType, class DistType>
void NNDescent::query_brute_force(
    const MatrixType &train_data,
    const MatrixType &query_data,
    const DistType &dist,
    int k
)
{
    HeapList<float> query_nn(query_data.nrows(), k, FLOAT_MAX);
    ProgressBar bar(query_data.nrows(), verbose);
    #pragma omp parallel for num_threads(n_threads)
    for (size_t idx_q = 0; idx_q < query_data.nrows(); ++idx_q)
    {
        bar.show();
        for (size_t idx_t = 0; idx_t < data_size; ++idx_t)
        {
            float d = dist(train_data, idx_t, query_data, idx_q);
            query_nn.checked_push(idx_q, idx_t, d);
        }
    }
    query_nn.heapsort();
    query_indices = query_nn.indices;
    query_distances = query_nn.keys;
    correct_distances(
        dist, query_distances, query_distances
    );
}


template<class MatrixType, class DistType>
void NNDescent::start_nn(
    DistType &dist,
    bool perform_query,
    const MatrixType &query_data,
    int query_k,
    float query_epsilon
)
{
    MatrixType *data_ptr = this->get_data<MatrixType>();
    if (perform_query)
    {
        query(*data_ptr, query_data, dist, query_k, query_epsilon);
    }
    else
    {
        run_nn_descent(*data_ptr, dist);
    }
}


template<class MatrixType>
void NNDescent::set_dist_and_start_nn(
    bool perform_query,
    const MatrixType &query_data,
    int query_k,
    float query_epsilon
)
{
    // MOST IMPORTANT METRICS
    if (metric == "cosine")
    {
        Cosine dist;
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "euclidean")
    {
        Euclidean dist;
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }

// Useful for development as compile time is much shorter.
#ifdef ALL_METRICS

    // METRICS WITH NO PARAMETERS
    else if (metric == "alternative_cosine")
    {
        AltCosine dist;
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "alternative_dot")
    {
        AltDot dist;
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "alternative_jaccard")
    {
        AltJaccard dist;
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "braycurtis")
    {
        BrayCurtis dist;
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "canberra")
    {
        Canberra dist;
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "chebyshev")
    {
        Chebyshev dist;
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "dice")
    {
        Dice dist;
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "dot")
    {
        Dot dist;
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "hamming")
    {
        Hamming dist;
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "haversine")
    {
        throw_exception_if_sparse(metric, is_sparse);
        if (data_dim != 2)
        {
            throw std::invalid_argument(
                "haversine is only defined for 2 dimensional graph_data"
            );
        }
        Haversine dist;
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "hellinger")
    {
        Hellinger dist;
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "jaccard")
    {
        Jaccard dist;
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "manhattan")
    {
        Manhattan dist;
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "matching")
    {
        Matching dist;
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "sokalsneath")
    {
        SokalSneath dist;
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "spearmanr")
    {
        throw_exception_if_sparse(metric, is_sparse);
        SpearmanR dist;
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "sqeuclidean")
    {
        SqEuclidean dist;
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "true_angular")
    {
        TrueAngular dist;
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "tsss")
    {
        Tsss dist;
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }

    // METRICS WITH ONE FLOAT PARAMETER
    else if (metric == "circular_kantorovich")
    {
        throw_exception_if_sparse(metric, is_sparse);
        CircularKantorovich dist(p_metric);
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "minkowski")
    {
        Minkowski dist(p_metric);
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "wasserstein_1d")
    {
        throw_exception_if_sparse(metric, is_sparse);
        Wasserstein dist(p_metric);
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }

    // METRICS WHERE THE SPARSE VERSION NEEDS KNOWLEDGE OF THE DIMENSION
    else if (metric == "correlation")
    {
        Correlation dist(data_dim);
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "jensen_shannon")
    {
        JensenShannon dist(data_dim);
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "kulsinski")
    {
        Kulsinski dist(data_dim);
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "rogerstanimoto")
    {
        RogersTanimoto dist(data_dim);
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "russellrao")
    {
        RussellRao dist(data_dim);
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "sokalmichener")
    {
        SokalMichener dist(data_dim);
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "symmetric_kl")
    {
        SymmetriyKL dist(data_dim);
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }
    else if (metric == "yule")
    {
        Yule dist(data_dim);
        start_nn(dist, perform_query, query_data, query_k, query_epsilon);
    }

#endif // ALL_METRICS

    else
    {
        throw std::invalid_argument("Invalid metric");
    }
}


template<class MatrixType>
void NNDescent::query(
    const MatrixType &query_data,
    int k,
    float epsilon
)
{
    set_dist_and_start_nn(true, query_data, k, epsilon);
}


template<class MatrixType, class DistType>
void NNDescent::query(
    const MatrixType &train_data,
    const MatrixType &query_data,
    DistType &dist,
    int k,
    float epsilon
)
{
    // Make shure original data cannot be modified.
    MatrixType _query_data = query_data;
    if (metric == "dot")
    {
        _query_data.deep_copy();
        _query_data.normalize();
    }

    if (algorithm == "bf")
    {
        query_brute_force(train_data, _query_data, dist, k);
        return;
    }
    // Check if search_graph already prepared.
    if (search_graph.nheaps() == 0)
    {
        prepare(dist);
    }
    HeapList<float> query_nn(_query_data.nrows(), k, FLOAT_MAX);
    for (size_t i = 0; i < query_nn.nheaps(); ++i)
    {

        // Initialization
        Heap<Candidate> search_candidates;
        std::vector<int> visited(data_size, 0);
        std::vector<int> initial_candidates = search_tree.get_leaf(
            _query_data, i, rng_state
        );

        for (auto const &idx : initial_candidates)
        {
            float d = dist(train_data, idx, _query_data, i);
            // Don't need to check as indices are guaranteed to be different.
            // TODO implement push without check.
            query_nn.checked_push(i, idx, d);
            visited[idx] = 1;
            search_candidates.push({idx, d});
        }
        int n_random_samples = k - initial_candidates.size();
        for (int j = 0; j < n_random_samples; ++j)
        {
            int idx = rand_int(rng_state) % data_size;
            if (!visited[idx])
            {
                float d = dist(train_data, idx, _query_data, i);
                query_nn.checked_push(i, idx, d);
                visited[idx] = 1;
                search_candidates.push({idx, d});
            }
        }

        // Search
        Candidate candidate = search_candidates.pop();
        float distance_bound = (1.0f + epsilon) * query_nn.max(i);
        while (candidate.key < distance_bound)
        {
            for (size_t j = 0; j < search_graph.nnodes(); ++j)
            {
                int idx = search_graph.indices(candidate.idx, j);
                if (idx == NONE)
                {
                    break;
                }
                if (visited[idx])
                {
                    continue;
                }
                visited[idx] = 1;
                float d = dist(train_data, idx, _query_data, i);
                if (d < distance_bound)
                {
                    query_nn.checked_push(i, idx, d);
                    search_candidates.push({idx, d});

                    // Update bound
                    distance_bound = (1.0f + epsilon) * query_nn.max(i);
                }
            }
            // Find new nearest candidate point.
            if (search_candidates.empty())
            {
                break;
            }
            else
            {
                candidate = search_candidates.pop();
            }
        }
    }
    query_nn.heapsort();
    query_indices = query_nn.indices;
    query_distances = query_nn.keys;
    correct_distances(
        dist, query_distances, query_distances
    );
}


} // namespace nndescent
