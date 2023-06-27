/**
 * @file nnd.h
 *
 * @author Jon Brugger
 *
 * @brief Implements the Nearest Neighbor Descent algorithm for approximate
 * nearest neighbor search.
 *
 * This file contains the C++ implementation of the pynndescent library,
 * originally written by Leland McInnes, which performs approximate nearest
 * neighbor search.
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
 * currently supports dense matrices and provides implementations for a subset
 * of distance functions. The main goal is to construct a k-nearest neighbor
 * graph quickly and accurately.
 */


#pragma once

#include <functional>

#include "distances.h"
#include "dtypes.h"
#include "utils.h"
#include "rp_trees.h"


namespace nndescent
{


/**
 * @version 0.1.0
 */
const std::string PROJECT_VERSION = "0.1.0";

// Constants
const char OLD = '0';
const char NEW = '1';
const int MAX_INT = std::numeric_limits<int>::max();
const int DEFAULT_K = 10;
const float DEFAULT_EPSILON = 0.1f;


/*
 * @brief Corrects distances using a distance correction function.
 *
 * Some metrics have alternative forms that allow for faster calculations.
 * However, these alternative forms may produce slightly different distances
 * compared to the original metric. This function applies a distance correction
 * by using the provided distance correction function to adjust the distances
 * calculated using the alternative form.
 *
 * @param distance_correction The distance correction function to be applied.
 * @param in The input matrix of distances to be corrected.
 * @param out The output matrix of corrected distances.
 */
void correct_distances
(
    Function1d distance_correction,
    Matrix<float> &in,
    Matrix<float> &out
);


/**
 * @brief Performs the NN-descent algorithm for approximate nearest neighbor
 * search.
 *
 * This function applies the NN-descent algorithm to construct an approximate
 * nearest neighbor graph. It iteratively refines the graph by exploring
 * neighbor candidates and updating the graph connections based on the
 * distances between nodes. The algorithm aims to find a graph that represents
 * the nearest neighbor relationships in the data.
 *
 * @param data The input data matrix.
 * @param current_graph The initial nearest neighbor graph.
 * @param n_neighbors The desired number of neighbors for each node.
 * @param rng_state The random state used for randomization.
 * @param max_candidates The maximum number of candidate neighbors to consider
 * during exploration.
 * @param dist The metric used for distance computation.
 * @param n_iters The number of iterations to perform.
 * @param delta The value controlling the early abort.
 * @param n_threads The number of threads to use for parallelization.
 * @param verbose Flag indicating whether to print progress and diagnostic
 * messages.
 */
void nn_descent
(
    const Matrix<float> &data,
    HeapList<float> &current_graph,
    int n_neighbors,
    RandomState &rng_state,
    int max_candidates,
    int n_iters,
    float delta,
    bool verbose
);


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
     * The distance metric used for computing distances between points.
     */
    // DistanceFunction* dist_fct;
    Distance dist;

    /*
     * The variables the distance metric 'dist_fct' points to. Some metric
     * distances are modifiable via parameters, therefore multiple types are
     * necessary.
     */
    DistanceFunction _dist_DF;
    DistanceFunction_p _dist_DFp;
    DistanceFunction__p _dist_DF_p;

    /*
     * The function used for distance correction if an alternative metric is
     * used.
     */
    Function1d distance_correction=nullptr;

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

    bool is_sparse;

    /*
     * Private member function to set 'dist' and 'distance_correction' from the
     * string 'metric'.
     */
    void get_distance_function();

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
     * Argument to pass on to the metric
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
     * The default of NONE means a value will be chosen based on n_neighbors.
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


    /*
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
    * The algorithm to use for construction of the k-neighbors
    * graph used as a search index. Available options are 'bf'
    * (brute force) and 'nnd' (nearest neighbor descent). Default is 'nnd'.
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
    * @param input_data The input data matrix.
    *
    * @param parms The parameters for the NNDescent algorithm.
    */
    NNDescent(Matrix<float> &input_data, Parms &parms);


    NNDescent(CSRMatrix<float> &input_data, Parms &parms);


    /**
     * @brief Set the parameters for NNDescent.
     *
     * This function allows setting the parameters for the NNDescent algorithm.
     *
     * @param parms The parameters to be set.
     */
    void set_parameters(Parms &parms);


    /**
     * @brief Start the NNDescent algorithm.
     *
     * This function starts the NNDescent algorithm by performing the necessary
     * computations to build the k-nearest neighbor graph. It iteratively
     * refines the graph until convergence based on the specified parameters.
     * After calling this function, the graph can be accessed through the
     * current_graph member variable.
     *
     * Note: The algorithm requires that the constructor has been called with
     * the appropriate parameters and input data before calling this function.
     */
    void start();


    void start_sparse();

    /**
     * @brief Perform k-nearest neighbors search using brute force for
     * debugging purposes.
     *
     * This function performs k-nearest neighbors search using brute force
     * method, which involves computing distances between all pairs of points
     * in the data.  It is used for debugging purposes to compare the results
     * with the approximate nearest neighbors obtained by the NNDescent
     * algorithm.
     *
     */
    template<class MatrixType>
    void start_brute_force(const MatrixType &mtx_data);


    /**
    * @brief Prepare the NNDescent object for querying.
    *
    * This function is invoked the first time 'query()' is called to construct
    * a 'search_tree' and a 'pruned search_graph'.
    */
    void prepare();


    /**
     * @brief Query the training data for the k nearest neighbors.
     *
     * This function queries the training data for the k nearest neighbors of
     * the provided query points and saves the indices and distances in the
     * respective member variables query_indices and query_distances.
     *
     * @param input_query_data A matrix of points to query. It should have the shape
     * (input_query_data.size(), self.data.ncols()).
     *
     * @param k The number of nearest neighbors to return (default: 10).
     *
     * @param epsilon Controls the trade-off between accuracy and search cost.
     * Larger values produce more accurate results at larger computational cost.
     * Values should be in the range 0.0 to 0.5, but typically not exceed 0.3
     * without good reason (default: 0.1).
     */
    template<class MatrixType>
    void query(
        const MatrixType &input_query_data,
        int k=DEFAULT_K,
        float epsilon=DEFAULT_EPSILON
    );


    /**
     * @brief Perform k-nearest neighbors search using brute force for query
     * data.
     *
     * This function performs k-nearest neighbors search using brute force
     * method for a given query data. It computes distances between each query
     * point and all points in the data to find their k-nearest neighbors and
     * saves the indices and distances in the respective member variables
     * query_indices and query_distances. It is used for debugging purposes to
     * compare the results with the approximate nearest neighbors obtained by
     * the NNDescent algorithm.
     *
     * @param query_data The query data for which k-nearest neighbors need to
     * be found.
     *
     * @param k The number of nearest neighbors to be returned.
     */
    template<class MatrixType>
    void query_brute_force(const MatrixType &query_data, int k);


    /*
     * @brief Prints a the parameters of an NNDescent object to an output
     * stream.
     */
    friend std::ostream& operator<<(std::ostream &out, const NNDescent &nnd);


    template<class MatrixType>
    MatrixType* get_data();
};


template<class MatrixType>
void NNDescent::query_brute_force(const MatrixType &query_data, int k)
{
    MatrixType *data_ptr = this->get_data<MatrixType>();
    HeapList<float> query_nn(query_data.nrows(), k, FLOAT_MAX);
    ProgressBar bar(query_data.nrows(), verbose);
    #pragma omp parallel for num_threads(n_threads)
    for (size_t idx_q = 0; idx_q < query_data.nrows(); ++idx_q)
    {
        bar.show();
        for (size_t idx_d = 0; idx_d < data_size; ++idx_d)
        {
            float d = (dist.get_fct())((*data_ptr), idx_d, query_data, idx_q);
            // float d = (*dist_fct)((*data_ptr), idx_d, query_data, idx_q);
            query_nn.checked_push(idx_q, idx_d, d);
        }
    }
    query_nn.heapsort();
    query_indices = query_nn.indices;
    query_distances = query_nn.keys;
    correct_distances(
        distance_correction, query_distances, query_distances
    );
}


template<class MatrixType>
void NNDescent::query
(
    const MatrixType &input_query_data,
    int k,
    float epsilon
)
{
    MatrixType *data_ptr = this->get_data<MatrixType>();
    // Make shure original data cannot be modified.
    MatrixType query_data = input_query_data;
    if (metric == "dot")
    {
        query_data.deep_copy();
        query_data.normalize();
    }

    if (algorithm == "bf")
    {
        query_brute_force(query_data, k);
        return;
    }
    // Check if search_graph already prepared.
    if (search_graph.nheaps() == 0)
    {
        prepare();
    }
    HeapList<float> query_nn(query_data.nrows(), k, FLOAT_MAX);
    for (size_t i = 0; i < query_nn.nheaps(); ++i)
    {

        // Initialization
        Heap<Candidate> search_candidates;
        std::vector<int> visited(data_size, 0);
        std::vector<int> initial_candidates = search_tree.get_leaf(
            // query_data.begin(i), rng_state
            query_data, i, rng_state
        );

        for (auto const &idx : initial_candidates)
        {
            float d = (dist.get_fct())((*data_ptr), idx, query_data, i);
            // float d = (*dist_fct)((*data_ptr), idx, query_data, i);
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
                float d = (dist.get_fct())((*data_ptr), idx, query_data, i);
                // float d = (*dist_fct)((*data_ptr), idx, query_data, i);
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
                float d = (dist.get_fct())((*data_ptr), idx, query_data, i);
                // float d = (*dist_fct)((*data_ptr), idx, query_data, i);
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
        distance_correction, query_distances, query_distances
    );
}


} // namespace nndescent
