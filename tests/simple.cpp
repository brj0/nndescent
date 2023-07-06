/*
 * Very simple dataset to illustrate the functionality of the library.
 */


#include <vector>

#include "../src/nnd.h"

using namespace nndescent;


int main()
{
    // NEAREST NEIGHBORS - TRAINING

    // Construct data from row major order vector.
    std::vector<float> values =
    {
        1, 10,
        17, 5,
        59, 5,
        60, 5,
        9, 13,
        60, 13,
        17, 19,
        54, 19,
        52, 20,
        54, 22,
        9, 25,
        70, 28,
        31, 31,
        9, 32,
        52, 32,
        67, 33
    };
    Matrix<float> data(16, values);

    // Print data
    std::cout << "\nInput data\n" << data;

    // Define algorithm parameters
    Parms parms;
    parms.n_neighbors=4;

    // Run nearest neighbor descent algorithm.
    NNDescent nnd = NNDescent(data, parms);

    // Return approximate NN-Matrix and distances.
    Matrix<int> nn_indices = nnd.neighbor_indices;
    Matrix<float> nn_distances = nnd.neighbor_distances;

    // Use brute force algorithm for exact values.
    parms.algorithm="bf";
    NNDescent nnd_bf = NNDescent(data, parms);

    // Return exact NN-Matrix.
    Matrix<int> nn_indices_ect = nnd_bf.neighbor_indices;
    Matrix<float> nn_distances_ect = nnd_bf.neighbor_distances;

    // Print NN-graphs
    std::cout << "\nNearest neighbor graph indices\n" << nn_indices
        << "\nExact nearest neighbor graph indices\n" << nn_indices_ect
        << "\nNearest neighbor graph distances\n" << nn_distances
        << "\nExact nearest neighbor graph distances\n" << nn_distances_ect;


    // NEAREST NEIGHBORS - TESTING

    // Construct query data from row major order vector.
    std::vector<float> query_values =
    {
        5, 34,
        65, 12,
        44, 0,
        18, 16,
        52, 19,
        35, 9,
        1, 9,
    };
    Matrix<float> query_data(7, query_values);

    // Print query data
    std::cout << "\nInput query data\n" << query_data;

    // Calculate 6-nearest neighbors for each query point
    nnd.query(query_data, 6);

    // Return approximate NN-Matrix and distances.
    Matrix<int> nn_query_indices = nnd.query_indices;
    Matrix<float> nn_query_distances = nnd.query_distances;

    // Same using brute force algorithm.
    nnd_bf.query(query_data, 6);
    Matrix<int> nn_query_indices_ect = nnd_bf.query_indices;
    Matrix<float> nn_query_distances_ect = nnd_bf.query_distances;

    // Print NN-graphs for query points
    std::cout << "\nApproximate query NN graph:\n" << nn_query_indices
        << "\nExact NN query graph:\n" << nn_query_indices_ect
        << "\nApproximate query NN graph distances:\n" << nn_query_distances
        << "\nExact NN query graph distances:\n" << nn_query_distances_ect;


    // DEBUG FEATURES

    // What follows is useful for debugging
    // Calculate accuracy of NNDescent.
    std::cout << "Accuracy of NN:\n";
    recall_accuracy(nn_indices, nn_indices_ect);

    std::cout << "Accuracy of NN query data:\n";
    recall_accuracy(nn_query_indices, nn_query_indices_ect);

    // Print current NN graph (internal representation of data):
    std::cout << "\nCurrent NN graph:\n" << nnd.current_graph
        << "\n Current NN graph indices:\n" << nnd.current_graph.indices
        << "\n Current NN graph keys/distances:\n"  << nnd.current_graph.keys
        << "\n Current NN graph flags:\n" << nnd.current_graph.flags;

    // Print 2d data as map
    std::cout <<
        "\nData as 2d map (last digit of data point no. printed on map):\n";
    print_map(data);

    return 0;
}
