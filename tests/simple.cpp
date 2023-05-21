#include <vector>

#include "../src/nnd.h"


// Timer for returning passed time in milliseconds.
Timer test_timer;

int main()
{
    // Reset the timer to 0.
    test_timer.start();

    std::vector<float> row_major_order =
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

    // Construct matrix from row major order vector.
    Matrix<float> data(16, row_major_order);

    test_timer.stop("Reading csv");

    Parms parms;
    parms.n_neighbors=3;
    parms.verbose=true;
    parms.seed=1234;

    test_timer.start();

    // Run nearest neighbor descent algorithm.
    NNDescent nnd = NNDescent(data, parms);

    // Return approximate NN-Matrix.
    Matrix<int> nn_apx = nnd.neighbor_indices;

    test_timer.stop("nnd");

    // Use brute force algorithm.
    parms.algorithm="bf";
    NNDescent nnd_bf = NNDescent(data, parms);

    // Return exact NN-Matrix.
    Matrix<int> nn_ect = nnd_bf.neighbor_indices;

    test_timer.stop("brute force");

    // Calculate accuracy of NNDescent.
    recall_accuracy(nn_apx, nn_ect);


    // What follows is useful for debugging
    // Print current NN graph:
    std::cout << "\nCurrent NN graph:\n" << nnd.current_graph
        << "\n Current NN graph indices:\n" << nnd.current_graph.indices
        << "\n Current NN graph keys/distances:\n"  << nnd.current_graph.keys
        << "\n Current NN graph flags:\n" << nnd.current_graph.flags;

    // Print NN-graphs
    std::cout << "\nApproximate NN graph:\n" << nn_apx;
    std::cout << "\nEcact NN graph:\n" << nn_ect;

    // Print data
    std::cout << "\nInput data:\n" <<  data
    << "\nData as 2d map (last digit of data point no. printed on map):\n";
    print_map(data);

    return 0;
}

