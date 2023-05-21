#include <chrono>
#include <fstream>
#include <iostream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include <utility>

#include "../src/dtypes.h"
#include "../src/nnd.h"
#include "../src/utils.h"
#include "../src/rp_trees.h"
#include "../src/distances.h"


using namespace std::chrono;


// Global timer for debugging
Timer test_timer;

Matrix<float> read_csv(std::string file_path)
{
    std::cout << "Reading " << file_path << "\n";

    std::fstream csv_file;
    csv_file.open(file_path, std::ios::in);

    if (!std::ifstream(file_path))
    {
        std::cerr << "Dataset '" << file_path
                << "' not found. Did you try to run 'make_test_data.py'?\n";
        exit(1);
    }

    size_t n_rows = 0;
    std::vector<float> vec_data;
    std::string line;
    while (std::getline(csv_file, line))
    {
        ++n_rows;
        std::vector<float> row;
        std::stringstream ss_line(line);
        while (ss_line.good())
        {
            std::string substr;
            getline(ss_line, substr, ',');
            vec_data.push_back(atof(substr.c_str()));
        }
    }
    csv_file.close();

    Matrix<float> matrix(n_rows, vec_data);

    return matrix;
}





void test_csv()
{
    test_timer.start();
    std::string data_dir = (std::string)getenv("HOME")
        + "/Downloads/nndescent_test_data/";

    std::string file_path = data_dir + "simple.csv";
    // std::string file_path = data_dir + "coil20.csv";
    // std::string file_path = data_dir + "fmnist_train.csv";
    // std::string file_path = data_dir + "mnist_train.csv";

    Matrix<float> data = read_csv(file_path);
    test_timer.stop("Reading csv");

    int k = 5;

    Parms parms;
    parms.n_neighbors=k;
    parms.verbose=true;
    parms.seed=1234;
    parms.n_threads = 1;
    // parms.n_iters=0;
    // parms.n_trees=72;


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

    nnd.init_search_graph();
}

// class MetricWrapper
// {
    // public:
        // MetricWrapper(float p)
            // : p_minkowski(p)
            // {}
        // float p_minkowski;
        // static float minkowski_wrapper(It first0, It last0, It first1)
        // {
            // return minkowski<It, It>(first0, last0, first1, p_minkowski);
        // }
        // Metric get_minkowski()
        // {
            // return &minkowski_wrapper;
        // }
// };
    // Metric d_minkowski = metric_wrapper.minkowski();


int main()
{
    test_csv();


    return 0;
}

