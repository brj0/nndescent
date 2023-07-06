/*
 * ANN Benchmark example of dimension 60'000x784 (train) and 10'000x784 (test)
 */


#include <fstream>

#include "../src/nnd.h"

using namespace nndescent;


// Timer for returning passed time in milliseconds.
Timer test_timer;


// Read csv from disk and return as Matrix.
template <class T>
Matrix<T> read_csv(std::string file_path)
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
    std::vector<T> vec_data;
    std::string line;
    while (std::getline(csv_file, line))
    {
        ++n_rows;
        std::stringstream ss_line(line);
        while (ss_line.good())
        {
            std::string substr;
            getline(ss_line, substr, ',');
            vec_data.push_back(atof(substr.c_str()));
        }
    }
    csv_file.close();

    Matrix<T> matrix(n_rows, vec_data);

    return matrix;
}


int main()
{
    // DATA INPUT

    // Reset the timer to 0.
    test_timer.start();

    std::string train_path = (std::string)getenv("HOME")
        + "/Downloads/nndescent_test_data/fmnist_train.csv";

    std::string test_path = (std::string)getenv("HOME")
        + "/Downloads/nndescent_test_data/fmnist_test.csv";

    std::string test_ect_path = (std::string)getenv("HOME")
        + "/Downloads/nndescent_test_data/fmnist_test_ect.csv";

    // Read data from csv file.
    Matrix<float> data = read_csv<float>(train_path);
    Matrix<float> query_data = read_csv<float>(test_path);
    Matrix<int> query_result = read_csv<int>(test_ect_path);

    test_timer.stop("Reading csv");


    // NEAREST NEIGHBORS - TRAINING

    // Define algorithm parameters
    Parms parms;
    parms.n_neighbors=30;
    parms.verbose=true;
    parms.seed=1234;

    test_timer.start();

    // Run nearest neighbor descent algorithm.
    NNDescent nnd = NNDescent(data, parms);

    test_timer.stop("nnd");

    // Return approximate NN-Matrix and distances.
    Matrix<int> nn_indices = nnd.neighbor_indices;
    Matrix<float> nn_distances = nnd.neighbor_distances;

    test_timer.start();


    // NEAREST NEIGHBORS - TESTING

    const int k_query = 10;
    test_timer.start();

    // Calculate k_query-nearest neighbors for each query point
    nnd.query(query_data, k_query);

    test_timer.stop("nnd query");

    // Return approximate NN-Matrix and distances.
    Matrix<int> nn_query_indices = nnd.query_indices;
    Matrix<float> nn_query_distances = nnd.query_distances;

    // Get precomputed exact values of k_query-NN
    Matrix<int> nn_query_indices_ect(
        query_result.nrows(), k_query
    );

    for (size_t i = 0; i < nn_query_indices_ect.nrows(); ++i)
    {
        for (size_t j = 0; j < k_query; ++j)
        {
            // First column of result csv is index.
            nn_query_indices_ect(i, j) = query_result(i, j + 1);
        }
    }

    // Calculate accuracy of NNDescent.
    std::cout << "Accuracy of NN query data:\n";
    recall_accuracy(nn_query_indices, nn_query_indices_ect);
    return 0;
}
