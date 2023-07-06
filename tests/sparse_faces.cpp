/*
 * Olivetty faces is a 400x4096 image dataset useful for quick checking and
 * debugging.
 */


#include <fstream>

#include "../src/nnd.h"

using namespace nndescent;


// Timer for returning passed time in milliseconds.
Timer test_timer;


// Read csv from disk and return as nndescent::Matrix.
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


int main()
{
    // DATA INPUT

    // Reset the timer to 0.
    test_timer.start();

    std::string train_path = (std::string)getenv("HOME")
        + "/Downloads/nndescent_test_data/faces_train.csv";

    std::string test_path = (std::string)getenv("HOME")
        + "/Downloads/nndescent_test_data/faces_test.csv";

    // Read data from csv file.
    Matrix<float> data = read_csv(train_path);
    Matrix<float> query_data = read_csv(test_path);

    // Convert data to csr form.
    CSRMatrix<float> csr_data(data);
    CSRMatrix<float> csr_query_data(query_data);

    test_timer.stop("Reading csv");


    // NEAREST NEIGHBORS - TRAINING

    // Define algorithm parameters
    Parms parms;
    parms.n_neighbors=30;
    parms.verbose=true;
    parms.seed=1234;

    test_timer.start();

    // Run nearest neighbor descent algorithm.
    NNDescent nnd = NNDescent(csr_data, parms);

    test_timer.stop("nnd");

    // Return approximate NN-Matrix and distances.
    Matrix<int> nn_indices = nnd.neighbor_indices;
    Matrix<float> nn_distances = nnd.neighbor_distances;

    test_timer.start();

    // Use brute force algorithm for exact values.
    parms.algorithm="bf";
    NNDescent nnd_bf = NNDescent(csr_data, parms);

    test_timer.stop("brute force");

    // Return exact NN-Matrix.
    Matrix<int> nn_indices_ect = nnd_bf.neighbor_indices;
    Matrix<float> nn_distances_ect = nnd_bf.neighbor_distances;

    // Calculate accuracy of NNDescent.
    std::cout << "Accuracy of NN:\n";
    recall_accuracy(nn_indices, nn_indices_ect);


    // NEAREST NEIGHBORS - TESTING

    const int k_query = 10;
    test_timer.start();

    // Calculate k_query-nearest neighbors for each query point
    nnd.query(csr_query_data, k_query);

    test_timer.stop("nnd query");

    // Return approximate NN-Matrix and distances.
    Matrix<int> nn_query_indices = nnd.query_indices;
    Matrix<float> nn_query_distances = nnd.query_distances;

    test_timer.start();

    // Same using brute force algorithm.
    nnd_bf.query(csr_query_data, k_query);

    test_timer.stop("brute force query");

    // Return approximate NN-Matrix and distances.
    Matrix<int> nn_query_indices_ect = nnd_bf.query_indices;
    Matrix<float> nn_query_distances_ect = nnd_bf.query_distances;

    // Calculate accuracy of NNDescent.
    std::cout << "Accuracy of NN query csr_data:\n";
    recall_accuracy(nn_query_indices, nn_query_indices_ect);

    return 0;
}
