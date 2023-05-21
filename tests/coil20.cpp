#include <fstream>

#include "../src/nnd.h"


// Timer for returning passed time in milliseconds.
Timer test_timer;

// Read csv from disk and return as Matrix.
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
    // Reset the timer to 0.
    test_timer.start();

    std::string file_path = (std::string)getenv("HOME")
        + "/Downloads/nndescent_test_data/coil20.csv";

    // Read data from csv file.
    Matrix<float> data = read_csv(file_path);

    test_timer.stop("Reading csv");

    Parms parms;
    parms.n_neighbors=30;
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

    return 0;
}

