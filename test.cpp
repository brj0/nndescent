#include <iostream>
#include <chrono>
#include <numeric>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>


#include "dtypes.h"
#include "nnd.h"
#include "utils.h"


using namespace std::chrono;


// Global timer for debugging
Timer ttest;

Matrix read_csv(std::string file_path)
{
    Matrix data;

    std::fstream csv_file;
    csv_file.open(file_path, std::ios::in);

    std::string line;
    while (std::getline(csv_file, line))
    {
        std::vector<double> row;
        std::stringstream ss_line(line);
        while (ss_line.good())
        {
            std::string substr;
            getline(ss_line, substr, ',');
            row.push_back(atof(substr.c_str()));
        }
        data.push_back(row);
    }

    csv_file.close();

    return data;
}

void test_csv()
{
    ttest.start();
    std::string file_path = "data/data16x2.csv";
    // std::string file_path = "data/NR208x3339.csv";
    // std::string file_path = (std::string)getenv("HOME")
        // + "/Downloads/fmnist_train.csv";
    Matrix data = read_csv(file_path);
    ttest.stop("Reading csv");

    int k = 3;

    Parms parms;
    parms.data=data;
    parms.n_neighbors=k;
    parms.n_iters=0;
    parms.verbose=true;
    parms.seed=1234;
    NNDescent nnd = NNDescent(parms);
    // nnd.print();

    IntMatrix nn_apx = nnd.neighbor_graph;
    // IntMatrix nn_apx = nn_descent(data, k);
    ttest.stop("nnd");

    IntMatrix nn_ect = nn_brute_force(data, k);
    ttest.stop("brute force");


    std::cout << "\nNNDescent\n==============\n";
    print(nn_apx);

    std::cout << "\nBrute force\n==============\n";
    print(nn_ect);

    print(data);
    print_map(data);
    recall_accuracy(nn_apx, nn_ect);

}


int main()
{
    test_csv();

    return 0;
}

