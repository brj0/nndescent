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

SlowMatrix read_csv(std::string file_path)
{
    std::cout << "Reading " << file_path << "\n";
    SlowMatrix data;

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
    SlowMatrix data = read_csv(file_path);
    ttest.stop("Reading csv");

    // int k = 30;

    // Parms parms;
    // parms.data=data;
    // parms.n_neighbors=k;
    // parms.n_iters=0;
    // parms.n_trees=1;
    // parms.verbose=true;
    // parms.seed=1234;

    // NNDescent nnd = NNDescent(parms);
    // nnd.print();

    // IntMatrix nn_apx = nnd.neighbor_graph;
    // // IntMatrix nn_apx = nn_descent(data, k);
    // ttest.stop("nnd");

    // IntMatrix nn_ect = nn_brute_force(data, k);
    // ttest.stop("brute force");


    // // std::cout << "\nNNDescent\n==============\n";
    // // print(nn_apx);

    // // std::cout << "\nBrute force\n==============\n";
    // // print(nn_ect);

    // // print(data);
    // // print_map(data);
    // recall_accuracy(nn_apx, nn_ect);
}

int main()
{
    test_csv();

    // int N = 800;
    // std::vector<double> v0 (N);
    // std::vector<double> v1 (N);
    // // all_points = [0,1,2,...]
    // std::iota(v0.begin(), v0.end(), 0.0);
    // std::iota(v1.begin(), v1.end(), 1.0);

    // std::cout << "v0=" << v0[0] << " " << v0[1] << " ... " << v0[v0.size()-1] << "\n";
    // std::cout << "v1=" << v1[0] << " " << v1[1] << " ... " << v1[v1.size()-1] << "\n";

    // double d0;

    // ttest.start();
    // for (int i = 0; i < 1e5; ++i){
    // d0 = std::inner_product(v0.begin(), v0.end(), v1.begin(), 0.0);
    // }
    // ttest.stop("dot product");
    // std::cout << "dotproduct=" << d0 << "\n";


    // ttest.start();
    // double *a0 = &v0[0];
    // double *a1 = &v1[0];
    // for (int i = 0; i < 1e5; ++i){
    // d0=0;
    // for (int i = 0; i < N; ++i)
        // d0+=a0[i]*a1[i];
    // }
    // ttest.stop("dot product 2");
    // std::cout << "dotproduct=" << d0 << "\n";

    // ttest.start();
    // for (int i = 0; i < 1e5; ++i){
    // d0=dot_product(v0,v1);
    // }
    // ttest.stop("dot product 2");
    // std::cout << "dotproduct=" << d0 << "\n";
    return 0;
}

