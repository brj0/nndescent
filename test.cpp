#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <utility>

#include "dtypes.h"
#include "nnd.h"
#include "utils.h"
#include "rp_trees.h"


using namespace std::chrono;


// Global timer for debugging
Timer ttest;

Matrix<float> read_csv(std::string file_path)
{
    std::cout << "Reading " << file_path << "\n";

    std::fstream csv_file;
    csv_file.open(file_path, std::ios::in);

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
    ttest.start();
    // std::string file_path = "data/data16x2.csv";
    std::string file_path = "data/NR208x3339.csv";
    // std::string file_path = (std::string)getenv("HOME")
        // + "/Downloads/fmnist_train.csv";
    Matrix<float> data = read_csv(file_path);
    // std::cout << data;
    ttest.stop("Reading csv");

    int k = 30;

    Parms parms;
    parms.data=data;
    parms.n_neighbors=k;
    // parms.n_iters=1;
    // parms.n_trees=0;
    parms.verbose=true;
    parms.seed=1234;

    ttest.start();
    NNDescent nnd = NNDescent(parms);
    Matrix<int> nn_apx = nnd.neighbor_graph;
    ttest.stop("nnd");

    parms.algorithm="bf";
    NNDescent nnd2 = NNDescent(parms);

    // IntMatrix nn_ect = nn_brute_force(data, k);
    Matrix<int> nn_ect = nnd2.neighbor_graph;
    // Matrix<int> nn_ect = nnd.brute_force();
    ttest.stop("brute force");


    // std::cout << "\nNNDescent\n==============\n";
    // std::cout << nn_apx;

    // std::cout << "\nBrute force\n==============\n";
    // std::cout << nn_ect;

    // std::cout << data;
    // print_map(data);

    // std::cout << data << nn_apx;
    // std::cout << nnd.current_graph << nnd.current_graph.indices
              // << nnd.current_graph.keys << nnd.current_graph.flags;

    // std::cout << "BF\n" << nn_ect;
    recall_accuracy(nn_apx, nn_ect);

}

int main()
{
    test_csv();

    // IntVec v0 = {2,2,2,2,5,6,7,8};
    // IntVec v1 = {2,2,3,4,5,6,7,8};
    // IntVec v2 = {0,1,4,4,5,6,7,8};
    // IntVec v3 = {0,2,3,5,7,7,8,8};
    // std::cout << "vectors=" << v0 << "\n";
    // IntMatrix m = {v0,v1,v2,v3};
    // std::cout << "mat=" << m << "\n";

    // std::vector<float> v4 = {
        // 0,1,-2,3,
        // 1,2,2,4,
        // 0,9,2,0
    // };

    // Matrix<float> mtx(3,v4);
    // std::cout << "m=" << mtx;
    // std::cout << "distance="
        // << bray_curtis(mtx.begin(0), mtx.end(0), mtx.begin(2))
        // << "\n";

    // int N = 60000*800;

    // std::vector<float> val (N);
    // std::iota(val.begin(), val.end(), 0.0);

    // Matrix<float> data(60000,val);


    // ttest.start();
    // double d0;
    // for (int k = 0; k < 10000; ++k)
    // {
        // for (int i = 0; i < 800; ++i)
        // {
            // d0 = dist(data, 0, i);
        // }
    // }
    // ttest.stop("dot product");
    // std::cout << "dotproduct=" << d0 << "\n";

    // ttest.start();
    // double d0;
    // for (int k = 0; k < 10000; ++k)
    // {
        // for (int i = 0; i < 800; ++i)
        // {
            // // d0 = squared_euclidean(data.begin(0), data.end(0), data.begin(i));
        // }
    // }
    // ttest.stop("dot product");
    // std::cout << "dotproduct=" << d0 << "\n";


    return 0;
}

