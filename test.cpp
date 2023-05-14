#include <chrono>
#include <fstream>
#include <iostream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include <utility>

#include "dtypes.h"
#include "nnd.h"
#include "utils.h"
#include "rp_trees.h"
#include "distances.h"


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
    parms.n_neighbors=k;
    parms.metric="cosine";
    // parms.n_iters=0;
    // parms.n_trees=72;
    parms.verbose=true;
    parms.seed=1234;

    ttest.start();
    NNDescent nnd = NNDescent(data, parms);
    Matrix<int> nn_apx = nnd.neighbor_graph;
    ttest.stop("nnd");

    parms.algorithm="bf";
    NNDescent nnd_bf = NNDescent(data, parms);
    Matrix<int> nn_ect = nnd_bf.neighbor_graph;
    recall_accuracy(nn_apx, nn_ect);
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

    std::vector<float> v0 = {2,2,-1,2,7,0,0,1,-5,10};
    std::vector<float> v1 = {2,0,3,4,-5,-6,1,0,1,-2};
    std::vector<float> v2 = {5,0,1,2,-9,0,0,0,1,-1,5};
    std::vector<float> v3 = {2,2, 10, 5, 6, 7};
    std::vector<float> v4 = {1,0};
    std::vector<float> w0 = {0,0,1.0/3,2.0/3};
    std::vector<float> w1 = {1.0/8,0,3.0/8,0.5};
    std::vector<float> w(10);
    std::vector<float> mtx_val ({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15});
    Matrix<float> mtx(4, mtx_val);
    std::vector<float> x = {8,2,19,1};
    std::vector<float> y = {-2,0,3,8};

    float d = spearmanr(v0.begin(), v0.end(), v0.begin());
    correct_alternative_hellinger(v0.begin(), v0.end(), w.begin());

    std::cout << "d=" << d << "\n";
    std::cout << "w=" << w << "\n";
    float m = median(w0);
    std::cout << "med0=" << m << "\n";

    std::cout << "mtx=" << mtx << "\n";
    float e = mahalanobis(x.begin(), x.end(), y.begin(), mtx.begin(0));
    std::cout << "e=" << e << "\n";

    auto v = rankdata(v3.begin(), v3.end(), "min");
    std::cout << "v3=" << v <<"\n";




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
        // << jaccard(mtx.begin(0), mtx.end(0), mtx.begin(2))
        // << "\n";

    // int N = 60000*800;

    // std::vector<float> val (N);
    // std::iota(val.begin(), val.end(), 0.0);

    // Matrix<float> data(60000,val);


    // double d0;

    // Metric fun;
    // // typedef float (*MetricPtr)(It, It, It);
    // using MetricPtr = float (*)(It, It, It);
    // // float (*dist_ptr)(It, It, It);
    // MetricPtr dist_ptr;
    // dist_ptr = squared_euclidean<It,It>;

    // fun = squared_euclidean<It,It>;

    // ttest.start();
    // for (int k = 0; k < 10000; ++k)
    // {
        // for (int i = 0; i < 800; ++i)
        // {
            // // d0 = _dist(data, 0, i);
            // // d0 = squared_euclidean(data.begin(0), data.end(0), data.begin(i));
            // // d0 = dist_ptr(data.begin(0), data.end(0), data.begin(i));
            // // d0 = (*dist_ptr)(data.begin(0), data.end(0), data.begin(i));
            // // d0 = fun(data.begin(0), data.end(0), data.begin(i));
            // d0 = my_dist(data.begin(0), data.end(0), data.begin(i));
        // }
    // }
    // ttest.stop("dot product");
    // std::cout << "dotproduct=" << d0 << "\n";


    // ttest.start();
    // for (int k = 0; k < 10000; ++k)
    // {
        // for (int i = 0; i < 800; ++i)
        // {
            // d0 = _dist(data, 0, i);
            // d0 = squared_euclidean(data.begin(0), data.end(0), data.begin(i));

            // d0 = dist_ptr(data.begin(0), data.end(0), data.begin(i));
            // d0 = (*dist_ptr)(data.begin(0), data.end(0), data.begin(i));
            // d0 = fun(data.begin(0), data.end(0), data.begin(i));
        // }
    // }
    // ttest.stop("dot product");
    // std::cout << "dotproduct=" << d0 << "\n";

    return 0;
}

