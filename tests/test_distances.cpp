/*
 * Tests all implemented functions of nndescent. The values are the same
 * as in the corresponding py file.
 */


#include <iostream>
#include <vector>

#include "../src/distances.h"
#include "../src/dtypes.h"

using namespace nndescent;


template<class MatrixType, class DistType>
void _test_distance(
    const std::string& dist_name,
    DistType &dist,
    const std::string &vec_name,
    MatrixType &matrix
)
{
    for (size_t i = 0; i < matrix.nrows(); ++i)
    {
        for (size_t j = i + 1; j < matrix.nrows(); ++j)
        {
            std::cout << dist_name << "(" << vec_name << i << ", " << vec_name
                << j << ") = " << dist(matrix, i, j) << "\n";
        }
    }
}


template<class DistType>
void test_distance(
    const std::string& dist_name,
    DistType &dist,
    const std::string &vec_name,
    Matrix<float> &matrix,
    bool no_sparse=false
)
{
    _test_distance(dist_name, dist, vec_name, matrix);
    CSRMatrix<float> csr_matrix(matrix);
    if (no_sparse)
    {
        std::cout << "\n";
        return;
    }
    _test_distance(
        std::string("sparse_") + dist_name,
        dist,
        vec_name,
        csr_matrix
    );
    std::cout << "\n";
}


void test_all_distances(
    const std::string& name, Matrix<float> mtx, float p_metric
)
{
    // Convert mtx to probability matrix
    Matrix<float> mtx_prob = mtx;
    for (size_t i = 0; i < mtx_prob.nrows(); ++i)
    {
        float sum = 0.0f;
        for (size_t j = 0; j < mtx_prob.ncols(); ++j)
        {
            mtx_prob(i, j) = std::abs(mtx_prob(i, j));
            sum += mtx_prob(i, j);
        }
        if (sum != 0.0f)
        {
            for (size_t j = 0; j < mtx_prob.ncols(); ++j)
            {
                mtx_prob(i, j) /= sum;
            }
        }
    }

    // 2D Matrix
    Matrix<float> mtx_2d(mtx.nrows(), 2);
    for (size_t i = 0; i < mtx_2d.nrows(); ++i)
    {
        for (size_t j = 0; j < mtx_2d.ncols(); ++j)
        {
            mtx_2d(i, j) = mtx(i, j);
        }
    }

    AltCosine d_alternative_cosine;
    AltDot d_alternative_dot;
    AltJaccard d_alternative_jaccard;
    BrayCurtis d_bray_curtis;
    Canberra d_canberra;
    Chebyshev d_chebyshev;
    Correlation d_correlation(mtx.ncols());
    Cosine d_cosine;
    Dice d_dice;
    Dot d_dot;
    Euclidean d_euclidean;
    Hamming d_hamming;
    Haversine d_haversine;
    Hellinger d_hellinger;
    Jaccard d_jaccard;
    Manhattan d_manhattan;
    Matching d_matching;
    SokalSneath d_sokalsneath;
    SpearmanR d_spearman_r;
    SqEuclidean d_sqeuclidean;
    TrueAngular d_true_angular;
    Tsss d_tsss;

    CircularKantorovich d_circular_kantorovich(p_metric);
    Minkowski d_minkowski(p_metric);
    Wasserstein d_wasserstein(p_metric);

    JensenShannon d_jensen_shannon(mtx_prob.ncols());
    Kulsinski d_kulsinski(mtx.ncols());
    RogersTanimoto d_rogerstanimoto(mtx.ncols());
    RussellRao d_russellrao(mtx.ncols());
    SokalMichener d_sokalmichener(mtx_prob.ncols());
    SymmetriyKL d_symmetric_kl(mtx.ncols());
    Yule d_yule(mtx.ncols());

    std::string name_prob = name + std::string("_prob");
    std::string name_2d = name + std::string("_2d");

    std::cout << "\n# Test distances for:\n"
        << name << " =\n" << mtx << "shape = (" << mtx.nrows()
        << ", " << mtx.ncols() << ")\n\n"
        << name_prob << " =\n" << mtx_prob << "shape = (" << mtx_prob.nrows()
        << ", " << mtx_prob.ncols() << ")\n\n"
        << name_2d << " =\n" << mtx_2d << "shape = (" << mtx_2d.nrows()
        << ", " << mtx_2d.ncols() << ")\n\n"
        << "\n# Distance functions:\n\n";

    test_distance("alternative_cosine", d_alternative_cosine, name, mtx);
    test_distance("alternative_dot", d_alternative_dot, name, mtx);
    test_distance("alternative_jaccard", d_alternative_jaccard, name, mtx);
    test_distance("braycurtis", d_bray_curtis, name, mtx);
    test_distance("canberra", d_canberra, name, mtx);
    test_distance("chebyshev", d_chebyshev, name, mtx);
    test_distance("circular_kantorovich", d_circular_kantorovich, name, mtx, true);
    test_distance("correlation", d_correlation, name, mtx);
    test_distance("cosine", d_cosine, name, mtx);
    test_distance("dice", d_dice, name, mtx);
    test_distance("dot", d_dot, name, mtx);
    test_distance("euclidean", d_euclidean, name, mtx);
    test_distance("hamming", d_hamming, name, mtx);
    test_distance("haversine", d_haversine, name_2d, mtx_2d, true);
    test_distance("hellinger", d_hellinger, name_prob, mtx_prob);
    test_distance("jaccard", d_jaccard, name, mtx);
    test_distance("jensen_shannon", d_jensen_shannon, name_prob, mtx_prob);
    test_distance("kulsinski", d_kulsinski, name, mtx);
    test_distance("manhattan", d_manhattan, name, mtx);
    test_distance("matching", d_matching, name, mtx);
    test_distance("minkowski", d_minkowski, name, mtx);
    test_distance("rogerstanimoto", d_rogerstanimoto, name, mtx);
    test_distance("russelrao", d_russellrao, name, mtx);
    test_distance("sokalmichener", d_sokalmichener, name, mtx);
    test_distance("sokalsneath", d_sokalsneath, name, mtx);
    test_distance("spearmanr", d_spearman_r, name, mtx, true);
    test_distance("sqeuclidean", d_sqeuclidean, name, mtx);
    test_distance("symmetric_kl", d_symmetric_kl, name_prob, mtx_prob);
    test_distance("true_angular", d_true_angular, name, mtx);
    test_distance("tsss", d_tsss, name, mtx);
    test_distance("wasserstein", d_wasserstein, name, mtx, true);
    test_distance("yule", d_yule, name, mtx);
}


int main()
{
    std::vector<float> data_U = {
        9,5,6,7,3,2,1,0,8,-4,
        6,8,-2,3,6,5,4,-9,1,0,
        -1,3,5,1,0,0,-7,6,5,0
    };
    Matrix<float> U(3, data_U);

    std::vector<float> data_V = {
        -7, 2,  0, 3, 0, 0, -1, 2, 0, 0, 1, 0, 2, -1, 2, 0, 0, 1, 0, 0,
         0, 3,  1, 2, 0, 0,  0, 0, 0, 0, 0, 1, 0,  3, 1, 0, 0, 2, 0, 2,
         0, 1, -1, 1, 0, 0,  0, 5, 0, 0, 0, 0, 0, -4, 7, 5, 9, 1, 1, 1
    };
    Matrix<float> V(3, data_V);

    const int N = 100;
    std::vector<float> data_W;
    unsigned int state = 0;
    for (size_t i = 0; i < 3*N; ++i)
    {
        state = ((state * 1664525) + 1013904223) % 4294967296;
        data_W.push_back((state % 13) - 3.0f);
    }
    Matrix<float> W(3, data_W);

    test_all_distances("U", U, 2.0f);
    test_all_distances("V", V, 2.0f);
    test_all_distances("W", W, 2.0f);

    return 0;
}
