/*
 * Tests all implemented functions of nndescent. The values are the same
 * as in the corresponding py file.
 */


#include <iostream>
#include <vector>

#include "../src/distances.h"
#include "../src/dtypes.h"

using namespace nndescent;


void test_distance
(
    const std::string& dist_name,
    Metric dist,
    const std::string& vec_name,
    std::vector<float>& v0,
    std::vector<float>& v1,
    std::vector<float>& v2
)
{
    std::cout << dist_name << "(" << vec_name << "0, " << vec_name << "1) = "
        << dist(v0.data(), v0.data() + v0.size(), v1.data()) << "\n"
        << dist_name << "(" << vec_name << "0, " << vec_name << "2) = "
        << dist(v0.data(), v0.data() + v0.size(), v2.data()) << "\n"
        << dist_name << "(" << vec_name << "1, " << vec_name << "2) = "
        << dist(v1.data(), v1.data() + v1.size(), v2.data()) << "\n";
}

void test_distance
(
    const std::string& dist_name,
    Metric_p dist,
    float p,
    const std::string& vec_name,
    std::vector<float>& v0,
    std::vector<float>& v1,
    std::vector<float>& v2
)
{
    std::cout << dist_name << "(" << vec_name << "0, " << vec_name << "1) = "
        << dist(v0.data(), v0.data() + v0.size(), v1.data(), p) << "\n"
        << dist_name << "(" << vec_name << "0, " << vec_name << "2) = "
        << dist(v0.data(), v0.data() + v0.size(), v2.data(), p) << "\n"
        << dist_name << "(" << vec_name << "1, " << vec_name << "2) = "
        << dist(v1.data(), v1.data() + v1.size(), v2.data(), p) << "\n";
}


float get_sparse_metric
(
    SparseMetric sparse_metric,
    const CSRMatrix<float> &data,
    int idx0,
    int idx1
)
{
    return sparse_metric(
        data.begin_col(idx0),
        data.end_col(idx0),
        data.begin_data(idx0),
        data.begin_col(idx1),
        data.end_col(idx1),
        data.begin_data(idx1)
    );
}


float get_sparse_metric
(
    Metric dense_metric,
    const Matrix<float> &data,
    int idx0,
    int idx1
)
{
    return dense_metric(
        data.begin(idx0), data.end(idx0), data.begin(idx1)
    );
}


float get_sparse_metric
(
    SparseMetric_p sparse_metric,
    const CSRMatrix<float> &data,
    float p_metric,
    int idx0,
    int idx1
)
{
    return sparse_metric(
        data.begin_col(idx0),
        data.end_col(idx0),
        data.begin_data(idx0),
        data.begin_col(idx1),
        data.end_col(idx1),
        data.begin_data(idx1),
        p_metric
    );
}


float get_sparse_metric
(
    Metric_p dense_metric,
    const Matrix<float> &data,
    float p_metric,
    int idx0,
    int idx1
)
{
    return dense_metric(
        data.begin(idx0), data.end(idx0), data.begin(idx1), p_metric
    );
}


void test_distance
(
    const std::string& dist_name,
    SparseMetric dist,
    CSRMatrix<float> &matrix,
    std::string vec_name=""
)
{
    for (size_t i = 0; i < matrix.nrows(); ++i)
    {
        for (size_t j = i + 1; j < matrix.nrows(); ++j)
        {
            std::cout << dist_name << "(" << vec_name << i << ", " << vec_name
                << j << ") = " << get_sparse_metric(dist, matrix, i, j) << "\n";
        }
    }
}

void test_distance
(
    const std::string& dist_name,
    SparseMetric_p dist,
    float p_metric,
    CSRMatrix<float> &matrix,
    std::string vec_name=""
)
{
    for (size_t i = 0; i < matrix.nrows(); ++i)
    {
        for (size_t j = i + 1; j < matrix.nrows(); ++j)
        {
            std::cout << dist_name << "(" << vec_name << i << ", " << vec_name
                << j << ") = " << get_sparse_metric(dist, matrix, p_metric, i, j)
                << "\n";
        }
    }
}



void test_distance
(
    const std::string& dist_name,
    Metric dist,
    Matrix<float> &matrix,
    std::string vec_name=""
)
{
    for (size_t i = 0; i < matrix.nrows(); ++i)
    {
        for (size_t j = i + 1; j < matrix.nrows(); ++j)
        {
            std::cout << dist_name << "(" << vec_name << i << ", " << vec_name
                << j << ") = " << get_sparse_metric(dist, matrix, i, j) << "\n";
        }
    }
}


void test_distance
(
    const std::string& dist_name,
    Metric_p dist,
    float p_metric,
    Matrix<float> &matrix,
    std::string vec_name=""
)
{
    for (size_t i = 0; i < matrix.nrows(); ++i)
    {
        for (size_t j = i + 1; j < matrix.nrows(); ++j)
        {
            std::cout << dist_name << "(" << vec_name << i << ", " << vec_name
                << j << ") = " << get_sparse_metric(dist, matrix, p_metric, i, j)
                << "\n";
        }
    }
}

std::vector<float> make_propability(std::vector<float> vec)
{
    float sum_vec = 0.0f;

    for (const float& element : vec)
    {
        sum_vec += std::abs(element);
    }

    std::vector<float> pvec;

    for (const float& element : vec)
    {
        pvec.push_back(std::abs(element) / sum_vec);
    }

    return pvec;
}



int main()
{
    std::vector<float> v0 = {9,5,6,7,3,2,1,0,8,-4};
    std::vector<float> v1 = {6,8,-2,3,6,5,4,-9,1,0};
    std::vector<float> v2 = {-1,3,5,1,0,0,-7,6,5,0};

    const int N = 100;
    std::vector<float> w0(N);
    std::vector<float> w1(N);
    std::vector<float> w2(N);

    std::iota(w0.begin(), w0.end(), 0);
    std::iota(w1.begin(), w1.end(), -10);
    std::iota(w2.begin(), w2.end(), 5);

    std::vector<float> x0 = {9,5};
    std::vector<float> x1 = {6,8};
    std::vector<float> x2 = {-1,3};

    std::vector<float> y0 = {3,-4};
    std::vector<float> y1 = {-8,8};
    std::vector<float> y2 = {9,4};

    std::vector<float> pv0 = make_propability(v0);
    std::vector<float> pv1 = make_propability(v1);
    std::vector<float> pv2 = make_propability(v2);

    std::vector<float> pw0 = make_propability(w0);
    std::vector<float> pw1 = make_propability(w1);
    std::vector<float> pw2 = make_propability(w2);


    std::vector<float> data_V = {
        -7, 2,  0, 3, 0, 0, -1, 2, 0, 0, 1, 0, 2, -1, 2, 0, 0, 1, 0, 0,
         0, 3,  1, 2, 0, 0,  0, 0, 0, 0, 0, 1, 0,  3, 1, 0, 0, 2, 0, 2,
         0, 1, -1, 1, 0, 0,  0, 5, 0, 0, 0, 0, 0, -4, 7, 5, 9, 1, 1, 1
    };
    Matrix<float> V(3, data_V);
    CSRMatrix<float> SV(V);

    std::vector<float> data_P = {
         7, 2, 0, 3, 0, 0, 1, 2, 0, 0, 1, 0, 0, 1, 2, 0, 0, 1, 0, 0,
         0, 3, 1, 2, 0, 0, 5, 0, 0, 0, 0, 1, 0, 3, 1, 0, 0, 2, 0, 2,
         0, 1, 1, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 2, 4, 1, 1, 1, 1
    };
    for (auto it = data_P.begin(); it != data_P.end(); ++it)
    {
        *it /= 20.0f;
    }
    Matrix<float> P(3, data_P);
    CSRMatrix<float> SP(P);

    float p_metric = 2.0f;

    std::cout << "\n# Test distances for:\n"
        << "v0 = " << v0 << "\n"
        << "v1 = " << v1 << "\n"
        << "v2 = " << v2 << "\n"

        << "w0 = " << "[" << w0[0] << ", " << w0[1] << ", " << w0[2] << ",..., "
            << *(w0.end() - 1) << "], size = " << w0.size() << "\n"
        << "w1 = " << "[" << w1[0] << ", " << w1[1] << ", " << w1[2] << ",..., "
            << *(w1.end() - 1) << "], size = " << w1.size() << "\n"
        << "w2 = " << "[" << w2[0] << ", " << w2[1] << ", " << w2[2] << ",..., "
            << *(w2.end() - 1) << "], size = " << w2.size() << "\n"
        << "x0 = " << x0 << "\n"
        << "x1 = " << x1 << "\n"
        << "x2 = " << x2 << "\n"
        << "y0 = " << y0 << "\n"
        << "y1 = " << y1 << "\n"
        << "y2 = " << y2 << "\n"
        << "pv0 = " << pv0 << "\n"
        << "pv1 = " << pv1 << "\n"
        << "pv2 = " << pv2 << "\n"
        << "pw0 = " << "[" << pw0[0] << ", " << pw0[1] << ", " << pw0[2] << ",..., "
            << *(pw0.end() - 1) << "], size = " << pw0.size() << "\n"
        << "pw1 = " << "[" << pw1[0] << ", " << pw1[1] << ", " << pw1[2] << ",..., "
            << *(pw1.end() - 1) << "], size = " << pw1.size() << "\n"
        << "pw2 = " << "[" << pw2[0] << ", " << pw2[1] << ", " << pw2[2] << ",..., "
            << *(pw2.end() - 1) << "], size = " << pw2.size() << "\n"
        << "SV, V = " << V << "\n"
        << "SP, P = " << P << "\n"
        << "\n# Distance functions:\n\n";


    test_distance("alternative_cosine", alternative_cosine, "v", v0, v1, v2);
    test_distance("alternative_cosine", alternative_cosine, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("alternative_dot", alternative_dot, "v", v0, v1, v2);
    test_distance("alternative_dot", alternative_dot, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("alternative_jaccard", alternative_jaccard, "v", v0, v1, v2);
    test_distance("alternative_jaccard", alternative_jaccard, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("braycurtis", bray_curtis, "v", v0, v1, v2);
    test_distance("braycurtis", bray_curtis, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("canberra", canberra, "v", v0, v1, v2);
    test_distance("canberra", canberra, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("chebyshev", chebyshev, "v", v0, v1, v2);
    test_distance("chebyshev", chebyshev, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("circular_kantorovich", circular_kantorovich, 1.0f, "v", v0, v1, v2);
    test_distance("circular_kantorovich", circular_kantorovich, 1.0f, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("correlation", correlation, "v", v0, v1, v2);
    test_distance("correlation", correlation, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("cosine", cosine, "v", v0, v1, v2);
    test_distance("cosine", cosine, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("dice", dice, "v", v0, v1, v2);
    test_distance("dice", dice, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("dot", dot, "v", v0, v1, v2);
    test_distance("dot", dot, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("euclidean", euclidean, "v", v0, v1, v2);
    test_distance("euclidean", euclidean, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("hamming", hamming, "v", v0, v1, v2);
    test_distance("hamming", hamming, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("haversine", haversine, "x", x0, x1, x2);
    test_distance("haversine", haversine, "y", y0, y1, y2);
    std::cout << "\n";

    test_distance("hellinger", hellinger, "p", pv0, pv1, pv2);
    test_distance("hellinger", hellinger, "p", pw0, pw1, pw2);
    std::cout << "\n";

    test_distance("jaccard", jaccard, "v", v0, v1, v2);
    test_distance("jaccard", jaccard, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("jensen_shannon", jensen_shannon_divergence, "p", pv0, pv1, pv2);
    test_distance("jensen_shannon", jensen_shannon_divergence, "p", pw0, pw1, pw2);
    std::cout << "\n";

    test_distance("kulsinski", kulsinski, "v", v0, v1, v2);
    test_distance("kulsinski", kulsinski, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("manhattan", manhattan, "v", v0, v1, v2);
    test_distance("manhattan", manhattan, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("matching", matching, "v", v0, v1, v2);
    test_distance("matching", matching, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("minkowski", minkowski, 2.0f, "v", v0, v1, v2);
    test_distance("minkowski", minkowski, 2.0f, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("rogerstanimoto", rogers_tanimoto, "v", v0, v1, v2);
    test_distance("rogerstanimoto", rogers_tanimoto, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("russellrao", russellrao, "v", v0, v1, v2);
    test_distance("russellrao", russellrao, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("sokalmichener", sokal_michener, "v", v0, v1, v2);
    test_distance("sokalmichener", sokal_michener, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("sokalsneath", sokal_sneath, "v", v0, v1, v2);
    test_distance("sokalsneath", sokal_sneath, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("spearmanr", spearmanr, "v", v0, v1, v2);
    test_distance("spearmanr", spearmanr, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("sqeuclidean", squared_euclidean, "v", v0, v1, v2);
    test_distance("sqeuclidean", squared_euclidean, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("symmetric_kl", symmetric_kl_divergence, "pv", pv0, pv1, pv2);
    test_distance("symmetric_kl", symmetric_kl_divergence, "pw", pw0, pw1, pw2);
    std::cout << "\n";

    test_distance("true_angular", true_angular, "v", v0, v1, v2);
    test_distance("true_angular", true_angular, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("tsss", tsss, "v", v0, v1, v2);
    test_distance("tsss", tsss, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("wasserstein_1d", wasserstein_1d, 1.0f, "v", v0, v1, v2);
    test_distance("wasserstein_1d", wasserstein_1d, 1.0f, "w", w0, w1, w2);
    std::cout << "\n";

    test_distance("yule", yule, "v", v0, v1, v2);
    test_distance("yule", yule, "w", w0, w1, w2);


    std::cout << "\n\n# Test sparse distances:\n\n";

    test_distance("sparse_alternative_cosine", sparse_alternative_cosine, SV, "SV");
    test_distance("alternative_cosine", alternative_cosine, V, "V");
    std::cout << "\n";

    test_distance("sparse_alternative_dot", sparse_alternative_dot, SV, "SV");
    test_distance("alternative_dot", alternative_dot, V, "V");
    std::cout << "\n";

    test_distance("sparse_alternative_hellinger", sparse_alternative_hellinger, SP, "SP");
    test_distance("alternative_hellinger", alternative_hellinger, P, "P");
    std::cout << "\n";

    test_distance("sparse_alternative_jaccard", sparse_alternative_jaccard, SV, "SV");
    test_distance("alternative_jaccard", alternative_jaccard, V, "V");
    std::cout << "\n";

    test_distance("sparse_braycurtis", sparse_bray_curtis, SV, "SV");
    test_distance("braycurtis", bray_curtis, V, "V");
    std::cout << "\n";

    test_distance("sparse_canberra", sparse_canberra, SV, "SV");
    test_distance("canberra", canberra, V, "V");
    std::cout << "\n";

    test_distance("sparse_chebyshev", sparse_chebyshev, SV, "SV");
    test_distance("chebyshev", chebyshev, V, "V");
    std::cout << "\n";

    // test_distance("sparse_circular_kantorovich", sparse_circular_kantorovich, SV, "SV");
    test_distance("sparse_correlation", sparse_correlation, SV.ncols(), SV, "SV");
    test_distance("correlation", correlation, V, "V");
    std::cout << "\n";

    test_distance("sparse_cosine", sparse_cosine, SV, "SV");
    test_distance("cosine", cosine, V, "V");
    std::cout << "\n";

    test_distance("sparse_dice", sparse_dice, SV, "SV");
    test_distance("dice", dice, V, "V");
    std::cout << "\n";

    test_distance("sparse_dot", sparse_dot, SV, "SV");
    test_distance("dot", dot, V, "SV");
    std::cout << "\n";

    test_distance("sparse_hamming", sparse_hamming, SV, "SV");
    test_distance("hamming", hamming, V, "V");
    std::cout << "\n";

    test_distance("sparse_hellinger", sparse_hellinger, SP, "SP");
    test_distance("hellinger", hellinger, P, "P");
    std::cout << "\n";

    test_distance("sparse_jaccard", sparse_jaccard, SV, "SV");
    test_distance("jaccard", jaccard, V, "V");
    std::cout << "\n";

    test_distance("sparse_jensen_shannon", sparse_jensen_shannon_divergence,
        SV.ncols(), SP, "SP");
    test_distance("jensen_shannon", jensen_shannon_divergence, P, "P");
    std::cout << "\n";

    test_distance("sparse_kulsinski", sparse_kulsinski, V.ncols(), SV, "SV");
    test_distance("kulsinski", kulsinski, V, "V");
    std::cout << "\n";

    test_distance("sparse_manhattan", sparse_manhattan, SV, "SV");
    test_distance("manhattan", manhattan, V, "V");
    std::cout << "\n";

    test_distance("sparse_matching", sparse_matching, SV, "SV");
    test_distance("matching", matching, V, "V");
    std::cout << "\n";

    test_distance("sparse_minkowski", sparse_minkowski, p_metric, SV, "SV");
    test_distance("minkowski", minkowski, p_metric, V, "V");
    std::cout << "\n";

    test_distance("sparse_rogerstanimoto", sparse_rogers_tanimoto, SV.ncols(), SV, "SV");
    test_distance("rogerstanimoto", rogers_tanimoto, V, "V");
    std::cout << "\n";

    test_distance("sparse_russellrao", sparse_russellrao, SV.ncols(), SV, "SV");
    test_distance("russellrao", russellrao, V, "V");
    std::cout << "\n";

    test_distance("sparse_sokalmichener", sparse_sokal_michener, SV.ncols(), SV, "SV");
    test_distance("sokalmichener", sokal_michener, V, "V");
    std::cout << "\n";

    test_distance("sparse_sokalsneath", sparse_sokal_sneath, SV, "SV");
    test_distance("sokalsneath", sokal_sneath, V, "V");
    std::cout << "\n";

    // test_distance("sparse_spearmanr", sparse_spearmanr, SV, "SV");
    test_distance("sparse_squared_euclidean", sparse_squared_euclidean, SV, "SV");
    test_distance("squared_euclidean", squared_euclidean, V, "V");
    std::cout << "\n";

    test_distance("sparse_symmetric_kl", sparse_symmetric_kl_divergence,
        SP.ncols(), SP, "SP");
    test_distance("symmetric_kl", symmetric_kl_divergence, P, "P");
    std::cout << "\n";

    test_distance("sparse_true_angular", sparse_true_angular, SV, "SV");
    test_distance("true_angular", true_angular, V, "V");
    std::cout << "\n";

    test_distance("sparse_tsss", sparse_tsss, SV, "SV");
    test_distance("tsss", tsss, V, "V");
    std::cout << "\n";

    // test_distance("sparse_wasserstein_1d", sparse_wasserstein_1d, SV, "SV");
    test_distance("sparse_yule", sparse_yule, SV.ncols(), SV, "SV");
    test_distance("yule", yule, V, "V");

    return 0;
}

