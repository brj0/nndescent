/*
 * Tests all implemented functions of nndescent. The values are the same
 * as in the corresponding py file.
 */


#include <iostream>
#include <vector>

#include "../src/distances.h"
#include "../src/dtypes.h"

using namespace nndescent;

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

    std::vector<float> p0 = {0.9,0.1,0.0};
    std::vector<float> p1 = {0.5,0.2,0.3};
    std::vector<float> p2 = {0,0.7,0.3};

    std::cout << "Test distances for:\n"
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
        << "p0 = " << p0 << "\n"
        << "p1 = " << p1 << "\n"
        << "p2 = " << p2 << "\n"
        << "\n"

        << "alternative_cosine(v0, v1) = "
            << alternative_cosine(v0.begin(), v0.end(), v1.begin()) << "\n"
        << "alternative_cosine(v0, v2) = "
            << alternative_cosine(v0.begin(), v0.end(), v2.begin()) << "\n"
        << "alternative_cosine(v1, v2) = "
            << alternative_cosine(v1.begin(), v1.end(), v2.begin()) << "\n"

        << "alternative_cosine(w0, w1) = "
            << alternative_cosine(w0.begin(), w0.end(), w1.begin()) << "\n"
        << "alternative_cosine(w0, w2) = "
            << alternative_cosine(w0.begin(), w0.end(), w2.begin()) << "\n"
        << "alternative_cosine(w1, w2) = "
            << alternative_cosine(w1.begin(), w1.end(), w2.begin()) << "\n"

        << "\n"

        << "alternative_dot(v0, v1) = "
            << alternative_dot(v0.begin(), v0.end(), v1.begin()) << "\n"
        << "alternative_dot(v0, v2) = "
            << alternative_dot(v0.begin(), v0.end(), v2.begin()) << "\n"
        << "alternative_dot(v1, v2) = "
            << alternative_dot(v1.begin(), v1.end(), v2.begin()) << "\n"

        << "alternative_dot(w0, w1) = "
            << alternative_dot(w0.begin(), w0.end(), w1.begin()) << "\n"
        << "alternative_dot(w0, w2) = "
            << alternative_dot(w0.begin(), w0.end(), w2.begin()) << "\n"
        << "alternative_dot(w1, w2) = "
            << alternative_dot(w1.begin(), w1.end(), w2.begin()) << "\n"

        << "\n"

        << "alternative_jaccard(v0, v1) = "
            << alternative_jaccard(v0.begin(), v0.end(), v1.begin()) << "\n"
        << "alternative_jaccard(v0, v2) = "
            << alternative_jaccard(v0.begin(), v0.end(), v2.begin()) << "\n"
        << "alternative_jaccard(v1, v2) = "
            << alternative_jaccard(v1.begin(), v1.end(), v2.begin()) << "\n"

        << "alternative_jaccard(w0, w1) = "
            << alternative_jaccard(w0.begin(), w0.end(), w1.begin()) << "\n"
        << "alternative_jaccard(w0, w2) = "
            << alternative_jaccard(w0.begin(), w0.end(), w2.begin()) << "\n"
        << "alternative_jaccard(w1, w2) = "
            << alternative_jaccard(w1.begin(), w1.end(), w2.begin()) << "\n"

        << "\n"

        << "bray_curtis(v0, v1) = "
            << bray_curtis(v0.begin(), v0.end(), v1.begin()) << "\n"
        << "bray_curtis(v0, v2) = "
            << bray_curtis(v0.begin(), v0.end(), v2.begin()) << "\n"
        << "bray_curtis(v1, v2) = "
            << bray_curtis(v1.begin(), v1.end(), v2.begin()) << "\n"

        << "bray_curtis(w0, w1) = "
            << bray_curtis(w0.begin(), w0.end(), w1.begin()) << "\n"
        << "bray_curtis(w0, w2) = "
            << bray_curtis(w0.begin(), w0.end(), w2.begin()) << "\n"
        << "bray_curtis(w1, w2) = "
            << bray_curtis(w1.begin(), w1.end(), w2.begin()) << "\n"

        << "\n"

        << "canberra(v0, v1) = "
            << canberra(v0.begin(), v0.end(), v1.begin()) << "\n"
        << "canberra(v0, v2) = "
            << canberra(v0.begin(), v0.end(), v2.begin()) << "\n"
        << "canberra(v1, v2) = "
            << canberra(v1.begin(), v1.end(), v2.begin()) << "\n"

        << "canberra(w0, w1) = "
            << canberra(w0.begin(), w0.end(), w1.begin()) << "\n"
        << "canberra(w0, w2) = "
            << canberra(w0.begin(), w0.end(), w2.begin()) << "\n"
        << "canberra(w1, w2) = "
            << canberra(w1.begin(), w1.end(), w2.begin()) << "\n"

        << "\n"

        << "chebyshev(v0, v1) = "
            << chebyshev(v0.begin(), v0.end(), v1.begin()) << "\n"
        << "chebyshev(v0, v2) = "
            << chebyshev(v0.begin(), v0.end(), v2.begin()) << "\n"
        << "chebyshev(v1, v2) = "
            << chebyshev(v1.begin(), v1.end(), v2.begin()) << "\n"

        << "chebyshev(w0, w1) = "
            << chebyshev(w0.begin(), w0.end(), w1.begin()) << "\n"
        << "chebyshev(w0, w2) = "
            << chebyshev(w0.begin(), w0.end(), w2.begin()) << "\n"
        << "chebyshev(w1, w2) = "
            << chebyshev(w1.begin(), w1.end(), w2.begin()) << "\n"

        << "\n"

        << "correlation(v0, v1) = "
            << correlation(v0.begin(), v0.end(), v1.begin()) << "\n"
        << "correlation(v0, v2) = "
            << correlation(v0.begin(), v0.end(), v2.begin()) << "\n"
        << "correlation(v1, v2) = "
            << correlation(v1.begin(), v1.end(), v2.begin()) << "\n"

        << "correlation(w0, w1) = "
            << correlation(w0.begin(), w0.end(), w1.begin()) << "\n"
        << "correlation(w0, w2) = "
            << correlation(w0.begin(), w0.end(), w2.begin()) << "\n"
        << "correlation(w1, w2) = "
            << correlation(w1.begin(), w1.end(), w2.begin()) << "\n"

        << "\n"

        << "cosine(v0, v1) = "
            << cosine(v0.begin(), v0.end(), v1.begin()) << "\n"
        << "cosine(v0, v2) = "
            << cosine(v0.begin(), v0.end(), v2.begin()) << "\n"
        << "cosine(v1, v2) = "
            << cosine(v1.begin(), v1.end(), v2.begin()) << "\n"

        << "cosine(w0, w1) = "
            << cosine(w0.begin(), w0.end(), w1.begin()) << "\n"
        << "cosine(w0, w2) = "
            << cosine(w0.begin(), w0.end(), w2.begin()) << "\n"
        << "cosine(w1, w2) = "
            << cosine(w1.begin(), w1.end(), w2.begin()) << "\n"

        << "\n"

        << "dice(v0, v1) = "
            << dice(v0.begin(), v0.end(), v1.begin()) << "\n"
        << "dice(v0, v2) = "
            << dice(v0.begin(), v0.end(), v2.begin()) << "\n"
        << "dice(v1, v2) = "
            << dice(v1.begin(), v1.end(), v2.begin()) << "\n"

        << "dice(w0, w1) = "
            << dice(w0.begin(), w0.end(), w1.begin()) << "\n"
        << "dice(w0, w2) = "
            << dice(w0.begin(), w0.end(), w2.begin()) << "\n"
        << "dice(w1, w2) = "
            << dice(w1.begin(), w1.end(), w2.begin()) << "\n"

        << "\n"

        << "dot(v0, v1) = "
            << dot(v0.begin(), v0.end(), v1.begin()) << "\n"
        << "dot(v0, v2) = "
            << dot(v0.begin(), v0.end(), v2.begin()) << "\n"
        << "dot(v1, v2) = "
            << dot(v1.begin(), v1.end(), v2.begin()) << "\n"

        << "dot(w0, w1) = "
            << dot(w0.begin(), w0.end(), w1.begin()) << "\n"
        << "dot(w0, w2) = "
            << dot(w0.begin(), w0.end(), w2.begin()) << "\n"
        << "dot(w1, w2) = "
            << dot(w1.begin(), w1.end(), w2.begin()) << "\n"

        << "\n"

        << "euclidean(v0, v1) = "
            << euclidean(v0.begin(), v0.end(), v1.begin()) << "\n"
        << "euclidean(v0, v2) = "
            << euclidean(v0.begin(), v0.end(), v2.begin()) << "\n"
        << "euclidean(v1, v2) = "
            << euclidean(v1.begin(), v1.end(), v2.begin()) << "\n"

        << "euclidean(w0, w1) = "
            << euclidean(w0.begin(), w0.end(), w1.begin()) << "\n"
        << "euclidean(w0, w2) = "
            << euclidean(w0.begin(), w0.end(), w2.begin()) << "\n"
        << "euclidean(w1, w2) = "
            << euclidean(w1.begin(), w1.end(), w2.begin()) << "\n"

        << "\n"

        << "hamming(v0, v1) = "
            << hamming(v0.begin(), v0.end(), v1.begin()) << "\n"
        << "hamming(v0, v2) = "
            << hamming(v0.begin(), v0.end(), v2.begin()) << "\n"
        << "hamming(v1, v2) = "
            << hamming(v1.begin(), v1.end(), v2.begin()) << "\n"

        << "hamming(w0, w1) = "
            << hamming(w0.begin(), w0.end(), w1.begin()) << "\n"
        << "hamming(w0, w2) = "
            << hamming(w0.begin(), w0.end(), w2.begin()) << "\n"
        << "hamming(w1, w2) = "
            << hamming(w1.begin(), w1.end(), w2.begin()) << "\n"

        << "\n"

        << "hellinger(p0, p1) = "
            << hellinger(p0.begin(), p0.end(), p1.begin()) << "\n"
        << "hellinger(p0, p2) = "
            << hellinger(p0.begin(), p0.end(), p2.begin()) << "\n"
        << "hellinger(p1, p2) = "
            << hellinger(p1.begin(), p1.end(), p2.begin()) << "\n"

        << "\n"

        << "haversine(x0, x1) = "
            << haversine(x0.begin(), x0.end(), x1.begin()) << "\n"
        << "haversine(x0, x2) = "
            << haversine(x0.begin(), x0.end(), x2.begin()) << "\n"
        << "haversine(x1, x2) = "
            << haversine(x1.begin(), x1.end(), x2.begin()) << "\n"

        << "\n"

        << "jaccard(v0, v1) = "
            << jaccard(v0.begin(), v0.end(), v1.begin()) << "\n"
        << "jaccard(v0, v2) = "
            << jaccard(v0.begin(), v0.end(), v2.begin()) << "\n"
        << "jaccard(v1, v2) = "
            << jaccard(v1.begin(), v1.end(), v2.begin()) << "\n"

        << "jaccard(w0, w1) = "
            << jaccard(w0.begin(), w0.end(), w1.begin()) << "\n"
        << "jaccar(w0, w2) = "
            << jaccard(w0.begin(), w0.end(), w2.begin()) << "\n"
        << "jaccar(w1, w2) = "
            << jaccard(w1.begin(), w1.end(), w2.begin()) << "\n"

        << "\n"

        << "kulsinski(v0, v1) = "
            << kulsinski(v0.begin(), v0.end(), v1.begin()) << "\n"
        << "kulsinski(v0, v2) = "
            << kulsinski(v0.begin(), v0.end(), v2.begin()) << "\n"
        << "kulsinski(v1, v2) = "
            << kulsinski(v1.begin(), v1.end(), v2.begin()) << "\n"

        << "kulsinski(w0, w1) = "
            << kulsinski(w0.begin(), w0.end(), w1.begin()) << "\n"
        << "kulsinski(w0, w2) = "
            << kulsinski(w0.begin(), w0.end(), w2.begin()) << "\n"
        << "kulsinski(w1, w2) = "
            << kulsinski(w1.begin(), w1.end(), w2.begin()) << "\n"

        << "\n"

        << "manhattan(v0, v1) = "
            << manhattan(v0.begin(), v0.end(), v1.begin()) << "\n"
        << "manhattan(v0, v2) = "
            << manhattan(v0.begin(), v0.end(), v2.begin()) << "\n"
        << "manhattan(v1, v2) = "
            << manhattan(v1.begin(), v1.end(), v2.begin()) << "\n"

        << "manhattan(w0, w1) = "
            << manhattan(w0.begin(), w0.end(), w1.begin()) << "\n"
        << "manhattan(w0, w2) = "
            << manhattan(w0.begin(), w0.end(), w2.begin()) << "\n"
        << "manhattan(w1, w2) = "
            << manhattan(w1.begin(), w1.end(), w2.begin()) << "\n"

        << "\n"

        << "matching(v0, v1) = "
            << matching(v0.begin(), v0.end(), v1.begin()) << "\n"
        << "matching(v0, v2) = "
            << matching(v0.begin(), v0.end(), v2.begin()) << "\n"
        << "matching(v1, v2) = "
            << matching(v1.begin(), v1.end(), v2.begin()) << "\n"

        << "matching(w0, w1) = "
            << matching(w0.begin(), w0.end(), w1.begin()) << "\n"
        << "matching(w0, w2) = "
            << matching(w0.begin(), w0.end(), w2.begin()) << "\n"
        << "matching(w1, w2) = "
            << matching(w1.begin(), w1.end(), w2.begin()) << "\n"

        << "\n"

        << "russellrao(v0, v1) = "
            << russellrao(v0.begin(), v0.end(), v1.begin()) << "\n"
        << "russellrao(v0, v2) = "
            << russellrao(v0.begin(), v0.end(), v2.begin()) << "\n"
        << "russellrao(v1, v2) = "
            << russellrao(v1.begin(), v1.end(), v2.begin()) << "\n"

        << "russellrao(w0, w1) = "
            << russellrao(w0.begin(), w0.end(), w1.begin()) << "\n"
        << "russellrao(w0, w2) = "
            << russellrao(w0.begin(), w0.end(), w2.begin()) << "\n"
        << "russellrao(w1, w2) = "
            << russellrao(w1.begin(), w1.end(), w2.begin()) << "\n"

        << "\n"

        << "rogers_tanimoto(v0, v1) = "
            << rogers_tanimoto(v0.begin(), v0.end(), v1.begin()) << "\n"
        << "rogers_tanimoto(v0, v2) = "
            << rogers_tanimoto(v0.begin(), v0.end(), v2.begin()) << "\n"
        << "rogers_tanimoto(v1, v2) = "
            << rogers_tanimoto(v1.begin(), v1.end(), v2.begin()) << "\n"

        << "rogers_tanimoto(w0, w1) = "
            << rogers_tanimoto(w0.begin(), w0.end(), w1.begin()) << "\n"
        << "rogers_tanimoto(w0, w2) = "
            << rogers_tanimoto(w0.begin(), w0.end(), w2.begin()) << "\n"
        << "rogers_tanimoto(w1, w2) = "
            << rogers_tanimoto(w1.begin(), w1.end(), w2.begin()) << "\n"

        << "\n"

        << "sokal_michener(v0, v1) = "
            << sokal_michener(v0.begin(), v0.end(), v1.begin()) << "\n"
        << "sokal_michener(v0, v2) = "
            << sokal_michener(v0.begin(), v0.end(), v2.begin()) << "\n"
        << "sokal_michener(v1, v2) = "
            << sokal_michener(v1.begin(), v1.end(), v2.begin()) << "\n"

        << "sokal_michener(w0, w1) = "
            << sokal_michener(w0.begin(), w0.end(), w1.begin()) << "\n"
        << "sokal_michener(w0, w2) = "
            << sokal_michener(w0.begin(), w0.end(), w2.begin()) << "\n"
        << "sokal_michener(w1, w2) = "
            << sokal_michener(w1.begin(), w1.end(), w2.begin()) << "\n"

        << "\n"

        << "sokal_sneath(v0, v1) = "
            << sokal_sneath(v0.begin(), v0.end(), v1.begin()) << "\n"
        << "sokal_sneath(v0, v2) = "
            << sokal_sneath(v0.begin(), v0.end(), v2.begin()) << "\n"
        << "sokal_sneath(v1, v2) = "
            << sokal_sneath(v1.begin(), v1.end(), v2.begin()) << "\n"

        << "sokal_sneath(w0, w1) = "
            << sokal_sneath(w0.begin(), w0.end(), w1.begin()) << "\n"
        << "sokal_sneath(w0, w2) = "
            << sokal_sneath(w0.begin(), w0.end(), w2.begin()) << "\n"
        << "sokal_sneath(w1, w2) = "
            << sokal_sneath(w1.begin(), w1.end(), w2.begin()) << "\n"

        << "\n"

        << "squared_euclidean(v0, v1) = "
            << squared_euclidean(v0.begin(), v0.end(), v1.begin()) << "\n"
        << "squared_euclidean(v0, v2) = "
            << squared_euclidean(v0.begin(), v0.end(), v2.begin()) << "\n"
        << "squared_euclidean(v1, v2) = "
            << squared_euclidean(v1.begin(), v1.end(), v2.begin()) << "\n"

        << "squared_euclidean(w0, w1) = "
            << squared_euclidean(w0.begin(), w0.end(), w1.begin()) << "\n"
        << "squared_euclidean(w0, w2) = "
            << squared_euclidean(w0.begin(), w0.end(), w2.begin()) << "\n"
        << "squared_euclidean(w1, w2) = "
            << squared_euclidean(w1.begin(), w1.end(), w2.begin()) << "\n"

        << "\n"

        << "yule(v0, v1) = "
            << yule(v0.begin(), v0.end(), v1.begin()) << "\n"
        << "yule(v0, v2) = "
            << yule(v0.begin(), v0.end(), v2.begin()) << "\n"
        << "yule(v1, v2) = "
            << yule(v1.begin(), v1.end(), v2.begin()) << "\n"

        << "yule(w0, w1) = "
            << yule(w0.begin(), w0.end(), w1.begin()) << "\n"
        << "yule(w0, w2) = "
            << yule(w0.begin(), w0.end(), w2.begin()) << "\n"
        << "yule(w1, w2) = "
            << yule(w1.begin(), w1.end(), w2.begin()) << "\n"

        << "\n"

        ;
    return 0;
}

