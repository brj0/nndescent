/**
 * @file distances.h
 *
 * @brief Distance functions.
 */


#pragma once

#include <algorithm>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "utils.h"
#include "dtypes.h"


namespace nndescent
{


const float PI = 3.14159265358979f;
const float FLOAT_MAX = std::numeric_limits<float>::max();
const float FLOAT_MIN = std::numeric_limits<float>::min();
const float FLOAT_EPS = std::numeric_limits<float>::epsilon();


// Types
using Function1d = float (*)(float);
using It = float*;
using Metric = float (*)(It, It, It);
using SparseMetric = float (*)(size_t*, size_t*, It, size_t*, size_t*, It);
using MetricP = float (*)(It, It, It, float);
using SparseMetricP = float (*)(size_t*, size_t*, It, size_t*, size_t*, It, float);


/*
 * @brief Identity function.
 */
template<class T>
T identity(T value)
{
    return value;
}


/*
 * @brief Squared euclidean distance.
 */
template<class Iter0, class Iter1>
float squared_euclidean(Iter0 first0, Iter0 last0, Iter1 first1)
{
    float result = 0.0f;
    for (; first0 != last0; ++first0, ++first1)
    {
        result = std::move(result) + (*first0 - *first1)*(*first0 - *first1);
    }
    return result;
}


/*
 * @brief Sparse squared euclidean distance.
 */
template<class IterCol, class IterData>
float sparse_squared_euclidean(
    IterCol first0,
    IterCol last0,
    IterData data0,
    IterCol first1,
    IterCol last1,
    IterData data1
)
{
    float result = 0.0f;
    // Pass through both index lists
    while(first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            result += (*data0 - *data1) * (*data0 - *data1);

            ++first0;
            ++data0;

            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            result += (*data0) * (*data0);

            ++first0;
            ++data0;
        }
        else
        {
            result += (*data1) * (*data1);

            ++first1;
            ++data1;
        }
    }
    // Pass over the tails
    while(first0 != last0)
    {
        result += (*data0) * (*data0);

        ++first0;
        ++data0;
    }
    while(first1 != last1)
    {
        result += (*data1) * (*data1);

        ++first1;
        ++data1;
    }
    return result;
}


/*
 * @brief Computes the element-wise sum of two sparse vectors.
 */
template<class IterCol, class IterData>
std::tuple<std::vector<size_t>, std::vector<float>> sparse_sum(
    IterCol first0,
    IterCol last0,
    IterData data0,
    IterCol first1,
    IterCol last1,
    IterData data1
)
{
    std::vector<size_t> result_col_ind;
    std::vector<float> result_data;

    // Pass through both index lists
    while(first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            result_col_ind.push_back(*first0);
            result_data.push_back(*data0 + *data1);

            ++first0;
            ++data0;

            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            result_col_ind.push_back(*first0);
            result_data.push_back(*data0);

            ++first0;
            ++data0;
        }
        else
        {
            result_col_ind.push_back(*first1);
            result_data.push_back(*data1);

            ++first1;
            ++data1;
        }
    }
    // Pass over the tails
    while(first0 != last0)
    {
        result_col_ind.push_back(*first0);
        result_data.push_back(*data0);

        ++first0;
        ++data0;
    }
    while(first1 != last1)
    {
        result_col_ind.push_back(*first1);
        result_data.push_back(*data1);

        ++first1;
        ++data1;
    }
    return std::make_tuple(result_col_ind, result_data);
}


/*
 * @brief Sparse inner product.
 */
template<class IterCol0, class IterCol1, class IterData0, class IterData1>
float sparse_inner_product(
    IterCol0 first0,
    IterCol0 last0,
    IterData0 data0,
    IterCol1 first1,
    IterCol1 last1,
    IterData1 data1
)
{
    float result = 0.0f;
    // Pass through both index lists
    while(first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            result += (*data0) * (*data1);

            ++first0;
            ++data0;

            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            ++first0;
            ++data0;
        }
        else
        {
            ++first1;
            ++data1;
        }
    }
    return result;
}


/*
 * @brief Computes the element-wise difference of two sparse vectors.
 */
template<class IterCol, class IterData>
std::tuple<std::vector<size_t>, std::vector<float>> sparse_diff(
    IterCol first0,
    IterCol last0,
    IterData data0,
    IterCol first1,
    IterCol last1,
    IterData data1
)
{
    std::vector<size_t> result_col_ind;
    std::vector<float> result_data;

    // Pass through both index lists
    while(first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            result_col_ind.push_back(*first0);
            result_data.push_back(*data0 - *data1);

            ++first0;
            ++data0;

            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            result_col_ind.push_back(*first0);
            result_data.push_back(*data0);

            ++first0;
            ++data0;
        }
        else
        {
            result_col_ind.push_back(*first1);
            result_data.push_back(-*data1);

            ++first1;
            ++data1;
        }
    }
    // Pass over the tails
    while(first0 != last0)
    {
        result_col_ind.push_back(*first0);
        result_data.push_back(*data0);

        ++first0;
        ++data0;
    }
    while(first1 != last1)
    {
        result_col_ind.push_back(*first1);
        result_data.push_back(-*data1);

        ++first1;
        ++data1;
    }
    return std::make_tuple(result_col_ind, result_data);
}


/*
 * @brief Computes the element-wise difference of the normalized form of two
 * sparse vectors.
 */
template<class IterCol, class IterData>
std::tuple<std::vector<size_t>, std::vector<float>> sparse_weighted_diff(
    IterCol first0,
    IterCol last0,
    IterData data0,
    float weight0,
    IterCol first1,
    IterCol last1,
    IterData data1,
    float weight1
)
{
    std::vector<size_t> result_col_ind;
    std::vector<float> result_data;

    // Pass through both index lists
    while(first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            result_col_ind.push_back(*first0);
            result_data.push_back(*data0/weight0 - *data1/weight1);

            ++first0;
            ++data0;

            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            result_col_ind.push_back(*first0);
            result_data.push_back(*data0/weight0);

            ++first0;
            ++data0;
        }
        else
        {
            result_col_ind.push_back(*first1);
            result_data.push_back(-*data1/weight1);

            ++first1;
            ++data1;
        }
    }
    // Pass over the tails
    while(first0 != last0)
    {
        result_col_ind.push_back(*first0);
        result_data.push_back(*data0/weight0);

        ++first0;
        ++data0;
    }
    while(first1 != last1)
    {
        result_col_ind.push_back(*first1);
        result_data.push_back(-*data1/weight1);

        ++first1;
        ++data1;
    }
    return std::make_tuple(result_col_ind, result_data);
}


/*
 * @brief Standard euclidean distance.
 */
template<class Iter0, class Iter1>
float euclidean(Iter0 first0, Iter0 last0, Iter1 first1)
{
    float result = 0.0f;
    while (first0 != last0)
    {
        result = std::move(result) + (*first0 - *first1)*(*first0 - *first1);
        ++first0;
        ++first1;
    }
    return std::sqrt(result);
}


/*
 * @brief Standardized euclidean distance.
 *
 * Euclidean distance standardized against a vector of standard
 * deviations per coordinate.
 *
 * \f[
 *     D(x, y) = \sqrt{\sum_i \frac{(x_i - y_i)^2}{v_i}}
 * \f]
 */
template<class Iter0, class Iter1, class Iter2>
float standardised_euclidean(
    Iter0 first0, Iter0 last0, Iter1 first1, Iter2 first2
)
{
    float result = 0.0f;
    while (first0 != last0)
    {
        result = std::move(result)
            + (*first0 - *first1)*(*first0 - *first1) / *first2;
        ++first0;
        ++first1;
        ++first2;
    }
    return std::sqrt(result);
}


/*
 * @brief Manhattan, taxicab, or l1 distance.
 *
 * \f[
 *     D(x, y) = \sum_i |x_i - y_i|
 * \f]
 */
template<class Iter0, class Iter1>
float manhattan(Iter0 first0, Iter0 last0, Iter1 first1)
{
    float result = 0.0f;
    for (;first0 != last0; ++first0, ++first1)
    {
        result = std::move(result) + std::abs(*first0 - *first1);
    }
    return result;
}


template<class IterCol, class IterData>
float sparse_manhattan(
    IterCol first0,
    IterCol last0,
    IterData data0,
    IterCol first1,
    IterCol last1,
    IterData data1
)
{
    float result = 0.0f;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            result += std::abs(*data0 - *data1);

            ++first0;
            ++data0;

            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            result += std::abs(*data0);

            ++first0;
            ++data0;
        }
        else
        {
            result += std::abs(*data1);

            ++first1;
            ++data1;
        }
    }

    // Pass over the tails
    while (first0 != last0)
    {
        result += std::abs(*data0);

        ++first0;
        ++data0;
    }

    while (first1 != last1)
    {
        result += std::abs(*data1);

        ++first1;
        ++data1;
    }

    return result;
}


/*
 * @brief Chebyshev or l-infinity distance.
 *
 * \f[
 *     D(x, y) = \max_i |x_i - y_i|
 * \f]
 */
template<class Iter0, class Iter1>
float chebyshev(Iter0 first0, Iter0 last0, Iter1 first1)
{
    float result = 0.0f;
    while (first0 != last0)
    {
        result = std::max(std::move(result), std::abs(*first0 - *first1));
        ++first0;
        ++first1;
    }
    return result;
}


/*
 * @brief Chebyshev or l-infinity distance.
 *
 * \f[
 *     D(x, y) = \max_i |x_i - y_i|
 * \f]
 */
template<class IterCol, class IterData>
float sparse_chebyshev(
    IterCol first0,
    IterCol last0,
    IterData data0,
    IterCol first1,
    IterCol last1,
    IterData data1
)
{
    float result = 0.0f;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            float diff = std::abs(*data0 - *data1);
            result = std::max(result, diff);

            ++first0;
            ++data0;

            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            float diff = std::abs(*data0);
            result = std::max(result, diff);

            ++first0;
            ++data0;
        }
        else
        {
            float diff = std::abs(*data1);
            result = std::max(result, diff);

            ++first1;
            ++data1;
        }
    }

    // Pass over the tails
    while (first0 != last0)
    {
        float diff = std::abs(*data0);
        result = std::max(result, diff);

        ++first0;
        ++data0;
    }
    while (first1 != last1)
    {
        float diff = std::abs(*data1);
        result = std::max(result, diff);

        ++first1;
        ++data1;
    }

    return result;
}


/*
 * @brief Minkowski distance.
 *
 * \f[
 *     D(x, y) = \left(\sum_i |x_i - y_i|^p\right)^{\frac{1}{p}}
 * \f]
 *
 * This is a general distance. For p=1 it is equivalent to
 * manhattan distance, for p=2 it is Euclidean distance, and
 * for p=infinity it is Chebyshev distance. In general it is better
 * to use the more specialised functions for those distances.
 */
template<class Iter0, class Iter1>
float minkowski(Iter0 first0, Iter0 last0, Iter1 first1, float p)
{
    float result = 0.0f;
    while (first0 != last0)
    {
        result = std::move(result) + std::pow(std::abs(*first0 - *first1), p);
        ++first0;
        ++first1;
    }
    return std::pow(result, 1.0f / p);
}


/*
 * @brief Sparse Minkowski distance.
 
 * \f[
 *     D(x, y) = \left(\sum_i |x_i - y_i|^p\right)^{\frac{1}{p}}
 * \f]
 *
 * This is a general distance. For p=1 it is equivalent to
 * manhattan distance, for p=2 it is Euclidean distance, and
 * for p=infinity it is Chebyshev distance. In general it is better
 * to use the more specialised functions for those distances.
 */
template<class IterCol, class IterData>
float sparse_minkowski(
    IterCol first0,
    IterCol last0,
    IterData data0,
    IterCol first1,
    IterCol last1,
    IterData data1,
    float p
)
{
    float result = 0.0f;
    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            float diff = std::abs(*data0 - *data1);
            result = std::move(result) + std::pow(diff, p);

            ++first0;
            ++data0;

            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            float diff = std::abs(*data0);
            result = std::move(result) + std::pow(diff, p);

            ++first0;
            ++data0;
        }
        else
        {
            float diff = std::abs(*data1);
            result = std::move(result) + std::pow(diff, p);

            ++first1;
            ++data1;
        }
    }

    // Pass over the tails
    while (first0 != last0)
    {
        float diff = std::abs(*data0);
        result = std::move(result) + std::pow(diff, p);

        ++first0;
        ++data0;
    }
    while (first1 != last1)
    {
        float diff = std::abs(*data1);
        result = std::move(result) + std::pow(diff, p);

        ++first1;
        ++data1;
    }

    return std::pow(result, 1.0f / p);
}


/*
 *  @brief A weighted version of Minkowski distance.
 *
 * \f[
       D(x, y) = \left(\sum_i w_i |x_i - y_i|^p\right)^{\frac{1}{p}}
 * \f]
 *
 * If weights w_i are inverse standard deviations of graph_data in each
 * dimension then this represented a standardised Minkowski distance (and
 * is equivalent to standardised Euclidean distance for p=1).
 */
template<class Iter0, class Iter1, class Iter2>
float weighted_minkowski(
    Iter0 first0, Iter0 last0, Iter1 first1, Iter2 first2, float p
)
{
    float result = 0.0f;
    while (first0 != last0)
    {
        result = std::move(result)
            + (*first2) * std::pow(std::abs(*first0 - *first1), p);
        ++first0;
        ++first1;
        ++first2;
    }
    return std::pow(result, 1.0f / p);
}


/*
 * @brief Mahalanobis distance.
 */
template<class Iter0, class Iter1, class Iter2>
float mahalanobis(Iter0 first0, Iter0 last0, Iter1 first1, Iter2 matrix_first)
{
    float result = 0.0f;
    int dim = std::distance(first0, last0);
    std::vector<float> diff(dim, 0.0f);

    for (int i = 0; i < dim; ++i, ++first0, ++first1)
    {
        diff[i] = *first0 - *first1;
    }

    first0 -= diff.size();

    for (int i = 0; i < dim; ++i)
    {
        float tmp = 0.0f;

        for (int j = 0; j < dim; ++j, ++matrix_first)
        {
            tmp += *matrix_first * diff[j];
        }

        result += tmp * diff[i];
    }

    return std::sqrt(result);
}


/*
 * @brief Hamming distance.
 */
template<class Iter0, class Iter1>
float hamming(Iter0 first0, Iter0 last0, Iter1 first1)
{
    int result = 0;
    while (first0 != last0)
    {
        if (*first0 != *first1)
        {
            ++result;
        }
        ++first0;
        ++first1;
    }
    return result;
}


/*
 * @brief Hamming distance.
 */
template<class IterCol, class IterData>
float sparse_hamming(
    IterCol first0,
    IterCol last0,
    IterData data0,
    IterCol first1,
    IterCol last1,
    IterData data1
)
{
    int result = 0;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            if (*data0 != *data1)
            {
                ++result;
            }
            ++first0;
            ++data0;

            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            ++result;

            ++first0;
            ++data0;
        }
        else
        {
            ++result;

            ++first1;
            ++data1;
        }
    }

    // Pass over the tails
    while (first0 != last0)
    {
        ++result;

        ++first0;
        ++data0;
    }
    while (first1 != last1)
    {
        ++result;

        ++first1;
        ++data1;
    }

    return result;
}


/*
 * @brief Canberra distance.
 */
template<class Iter0, class Iter1>
float canberra(Iter0 first0, Iter0 last0, Iter1 first1)
{
    float result = 0.0f;
    while (first0 != last0)
    {
        float denominator = std::abs(*first0) + std::abs(*first1);
        if (denominator > 0)
        {
            result = std::move(result)
                + std::abs(*first0 - *first1) / denominator;
        }
        ++first0;
        ++first1;
    }
    return result;
}


/*
 * @brief Sparse Canberra distance.
 */
template<class IterCol, class IterData>
float sparse_canberra(
    IterCol first0,
    IterCol last0,
    IterData data0,
    IterCol first1,
    IterCol last1,
    IterData data1
)
{
    float result = 0.0f;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            float denominator = std::abs(*data0) + std::abs(*data1);
            if (denominator > 0)
            {
                result += std::abs(*data0 - *data1) / denominator;
            }

            ++first0;
            ++data0;

            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            float denominator = std::abs(*data0);
            if (denominator > 0)
            {
                result += std::abs(*data0) / denominator;
            }

            ++first0;
            ++data0;
        }
        else
        {
            float denominator = std::abs(*data1);
            if (denominator > 0)
            {
                result += std::abs(*data1) / denominator;
            }

            ++first1;
            ++data1;
        }
    }

    // Pass over the tails
    while (first0 != last0)
    {
        float denominator = std::abs(*data0);
        if (denominator > 0)
        {
            result += std::abs(*data0) / denominator;
        }

        ++first0;
        ++data0;
    }
    while (first1 != last1)
    {
        float denominator = std::abs(*data1);
        if (denominator > 0)
        {
            result += std::abs(*data1) / denominator;
        }

        ++first1;
        ++data1;
    }

    return result;
}


/*
 * @brief Bray–Curtis dissimilarity.
 */
template<class Iter0, class Iter1>
float bray_curtis(Iter0 first0, Iter0 last0, Iter1 first1)
{
    float numerator = 0.0f;
    float denominator = 0.0f;
    while (first0 != last0)
    {
        numerator = std::move(numerator) + std::abs(*first0 - *first1);
        denominator = std::move(denominator) + std::abs(*first0 + *first1);
        ++first0;
        ++first1;
    }
    if (denominator > 0.0f)
    {
        return numerator / denominator;
    }
    return 0.0f;
}


/*
 * @brief Sparse Bray–Curtis dissimilarity.
 */
template<class IterCol0, class IterData0, class IterCol1, class IterData1>
float sparse_bray_curtis(
    IterCol0 first0,
    IterCol0 last0,
    IterData0 data0,
    IterCol1 first1,
    IterCol1 last1,
    IterData1 data1
)
{
    float numerator = 0.0f;
    float denominator = 0.0f;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            numerator += std::abs(*data0 - *data1);
            denominator += std::abs(*data0 + *data1);
            ++first0;
            ++data0;
            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            numerator += std::abs(*data0);
            denominator += std::abs(*data0);
            ++first0;
            ++data0;
        }
        else
        {
            numerator += std::abs(*data1);
            denominator += std::abs(*data1);
            ++first1;
            ++data1;
        }
    }

    // Pass over the tails
    while (first0 != last0)
    {
        numerator += std::abs(*data0);
        denominator += std::abs(*data0);
        ++first0;
        ++data0;
    }
    while (first1 != last1)
    {
        numerator += std::abs(*data1);
        denominator += std::abs(*data1);
        ++first1;
        ++data1;
    }

    if (denominator > 0.0f)
    {
        return numerator / denominator;
    }

    return 0.0f;
}


/*
 * @brief Jaccard distance.
 */
template<class Iter0, class Iter1>
float jaccard(Iter0 first0, Iter0 last0, Iter1 first1)
{
    int num_non_zero = 0;
    int num_equal = 0;
    while (first0 != last0)
    {
        bool first0_true = ((*first0) != 0);
        bool first1_true = ((*first1) != 0);
        num_non_zero = std::move(num_non_zero) + (first0_true || first1_true);
        num_equal = std::move(num_equal) + (first0_true && first1_true);
        ++first0;
        ++first1;
    }
    if (num_non_zero == 0)
    {
        return 0.0f;
    }
    return static_cast<float>(num_non_zero - num_equal) / num_non_zero;
}


/*
 * @brief Sparse jaccard distance.
 */
template<class IterCol, class IterData>
float sparse_jaccard(
    IterCol first0,
    IterCol last0,
    IterData data0,
    IterCol first1,
    IterCol last1,
    IterData data1
)
{
    int num_non_zero = 0;
    int num_equal = 0;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            ++num_non_zero;
            ++num_equal;

            ++first0;
            ++data0;

            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            ++num_non_zero;

            ++first0;
            ++data0;
        }
        else
        {
            ++num_non_zero;

            ++first1;
            ++data1;
        }
    }

    // Pass over the tails
    while (first0 != last0)
    {
        ++num_non_zero;

        ++first0;
        ++data0;
    }
    while (first1 != last1)
    {
        ++num_non_zero;

        ++first1;
        ++data1;
    }

    if (num_non_zero == 0)
    {
        return 0.0f;
    }

    return static_cast<float>(num_non_zero - num_equal) / num_non_zero;
}


/*
 * @brief Alternative Jaccard distance.
 */
template<class Iter0, class Iter1>
float alternative_jaccard(Iter0 first0, Iter0 last0, Iter1 first1)
{
    int num_non_zero = 0;
    int num_equal = 0;
    for (; first0 != last0; ++first0, ++first1)
    {
        bool first0_true = ((*first0) != 0);
        bool first1_true = ((*first1) != 0);
        num_non_zero = std::move(num_non_zero) + (first0_true || first1_true);
        num_equal = std::move(num_equal) + (first0_true && first1_true);
    }
    if (num_non_zero == 0)
    {
        return 0.0f;
    }
    return -std::log2((float)num_equal / num_non_zero);
}


/*
 * @brief Sparse alternative jaccard distance.
 */
template<class IterCol, class IterData>
float sparse_alternative_jaccard(
    IterCol first0,
    IterCol last0,
    IterData data0,
    IterCol first1,
    IterCol last1,
    IterData data1
)
{
    int num_non_zero = 0;
    int num_equal = 0;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            ++num_non_zero;
            ++num_equal;

            ++first0;
            ++data0;

            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            ++num_non_zero;

            ++first0;
            ++data0;
        }
        else
        {
            ++num_non_zero;

            ++first1;
            ++data1;
        }
    }

    // Pass over the tails
    while (first0 != last0)
    {
        ++num_non_zero;

        ++first0;
        ++data0;
    }
    while (first1 != last1)
    {
        ++num_non_zero;

        ++first1;
        ++data1;
    }
    if (num_non_zero == 0)
    {
        return 0.0f;
    }
    return -std::log2(
        static_cast<float>(num_equal) / num_non_zero
    );
}


/*
 * @brief Correction function for Jaccard distance.
 */
inline float correct_alternative_jaccard(float value)
{
    return 1.0f - std::pow(2.0f, -value);
}


/*
 * @brief Matching.
 */
template<class Iter0, class Iter1>
float matching(Iter0 first0, Iter0 last0, Iter1 first1)
{
    int num_not_equal = 0;
    for (; first0 != last0; ++first0, ++first1)
    {
        bool first0_true = (*first0 != 0);
        bool first1_true = (*first1 != 0);
        num_not_equal = std::move(num_not_equal) + (first0_true != first1_true);
    }
    return static_cast<float>(num_not_equal);
}


/*
 * @brief Sparse matching.
 */
template<class IterCol, class IterData>
float sparse_matching(
    IterCol first0,
    IterCol last0,
    IterData data0,
    IterCol first1,
    IterCol last1,
    IterData data1
)
{
    int num_not_equal = 0;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            ++first0;
            ++data0;

            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            ++num_not_equal;

            ++first0;
            ++data0;
        }
        else
        {
            ++num_not_equal;

            ++first1;
            ++data1;
        }
    }

    // Pass over the tails
    while (first0 != last0)
    {
        ++num_not_equal;

        ++first0;
        ++data0;
    }
    while (first1 != last1)
    {
        ++num_not_equal;

        ++first1;
        ++data1;
    }

    return static_cast<float>(num_not_equal);
}


/*
 * @brief Dice.
 */
template<class Iter0, class Iter1>
float dice(Iter0 first0, Iter0 last0, Iter1 first1)
{
    int num_true_true = 0;
    int num_not_equal = 0;
    for (; first0 != last0; ++first0, ++first1)
    {
        bool first0_true = (*first0 != 0);
        bool first1_true = (*first1 != 0);
        num_true_true = std::move(num_true_true) + (first0_true && first1_true);
        num_not_equal = std::move(num_not_equal) + (first0_true != first1_true);
    }
    if (num_not_equal == 0)
    {
        return 0.0f;
    }
    return static_cast<float>(num_not_equal)
        / (2.0f * num_true_true + num_not_equal);
}


/*
 * @brief Sparse dice.
 */
template<class IterCol, class IterData>
float sparse_dice(
    IterCol first0,
    IterCol last0,
    IterData data0,
    IterCol first1,
    IterCol last1,
    IterData data1
)
{
    int num_true_true = 0;
    int num_not_equal = 0;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            ++num_true_true;

            ++first0;
            ++data0;

            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            ++num_not_equal;

            ++first0;
            ++data0;
        }
        else
        {
            ++num_not_equal;

            ++first1;
            ++data1;
        }
    }

    // Pass over the tails
    while (first0 != last0)
    {
        ++num_not_equal;

        ++first0;
        ++data0;
    }
    while (first1 != last1)
    {
        ++num_not_equal;

        ++first1;
        ++data1;
    }

    if (num_not_equal == 0)
    {
        return 0.0f;
    }

    return static_cast<float>(num_not_equal)
        / (2.0f * num_true_true + num_not_equal);
}


/*
 * @brief Kulsinski dissimilarity.
 */
template<class Iter0, class Iter1>
float kulsinski(Iter0 first0, Iter0 last0, Iter1 first1)
{
    int num_true_true = 0;
    int num_not_equal = 0;
    float dim = last0 - first0;
    for (; first0 != last0; ++first0, ++first1)
    {
        bool first0_true = (*first0 != 0);
        bool first1_true = (*first1 != 0);
        num_true_true = std::move(num_true_true) + (first0_true && first1_true);
        num_not_equal = std::move(num_not_equal) + (first0_true != first1_true);
    }
    if (num_not_equal == 0)
    {
        return 0.0f;
    }
    return static_cast<float>(num_not_equal - num_true_true + dim)
        / (num_not_equal + dim);
}


/*
 * @brief Sparse kulsinski dissimilarity.
 */
template<class IterCol, class IterData>
float sparse_kulsinski(
    IterCol first0,
    IterCol last0,
    IterData data0,
    IterCol first1,
    IterCol last1,
    IterData data1,
    float dim
)
{
    int num_true_true = 0;
    int num_not_equal = 0;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            ++num_true_true;

            ++first0;
            ++data0;

            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            ++num_not_equal;

            ++first0;
            ++data0;
        }
        else
        {
            ++num_not_equal;

            ++first1;
            ++data1;
        }
    }

    // Pass over the tails
    while (first0 != last0)
    {
        ++num_not_equal;

        ++first0;
        ++data0;
    }
    while (first1 != last1)
    {
        ++num_not_equal;

        ++first1;
        ++data1;
    }

    if (num_not_equal == 0)
    {
        return 0.0f;
    }

    return static_cast<float>(num_not_equal - num_true_true + dim)
        / (num_not_equal + dim);
}


/*
 * @brief Rogers-Tanimoto dissimilarity.
 */
template<class Iter0, class Iter1>
float rogers_tanimoto(Iter0 first0, Iter0 last0, Iter1 first1)
{
    int num_not_equal = 0;
    float dim = last0 - first0;
    for (; first0 != last0; ++first0, ++first1)
    {
        bool first0_true = (*first0 != 0);
        bool first1_true = (*first1 != 0);
        num_not_equal = std::move(num_not_equal) + (first0_true != first1_true);
    }
    return (2.0f * num_not_equal) / (dim + num_not_equal);
}


/*
 * @brief Sparse Rogers-Tanimoto dissimilarity.
 */
template<class IterCol, class IterData>
float sparse_rogers_tanimoto(
    IterCol first0,
    IterCol last0,
    IterData data0,
    IterCol first1,
    IterCol last1,
    IterData data1,
    float dim
)
{
    int num_not_equal = 0;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            ++first0;
            ++data0;

            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            ++num_not_equal;

            ++first0;
            ++data0;
        }
        else
        {
            ++num_not_equal;

            ++first1;
            ++data1;
        }
    }

    // Pass over the tails
    while (first0 != last0)
    {
        ++num_not_equal;

        ++first0;
        ++data0;
    }
    while (first1 != last1)
    {
        ++num_not_equal;

        ++first1;
        ++data1;
    }

    return (2.0f * num_not_equal) / (dim + num_not_equal);
}


/*
 * @brief Russell-Rao dissimilarity.
 */
template<class Iter0, class Iter1>
float russellrao(Iter0 first0, Iter0 last0, Iter1 first1)
{
    int num_true_true = 0;
    size_t dim = last0 - first0;
    int first0_non0 = count_if_not_equal(first0, last0, 0);
    int first1_non0 = count_if_not_equal(first1, first1 + dim, 0);
    for (; first0 != last0; ++first0, ++first1)
    {
        bool first0_true = (*first0 != 0);
        bool first1_true = (*first1 != 0);
        num_true_true = std::move(num_true_true) + (first0_true && first1_true);
    }
    if ((num_true_true == first0_non0) && (num_true_true == first1_non0))
    {
        return 0.0f;
    }
    return (float)(dim - num_true_true) / dim;
}


/*
 * @brief Sparse Russell-Rao dissimilarity.
 */
template<class IterCol, class IterData>
float sparse_russellrao(
    IterCol first0,
    IterCol last0,
    IterData data0,
    IterCol first1,
    IterCol last1,
    IterData data1,
    float dim
)
{
    int num_true_true = 0;

    int first0_non0 = sparse_count_if_not_equal(first0, last0, data0, 0);
    int first1_non0 = sparse_count_if_not_equal(first1, last1, data0, 0);

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            ++num_true_true;

            ++first0;
            ++data0;

            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            ++first0;
            ++data0;
        }
        else
        {
            ++first1;
            ++data1;
        }
    }

    if ((num_true_true == first0_non0) && (num_true_true == first1_non0))
    {
        return 0.0f;
    }

    return (float)(dim - num_true_true) / dim;
}


/*
 * @brief Sokal-Michener dissimilarity.
 */
template<class Iter0, class Iter1>
float sokal_michener(Iter0 first0, Iter0 last0, Iter1 first1)
{
    int num_not_equal = 0;
    float dim = last0 - first0;
    for (; first0 != last0; ++first0, ++first1)
    {
        bool first0_true = (*first0 != 0);
        bool first1_true = (*first1 != 0);
        num_not_equal = std::move(num_not_equal) + (first0_true != first1_true);
    }
    return (2.0f * num_not_equal) / (dim + num_not_equal);
}


/*
 * @brief Sparse Sokal-Michener dissimilarity.
 */
template<class IterCol, class IterData>
float sparse_sokal_michener(
    IterCol first0,
    IterCol last0,
    IterData data0,
    IterCol first1,
    IterCol last1,
    IterData data1,
    float dim
)
{
    int num_not_equal = 0;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            ++first0;
            ++data0;

            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            ++num_not_equal;

            ++first0;
            ++data0;
        }
        else
        {
            ++num_not_equal;

            ++first1;
            ++data1;
        }
    }

    // Pass over the tails
    while (first0 != last0)
    {
        ++num_not_equal;

        ++first0;
        ++data0;
    }

    while (first1 != last1)
    {
        ++num_not_equal;

        ++first1;
        ++data1;
    }

    return (2.0f * num_not_equal) / (dim + num_not_equal);
}


/*
 * @brief Sokal-Sneath dissimilarity.
 */
template<class Iter0, class Iter1>
float sokal_sneath(Iter0 first0, Iter0 last0, Iter1 first1)
{
    int num_true_true = 0;
    int num_not_equal = 0;
    for (; first0 != last0; ++first0, ++first1)
    {
        bool first0_true = (*first0 != 0);
        bool first1_true = (*first1 != 0);
        num_true_true = std::move(num_true_true) + (first0_true && first1_true);
        num_not_equal = std::move(num_not_equal) + (first0_true != first1_true);
    }
    if (num_not_equal == 0)
    {
        return 0.0f;
    }
    return (float)(num_not_equal) / (0.5f * num_true_true + num_not_equal);
}


/*
 * @brief Sparse Sokal-Sneath dissimilarity.
 */
template<class IterCol, class IterData>
float sparse_sokal_sneath(
    IterCol first0,
    IterCol last0,
    IterData data0,
    IterCol first1,
    IterCol last1,
    IterData data1
)
{
    int num_true_true = 0;
    int num_not_equal = 0;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            ++num_true_true;

            ++first0;
            ++data0;

            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            ++num_not_equal;

            ++first0;
            ++data0;
        }
        else
        {
            ++num_not_equal;

            ++first1;
            ++data1;
        }
    }

    // Pass over the tails
    while (first0 != last0)
    {
        ++num_not_equal;

        ++first0;
        ++data0;
    }

    while (first1 != last1)
    {
        ++num_not_equal;

        ++first1;
        ++data1;
    }

    if (num_not_equal == 0)
    {
        return 0.0f;
    }

    return static_cast<float>(num_not_equal)
        / (0.5f * num_true_true + num_not_equal);
}


/*
 * @brief Haversine distance.
 */
template<class Iter0, class Iter1>
float haversine(Iter0 first0, Iter0 last0, Iter1 first1)
{
    float sin_lat = std::sin(0.5f * (*first0 - *first1));
    float sin_long = std::sin(0.5f * (*(first0 + 1) - *(first1 + 1)));
    float result = std::sqrt(
        sin_lat*sin_lat
        + std::cos(*first0) * std::cos(*first1) * sin_long*sin_long
    );
    return 2.0f * std::asin(result);
}


/*
 * @brief Yule dissimilarity.
 */
template<class Iter0, class Iter1>
float yule(Iter0 first0, Iter0 last0, Iter1 first1)
{
    int num_true_true = 0;
    int num_true_false = 0;
    int num_false_true = 0;
    int dim = last0 - first0;
    for (; first0 != last0; ++first0, ++first1)
    {
        bool first0_true = (*first0 != 0);
        bool first1_true = (*first1 != 0);
        num_true_true = std::move(num_true_true) + (first0_true && first1_true);
        num_true_false = std::move(num_true_false) + (first0_true && !first1_true);
        num_false_true = std::move(num_false_true) + (!first0_true && first1_true);
    }
    int num_false_false = dim - num_true_true - num_true_false - num_false_true;
    if ((num_true_false == 0) || (num_false_true == 0))
    {
        return 0.0f;
    }
    return (float)(2.0f * num_true_false * num_false_true) / (
        num_true_true * num_false_false + num_true_false * num_false_true
    );
}


/*
 * @brief Sparse Yule dissimilarity.
 */
template<class IterCol0, class IterData0, class IterCol1, class IterData1>
float sparse_yule(
    IterCol0 first0,
    IterCol0 last0,
    IterData0 data0,
    IterCol1 first1,
    IterCol1 last1,
    IterData1 data1,
    float dim
)
{
    int num_true_true = 0;
    int num_true_false = 0;
    int num_false_true = 0;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            ++num_true_true;

            ++first0;
            ++data0;
            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            ++num_true_false;

            ++first0;
            ++data0;
        }
        else
        {
            ++num_false_true;

            ++first1;
            ++data1;
        }
    }

    // Pass over the tails
    while (first0 != last0)
    {
        ++num_true_false;

        ++first0;
        ++data0;
    }
    while (first1 != last1)
    {
        ++num_false_true;

        ++first1;
        ++data1;
    }

    int num_false_false = dim - num_true_true - num_true_false - num_false_true;

    if ((num_true_false == 0) || (num_false_true == 0))
    {
        return 0.0f;
    }

    return (float)(2.0f * num_true_false * num_false_true) / (
        num_true_true * num_false_false + num_true_false * num_false_true
    );
}


/*
 * @brief Cosine similarity.
 */
template<class Iter0, class Iter1>
float cosine(Iter0 first0, Iter0 last0, Iter1 first1)
{
    float result = 0.0f;
    float norm0 = 0.0f;
    float norm1 = 0.0f;
    for (; first0 != last0; ++first0, ++first1)
    {
        result = std::move(result) + (*first0) * (*first1);
        norm0 = std::move(norm0) + (*first0) * (*first0);
        norm1 = std::move(norm1) + (*first1) * (*first1);
    }
    if ((norm0 == 0.0f) && (norm1 == 0.0f))
    {
        return 0.0f;
    }
    else if ((norm0 == 0.0f) || (norm1 == 0.0f))
    {
        return 1.0f;
    }
    return 1.0f - (result / std::sqrt(norm0 * norm1));
}


/*
 * @brief Sparse cosine similarity.
 */
template<class IterCol, class IterData>
float sparse_cosine(
    IterCol first0,
    IterCol last0,
    IterData data0,
    IterCol first1,
    IterCol last1,
    IterData data1
)
{
    float result = 0.0f;
    float norm0 = 0.0f;
    float norm1 = 0.0f;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            result += (*data0) * (*data1);
            norm0 += (*data0) * (*data0);
            norm1 += (*data1) * (*data1);

            ++first0;
            ++data0;

            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            norm0 += (*data0) * (*data0);

            ++first0;
            ++data0;
        }
        else
        {
            norm1 += (*data1) * (*data1);

            ++first1;
            ++data1;
        }
    }
    // Pass over the tails
    while (first0 != last0)
    {
        norm0 += (*data0) * (*data0);

        ++first0;
        ++data0;
    }
    while (first1 != last1)
    {
        norm1 += (*data1) * (*data1);

        ++first1;
        ++data1;
    }

    if ((norm0 == 0.0f) && (norm1 == 0.0f))
    {
        return 0.0f;
    }
    else if ((norm0 == 0.0f) || (norm1 == 0.0f))
    {
        return 1.0f;
    }

    return 1.0f - (result / std::sqrt(norm0 * norm1));
}


/*
 * @brief Alternative cosine similarity.
 */
template<class Iter0, class Iter1>
float alternative_cosine(Iter0 first0, Iter0 last0, Iter1 first1)
{
    float result = 0.0f;
    float norm0 = 0.0f;
    float norm1 = 0.0f;
    for (; first0 != last0; ++first0, ++first1)
    {
        result = std::move(result) + (*first0) * (*first1);
        norm0 = std::move(norm0) + (*first0) * (*first0);
        norm1 = std::move(norm1) + (*first1) * (*first1);
    }
    if ((norm0 == 0.0f) && (norm1 == 0.0f))
    {
        return 0.0f;
    }
    else if ((norm0 == 0.0f) || (norm1 == 0.0f))
    {
        return FLOAT_MAX;
    }
    else if (result <= 0.0f)
    {
        return FLOAT_MAX;
    }
    result = std::sqrt(norm0 * norm1) / result;
    return std::log2(result);
}


/*
 * @brief Sparse alternative cosine similarity.
 */
template<class IterCol, class IterData>
float sparse_alternative_cosine(
    IterCol first0,
    IterCol last0,
    IterData data0,
    IterCol first1,
    IterCol last1,
    IterData data1
)
{
    float result = 0.0f;
    float norm0 = 0.0f;
    float norm1 = 0.0f;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            result += (*data0) * (*data1);
            norm0 += (*data0) * (*data0);
            norm1 += (*data1) * (*data1);

            ++first0;
            ++data0;

            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            norm0 += (*data0) * (*data0);

            ++first0;
            ++data0;
        }
        else
        {
            norm1 += (*data1) * (*data1);

            ++first1;
            ++data1;
        }
    }
    // Pass over the tails
    while (first0 != last0)
    {
        norm0 += (*data0) * (*data0);

        ++first0;
        ++data0;
    }
    while (first1 != last1)
    {
        norm1 += (*data1) * (*data1);

        ++first1;
        ++data1;
    }

    if ((norm0 == 0.0f) && (norm1 == 0.0f))
    {
        return 0.0f;
    }
    else if ((norm0 == 0.0f) || (norm1 == 0.0f))
    {
        return FLOAT_MAX;
    }
    else if (result <= 0.0f)
    {
        return FLOAT_MAX;
    }

    result = std::sqrt(norm0 * norm1) / result;
    return std::log2(result);
}


/*
 * @brief Dot.
 */
template<class Iter0, class Iter1>
float dot(Iter0 first0, Iter0 last0, Iter1 first1)
{
    float result = 0.0f;
    for (; first0 != last0; ++first0, ++first1)
    {
        result = std::move(result) + (*first0) * (*first1);
    }
    if (result <= 0.0f)
    {
        return 1.0f;
    }
    return 1.0f - result;
}


/*
 * @brief Sparse dot.
 */
template<class IterCol, class IterData>
float sparse_dot(
    IterCol first0,
    IterCol last0,
    IterData data0,
    IterCol first1,
    IterCol last1,
    IterData data1
)
{
    float result = 0.0f;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            result += (*data0) * (*data1);

            ++first0;
            ++data0;

            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            ++first0;
            ++data0;
        }
        else
        {
            ++first1;
            ++data1;
        }
    }

    if (result <= 0.0f)
    {
        return 1.0f;
    }

    return 1.0f - result;
}


/*
 * @brief Alternative dot.
 */
template<class Iter0, class Iter1>
float alternative_dot(Iter0 first0, Iter0 last0, Iter1 first1)
{
    float result = 0.0f;
    for (; first0 != last0; ++first0, ++first1)
    {
        result = std::move(result) + (*first0) * (*first1);
    }
    if (result <= 0.0f)
    {
        return FLOAT_MAX;
    }
    return -std::log2(result);
}


/*
 * @brief Sparse alternative dot.
 */
template<class IterCol, class IterData>
float sparse_alternative_dot(
    IterCol first0,
    IterCol last0,
    IterData data0,
    IterCol first1,
    IterCol last1,
    IterData data1
)
{
    float result = 0.0f;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            result += (*data0) * (*data1);

            ++first0;
            ++data0;

            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            ++first0;
            ++data0;
        }
        else
        {
            ++first1;
            ++data1;
        }
    }

    if (result <= 0.0f)
    {
        return FLOAT_MAX;
    }
    return -std::log2(result);
}


/*
 * @brief Correction function for cosine.
 */
inline float correct_alternative_cosine(float value)
{
    return 1.0f - std::pow(2.0f, -value);
}


/*
 * @brief TS-SS Similarity.
 */
template<class Iter0, class Iter1>
float tsss(Iter0 first0, Iter0 last0, Iter1 first1)
{
    float d_euc_squared = 0.0f;
    float d_cos = 0.0f;
    float norm0 = 0.0f;
    float norm1 = 0.0f;
    for (; first0 != last0; ++first0, ++first1)
    {
        float diff = (*first0) - (*first1);
        d_euc_squared = std::move(d_euc_squared) + diff*diff;
        d_cos = std::move(d_cos) + (*first0) * (*first1);
        norm0 = std::move(norm0) + (*first0) * (*first0);
        norm1 = std::move(norm1) + (*first1) * (*first1);
    }
    norm0 = std::sqrt(norm0);
    norm1 = std::sqrt(norm1);
    float magnitude_difference = std::abs(norm0 - norm1);
    d_cos /= norm0 * norm1;
    // Add 10 degrees as an "epsilon" to avoid problems.
    float theta = std::acos(d_cos) + PI / 18.0f;
    float sector = std::sqrt(d_euc_squared) + magnitude_difference;
    sector = sector*sector*theta;
    float triangle = norm0 * norm1 * std::sin(theta) / 2.0f;
    return triangle * sector;
}


/*
 * @brief Sparse TS-SS Similarity.
 */
template<class IterCol0, class IterData0, class IterCol1, class IterData1>
float sparse_tsss(
    IterCol0 first0,
    IterCol0 last0,
    IterData0 data0,
    IterCol1 first1,
    IterCol1 last1,
    IterData1 data1
)
{
    float d_euc_squared = 0.0f;
    float d_cos = 0.0f;
    float norm0 = 0.0f;
    float norm1 = 0.0f;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            float diff = (*data0) - (*data1);
            d_euc_squared += diff * diff;
            d_cos += (*data0) * (*data1);
            norm0 += (*data0) * (*data0);
            norm1 += (*data1) * (*data1);

            ++first0;
            ++data0;
            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            float diff = (*data0);
            d_euc_squared += diff * diff;
            norm0 += (*data0) * (*data0);

            ++first0;
            ++data0;
        }
        else
        {
            float diff = *data1;
            d_euc_squared += diff * diff;
            norm1 += (*data1) * (*data1);

            ++first1;
            ++data1;
        }
    }

    // Pass over the tails
    while (first0 != last0)
    {
        float diff = (*data0);
        d_euc_squared += diff * diff;
        norm0 += (*data0) * (*data0);

        ++first0;
        ++data0;
    }
    while (first1 != last1)
    {
        float diff = *data1;
        d_euc_squared += diff * diff;
        norm1 += (*data1) * (*data1);

        ++first1;
        ++data1;
    }

    norm0 = std::sqrt(norm0);
    norm1 = std::sqrt(norm1);

    float magnitude_difference = std::abs(norm0 - norm1);
    d_cos /= (norm0 * norm1);

    // Add 10 degrees as an "epsilon" to avoid problems.
    float theta = std::acos(d_cos) + PI / 18.0f;

    float sector = std::sqrt(d_euc_squared) + magnitude_difference;
    sector = sector * sector * theta;

    float triangle = norm0 * norm1 * std::sin(theta) / 2.0f;

    return triangle * sector;
}


/*
 * @brief True angular.
 */
template<class Iter0, class Iter1>
float true_angular(Iter0 first0, Iter0 last0, Iter1 first1)
{
    float result = 0.0f;
    float norm0 = 0.0f;
    float norm1 = 0.0f;
    for (; first0 != last0; ++first0, ++first1)
    {
        result = std::move(result) + (*first0) * (*first1);
        norm0 = std::move(norm0) + (*first0) * (*first0);
        norm1 = std::move(norm1) + (*first1) * (*first1);
    }
    if ((norm0 == 0.0f) && (norm1 == 0.0f))
    {
        return 0.0f;
    }
    else if ((norm0 == 0.0f) || (norm1 == 0.0f))
    {
        return FLOAT_MAX;
    }
    else if (result <= 0.0f)
    {
        return FLOAT_MAX;
    }
    result = result / std::sqrt(norm0 * norm1);
    return 1.0f - std::acos(result) / PI;
}


/*
 * @brief Sparse true angular.
 */
template<class IterCol, class IterData>
float sparse_true_angular(
    IterCol first0,
    IterCol last0,
    IterData data0,
    IterCol first1,
    IterCol last1,
    IterData data1
)
{
    float result = 0.0f;
    float norm0 = 0.0f;
    float norm1 = 0.0f;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            result += (*data0) * (*data1);
            norm0 += (*data0) * (*data0);
            norm1 += (*data1) * (*data1);

            ++first0;
            ++data0;

            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            norm0 += (*data0) * (*data0);

            ++first0;
            ++data0;
        }
        else
        {
            norm1 += (*data1) * (*data1);

            ++first1;
            ++data1;
        }
    }

    // Pass over the tails
    while (first0 != last0)
    {
        norm0 += (*data0) * (*data0);

        ++first0;
        ++data0;
    }
    while (first1 != last1)
    {
        norm1 += (*data1) * (*data1);

        ++first1;
        ++data1;
    }

    if ((norm0 == 0.0f) && (norm1 == 0.0f))
    {
        return 0.0f;
    }
    else if ((norm0 == 0.0f) || (norm1 == 0.0f))
    {
        return FLOAT_MAX;
    }
    else if (result <= 0.0f)
    {
        return FLOAT_MAX;
    }
    result = result / std::sqrt(norm0 * norm1);
    return 1.0f - std::acos(result) / PI;
}


/*
 * @brief Correction function true angular.
 */
inline float true_angular_from_alt_cosine(float value)
{
    return 1.0f - std::acos(std::pow(2.0f, -value)) / PI;
}


/*
 * @brief Correlation.
 */
template<class Iter0, class Iter1>
float correlation(Iter0 first0, Iter0 last0, Iter1 first1)
{
    float mu0 = 0.0f;
    float mu1 = 0.0f;
    float norm0 = 0.0f;
    float norm1 = 0.0f;
    float dot_product = 0.0f;
    float dim = last0 - first0;
    Iter0 it0 = first0;
    Iter1 it1 = first1;
    for (; it0 != last0; ++it0, ++it1)
    {
        mu0 = std::move(mu0) + (*it0);
        mu1 = std::move(mu1) + (*it1);
    }
    mu0 /= dim;
    mu1 /= dim;

    for (; first0 != last0; ++first0, ++first1)
    {
        float shifted0 = *first0 - mu0;
        float shifted1 = *first1 - mu1;
        norm0 = std::move(norm0) + shifted0 * shifted0;
        norm1 = std::move(norm1) + shifted1 * shifted1;
        dot_product = std::move(dot_product) + shifted0 * shifted1;
    }
    if ((norm0 == 0.0f) && (norm1 == 0.0f))
    {
        return 0.0f;
    }
    else if (dot_product == 0.0f)
    {
        return 1.0f;
    }
    return 1.0f - dot_product / std::sqrt(norm0 * norm1);
}


/*
 * @brief Sparse correlation.
 */
template<class IterCol0, class IterData0, class IterCol1, class IterData1>
float sparse_correlation(
    IterCol0 first0,
    IterCol0 last0,
    IterData0 data0,
    IterCol1 first1,
    IterCol1 last1,
    IterData1 data1,
    float dim
)
{
    float mu0 = 0.0f;
    float mu1 = 0.0f;
    float norm0 = 0.0f;
    float norm1 = 0.0f;
    float dot_product = 0.0f;

    // Calculate means
    const IterCol0 _first0 = first0;
    const IterData0 _data0 = data0;
    for (; first0 != last0; ++first0, ++data0)
    {
        mu0 += *data0;
    }
    mu0 /= dim;
    first0 = _first0;
    data0 = _data0;

    const IterCol1 _first1 = first1;
    const IterData1 _data1 = data1;
    for (; first1 != last1; ++first1, ++data1)
    {
        mu1 += *data1;
    }
    mu1 /= dim;
    first1 = _first1;
    data1 = _data1;

    int both_indices_zero = dim;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        --both_indices_zero;
        if (*first0 == *first1)
        {
            float shifted0 = *data0 - mu0;
            float shifted1 = *data1 - mu1;
            norm0 += shifted0 * shifted0;
            norm1 += shifted1 * shifted1;
            dot_product += shifted0 * shifted1;

            ++first0;
            ++data0;
            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            float shifted0 = *data0 - mu0;
            float shifted1 = -mu1;
            norm0 += shifted0 * shifted0;
            norm1 += shifted1 * shifted1;
            dot_product += shifted0 * shifted1;

            ++first0;
            ++data0;
        }
        else
        {
            float shifted0 = -mu0;
            float shifted1 = *data1 - mu1;
            norm0 += shifted0 * shifted0;
            norm1 += shifted1 * shifted1;
            dot_product += shifted0 * shifted1;

            ++first1;
            ++data1;
        }
    }

    // Pass over the tails
    while (first0 != last0)
    {
        --both_indices_zero;
        float shifted0 = *data0 - mu0;
        float shifted1 = -mu1;
        norm0 += shifted0 * shifted0;
        norm1 += shifted1 * shifted1;
        dot_product += shifted0 * shifted1;

        ++first0;
        ++data0;
    }
    while (first1 != last1)
    {
        --both_indices_zero;
        float shifted0 = -mu0;
        float shifted1 = *data1 - mu1;
        norm0 += shifted0 * shifted0;
        norm1 += shifted1 * shifted1;
        dot_product += shifted0 * shifted1;

        ++first1;
        ++data1;
    }
    // Correct for positions where both indices are 0
    float shifted0 = -mu0;
    float shifted1 = -mu1;
    norm0 += both_indices_zero * shifted0 * shifted0;
    norm1 += both_indices_zero * shifted1 * shifted1;
    dot_product += both_indices_zero * shifted0 * shifted1;


    if ((norm0 == 0.0f) && (norm1 == 0.0f))
    {
        return 0.0f;
    }
    else if (dot_product == 0.0f)
    {
        return 1.0f;
    }

    return 1.0f - dot_product / std::sqrt(norm0 * norm1);
}


/*
 * @brief Hellinger.
 */
template<class Iter0, class Iter1>
float hellinger(Iter0 first0, Iter0 last0, Iter1 first1)
{
    float result = 0.0f;
    float l1_norm0 = 0.0f;
    float l1_norm1 = 0.0f;

    for (; first0 != last0; ++first0, ++first1)
    {
        result = std::move(result) + std::sqrt((*first0) * (*first1));

        l1_norm0 = std::move(l1_norm0) + *first0;
        l1_norm1 = std::move(l1_norm1) + *first1;
    }

    if ((l1_norm0 == 0.0f) && (l1_norm1 == 0.0f))
    {
        return 0.0f;
    }
    else if ((l1_norm0 == 0.0f) || (l1_norm1 == 0.0f))
    {
        return 1.0f;
    }

    return std::sqrt(1.0f - result / std::sqrt(l1_norm0 * l1_norm1));
}


/*
 * @brief Sparse Hellinger.
 */
template<class IterCol0, class IterData0, class IterCol1, class IterData1>
float sparse_hellinger(
    IterCol0 first0,
    IterCol0 last0,
    IterData0 data0,
    IterCol1 first1,
    IterCol1 last1,
    IterData1 data1
)
{
    float result = 0.0f;
    float l1_norm0 = 0.0f;
    float l1_norm1 = 0.0f;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            float value = std::sqrt((*data0) * (*data1));
            result += value;

            l1_norm0 += (*data0);
            l1_norm1 += (*data1);

            ++first0;
            ++data0;
            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            l1_norm0 += (*data0);

            ++first0;
            ++data0;
        }
        else
        {
            l1_norm1 += (*data1);

            ++first1;
            ++data1;
        }
    }

    // Pass over the tails
    while (first0 != last0)
    {
        l1_norm0 += (*data0);

        ++first0;
        ++data0;
    }
    while (first1 != last1)
    {
        l1_norm1 += (*data1);

        ++first1;
        ++data1;
    }

    if ((l1_norm0 == 0.0f) && (l1_norm1 == 0.0f))
    {
        return 0.0f;
    }
    else if ((l1_norm0 == 0.0f) || (l1_norm1 == 0.0f))
    {
        return 1.0f;
    }

    return std::sqrt(1.0f - result / std::sqrt(l1_norm0 * l1_norm1));
}


/*
 * @brief Alternative Hellinger.
 */
template<class Iter0, class Iter1>
float alternative_hellinger(Iter0 first0, Iter0 last0, Iter1 first1)
{
    float result = 0.0f;
    float l1_norm0 = 0.0f;
    float l1_norm1 = 0.0f;

    for (; first0 != last0; ++first0, ++first1)
    {
        result = std::move(result) + std::sqrt((*first0) * (*first1));

        l1_norm0 = std::move(l1_norm0) + *first0;
        l1_norm1 = std::move(l1_norm1) + *first1;
    }

    if ((l1_norm0 == 0.0f) && (l1_norm1 == 0.0f))
    {
        return 0.0f;
    }
    else if ((l1_norm0 == 0.0f) || (l1_norm1 == 0.0f))
    {
        return FLOAT_MAX;
    }
    else if (result <= 0.0f)
    {
        return FLOAT_MAX;
    }

    result = std::sqrt(l1_norm0 * l1_norm1) / result;
    return std::log2(result);
}


/*
 * @brief Sparse alternative Hellinger.
 */
template<class IterCol0, class IterData0, class IterCol1, class IterData1>
float sparse_alternative_hellinger(
    IterCol0 first0,
    IterCol0 last0,
    IterData0 data0,
    IterCol1 first1,
    IterCol1 last1,
    IterData1 data1
)
{
    float result = 0.0f;
    float l1_norm0 = 0.0f;
    float l1_norm1 = 0.0f;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            float value = std::sqrt((*data0) * (*data1));
            result += value;

            l1_norm0 += (*data0);
            l1_norm1 += (*data1);

            ++first0;
            ++data0;
            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            l1_norm0 += (*data0);

            ++first0;
            ++data0;
        }
        else
        {
            l1_norm1 += (*data1);

            ++first1;
            ++data1;
        }
    }

    // Pass over the tails
    while (first0 != last0)
    {
        l1_norm0 += (*data0);

        ++first0;
        ++data0;
    }
    while (first1 != last1)
    {
        l1_norm1 += (*data1);

        ++first1;
        ++data1;
    }

    if ((l1_norm0 == 0.0f) && (l1_norm1 == 0.0f))
    {
        return 0.0f;
    }
    else if ((l1_norm0 == 0.0f) || (l1_norm1 == 0.0f))
    {
        return FLOAT_MAX;
    }
    else if (result <= 0.0f)
    {
        return FLOAT_MAX;
    }

    result = std::sqrt(l1_norm0 * l1_norm1) / result;
    return std::log2(result);
}


/*
 * @brief Correction function for alternative Hellinger.
 */
inline float correct_alternative_hellinger(float value)
{
    return std::sqrt(1.0f - std::pow(2.0f, -value));
}


template<class Iter>
std::vector<float> rankdata(
    Iter first, Iter last,
    const std::string& method="average"
)
{
    using T = typename std::iterator_traits<Iter>::value_type;
    std::vector<T> arr(first, last);
    std::vector<size_t> sorter(arr.size());
    std::iota(sorter.begin(), sorter.end(), 0);
    std::sort(sorter.begin(), sorter.end(),
        [&arr](size_t i, size_t j) { return arr[i] < arr[j]; });

    std::vector<size_t> inv(sorter.size());
    for (size_t i = 0; i < sorter.size(); ++i)
    {
        inv[sorter[i]] = i;
    }

    if (method == "ordinal")
    {
        std::vector<float> result(inv.size());
        for (size_t i = 0; i < inv.size(); ++i)
        {
            result[i] = inv[i] + 1;
        }
        return result;
    }

    std::vector<T> sorted_arr(arr.size());
    std::vector<int> obs(arr.size());

    sorted_arr[0] = arr[sorter[0]];
    obs[0] = 1;

    for (size_t i = 1; i < arr.size(); ++i)
    {
        sorted_arr[i] = arr[sorter[i]];
        obs[i] = arr[sorter[i]] != arr[sorter[i - 1]];
    }

    std::vector<float> dense(inv.size());
    std::vector<float> obs_partsum(obs.size());
    std::partial_sum(obs.begin(), obs.end(), obs_partsum.begin());

    for (size_t i = 0; i < inv.size(); ++i)
    {
        dense[i] = obs_partsum[inv[i]];
    }

    if (method == "dense")
    {
        return dense;
    }

    // Cumulative counts of each unique value.
    std::vector<size_t> count;
    for (size_t i = 0; i < obs.size(); ++i)
    {
        if (obs[i])
        {
            count.push_back(i);
        }
    }
    count.push_back(obs.size());

    if (method == "max")
    {
        std::vector<float> result(dense.size());
        for (size_t i = 0; i < dense.size(); ++i)
        {
            result[i] = count[dense[i]];
        }
        return result;
    }

    if (method == "min")
    {
        std::vector<float> result(dense.size());
        for (size_t i = 0; i < dense.size(); ++i)
        {
            result[i] = count[dense[i] - 1] + 1.0f;
        }
        return result;
    }

    // Average method
    std::vector<float> result(dense.size());
    for (size_t i = 0; i < dense.size(); ++i)
    {
        result[i] = 0.5 * (count[dense[i]] + count[dense[i] - 1] + 1);
    }
    return result;
}


/*
 * @brief Spearman rho.
 */
template<class Iter0, class Iter1>
float spearmanr(Iter0 first0, Iter0 last0, Iter1 first1)
{
    std::vector<float> x_rank = rankdata(first0, last0);
    Iter1 last1 = first1 + std::distance(first0, last0);
    std::vector<float> y_rank = rankdata(first1, last1);

    return correlation(x_rank.begin(), x_rank.end(), y_rank.begin());
}


/*
 * @brief Jensen Shannon divergence.
 */
template<class Iter0, class Iter1>
float jensen_shannon(Iter0 first0, Iter0 last0, Iter1 first1)
{
    float result = 0.0f;
    float l1_norm0 = 0.0f;
    float l1_norm1 = 0.0f;
    size_t dim = last0 - first0;

    Iter0 it0 = first0;
    Iter1 it1 = first1;
    for (; it0 != last0; ++it0, ++it1)
    {
        l1_norm0 = std::move(l1_norm0) + (*it0);
        l1_norm1 = std::move(l1_norm1) + (*it1);
    }

    l1_norm0 = std::move(l1_norm0) + FLOAT_EPS * dim;
    l1_norm1 = std::move(l1_norm1) + FLOAT_EPS * dim;

    for (; first0 != last0; ++first0, ++first1)
    {
        float pdf0 = ((*first0) + FLOAT_EPS) / l1_norm0;
        float pdf1 = ((*first1) + FLOAT_EPS) / l1_norm1;
        float m = 0.5f * (pdf0 + pdf1);

        result = std::move(result) + 0.5f*(
            pdf0*std::log(pdf0 / m) + pdf1*std::log(pdf1 / m)
        );
    }

    return result;
}


/*
 * @brief Sparse Jensen Shannon divergence.
 */
template<class IterCol0, class IterData0, class IterCol1, class IterData1>
float sparse_jensen_shannon(
    IterCol0 first0,
    IterCol0 last0,
    IterData0 data0,
    IterCol1 first1,
    IterCol1 last1,
    IterData1 data1,
    float dim
)
{
    float result = 0.0f;
    float l1_norm0 = 0.0f;
    float l1_norm1 = 0.0f;

    const IterCol0 _first0 = first0;
    const IterData0 _data0 = data0;
    const IterCol1 _first1 = first1;
    const IterData1 _data1 = data1;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            l1_norm0 += (*data0);
            l1_norm1 += (*data1);

            ++first0;
            ++data0;
            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            l1_norm0 += (*data0);

            ++first0;
            ++data0;
        }
        else
        {
            l1_norm1 += (*data1);

            ++first1;
            ++data1;
        }
    }

    //Pass over the tails
    while (first0 != last0)
    {
        l1_norm0 += (*data0);

        ++first0;
        ++data0;
    }
    while (first1 != last1)
    {
        l1_norm1 += (*data1);

        ++first1;
        ++data1;
    }

    first0 = _first0;
    data0 = _data0;
    first1 = _first1;
    data1 = _data1;

    l1_norm0 += FLOAT_EPS * dim;
    l1_norm1 += FLOAT_EPS * dim;

    int both_indices_zero = dim;

    // Pass through both index lists
    while (first0 != last0 && first1 != last1)
    {
        --both_indices_zero;
        if (*first0 == *first1)
        {
            float pdf0 = ((*data0) + FLOAT_EPS) / l1_norm0;
            float pdf1 = ((*data1) + FLOAT_EPS) / l1_norm1;
            float m = 0.5f * (pdf0 + pdf1);

            result += 0.5f * (
                pdf0 * std::log(pdf0 / m) + pdf1 * std::log(pdf1 / m)
            );

            ++first0;
            ++data0;
            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            float pdf0 = ((*data0) + FLOAT_EPS) / l1_norm0;
            float pdf1 = FLOAT_EPS / l1_norm1;
            float m = 0.5f * (pdf0 + pdf1);

            result += 0.5f * (
                pdf0 * std::log(pdf0 / m) + pdf1 * std::log(pdf1 / m)
            );

            ++first0;
            ++data0;
        }
        else
        {
            float pdf0 = FLOAT_EPS / l1_norm0;
            float pdf1 = ((*data1) + FLOAT_EPS) / l1_norm1;
            float m = 0.5f * (pdf0 + pdf1);

            result += 0.5f * (
                pdf0 * std::log(pdf0 / m) + pdf1 * std::log(pdf1 / m)
            );

            ++first1;
            ++data1;
        }
    }

    //Pass over the tails
    while (first0 != last0)
    {
        --both_indices_zero;
        float pdf0 = ((*data0) + FLOAT_EPS) / l1_norm0;
        float pdf1 = FLOAT_EPS / l1_norm1;
        float m = 0.5f * (pdf0 + pdf1);

        result += 0.5f * (
            pdf0 * std::log(pdf0 / m) + pdf1 * std::log(pdf1 / m)
        );

        ++first0;
        ++data0;
    }
    while (first1 != last1)
    {
        --both_indices_zero;
        float pdf0 = FLOAT_EPS / l1_norm0;
        float pdf1 = ((*data1) + FLOAT_EPS) / l1_norm1;
        float m = 0.5f * (pdf0 + pdf1);

        result += 0.5f * (
            pdf0 * std::log(pdf0 / m) + pdf1 * std::log(pdf1 / m)
        );

        ++first1;
        ++data1;
    }
    float pdf0 = FLOAT_EPS / l1_norm0;
    float pdf1 = FLOAT_EPS / l1_norm1;
    float m = 0.5f * (pdf0 + pdf1);

    result += both_indices_zero * 0.5f * (
        pdf0 * std::log(pdf0 / m) + pdf1 * std::log(pdf1 / m)
    );

    return result;
}


/*
 * @brief Wasserstein 1d.
 */
template<class Iter0, class Iter1>
float wasserstein_1d(Iter0 first0, Iter0 last0, Iter1 first1, float p)
{
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    size_t dim = last0 - first0;

    Iter0 _first0 = first0;
    Iter1 _first1 = first1;

    for (; first0 != last0; ++first0, ++first1)
    {
        sum0 = std::move(sum0) + (*first0);
        sum1 = std::move(sum1) + (*first1);
    }

    // Reset iterators
    first0 = _first0;
    first1 = _first1;

    std::vector<float> cdf0(dim);
    std::vector<float> cdf1(dim);

    auto cdf0_it = cdf0.begin();
    auto cdf1_it = cdf1.begin();

    for (; first0 != last0; ++first0, ++first1, ++cdf0_it, ++cdf1_it)
    {
        *cdf0_it = (*first0) / sum0;
        *cdf1_it = (*first1) / sum1;
    }

    cdf0_it = cdf0.begin();
    cdf1_it = cdf1.begin();
    for (; cdf0_it + 1 != cdf0.end(); ++cdf0_it, ++cdf1_it)
    {
        *(cdf0_it + 1) += *cdf0_it;
        *(cdf1_it + 1) += *cdf1_it;
    }

    return minkowski(cdf0.begin(), cdf0.end(), cdf1.begin(), p);
}


template<class T>
T median(std::vector<T> &vec)
{
    if(vec.empty())
    {
        return 0;
    }
    size_t n = vec.size() / (T)2;
    nth_element(vec.begin(), vec.begin() + n, vec.end());
    auto med = vec[n];
    if (vec.size() % 2 == 0)
    {
        auto max_it = max_element(vec.begin(), vec.begin() + n);
        med = (*max_it + med) / (T)2;
    }
    return med;
}


/*
 * @brief Circular Kantorovich.
 */
template<class Iter0, class Iter1>
float circular_kantorovich(Iter0 first0, Iter0 last0, Iter1 first1, float p)
{
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    size_t dim = last0 - first0;

    for (size_t i = 0; i < dim; ++i)
    {
        sum0 = std::move(sum0) + *(first0 + i);
        sum1 = std::move(sum1) + *(first1 + i);
    }

    std::vector<float> cdf0(dim);
    std::vector<float> cdf1(dim);

    for (size_t i = 0; i < dim; ++i)
    {
        cdf0[i] = *(first0 + i) / sum0;
        cdf1[i] = *(first1 + i) / sum1;
    }

    for (size_t i = 1; i < dim; ++i)
    {
        cdf0[i] += cdf0[i - 1];
        cdf1[i] += cdf1[i - 1];
    }

    std::vector<float> diff_p(dim);

    for (size_t i = 0; i < dim; ++i)
    {
        diff_p[i] = std::pow(cdf0[i] - cdf1[i], p);
    }

    float mu = median(diff_p);
    // Now we just want minkowski distance on the CDFs shifted by mu.
    float result = 0.0f;
    if (p > 2.0f)
    {
        for (size_t i = 0; i < dim; ++i)
        {
            result = std::move(result) + std::abs(
                std::pow(cdf0[i] - cdf1[i] - mu , p)
            );
        }

        return std::pow(result, (1.0f / p));
    }
    else if (p == 2.0f)
    {
        for (size_t i = 0; i < dim; ++i)
        {
            float val = cdf0[i] - cdf1[i] - mu;
            result = std::move(result) + val*val;
        }
        return std::sqrt(result);
    }

    else if (p == 1.0f)
    {
        for (size_t i = 0; i < dim; ++i)
        {
            result = std::move(result) + std::abs(cdf0[i] - cdf1[i] - mu);
        }
        return result;
    }
    throw std::invalid_argument("Invalid p supplied to Kantorvich distance");
}


/*
 * @brief Symmetric Kullback-Leibler divergence.
 */
template<class Iter0, class Iter1>
float symmetric_kl(Iter0 first0, Iter0 last0, Iter1 first1)
{
    float result = 0.0f;
    float l1_norm0 = 0.0f;
    float l1_norm1 = 0.0f;
    int dim = std::distance(first0, last0);

    for (int i = 0; i < dim; ++i)
    {
        l1_norm0 += *first0;
        l1_norm1 += *first1;
        ++first0;
        ++first1;
    }

    l1_norm0 += FLOAT_EPS * dim;
    l1_norm1 += FLOAT_EPS * dim;

    first0 -= dim;
    first1 -= dim;

    for (int i = 0; i < dim; ++i)
    {
        float pdf0 = (*first0 + FLOAT_EPS) / l1_norm0;
        float pdf1 = (*first1 + FLOAT_EPS) / l1_norm1;

        result += pdf0 * std::log(pdf0 / pdf1) + pdf1 * std::log(pdf1 / pdf0);

        ++first0;
        ++first1;
    }

    return result;
}


/*
 * @brief Sparse symmetric Kullback-Leibler divergence.
 */
template<class IterCol0, class IterData0, class IterCol1, class IterData1>
float sparse_symmetric_kl(
    IterCol0 first0,
    IterCol0 last0,
    IterData0 data0,
    IterCol1 first1,
    IterCol1 last1,
    IterData1 data1,
    float dim
)
{
    float result = 0.0f;
    float l1_norm0 = 0.0f;
    float l1_norm1 = 0.0f;

    // Pass through both index lists
    const IterCol0 _first0 = first0;
    const IterData0 _data0 = data0;
    const IterCol1 _first1 = first1;
    const IterData1 _data1 = data1;

    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            l1_norm0 += (*data0);
            l1_norm1 += (*data1);

            ++first0;
            ++data0;
            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            l1_norm0 += (*data0);

            ++first0;
            ++data0;
        }
        else
        {
            l1_norm1 += (*data1);

            ++first1;
            ++data1;
        }
    }

    // Pass over the tails
    while (first0 != last0)
    {
        l1_norm0 += (*data0);

        ++first0;
        ++data0;
    }
    while (first1 != last1)
    {
        l1_norm1 += (*data1);

        ++first1;
        ++data1;
    }

    l1_norm0 += FLOAT_EPS * dim;
    l1_norm1 += FLOAT_EPS * dim;

    first0 = _first0;
    data0 = _data0;
    first1 = _first1;
    data1 = _data1;

    while (first0 != last0 && first1 != last1)
    {
        if (*first0 == *first1)
        {
            float pdf0 = ((*data0) + FLOAT_EPS) / l1_norm0;
            float pdf1 = ((*data1) + FLOAT_EPS) / l1_norm1;

            result += pdf0 * std::log(pdf0 / pdf1) + pdf1 * std::log(pdf1 / pdf0);

            ++first0;
            ++data0;
            ++first1;
            ++data1;
        }
        else if (*first0 < *first1)
        {
            float pdf0 = ((*data0) + FLOAT_EPS) / l1_norm0;
            float pdf1 = FLOAT_EPS / l1_norm1;

            result += pdf0 * std::log(pdf0 / pdf1) + pdf1 * std::log(pdf1 / pdf0);

            ++first0;
            ++data0;
        }
        else
        {
            float pdf0 = FLOAT_EPS / l1_norm0;
            float pdf1 = ((*data1) + FLOAT_EPS) / l1_norm1;

            result += pdf0 * std::log(pdf0 / pdf1) + pdf1 * std::log(pdf1 / pdf0);

            ++first1;
            ++data1;
        }
    }

    // Pass over the tails
    while (first0 != last0)
    {
        float pdf0 = ((*data0) + FLOAT_EPS) / l1_norm0;
        float pdf1 = FLOAT_EPS / l1_norm1;

        result += pdf0 * std::log(pdf0 / pdf1) + pdf1 * std::log(pdf1 / pdf0);

        ++first0;
        ++data0;
    }
    while (first1 != last1)
    {
        float pdf0 = FLOAT_EPS / l1_norm0;
        float pdf1 = ((*data1) + FLOAT_EPS) / l1_norm1;

        result += pdf0 * std::log(pdf0 / pdf1) + pdf1 * std::log(pdf1 / pdf0);

        ++first1;
        ++data1;
    }

    return result;
}


/*
 * @brief Class template representing a distance function.
 *
 * This class encapsulates a dense metric function, a sparse metric function
 * and a correction function for calculating distances between two points.
 *
 * @tparam Dense The dense metric function.
 * @tparam Sparse The sparse metric function.
 * @tparam Correction function if a faster alternative of the distance function
 * is used. Otherwise this is simply the identity function.
 */
template<
    float (*Dense)(It, It, It),
    float (*Sparse)(size_t*, size_t*, It, size_t*, size_t*, It),
    float (*Correction)(float)
>
class Dist
{
public:

    /*
     * Some metrics have alternative forms that allow for faster calculations,
     * but these forms may produce slightly different distances. This function
     * applies distance correction. If no alternative metric is used, this is
     * simply the identity function.
     */
    inline float correction(float value) const { return Correction(value); }

    /*
     * Calculates the distance between two data points using the dense metric
     * function.
     *
     * @param data The input data matrix.
     * @param idx0 The index of the first data point.
     * @param idx1 The index of the second data point.
     *
     * @return The distance between the two data points.
     */
    inline float operator()
    (
        const Matrix<float> &data,
        int idx0,
        int idx1
    ) const
    {
        return Dense(
            data.begin(idx0), data.end(idx0), data.begin(idx1)
        );
    }

    /*
     * Calculates the distance between a data point and a query point using the
     * dense metric function.
     *
     * @param data The input data matrix.
     * @param idx_d The index of the data point.
     * @param query_data The query data matrix.
     * @param idx_q The index of the query point.
     *
     * @return The distance between the data point and the query point.
     */
    inline float operator()
    (
        const Matrix<float> &data,
        int idx_d,
        const Matrix<float> &query_data,
        int idx_q
    ) const
    {
        return Dense(
            data.begin(idx_d), data.end(idx_d), query_data.begin(idx_q)
        );
    }

    /*
     * Calculates the distance between two data points using the sparse metric
     * function.
     *
     * @param data The input CSR matrix.
     * @param idx0 The index of the first data point.
     * @param idx1 The index of the second data point.
     *
     * @return The distance between the two data points.
     */
    inline float operator()
    (
        const CSRMatrix<float> &data,
        int idx0,
        int idx1
    ) const
    {
        return Sparse(
            data.begin_col(idx0),
            data.end_col(idx0),
            data.begin_data(idx0),
            data.begin_col(idx1),
            data.end_col(idx1),
            data.begin_data(idx1)
        );
    }

    /*
     * Calculates the distance between a data point and a query point using the
     * sparse metric function.
     *
     * @param data The input CSR matrix.
     * @param idx_d The index of the data point.
     * @param query_data The query CSR matrix.
     * @param idx_q The index of the query point.
     *
     * @return The distance between the data point and the query point.
     */
    inline float operator()
    (
        const CSRMatrix<float> &data,
        int idx_d,
        const CSRMatrix<float> &query_data,
        int idx_q
    ) const
    {
        return Sparse(
            data.begin_col(idx_d),
            data.end_col(idx_d),
            data.begin_data(idx_d),
            query_data.begin_col(idx_q),
            query_data.end_col(idx_q),
            query_data.begin_data(idx_q)
        );
    }

};


/*
 * @brief This class template is similar to the 'Dist' class, but with
 * the difference that the dense and the sparse metrics depend on an external
 * float variable 'p_metric'.
 */
template<
    float (*Dense)(It, It, It, float),
    float (*Sparse)(size_t*, size_t*, It, size_t*, size_t*, It, float),
    float (*Correction)(float)
>
class DistP
{
public:

    float p_metric;

    explicit DistP(float p)
        : p_metric(p)
    {
    }

    inline float correction(float value) const { return Correction(value); }

    inline float operator()
    (
        const Matrix<float> &data,
        int idx0,
        int idx1
    ) const
    {
        return Dense(
            data.begin(idx0), data.end(idx0), data.begin(idx1), p_metric
        );
    }

    inline float operator()
    (
        const Matrix<float> &data,
        int idx_d,
        const Matrix<float> &query_data,
        int idx_q
    ) const
    {
        return Dense(
            data.begin(idx_d), data.end(idx_d), query_data.begin(idx_q), p_metric
        );
    }

    inline float operator()
    (
        const CSRMatrix<float> &data,
        int idx0,
        int idx1
    ) const
    {
        return Sparse(
            data.begin_col(idx0),
            data.end_col(idx0),
            data.begin_data(idx0),
            data.begin_col(idx1),
            data.end_col(idx1),
            data.begin_data(idx1),
            p_metric
        );
    }

    inline float operator()
    (
        const CSRMatrix<float> &data,
        int idx_d,
        const CSRMatrix<float> &query_data,
        int idx_q
    ) const
    {
        return Sparse(
            data.begin_col(idx_d),
            data.end_col(idx_d),
            data.begin_data(idx_d),
            query_data.begin_col(idx_q),
            query_data.end_col(idx_q),
            query_data.begin_data(idx_q),
            p_metric
        );
    }

};


/*
 * @brief This class template is similar to the 'Dist' class, but with the
 * difference that the sparse metric needs knowledge of the dimension.
 */
template<
    float (*Dense)(It, It, It),
    float (*Sparse)(size_t*, size_t*, It, size_t*, size_t*, It, float),
    float (*Correction)(float)
>
class DistD
{
public:

    float dim;

    explicit DistD(float d)
        : dim(d)
    {
    }

    inline float correction(float value) const { return Correction(value); }

    inline float operator()
    (
        const Matrix<float> &data,
        int idx0,
        int idx1
    ) const
    {
        return Dense(
            data.begin(idx0), data.end(idx0), data.begin(idx1)
        );
    }

    inline float operator()
    (
        const Matrix<float> &data,
        int idx_d,
        const Matrix<float> &query_data,
        int idx_q
    ) const
    {
        return Dense(
            data.begin(idx_d), data.end(idx_d), query_data.begin(idx_q)
        );
    }

    inline float operator()
    (
        const CSRMatrix<float> &data,
        int idx0,
        int idx1
    ) const
    {
        return Sparse(
            data.begin_col(idx0),
            data.end_col(idx0),
            data.begin_data(idx0),
            data.begin_col(idx1),
            data.end_col(idx1),
            data.begin_data(idx1),
            dim
        );
    }

    inline float operator()
    (
        const CSRMatrix<float> &data,
        int idx_d,
        const CSRMatrix<float> &query_data,
        int idx_q
    ) const
    {
        return Sparse(
            data.begin_col(idx_d),
            data.end_col(idx_d),
            data.begin_data(idx_d),
            query_data.begin_col(idx_q),
            query_data.end_col(idx_q),
            query_data.begin_data(idx_q),
            dim
        );
    }

};


// METRICS WITH NO PARAMETERS
using AltCosine = Dist<alternative_cosine, sparse_alternative_cosine, identity>;
using AltDot = Dist<alternative_dot, sparse_alternative_dot, identity>;
using AltJaccard = Dist<
    alternative_jaccard, sparse_alternative_jaccard, correct_alternative_jaccard
>;
using BrayCurtis = Dist<bray_curtis, sparse_bray_curtis, identity>;
using Canberra = Dist<canberra, sparse_canberra, identity>;
using Chebyshev = Dist<chebyshev, sparse_chebyshev, identity>;
using Cosine = Dist<cosine, sparse_cosine, identity>;
using Dice = Dist<dice, sparse_dice, identity>;
using Dot = Dist<dot, sparse_dot, identity>;
using Euclidean = Dist<squared_euclidean, sparse_squared_euclidean, std::sqrt>;
using Hamming = Dist<hamming, sparse_hamming, identity>;
using Haversine = Dist<haversine, nullptr, identity>;
using Hellinger = Dist<hellinger, sparse_hellinger, identity>;
using Jaccard = Dist<jaccard, sparse_jaccard, identity>;
using Manhattan = Dist<manhattan, sparse_manhattan, identity>;
using Matching = Dist<matching, sparse_matching, identity>;
using SokalSneath = Dist<sokal_sneath, sparse_sokal_sneath, identity>;
using SpearmanR = Dist<spearmanr, nullptr, identity>;
using SqEuclidean = Dist<squared_euclidean, sparse_squared_euclidean, identity>;
using TrueAngular = Dist<true_angular, sparse_true_angular, identity>;
using Tsss = Dist<tsss, sparse_tsss, identity>;


// METRICS WITH ONE FLOAT PARAMETER
using CircularKantorovich = DistP<circular_kantorovich, nullptr, identity>;
using Minkowski = DistP<minkowski, sparse_minkowski, identity>;
using Wasserstein = DistP<wasserstein_1d, nullptr, identity>;


// METRICS WHERE THE SPARSE VERSION NEEDS KNOWLEDGE OF THE DIMENSION
using Correlation = DistD<correlation, sparse_correlation, identity>;
using JensenShannon = DistD<jensen_shannon, sparse_jensen_shannon, identity>;
using Kulsinski = DistD<kulsinski, sparse_kulsinski, identity>;
using RogersTanimoto = DistD<rogers_tanimoto, sparse_rogers_tanimoto, identity>;
using RussellRao = DistD<russellrao, sparse_russellrao, identity>;
using SokalMichener = DistD<sokal_michener, sparse_sokal_michener, identity>;
using SymmetriyKL = DistD<symmetric_kl, sparse_symmetric_kl, identity>;
using Yule = DistD<yule, sparse_yule, identity>;


} // namespace nndescent
