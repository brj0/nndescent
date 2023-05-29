/**
 * @file distances.h
 *
 * @brief Distance functions.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace nndescent
{


const float PI = 3.14159265358979f;
const float FLOAT_MAX = std::numeric_limits<float>::max();
const float FLOAT_MIN = std::numeric_limits<float>::min();
const float FLOAT_EPS = std::numeric_limits<float>::epsilon();


/**
 * @brief Identity function.
 */
template<class T>
T identity_function(T value)
{
    return value;
}


/**
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


/**
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


/**
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
float standardised_euclidean
(
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


/**
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
    while (first0 != last0)
    {
        result = std::move(result) + std::abs(*first0 - *first1);
        ++first0;
        ++first1;
    }
    return result;
}


/**
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


/**
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


/**
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
float weighted_minkowski
(
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


/**
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


/**
 * @brief Hamming distance.
 */
template<class Iter0, class Iter1>
float hamming(Iter0 first0, Iter0 last0, Iter1 first1)
{
    int result = 0;
    int size = last0 - first0;
    while (first0 != last0)
    {
        if (*first0 != *first1)
        {
            ++result;
        }
        ++first0;
        ++first1;
    }
    return (float)(result) / size;
}


/**
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


/**
 * @brief Bray–Curtis dissimilarity.
 */
template<class Iter0, class Iter1>
float bray_curtis(Iter0 first0, Iter0 last0, Iter1 first1)
{
    int numerator = 0;
    int denominator = 0;
    while (first0 != last0)
    {
        numerator = std::move(numerator) + std::abs(*first0 - *first1);
        denominator = std::move(denominator) + std::abs(*first0 + *first1);
        ++first0;
        ++first1;
    }
    if (denominator > 0)
    {
        return (float)(numerator)/denominator;
    }
    return 0.0f;
}


/**
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
    return (float)(num_non_zero - num_equal) / num_non_zero;
}


/**
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


/**
 * @brief Correction function for Jaccard distance.
 */
template<class Iter>
void correct_alternative_jaccard(Iter first0, Iter last0, Iter first1)
{
    for (; first0 != last0; ++first0, ++first1)
    {
        (*first1) = 1.0f - std::pow(2.0f, -*first0);
    }
}


/**
 * @brief Matching.
 */
template<class Iter0, class Iter1>
float matching(Iter0 first0, Iter0 last0, Iter1 first1)
{
    int num_not_equal = 0;
    size_t size = last0 - first0;
    for (; first0 != last0; ++first0, ++first1)
    {
        bool first0_true = (*first0 != 0);
        bool first1_true = (*first1 != 0);
        num_not_equal = std::move(num_not_equal) + (first0_true != first1_true);
    }
    return (float)num_not_equal / size;
}


/**
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
    return (float)num_not_equal / (2.0f * num_true_true + num_not_equal);
}


/**
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
    return (float)(num_not_equal - num_true_true + dim) / (
        num_not_equal + dim
    );
}


/**
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


/**
 * @brief Russell-Rao dissimilarity.
 */
template<class Iter0, class Iter1>
float russellrao(Iter0 first0, Iter0 last0, Iter1 first1)
{
    int num_true_true = 0;
    size_t dim = last0 - first0;
    int first0_non0 = std::count_if(
        first0, last0, [&](auto const &x){ return x != 0; }
    );
    int first1_non0 = std::count_if(
        first1, first1 + dim, [&](auto const &x){ return x != 0; }
    );
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


/**
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


/**
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


/**
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


/**
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


/**
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


/**
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


/**
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


/**
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


/**
 * @brief Correction function for cosine.
 */
template<class Iter>
void correct_alternative_cosine(Iter first0, Iter last0, Iter first1)
{
    for (; first0 != last0; ++first0, ++first1)
    {
        (*first1) = 1.0f - std::pow(2.0f, -*first0);
    }
}


/**
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


/**
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


/**
 * @brief Correction function true angular.
 */
template<class Iter>
void true_angular_from_alt_cosine(Iter first0, Iter last0, Iter first1)
{
    for (; first0 != last0; ++first0, ++first1)
    {
        (*first1) = 1.0f - std::acos(std::pow(2.0f, -*first0)) / PI;
    }
}


/**
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


/**
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


/**
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


/**
 * @brief Correction function for alternative Hellinger.
 */
template<class Iter>
void correct_alternative_hellinger(Iter first0, Iter last0, Iter first1)
{
    for (; first0 != last0; ++first0, ++first1)
    {
        (*first1) = std::sqrt(1.0f - std::pow(2.0f, -*first0));
    }
}


template<class Iter>
std::vector<float> rankdata
(
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


/**
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


/**
 * @brief Jensen Shannon divergence.
 */
template<class Iter0, class Iter1>
float jensen_shannon_divergence(Iter0 first0, Iter0 last0, Iter1 first1)
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


/**
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


/**
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


/**
 * @brief Symmetric Kullback-Leibler divergence.
 */
template<class Iter0, class Iter1>
float symmetric_kl_divergence(Iter0 first0, Iter0 last0, Iter1 first1)
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


} // namespace nndescent