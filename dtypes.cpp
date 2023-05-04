#include <chrono>
#include <iostream>
#include <limits>
#include <random>
#include <string>

#include "dtypes.h"

Timer timer_dtyp;
Timer timer_dtyp2;

void print(IntMatrix &matrix)
{
    for (size_t i = 0; i < matrix.size(); i++)
    {
        std::cout << i << ": ";
        for (size_t j = 0; j < matrix[i].size(); ++j)
        {
            std::cout << " " << matrix[i][j];
        }
        std::cout << "\n";
    }
}

void print(std::vector<IntMatrix> &array)
{
    for (size_t i = 0; i < array.size(); i++)
    {
        std::cout << "[" <<i << "]\n";
        print(array[i]);
        std::cout << "\n";
    }
}

// Print the data as 2d map.
void print_map(Matrix<float> matrix)
{
    // Calculate maximum coordinates.
    int x_max = 0;
    int y_max = 0;
    for (size_t j = 0; j < matrix.nrows(); ++j)
    {
        int x_cur = matrix(j, 0);
        int y_cur = matrix(j, 1);
        x_max = (x_cur > x_max) ? x_cur : x_max;
        y_max = (y_cur > y_max) ? y_cur : y_max;
    }

    // Initialize 2d map to be printed to console.
    std::vector<std::vector<char>> map(
        y_max + 1, std::vector<char>(x_max+ 4)
    );
    for (int i = 0; i <= y_max; ++i)
    {
        for (int j = 0; j <= x_max; ++j)
        {
            map[i][j] = ' ';
        }
        map[i][x_max + 1] = '|';
        map[i][x_max + 2] = '\n';
        map[i][x_max + 3] = '\0';
    }

    // Draw data as single digits.
    for (size_t j = 0; j < matrix.nrows(); ++j)
    {
        int x_cur = matrix(j, 0);
        int y_cur = matrix(j, 1);
        map[y_cur][x_cur] = '0' + j % 10;
    }

    // Print map.
    for (int i = 0; i <= y_max; ++i)
    {
        std::string row(map[i].begin(), map[i].end());
        std::cout << row;
    }
}

void print(IntVec vec)
{
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i)
    {
        std::cout << vec[i];
        if (i + 1 != vec.size())
        {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}

void print(std::vector<float> vec)
{
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i)
    {
        std::cout << vec[i];
        if (i + 1 != vec.size())
        {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}

