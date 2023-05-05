#include <iostream>
#include <string>

#include "dtypes.h"


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
