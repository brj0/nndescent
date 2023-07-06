/**
 * @file dtypes.cpp
 *
 * @brief Data types used (Matrix, Heap, HeapList).
 */


#include "dtypes.h"


namespace nndescent
{


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


std::ostream& operator<<(std::ostream &out, NNUpdate &update)
{
    out << "(idx0=" << update.idx0
        << ", idx1=" << update.idx1
        << ", key=" << update.key
        << ")";
    return out;
}


std::ostream& operator<<(std::ostream &out, std::vector<NNUpdate> &updates)
{
    out << "[";
    for (size_t i = 0; i < updates.size(); ++i)
    {
        if (i > 0)
        {
            out << " ";
        }
        out << updates[i];
        if (i + 1 != updates.size())
        {
            out << ",\n";
        }
    }
    out << "]\n";
    return out;
}


} // namespace nndescent
