#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "utils.h"


typedef std::vector<int> IntVec;
typedef std::vector<IntVec> IntMatrix;

template <class T>
class Matrix
{
    public:
        Matrix() {}
        Matrix(size_t rows, size_t cols);
        Matrix(size_t rows, size_t cols, const T &const_val);
        Matrix(size_t rows, std::vector<T> &val);
        T& operator()(size_t i, size_t j);
        const T operator()(size_t i, size_t j) const;
        size_t nrows() const { return n_rows; }
        size_t ncols() const { return n_cols; }
        float* operator[](size_t i){ return &val[i*n_cols]; }
        auto begin(size_t i) const { return val.begin() + i*n_cols; }
        auto end(size_t i) const { return val.begin() + (i + 1)*n_cols; }
        T *to_pnt() { return &val[0];}
        std::vector<T> val;

    private:
        size_t n_rows;
        size_t n_cols;
};

template <class T>
Matrix<T>::Matrix(size_t rows, size_t cols)
    : val(rows * cols)
    , n_rows(rows)
    , n_cols(cols)
{
}

template <class T>
Matrix<T>::Matrix(size_t rows, size_t cols, const T &const_val)
    : val(rows * cols, const_val)
    , n_rows(rows)
    , n_cols(cols)
{
}

template <class T>
Matrix<T>::Matrix(size_t rows, std::vector<T> &val)
    : val(val)
    , n_rows(rows)
    , n_cols(val.size()/rows)
{
}

template <class T>
inline
T& Matrix<T>::operator()(size_t i, size_t j)
{
    return val[i * n_cols + j];
}

template <class T>
inline
const T Matrix<T>::operator()(size_t i, size_t j) const
{
    return val[i * n_cols + j];
}

template <class T>
std::ostream& operator<<(std::ostream &out, Matrix<T> const& matrix)
{
    out << "[";

    for(size_t i = 0; i < matrix.nrows(); ++i)
    {
        if (i > 0)
        {
            out << " ";
        }
        out << "[";
        for(size_t j = 0; j < matrix.ncols(); j++)
        {
            out << matrix(i, j);
            if (j + 1 != matrix.ncols())
            {
                out << ", ";
            }
        }
        out << "]";
        if (i + 1 != matrix.nrows())
        {
            out << ",\n";
        }
    }

    out << "]\n";
    return out;
}

template <class KeyType>
class HeapList
{
    private:
        size_t n_heaps;
        size_t n_nodes;

    public:
        Matrix<int> indices;
        Matrix<KeyType> keys;
        // std::vector<bool> is special from all other std::vector
        // specializations. Use char instead
        Matrix<char> flags;
        HeapList(size_t n_heaps, size_t n_nodes, KeyType key0, char flag0)
            : n_heaps(n_heaps)
            , n_nodes(n_nodes)
            , indices(n_heaps, n_nodes, NONE)
            , keys(n_heaps, n_nodes, key0)
            , flags(n_heaps, n_nodes, flag0)
            {}
        HeapList(size_t n_heaps, size_t n_nodes, KeyType key0)
            : n_heaps(n_heaps)
            , n_nodes(n_nodes)
            , indices(n_heaps, n_nodes, NONE)
            , keys(n_heaps, n_nodes, key0)
            , flags(0, 0)
            {}
        size_t nheaps() {return n_heaps;}
        size_t nnodes() {return n_nodes;}
        bool noflags() { return flags.nrows() == 0; }
        KeyType max(size_t i) { return keys(i, 0); }
        size_t size(size_t i);
        int checked_push(size_t i, int idx, KeyType key, char flag);
        int checked_push(size_t i, int idx, KeyType key);
        void siftdown(size_t i, size_t stop);
        void heapsort();
};

// TODO parallel
template <class KeyType>
void HeapList<KeyType>::heapsort()
{
    int tmp_id;
    KeyType tmp_key;
    // char tmp_flag;

    for (size_t i = 0; i < n_heaps; ++i)
    {
        for (size_t j = n_nodes - 1; j > 0; --j)
        {
            tmp_id = indices(i, 0);
            tmp_key = keys(i, 0);
            // tmp_flag = flags(i, 0);

            indices(i, 0) = indices(i, j);
            keys(i, 0) = keys(i, j);
            // flags(i, 0) = flags(i, j);

            indices(i, j) = tmp_id;
            keys(i, j) = tmp_key;
            // flags(i, j) = tmp_flag;

            this->siftdown(i, j);
        }
    }
}

template <class KeyType>
void HeapList<KeyType>::siftdown(size_t i, size_t stop)
{
    // Siftdown: Descend the heap, swapping values until the max heap
    // criterion is met.
    KeyType key = keys(i, 0);
    int idx = indices(i, 0);
    // char flag = flags(i, 0);

    size_t current = 0;
    size_t left_child;
    size_t right_child;
    size_t swap;

    while (true)
    {
        left_child = 2*current + 1;
        right_child = left_child + 1;

        if (left_child >= stop)
        {
            break;
        }
        else if (right_child >= stop)
        {
            if (keys(i, left_child) > key)
            {
                swap = left_child;
            }
            else
            {
                break;
            }
        }
        else if (keys(i, left_child) >= keys(i, right_child))
        {
            if (keys(i, left_child) > key)
            {
                swap = left_child;
            }
            else
            {
                break;
            }
        }
        else
        {
            if (keys(i, right_child) > key)
            {
                swap = right_child;
            }
            else
            {
                break;
            }
        }
        indices(i, current) = indices(i, swap);
        keys(i, current) = keys(i, swap);
        // flags(i, current) = flags(i, swap);

        current = swap;
    }
    // Insert node at current position.
    indices(i, current) = idx;
    keys(i, current) = key;
    // flags(i, current) = flag;
}


template <class KeyType>
int HeapList<KeyType>::checked_push(size_t i, int idx, KeyType key, char flag)
{
    if (key >= keys(i, 0))
    {
        return 0;
    }

    // Break if we already have this element.
    for (size_t j = 0; j < n_nodes; ++j)
    {
        if (indices(i, j) == idx)
        {
            return 0;
        }
    }

    // Siftdown: Descend the heap, swapping values until the max heap
    // criterion is met.
    size_t current = 0;
    size_t left_child;
    size_t right_child;
    size_t swap;

    while (true)
    {
        left_child = 2*current + 1;
        right_child = left_child + 1;

        if (left_child >= n_nodes)
        {
            break;
        }
        else if (right_child >= n_nodes)
        {
            if (keys(i, left_child) > key)
            {
                swap = left_child;
            }
            else
            {
                break;
            }
        }
        else if (keys(i, left_child) >= keys(i, right_child))
        {
            if (keys(i, left_child) > key)
            {
                swap = left_child;
            }
            else
            {
                break;
            }
        }
        else
        {
            if (keys(i, right_child) > key)
            {
                swap = right_child;
            }
            else
            {
                break;
            }
        }
        indices(i, current) = indices(i, swap);
        keys(i, current) = keys(i, swap);
        flags(i, current) = flags(i, swap);

        current = swap;
    }
    // Insert node at current position.
    indices(i, current) = idx;
    keys(i, current) = key;
    flags(i, current) = flag;

    return 1;
}

template <class KeyType>
int HeapList<KeyType>::checked_push(size_t i, int idx, KeyType key)
{
    if (key >= keys(i, 0))
    {
        return 0;
    }

    // Break if we already have this element.
    for (size_t j = 0; j < n_nodes; ++j)
    {
        if (indices(i, j) == idx)
        {
            return 0;
        }
    }

    // Siftdown: Descend the heap, swapping values until the max heap
    // criterion is met.
    size_t current = 0;
    size_t left_child;
    size_t right_child;
    size_t swap;

    while (true)
    {
        left_child = 2*current + 1;
        right_child = left_child + 1;

        if (left_child >= n_nodes)
        {
            break;
        }
        else if (right_child >= n_nodes)
        {
            if (keys(i, left_child) > key)
            {
                swap = left_child;
            }
            else
            {
                break;
            }
        }
        else if (keys(i, left_child) >= keys(i, right_child))
        {
            if (keys(i, left_child) > key)
            {
                swap = left_child;
            }
            else
            {
                break;
            }
        }
        else
        {
            if (keys(i, right_child) > key)
            {
                swap = right_child;
            }
            else
            {
                break;
            }
        }
        indices(i, current) = indices(i, swap);
        keys(i, current) = keys(i, swap);

        current = swap;
    }
    // Insert node at current position.
    indices(i, current) = idx;
    keys(i, current) = key;

    return 1;
}

template <class KeyType>
size_t HeapList<KeyType>::size(size_t i)
{
    size_t count = std::count_if(
        indices.begin(i),
        indices.end(i),
        [&](int const &idx){ return idx != NONE; }
    );
    return count;
}

// // Auxiliary function for recursively printing a binary Heap.
template <class KeyType>
void _add_heap_from_to_stream
(
    std::ostream &out,
    std::string prefix,
    HeapList<KeyType> heaplist,
    size_t i,
    size_t from,
    char is_left
)
{
    if (from >= heaplist.nnodes())
    {
        return;
    }

    out << prefix;
    out << (is_left && (from + 1 < heaplist.nnodes()) ? "├──" : "└──");

    // Print current node
    char flag = heaplist.noflags() ? 'x' : heaplist.flags(i, from);
    out << "(idx=" << heaplist.indices(i, from)
        << " key=" << heaplist.keys(i, from)
        << " flag=" << flag
        << ")\n";
    std::string prefix_children = prefix + (is_left ? "│   " : "    ");

    // Add children of current node.
    _add_heap_from_to_stream(out, prefix_children, heaplist, i, from*2 + 1, true);
    _add_heap_from_to_stream(out, prefix_children, heaplist, i, from*2 + 2, false);
}

template <class KeyType>
void add_heap_to_stream(std::ostream &out, HeapList<KeyType> heaplist, size_t i)
{
    out << i << " [size=" << heaplist.nheaps() << "]\n";
    _add_heap_from_to_stream(out, "    ", heaplist, i, 0, false);
}


template <class KeyType>
std::ostream& operator<<(std::ostream &out, HeapList<KeyType> &heaplist)
{
    out << "HeapList(n_heaps=" << heaplist.nheaps() << ", n_nodes="
        << heaplist.nnodes() << ", KeyType=" << typeid(KeyType).name()
        << ",\n";
    for (size_t i = 0; i < heaplist.nheaps(); ++i)
    {
        out << "    ";
        add_heap_to_stream(out, heaplist, i);
    }
    out << ")\n";
    return out;
}

void print_map(Matrix<float> matrix);

template <class T>
std::ostream& operator<<(std::ostream &out, std::vector<T> &vec)
{
    out << "[";
    for (size_t i = 0; i < vec.size(); ++i)
    {
        out << vec[i];
        if (i + 1 != vec.size())
        {
            out << ", ";
        }
    }
    out << "]";
    return out;
}

template <class T>
std::ostream& operator<<(std::ostream &out, std::vector<std::vector<T>> &matrix)
{
    out << "[\n";
    for(size_t i = 0; i < matrix.size(); ++i)
    {
        out << "    " << i << ": " <<  matrix[i] << ",\n";
    }
    out << "]";
    return out;
}
