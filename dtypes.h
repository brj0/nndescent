#pragma once

#include <algorithm>
#include <random>
#include <vector>
#include <valarray>
#include <chrono>
#include <string>
#include <iostream>

#include "utils.h"

typedef std::vector<int> IntVec;
typedef std::vector<IntVec> IntMatrix;

const float EPS = 1e-8;
const float MAX_FLOAT = std::numeric_limits<float>::max();
const int MAX_INT = std::numeric_limits<int>::max();
const char FALSE = '0';
const char TRUE = '1';

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
        size_t nrows() const {return n_rows;}
        size_t ncols() const {return n_cols;}
        float* operator[](size_t i){ return &val[i*n_cols]; }
        auto begin(size_t i) const {return val.begin() + i*n_cols;}
        auto end(size_t i) const {return val.begin() + (i + 1)*n_cols;}
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

// Timer for debugging
class Timer
{
    private:
        std::chrono::time_point<std::chrono::system_clock> time;
    public:
        void start()
        {
            time = std::chrono::high_resolution_clock::now();
        }
        void stop(std::string text)
        {
            auto end = std::chrono::high_resolution_clock::now();
            std::cout << "Time passed: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       end - time
                   ).count()
                << " ms ("
                << text
                << ")\n";
            this->start();
        }
};

// Binary max-heap data structure.
template <class NodeType>
class Heap
{
    private:
        std::vector<NodeType> nodes_;
    public:
        Heap() {}
        Heap(size_t n):
            nodes_(std::vector<NodeType>(n)) {}
        Heap(size_t n, NodeType node):
            nodes_(std::vector<NodeType>(n, node)) {}
        auto max() {return nodes_[0].key;}
        size_t size() {return nodes_.size();}
        size_t valid_idx_size();
        void insert(NodeType node);
        void controlled_insert(NodeType node);
        void siftup(int i_node);
        void siftdown(int i_node);
        bool node_in_heap(NodeType &node);
        int update_max(NodeType &node);
        int replace_max(NodeType &node);
        void update_max(NodeType &node, unsigned int limit);
        NodeType operator [] (int i) const {return nodes_[i];}
        NodeType& operator [] (int i) {return nodes_[i];}
};

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

void print(IntVec vec);
void print(std::vector<float> vec);
void print(std::vector<IntMatrix> &array);
void print_map(Matrix<float> matrix);
void print(IntMatrix &matrix);

std::vector<IntMatrix> make_forest(
    const Matrix<float> &data,
    int n_trees,
    int leaf_size,
    RandomState &rng_state
);


template <class NodeType>
void Heap<NodeType>::siftdown(int i_node)
{
    int current = i_node;
    while ((current*2 + 1) < (int) nodes_.size())
    {
        int left_child = current*2 + 1;
        int right_child = left_child + 1;
        int swap = current;

        if (nodes_[swap].key < nodes_[left_child].key)
        {
            swap = left_child;
        }

        if (
            (right_child < (int) nodes_.size()) &&
            (nodes_[swap].key < nodes_[right_child].key)
        )
        {
            swap = right_child;
        }

        if (swap == current)
        {
            return;
        }

        NodeType tmp = nodes_[current];
        nodes_[current] = nodes_[swap];
        nodes_[swap] = tmp;
        current = swap;
    }
}

template <class NodeType>
void Heap<NodeType>::siftup(int i_node)
{
    int current = i_node;
    int parent = (current - 1)/2;
    while ((parent >= 0) && (nodes_[parent].key < nodes_[current].key))
    {
        NodeType tmp = nodes_[current];
        nodes_[current] = nodes_[parent];
        nodes_[parent] = tmp;

        current = parent;
        parent = (current - 1)/2;
    }
}

template <class NodeType>
bool Heap<NodeType>::node_in_heap(NodeType &node)
{
    for (size_t i = 0; i < this->size(); i++)
    {
        if (nodes_[i].idx == node.idx)
        {
            return true;
        }
    }
    return false;
}

template <class NodeType>
size_t Heap<NodeType>::valid_idx_size()
{
    size_t count = std::count_if(
        nodes_.begin(),
        nodes_.end(),
        [&](NodeType const &node){ return node.idx != NONE; }
    );
    return count;
}

template <class NodeType>
void Heap<NodeType>::insert(NodeType node)
{
    nodes_.push_back(node);
    this->siftup(this->size() - 1);
}

// Insert 'node' if it is not allready in Heap
template <class NodeType>
void Heap<NodeType>::controlled_insert(NodeType node)
{
    if (!this->node_in_heap(node))
    {
        this->insert(node);
    }
}

// Replaces max-element by 'node' if it has a smaller distance.
template <class NodeType>
int Heap<NodeType>::update_max(NodeType &node)
{
    if (node.key >= nodes_[0].key || this->node_in_heap(node))
    {
        return 0;
    }
    nodes_[0] = node;
    this->siftdown(0);
    return 1;
}

// Replaces max-element by 'node'.
template <class NodeType>
int Heap<NodeType>::replace_max(NodeType &node)
{
    if (this->node_in_heap(node))
    {
        return 0;
    }
    nodes_[0] = node;
    this->siftdown(0);
    return 1;
}

// Replaces max-element by 'node' if it has a smaller distance or if heap
// has size smaller than 'limit'.
template <class NodeType>
void Heap<NodeType>::update_max(NodeType &node, unsigned int limit)
{
    if (this->size() < limit)
    {
        this->controlled_insert(node);
    }
    else
    {
        this->update_max(node);
    }
}

