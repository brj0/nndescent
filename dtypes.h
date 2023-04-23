#pragma once

#include <algorithm>
#include <random>
#include <vector>
#include <chrono>
#include <string>

#include "utils.h"

/*
 * DASGUPTA, Sanjoy; FREUND, Yoav. Random projection trees and low dimensional
 * manifolds. In: Proceedings of the fortieth annual ACM symposium on Theory of
 * computing. 2008. S. 537-546.
 *
 * https://dl.acm.org/doi/pdf/10.1145/1374376.1374452
 * https://cseweb.ucsd.edu/~dasgupta/papers/rptree-stoc.pdf
 */

typedef std::vector<std::vector<double>> Matrix;
typedef std::vector<int> IntVec;
typedef std::vector<IntVec> IntMatrix;

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

// Elements of the nearest neighbors heap
typedef struct
{
    int idx;
    double key;
    bool visited;
} NNNode;

// Elements of the randomly selected candidates for loca joins
typedef struct
{
    int idx;
    uint64_t key;
} RandNode;

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
        size_t size() {return nodes_.size();}
        void insert(NodeType node);
        void controlled_insert(NodeType node);
        void siftup(int i_node);
        void siftdown(int i_node);
        bool node_in_heap(NodeType &node);
        int update_max(NodeType &node);
        void update_max(NodeType &node, unsigned int limit);
        NodeType operator [] (int i) const {return nodes_[i];}
        NodeType& operator [] (int i) {return nodes_[i];}
};

typedef Heap<NNNode> NNHeap;
typedef Heap<RandNode> RandHeap;

IntMatrix make_rp_tree
(
    const Matrix &data,
    unsigned int leaf_size,
    RandomState &rng_state
);

void rand_tree_split
(
    const Matrix &data,
    IntVec &parent,
    IntVec &child0,
    IntVec &child1
);

IntMatrix get_index_matrix(std::vector<NNHeap> &graph);

void print(IntVec vec);
void print(std::vector<IntMatrix> &array);
void print(NNNode nd);
void print(RandNode nd);
void print(NNHeap heap);
void print(RandHeap heap);
void print(std::vector<NNHeap> graph);
void print(std::vector<RandHeap> graph);
void print(Matrix matrix);
void print_map(Matrix matrix);
void print(IntMatrix &matrix);

namespace RandNum
{
    extern std::mt19937 mersenne;
}

std::vector<IntMatrix> make_forest(
    const Matrix &data,
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
        int swab = current;

        if (nodes_[swab].key < nodes_[left_child].key)
        {
            swab = left_child;
        }

        if (
            (right_child < (int) nodes_.size()) &&
            (nodes_[swab].key < nodes_[right_child].key)
        )
        {
            swab = right_child;
        }

        if (swab == current)
        {
            return;
        }

        NodeType tmp = nodes_[current];
        nodes_[current] = nodes_[swab];
        nodes_[swab] = tmp;
        current = swab;
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

// Auxiliary function for recursively printing a binary NNHeap.
template <class HeapType>
void _print_from(std::string prefix, HeapType heap, size_t from, bool is_left)
{
    if (from >= heap.size())
    {
        return;
    }

    std::cout << prefix;
    std::cout << (is_left && (from + 1 < heap.size()) ? "├──" : "└──");

    // Print current node
    print(heap[from]);
    std::string prefix_children = prefix + (is_left ? "│   " : "    ");

    // Add children of current node.
    _print_from(prefix_children, heap, from*2 + 1, true);
    _print_from(prefix_children, heap, from*2 + 2, false);
}

template <class HeapType>
void print(HeapType heap, int number=0)
{
    std::cout << number << " [size=" << heap.size() << "]\n";
    _print_from("", heap, 0, false);
}

template <class HeapType>
void print(std::vector<HeapType> graph)
{
    std::cout << "*****************Graph*****************\n";
    for (size_t i = 0; i < graph.size(); ++i)
    {
        print<HeapType>(graph[i], i);
    }
    std::cout << "***************************************\n";
}
