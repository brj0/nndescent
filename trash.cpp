
float _dist
(
    const Matrix<float> &data,
    size_t row0,
    size_t row1
);
float _dist
(
    const Matrix<float> &data,
    size_t row0,
    size_t row1
)
{
    float sum = 0.0f;
    size_t dim = data.ncols();
    for (size_t i = 0; i < dim; ++i)
    {
        sum += (data(row0, i) - data(row1, i))
             * (data(row0, i) - data(row1, i));
    }
    return sum;
}
inline float dist
(
    const Matrix<float> &data,
    size_t row0,
    size_t row1
)
{
    float sum = 0.0f;
    float delta;
    auto it0 = data.begin(row0);
    auto it1 = data.begin(row1);
    while (it0 != data.end(row0))
    {
        delta = (*it0) * (*it1);
        sum = std::move(sum) + delta*delta;
        ++it0;
        ++it1;
    }
    return sum;
}

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


Matrix<int> nn_brute_force(Matrix<float> &data, int n_neighbors);

void print(std::vector<NNHeap> &graph)
{
    for (size_t i = 0; i < graph.size(); i++)
    {
        std::cout << i << ": ";
        for (size_t j = 0; j < graph[i].size(); ++j)
        {
            std::cout << " " << graph[i][j].idx;
        }
        std::cout << "\n";
    }
}

void print(Matrix<float> matrix)
{
    std::cout << "[";

    for(size_t i = 0; i < matrix.nrows(); ++i)
    {
        if (i > 0)
        {
            std::cout << " ";
        }
        std::cout << "[";
        for(size_t j = 0; j < matrix.ncols(); j++)
        {
            std::cout << matrix(i, j);
            if (j + 1 != matrix.ncols())
            {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (i + 1 != matrix.nrows())
        {
            std::cout << ",\n";
        }
    }

    std::cout << "]\n";
}

void print(NNNode nd)
{
    std::cout << "(idx=" << nd.idx
              << " dist=" << nd.key
              << " visited=" << nd.visited
              << ")\n";
}

void print(RandNode nd)
{
    std::cout << "(idx=" << nd.idx
              << " priority=" << nd.key
              << ")\n";
}

void print(NNHeap heap)
{
    print<NNHeap>(heap);
}

void print(RandHeap heap)
{
    print<RandHeap>(heap);
}

void print(std::vector<NNHeap> graph)
{
    print<NNHeap>(graph);
}

void print(std::vector<RandHeap> graph)
{
    print<RandHeap>(graph);
}

typedef std::vector<std::vector<float>> SlowMatrix;

// Elements of the nearest neighbors heap
typedef struct
{
    int idx;
    float key;
    bool visited;
} NNNode;

// Elements of the randomly selected candidates for loca joins
typedef struct
{
    int idx;
    uint64_t key;
} RandNode;

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

{
    float sum = 0.0f;
    size_t size = matrix.ncols();
    for (size_t j = 0; j < size; ++j)
    {
        sum += matrix(row0, j) * matrix(row1, j);
    }
    return sum;
}

float cdist
(
    Matrix<float> &matrix,
    size_t row0,
    size_t row1
)
{
    float sum = 0.0f;
    size_t size = matrix.ncols();
    for (size_t i = 0; i < size; ++i)
    {
        sum += (matrix(row0,i) - matrix(row1, i))
            * (matrix(row0,i) - matrix(row1, i));
    }
    return sum;
}

float dot_product
(
    Matrix<float> &matrix,
    size_t row0,
    size_t row1
)
{
    float sum = 0.0f;
    size_t size = matrix.ncols();
    for (size_t j = 0; j < size; ++j)
    {
        sum += matrix(row0, j) * matrix(row1, j);
    }
    return sum;
}
// Improves nearest neighbors heaps by a single random projection tree.
void update_nn_graph_by_rp_tree(
    const Matrix<float> &data, std::vector<NNHeap> &current_graph, unsigned int leaf_size
)
{
    // timer.start();
    RandomState rng_state;
    IntMatrix rp_tree = make_rp_tree(data, leaf_size, rng_state);
    // timer.stop("make_rp_tree");
    // std::cout << "\nRPTree="; print(rp_tree);
    for (IntVec leaf : rp_tree)
    {
        for (int p0 : leaf)
        {
            for (int p1 : leaf)
            {
                if (p0 >= p1)
                {
                    continue;
                }
                float l = dist(data, p0, p1);
                NNNode node0 = {.idx=p0, .key=l, .visited=false};
                NNNode node1 = {.idx=p1, .key=l, .visited=false};
                current_graph[p0].update_max(node1);
                current_graph[p1].update_max(node0);
            }
        }
    }
    // timer.stop("heap updates");
}

// Converts heap list to Matrix.
IntMatrix get_index_matrix(std::vector<NNHeap> &graph)
{
    size_t nrows = graph.size();
    if (nrows == 0)
    {
        return IntMatrix(0, IntVec(0));
    }
    size_t ncols = graph[0].size();
    IntMatrix matrix (nrows, IntVec(ncols));
    for (size_t i = 0; i < nrows; ++i)
    {
        for (size_t j = 0; j < graph[i].size(); ++j)
        {
            matrix[i][j] = graph[i][j].idx;
        }
        // TODO del after changes to algorithm
        for (size_t j = graph[i].size(); j < ncols; ++j)
        {
            matrix[i][j] = -1;
        }
    }
    return matrix;
}

IntMatrix nn_brute_force(Matrix<float> &data, int n_neighbors)
{
    std::vector<NNHeap> current_graph(data.nrows());
    for (size_t i = 0; i < data.nrows(); ++i)
    {
        for (size_t j = 0; j < data.nrows(); ++j)
        {
            float l = dist(data, i, j);
            NNNode u = {.idx=(int) j, .key=l, .visited=true};
            current_graph[i].update_max(u, n_neighbors);
        }
    }
    return get_index_matrix(current_graph);
}

std::vector<float> midpoint(
    std::vector<float> &vec0,
    std::vector<float> &vec1
)
{
    std::vector<float> midpnt (vec0.size());
    for (size_t i = 0; i < vec0.size(); ++i)
    {
        midpnt[i] = (vec0[i] + vec1[i]) / 2;
    }
    return midpnt;
}

// Initializes heaps by choosing nodes randomly.
std::vector<NNHeap> empty_graph
(
    unsigned int n_heaps,
    unsigned int n_nodes
)
{
    NNNode nullnode = {
        .idx=-1,
        .key=MAX_FLOAT,
        .visited=false
    };
    NNHeap nullheap(n_nodes, nullnode);
    std::vector<NNHeap> heaplist(n_heaps, nullheap);
    return heaplist;
}

if (left_child >= n_nodes)
{
    break;
}
else if (right_child >= n_nodes)
{
    if (keys(i, left_child) > key)
    {
        swab = left_child;
    }
    else
    {
        break;
    }
}
else if (keys(i, left_child) >= keys(i, right_child))

// Improves nearest neighbors heaps by multiple random projection trees.
void _update_nn_graph_by_rp_forest(
    const Matrix<float> &data,
    std::vector<NNHeap> &current_graph,
    unsigned int forest_size,
    unsigned int leaf_size
)
{
    for (size_t i = 0; i < forest_size; ++i)
    {
        update_nn_graph_by_rp_tree(data, current_graph, leaf_size);
    }
}
// Calculates reverse nearest neighbours.
void reverse_neighbours(std::vector<IntVec> &dst, std::vector<IntVec> &src)
{
    for (size_t u = 0; u < src.size(); u++)
    {
        for (int v : src[u])
        {
            // std::cout << "u=" << u << " v=" << v << " dst[v]=";
            // print(dst[v]);
            dst[v].push_back(u);
        }
    }
}


template <class ArrayType>
float euclid_sqr
(
    const ArrayType &v0,
    const ArrayType &v1
)
{
    float sum = 0.0f;
    for (size_t i = 0; i < v0.size(); ++i)
    {
        sum += (v0[i] - v1[i])*(v0[i] - v1[i]);
    }
    return sum;
}

float dist
(
    const std::vector<float> &v0,
    const std::vector<float> &v1
);

float dist
(
    const std::vector<float> &v0,
    const std::vector<float> &v1
)
{
    return euclid_sqr<std::vector<float>>(v0, v1);
}

namespace RandNum
{
    extern std::mt19937 mersenne;
}

namespace RandNum
{
    // int seed = 1234;
    // int seed = time(NULL);
    uint64_t seed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
    std::mt19937 mersenne(seed);
}

#include <experimental/algorithm>
void sample_neighbors(
    IntVec &dst,
    NNHeap &src,
    IntVec &indices,
    int sample_size
)
{
    IntVec samples;

    // Sample without replacement
    // std::sample(
    std::experimental::sample(
        indices.begin(),
        indices.end(),
        std::back_inserter(samples),
        sample_size,
        RandNum::mersenne
    );

    // std::cout << "popsize=" << indices.size()
              // << " smpsize=" << sample_size
              // << "\nsamples=";
    // print(samples);

    for (size_t i = 0; i < samples.size(); ++i)
    {
        // print(indices);
        // print(samples);
        // std::cout << "sample_size=" << sample_size << ",";
        // std::cout << "XXXXXXXXXX\n\n";

        int index = samples[i];
        NNNode node = src[index];
        dst.push_back(node.idx);
        node.visited = true;
    }
    // print(src);
    // print(dst);
}

void set_new_and_old(
    IntVec &idx_new_nodes,
    IntVec &idx_old_nodes,
    NNHeap &all_nodes,
    int k_part
)
{
    IntVec indices_new;
    for (size_t i = 0; i < all_nodes.size(); i++)
    {
        NNNode node = all_nodes[i];
        if (!node.visited)
        {
            indices_new.push_back(i);
        }
        else
        {
            idx_old_nodes.push_back(node.idx);
        }
    }

    // print(idx_new_nodes);
    // print(idx_old_nodes);
    // print(all_nodes);

    sample_neighbors(
        idx_new_nodes,
        all_nodes,
        indices_new,
        k_part
    );
}


int local_join(
    const Matrix<float> &data,
    IntVec &idx_new_nodes,
    IntVec &idx_old_nodes,
    IntVec &idx_new_nodes_r,
    IntVec &idx_old_nodes_r,
    std::vector<NNHeap> &current_graph,
    int k_part
)
{
    IntVec buf_new;
    IntVec buf_old;

    // // Sample without replacement
    // std::sample(
    std::experimental::sample(
        idx_new_nodes_r.begin(),
        idx_new_nodes_r.end(),
        std::back_inserter(buf_new),
        k_part,
        RandNum::mersenne
    );
    // std::sample(
    std::experimental::sample(
        idx_old_nodes_r.begin(),
        idx_old_nodes_r.end(),
        std::back_inserter(buf_old),
        k_part,
        RandNum::mersenne
    );

    // buf_new = idx_new_nodes_r;
    // buf_old = idx_old_nodes_r;
    // std::cout << "size new=" << buf_new.size() << "\told=" << buf_old.size() << "\n";

    // Extend vectors by buf
    idx_new_nodes.insert(idx_new_nodes.end(), buf_new.begin(), buf_new.end());
    idx_old_nodes.insert(idx_old_nodes.end(), buf_old.begin(), buf_old.end());

    int cnt = 0;

    for (size_t i = 0; i < idx_new_nodes.size(); ++i)
    {
        int idx0 = idx_new_nodes[i];
        for (size_t j = 0; j < idx_new_nodes.size(); ++j)
        {
            int idx1 = idx_new_nodes[j];

            if (idx0 >= idx1)
            {
                continue;
            }
            // TODO size_t statt int
            float l = dist(data, idx0, idx1);
            NNNode u0 = {.idx=idx0, .key=l, .visited=true};
            NNNode u1 = {.idx=idx1, .key=l, .visited=true};

            cnt += current_graph[u0.idx].update_max(u1);
            cnt += current_graph[u1.idx].update_max(u0);
        }
        for (size_t j = 0; j < idx_old_nodes.size(); ++j)
        {
            int idx1 = idx_old_nodes[j];

            float l = dist(data, idx0, idx1);
            NNNode u0 = {.idx=idx0, .key=l, .visited=true};
            NNNode u1 = {.idx=idx1, .key=l, .visited=true};

            cnt += current_graph[u0.idx].update_max(u1);
            cnt += current_graph[u1.idx].update_max(u0);
        }
    }
    return cnt;
}

void euclidean_random_projection_split
{
    if (std::abs(margin) < EPS)
    {
        side[i] = rand_int(rng_state) % 2;
        if (side[i] == 0)
        {
            ++cnt0;
        }
        else
        {
            ++cnt1;
        }
    }
    else if (margin > 0)
    {
        side[i] = 0;
        ++cnt0;
    }
    else
    {
        side[i] = 1;
        ++cnt1;
    }
}

SlowMatrix vec_read_csv(std::string file_path)
{
    std::cout << "Reading " << file_path << "\n";
    SlowMatrix data;

    std::fstream csv_file;
    csv_file.open(file_path, std::ios::in);

    std::string line;
    while (std::getline(csv_file, line))
    {
        std::vector<double> row;
        std::stringstream ss_line(line);
        while (ss_line.good())
        {
            std::string substr;
            getline(ss_line, substr, ',');
            row.push_back(atof(substr.c_str()));
        }
        data.push_back(row);
    }

    csv_file.close();

    return data;
}
SlowMatrix slowdata = vec_read_csv(file_path);



class RandomNumberGenerator
{
    private:
        uint64_t s[4];
    public:
        uint64_t next(void)
        {
            const uint64_t result = s[0] + s[3];
            const uint64_t t = s[1] << 17;

            s[2] ^= s[0];
            s[3] ^= s[1];
            s[1] ^= s[2];
            s[0] ^= s[3];

            s[2] ^= t;
            s[3] = ((s[3] << 45) | (s[3] >> 19));

            return result;
        }

        void seed()
        {
            seed_state(s);
        }
};

typedef uint32_t RandomState32[STATE_SIZE];

/* This is xoshiro128++ 1.0, one of our 32-bit all-purpose, rock-solid
   generators. It has excellent speed, a state size (128 bits) that is
   large enough for mild parallelism, and it passes all tests we are aware
   of.

   For generating just single-precision (i.e., 32-bit) floating-point
   numbers, xoshiro128+ is even faster.

   The state must be seeded so that it is not everywhere zero. */
// static uint32_t s[4];
// https://prng.di.unimi.it/xoshiro128plusplus.c
uint32_t xoshiro128pp(RandomState32 &s)
{
    const uint32_t result = (((s[0] + s[3]) << 7) | ((s[0] + s[3]) >> 25))
        + s[0];
    const uint32_t t = s[1] << 9;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = ((s[3] << 11) | (s[3] >> 21));
    return result;
}




/* This is xoshiro256+ 1.0, our best and fastest generator for floating-point
   numbers. We suggest to use its upper bits for floating-point
   generation, as it is slightly faster than xoshiro256++/xoshiro256**. It
   passes all tests we are aware of except for the lowest three bits,
   which might fail linearity tests (and just those), so if low linear
   complexity is not considered an issue (as it is usually the case) it
   can be used to generate 64-bit outputs, too.

   We suggest to use a sign test to extract a random Boolean value, and
   right shifts to extract subsets of bits.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s. */
static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}
static uint64_t s[4]={1234,2345,3456,4567};
static uint64_t s[4];
uint64_t next(void) {
    const uint64_t result = s[0] + s[3];
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
}


xoshiro256+ 1.0 fast random number generator. Code modified from
https://prng.di.unimi.it/xoshiro256plus.c
uint64_t rand_int(void)
{
    const uint64_t result = s[0] + s[3];
    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = ((s[3] << 45) | (s[3] >> 19));

    return result;
}


/* This is xoshiro128** 1.1, one of our 32-bit all-purpose, rock-solid
   generators. It has excellent speed, a state size (128 bits) that is
   large enough for mild parallelism, and it passes all tests we are aware
   of.

   Note that version 1.0 had mistakenly s[0] instead of s[1] as state
   word passed to the scrambler.

   For generating just single-precision (i.e., 32-bit) floating-point
   numbers, xoshiro128+ is even faster.

   The state must be seeded so that it is not everywhere zero. */
uint32_t xoshiro129ss(RandomState32 &s)
{
    const uint32_t result = (((s[1] * 5) << 7) | ((s[1] * 5) >> 25)) * 9;
    const uint32_t t = s[1] << 9;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = ((s[3] << 11) | (s[3] >> 21));
    return result;
}

int main()
{

    std::vector<float> v0 = {2,2,-1,2,7,0,0,1,-5,10};
    std::vector<float> v1 = {2,0,3,4,-5,-6,1,0,1,-2};
    std::vector<float> v2 = {5,0,1,2,-9,0,0,0,1,-1,5};
    std::vector<float> v3 = {2,2, 10, 5, 6, 7};
    std::vector<float> v4 = {1,0};
    std::vector<float> w0 = {0,0,1.0/3,2.0/3};
    std::vector<float> w1 = {1.0/8,0,3.0/8,0.5};
    std::vector<float> w(10);
    std::vector<float> mtx_val ({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15});
    Matrix<float> mtx(4, mtx_val);
    std::vector<float> x = {8,2,19,1};
    std::vector<float> y = {-2,0,3,8};

    float d = spearmanr(v0.begin(), v0.end(), v0.begin());
    correct_alternative_hellinger(v0.begin(), v0.end(), w.begin());

    std::cout << "d=" << d << "\n";
    std::cout << "w=" << w << "\n";
    float m = median(w0);
    std::cout << "med0=" << m << "\n";

    std::cout << "mtx=" << mtx << "\n";
    float e = mahalanobis(x.begin(), x.end(), y.begin(), mtx.begin(0));
    std::cout << "e=" << e << "\n";

    auto v = rankdata(v3.begin(), v3.end(), "min");
    std::cout << "v3=" << v <<"\n";




    IntVec v0 = {2,2,2,2,5,6,7,8};
    IntVec v1 = {2,2,3,4,5,6,7,8};
    IntVec v2 = {0,1,4,4,5,6,7,8};
    IntVec v3 = {0,2,3,5,7,7,8,8};


    std::cout << "vectors=" << v0 << "\n";
    IntMatrix m = {v0,v1,v2,v3};
    std::cout << "mat=" << m << "\n";

    std::vector<float> v4 = {
        0,1,-2,3,
        1,2,2,4,
        0,9,2,0
    };

    Matrix<float> mtx(3,v4);
    std::cout << "m=" << mtx;
    std::cout << "distance="
        << jaccard(mtx.begin(0), mtx.end(0), mtx.begin(2))
        << "\n";

    int N = 60000*800;

    std::vector<float> val (N);
    std::iota(val.begin(), val.end(), 0.0);

    Matrix<float> data(60000,val);


    double d0;

    Metric fun;
    // typedef float (*MetricPtr)(It, It, It);
    using MetricPtr = float (*)(It, It, It);
    // float (*dist_ptr)(It, It, It);
    MetricPtr dist_ptr;
    dist_ptr = squared_euclidean<It,It>;

    fun = squared_euclidean<It,It>;

    ttest.start();
    for (int k = 0; k < 10000; ++k)
    {
        for (int i = 0; i < 800; ++i)
        {
            // d0 = _dist(data, 0, i);
            // d0 = squared_euclidean(data.begin(0), data.end(0), data.begin(i));
            // d0 = dist_ptr(data.begin(0), data.end(0), data.begin(i));
            // d0 = (*dist_ptr)(data.begin(0), data.end(0), data.begin(i));
            // d0 = fun(data.begin(0), data.end(0), data.begin(i));
            d0 = my_dist(data.begin(0), data.end(0), data.begin(i));
        }
    }
    ttest.stop("dot product");
    std::cout << "dotproduct=" << d0 << "\n";


    ttest.start();
    for (int k = 0; k < 10000; ++k)
    {
        for (int i = 0; i < 800; ++i)
        {
            d0 = _dist(data, 0, i);
            d0 = squared_euclidean(data.begin(0), data.end(0), data.begin(i));

            d0 = dist_ptr(data.begin(0), data.end(0), data.begin(i));
            d0 = (*dist_ptr)(data.begin(0), data.end(0), data.begin(i));
            d0 = fun(data.begin(0), data.end(0), data.begin(i));
        }
    }
    ttest.stop("dot product");
    std::cout << "dotproduct=" << d0 << "\n";
    HeapList<float> hl = HeapList<float>(4,3,99,FALSE);
    std::cout << hl;

    hl.checked_push(2, 0, 66, FALSE);
    std::cout << hl;

    hl.checked_push(2, 1, 16, TRUE);
    std::cout << hl;

    hl.checked_push(2, 2, 6, FALSE);
    std::cout << hl;

    hl.checked_push(2, 3, 766, FALSE);
    std::cout << hl;

    hl.checked_push(3, 4, 76, TRUE);
    std::cout << hl;

    hl.checked_push(2, 5, 1, TRUE);
    std::cout << hl;

    hl.checked_push(2, 5, 0, TRUE);
    std::cout << hl;

    hl.checked_push(2, 6, 0, TRUE);
    std::cout << hl;

    hl.checked_push(0, 6, 0, TRUE);
    std::cout << hl;

    hl.checked_push(0, 16, 10, TRUE);
    std::cout << hl;

    hl.heapsort();
    std::cout << hl;



    int N = 800;
    std::vector<double> v0 (N);
    std::vector<double> v1 (N);
    // all_points = [0,1,2,...]
    std::iota(v0.begin(), v0.end(), 0.0);
    std::iota(v1.begin(), v1.end(), 1.0);

    std::cout << "v0=" << v0[0] << " " << v0[1] << " ... " << v0[v0.size()-1] << "\n";
    std::cout << "v1=" << v1[0] << " " << v1[1] << " ... " << v1[v1.size()-1] << "\n";

    double d0;

    ttest.start();
    for (int i = 0; i < 1e5; ++i){
    d0 = std::inner_product(v0.begin(), v0.end(), v1.begin(), 0.0);
    }
    ttest.stop("dot product");
    std::cout << "dotproduct=" << d0 << "\n";


    ttest.start();
    double *a0 = &v0[0];
    double *a1 = &v1[0];
    for (int i = 0; i < 1e5; ++i){
    d0=0;
    for (int i = 0; i < N; ++i)
        d0+=a0[i]*a1[i];
    }
    ttest.stop("dot product 2");
    std::cout << "dotproduct=" << d0 << "\n";

    ttest.start();
    for (int i = 0; i < 1e5; ++i){
    d0=dot_product(v0,v1);
    }
    ttest.stop("dot product 2");
    std::cout << "dotproduct=" << d0 << "\n";






    std::cout << "d(0,1)=" << dist(slowdata[0], slowdata[1]) << "\n";
    std::cout << "d(0,2)=" << dist(slowdata[0], slowdata[2]) << "\n";
    std::cout << "d(0,5)=" << dist(slowdata[0], slowdata[5]) << "\n";
    std::cout << "d(2,4)=" << dist(slowdata[2], slowdata[4]) << "\n";
    std::cout << "d(3,7)=" << dist(slowdata[3], slowdata[7]) << "\n";

    ttest.start();
    for (size_t i = 0; i < data.nrows(); ++i)
    {
        dist(slowdata[0], slowdata[i]);
    }
    ttest.stop("slowdata");

    std::cout << "cd(0,1)=" << cdist(data, 0, 1) << "\n";
    std::cout << "cd(0,2)=" << cdist(data, 0, 2) << "\n";
    std::cout << "cd(0,5)=" << cdist(data, 0, 5) << "\n";
    std::cout << "cd(2,4)=" << cdist(data, 2, 4) << "\n";
    std::cout << "cd(3,7)=" << cdist(data, 3, 7) << "\n";



    ttest.start();
    for (size_t i = 0; i < data.nrows(); ++i)
    {
        cdist(data, 0, i);
    }
    ttest.stop("cdust");

    std::cout << "dot(0,19)=" << dot_product(slowdata[0], slowdata[19]) << "\n";
    std::cout << "arrdot(0,19)=" << dot_product(data, 0, 19) << "\n";

    ttest.start();
    for (size_t i = 0; i < data.nrows(); ++i)
    {
        dot_product(slowdata[0], slowdata[i]);
    }
    ttest.stop("dot: slowdata");

    for (int j = 0; j<5; ++j){
    ttest.start();
    for (size_t i = 0; i < data.nrows(); ++i)
    {
        dot_product(data, 0, i);
    }
    ttest.stop("dot: carray");
    }


    std::cout << "ad(0,1)=" << arrdist(data[0], data[1]) << "\n";
    std::cout << "ad(0,2)=" << arrdist(data[0], data[2]) << "\n";
    std::cout << "ad(0,5)=" << arrdist(data[0], data[5]) << "\n";
    std::cout << "ad(2,4)=" << arrdist(data[2], data[4]) << "\n";
    std::cout << "ad(3,7)=" << arrdist(data[3], data[7]) << "\n";

    ttest.start();
    for (size_t i = 0; i < data.nrows(); ++i)
    {
        arrdist(data[0], data[i]);
    }
    ttest.stop("arrdist");




    std::vector<double> v0 (1e7);
    std::vector<double> v1 (1e7);
    // all_points = [0,1,2,...]
    std::iota(v0.begin(), v0.end(), 0.0);
    std::iota(v1.begin(), v1.end(), 1.0);

    std::cout << "v0=" << v0[0] << " " << v0[1] << " ... " << v0[v0.size()-1] << "\n";
    std::cout << "v1=" << v1[0] << " " << v1[1] << " ... " << v1[v1.size()-1] << "\n";

    double d0;
    double d1;

    ttest.start();
    d0 = std::inner_product(v0.begin(), v0.end(), v1.begin(), 0.0);
    ttest.stop("dot product");
    std::cout << "dotproduct=" << d0 << "\n";


    ttest.start();
    double *a0 = &v0[0];
    double *a1 = &v1[0];
    d1=0;
    for (int i = 0; i < 1e7; ++i)
        d1+=a0[i]*a1[i];
    ttest.stop("dot product 2");
    std::cout << "dotproduct=" << d1 << "\n";

    ttest.start();
    d1=dot_product(v0,v1);
    ttest.stop("dot product 2");
    std::cout << "dotproduct=" << d1 << "\n";


    test_csv();

    Timer ttyp;
    int z;

    RandomState state = {20,1,2,3};

    ttyp.start();
    for (int i = 0; i < 1e8; ++i)
    {
        // z = rand_int(RandNumGen::rng_state);
        z = rand_int(state);
    }
        std::cout << "xorshift256=" <<  z << "\n";
    ttyp.stop("RandNumGen");

    std::cout << "xorshift256=" <<  z << "\n";


    RandomState32 state32 = {0,1,2,3};
    int z32;

    ttyp.start();
    for (int i = 0; i < 1e8; ++i)
    {
        z32 = xoshiro128pp(state32);
    }
        std::cout << "xoshiro128pp=" <<  z32 << "\n";
    ttyp.stop("xoshiro128pp");

    state32[0]=20;
    state32[1]=1;
    state32[2]=2;
    state32[3]=3;

    ttyp.start();
    for (int i = 0; i < 1e8; ++i)
    {
        z32 = xoshiro129ss(state32);
    }
        std::cout << "xoshiro129ss=" <<  z32 << "\n";
    ttyp.stop("xoshiro129ss");

    int myint = xoshiro128pp(state32) % 1234;
    int myint2 = rand_int(state) % 1234;


    std::pair<double, double> ab = find_ab_params(1, 0.05);
    std::cout << "a=" << ab.first << " b=" << ab.second << "\n";
    std::pair<double, double> abc = find_ab(1, 3);

    IntVec ivec;
    ivec.push_back(12);
    ivec.push_back(0);
    ivec.push_back(88);

    print(ivec);

    Node n0 = {0, 0.7, false};
    Node n1 = {1, 1.1, true};
    Node n2 = {2, 0.2, false};
    Node n3 = {3, 0.3, false};
    Node n4 = {4, 0.4, true};
    Node n5 = {5, 0.5, false};
    Node n6 = {6, 0.6, true};

    Heap hp;
    hp.push(n0);
    hp.push(n1);
    hp.push(n2);
    hp.push(n3);
    hp.push(n4);
    hp.push(n5);
    hp.push(n6);
    print(hp);

    uint64_t y;
    ttyp.start();


    s[0]=1234; s[1]=2345; s[2]=3456; s[3]=4567;

    ttyp.start();
    for (int i = 0; i < 1e8; ++i)
    {
        y = rand_int();
    }
    ttyp.stop("xorshift256");

    RandomNumberGenerator rng;
    rng.seed();

    ttyp.start();
    for (int i = 0; i < 1e8; ++i)
    {
        z = rng.next();
    }
    ttyp.stop("rng class");
    std::cout << "xorshift256=" << y << " " << z << "\n";

    ttyp.start();
    for (int i = 0; i < 1e8; ++i)
    {
        z = rand_int(s3);
    }
    ttyp.stop("randint3");
    std::cout << "xorshift256=" << y << " " << z << "\n";


    NNHeap heap (5);
    std::cout << "size=" << heap.size() << "\n";

    NNNode node0 = {.idx=11, .key=3.2, .visited=false};
    NNNode node1 = {.idx=13, .key=13.2, .visited=false};
    NNNode node2 = {.idx=15, .key=2.0, .visited=false};
    NNNode node3 = {.idx=1, .key=9.2, .visited=false};
    NNNode node4 = {.idx=5, .key=1.2, .visited=false};
    heap.insert(node0);
    std::cout << "heap0=" << heap[0].idx << "\n";
    heap.insert(node1);
    std::cout << "heap0=" << heap[0].idx << "\n";
    heap.insert(node2);
    std::cout << "heap0=" << heap[0].idx << "\n";
    heap.insert(node3);
    std::cout << "heap0=" << heap[0].idx << "\n";
    heap.insert(node4);
    std::cout << "heap0=" << heap[0].idx << "\n";
    heap.siftdown(0);
    std::cout << "heap0=" << heap[0].idx << "\n";

    print(heap);


   return 0;
}

// TODO do parallel
void nn_descent(
    const Matrix &data,
    std::vector<NNHeap> &current_graph,
    int n_neighbors,
    RandomState rng_state,
    int max_candidates,
    int n_iters,
    float delta,
    // bool rp_tree_init,
    bool verbose
)
{
    timer.start();

    assert(current_graph.size() == data.size());
    log("NN descent for " + std::to_string(n_iters) + " iterations", verbose);

    double rho = 1.0;
    int k_part = n_neighbors * rho;

    if (verbose)
    {
        std::cout << "Start with parameters:"
            << " n_neighbors=" << n_neighbors
            << " rho=" << rho
            << " k_part=" << k_part
            << " delta=" << delta
            << "\n";
    }

    int cnt = 0;
    timer.stop("init0");

    for (int iter = 0; iter < n_iters; ++iter)
    {
        log(
            (
                "\t" + std::to_string(iter + 1) + "  /  "
                + std::to_string(n_iters)
            ),
            verbose
        );

        std::vector<RandHeap> new_candidates;
        std::vector<RandHeap> old_candidates;
        int n_threads = 1;
        sample_candidates(
            current_graph,
            new_candidates,
            old_candidates,
            max_candidates,
            rng_state,
            n_threads
        );
        timer.stop("sample_candidates");
        update_nn_graph_by_joining_candidates(
            data,
            current_graph,
            new_candidates,
            old_candidates,
            n_threads
        );
        timer.stop("update by joining candidates");
        return;

        std::vector<IntVec> visited(data.size());
        std::vector<IntVec> visited_rev(data.size());
        std::vector<IntVec> unvisited(data.size());
        std::vector<IntVec> unvisited_rev(data.size());

        // Define old and new.
        for (size_t v = 0; v < data.size(); v++)
        {
            set_new_and_old(unvisited[v], visited[v], current_graph[v], k_part);
        }

        // log("\t\tOld and new set.", verbose);

        // Reverse.
        reverse_neighbours(unvisited_rev, unvisited);
        reverse_neighbours(visited_rev, visited);

        // log("\t\tReverse done.", verbose);

        // Local joins.
        cnt = 0;
        for (size_t v = 0; v < data.size(); v++)
        {
            log(
                "\t\tLocal join: " + std::to_string(v) + "/"
                    + std::to_string(data.size()),
                (verbose && (v % (data.size() / 4) == 0))
            );
            cnt += local_join(
                data,
                unvisited[v],
                visited[v],
                unvisited_rev[v],
                visited_rev[v],
                current_graph,
                k_part
            );
        }

        if (cnt < delta * data.size() * n_neighbors)
        {
            log(
                "Stopping threshold met -- exiting after "
                    + std::to_string(iter) + " iterations",
                verbose
            );
            break;
        }
    }

    log("NN descent done.", verbose);
}
