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
