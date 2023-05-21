#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "../src/dtypes.h"
#include "../src/nnd.h"

namespace py = pybind11;

const Parms DEFAULT_PARMS;

class NNDWrapper
{
private:
public:
    Matrix<float> data;
    NNDescent nnd;

    NNDWrapper
    (
        py::array_t<float> &py_data,
        std::string metric,
        int n_neighbors,
        int n_trees,
        int leaf_size,
        float pruning_degree_multiplier,
        float diversify_prob,
        bool tree_init,
        int seed,
        bool low_memory,
        int max_candidates,
        int n_iters,
        float delta,
        int n_threads,
        bool compressed,
        bool parallel_batch_queries,
        bool verbose,
        std::string algorithm
    )
        : data(
            py_data.shape()[0],
            py_data.shape()[1],
            static_cast<float*>(py_data.request().ptr))
        , nnd(data, n_neighbors)
    {
        if (py_data.ndim() != 2)
        {
            throw std::runtime_error("Input should be 2-D NumPy array");
        }

        Parms parms;
        parms.metric = metric;
        parms.n_neighbors = n_neighbors;
        parms.n_trees = n_trees;
        parms.leaf_size = leaf_size;
        parms.pruning_degree_multiplier = pruning_degree_multiplier;
        parms.diversify_prob = diversify_prob;
        parms.tree_init = tree_init;
        parms.seed = seed;
        parms.low_memory = low_memory;
        parms.max_candidates = max_candidates;
        parms.n_iters = n_iters;
        parms.delta = delta;
        parms.n_threads = n_threads;
        parms.compressed = compressed;
        parms.parallel_batch_queries = parallel_batch_queries;
        parms.verbose = verbose;
        parms.algorithm = algorithm;

        nnd.set_parameters(parms);
        nnd.start();
    }

    std::string get_metric() const { return nnd.metric; }
    int get_n_neighbors() const { return nnd.n_neighbors; }
    int get_n_trees() const { return nnd.n_trees; }
    int get_leaf_size() const { return nnd.leaf_size; }
    float get_pruning_degree_multiplier() const
        { return nnd.pruning_degree_multiplier; }
    float get_diversify_prob() const { return nnd.diversify_prob; }
    bool get_tree_init() const { return nnd.tree_init; }
    int get_seed() const { return nnd.seed; }
    bool get_low_memory() const { return nnd.low_memory; }
    int get_max_candidates() const { return nnd.max_candidates; }
    int get_n_iters() const { return nnd.n_iters; }
    float get_delta() const { return nnd.delta; }
    int get_n_threads() const { return nnd.n_threads; }
    bool get_compressed() const { return nnd.compressed; }
    bool get_parallel_batch_queries() const
        { return nnd.parallel_batch_queries; }
    bool get_verbose() const { return nnd.verbose; }
    std::string get_algorithm() const { return nnd.algorithm; }
    py::array_t<float> get_data() const
    {
        size_t dim = 2;
        size_t nrows = data.nrows();
        size_t ncols = data.ncols();
        std::vector<size_t> strides = {sizeof(float)*ncols, sizeof(float)};
        std::vector<size_t> shape = {nrows , ncols};

        return py::array_t<float>(py::buffer_info(
            (float*) data.m_ptr,                     // data as contiguous array
            sizeof(float),                           // size of one scalar
            py::format_descriptor<float>::format(),  // data type
            dim,                                     // number of dimensions
            shape,                                   // shape of the matrix
            strides                                  // strides for each axis
        ));
    }

    py::tuple get_neighbor_graph() const
    {
        return py::make_tuple(get_indices(), get_distances());
    }

    py::array_t<int> get_indices() const
    {
        size_t dim = 2;
        size_t nrows = nnd.current_graph.indices.nrows();
        size_t ncols = nnd.current_graph.indices.ncols();
        std::vector<size_t> strides = {sizeof(int)*ncols, sizeof(int)};
        std::vector<size_t> shape = {nrows , ncols};

        return py::array_t<int>(py::buffer_info(
            (int*) nnd.current_graph.indices.m_ptr,  // data as contiguous array
            sizeof(int),                             // size of one scalar
            py::format_descriptor<int>::format(),    // data type
            dim,                                     // number of dimensions
            shape,                                   // shape of the matrix
            strides                                  // strides for each axis
        ));
    }

    py::array_t<float> get_distances() const
    {
        size_t dim = 2;
        size_t nrows = nnd.neighbor_distances.nrows();
        size_t ncols = nnd.neighbor_distances.ncols();
        std::vector<size_t> strides = {sizeof(float)*ncols, sizeof(float)};
        std::vector<size_t> shape = {nrows , ncols};

        return py::array_t<float>(py::buffer_info(
            (float*) nnd.neighbor_distances.m_ptr,   // data as contiguous array
            sizeof(float),                           // size of one scalar
            py::format_descriptor<float>::format(),  // data type
            dim,                                     // number of dimensions
            shape,                                   // shape of the matrix
            strides                                  // strides for each axis
        ));
    }

    py::array_t<char> get_flags() const
    {
        size_t dim = 2;
        size_t nrows = nnd.current_graph.flags.nrows();
        size_t ncols = nnd.current_graph.flags.ncols();
        std::vector<size_t> strides = {sizeof(char)*ncols, sizeof(char)};
        std::vector<size_t> shape = {nrows , ncols};

        return py::array_t<char>(py::buffer_info(
            (char*) nnd.current_graph.flags.m_ptr,   // data as contiguous array
            sizeof(char),                            // size of one scalar
            py::format_descriptor<char>::format(),   // data type
            dim,                                     // number of dimensions
            shape,                                   // shape of the matrix
            strides                                  // strides for each axis
        ));
    }


    void set_metric(const std::string& m) { nnd.metric = m; }
    void set_n_neighbors(int n) { nnd.n_neighbors = n; }
    void set_n_trees(int n) { nnd.n_trees = n; }
    void set_leaf_size(int n) { nnd.leaf_size = n; }
    void set_pruning_degree_multiplier(float x)
        { nnd.pruning_degree_multiplier = x; }
    void set_diversify_prob(float x) { nnd.diversify_prob = x; }
    void set_tree_init(bool x) { nnd.tree_init = x; }
    void set_seed(int x) { nnd.seed = x; }
    void set_low_memory(bool x) { nnd.low_memory = x; }
    void set_max_candidates(int x) { nnd.max_candidates = x; }
    void set_n_iters(int x) { nnd.n_iters = x; }
    void set_delta(float x) { nnd.delta = x; }
    void set_n_threads(int x) { nnd.n_threads = x; }
    void set_compressed(bool x) { nnd.compressed = x; }
    void set_parallel_batch_queries(bool x)
        { nnd.parallel_batch_queries = x; }
    void set_verbose(bool x) { nnd.verbose = x; }
    void set_algorithm(const std::string& alg) { nnd.algorithm = alg; }
};


PYBIND11_MODULE(nndescent, m)
{
    m.doc() = "Calculates approximate k-nearest neighbors";
    m.attr("__version__") = PROJECT_VERSION;
    py::class_<NNDWrapper>(m, "NNDescent")
        .def(py::init<py::array_t<float>&, const std::string&, int, int, int,
            float, float, bool, int, bool, int, int, float, int, bool, bool,
            bool, const std::string&>(),
            py::arg("data"),
            py::arg("metric")=DEFAULT_PARMS.metric,
            py::arg("n_neighbors")=DEFAULT_PARMS.n_neighbors,
            py::arg("n_trees")=DEFAULT_PARMS.n_trees,
            py::arg("leaf_size")=DEFAULT_PARMS.leaf_size,
            py::arg("pruning_degree_multiplier")
                =DEFAULT_PARMS.pruning_degree_multiplier,
            py::arg("diversify_prob")=DEFAULT_PARMS.diversify_prob,
            py::arg("tree_init")=DEFAULT_PARMS.tree_init,
            py::arg("seed")=DEFAULT_PARMS.seed,
            py::arg("low_memory")=DEFAULT_PARMS.low_memory,
            py::arg("max_candidates")=DEFAULT_PARMS.max_candidates,
            py::arg("n_iters")=DEFAULT_PARMS.n_iters,
            py::arg("delta")=DEFAULT_PARMS.delta,
            py::arg("n_threads")=DEFAULT_PARMS.n_threads,
            py::arg("compressed")=DEFAULT_PARMS.compressed,
            py::arg("parallel_batch_queries")
                =DEFAULT_PARMS.parallel_batch_queries,
            py::arg("verbose")=DEFAULT_PARMS.verbose,
            py::arg("algorithm")=DEFAULT_PARMS.algorithm
        )
        .def_property("metric", &NNDWrapper::get_metric,
            &NNDWrapper::set_metric)
        .def_property("n_neighbors", &NNDWrapper::get_n_neighbors,
            &NNDWrapper::set_n_neighbors)
        .def_property("n_trees", &NNDWrapper::get_n_trees,
            &NNDWrapper::set_n_trees)
        .def_property("leaf_size", &NNDWrapper::get_leaf_size,
            &NNDWrapper::set_leaf_size)
        .def_property("pruning_degree_multiplier",
            &NNDWrapper::get_pruning_degree_multiplier,
            &NNDWrapper::set_pruning_degree_multiplier)
        .def_property("diversify_prob", &NNDWrapper::get_diversify_prob,
            &NNDWrapper::set_diversify_prob)
        .def_property("tree_init", &NNDWrapper::get_tree_init,
            &NNDWrapper::set_tree_init)
        .def_property("seed", &NNDWrapper::get_seed, &NNDWrapper::set_seed)
        .def_property("low_memory", &NNDWrapper::get_low_memory,
            &NNDWrapper::set_low_memory)
        .def_property("max_candidates", &NNDWrapper::get_max_candidates,
            &NNDWrapper::set_max_candidates)
        .def_property("n_iters", &NNDWrapper::get_n_iters,
            &NNDWrapper::set_n_iters)
        .def_property("delta", &NNDWrapper::get_delta,
            &NNDWrapper::set_delta)
        .def_property("n_threads", &NNDWrapper::get_n_threads,
            &NNDWrapper::set_n_threads)
        .def_property("compressed", &NNDWrapper::get_compressed,
            &NNDWrapper::set_compressed)
        .def_property("parallel_batch_queries",
            &NNDWrapper::get_parallel_batch_queries,
            &NNDWrapper::set_parallel_batch_queries)
        .def_property("verbose", &NNDWrapper::get_verbose,
            &NNDWrapper::set_verbose)
        .def_property("algorithm", &NNDWrapper::get_algorithm,
            &NNDWrapper::set_algorithm)
        .def_property_readonly("data", &NNDWrapper::get_data)
        .def_property_readonly("indices", &NNDWrapper::get_indices)
        .def_property_readonly("distances", &NNDWrapper::get_distances)
        .def_property_readonly("neighbor_graph",
            &NNDWrapper::get_neighbor_graph)
        .def_property_readonly("flags", &NNDWrapper::get_flags)
        ;
    ;
}
