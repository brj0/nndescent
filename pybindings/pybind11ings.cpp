/*
 * @file pybind11ings.cpp
 *
 * @brief Contaings the python bindings.
 */


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "../src/dtypes.h"
#include "../src/nnd.h"

namespace py = pybind11;
using namespace nndescent;

const Parms DEFAULT_PARMS;


/**
 * @brief Convert a nndescent::Matrix<T> object to a NumPy array in Python.
 */
template<class T>
py::array_t<T> to_pyarray(const Matrix<T> &matrix)
{
    size_t dim = 2;
    size_t nrows = matrix.nrows();
    size_t ncols = matrix.ncols();
    std::vector<size_t> strides = {sizeof(T)*ncols, sizeof(T)};
    std::vector<size_t> shape = {nrows , ncols};

    return py::array_t<T>(py::buffer_info(
        (T*) matrix.m_ptr,                   // data as contiguous array
        sizeof(T),                           // size of one scalar
        py::format_descriptor<T>::format(),  // data type
        dim,                                 // number of dimensions
        shape,                               // shape of the matrix
        strides                              // strides for each axis
    ));
}


 /**
 * @brief Wrapper class for binding the NND (Nearest Neighbor Descent)
 * algorithm in Python.
 */
class NNDWrapper
{
private:
public:
    Matrix<float> data;
    NNDescent nnd;

    NNDWrapper
    (
        py::object &py_obj,
        std::string metric,
        int n_neighbors,
        int n_trees,
        int leaf_size,
        float pruning_degree_multiplier,
        float pruning_prob,
        bool tree_init,
        int seed,
        int max_candidates,
        int n_iters,
        float delta,
        int n_threads,
        bool verbose,
        std::string algorithm
    )
    {
        py::array_t<float> py_data(py_obj);
        if (!py::isinstance<py::array_t<float>>(py_obj))
        {
            throw std::runtime_error("Input should be 2-D float32 NumPy array");
        }
        if (py_data.ndim() != 2)
        {
            throw std::runtime_error("Input should be 2-D float32 NumPy array");
        }
        data = Matrix<float>(
            py_data.shape()[0],
            py_data.shape()[1],
            static_cast<float*>(py_data.request().ptr)
        );
        nnd = NNDescent(data, n_neighbors);

        Parms parms;
        parms.metric = metric;
        parms.n_neighbors = n_neighbors;
        parms.n_trees = n_trees;
        parms.leaf_size = leaf_size;
        parms.pruning_degree_multiplier = pruning_degree_multiplier;
        parms.pruning_prob = pruning_prob;
        parms.tree_init = tree_init;
        parms.seed = seed;
        parms.max_candidates = max_candidates;
        parms.n_iters = n_iters;
        parms.delta = delta;
        parms.n_threads = n_threads;
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
    float get_pruning_prob() const
        { return nnd.pruning_prob; }
    bool get_tree_init() const { return nnd.tree_init; }
    int get_seed() const { return nnd.seed; }
    int get_max_candidates() const { return nnd.max_candidates; }
    int get_n_iters() const { return nnd.n_iters; }
    float get_delta() const { return nnd.delta; }
    int get_n_threads() const { return nnd.n_threads; }
    bool get_verbose() const { return nnd.verbose; }
    std::string get_algorithm() const { return nnd.algorithm; }
    py::array_t<float> get_data() const
    {
        return to_pyarray(data);
    }
    py::tuple get_neighbor_graph() const
    {
        return py::make_tuple(get_indices(), get_distances());
    }
    py::array_t<int> get_indices() const
    {
        return to_pyarray(nnd.current_graph.indices);
    }
    py::array_t<float> get_distances() const
    {
        return to_pyarray(nnd.neighbor_distances);
    }
    py::array_t<char> get_flags() const
    {
        return to_pyarray(nnd.current_graph.flags);
    }

    py::tuple query
    (
        py::array_t<float> &py_query_data, int k, float epsilon
    )
    {
        Matrix<float> query_data(
            py_query_data.shape()[0],
            py_query_data.shape()[1],
            static_cast<float*>(py_query_data.request().ptr)
        );
        nnd.query(query_data, k, epsilon);
        Matrix<int> query_indices = nnd.query_indices;
        Matrix<float> query_distances = nnd.query_distances;
        return py::make_tuple(
            to_pyarray(query_indices), to_pyarray(query_distances)
        );
    }
    void set_metric(const std::string& m) { nnd.metric = m; }
    void set_n_neighbors(int n) { nnd.n_neighbors = n; }
    void set_n_trees(int n) { nnd.n_trees = n; }
    void set_leaf_size(int n) { nnd.leaf_size = n; }
    void set_pruning_degree_multiplier(float x)
        { nnd.pruning_degree_multiplier = x; }
    void set_pruning_prob(float x)
        { nnd.pruning_prob = x; }
    void set_tree_init(bool x) { nnd.tree_init = x; }
    void set_seed(int x) { nnd.seed = x; }
    void set_max_candidates(int x) { nnd.max_candidates = x; }
    void set_n_iters(int x) { nnd.n_iters = x; }
    void set_delta(float x) { nnd.delta = x; }
    void set_n_threads(int x) { nnd.n_threads = x; }
    void set_verbose(bool x) { nnd.verbose = x; }
    void set_algorithm(const std::string& alg) { nnd.algorithm = alg; }
};


PYBIND11_MODULE(nndescent, m)
{
    m.doc() = "Calculates approximate k-nearest neighbors";
    m.attr("__version__") = PROJECT_VERSION;
    py::class_<NNDWrapper>(m, "NNDescent")
        .def(py::init<py::object&, const std::string&, int, int, int,
            float, float, bool, int, int, int, float, int, bool,
            const std::string&>(),
            py::arg("data"),
            py::arg("metric")=DEFAULT_PARMS.metric,
            py::arg("n_neighbors")=DEFAULT_PARMS.n_neighbors,
            py::arg("n_trees")=DEFAULT_PARMS.n_trees,
            py::arg("leaf_size")=DEFAULT_PARMS.leaf_size,
            py::arg("pruning_degree_multiplier")
                =DEFAULT_PARMS.pruning_degree_multiplier,
            py::arg("pruning_prob")
                =DEFAULT_PARMS.pruning_prob,
            py::arg("tree_init")=DEFAULT_PARMS.tree_init,
            py::arg("seed")=DEFAULT_PARMS.seed,
            py::arg("max_candidates")=DEFAULT_PARMS.max_candidates,
            py::arg("n_iters")=DEFAULT_PARMS.n_iters,
            py::arg("delta")=DEFAULT_PARMS.delta,
            py::arg("n_threads")=DEFAULT_PARMS.n_threads,
            py::arg("verbose")=DEFAULT_PARMS.verbose,
            py::arg("algorithm")=DEFAULT_PARMS.algorithm
        )
        .def("query", &NNDWrapper::query,
            py::arg("query_data"),
            py::arg("k")=DEFAULT_K,
            py::arg("epsilon")=DEFAULT_EPSILON
        )
        .def_property("metric", &NNDWrapper::get_metric,
            &NNDWrapper::set_metric
        )
        .def_property("n_neighbors", &NNDWrapper::get_n_neighbors,
            &NNDWrapper::set_n_neighbors
        )
        .def_property("n_trees", &NNDWrapper::get_n_trees,
            &NNDWrapper::set_n_trees
        )
        .def_property("leaf_size", &NNDWrapper::get_leaf_size,
            &NNDWrapper::set_leaf_size
        )
        .def_property("pruning_degree_multiplier",
            &NNDWrapper::get_pruning_degree_multiplier,
            &NNDWrapper::set_pruning_degree_multiplier
        )
        .def_property("pruning_prob",
            &NNDWrapper::get_pruning_prob,
            &NNDWrapper::set_pruning_prob
        )
        .def_property("tree_init", &NNDWrapper::get_tree_init,
            &NNDWrapper::set_tree_init
        )
        .def_property("seed", &NNDWrapper::get_seed, &NNDWrapper::set_seed)
        .def_property("max_candidates", &NNDWrapper::get_max_candidates,
            &NNDWrapper::set_max_candidates
        )
        .def_property("n_iters", &NNDWrapper::get_n_iters,
            &NNDWrapper::set_n_iters
        )
        .def_property("delta", &NNDWrapper::get_delta,
            &NNDWrapper::set_delta
        )
        .def_property("n_threads", &NNDWrapper::get_n_threads,
            &NNDWrapper::set_n_threads
        )
        .def_property("verbose", &NNDWrapper::get_verbose,
            &NNDWrapper::set_verbose
        )
        .def_property("algorithm", &NNDWrapper::get_algorithm,
            &NNDWrapper::set_algorithm
        )
        .def_property_readonly("data", &NNDWrapper::get_data)
        .def_property_readonly("indices", &NNDWrapper::get_indices)
        .def_property_readonly("distances", &NNDWrapper::get_distances)
        .def_property_readonly("neighbor_graph",
            &NNDWrapper::get_neighbor_graph
        )
        .def_property_readonly("flags", &NNDWrapper::get_flags)
    ;
}
