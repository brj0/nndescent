#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <chrono>
#include <thread>
#include <iostream>

#include "dtypes.h"
#include "nnd.h"

namespace py = pybind11;

const std::string MODULE_VERSION = "0.0.0";
const Parms DEFAULT_PARMS;

Timer timer11;


Matrix<int> nnd_algorithm(Matrix<float> &points, int n_neighbors)
{
    Parms parms;
    parms.n_neighbors = n_neighbors;
    parms.verbose = true;
    NNDescent nnd = NNDescent(points, parms);
    return nnd.neighbor_graph;
}


Matrix<int> bfnn_algorithm(Matrix<float> &points, int n_neighbors)
{
    Parms parms;
    parms.n_neighbors = n_neighbors;
    parms.verbose = true;
    parms.algorithm = "bf";
    NNDescent nnd = NNDescent(points, parms);
    return nnd.neighbor_graph;
}


std::string version()
{
    return MODULE_VERSION;
}

// py::array_t<float> _test(
    // py::array_t<float, py::array::c_style | py::array::forcecast> py_data,
    // int n_neighbors
// )
py::array_t<float> _test(py::array_t<float> py_data, int n_neighbors)
{
    timer11.start();
    // Allocate the buffer
    py::buffer_info buf = py_data.request();
    // float *ptr = (float *) buf.ptr;

    if (buf.ndim != 2)
    {
        throw std::runtime_error("Input should be 2-D NumPy array");
    }

    // Translate input matrix from Python to C++
    size_t dim = 2;
    size_t nrows = buf.shape[0];
    size_t ncols = buf.shape[1];
    std::vector<size_t> strides = { sizeof(float)*ncols , sizeof(float) };
    timer11.stop("--0");
    std::vector<size_t> shape = {nrows , ncols};
    timer11.stop("--1");

    // Matrix<float> data (nrows, ncols);
    Matrix<float> data(nrows, ncols);
    timer11.stop("--2");

    std::memcpy(data.m_data.data(), py_data.data(), py_data.size()*sizeof(float));
    timer11.stop("--3");
    std::cout << "dim="<<dim<<" row="<<nrows<<" col="<<ncols<<"\n";
    // for(int i = 0; i < nrows; i++)
    // {
        // for (int j = 0; j < ncols; ++j)
        // {
            // // std::cout << data[i][j] << " ";
            // data(i, j) = ptr[i*ncols + j];
        // }
    // }

    // Next neighbour descent algorithm
    // Matrix<int> nn_mat = nnd_algorithm(data, n_neighbors);

    // return py::cast(mtx_storage);
    // return py::cast(nn_mat.m_data);
    py::buffer_info result_buf = py::buffer_info(
        data.m_data.data(),                           /* data as contiguous array  */
        sizeof(float),                          /* size of one scalar        */
        py::format_descriptor<float>::format(), /* data type                 */
        dim,                                    /* number of dimensions      */
        shape,                                   /* shape of the matrix       */
        strides                                  /* strides for each axis     */
    );

    timer11.stop("--4");

    py::array_t<float> result =  py::array(result_buf);

    timer11.stop("--5");
    return result;
}

// py::array py_length(py::array_t<double, py::array::c_style | py::array::forcecast> array)
// {
  // // check input dimensions
  // if ( array.ndim()     != 2 )
    // throw std::runtime_error("Input should be 2-D NumPy array");
  // if ( array.shape()[1] != 2 )
    // throw std::runtime_error("Input should have size [N,2]");

  // // allocate std::vector (to pass to the C++ function)
  // std::vector<double> pos(array.size());

  // // copy py::array -> std::vector
  // std::memcpy(pos.data(),array.data(),array.size()*sizeof(double));

  // // call pure C++ function
  // std::vector<double> result = length(pos);

  // ssize_t              ndim    = 2;
  // std::vector<ssize_t> shape   = { array.shape()[0] , 3 };
  // std::vector<ssize_t> strides = { sizeof(double)*3 , sizeof(double) };

  // // return 2-D NumPy array
  // return py::array(py::buffer_info(
    // result.data(),                           /* data as contiguous array  */
    // sizeof(double),                          /* size of one scalar        */
    // py::format_descriptor<double>::format(), /* data type                 */
    // ndim,                                    /* number of dimensions      */
    // shape,                                   /* shape of the matrix       */
    // strides                                  /* strides for each axis     */
  // ));
// }


py::array_t<int> test(py::array_t<float> py_data, int n_neighbors)
{
    py::buffer_info buf = py_data.request();
    if (buf.ndim != 2)
    {
        throw std::runtime_error("Input should be 2-D NumPy array");
    }


    // Allocate the buffer
    py::array_t<float> result = py::array_t<float>(buf.size);
    float *ptr = (float *) buf.ptr;

    int nrows = buf.shape[0];
    int ncols = buf.shape[1];

    Matrix<float> data (nrows, ncols);
    for(int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; ++j)
        {
            data(i, j) = ptr[i*ncols + j];
        }
    }

    // Do stuff with data

    Matrix<float> out (nrows, n_neighbors);
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < n_neighbors; j++)
        {
            out(i, j) = j;
        }
    }

    return py::cast(out.m_data);
}


py::array_t<int> nnd(py::array_t<float> py_data, int n_neighbors)
{
    // Allocate the buffer
    py::buffer_info buf = py_data.request();
    float *ptr = (float *) buf.ptr;

    if (buf.ndim != 2)
    {
        throw std::runtime_error("Input should be 2-D NumPy array");
    }


    // Translate input matrix from Python to C++
    int nrows = buf.shape[0];
    int ncols = buf.shape[1];
    Matrix<float> data (nrows, ncols);
    // data = py_data.cast<std::vector<std::vector<float>>>();
    for(int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; ++j)
        {
            // std::cout << data[i][j] << " ";
            data(i, j) = ptr[i*ncols + j];
        }
    }

    // Next neighbour descent algorithm
    Matrix<int> nn_mat = nnd_algorithm(data, n_neighbors);

    return py::cast(nn_mat.m_data);
}

class NNDWrapper
{
private:
public:
    Matrix<float> data;
    NNDescent nnd;

    NNDWrapper(
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
        std::cout << nnd;

        py::buffer_info py_buf = py_data.request();

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
        std::cout << nnd;
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

    py::array_t<int> get_neighbor_graph() const
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

class SomeClass {
    float multiplier;
public:
    SomeClass(float multiplier_) : multiplier(multiplier_) {};

    float multiply(float input) {
        return multiplier * input;
    }

    std::vector<float> multiply_list(std::vector<float> items) {
        for (size_t i = 0; i < items.size(); i++) {
            items[i] = multiply(items.at(i));
        }
        return items;
    }

    std::vector<std::vector<uint8_t>> make_image() {
        auto out = std::vector<std::vector<uint8_t>>();
        for (auto i = 0; i < 128; i++) {
            out.push_back(std::vector<uint8_t>(64));
        }
        for (auto i = 0; i < 30; i++) {
            for (auto j = 0; j < 30; j++) { out[i][j] = 255; }
        }
        return out;
    }

    void set_mult(float val) {
        multiplier = val;
    }

    float get_mult() {
        return multiplier;
    }

};

SomeClass some_class_factory(float multiplier) {
    return SomeClass(multiplier);
}


PYBIND11_MODULE(nndescent, m) {
    m.doc() = "Calculates approximate k-nearest neighbors.";
    m.attr("__version__") = MODULE_VERSION;
    m.def("version", &version);
    m.def("test", &test);
    m.def("_test", &_test);
    m.def("nnd", &nnd);
    m.def("some_class_factory", &some_class_factory);
    py::class_<SomeClass>(
        m, "PySomeClass"
        )
    .def(py::init<float>())
    .def_property("multiplier", &SomeClass::get_mult, &SomeClass::set_mult)
    .def("multiply", &SomeClass::multiply)
    .def("multiply_list", &SomeClass::multiply_list)
    // .def_property_readonly("image", &SomeClass::make_image)
    .def_property_readonly("image", [](SomeClass &self) {
                        py::array out = py::cast(self.make_image());
                        return out;
                    })
    // .def("multiply_two", &SomeClass::multiply_two)
    .def("multiply_two", [](SomeClass &self, float one, float two) {
                return py::make_tuple(self.multiply(one), self.multiply(two));
                })
    ;

    py::class_<NNDWrapper>(m, "NNDescent")
        .def(py::init<py::array_t<float>&, const std::string&, int, int, int,
            float, float, bool, int, bool, int, int, float, int, bool, bool,
            bool, const std::string&>(),
            py::arg("data"),
            py::arg("metric")=DEFAULT_PARMS.metric,
            py::arg("n_neighbors")=DEFAULT_PARMS.n_neighbors,
            py::arg("n_trees")=DEFAULT_PARMS.n_trees,
            py::arg("leaf_size")=DEFAULT_PARMS.leaf_size,
            py::arg("pruning_degree_multiplier")=DEFAULT_PARMS.pruning_degree_multiplier,
            py::arg("diversify_prob")=DEFAULT_PARMS.diversify_prob,
            py::arg("tree_init")=DEFAULT_PARMS.tree_init,
            py::arg("seed")=DEFAULT_PARMS.seed,
            py::arg("low_memory")=DEFAULT_PARMS.low_memory,
            py::arg("max_candidates")=DEFAULT_PARMS.max_candidates,
            py::arg("n_iters")=DEFAULT_PARMS.n_iters,
            py::arg("delta")=DEFAULT_PARMS.delta,
            py::arg("n_threads")=DEFAULT_PARMS.n_threads,
            py::arg("compressed")=DEFAULT_PARMS.compressed,
            py::arg("parallel_batch_queries")=DEFAULT_PARMS.parallel_batch_queries,
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
        .def_property_readonly("neighbor_graph", &NNDWrapper::get_neighbor_graph)
        ;
    ;
}
