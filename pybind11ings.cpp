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

IntMatrix nnd_algorithm(SlowMatrix &points, int k)
{
    IntMatrix adj_mat = nn_descent(points, k);
    return adj_mat;
}

IntMatrix bfnn_algorithm(SlowMatrix &points, int k)
{
    IntMatrix adj_mat = nn_brute_force(points, k);
    return adj_mat;
}

std::string version()
{
    return MODULE_VERSION;
}

py::array_t<int> _test(py::array_t<double> py_pnts, int k)
{
    std::vector<double> pnts(py_pnts.size());
    std::memcpy(pnts.data(),py_pnts.data(),py_pnts.size()*sizeof(double));

    std::cout << "Start\n";
    int nrows = 30000;
    IntMatrix out (nrows, std::vector<int>(k));
    std::cout << "IntMatrix init\n";
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < k; j++)
        {
            out[i][j] = j;
        }
    }
    std::cout << "loop done\n";
    return py::cast(out);
}


py::array_t<int> test(py::array_t<double> py_pnts, int k)
{
    py::buffer_info buf = py_pnts.request();
    if (buf.ndim != 2)
    {
        throw std::runtime_error("Input should be 2-D NumPy array");
    }


    // Allocate the buffer
    py::array_t<double> result = py::array_t<double>(buf.size);
    double *ptr = (double *) buf.ptr;

    int nrows = buf.shape[0];
    int ncols = buf.shape[1];

    SlowMatrix pnts (nrows, std::vector<double>(ncols));
    for(int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; ++j)
        {
            pnts[i][j] = ptr[i*ncols + j];
        }
    }

    // Do stuff with pnts

    IntMatrix out (nrows, std::vector<int>(k));
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < k; j++)
        {
            out[i][j] = j;
        }
    }

    return py::cast(out);
}

py::array_t<int> nnd(py::array_t<double> py_pnts, int k)
{
    // Allocate the buffer
    py::buffer_info buf = py_pnts.request();
    double *ptr = (double *) buf.ptr;

    if (buf.ndim != 2)
    {
        throw std::runtime_error("Input should be 2-D NumPy array");
    }


    // Translate input matrix from Python to C++
    int nrows = buf.shape[0];
    int ncols = buf.shape[1];
    SlowMatriSlowMatrix pnts (nrows, std::vector<double>(ncols));
    // pnts = py_pnts.cast<std::vector<std::vector<double>>>();
    for(int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; ++j)
        {
            // std::cout << pnts[i][j] << " ";
            pnts[i][j] = ptr[i*ncols + j];
        }
    }

    // Next neighbour descent algorithm
    IntMatrix nn_mat = nnd_algorithm(pnts, k);

    return py::cast(nn_mat);
}


class SomeClass {
    float multiplier;
public:
    SomeClass(float multiplier_) : multiplier(multiplier_) {};

    float multiply(float input) {
        return multiplier * input;
    }

    std::vector<float> multiply_list(std::vector<float> items) {
        for (auto i = 0; i < items.size(); i++) {
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


PYBIND11_MODULE(nndescent, module_handle) {
    module_handle.doc() = "Calculates approximate k-nearest neighbors.";
    module_handle.attr("__version__") = MODULE_VERSION;
    module_handle.def("version", &version);
    module_handle.def("test", &test);
    module_handle.def("_test", &_test);
    module_handle.def("nnd", &nnd);
    module_handle.def("some_class_factory", &some_class_factory);
    py::class_<SomeClass>(
        module_handle, "PySomeClass"
        ).def(py::init<float>())
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
}
