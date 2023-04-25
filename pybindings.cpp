#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <vector>
#include <iostream>
#include "nnd.h"

extern "C"
{
    #include <numpy/arrayobject.h>
    static PyObject* version(PyObject* self);
    PyMODINIT_FUNC PyInit_nndescent(void);
    // static PyObject* nnd(PyObject* self, PyObject* args);
    // static PyObject* test(PyObject* self, PyObject* args);
}

IntMatrix nnd_algorithm(SlowMatrix &points, int k)
{
    Parms parms;
    parms.data = points;
    parms.n_neighbors = k;
    NNDescent nnd = NNDescent(parms);
    return nnd.neighbor_graph;
}

IntMatrix bfnn_algorithm(SlowMatrix &points, int k)
{
    IntMatrix adj_mat = nn_brute_force(points, k);
    return adj_mat;
}

int Cfib(int n)
{
    if (n < 2)
        return n;
    else
        return Cfib(n - 1) + Cfib(n - 2);
}

static PyObject* fib(PyObject* self, PyObject* args)
{
    int n;
    printf("START fib\n");

    if (!PyArg_ParseTuple(args, "i", &n))
        return NULL;
    printf("return fib\n");
    return Py_BuildValue("i", Cfib(n));
}

PyObject *to_py(std::vector<NNHeap> heap)
{
    int nrows = (int) heap.size();
    int ncols = (int) heap[0].size();
    double *data= (double*) malloc(sizeof(double)*nrows*ncols);
    npy_intp dims[] = {nrows, ncols};
    // TODO ...
    PyObject *pyobj = PyArray_SimpleNewFromData(
        2, dims, NPY_DOUBLE, (void*) data
    );
    return pyobj;
}

static PyObject* test(PyObject* self, PyObject* args)
{
    // Parse Python objects
    PyArrayObject *mat_in;
    int k;

    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &mat_in, &k))
    {
        return NULL;
    }

    if (PyArray_TYPE(mat_in) != NPY_DOUBLE)
    {
        PyErr_SetString(PyExc_TypeError, "Array must be of type double.");
        return NULL;
    }

    // Translate input matrix from Python to C++
    int nrows = PyArray_DIM(mat_in, 0);
    int ncols = PyArray_DIM(mat_in, 1);
    SlowMatrix pnts (nrows, std::vector<double>(ncols));

    for(int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; ++j)
        {
            pnts[i][j] = *(double*)(PyArray_GETPTR2(mat_in, i, j));
        }
    }

    // Do stuff with pnts

    int nrows_out = nrows;
    int ncols_out = k;

    int *lin_mat = new int[nrows_out*ncols_out];

    for(int i = 0; i < nrows_out; i++)
    {
        for (int j = 0; j < ncols_out; ++j)
        {
            lin_mat[i*ncols_out + j] = j;
        }
    }

    npy_intp dims[] = {nrows_out, ncols_out};

    // PyObject *mat_out = PyArray_SimpleNewFromData(
        // 2, dims, NPY_DOUBLE, (void*) lin_mat
    // );
    PyObject *mat_out = PyArray_SimpleNewFromData(
        2, dims, NPY_INT, (void*) lin_mat
    );

    return mat_out;
}

static PyObject* nnd(PyObject* self, PyObject* args)
{
    // Parse Python objects
    PyArrayObject *mat_in;
    int k;

    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &mat_in, &k))
    {
        return NULL;
    }

    if (PyArray_TYPE(mat_in) != NPY_DOUBLE)
    {
        PyErr_SetString(PyExc_TypeError, "Array must be of type double.");
        return NULL;
    }

    // Translate input matrix from Python to C++
    int nrows = PyArray_DIM(mat_in, 0);
    int ncols = PyArray_DIM(mat_in, 1);
    SlowMatrix pnts (nrows, std::vector<double>(ncols));

    for(int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; ++j)
        {
            pnts[i][j] = *(double*)(PyArray_GETPTR2(mat_in, i, j));
        }
    }

    // Next neighbour descent algorithm
    IntMatrix nn_mat = nnd_algorithm(pnts, k);
    nrows = nn_mat.size();
    ncols = nn_mat[0].size();

    int *data = new int[nrows * nn_mat[0].size()];
    for(int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; ++j)
        {
            data[i*ncols + j] = nn_mat[i][j];
        }
    }
    npy_intp dims[] = {nn_mat.size(), nn_mat[0].size()};
    PyObject *mat_out = PyArray_SimpleNewFromData(
        2, dims, NPY_INT, (void*) data
    );

    return mat_out;
}

static PyObject* bfnn(PyObject* self, PyObject* args)
{
    // Parse Python objects
    PyArrayObject *mat_in;
    int k;

    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &mat_in, &k))
    {
        return NULL;
    }

    if (PyArray_TYPE(mat_in) != NPY_DOUBLE)
    {
        PyErr_SetString(PyExc_TypeError, "Array must be of type double.");
        return NULL;
    }

    // Translate input matrix from Python to C++
    int nrows = PyArray_DIM(mat_in, 0);
    int ncols = PyArray_DIM(mat_in, 1);
    SlowMatrix pnts (nrows, std::vector<double>(ncols));

    for(int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; ++j)
        {
            pnts[i][j] = *(double*)(PyArray_GETPTR2(mat_in, i, j));
        }
    }

    // Next neighbour descent algorithm
    IntMatrix nn_mat = bfnn_algorithm(pnts, k);
    nrows = nn_mat.size();
    ncols = nn_mat[0].size();

    int *data = new int[nrows * nn_mat[0].size()];
    for(int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; ++j)
        {
            data[i*ncols + j] = nn_mat[i][j];
        }
    }
    npy_intp dims[] = {nn_mat.size(), nn_mat[0].size()};
    PyObject *mat_out = PyArray_SimpleNewFromData(
        2, dims, NPY_INT, (void*) data
    );

    return mat_out;
}

static PyObject* version(PyObject* self)
{
    return Py_BuildValue("s", "version 1.0");
}

static PyMethodDef nnd_methods[] = {
    {"nnd", nnd, METH_VARARGS, "Calculates approximate k-nearest neighbors."},
    {"bfnn", bfnn, METH_VARARGS, "Calculates k-nearest neighbors."},
    {"test", test, METH_VARARGS, "TEST"},
    {"fib", fib, METH_VARARGS, "Fibonacci"},
    {"version", (PyCFunction)version, METH_NOARGS, "Returns the version."},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef nndescent = {
    PyModuleDef_HEAD_INIT,
    "nndescent",
    "NNDescent Module",
    -1,
    nnd_methods
};

PyMODINIT_FUNC PyInit_nndescent(void)
{
    import_array();
    return PyModule_Create(&nndescent);
}
