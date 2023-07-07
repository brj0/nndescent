/**
 * @file dtypes.h
 *
 * @brief Data types used (Matrix, CSRMatrix, Heap, HeapList).
 */


#pragma once

#include <cmath>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#include "utils.h"


namespace nndescent
{


/**
 * @brief A template class representing a 2D matrix of elements of type T.
 */
template <class T>
class Matrix
{
private:

    /**
     * The number of rows in the matrix.
     */
    size_t m_rows;

    /**
     * The number of columns in the matrix.
     */
    size_t m_cols;

public:

    /**
     * A vector containing the data elements of the matrix.
     */
    std::vector<T> m_data;

    /**
     * A pointer to the raw data of the matrix.
     */
    T* m_ptr;

    /**
     * Default constructor. Creates an empty matrix.
     */
    Matrix();

    /**
     * Constructor to create a matrix with the specified number of rows and
     * columns. The matrix elements are uninitialized.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     */
    Matrix(size_t rows, size_t cols);

    /**
     * Constructor to create a matrix with the specified number of rows and
     * columns, and initialize all elements to a constant value.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param const_val The constant value to initialize all elements.
     */
    Matrix(size_t rows, size_t cols, const T &const_val);

    /**
     * Constructor to create a matrix with the specified number of rows and
     * columns, and initialize elements from a pointer to external data.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param data_ptr A pointer to the external data to be used as matrix
     * elements.
     */
    Matrix(size_t rows, size_t cols, T *data_ptr);

    /**
     * Constructor to create a matrix with the specified number of rows and
     * columns, and initialize elements from a vector.
     *
     * @param rows The number of rows in the matrix.
     * @param data A vector containing the data elements to be used as matrix
     * elements.
     */
    Matrix(size_t rows, std::vector<T> &data);

    /**
     * Copy constructor. Creates a new matrix as a copy of another matrix.
     *
     * @param other The matrix to be copied.
     */
    Matrix(const Matrix<T>& other);

    /**
     * Move constructor. Creates a new matrix by moving the data from another
     * matrix.
     *
     * @param other The matrix to be moved.
     */
    Matrix(Matrix<T>&& other) noexcept;

    /**
     * Copy assignment operator. Assigns the contents of another matrix to
     * this matrix.
     *
     * @param other The matrix to be copied.
     * @return A reference to this matrix after the assignment.
     */
    Matrix<T>& operator=(const Matrix<T>& other);

    /**
     * Resizes an empty matrix to the specified number of rows and columns.
     *
     * @param rows The new number of rows in the matrix.
     * @param cols The new number of columns in the matrix.
     */
    void resize(size_t rows, size_t cols);

    /**
     * Accesses the element at the specified row and column index in the
     * matrix.
     *
     * @param i The row index.
     * @param j The column index.
     * @return A reference to the element at the specified position.
     */
    inline T& operator()(size_t i, size_t j)
    {
        return m_ptr[i * m_cols + j];
    }

    /**
     * Accesses the element at the specified row and column index in the
     * matrix (const version).
     *
     * @param i The row index.
     * @param j The column index.
     * @return The value of the element at the specified position.
     */
    inline const T operator()(size_t i, size_t j) const
    {
        return m_ptr[i * m_cols + j];
    }

    /**
     * Returns the number of rows in the matrix.
     *
     * @return The number of rows.
     */
    size_t nrows() const { return m_rows; }

    /**
     * Returns the number of columns in the matrix.
     *
     * @return The number of columns.
     */
    size_t ncols() const { return m_cols; }

    /**
     * Accesses the element at the specified row index in the matrix.
     *
     * @param i The row index.
     * @return A pointer to the elements of the specified row.
     */
    T* operator[](size_t i){ return m_ptr + i*m_cols; }

    /**
     * Returns a pointer to the beginning of the specified row in the matrix.
     *
     * @param i The row index.
     * @return A pointer to the beginning of the specified row.
     */
    T* begin(size_t i) const { return m_ptr + i*m_cols; }

    /**
     * Returns a pointer to the end of the specified row in the matrix.
     *
     * @param i The row index.
     * @return A pointer to the end of the specified row.
     */
    T* end(size_t i) const { return m_ptr + (i + 1)*m_cols; }

    /**
     * Returns the count of non-none elements in the matrix.
     *
     * @return The count of non-none elements.
     */
    int non_none_cnt();

    /**
     * @brief Normalize each row of the matrix using the L2 norm.
     *
     * Note that the normalization is only performed on non-zero norm rows to
     * avoid division by zero.
     */
    void normalize();

    /**
     * @brief Creates a deep copy of the data storage if necessary.
     *
     *  If the matrix was initialized with a pointer, this function creates a
     *  deep copy of the data storage, ensuring that modifications to the copy
     *  do not affect the original data.
     */
    void deep_copy()
    {
        if (m_data.empty())
        {
            m_data.assign(m_ptr, m_ptr + m_rows*m_cols);
        }
        m_ptr = &m_data[0];
    }
};


template <class T>
Matrix<T>::Matrix()
    : m_rows(0)
    , m_cols(0)
    , m_data(0)
    , m_ptr(&m_data[0])
{
}


template <class T>
Matrix<T>::Matrix(size_t rows, size_t cols)
    : m_rows(rows)
    , m_cols(cols)
    , m_data(rows * cols)
    , m_ptr(&m_data[0])
{
}


template <class T>
Matrix<T>::Matrix(size_t rows, size_t cols, T *data_ptr)
    : m_rows(rows)
    , m_cols(cols)
    , m_data(0)
    , m_ptr(data_ptr)
{
}


template <class T>
Matrix<T>::Matrix(size_t rows, size_t cols, const T &const_val)
    : m_rows(rows)
    , m_cols(cols)
    , m_data(rows * cols, const_val)
    , m_ptr(&m_data[0])
{
}


template <class T>
Matrix<T>::Matrix(size_t rows, std::vector<T> &data)
    : m_rows(rows)
    , m_cols(data.size()/rows)
    , m_data(data)
    , m_ptr(&m_data[0])
{
}


template <class T>
Matrix<T>::Matrix(const Matrix<T>& other)
    : m_rows(other.m_rows)
    , m_cols(other.m_cols)
    , m_data(other.m_data)
    , m_ptr(m_data.empty() ? other.m_ptr : &m_data[0])
{
}


template <class T>
Matrix<T>::Matrix(Matrix<T>&& other) noexcept
    : m_rows(other.m_rows)
    , m_cols(other.m_cols)
    , m_data(std::move(other.m_data))
    , m_ptr(m_data.empty() ? other.m_ptr : &m_data[0])
{
    other.m_ptr = nullptr;
}


template <class T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other)
{
    if (this != &other)
    {
        m_data = other.m_data;
        m_ptr = m_data.empty() ? other.m_ptr : &m_data[0];
        m_rows = other.m_rows;
        m_cols = other.m_cols;
    }
    return *this;
}


template <class T>
void Matrix<T>::resize(size_t rows, size_t cols)
{
    m_rows = rows;
    m_cols = cols;
    m_data.resize(rows*cols);
    m_ptr = &m_data[0];
}


template <class T>
int Matrix<T>::non_none_cnt()
{
    int cnt = 0;
    for (auto ptr = m_ptr; ptr != m_ptr + m_rows*m_cols; ++ptr)
    {
        cnt += (*ptr) == NONE ? 0 : 1;
    }
    return cnt;
}


template <class T>
void Matrix<T>::normalize()
{
    for (size_t i = 0; i < m_rows; ++i)
    {
        float norm = 0.0f;
        for (size_t j = 0; j < m_cols; ++j)
        {
            norm += (*this)(i, j) * (*this)(i, j);
        }
        norm = std::sqrt(norm);

        // Avoid division by zero
        if (norm > 0.0f)
        {
            for (size_t j = 0; j < m_cols; ++j)
            {
                (*this)(i, j) /= norm;
            }
        }
    }

}


/*
 * @brief Prints an Matrix<T> object to an output stream.
 */
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


/**
 * @brief CSRMatrix class represents a compressed sparse row (CSR) matrix.
 *
 * The CSRMatrix class is used to store and manipulate sparse matrices in
 * compressed sparse row format. It provides efficient storage and access to
 * the non-zero elements of the matrix.
 *
 * @tparam T The data type of the matrix elements.
 */
template<class T>
class CSRMatrix
{
private:

    /**
     * The number of rows in the matrix.
     */
    size_t m_rows;

    /**
     * The number of columns in the matrix.
     */
    size_t m_cols;

public:

    /**
     * Vector storing the non-zero values of the matrix.
     */
    std::vector<T> m_data;

    /**
     * Vector storing the column indices of the non-zero values.
     */
    std::vector<size_t> m_col_ind;

    /**
     * Vector storing the row pointers indicating the start of each row.
     */
    std::vector<size_t> m_row_ptr;

    /**
     * Pointer to the non-zero values of the matrix.
     */
    T* m_ptr_data;

    size_t* m_ptr_col_ind;

    /**
     * Default constructor. Creates an empty matrix.
     */
    CSRMatrix();

    /**
     * @brief Constructor for the CSRMatrix class.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param data Vector storing the non-zero values of the matrix.
     * @param col_ind Vector storing the column indices of the non-zero values.
     * @param row_ptr Vector storing the row pointers indicating the start of
     * each row.
     */
    CSRMatrix(
        size_t rows,
        size_t cols,
        std::vector<T> &data,
        std::vector<size_t> &col_ind,
        std::vector<size_t> &row_ptr
    );

    /**
     * @brief Constructor for the CSRMatrix class.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param nnz The number of non-zero elements in the matrix.
     * @param data Pointer to the non-zero values of the matrix.
     * @param col_ind Pointer to the column indices of the non-zero values.
     * @param row_ptr Pointer to the row pointers indicating the start of each
     * row.
     */
    CSRMatrix(
        size_t rows,
        size_t cols,
        size_t nnz,
        T *data,
        size_t *col_ind,
        size_t *row_ptr
    );

    /**
     * @brief Constructor for the CSRMatrix class.
     *
     * @param matrix The dense matrix to convert to CSR format.
     */
    explicit CSRMatrix(Matrix<T> matrix);

    /**
     * @brief Copy constructor for the CSRMatrix class.
     *
     * @param other The CSRMatrix object to be copied.
     */
    CSRMatrix(const CSRMatrix<T>& other);

    /**
     * @brief Move constructor for the CSRMatrix class.
     *
     * @param other The CSRMatrix object to be moved.
     */
    CSRMatrix(CSRMatrix<T>&& other) noexcept;

    /**
     * @brief Assignment operator for the CSRMatrix class.
     *
     * @param other The CSRMatrix object to be assigned.
     *
     * @return Reference to the assigned CSRMatrix object.
     */
    CSRMatrix<T>& operator=(const CSRMatrix<T>& other);

    /**
     * Accesses the element at the specified row and column index in the matrix
     * (const version).
     *
     * @param i The row index.
     * @param j The column index.
     *
     * @return The value of the element at the specified position.
     */
    inline const T operator()(size_t i, size_t j) const
    {
        for (size_t k = m_row_ptr[i]; k < m_row_ptr[i + 1]; k++)
        {
            if (m_col_ind[k] == j)
            {
                return m_data[k];
            }
        }
        return (T)0;
    }

    /**
     * @brief Returns a pointer to the beginning of the column indices for the
     * specified row.
     *
     * @param i The index of the row.
     *
     * @return Pointer to the beginning of the column indices for the specified
     * row.
     */
    size_t* begin_col(size_t i) const
    {
        return m_ptr_col_ind + m_row_ptr[i];
    }

    /**
     * @brief Returns a pointer to the end of the column indices for the
     * specified row.
     *
     * @param i The index of the row.
     *
     * @return Pointer to the end of the column indices for the specified row.
     */
    size_t* end_col(size_t i) const
    {
        return m_ptr_col_ind + m_row_ptr[i + 1];
    }

    /**
     * @brief Returns a pointer to the beginning of the non-zero values for the
     * specified row.
     *
     * @param i The index of the row.
     *
     * @return Pointer to the beginning of the non-zero values for the
     * specified row.
     */
    T* begin_data(size_t i) const
    {
        return m_ptr_data + m_row_ptr[i];
    }

    /**
     * @brief Returns a pointer to the end of the non-zero values for the
     * specified row.
     *
     * @param i The index of the row.
     * @return Pointer to the end of the non-zero values for the specified row.
     */
    T* end_data(size_t i) const
    {
        return m_ptr_data + m_row_ptr[i + 1];
    }

    /**
     * @brief Normalize each row of the matrix using the L2 norm.
     *
     * Note that the normalization is only performed on non-zero norm rows to
     * avoid division by zero.
     */
    void normalize();

    /**
     *  This function does not modify the current object. It is provided for
     *  compatibility purposes.
     */
    void deep_copy() {}

    /**
     * Returns the number of rows in the matrix.
     *
     * @return The number of rows.
     */
    size_t nrows() const { return m_rows; }

    /**
     * Returns the number of columns in the matrix.
     *
     * @return The number of columns.
     */
    size_t ncols() const { return m_cols; }
};


template <class T>
CSRMatrix<T>::CSRMatrix()
    : m_rows(0)
    , m_cols(0)
    , m_data(0)
    , m_col_ind(0)
    , m_row_ptr(0)
    , m_ptr_data(&m_data[0])
    , m_ptr_col_ind(&m_col_ind[0])
{
}


template <class T>
CSRMatrix<T>::CSRMatrix(
    size_t rows,
    size_t cols,
    std::vector<T> &data,
    std::vector<size_t> &col_ind,
    std::vector<size_t> &row_ptr
)
    : m_rows(rows)
    , m_cols(cols)
    , m_data(data)
    , m_col_ind(col_ind)
    , m_row_ptr(row_ptr)
    , m_ptr_data(&m_data[0])
    , m_ptr_col_ind(&m_col_ind[0])
{
}


template <class T>
CSRMatrix<T>::CSRMatrix(
    size_t rows,
    size_t cols,
    size_t nnz,
    T *data,
    size_t *col_ind,
    size_t *row_ptr
)
    : m_rows(rows)
    , m_cols(cols)
    , m_data(data, data + nnz)
    , m_col_ind(col_ind, col_ind + nnz)
    , m_row_ptr(row_ptr, row_ptr + rows + 1)
    , m_ptr_data(&m_data[0])
    , m_ptr_col_ind(&m_col_ind[0])
{
}


template <class T>
CSRMatrix<T>::CSRMatrix(Matrix<T> matrix)
    : m_rows(matrix.nrows())
    , m_cols(matrix.ncols())
    , m_row_ptr(matrix.nrows() + 1, (T)0)
    , m_ptr_col_ind(&m_col_ind[0])
{
    for (size_t i = 0; i < m_rows; ++i)
    {
        for (size_t j = 0; j < m_cols; ++j)
        {
            if (matrix(i, j) != (T)0)
            {
                m_data.push_back(matrix(i, j));
                m_col_ind.push_back(j);
                ++m_row_ptr[i + 1];
            }
        }
    }
    m_ptr_col_ind = &m_col_ind[0];
    m_ptr_data = &m_data[0];
    for (size_t i = 1; i <= m_rows; i++)
    {
        m_row_ptr[i] += m_row_ptr[i - 1];
    }
}


template <class T>
CSRMatrix<T>::CSRMatrix(const CSRMatrix<T>& other)
    : m_rows(other.m_rows)
    , m_cols(other.m_cols)
    , m_data(other.m_data)
    , m_col_ind(other.m_col_ind)
    , m_row_ptr(other.m_row_ptr)
    , m_ptr_data(&m_data[0])
    , m_ptr_col_ind(&m_col_ind[0])
{
}


template <class T>
CSRMatrix<T>::CSRMatrix(CSRMatrix<T>&& other) noexcept
    : m_rows(other.m_rows)
    , m_cols(other.m_cols)
    , m_data(std::move(other.m_data))
    , m_col_ind(std::move(other.m_col_ind))
    , m_row_ptr(std::move(other.m_row_ptr))
    , m_ptr_data(&m_data[0])
    , m_ptr_col_ind(&m_col_ind[0])
{
    other.m_ptr_data = nullptr;
    other.m_ptr_col_ind = nullptr;
}


template <class T>
CSRMatrix<T>& CSRMatrix<T>::operator=(const CSRMatrix<T>& other)
{
    if (this != &other)
    {
        m_rows = other.m_rows;
        m_cols = other.m_cols;
        m_data = other.m_data;
        m_col_ind = other.m_col_ind;
        m_row_ptr = other.m_row_ptr;
        m_ptr_data = &m_data[0];
        m_ptr_col_ind = &m_col_ind[0];
    }
    return *this;
}


template <class T>
void CSRMatrix<T>::normalize()
{
    for (size_t i = 0; i < m_rows; ++i)
    {
        float norm = 0.0f;
        for (size_t j = m_row_ptr[i]; j < m_row_ptr[i + 1]; ++j)
        {
            norm += m_data[j] * m_data[j];
        }
        norm = std::sqrt(norm);

        // Avoid division by zero
        if (norm > 0.0f)
        {
            for (size_t j = m_row_ptr[i]; j < m_row_ptr[i + 1]; ++j)
            {
                m_data[j] /= norm;
            }
        }
    }

}


/*
* @brief Prints an CSRMatrix<T> object to an output stream.
*/
template <class T>
std::ostream& operator<<(std::ostream &out, CSRMatrix<T> const& matrix)
{
    out << "CSRMatrix(m_rows=" << matrix.nrows()
        << ", m_cols=" << matrix.ncols() << ",\n";
    for (size_t i = 0; i < matrix.nrows(); ++i)
    {
        for (size_t j = matrix.m_row_ptr[i]; j < matrix.m_row_ptr[i + 1]; ++j)
        {
            out << "    (" << i << ", " << matrix.m_col_ind[j] << ")\t"
                << matrix.m_data[j] << "\n";
        }
    }
    out << ")\n";
    return out;
}


/*
 * @brief A struct for nearst neighbor candidates in a query search.
 */
struct Candidate
{
    /*
     * The identifier of the candidate.
     */
    int idx;

    /*
     * The key/distance of the candidate.
     */
    float key;

    /*
     * Overloaded less-than operator used to store Candidates in a MinHeap.
     */
    bool operator<(const Candidate& other) const
    {
        return key > other.key;
    }
};


/*
 * @brief A template class representing a maximum heap data structure.
 */
template<class T>
class Heap
{

private:

    /*
     * The underlying priority queue used to implement the heap.
     */
    std::priority_queue<T> heap;

public:

    /*
     * Pushes a new element into the heap.
     *
     * @param value The value to be pushed into the heap.
     */
    void push(const T& value)
    {
        heap.push(value);
    }

    /*
     * Removes and returns the top element from the heap.
     *
     * @return The top element of the heap.
     */
    T pop()
    {
        T top = heap.top();
        heap.pop();
        return top;
    }

    /*
     * Checks if the heap is empty.
     *
     * @return True if the heap is empty, false otherwise.
     */
    bool empty() const
    {
        return heap.empty();
    }

};


/*
 * @brief A cache-friendly implementation of a list of maximum heaps.
 *
 * The HeapList class provides a cache-friendly representation of multiple
 * heaps, each containing nodes with associated indices, keys and flags.
 * It supports operations such as pushing nodes into the heaps, sorting
 * the heaps, and retrieving information about the heaps.
 *
 * @tparam KeyType The type of keys associated with the nodes.
 */
template <class KeyType>
class HeapList
{

private:

    /*
     * Number of heaps.
     */
    size_t n_heaps;

    /*
     * Number of nodes.
     */
    size_t n_nodes;

public:

    /*
     * Matrix storing indices of nodes in the heaps.
     */
    Matrix<int> indices;

    /*
     * Matrix storing keys associated with nodes.
     */
    Matrix<KeyType> keys;

    /*
     * Matrix storing flags associated with nodes.
     *
     * As bool leads to problems since std::vector<bool> is special from all
     * other std::vector specializations, char is used instead.
     */
    Matrix<char> flags;

    /*
     * Default constructor. Creates an empty HeapList.
     */
    HeapList() : n_heaps(0), n_nodes(0) {}

    /*
     * Constructor that initializes the HeapList with specified parameters.
     *
     * @param n_heaps The number of heaps.
     * @param n_nodes The number of nodes in each heap.
     * @param key0 The initial key value for all nodes.
     * @param flag0 The initial flag value for all nodes.
     */
    HeapList(size_t n_heaps, size_t n_nodes, KeyType key0, char flag0)
        : n_heaps(n_heaps)
        , n_nodes(n_nodes)
        , indices(n_heaps, n_nodes, NONE)
        , keys(n_heaps, n_nodes, key0)
        , flags(n_heaps, n_nodes, flag0)
    {
    }

    /*
     * Constructor that initializes the HeapList with specified parameters,
     * using the same initial key value for all nodes and with no flags.
     *
     * @param n_heaps The number of heaps.
     * @param n_nodes The number of nodes in each heap.
     * @param key0 The initial key value for all nodes.
     */
    HeapList(size_t n_heaps, size_t n_nodes, KeyType key0)
        : n_heaps(n_heaps)
        , n_nodes(n_nodes)
        , indices(n_heaps, n_nodes, NONE)
        , keys(n_heaps, n_nodes, key0)
        , flags(0, 0)
    {
    }

    /*
     * Retrieves the number of heaps in the HeapList.
     *
     * @return The number of heaps.
     */
    size_t nheaps() const { return n_heaps; }

    /*
     * Retrieves the number of nodes in each heap of the HeapList.
     *
     * @return The number of nodes in each heap.
     */
    size_t nnodes() const { return n_nodes; }

    /*
     * Checks if the HeapList has no flags associated with the nodes.
     *
     * @return True if the HeapList has no flags, false otherwise.
     */
    bool noflags() const { return flags.nrows() == 0; }

    /*
     * Retrieves the maximum key value in the specified heap.
     *
     * @param i The index of the heap.
     *
     * @return The maximum key value in the heap.
     */
    KeyType max(size_t i) const { return keys(i, 0); }

    /*
     * Retrieves the number of non-'NONE' nodes in the specified heap.
     *
     * @param i The index of the heap.
     *
     * @return The number of non-'NONE' nodes of the heap.
     */
    size_t size(size_t i) const;

    /*
     * Pushes a node with the specified index, key, and flag into the
     * specified heap if its key is smaller and it is not already in the heap.
     *
     * @param i The index of the heap.
     * @param idx The index of the node.
     * @param key The key associated with the node.
     * @param flag The flag associated with the node.
     *
     * @return 1 if the node was added to the heap, 0 otherwise.
     */
    int checked_push(size_t i, int idx, KeyType key, char flag);

    /*
     * Pushes a node with the specified index and key into the specified heap
     * if its key is smaller and it is not already in the heap.
     *
     * @param i The index of the heap.
     * @param idx The index of the node.
     * @param key The key associated with the node.
     *
     * @return 1 if the node was added to the heap, 0 otherwise.
     */
    int checked_push(size_t i, int idx, KeyType key);

    /*
     * Performs a "siftdown" operation on the specified heap starting from the
     * given index.
     *
     * The "siftdown" operation descends the top node down the heap by swapping
     * values until the maximum heap criterion is met or 'stop' is reached.
     *
     * @param i The index of the heap to perform the siftdown operation on.
     * @param stop The index at which to stop the siftdown operation.
     */
    void siftdown(size_t i, size_t stop);

    /*
     * @brief Sorts all heaps in ascending key order.
     *
     * As the heap criterion is already met only the second part of the
     * "Heapsort" algorithm is executed.
     */
    void heapsort();
};


template <class KeyType>
void HeapList<KeyType>::heapsort()
{
    int tmp_id;
    KeyType tmp_key;

    for (size_t i = 0; i < n_heaps; ++i)
    {
        for (size_t j = n_nodes - 1; j > 0; --j)
        {
            tmp_id = indices(i, 0);
            tmp_key = keys(i, 0);

            indices(i, 0) = indices(i, j);
            keys(i, 0) = keys(i, j);

            indices(i, j) = tmp_id;
            keys(i, j) = tmp_key;

            this->siftdown(i, j);
        }
    }
}


template <class KeyType>
void HeapList<KeyType>::siftdown(size_t i, size_t stop)
{
    KeyType key = keys(i, 0);
    int idx = indices(i, 0);

    size_t current = 0;
    size_t swap;

    while (true)
    {
        size_t left_child = 2*current + 1;
        size_t right_child = left_child + 1;

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

        current = swap;
    }
    // Insert node at current position.
    indices(i, current) = idx;
    keys(i, current) = key;
}


template <class KeyType>
int HeapList<KeyType>::checked_push(size_t i, int idx, KeyType key, char flag)
{
    if (key >= keys(i, 0))
    {
        return 0;
    }

    // Break if we already have this element.
    for (auto it = indices.begin(i); it != indices.end(i); ++it)
    {
        if (*it == idx)
        {
            return 0;
        }
    }

    // Siftdown: Descend the heap, swapping values until the max heap
    // criterion is met.
    size_t current = 0;
    size_t swap;

    while (true)
    {
        size_t left_child = 2*current + 1;
        size_t right_child = left_child + 1;

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
    for (auto it = indices.begin(i); it != indices.end(i); ++it)
    {
        if (*it == idx)
        {
            return 0;
        }
    }

    // Siftdown: Descend the heap, swapping values until the max heap
    // criterion is met.
    size_t current = 0;
    size_t swap;

    while (true)
    {
        size_t left_child = 2*current + 1;
        size_t right_child = left_child + 1;

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
size_t HeapList<KeyType>::size(size_t i) const
{
    size_t count = count_if_not_equal(indices.begin(i), indices.end(i), NONE);
    return count;
}


/*
 * Auxiliary function for recursively printing a binary Heap.
 */
template <class KeyType>
void _add_heap_from_to_stream(
    std::ostream &out,
    const std::string &prefix,
    HeapList<KeyType> &heaplist,
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


/*
 * Auxiliary function for printing a binary Heap.
 */
template <class KeyType>
void add_heap_to_stream(std::ostream &out, HeapList<KeyType> &heaplist, size_t i)
{
    out << i << " [size=" << heaplist.nnodes() << "]\n";
    _add_heap_from_to_stream(out, "    ", heaplist, i, 0, false);
}


/*
 * @brief Prints a HeapList<KeyType> object to an output stream.
 */
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


/*
 * @brief Debug function to print the data as 2d map.
 */
void print_map(Matrix<float> matrix);


/*
 * @brief Prints a vector object to an output stream.
 */
template <class T>
std::ostream& operator<<(std::ostream &out, const std::vector<T> &vec)
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


/*
 * @brief Prints a vectorized Matrix object to an output stream.
 */
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


/*
 * @brief A struct holding two node identifiers and the key/distance between
 * the nodes.
 */
typedef struct
{
    int idx0;
    int idx1;
    float key;
} NNUpdate;


/*
 * @brief Prints a NNUpdate object to an output stream.
 */
std::ostream& operator<<(std::ostream &out, NNUpdate &update);


/*
 * @brief Prints a vector of NNUpdate objects to an output stream.
 */
std::ostream& operator<<(std::ostream &out, std::vector<NNUpdate> &updates);


} // namespace nndescent
