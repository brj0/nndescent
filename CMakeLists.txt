cmake_minimum_required(VERSION 3.10)

project(nndescent
    VERSION 1.0.3
    LANGUAGES CXX
    DESCRIPTION "A C++ implementation of the nearest neighbour descent algorithm")

set(CMAKE_CXX_STANDARD 11)


# Enable the fast mode option that uses only euclidean and cosine metrics
option(FAST_MODE "Enable fast compile mode" OFF)

# Standard mode includes all distance functions (slow)
set(STANDARD_CXX_FLAGS
    "-Wall -g -pg -Ofast -DALL_METRICS -march=native -flto -fno-math-errno -fopenmp -pg")

# Fast mode (only euclidean/cosine distance)
set(FAST_CXX_FLAGS
    "-Wall -g -pg -Ofast -march=native -flto -fno-math-errno -fopenmp -pg")

# Select the appropriate compiler flags
if(FAST_MODE)
    set(CMAKE_CXX_FLAGS "${FAST_CXX_FLAGS}")
else()
    set(CMAKE_CXX_FLAGS "${STANDARD_CXX_FLAGS}")
endif()


# Collect source files
file(GLOB_RECURSE SRC_FILES src/*.cpp)

# Library
add_library(nndescent STATIC ${SRC_FILES})

# Source code
target_include_directories(nndescent PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)


# Tests
add_executable(test_distances tests/test_distances.cpp)
target_link_libraries(test_distances PRIVATE nndescent)

add_executable(simple tests/simple.cpp)
target_link_libraries(simple PRIVATE nndescent)

add_executable(faces tests/faces.cpp)
target_link_libraries(faces PRIVATE nndescent)

add_executable(sparse_faces tests/sparse_faces.cpp)
target_link_libraries(sparse_faces PRIVATE nndescent)

add_executable(fmnist tests/fmnist.cpp)
target_link_libraries(fmnist PRIVATE nndescent)
