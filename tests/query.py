import nndescent
import pynndescent
from urllib.request import urlretrieve

import time
import numpy as np

import os
import h5py

DATA_PATH = os.path.expanduser("~/Downloads/nndescent_test_data")
os.makedirs(DATA_PATH, exist_ok=True)


class Timer:
    """Measures the time elapsed in milliseconds."""

    def __init__(self):
        self.time0 = time.time()

    def start(self):
        self.time0 = time.time()

    def stop(self, text=None):
        delta_time = 1000 * (time.time() - self.time0)
        comment = "" if text is None else "(" + text + ")"
        print("Time passed:", delta_time, "ms", comment)
        self.time0 = time.time()
        return delta_time


timer = Timer()

def ann_benchmark_data(dataset_name):
    """Downloads data if necessary and returns as arrays."""
    data_path = os.path.join(DATA_PATH, f"{dataset_name}.hdf5")
    if not os.path.exists(data_path):
        print(f"Dataset {dataset_name} is not cached; downloading now ...")
        urlretrieve(
            f"http://ann-benchmarks.com/{dataset_name}.hdf5",
            data_path,
        )
    hdf5_file = h5py.File(data_path, "r")
    nn_ect = np.array(hdf5_file["neighbors"])
    nn_ect = np.c_[range(nn_ect.shape[0]), nn_ect]
    return (
        np.array(hdf5_file["train"]),
        np.array(hdf5_file["test"]),
        hdf5_file.attrs["distance"],
        nn_ect,
    )


def accuracy(approx_neighbors, true_neighbors):
    result = np.zeros(approx_neighbors.shape[0])
    for i in range(approx_neighbors.shape[0]):
        n_correct = np.intersect1d(
            approx_neighbors[i], true_neighbors[i]
        ).shape[0]
        result[i] = n_correct / true_neighbors.shape[1]
    print(f"Average accuracy of {np.mean(result)}\n")
    return result


train, test, dist, nn_train = ann_benchmark_data("fashion-mnist-784-euclidean")


kwargs = {
    "data": train,
    "n_neighbors": 30,
    "verbose": True,
    "metric": "euclidean",
}

pyindex = pynndescent.NNDescent(**kwargs)
index = nndescent.NNDescent(**kwargs)

accuracy(pyindex.neighbor_graph[0], index.neighbor_graph[0])

pyindex.prepare()
neighbors = pyindex.query(test)
