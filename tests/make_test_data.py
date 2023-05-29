"""
Generates and downloads test data. Most tests require this script to be
run in advance.
"""

from urllib.request import urlretrieve
import os

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces

DATA_PATH = os.path.expanduser("~/Downloads/nndescent_test_data")
os.makedirs(DATA_PATH, exist_ok=True)

# Toy dataset useful for debugging.
simple = np.array(
    [
        [1, 10],
        [17, 5],
        [59, 5],
        [60, 5],
        [9, 13],
        [60, 13],
        [17, 19],
        [54, 19],
        [52, 20],
        [54, 22],
        [9, 25],
        [70, 28],
        [31, 31],
        [9, 32],
        [52, 32],
        [67, 33],
    ]
)

# Small data set, useful to quickly check correctness of the algorithm.
olivetti_faces = fetch_olivetti_faces(data_home=DATA_PATH).data
faces_train, faces_test = train_test_split(
    olivetti_faces, test_size=0.2, random_state=1234
)

# ANN Benchmark datasets. Comment out accordingly.
# https://github.com/erikbern/ann-benchmarks
ANNBEN = {
    # "deep": "deep-image-96-angular",
    "fmnist": "fashion-mnist-784-euclidean",
    # "gist": "gist-960-euclidean",
    # "glove25": "glove-25-angular",
    # "glove50": "glove-50-angular",
    # "glove100": "glove-100-angular",
    # "glove200": "glove-200-angular",
    # "kosark": "kosarak-jaccard", # url not working
    "mnist": "mnist-784-euclidean",
    # "movielens": "movielens10m-jaccard", # url not working
    # "nytimes": "nytimes-256-angular",
    # "sift": "sift-128-euclidean",
    # "lastfm": "lastfm-64-dot", # url not working
}


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


annb = {}
for nm, nm_long in ANNBEN.items():
    train, test, _, ect = ann_benchmark_data(nm_long)
    annb[nm + "_train"] = train
    annb[nm + "_test"] = test
    annb[nm + "_test_ect"] = ect


# Save datasets as csv to Downloads folder.
datasets = {
    "simple": simple,
    "faces_train": faces_train,
    "faces_test": faces_test,
    **annb,
}

for name, data in datasets.items():
    print("Saving", name, "as csv")
    path = os.path.join(DATA_PATH, f"{name}.csv")
    np.savetxt(path, data, delimiter=",", fmt="%g")
    del data

print("Test data created successfully")
