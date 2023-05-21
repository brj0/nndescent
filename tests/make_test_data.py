"""
Generates and downloads test data. Most tests require this script to be
run in advance.
"""

from urllib.request import urlretrieve
import os

import h5py
import numpy as np
import pandas as pd

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
coil20 = np.array(
    pd.read_csv(
        "https://raw.githubusercontent.com/akmand/datasets/main/coil20.csv"
    )
)

# ANN Benchmark datasets. Comment out accordingly.
# https://github.com/erikbern/ann-benchmarks
ANNBEN = {
    "deep": "deep-image-96-angular",
    "fmnist": "fashion-mnist-784-euclidean",
    "gist": "gist-960-euclidean",
    "glove25": "glove-25-angular",
    "glove50": "glove-50-angular",
    "glove100": "glove-100-angular",
    "glove200": "glove-200-angular",
    # "kosark": "kosarak-jaccard", # url not working
    "mnist": "mnist-784-euclidean",
    # "movielens": "movielens10m-jaccard", # url not working
    "nytimes": "nytimes-256-angular",
    "sift": "sift-128-euclidean",
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
    else:
        print(f"Reading {dataset_name} from disk ...")
    hdf5_file = h5py.File(data_path, "r")
    return (
        np.array(hdf5_file["train"]),
        np.array(hdf5_file["test"]),
    )

annb = {}
for nm, nm_long in ANNBEN.items():
    train, test = ann_benchmark_data(nm_long)
    annb[nm + "_train"] = train


# Save datasets as csv to Downloads folder.
datasets = {
    "simple": simple,
    "coil20": coil20,
    **annb
}

for name, data in datasets.items():
    print("Saving", name, "as csv")
    path = os.path.join(DATA_PATH, f"{name}.csv")
    np.savetxt(path, data, delimiter=",", fmt="%g")
    del(data)

print("Test data created successfully")
