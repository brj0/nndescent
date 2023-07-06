"""
Test distance functions of pynndescent. The values are the same as in the
corresponding cpp file.
"""


import numpy as np
from pynndescent.distances import *
from pynndescent.sparse import *


def get_distances():
    return {
        "alternative_cosine": alternative_cosine,
        "alternative_dot": alternative_dot,
        "alternative_jaccard": alternative_jaccard,
        "braycurtis": bray_curtis,
        "canberra": canberra,
        "chebyshev": chebyshev,
        "circular_kantorovich": circular_kantorovich,
        "correlation": correlation,
        "cosine": cosine,
        "dice": dice,
        "dot": dot,
        "euclidean": euclidean,
        "hamming": hamming,
        "haversine": haversine,
        "hellinger": hellinger,
        "jaccard": jaccard,
        "jensen_shannon": jensen_shannon_divergence,
        "kulsinski": kulsinski,
        "manhattan": manhattan,
        "matching": matching,
        "minkowski": minkowski,
        "rogerstanimoto": rogers_tanimoto,
        "russellrao": russellrao,
        "sokalmichener": sokal_michener,
        "sokalsneath": sokal_sneath,
        "spearmanr": spearmanr,
        "sqeuclidean": squared_euclidean,
        "symmetric_kl": symmetric_kl_divergence,
        "true_angular": true_angular,
        "tsss": tsss,
        "wasserstein_1d": wasserstein_1d,
        "yule": yule,
        # "kantorovich": kantorovich,
        # "mahalanobis": mahalanobis,
        # "sinkhorn": sinkhorn,
        # "seuclidean": standardised_euclidean,
        # "wminkowski": weighted_minkowski,
    }


def test_distance(dist_name, dist, name, mtx, p_metric):
    for i in range(mtx.shape[0]):
        for j in range(i + 1, mtx.shape[0]):
            if dist_name in [
                "circular_kantorovich",
                "wasserstein_1d",
                "minkowski",
            ]:
                d = dist(mtx[i], mtx[j], p_metric)
            else:
                d = dist(mtx[i], mtx[j])
            print(f"{dist_name}({name}{i}, {name}{j}) =", d)
    print()


def test_all_distances(name, mtx, p_metric):
    mtx_prob = np.abs(mtx)
    row_sum = np.sum(mtx_prob, axis=1)
    mtx_prob = mtx_prob / row_sum[:, np.newaxis]
    mtx_2d = mtx[:, :2]
    name_prob = name + "_prob"
    name_2d = name + "_2d"
    print("# Test distances for:")
    print(name, "=\n", mtx, "\nshape =", mtx.shape)
    print()
    print(name_prob, "=\n", mtx_prob, "\nshape =", mtx_prob.shape)
    print()
    print(name_2d, "=\n", mtx_2d, "\nshape =", mtx_2d.shape)
    print()
    print()
    print("# Distance functions:\n")

    metrics = get_distances()

    for dist_name, dist in metrics.items():
        if dist_name == "haversine":
            test_distance(dist_name, dist, name_2d, mtx_2d, p_metric)
            continue
        if dist_name in ["hellinger", "jensen_shannon", "symmetric_kl"]:
            test_distance(dist_name, dist, name_prob, mtx_prob, p_metric)
            continue
        test_distance(dist_name, dist, name, mtx, p_metric)
    print()


U = np.array(
    [
        [9, 5, 6, 7, 3, 2, 1, 0, 8, -4],
        [6, 8, -2, 3, 6, 5, 4, -9, 1, 0],
        [-1, 3, 5, 1, 0, 0, -7, 6, 5, 0],
    ],
    dtype=np.float32,
)

V = np.array(
    [
        [-7, 2, 0, 3, 0, 0, -1, 2, 0, 0, 1, 0, 2, -1, 2, 0, 0, 1, 0, 0],
        [0, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 1, 0, 0, 2, 0, 2],
        [0, 1, -1, 1, 0, 0, 0, 5, 0, 0, 0, 0, 0, -4, 7, 5, 9, 1, 1, 1],
    ],
    dtype=np.float32,
)

N = 100
data_W = []
state = 0
for _ in range(3 * N):
    state = ((state * 1664525) + 1013904223) % 4294967296
    data_W.append(state % 13 - 3)
W = np.array(data_W, dtype=np.float32).reshape((3, 100))

test_all_distances("U", U, 2)
test_all_distances("V", V, 2)
test_all_distances("W", W, 2)
