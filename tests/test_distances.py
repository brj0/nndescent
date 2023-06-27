"""
Test distance functions of pynndescent. The values are the same as in the
corresponding cpp file.
"""


import numpy as np
from pynndescent.distances import *
from pynndescent.sparse import *


def test_distance(dist_name, dist, vec_name, v0, v1, v2):
    print(f"{dist_name}({vec_name}0, {vec_name}1) =", dist(v0, v1))
    print(f"{dist_name}({vec_name}0, {vec_name}2) =", dist(v0, v2))
    print(f"{dist_name}({vec_name}1, {vec_name}2) =", dist(v1, v2))


def get_distances(data_dim, p_metric):
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
        "mahalanobis": mahalanobis,
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
        "weighted_minkowski": weighted_minkowski,
        "yule": yule,
    }


v0 = np.array([9, 5, 6, 7, 3, 2, 1, 0, 8, -4], dtype=np.float32)
v1 = np.array([6, 8, -2, 3, 6, 5, 4, -9, 1, 0], dtype=np.float32)
v2 = np.array([-1, 3, 5, 1, 0, 0, -7, 6, 5, 0], dtype=np.float32)

N = 100
w0 = np.arange(0, N, dtype=np.float32)
w1 = np.arange(-10, N - 10, dtype=np.float32)
w2 = np.arange(5, N + 5, dtype=np.float32)

x0 = np.array([9, 5], dtype=np.float32)
x1 = np.array([6, 8], dtype=np.float32)
x2 = np.array([-1, 3], dtype=np.float32)

y0 = np.array([3, -4], dtype=np.float32)
y1 = np.array([-8, 8], dtype=np.float32)
y2 = np.array([9, 4], dtype=np.float32)

pv0 = np.abs(v0) / sum(np.abs(v0))
pv1 = np.abs(v1) / sum(np.abs(v1))
pv2 = np.abs(v2) / sum(np.abs(v2))

pw0 = np.abs(w0) / sum(np.abs(w0))
pw1 = np.abs(w1) / sum(np.abs(w1))
pw2 = np.abs(w2) / sum(np.abs(w2))

print("# Test distances for:")

print("v0 =", v0, "size =", len(v0))
print("v1 =", v1, "size =", len(v1))
print("v2 =", v2, "size =", len(v2))

print("w0 =", w0, "size =", len(w0))
print("w1 =", w1, "size =", len(w1))
print("w2 =", w2, "size =", len(w2))

print("x0 =", x0, "size =", len(x0))
print("x1 =", x1, "size =", len(x1))
print("x2 =", x2, "size =", len(x2))

print("y0 =", y0, "size =", len(y0))
print("y1 =", y1, "size =", len(y1))
print("y2 =", y2, "size =", len(y2))

print("pv0 =", pv0, "size =", len(pv0))
print("pv1 =", pv1, "size =", len(pv1))
print("pv2 =", pv2, "size =", len(pv2))

print("pw0 =", pw0, "size =", len(pw0))
print("pw1 =", pw1, "size =", len(pw1))
print("pw2 =", pw2, "size =", len(pw2))

print()

metrics_v = get_distances(len(v0), 2)
metrics_w = get_distances(len(w0), 2)

for name, dense in metrics_v.items():
    if name == "haversine":
        test_distance(name, dense, "x", x0, x1, x2)
        test_distance(name, dense, "y", y0, y1, y2)
        print()
        continue
    if name in ["hellinger", "jensen_shannon", "symmetric_kl"]:
        test_distance(name, dense, "p", pv0, pv1, pv2)
        test_distance(name, dense, "p", pw0, pw1, pw2)
        print()
        continue
    test_distance(name, dense, "v", v0, v1, v2)
    test_distance(name, dense, "w", w0, w1, w2)
    print()
