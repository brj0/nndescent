"""
Test distance functions of pynndescent. The values are the same as in the
corresponding cpp file.
"""


import numpy as np
from pynndescent.distances import *


v0 = np.array([9,5,6,7,3,2,1,0,8,-4], dtype=np.float32)
v1 = np.array([6,8,-2,3,6,5,4,-9,1,0], dtype=np.float32)
v2 = np.array([-1,3,5,1,0,0,-7,6,5,0], dtype=np.float32)

N = 100
w0 = np.arange(0, N, dtype=np.float32)
w1 = np.arange(-10, N - 10, dtype=np.float32)
w2 = np.arange(5, N + 5, dtype=np.float32)

x0 = np.array([9,5], dtype=np.float32)
x1 = np.array([6,8], dtype=np.float32)
x2 = np.array([-1,3], dtype=np.float32)

p0 = np.array([0.9,0.1,0.0], dtype=np.float32)
p1 = np.array([0.5,0.2,0.3], dtype=np.float32)
p2 = np.array([0,0.7,0.3], dtype=np.float32)

print("v0 =", v0, "size =", len(v0))
print("v1 =", v1, "size =", len(v1))
print("v2 =", v2, "size =", len(v2))

print("w0 =", w0, "size =", len(w0))
print("w1 =", w1, "size =", len(w1))
print("w2 =", w2, "size =", len(w2))

print("x0 =", x0, "size =", len(x0))
print("x1 =", x1, "size =", len(x1))
print("x2 =", x2, "size =", len(x2))

print("p0 =", p0, "size =", len(p0))
print("p1 =", p1, "size =", len(p1))
print("p2 =", p2, "size =", len(p2))

print("\nTest distances for:")

print("alternative_cosine(v0, v1) =", alternative_cosine(v0, v1))
print("alternative_cosine(v0, v2) =", alternative_cosine(v0, v2))
print("alternative_cosine(v1, v2) =", alternative_cosine(v1, v2))

print("alternative_cosine(w0, w1) =", alternative_cosine(w0, w1))
print("alternative_cosine(w0, w2) =", alternative_cosine(w0, w2))
print("alternative_cosine(w1, w2) =", alternative_cosine(w1, w2))

print()

print("alternative_dot(v0, v1) =", alternative_dot(v0, v1))
print("alternative_dot(v0, v2) =", alternative_dot(v0, v2))
print("alternative_dot(v1, v2) =", alternative_dot(v1, v2))

print("alternative_dot(w0, w1) =", alternative_dot(w0, w1))
print("alternative_dot(w0, w2) =", alternative_dot(w0, w2))
print("alternative_dot(w1, w2) =", alternative_dot(w1, w2))

print()

print("alternative_jaccard(v0, v1) =", alternative_jaccard(v0, v1))
print("alternative_jaccard(v0, v2) =", alternative_jaccard(v0, v2))
print("alternative_jaccard(v1, v2) =", alternative_jaccard(v1, v2))

print("alternative_jaccard(w0, w1) =", alternative_jaccard(w0, w1))
print("alternative_jaccard(w0, w2) =", alternative_jaccard(w0, w2))
print("alternative_jaccard(w1, w2) =", alternative_jaccard(w1, w2))

print()

print("bray_curtis(v0, v1) =", bray_curtis(v0, v1))
print("bray_curtis(v0, v2) =", bray_curtis(v0, v2))
print("bray_curtis(v1, v2) =", bray_curtis(v1, v2))

print("bray_curtis(w0, w1) =", bray_curtis(w0, w1))
print("bray_curtis(w0, w2) =", bray_curtis(w0, w2))
print("bray_curtis(w1, w2) =", bray_curtis(w1, w2))

print()

print("canberra(v0, v1) =", canberra(v0, v1))
print("canberra(v0, v2) =", canberra(v0, v2))
print("canberra(v1, v2) =", canberra(v1, v2))

print("canberra(w0, w1) =", canberra(w0, w1))
print("canberra(w0, w2) =", canberra(w0, w2))
print("canberra(w1, w2) =", canberra(w1, w2))

print()

print("chebyshev(v0, v1) =", chebyshev(v0, v1))
print("chebyshev(v0, v2) =", chebyshev(v0, v2))
print("chebyshev(v1, v2) =", chebyshev(v1, v2))

print("chebyshev(w0, w1) =", chebyshev(w0, w1))
print("chebyshev(w0, w2) =", chebyshev(w0, w2))
print("chebyshev(w1, w2) =", chebyshev(w1, w2))

print()

print("correlation(v0, v1) =", correlation(v0, v1))
print("correlation(v0, v2) =", correlation(v0, v2))
print("correlation(v1, v2) =", correlation(v1, v2))

print("correlation(w0, w1) =", correlation(w0, w1))
print("correlation(w0, w2) =", correlation(w0, w2))
print("correlation(w1, w2) =", correlation(w1, w2))

print()

print("cosine(v0, v1) =", cosine(v0, v1))
print("cosine(v0, v2) =", cosine(v0, v2))
print("cosine(v1, v2) =", cosine(v1, v2))

print("cosine(w0, w1) =", cosine(w0, w1))
print("cosine(w0, w2) =", cosine(w0, w2))
print("cosine(w1, w2) =", cosine(w1, w2))

print()

print("dice(v0, v1) =", dice(v0, v1))
print("dice(v0, v2) =", dice(v0, v2))
print("dice(v1, v2) =", dice(v1, v2))

print("dice(w0, w1) =", dice(w0, w1))
print("dice(w0, w2) =", dice(w0, w2))
print("dice(w1, w2) =", dice(w1, w2))

print()

print("dot(v0, v1) =", dot(v0, v1))
print("dot(v0, v2) =", dot(v0, v2))
print("dot(v1, v2) =", dot(v1, v2))

print("dot(w0, w1) =", dot(w0, w1))
print("dot(w0, w2) =", dot(w0, w2))
print("dot(w1, w2) =", dot(w1, w2))

print()

print("euclidean(v0, v1) =", euclidean(v0, v1))
print("euclidean(v0, v2) =", euclidean(v0, v2))
print("euclidean(v1, v2) =", euclidean(v1, v2))

print("euclidean(w0, w1) =", euclidean(w0, w1))
print("euclidean(w0, w2) =", euclidean(w0, w2))
print("euclidean(w1, w2) =", euclidean(w1, w2))

print()

print("hamming(v0, v1) =", hamming(v0, v1))
print("hamming(v0, v2) =", hamming(v0, v2))
print("hamming(v1, v2) =", hamming(v1, v2))

print("hamming(w0, w1) =", hamming(w0, w1))
print("hamming(w0, w2) =", hamming(w0, w2))
print("hamming(w1, w2) =", hamming(w1, w2))

print()

print("hellinger(p0, p1) =", hellinger(p0, p1))
print("hellinger(p0, p2) =", hellinger(p0, p2))
print("hellinger(p1, p2) =", hellinger(p1, p2))

print()

print("haversine(x0, x1) =", haversine(x0, x1))
print("haversine(x0, x2) =", haversine(x0, x2))
print("haversine(x1, x2) =", haversine(x1, x2))

print()

print("jaccard(v0, v1) =", jaccard(v0, v1))
print("jaccard(v0, v2) =", jaccard(v0, v2))
print("jaccard(v1, v2) =", jaccard(v1, v2))

print("jaccard(w0, w1) =", jaccard(w0, w1))
print("jaccard(w0, w2) =", jaccard(w0, w2))
print("jaccard(w1, w2) =", jaccard(w1, w2))

print()

print("kulsinski(v0, v1) =", kulsinski(v0, v1))
print("kulsinski(v0, v2) =", kulsinski(v0, v2))
print("kulsinski(v1, v2) =", kulsinski(v1, v2))

print("kulsinski(w0, w1) =", kulsinski(w0, w1))
print("kulsinski(w0, w2) =", kulsinski(w0, w2))
print("kulsinski(w1, w2) =", kulsinski(w1, w2))

print()

print("manhattan(v0, v1) =", manhattan(v0, v1))
print("manhattan(v0, v2) =", manhattan(v0, v2))
print("manhattan(v1, v2) =", manhattan(v1, v2))

print("manhattan(w0, w1) =", manhattan(w0, w1))
print("manhattan(w0, w2) =", manhattan(w0, w2))
print("manhattan(w1, w2) =", manhattan(w1, w2))

print()

print("matching(v0, v1) =", matching(v0, v1))
print("matching(v0, v2) =", matching(v0, v2))
print("matching(v1, v2) =", matching(v1, v2))

print("matching(w0, w1) =", matching(w0, w1))
print("matching(w0, w2) =", matching(w0, w2))
print("matching(w1, w2) =", matching(w1, w2))

print()

print("russellrao(v0, v1) =", russellrao(v0, v1))
print("russellrao(v0, v2) =", russellrao(v0, v2))
print("russellrao(v1, v2) =", russellrao(v1, v2))

print("russellrao(w0, w1) =", russellrao(w0, w1))
print("russellrao(w0, w2) =", russellrao(w0, w2))
print("russellrao(w1, w2) =", russellrao(w1, w2))

print()

print("rogers_tanimoto(v0, v1) =", rogers_tanimoto(v0, v1))
print("rogers_tanimoto(v0, v2) =", rogers_tanimoto(v0, v2))
print("rogers_tanimoto(v1, v2) =", rogers_tanimoto(v1, v2))

print("rogers_tanimoto(w0, w1) =", rogers_tanimoto(w0, w1))
print("rogers_tanimoto(w0, w2) =", rogers_tanimoto(w0, w2))
print("rogers_tanimoto(w1, w2) =", rogers_tanimoto(w1, w2))

print()

print("sokal_michener(v0, v1) =", sokal_michener(v0, v1))
print("sokal_michener(v0, v2) =", sokal_michener(v0, v2))
print("sokal_michener(v1, v2) =", sokal_michener(v1, v2))

print("sokal_michener(w0, w1) =", sokal_michener(w0, w1))
print("sokal_michener(w0, w2) =", sokal_michener(w0, w2))
print("sokal_michener(w1, w2) =", sokal_michener(w1, w2))

print()

print("sokal_sneath(v0, v1) =", sokal_sneath(v0, v1))
print("sokal_sneath(v0, v2) =", sokal_sneath(v0, v2))
print("sokal_sneath(v1, v2) =", sokal_sneath(v1, v2))

print("sokal_sneath(w0, w1) =", sokal_sneath(w0, w1))
print("sokal_sneath(w0, w2) =", sokal_sneath(w0, w2))
print("sokal_sneath(w1, w2) =", sokal_sneath(w1, w2))

print()

print("squared_euclidean(v0, v1) =", squared_euclidean(v0, v1))
print("squared_euclidean(v0, v2) =", squared_euclidean(v0, v2))
print("squared_euclidean(v1, v2) =", squared_euclidean(v1, v2))

print("squared_euclidean(w0, w1) =", squared_euclidean(w0, w1))
print("squared_euclidean(w0, w2) =", squared_euclidean(w0, w2))
print("squared_euclidean(w1, w2) =", squared_euclidean(w1, w2))

print()

print("yule(v0, v1) =", yule(v0, v1))
print("yule(v0, v2) =", yule(v0, v2))
print("yule(v1, v2) =", yule(v1, v2))

print("yule(w0, w1) =", yule(w0, w1))
print("yule(w0, w2) =", yule(w0, w2))
print("yule(w1, w2) =", yule(w1, w2))

print()

