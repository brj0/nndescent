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

print("v0 =", v0, "size =", len(v0))
print("v1 =", v1, "size =", len(v1))
print("v2 =", v2, "size =", len(v2))

print("w0 =", w0, "size =", len(w0))
print("w1 =", w1, "size =", len(w1))
print("w2 =", w2, "size =", len(w2))


print("\nTest distances for:")

print("euclidean(v0, v1) =", euclidean(v0, v1))
print("euclidean(v0, v2) =", euclidean(v0, v2))
print("euclidean(v1, v2) =", euclidean(v1, v2))

print()

print("euclidean(w0, w1) =", euclidean(w0, w1))
print("euclidean(w0, w2) =", euclidean(w0, w2))
print("euclidean(w1, w2) =", euclidean(w1, w2))

print()
print()

print("sqeuclidean(v0, v1) =", squared_euclidean(v0, v1))
print("sqeuclidean(v0, v2) =", squared_euclidean(v0, v2))
print("sqeuclidean(v1, v2) =", squared_euclidean(v1, v2))

print()

print("sqeuclidean(w0, w1) =", squared_euclidean(w0, w1))
print("sqeuclidean(w0, w2) =", squared_euclidean(w0, w2))
print("sqeuclidean(w1, w2) =", squared_euclidean(w1, w2))

print()
print()

print("dot(v0, v1) =", dot(v0, v1))
print("dot(v0, v2) =", dot(v0, v2))
print("dot(v1, v2) =", dot(v1, v2))

print()

print("dot(w0, w1) =", dot(w0, w1))
print("dot(w0, w2) =", dot(w0, w2))
print("dot(w1, w2) =", dot(w1, w2))

print()
print()

print("alternative_dot(v0, v1) =", alternative_dot(v0, v1))
print("alternative_dot(v0, v2) =", alternative_dot(v0, v2))
print("alternative_dot(v1, v2) =", alternative_dot(v1, v2))

print()

print("alternative_dot(w0, w1) =", alternative_dot(w0, w1))
print("alternative_dot(w0, w2) =", alternative_dot(w0, w2))
print("alternative_dot(w1, w2) =", alternative_dot(w1, w2))

print()
print()

print("cosine(v0, v1) =", cosine(v0, v1))
print("cosine(v0, v2) =", cosine(v0, v2))
print("cosine(v1, v2) =", cosine(v1, v2))

print()

print("cosine(w0, w1) =", cosine(w0, w1))
print("cosine(w0, w2) =", cosine(w0, w2))
print("cosine(w1, w2) =", cosine(w1, w2))

print()
print()

print("hamming(v0, v1) =", hamming(v0, v1))
print("hamming(v0, v2) =", hamming(v0, v2))
print("hamming(v1, v2) =", hamming(v1, v2))

print()

print("hamming(w0, w1) =", hamming(w0, w1))
print("hamming(w0, w2) =", hamming(w0, w2))
print("hamming(w1, w2) =", hamming(w1, w2))

print()
print()

print("jaccard(v0, v1) =", jaccard(v0, v1))
print("jaccard(v0, v2) =", jaccard(v0, v2))
print("jaccard(v1, v2) =", jaccard(v1, v2))

print()

print("jaccard(w0, w1) =", jaccard(w0, w1))
print("jaccard(w0, w2) =", jaccard(w0, w2))
print("jaccard(w1, w2) =", jaccard(w1, w2))

print()
print()
