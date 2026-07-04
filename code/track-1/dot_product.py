"""
Track 1 · Lesson 3 — The dot product: length, angle, projection
Run:  python dot_product.py
Two vectors in, one number out. That number secretly encodes length, angle,
and similarity -- the most important computation in machine learning.
"""
import numpy as np


def dot_from_scratch(a, b):
    """The definition, by hand: sum of products of matching components."""
    assert len(a) == len(b), "vectors must have the same dimension"
    total = 0.0
    for i in range(len(a)):       # i = 0..n-1  (code indexing)
        total += a[i] * b[i]
    return total


def norm(v):
    return np.sqrt(v @ v)


def angle_deg(a, b):
    cos_t = (a @ b) / (norm(a) * norm(b))
    cos_t = np.clip(cos_t, -1.0, 1.0)     # guard tiny float overshoot past +-1
    return np.degrees(np.arccos(cos_t))


def cosine_sim(a, b):
    return (a @ b) / (norm(a) * norm(b))


def main():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])

    print("dot (by hand) =", dot_from_scratch(a, b))   # 32.0
    print("dot (sum a*b) =", np.sum(a * b))            # 32.0
    print("dot (a @ b)   =", a @ b)                    # 32.0

    print("norm (4,3)    =", norm(np.array([4.0, 3.0])))            # 5.0
    print("angle e1,e2   =", angle_deg(np.array([1, 0]), np.array([0, 1])))   # 90.0
    print("angle (1,0),(1,1) =", angle_deg(np.array([1, 0]), np.array([1, 1])))  # 45.0
    print("cos_sim same dir  =", cosine_sim(np.array([1, 1]), np.array([2, 2])))  # 1.0

    # A tiny "vector search": find the item most similar to a query
    query = np.array([1.0, 1.0])
    items = [np.array([2.0, 0.0]), np.array([1.0, 2.0]), np.array([-1.0, -1.0])]
    best = max(items, key=lambda x: cosine_sim(query, x))
    print("most similar to (1,1):", best)              # [1. 2.]

    # --- Try it yourself -----------------------------------------------------
    # 1) For a=(3,4), b=(4,3): compute a.b, |a|, |b|, and the angle. (~16.3 deg)
    # 2) Remove the np.clip in angle_deg and feed two identical vectors.
    #    You may get a RuntimeWarning / nan. That's the bug clip prevents.


if __name__ == "__main__":
    main()
