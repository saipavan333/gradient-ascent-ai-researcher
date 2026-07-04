"""
Track 1 · Lesson 1 — Vectors: the atoms of ML
Run:  python vectors.py
A vector is just an ordered list of numbers. This script shows the three things
you most need: creating vectors, reading components, and testing equality.
"""
import numpy as np


def main():
    # A vector is just an ordered list of numbers.
    v = np.array([3, 2])          # the vector v = (3, 2)
    print("v        =", v)        # [3 2]
    print("v.shape  =", v.shape)  # (2,)   -> "2 components, i.e. v is in R^2"

    # Components. NOTE: math counts from 1 (v_1, v_2); Python counts from 0.
    print("v[0], v[1] =", v[0], v[1])   # 3 2  -> v[0] is v_1, v[1] is v_2

    # Dimension = number of components
    print("dimension:", v.shape[0])      # 2

    # A 3-D vector and the zero vector in R^3
    w = np.array([2, 3, 2])
    zero3 = np.zeros(3)
    print("w =", w, " zero3 =", zero3)   # [2 3 2] [0. 0. 0.]

    # Vector EQUALITY: same dimension AND every component matches
    a = np.array([3, 2])
    b = np.array([3, 2])
    c = np.array([2, 3])
    print("a == b ?", np.array_equal(a, b))   # True
    print("a == c ?", np.array_equal(a, c))   # False -> order matters!

    # An ML feature vector: a house as a point in R^4
    house = np.array([1500, 3, 30, 450])      # size, bedrooms, age, price(k)
    print("house lives in R^%d" % house.shape[0])   # R^4

    # --- Try it yourself -----------------------------------------------------
    # 1) Change v to a 4-component vector. Predict v.shape BEFORE running.
    # 2) Print the math 3rd component of `house` (careful: that's index 2).


if __name__ == "__main__":
    main()
