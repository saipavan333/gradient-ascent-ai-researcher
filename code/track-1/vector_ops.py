"""
Track 1 · Lesson 2 — Vector operations & geometry
Run:  python vector_ops.py
There are only two fundamental operations on vectors: addition and scalar
multiplication. Subtraction and linear combinations are built from them.
"""
import numpy as np


def main():
    u = np.array([3, 1])
    v = np.array([1, 2])

    print("u + v   =", u + v)     # [4 3]   addition: component-wise
    print("u - v   =", u - v)     # [2 -1]  subtraction: points from v to u
    print("2 * v   =", 2 * v)     # [2 4]   scalar multiplication: stretch
    print("-1 * v  =", -1 * v)    # [-1 -2] flip direction
    print("0.5 * u =", 0.5 * u)   # [1.5 0.5] shrink

    # A linear combination  c*u + d*v
    c, d = 2.0, -1.0
    print("2u - v  =", c * u + d * v)   # [5. 0.]

    # The mean (center of mass) of several vectors
    X = np.array([[3, 1],
                  [1, 2],
                  [2, 6]])        # three vectors, one per row
    mean = X.mean(axis=0)         # average down the columns
    print("mean    =", mean)      # [2. 3.]  == (1/3)*((3,1)+(1,2)+(2,6))

    # Verify a distributive law numerically: c(u+v) == c*u + c*v
    print("distributive holds:", np.array_equal(c * (u + v), c * u + c * v))  # True

    # The training-loop preview: w <- w - alpha * g  (scale, then subtract)
    w = np.array([0.5, -0.2])
    g = np.array([0.1, 0.4])      # a "gradient" direction
    alpha = 0.1                   # step size (a scalar)
    w_new = w - alpha * g
    print("param update:", w, "->", w_new)

    # --- Try it yourself -----------------------------------------------------
    # 1) A robot at p=(5,5) wants target t=(8,1). What vector takes it there?
    #    (Hint: t - p.) Predict, then print it.


if __name__ == "__main__":
    main()
