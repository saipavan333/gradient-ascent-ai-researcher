"""
Track 1 · Lesson 5 — Matrices as linear transformations
Run:  python matrix_vector.py
A matrix is a function that moves vectors. The matrix-vector product can be
read two ways (rows = dot products, columns = weighted sum) -- both agree.
"""
import numpy as np


def matvec_rows(A, x):
    """Rows view: output[i] = dot(row i of A, x)."""
    m, n = A.shape
    out = np.zeros(m)
    for i in range(m):
        s = 0.0
        for j in range(n):
            s += A[i, j] * x[j]      # row i dotted with x
        out[i] = s
    return out


def matvec_cols(A, x):
    """Columns view: output = sum_j x[j] * (column j of A)."""
    m, n = A.shape
    out = np.zeros(m)
    for j in range(n):
        out += x[j] * A[:, j]        # weighted sum of columns
    return out


def main():
    A = np.array([[2.0, 1.0],
                  [0.0, 1.5]])
    x = np.array([3.0, 2.0])

    print("rows view :", matvec_rows(A, x))   # [8. 3.]
    print("cols view :", matvec_cols(A, x))   # [8. 3.]
    print("A @ x     :", A @ x)               # [8. 3.]

    # "Columns are where the basis lands":
    print("A e1 =", A @ np.array([1.0, 0.0]))  # [2. 0.] == first column
    print("A e2 =", A @ np.array([0.0, 1.0]))  # [1. 1.5] == second column

    # A rotation by 90 degrees, applied to e1
    R = np.array([[0.0, -1.0],
                  [1.0,  0.0]])
    print("rotate e1 90deg:", R @ np.array([1.0, 0.0]))   # [0. 1.] -> e1 -> e2

    # Reflect across the x-axis sends (3,2) -> (3,-2)
    Refl = np.array([[1.0, 0.0],
                     [0.0, -1.0]])
    print("reflect (3,2):", Refl @ np.array([3.0, 2.0]))  # [3. -2.]

    # --- Try it yourself -----------------------------------------------------
    # 1) Build A = [[1,2],[3,4]], x=(5,6). Predict A@x, then check (=(17,39)).
    # 2) Apply R three times to (1,0). Predict each step before running.


if __name__ == "__main__":
    main()
