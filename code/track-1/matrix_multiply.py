"""
Track 1 · Lesson 6 — Matrix multiplication = composition
Run:  python matrix_multiply.py
AB means "do B first, then A". The row-times-column rule is FORCED by the
requirement (AB)x = A(Bx). Here we derive-by-code and confirm the key facts.
"""
import numpy as np


def matmul_from_scratch(A, B):
    """C[i,j] = sum_k A[i,k] * B[k,j]  -- the definition, by hand."""
    m, k = A.shape
    k2, n = B.shape
    assert k == k2, f"shape mismatch: {A.shape} x {B.shape} (inner dims must match)"
    C = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            s = 0.0
            for kk in range(k):
                s += A[i, kk] * B[kk, j]     # row i of A, column j of B
            C[i, j] = s
    return C


def main():
    A = np.array([[1.0, 2.0],
                  [3.0, 4.0]])
    B = np.array([[5.0, 6.0],
                  [7.0, 8.0]])

    print("from scratch:\n", matmul_from_scratch(A, B))  # [[19 22][43 50]]
    print("A @ B:\n", A @ B)                              # matches

    # Composition law:  (AB)x == A(Bx)
    x = np.array([1.0, -1.0])
    print("(AB)x == A(Bx):", np.allclose((A @ B) @ x, A @ (B @ x)))   # True

    # Non-commutativity: AB != BA in general
    print("AB == BA:", np.allclose(A @ B, B @ A))         # False
    print("BA:\n", B @ A)                                 # [[23 34][31 46]]

    # Associativity and the identity matrix
    C = np.random.randn(2, 2)
    print("assoc A(BC)=(AB)C:", np.allclose(A @ (B @ C), (A @ B) @ C))  # True
    I = np.eye(2)
    print("A I == A:", np.allclose(A @ I, A))             # True

    # Stacking two LINEAR layers collapses to one matrix (why nonlinearities exist)
    W1 = np.random.randn(4, 3)
    W2 = np.random.randn(2, 4)
    xx = np.random.randn(3)
    print("W2(W1 x) == (W2 W1) x:",
          np.allclose(W2 @ (W1 @ xx), (W2 @ W1) @ xx))    # True

    # --- Try it yourself -----------------------------------------------------
    # 1) Confirm two rotations (30deg then 60deg) commute (both give 90deg).
    # 2) Time matmul_from_scratch vs A@B on 200x200 matrices. Feel the difference.


if __name__ == "__main__":
    main()
