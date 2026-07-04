"""
Track 1 · Lesson 8 — Eigenvectors & eigenvalues
Run:  python eigen.py
A v = lambda v: eigenvectors keep their direction under A; lambda is the stretch.
Eigenvectors of a covariance matrix are the principal components (PCA).
"""
import numpy as np


def power_iteration(A, steps=100):
    """Find the dominant eigenvector by repeated multiplication."""
    v = np.random.default_rng(0).standard_normal(A.shape[0])
    for _ in range(steps):
        v = A @ v
        v = v / np.linalg.norm(v)        # renormalize so it doesn't blow up
    lam = (v @ A @ v) / (v @ v)          # Rayleigh quotient -> the eigenvalue
    return lam, v


def main():
    A = np.array([[2.0, 1.0],
                  [1.0, 2.0]])           # eigenvalues 3 and 1

    vals, vecs = np.linalg.eig(A)
    print("eigenvalues:", np.round(np.sort(vals), 3))     # [1. 3.]

    # Verify A v = lambda v for each pair
    for i in range(len(vals)):
        v, lam = vecs[:, i], vals[i]
        assert np.allclose(A @ v, lam * v)
    print("A v = lambda v verified for all eigenpairs")

    # Power iteration finds the DOMINANT eigenpair
    lam, v = power_iteration(A)
    print("dominant lambda ~", round(lam, 4),             # ~3.0
          " vector ~", np.round(np.abs(v), 3))            # ~(0.707, 0.707)

    # A^k v = lambda^k v  (the fact behind power iteration)
    v1 = np.array([1.0, 1.0]) / np.sqrt(2)                # eigenvector for lambda=3
    print("A^5 v == 3^5 v:", np.allclose(np.linalg.matrix_power(A, 5) @ v1, 3**5 * v1))

    # A tiny PCA: top eigenvector of the covariance is the data's main axis
    rng = np.random.default_rng(1)
    X = rng.standard_normal((500, 2)) @ np.array([[3.0, 1.0], [1.0, 1.0]])
    Sigma = np.cov(X.T)                                   # 2x2 covariance
    evals, evecs = np.linalg.eigh(Sigma)                 # eigh: symmetric matrices
    top = evecs[:, np.argmax(evals)]
    print("main axis of data ~", np.round(top, 3))

    # --- Try it yourself -----------------------------------------------------
    # 1) Find eigenvalues of [[4,1],[2,3]] by hand (5 and 2), then check here.
    # 2) Make A a rotation by 90 deg; its eigenvalues are complex -- inspect them.


if __name__ == "__main__":
    main()
