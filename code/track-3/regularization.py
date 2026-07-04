"""
Track 3 · Lesson 2 — Regularization (L1/L2)
Run:  python regularization.py
L2 (ridge) shrinks all weights smoothly; L1 (lasso) drives some to EXACTLY zero
(feature selection). Ridge has a closed form; lasso uses soft-thresholding.
"""
import numpy as np


def ridge(X, y, lam):
    """L2 closed form: w = (X^T X + lam I)^{-1} X^T y."""
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam*np.eye(d), X.T @ y)


def soft_threshold(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def lasso(X, y, lam, lr=0.01, steps=4000):
    """L1 via proximal gradient: gradient step on MSE, then shrink toward 0."""
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        grad = X.T @ (X @ w - y) / len(y)
        w = soft_threshold(w - lr*grad, lr*lam)    # the shrink-toward-zero step
    return w


def main():
    rng = np.random.default_rng(0)
    n, d = 60, 5
    X = rng.normal(size=(n, d))
    true_w = np.array([3.0, -2.0, 0.0, 0.0, 0.0])   # only features 0,1 matter
    y = X @ true_w + rng.normal(0, 0.5, n)

    print("true weights :", true_w)
    print("no reg (lam=0):", np.round(ridge(X, y, 0.0), 2))
    print("ridge  (L2)  :", np.round(ridge(X, y, 5.0), 2))    # all small, none exactly 0
    print("lasso  (L1)  :", np.round(lasso(X, y, 0.3), 2))    # noise features -> 0
    print("lasso strong :", np.round(lasso(X, y, 1.0), 2))    # over-shrinks even true ones

    # --- Try it yourself -----------------------------------------------------
    # 1) soft_threshold(0.2, 0.3) -> 0.0 ; soft_threshold(1.0, 0.3) -> 0.7


if __name__ == "__main__":
    main()
