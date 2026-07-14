"""
Track 5 (Classical ML) · Linear & Logistic Regression from scratch
==================================================================
Linear regression: fit the best straight line (or plane) through data by
minimizing mean-squared error. It's the "hello world" of machine learning and
the ancestor of every model in this course — a single layer with no activation,
trained by gradient descent.

Run:  python linear_regression.py
"""
import numpy as np

rng = np.random.default_rng(0)


def make_data(n=120, true_w=2.5, true_b=-1.0, noise=1.2):
    X = rng.uniform(-3, 3, size=(n, 1))
    y = true_w * X[:, 0] + true_b + noise * rng.standard_normal(n)
    return X, y


def fit_gd(X, y, lr=0.05, steps=400):
    """Fit y ≈ wX + b by gradient descent on mean-squared error."""
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(steps):
        pred = X @ w + b
        err = pred - y                      # residual
        # gradients of (1/n) Σ err² :
        gw = (2 / n) * X.T @ err
        gb = (2 / n) * err.sum()
        w -= lr * gw; b -= lr * gb
    return w, b


def normal_equation(X, y):
    """Closed-form optimum: w = (XᵀX)⁻¹ Xᵀy (with a bias column)."""
    Xb = np.c_[np.ones(len(X)), X]          # prepend a column of 1s for the bias
    theta = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)
    return theta[1:], theta[0]              # w, b


def r2(y, pred):
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


if __name__ == "__main__":
    X, y = make_data()
    w_gd, b_gd = fit_gd(X, y)
    w_ne, b_ne = normal_equation(X, y)
    print(f"true:            w=2.500  b=-1.000")
    print(f"gradient descent w={w_gd[0]:.3f}  b={b_gd:.3f}")
    print(f"normal equation  w={w_ne[0]:.3f}  b={b_ne:.3f}")
    print(f"R^2 (fit quality, 1.0 = perfect): {r2(y, X @ w_gd + b_gd):.3f}")
    # the two methods should agree, and recover the true slope/intercept closely
    assert abs(w_gd[0] - w_ne[0]) < 0.05 and abs(b_gd - b_ne) < 0.05
    assert abs(w_gd[0] - 2.5) < 0.4
    print("OK — gradient descent and the closed-form solution agree and recover the line.")
