"""
The double-descent experiment, reproducible and parameterized.
Random ReLU features + (ridge) least squares; sweep model size across P = n.
This is the result from Track 8 of the AI Researcher Path, packaged for serving.
"""
import numpy as np

def make_data(n=40, d=8, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)); w = rng.normal(size=d)
    y = np.tanh(X @ w) + noise * rng.normal(size=n)
    return X, y, w

def _test_error(Xtr, ytr, Xte, yte, P, d, ridge, rng):
    W = rng.normal(size=(d, P)) / np.sqrt(d); b = rng.normal(size=P)
    Ftr = np.maximum(0.0, Xtr @ W + b); Fte = np.maximum(0.0, Xte @ W + b)
    A = Ftr.T @ Ftr + ridge * np.eye(P)
    w = np.linalg.solve(A, Ftr.T @ ytr)
    return float(np.mean((Fte @ w - yte) ** 2))

def double_descent_curve(n=40, noise=0.5, ridge=1e-6, d=8, repeats=20, seed=0):
    """Return (ratios P/n, mean test errors) tracing the double-descent curve."""
    rng = np.random.default_rng(seed)
    Xtr, ytr, w = make_data(n, d, noise, seed)
    Xte = rng.normal(size=(500, d)); yte = np.tanh(Xte @ w)
    Ps = [2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 38, 40, 42, 44, 48, 52, 60,
          72, 90, 120, 160, 220, 320, 480]
    ratios, errs = [], []
    for P in Ps:
        e = np.mean([_test_error(Xtr, ytr, Xte, yte, P, d, ridge, rng) for _ in range(repeats)])
        ratios.append(P / n); errs.append(float(e))
    return ratios, errs
