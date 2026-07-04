"""
Track 8 · Lesson 1 — Reproducing "double descent" from scratch
Run:  python reproduce_double_descent.py

A faithful, small reproduction of one of the most surprising modern ML results
(Belkin et al. 2019; Nakkiran et al. 2019): as a model grows past the point where
it can exactly fit the training data (the "interpolation threshold", #params = #data),
TEST error first SPIKES, then DESCENDS AGAIN — contradicting the classic U-shaped
bias-variance story. We reproduce it with random ReLU features + least squares.
"""
import numpy as np
rng = np.random.default_rng(0)

def make_data(n, d=8, noise=0.5):
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    y = np.tanh(X @ w_true) + noise * rng.normal(size=n)   # nonlinear target + noise
    return X, y, w_true

def random_features(X, W, b):
    return np.maximum(0.0, X @ W + b)                      # ReLU random features

def fit_predict(Xtr, ytr, Xte, yte, P, d, ridge=1e-6):
    """P random ReLU features, min-norm / ridge least squares, return test MSE."""
    W = rng.normal(size=(d, P)) / np.sqrt(d); b = rng.normal(size=P)
    Ftr, Fte = random_features(Xtr, W, b), random_features(Xte, W, b)
    # min-norm solution via ridge (tiny lambda) — stable pseudo-inverse
    A = Ftr.T @ Ftr + ridge * np.eye(P)
    w = np.linalg.solve(A, Ftr.T @ ytr)
    return float(np.mean((Fte @ w - yte)**2))

if __name__ == "__main__":
    n = 40                                                  # training points
    Xtr, ytr, wt = make_data(n)
    Xte = rng.normal(size=(500, 8)); yte = np.tanh(Xte @ wt)  # clean test target
    Ps = [2,4,8,12,16,20,28,34,38,40,42,46,52,64,80,110,160,240,360,520]
    print(f"{'P (features)':>12} {'P/n':>6} {'test MSE':>10}")
    curve = []
    for P in Ps:
        errs = [fit_predict(Xtr, ytr, Xte, yte, P, 8) for _ in range(20)]   # avg over feature draws
        m = float(np.mean(errs)); curve.append((P, m))
        print(f"{P:>12} {P/n:>6.2f} {m:>10.3f}")
    peak = max(curve, key=lambda t: t[1])
    print(f"\nTest error peaks near P/n = {peak[0]/n:.2f} (the interpolation threshold),")
    print("then DESCENDS again for larger models — classic double descent.")
