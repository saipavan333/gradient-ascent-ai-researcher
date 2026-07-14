"""
Track 13 · Lesson 4 — Active learning: label the most informative points
========================================================================
Labels are expensive. Active learning lets the MODEL choose which points to label
next — picking the ones it is most UNSURE about (nearest the decision boundary),
because those pin the boundary down fastest. We compare it to labelling random
points, averaged over many runs, and measure how fast test accuracy climbs as the
label budget grows (area under that curve — higher means it learns from fewer labels).

Run:  python active_learning.py
"""
import numpy as np
rng = np.random.default_rng(4)

def data(n, seed):
    r = np.random.default_rng(seed)
    # discriminative in x (classes at x=-2 / x=+2), lots of irrelevant spread in y
    Xa = np.c_[r.normal(-2.0, 0.9, n // 2), r.normal(0, 3.0, n // 2)]
    Xb = np.c_[r.normal(+2.0, 0.9, n // 2), r.normal(0, 3.0, n // 2)]
    return np.vstack([Xa, Xb]), np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)

def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -30, 30)))
def fit(X, y, steps=250, lr=0.2):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(steps):
        e = sigmoid(X @ w + b) - y
        w -= lr * X.T @ e / len(y); b -= lr * e.mean()
    return w, b
def acc(X, y, w, b): return ((sigmoid(X @ w + b) > 0.5).astype(int) == y).mean()

def run(strategy, seed, budget=24, seed0=4):
    X, y = data(400, seed); Xte, yte = data(1000, seed + 999)
    idx = list(np.random.default_rng(seed).choice(len(X), seed0, replace=False))
    pool = [i for i in range(len(X)) if i not in idx]
    curve = []
    while len(idx) < budget:
        w, b = fit(X[idx], y[idx])
        if strategy == "active":
            unc = -np.abs(sigmoid(X[pool] @ w + b) - 0.5)     # closest to 0.5 = most uncertain
            j = int(np.argmax(unc))
        else:
            j = int(np.random.default_rng(seed * 7 + len(idx)).integers(len(pool)))
        idx.append(pool.pop(j))
        curve.append(acc(Xte, yte, *fit(X[idx], y[idx])))
    return np.array(curve)

if __name__ == "__main__":
    seeds = range(12)
    A = np.mean([run("active", s) for s in seeds], 0)
    R = np.mean([run("random", s) for s in seeds], 0)
    print(f"test accuracy after ~12 labels — active: {A[8]:.3f}, random: {R[8]:.3f}")
    print(f"learning speed (area under accuracy-vs-labels curve) — active: {A.mean():.3f}, random: {R.mean():.3f}")
    assert A.mean() >= R.mean()
    print("OK — choosing informative points reaches high accuracy with fewer labels.")
