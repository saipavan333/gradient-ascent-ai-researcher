"""
Track 13 · Lesson 3 — Semi-supervised learning: label propagation
=================================================================
Semi-supervised learning uses a FEW labels plus MANY unlabelled points. The idea:
points that are close together probably share a label, so we can spread ("propagate")
the few labels through the data's structure. Here we label just 6 seeds and let the
labels flow along a nearest-neighbour graph to classify all 300 points.

Run:  python semi_supervised.py
"""
import numpy as np
rng = np.random.default_rng(3)

def two_clusters(n=300):
    Xa = rng.normal([-2.2, 0], 0.75, (n // 2, 2))
    Xb = rng.normal([2.2, 0], 0.75, (n // 2, 2))
    X = np.vstack([Xa, Xb]); y = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)
    return X, y

def propagate(X, seed_idx, seed_lab, k=6, iters=50):
    n = len(X)
    d = ((X[:, None] - X[None]) ** 2).sum(2)
    nn = np.argsort(d, 1)[:, 1:k + 1]                 # k nearest neighbours
    lab = -np.ones(n, int)
    lab[seed_idx] = seed_lab
    scores = np.zeros((n, 2)); scores[seed_idx, seed_lab] = 1.0
    for _ in range(iters):
        new = scores.copy()
        for i in range(n):
            if i in seed_idx: continue
            new[i] = scores[nn[i]].sum(0)             # vote from neighbours
        s = new.sum(1, keepdims=True); s[s == 0] = 1
        scores = new / s
        scores[seed_idx] = 0; scores[seed_idx, seed_lab] = 1.0
    return scores.argmax(1)

if __name__ == "__main__":
    X, y = two_clusters()
    seed_idx = np.array([0, 1, 2, 150, 151, 152])     # only 6 labels
    pred = propagate(X, seed_idx, y[seed_idx])
    acc = (pred == y).mean()
    # baseline: nearest-seed classifier (supervised, same 6 labels)
    d = ((X[:, None] - X[seed_idx][None]) ** 2).sum(2)
    base = (y[seed_idx][d.argmin(1)] == y).mean()
    print(f"labelled points used: {len(seed_idx)} of {len(X)}")
    print(f"label propagation accuracy: {acc:.3f}")
    print(f"(nearest-seed baseline:     {base:.3f})")
    assert acc > 0.95
    print("OK — six labels, spread through the data's structure, classified all 300 points.")
