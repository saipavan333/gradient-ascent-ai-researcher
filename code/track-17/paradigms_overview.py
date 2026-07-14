"""
Track 13 (Learning Paradigms) · Lesson 0 — Learning without labels: k-means
===========================================================================
Supervised learning needs labels for every example. But most data in the world
is UNLABELLED. The other paradigms in this track squeeze signal out of unlabelled
data. The simplest example is clustering: find groups in data with no labels at
all. k-means does it by repeating two steps — assign each point to its nearest
centre, then move each centre to the mean of its points.

Run:  python paradigms_overview.py
"""
import numpy as np
rng = np.random.default_rng(0)

def make_blobs(n=300):
    centres = np.array([[-3, 0], [0, 3], [3, 0]])
    X = np.vstack([c + rng.standard_normal((n // 3, 2)) for c in centres])
    y = np.repeat([0, 1, 2], n // 3)          # TRUE labels — hidden from k-means
    return X, y

def kmeans(X, k=3, iters=20):
    C = X[rng.choice(len(X), k, replace=False)]           # random initial centres
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)  # distance to each centre
        a = d.argmin(1)                                    # assign nearest centre
        for j in range(k):
            if (a == j).any(): C[j] = X[a == j].mean(0)    # move centre to mean
    return a, C

def purity(assign, y):
    tot = 0
    for j in np.unique(assign):
        labels = y[assign == j]
        tot += np.bincount(labels).max()
    return tot / len(y)

if __name__ == "__main__":
    X, y = make_blobs()
    a, C = kmeans(X, k=3)
    p = purity(a, y)
    print(f"k-means found 3 clusters in unlabelled data.")
    print(f"cluster purity vs the (hidden) true labels: {p:.3f}")
    assert p > 0.9
    print("OK — with NO labels, clustering recovered the real groups.")
