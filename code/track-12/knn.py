"""
Track 12 · Lesson 0 — k-Nearest Neighbors from scratch
Run:  python knn.py

The simplest possible classifier: to label a new point, find the k closest training
points (by Euclidean distance — Track 1) and take a majority vote. No training, no
parameters — just memory and distance. It's the perfect entry point to classical ML.
"""
import numpy as np
rng = np.random.default_rng(0)

def two_moons(n):
    t = rng.uniform(0, np.pi, n)
    a = np.c_[np.cos(t), np.sin(t)] + rng.normal(0, 0.25, (n, 2)); ya = np.zeros(n)
    b = np.c_[1-np.cos(t), 0.4-np.sin(t)] + rng.normal(0, 0.25, (n, 2)); yb = np.ones(n)
    X = np.vstack([a, b]); y = np.r_[ya, yb]; return X, y.astype(int)

def knn_predict(Xtr, ytr, Xte, k):
    preds = []
    for x in Xte:
        d = np.sum((Xtr - x)**2, axis=1)          # squared Euclidean distance to all
        nn = np.argsort(d)[:k]                     # indices of the k nearest
        preds.append(np.bincount(ytr[nn]).argmax())  # majority vote
    return np.array(preds)

def main():
    Xtr, ytr = two_moons(200); Xte, yte = two_moons(200)
    print("k-NN accuracy on two-moons:\n")
    print(f"{'k':>4} {'test accuracy':>14}")
    for k in [1, 5, 15, 51]:
        acc = (knn_predict(Xtr, ytr, Xte, k) == yte).mean()
        print(f"{k:>4} {acc:>14.3f}")
    print("\nSmall k = flexible but noisy (memorizes); large k = smooth but can underfit.")
    print("k is the bias-variance knob (Track 3), chosen by validation.")

if __name__ == "__main__":
    main()
