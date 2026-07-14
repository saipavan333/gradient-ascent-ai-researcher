"""
Track 5 (Classical ML) · Logistic Regression from scratch
=========================================================
Logistic regression turns linear regression into a classifier: push the linear
score wx+b through a sigmoid to get a probability in (0,1), and train it by
gradient descent on the cross-entropy loss (Track 3). It's a single neuron with
a sigmoid — the smallest possible neural network — and a workhorse in practice.

Run:  python logistic_regression.py
"""
import numpy as np

rng = np.random.default_rng(1)


def make_data(n=200):
    # two Gaussian blobs -> a linearly separable-ish binary problem
    n2 = n // 2
    Xa = rng.normal([-1.5, -1.0], 1.0, size=(n2, 2))
    Xb = rng.normal([1.5, 1.0], 1.0, size=(n2, 2))
    X = np.vstack([Xa, Xb]); y = np.r_[np.zeros(n2), np.ones(n2)]
    return X, y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit(X, y, lr=0.1, steps=500):
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(steps):
        p = sigmoid(X @ w + b)              # predicted probability of class 1
        err = p - y                         # gradient of cross-entropy is this simple!
        w -= lr * (X.T @ err) / n
        b -= lr * err.mean()
    return w, b


def accuracy(X, y, w, b):
    return ((sigmoid(X @ w + b) > 0.5).astype(int) == y).mean()


if __name__ == "__main__":
    X, y = make_data()
    w, b = fit(X, y)
    acc = accuracy(X, y, w, b)
    print(f"learned weights: {w.round(3)}, bias {b:.3f}")
    print(f"training accuracy: {acc:.3f}")
    # a probability for a clearly class-1-looking point should be high
    p = sigmoid(np.array([2.0, 1.5]) @ w + b)
    print(f"P(class 1 | a top-right point) = {p:.3f}")
    assert acc > 0.9 and p > 0.8
    print("OK — logistic regression learned a linear decision boundary with calibrated probabilities.")
