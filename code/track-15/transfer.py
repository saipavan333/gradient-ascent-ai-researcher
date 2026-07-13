"""
Track 15 · Lesson 4 — Transfer learning (features first, labels later)
======================================================================
Training a big vision model needs a lot of labels. But the *early* features a
CNN learns — oriented edges, textures — are generic: they transfer to new tasks.
So the dominant recipe in practice is: take a network pretrained on abundant
data, FREEZE its feature extractor, and train only a small new head on your
(scarce) target labels.

Demo: we pretrain the tiny CNN's convolutional filters on a large "bars" dataset,
then reuse those frozen features to solve a target task with only a handful of
labels — and compare against a linear model on raw pixels given the same few
labels. Reusing learned features wins decisively when labels are scarce.

Run:  python transfer.py
"""
import numpy as np
from convnet import (TinyCNN, make_dataset, conv_forward, relu_forward,
                     maxpool_forward, softmax)

rng = np.random.default_rng(2)


def pretrain(model, X, y, epochs=8, lr=0.02):
    for _ in range(epochs):
        for idx in rng.permutation(len(X)):
            probs, cache = model.forward(X[idx])
            model.step(model.backward(probs, y[idx], cache), lr)
    return model


def extract_features(model, x):
    """Frozen feature extractor: conv -> relu -> maxpool -> flatten."""
    c = conv_forward(x, model.W)
    r = relu_forward(c)
    p, _ = maxpool_forward(r, model.size)
    return p.reshape(-1)


def train_linear(feats, y, n_classes=2, epochs=300, lr=0.1):
    """Plain multinomial logistic regression trained by gradient descent."""
    d = feats.shape[1]
    W = np.zeros((n_classes, d))
    b = np.zeros(n_classes)
    for _ in range(epochs):
        for xi, yi in zip(feats, y):
            p = softmax(W @ xi + b)
            p[yi] -= 1.0
            W -= lr * np.outer(p, xi)
            b -= lr * p
    return W, b


def linear_acc(W, b, feats, y):
    pred = np.array([np.argmax(softmax(W @ xi + b)) for xi in feats])
    return (pred == y).mean()


if __name__ == "__main__":
    # 1) Pretrain conv features on ABUNDANT source data.
    Xsrc, ysrc = make_dataset(600)
    base = pretrain(TinyCNN(n_filters=6), Xsrc, ysrc)

    # 2) Target task with only a FEW labels (same family, fresh samples).
    Xfew, yfew = make_dataset(16)          # scarce labels
    Xte, yte = make_dataset(400)

    # 2a) Transfer: frozen CNN features + small linear head.
    Ftr = np.array([extract_features(base, x) for x in Xfew])
    Fte = np.array([extract_features(base, x) for x in Xte])
    Wt, bt = train_linear(Ftr, yfew)
    acc_transfer = linear_acc(Wt, bt, Fte, yte)

    # 2b) Baseline: linear model straight on raw pixels, same few labels.
    Ptr = Xfew.reshape(len(Xfew), -1)
    Pte = Xte.reshape(len(Xte), -1)
    Wp, bp = train_linear(Ptr, yfew)
    acc_raw = linear_acc(Wp, bp, Pte, yte)

    print(f"Target labels available: {len(yfew)}")
    print(f"Test acc — transfer (frozen CNN features + head): {acc_transfer:.3f}")
    print(f"Test acc — raw pixels + linear head:              {acc_raw:.3f}")
    print(f"Transfer advantage: {acc_transfer - acc_raw:+.3f}")
    assert acc_transfer >= acc_raw, "reused features should beat raw pixels at low label counts"
    print("OK — features learned on abundant data solved a new task from very few labels.")
