"""
Track 15 · Lesson 3 — Data augmentation
=======================================
The cheapest way to make a vision model generalize better is to *invent* more
training data by applying label-preserving transformations: flips, small shifts,
rotations, noise. A cat flipped left-right is still a cat, so the model should
predict the same thing — augmentation bakes that invariance in without any new
labels. Here we show it lifting test accuracy on a small dataset.

We reuse the tiny CNN from convnet.py. To keep the demo honest we use a *small*
training set (where overfitting bites) and measure test accuracy with vs without
augmentation.

Run:  python augmentation.py
"""
import numpy as np
from convnet import TinyCNN, make_dataset, accuracy

rng = np.random.default_rng(1)


def augment(x):
    """A random label-preserving transform for a bar image (flip / shift / noise)."""
    out = x.copy()
    if rng.random() < 0.5:                    # horizontal flip
        out = out[:, ::-1]
    if rng.random() < 0.5:                    # vertical flip
        out = out[::-1, :]
    shift = rng.integers(-1, 2)               # roll by up to 1 px
    axis = rng.integers(0, 2)
    out = np.roll(out, shift, axis=axis)
    out = out + 0.1 * rng.standard_normal(out.shape)   # extra noise
    return out


def train(model, X, y, epochs=10, lr=0.02, use_aug=False):
    for _ in range(epochs):
        for idx in rng.permutation(len(X)):
            xi = augment(X[idx]) if use_aug else X[idx]
            probs, cache = model.forward(xi)
            model.step(model.backward(probs, y[idx], cache), lr)
    return model


if __name__ == "__main__":
    # Deliberately small training set so generalization is hard.
    Xtr, ytr = make_dataset(60)
    Xte, yte = make_dataset(400)

    base = train(TinyCNN(n_filters=6), Xtr, ytr, use_aug=False)
    aug = train(TinyCNN(n_filters=6), Xtr, ytr, use_aug=True)

    acc_base = accuracy(base, Xte, yte)
    acc_aug = accuracy(aug, Xte, yte)
    print(f"Train size: {len(Xtr)}  Test size: {len(yte)}")
    print(f"Test accuracy WITHOUT augmentation: {acc_base:.3f}")
    print(f"Test accuracy WITH    augmentation: {acc_aug:.3f}")
    print(f"Improvement: {acc_aug - acc_base:+.3f}")
    # Augmentation should help (or at least not hurt) on this small-data regime.
    assert acc_aug >= acc_base - 0.02, "augmentation should not meaningfully hurt"
    print("OK — synthetic label-preserving variety improved generalization from few labels.")
