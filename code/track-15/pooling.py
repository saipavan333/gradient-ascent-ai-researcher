"""
Track 15 · Lesson 1 — Pooling & stride (downsampling)
=====================================================
Convolutions detect features but keep the image large. *Pooling* shrinks the
feature map by summarizing each little window with one number — its max (the
strongest response) or its mean. This buys two things: fewer numbers to process,
and a little translation invariance (a feature shifted by one pixel still lands
in the same pooled cell). Stride does the same shrinking inside the convolution
itself.

Run:  python pooling.py
"""
import numpy as np


def pool2d(x, size=2, stride=2, mode="max"):
    """Downsample a (H, W) map with non-overlapping (by default) windows."""
    H, W = x.shape
    H_out = (H - size) // stride + 1
    W_out = (W - size) // stride + 1
    out = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            r, c = i * stride, j * stride
            window = x[r:r + size, c:c + size]
            out[i, j] = window.max() if mode == "max" else window.mean()
    return out


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    rng = np.random.default_rng(0)

    # A 6x6 feature map with one strong activation (a "detected feature").
    fmap = rng.random((6, 6)) * 0.3
    fmap[1, 4] = 1.0  # a sharp peak

    mx = pool2d(fmap, size=2, stride=2, mode="max")
    av = pool2d(fmap, size=2, stride=2, mode="mean")

    print("Feature map (6x6):")
    print(fmap)
    print(f"\nMax-pool 2x2  -> shape {mx.shape} (kept the peak = {mx.max():.2f}):")
    print(mx)
    print(f"\nMean-pool 2x2 -> shape {av.shape} (averaged, peak diluted):")
    print(av)

    # Translation invariance: shift the peak by one pixel inside its 2x2 window;
    # max-pool returns the SAME downsampled map.
    shifted = rng.random((6, 6)) * 0.3
    shifted[0, 5] = 1.0  # same 2x2 block as (1,4)
    mx2 = pool2d(shifted, size=2, stride=2, mode="max")
    same_cell = np.isclose(mx[0, 2], 1.0) and np.isclose(mx2[0, 2], 1.0)
    print(f"\nPeak moved within its window; max-pool cell unchanged? {same_cell}")
    assert same_cell, "max-pool should be invariant to small shifts within a window"
    print("OK — pooling shrank the map and gave small-shift invariance.")
