"""
Track 15 · Lesson 0 — Convolution from scratch
===============================================
A convolution (really: cross-correlation) slides a small weight grid — a
*kernel* — over an image and, at every position, computes a weighted sum of the
pixels under it. One kernel = one learned feature detector. Here we implement it
in a few lines of NumPy and watch a hand-designed kernel find vertical edges.

Run:  python conv2d.py
"""
import numpy as np


def conv2d(image, kernel, stride=1, pad=0):
    """Valid 2D cross-correlation of a single-channel image with a kernel.

    image : (H, W)      kernel : (kh, kw)
    returns (H_out, W_out) where H_out = (H + 2*pad - kh)//stride + 1
    """
    if pad:
        image = np.pad(image, pad, mode="constant")
    H, W = image.shape
    kh, kw = kernel.shape
    H_out = (H - kh) // stride + 1
    W_out = (W - kw) // stride + 1
    out = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            r, c = i * stride, j * stride
            patch = image[r:r + kh, c:c + kw]      # the receptive field
            out[i, j] = np.sum(patch * kernel)      # weighted sum
    return out


def make_test_image(n=9):
    """A synthetic image: left half dark, right half bright => one vertical edge."""
    img = np.zeros((n, n))
    img[:, n // 2:] = 1.0
    return img


if __name__ == "__main__":
    np.set_printoptions(precision=1, suppress=True)
    img = make_test_image(9)

    # A Sobel-style vertical-edge detector: big response where brightness
    # jumps horizontally, ~0 in flat regions.
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=float)

    edges = conv2d(img, sobel_x, stride=1, pad=1)

    print("Input image (0=dark, 1=bright):")
    print(img)
    print("\nAfter vertical-edge kernel (large |value| = an edge):")
    print(edges)

    # The strongest responses should sit on the middle column where dark meets
    # bright. Flat regions (all-dark or all-bright) should be ~0.
    col_energy = np.abs(edges).sum(axis=0)
    detected = int(np.argmax(col_energy))
    print(f"\nColumn with strongest edge response: {detected} "
          f"(true edge is between col {img.shape[1]//2 - 1} and {img.shape[1]//2})")
    assert abs(detected - img.shape[1] // 2) <= 1, "edge should be found at the boundary"
    print("OK — the kernel located the vertical edge from raw pixels alone.")
