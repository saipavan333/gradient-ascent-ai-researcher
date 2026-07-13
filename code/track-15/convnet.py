"""
Track 15 · Lesson 2 — A tiny ConvNet, trained from scratch (pure NumPy)
=======================================================================
We put the pieces together into a real, trainable convolutional network and
teach it to tell VERTICAL bars from HORIZONTAL bars in small noisy images.

    image 8x8  ->  conv (F 3x3 filters)  ->  ReLU  ->  maxpool 2x2
               ->  flatten  ->  linear -> softmax (2 classes)

Everything — the forward pass AND the backward pass (gradients for the filters
and the dense layer) — is written by hand. No autograd, no framework. The point
is to prove a CNN is just convolution + pooling + a linear head, trained by the
same gradient descent as everything else in this course.

Run:  python convnet.py
"""
import numpy as np

rng = np.random.default_rng(0)


# ----------------------------------------------------------------------------
# data: 8x8 images, class 0 = a horizontal bar, class 1 = a vertical bar
# ----------------------------------------------------------------------------
def make_dataset(n, size=8, noise=0.15):
    X = np.zeros((n, size, size))
    y = np.zeros(n, dtype=int)
    for i in range(n):
        if rng.random() < 0.5:                       # horizontal bar
            r = rng.integers(1, size - 1)
            X[i, r, :] = 1.0
            y[i] = 0
        else:                                        # vertical bar
            c = rng.integers(1, size - 1)
            X[i, :, c] = 1.0
            y[i] = 1
        X[i] += noise * rng.standard_normal((size, size))   # sensor noise
    return X, y


# ----------------------------------------------------------------------------
# layer primitives (forward + backward)
# ----------------------------------------------------------------------------
def conv_forward(x, W):
    """x:(H,W_) single image, W:(F,kh,kw) filters -> (F,Ho,Wo). Valid conv."""
    F, kh, kw = W.shape
    H, Wd = x.shape
    Ho, Wo = H - kh + 1, Wd - kw + 1
    out = np.zeros((F, Ho, Wo))
    for f in range(F):
        for i in range(Ho):
            for j in range(Wo):
                out[f, i, j] = np.sum(x[i:i + kh, j:j + kw] * W[f])
    return out


def conv_backward(dout, x, W):
    """Gradient of conv output w.r.t. filters W. dout:(F,Ho,Wo) -> dW:(F,kh,kw)."""
    F, kh, kw = W.shape
    _, Ho, Wo = dout.shape
    dW = np.zeros_like(W)
    for f in range(F):
        for i in range(Ho):
            for j in range(Wo):
                dW[f] += dout[f, i, j] * x[i:i + kh, j:j + kw]
    return dW


def relu_forward(x):
    return np.maximum(0, x)


def maxpool_forward(x, size=2):
    """x:(F,H,W) -> (F,H//s,W//s); also return argmax mask for backprop."""
    F, H, W = x.shape
    Ho, Wo = H // size, W // size
    out = np.zeros((F, Ho, Wo))
    mask = np.zeros_like(x)
    for f in range(F):
        for i in range(Ho):
            for j in range(Wo):
                r, c = i * size, j * size
                win = x[f, r:r + size, c:c + size]
                m = win.max()
                out[f, i, j] = m
                idx = np.unravel_index(np.argmax(win), win.shape)
                mask[f, r + idx[0], c + idx[1]] = 1.0   # route grad here
    return out, mask


def maxpool_backward(dout, mask, size=2):
    """Scatter pooled grads back to the argmax positions."""
    F, Ho, Wo = dout.shape
    dx = np.zeros_like(mask)
    for f in range(F):
        for i in range(Ho):
            for j in range(Wo):
                r, c = i * size, j * size
                block = mask[f, r:r + size, c:c + size]
                dx[f, r:r + size, c:c + size] = block * dout[f, i, j]
    return dx


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ----------------------------------------------------------------------------
# the model
# ----------------------------------------------------------------------------
class TinyCNN:
    def __init__(self, n_filters=6, ksize=3, img=8, n_classes=2):
        self.W = rng.standard_normal((n_filters, ksize, ksize)) * 0.3
        conv_out = img - ksize + 1
        pooled = conv_out // 2
        self.flat = n_filters * pooled * pooled
        self.V = rng.standard_normal((n_classes, self.flat)) * 0.1
        self.b = np.zeros(n_classes)
        self.size = 2

    def forward(self, x):
        c = conv_forward(x, self.W)
        r = relu_forward(c)
        p, mask = maxpool_forward(r, self.size)
        flat = p.reshape(-1)
        logits = self.V @ flat + self.b
        probs = softmax(logits)
        cache = (x, c, r, mask, p.shape, flat)
        return probs, cache

    def backward(self, probs, y, cache):
        x, c, r, mask, pshape, flat = cache
        dlogits = probs.copy()
        dlogits[y] -= 1.0                              # softmax + cross-entropy
        dV = np.outer(dlogits, flat)
        db = dlogits
        dflat = self.V.T @ dlogits
        dp = dflat.reshape(pshape)
        dr = maxpool_backward(dp, mask, self.size)
        dc = dr * (c > 0)                              # ReLU grad
        dW = conv_backward(dc, x, self.W)
        return dW, dV, db

    def step(self, grads, lr):
        dW, dV, db = grads
        self.W -= lr * dW
        self.V -= lr * dV
        self.b -= lr * db


def accuracy(model, X, y):
    correct = 0
    for xi, yi in zip(X, y):
        probs, _ = model.forward(xi)
        correct += int(np.argmax(probs) == yi)
    return correct / len(y)


if __name__ == "__main__":
    Xtr, ytr = make_dataset(400)
    Xte, yte = make_dataset(200)
    model = TinyCNN(n_filters=6)

    print(f"Start test accuracy (random weights): {accuracy(model, Xte, yte):.2f}")
    lr, epochs = 0.02, 8
    for ep in range(epochs):
        order = rng.permutation(len(Xtr))
        loss = 0.0
        for idx in order:
            probs, cache = model.forward(Xtr[idx])
            loss += -np.log(probs[ytr[idx]] + 1e-9)
            model.step(model.backward(probs, ytr[idx], cache), lr)
        print(f"epoch {ep + 1}: loss/ex={loss / len(Xtr):.3f}  "
              f"test acc={accuracy(model, Xte, yte):.3f}")

    final = accuracy(model, Xte, yte)
    print(f"\nFinal test accuracy: {final:.3f}")
    assert final > 0.9, "the CNN should learn bars vs bars well above chance"
    print("OK — a hand-built CNN learned oriented-edge detectors and classified from raw pixels.")
