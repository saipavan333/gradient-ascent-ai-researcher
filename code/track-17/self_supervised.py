"""
Track 13 · Lesson 1 — Self-supervised learning: an autoencoder
==============================================================
Self-supervised learning invents its own labels from the data itself. A classic
pretext task is reconstruction: squeeze the input through a small bottleneck code,
then rebuild it. To rebuild well the code must capture the data's real structure —
so the code becomes a useful representation, learned with NO human labels. Here a
tiny autoencoder compresses 8-D data to a 2-D code; afterwards the 2-D codes still
separate the three hidden groups, so a downstream classifier would have an easy job.

Run:  python self_supervised.py
"""
import numpy as np
rng = np.random.default_rng(1)

def make_data(n=300, D=8):
    centres = rng.standard_normal((3, D)) * 3.0
    X = np.vstack([c + 0.6 * rng.standard_normal((n // 3, D)) for c in centres])
    y = np.repeat([0, 1, 2], n // 3)          # TRUE groups — hidden from the model
    return X, y

class AutoEncoder:
    """8 -> tanh(16) -> code(2) -> tanh(16) -> 8, trained by full-batch gradient descent."""
    def __init__(self, D=8, H=16, code=2, seed=0):
        r = np.random.default_rng(seed); s = 0.5
        self.W1 = r.standard_normal((H, D)) * s; self.b1 = np.zeros(H)
        self.W2 = r.standard_normal((code, H)) * s; self.b2 = np.zeros(code)
        self.W3 = r.standard_normal((H, code)) * s; self.b3 = np.zeros(H)
        self.W4 = r.standard_normal((D, H)) * s; self.b4 = np.zeros(D)
    def forward(self, X):
        h1 = np.tanh(X @ self.W1.T + self.b1)
        z  = h1 @ self.W2.T + self.b2            # the code (bottleneck)
        h2 = np.tanh(z @ self.W3.T + self.b3)
        xr = h2 @ self.W4.T + self.b4            # reconstruction
        return xr, z, h1, h2
    def train(self, X, lr=0.03, epochs=2500):
        n = len(X)
        for _ in range(epochs):
            xr, z, h1, h2 = self.forward(X)
            d = 2 * (xr - X) / n
            dW4 = d.T @ h2; db4 = d.sum(0)
            dh2 = (d @ self.W4) * (1 - h2 ** 2); dW3 = dh2.T @ z; db3 = dh2.sum(0)
            dz = dh2 @ self.W3; dW2 = dz.T @ h1; db2 = dz.sum(0)
            dh1 = (dz @ self.W2) * (1 - h1 ** 2); dW1 = dh1.T @ X; db1 = dh1.sum(0)
            self.W4 -= lr*dW4; self.b4 -= lr*db4; self.W3 -= lr*dW3; self.b3 -= lr*db3
            self.W2 -= lr*dW2; self.b2 -= lr*db2; self.W1 -= lr*dW1; self.b1 -= lr*db1
    def codes(self, X): return self.forward(X)[1]
    def recon_err(self, X): return ((self.forward(X)[0] - X) ** 2).sum(1).mean()

def kmeans_purity(Z, y, k=3, iters=40, seed=0):
    r = np.random.default_rng(seed); C = Z[r.choice(len(Z), k, replace=False)].copy()
    for _ in range(iters):
        a = ((Z[:, None] - C[None]) ** 2).sum(2).argmin(1)
        for j in range(k):
            if (a == j).any(): C[j] = Z[a == j].mean(0)
    return sum(np.bincount(y[a == j]).max() for j in np.unique(a)) / len(y)

if __name__ == "__main__":
    X, y = make_data()
    X = (X - X.mean(0)) / X.std(0)            # standardize for stable training
    ae = AutoEncoder()
    err0 = ae.recon_err(X)
    ae.train(X)
    err1 = ae.recon_err(X)
    pur = kmeans_purity(ae.codes(X), y)
    print(f"reconstruction error: {err0:.2f} -> {err1:.3f}  (learned to compress + rebuild)")
    print(f"the 2-D codes still separate the 3 hidden groups: purity {pur:.3f}")
    assert err1 < err0 * 0.2 and pur > 0.9
    print("OK — self-supervision (no labels) learned a representation that keeps the structure.")
