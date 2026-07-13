"""
Track 16 · Lesson 0 — A recurrent neural network from scratch
=============================================================
An MLP sees a fixed-size vector. But language, audio, and time series are
*sequences* of varying length, where order matters. A recurrent neural network
(RNN) processes a sequence one step at a time, carrying a hidden state — a
memory — from step to step. The same weights are reused at every step (like a
CNN reuses a kernel across space, an RNN reuses a cell across time).

Task: read a sequence of bits and report the VERY FIRST bit. To do that the
network must carry information from step 0 all the way to the end — a genuine
(if small) long-term dependency. We implement the forward pass and
backpropagation-through-time (BPTT) by hand.

Run:  python rnn.py
"""
import numpy as np

rng = np.random.default_rng(0)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def make_data(n, T):
    X = rng.integers(0, 2, size=(n, T)).astype(float)
    y = X[:, 0].astype(int)            # label = the first bit
    return X, y


class VanillaRNN:
    def __init__(self, H=32):
        s = 0.4
        self.Wxh = rng.standard_normal((H, 1)) * s
        self.Whh = np.eye(H) * 0.5 + rng.standard_normal((H, H)) * 0.05  # near-identity helps memory
        self.Why = rng.standard_normal((2, H)) * s
        self.bh = np.zeros(H)
        self.by = np.zeros(2)
        self.H = H

    def forward(self, x):
        H, T = self.H, len(x)
        hs = {-1: np.zeros(H)}
        for t in range(T):
            a = self.Wxh[:, 0] * x[t] + self.Whh @ hs[t - 1] + self.bh
            hs[t] = np.tanh(a)
        p = softmax(self.Why @ hs[T - 1] + self.by)
        return p, hs

    def loss_and_grad(self, x, y):
        T = len(x)
        p, hs = self.forward(x)
        loss = -np.log(p[y] + 1e-9)
        dWxh = np.zeros_like(self.Wxh); dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why); dbh = np.zeros_like(self.bh); dby = np.zeros_like(self.by)
        dlogits = p.copy(); dlogits[y] -= 1.0
        dWhy += np.outer(dlogits, hs[T - 1]); dby += dlogits
        dh = self.Why.T @ dlogits                     # gradient into h_{T-1}
        for t in reversed(range(T)):
            da = (1 - hs[t] ** 2) * dh                 # through tanh
            dbh += da
            dWxh += np.outer(da, [x[t]])
            dWhh += np.outer(da, hs[t - 1])
            dh = self.Whh.T @ da                       # pass to previous step
        grads = [dWxh, dWhh, dWhy, dbh, dby]
        return loss, grads

    def step(self, grads, lr, clip=5.0):
        # gradient clipping keeps BPTT stable (exploding-gradient guard)
        total = np.sqrt(sum(np.sum(g * g) for g in grads))
        scale = min(1.0, clip / (total + 1e-9))
        for p, g in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], grads):
            p -= lr * scale * g


def accuracy(model, X, y):
    return np.mean([np.argmax(model.forward(xi)[0]) == yi for xi, yi in zip(X, y)])


if __name__ == "__main__":
    T = 8
    Xtr, ytr = make_data(600, T)
    Xte, yte = make_data(300, T)
    model = VanillaRNN(H=32)
    print(f"Task: report the first bit of a length-{T} sequence.")
    print(f"Start test accuracy: {accuracy(model, Xte, yte):.2f}")
    lr = 0.05
    for ep in range(25):
        for i in rng.permutation(len(Xtr)):
            _, g = model.loss_and_grad(Xtr[i], ytr[i])
            model.step(g, lr)
        if (ep + 1) % 5 == 0:
            print(f"epoch {ep+1:2d}: test acc = {accuracy(model, Xte, yte):.3f}")
    final = accuracy(model, Xte, yte)
    print(f"\nFinal test accuracy: {final:.3f}")
    assert final > 0.9, "RNN should learn to remember the first bit at T=8"
    print("OK — the RNN carried information across 8 time steps using one reused cell.")
