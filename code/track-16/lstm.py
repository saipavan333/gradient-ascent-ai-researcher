"""
Track 16 · Lesson 2 — LSTM from scratch: gates that remember
============================================================
The LSTM (Long Short-Term Memory) fixes the vanishing-gradient problem with a
protected memory lane — the cell state c_t — that information can travel along
almost untouched. Learnable gates decide what to forget, what to write, and what
to read:

    f = σ(Wf·[x,h])   forget gate   — keep or erase old memory
    i = σ(Wi·[x,h])   input gate    — how much new info to write
    g = tanh(Wg·[x,h])              — the candidate new info
    o = σ(Wo·[x,h])   output gate   — how much memory to expose
    c_t = f * c_{t-1} + i * g       — additive memory update (the key!)
    h_t = o * tanh(c_t)

Because c_t updates by ADDITION (not repeated matrix multiply), gradients flow
back through it without vanishing. We build the cell and its BPTT by hand and
solve the first-bit task at T=30 — a length where the vanilla RNN fails.

Run:  python lstm.py
"""
import numpy as np
from rnn import VanillaRNN, accuracy as rnn_accuracy

rng = np.random.default_rng(1)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def make_data(n, T):
    X = rng.integers(0, 2, size=(n, T)).astype(float)
    y = X[:, 0].astype(int)
    return X, y


def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


class LSTM:
    def __init__(self, H=32):
        self.H = H
        D = 1 + H
        s = 1.0 / np.sqrt(D)
        self.Wf = rng.standard_normal((H, D)) * s
        self.Wi = rng.standard_normal((H, D)) * s
        self.Wg = rng.standard_normal((H, D)) * s
        self.Wo = rng.standard_normal((H, D)) * s
        self.bf = np.ones(H)          # forget-bias 1 => remember by default
        self.bi = np.zeros(H); self.bg = np.zeros(H); self.bo = np.zeros(H)
        self.Why = rng.standard_normal((2, H)) * s
        self.by = np.zeros(2)

    def forward(self, x):
        H, T = self.H, len(x)
        h = np.zeros(H); c = np.zeros(H)
        cache = []
        for t in range(T):
            z = np.concatenate([[x[t]], h])
            f = sigmoid(self.Wf @ z + self.bf)
            i = sigmoid(self.Wi @ z + self.bi)
            g = np.tanh(self.Wg @ z + self.bg)
            o = sigmoid(self.Wo @ z + self.bo)
            c_prev = c
            c = f * c_prev + i * g
            h = o * np.tanh(c)
            cache.append((z, f, i, g, o, c_prev, c, h))
        p = softmax(self.Why @ h + self.by)
        return p, cache

    def loss_and_grad(self, x, y):
        p, cache = self.forward(x)
        T = len(x)
        loss = -np.log(p[y] + 1e-9)
        grads = {k: np.zeros_like(getattr(self, k)) for k in
                 ["Wf", "Wi", "Wg", "Wo", "bf", "bi", "bg", "bo", "Why", "by"]}
        dlogits = p.copy(); dlogits[y] -= 1.0
        _, _, _, _, _, _, _, hT = cache[-1]
        grads["Why"] += np.outer(dlogits, hT); grads["by"] += dlogits
        dh = self.Why.T @ dlogits
        dc = np.zeros(self.H)
        for t in reversed(range(T)):
            z, f, i, g, o, c_prev, c, h = cache[t]
            tanhc = np.tanh(c)
            do = dh * tanhc;              do_r = do * o * (1 - o)
            dc = dc + dh * o * (1 - tanhc ** 2)
            df = dc * c_prev;             df_r = df * f * (1 - f)
            di = dc * g;                  di_r = di * i * (1 - i)
            dg = dc * i;                  dg_r = dg * (1 - g ** 2)
            grads["Wf"] += np.outer(df_r, z); grads["bf"] += df_r
            grads["Wi"] += np.outer(di_r, z); grads["bi"] += di_r
            grads["Wg"] += np.outer(dg_r, z); grads["bg"] += dg_r
            grads["Wo"] += np.outer(do_r, z); grads["bo"] += do_r
            dz = self.Wf.T @ df_r + self.Wi.T @ di_r + self.Wg.T @ dg_r + self.Wo.T @ do_r
            dh = dz[1:]                   # gradient into h_{t-1}
            dc = dc * f                   # gradient into c_{t-1}
        return loss, grads

    def step(self, grads, lr, clip=5.0):
        total = np.sqrt(sum(np.sum(g * g) for g in grads.values()))
        scale = min(1.0, clip / (total + 1e-9))
        for k, g in grads.items():
            setattr(self, k, getattr(self, k) - lr * scale * g)

    def accuracy(self, X, y):
        return np.mean([np.argmax(self.forward(xi)[0]) == yi for xi, yi in zip(X, y)])


if __name__ == "__main__":
    T = 30
    Xtr, ytr = make_data(600, T)
    Xte, yte = make_data(300, T)
    print(f"Task: report the first bit across T={T} steps (a long dependency).\n")

    # Contrast: a vanilla RNN at this length struggles to connect step 0 to step T.
    rnn = VanillaRNN(H=32)
    for _ in range(15):
        for idx in rng.permutation(len(Xtr)):
            _, g = rnn.loss_and_grad(Xtr[idx], ytr[idx]); rnn.step(g, 0.05)
    print(f"Vanilla RNN  test accuracy at T={T}: {rnn_accuracy(rnn, Xte, yte):.3f}")

    lstm = LSTM(H=32)
    for ep in range(15):
        for idx in rng.permutation(len(Xtr)):
            _, g = lstm.loss_and_grad(Xtr[idx], ytr[idx]); lstm.step(g, 0.1)
    acc = lstm.accuracy(Xte, yte)
    print(f"LSTM         test accuracy at T={T}: {acc:.3f}")
    assert acc > 0.9, "LSTM should master the long-range first-bit task"
    print("\nOK — the additive cell state let gradients survive 30 steps; the LSTM remembers.")
