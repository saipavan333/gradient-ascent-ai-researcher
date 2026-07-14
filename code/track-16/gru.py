"""
Track 16 · Lesson 3 — GRU from scratch: a simpler gated cell
============================================================
The GRU (Gated Recurrent Unit) is a streamlined LSTM: it merges the cell and
hidden state into one, and uses two gates instead of three. Fewer parameters,
often similar accuracy.

    r = σ(Wr·[x,h])                 reset gate  — how much past to use in the candidate
    z = σ(Wz·[x,h])                 update gate — blend old state vs new
    n = tanh(Wn·[x, r*h])           candidate state
    h_t = (1 - z) * n + z * h_{t-1} — a leaky, gated update

The update gate z plays the LSTM forget-gate's role: when z≈1 the unit copies
its previous state forward almost unchanged, giving gradients a clear path back.
We build it and its BPTT, and solve a long-dependency task (T=20) that a
plain RNN cannot.

Run:  python gru.py
"""
import numpy as np

rng = np.random.default_rng(2)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def make_data(n, T):
    X = rng.integers(0, 2, size=(n, T)).astype(float)
    return X, X[:, 0].astype(int)


class GRU:
    def __init__(self, H=32):
        self.H = H
        D = 1 + H
        s = 1.0 / np.sqrt(D)
        self.Wr = rng.standard_normal((H, D)) * s
        self.Wz = rng.standard_normal((H, D)) * s
        self.Wn = rng.standard_normal((H, D)) * s
        self.br = np.zeros(H); self.bz = np.full(H, 2.0); self.bn = np.zeros(H)  # strong update-bias => carry memory by default
        self.Why = rng.standard_normal((2, H)) * s
        self.by = np.zeros(2)

    def forward(self, x):
        H, T = self.H, len(x)
        h = np.zeros(H)
        cache = []
        for t in range(T):
            zin = np.concatenate([[x[t]], h])
            r = sigmoid(self.Wr @ zin + self.br)
            z = sigmoid(self.Wz @ zin + self.bz)
            zin_n = np.concatenate([[x[t]], r * h])
            n = np.tanh(self.Wn @ zin_n + self.bn)
            h_prev = h
            h = (1 - z) * n + z * h_prev
            cache.append((x[t], h_prev, r, z, n, zin, zin_n))
        p = softmax(self.Why @ h + self.by)
        return p, cache

    def loss_and_grad(self, x, y):
        p, cache = self.forward(x)
        T = len(x)
        loss = -np.log(p[y] + 1e-9)
        grads = {k: np.zeros_like(getattr(self, k)) for k in
                 ["Wr", "Wz", "Wn", "br", "bz", "bn", "Why", "by"]}
        dlogits = p.copy(); dlogits[y] -= 1.0
        hT = cache[-1][1] * 0 + ((1 - cache[-1][3]) * cache[-1][4] + cache[-1][3] * cache[-1][1])
        grads["Why"] += np.outer(dlogits, hT); grads["by"] += dlogits
        dh = self.Why.T @ dlogits
        for t in reversed(range(T)):
            xt, h_prev, r, z, n, zin, zin_n = cache[t]
            dn = dh * (1 - z);            dn_r = dn * (1 - n ** 2)
            dz = dh * (h_prev - n);       dz_r = dz * z * (1 - z)
            # candidate depends on [x, r*h_prev]
            grads["Wn"] += np.outer(dn_r, zin_n); grads["bn"] += dn_r
            d_zin_n = self.Wn.T @ dn_r
            d_rh = d_zin_n[1:]            # gradient into (r*h_prev)
            dr = d_rh * h_prev;          dr_r = dr * r * (1 - r)
            grads["Wr"] += np.outer(dr_r, zin); grads["br"] += dr_r
            grads["Wz"] += np.outer(dz_r, zin); grads["bz"] += dz_r
            # gradient into h_prev from three paths: reset branch, update-carry, gates' inputs
            dh_prev = d_rh * r + dh * z
            dh_prev += (self.Wr.T @ dr_r)[1:]
            dh_prev += (self.Wz.T @ dz_r)[1:]
            dh = dh_prev
        return loss, grads

    def step(self, grads, lr, clip=5.0):
        total = np.sqrt(sum(np.sum(g * g) for g in grads.values()))
        scale = min(1.0, clip / (total + 1e-9))
        for k, g in grads.items():
            setattr(self, k, getattr(self, k) - lr * scale * g)

    def accuracy(self, X, y):
        return np.mean([np.argmax(self.forward(xi)[0]) == yi for xi, yi in zip(X, y)])

    def n_params(self):
        return sum(getattr(self, k).size for k in
                   ["Wr", "Wz", "Wn", "br", "bz", "bn", "Why", "by"])


if __name__ == "__main__":
    T = 20
    Xtr, ytr = make_data(600, T)
    Xte, yte = make_data(300, T)
    gru = GRU(H=32)
    print(f"Task: first-bit memory at T={T}.  GRU parameters: {gru.n_params()}")
    for ep in range(30):
        for idx in rng.permutation(len(Xtr)):
            _, g = gru.loss_and_grad(Xtr[idx], ytr[idx]); gru.step(g, 0.1)
        if (ep + 1) % 5 == 0:
            print(f"epoch {ep+1:2d}: test acc = {gru.accuracy(Xte, yte):.3f}")
    acc = gru.accuracy(Xte, yte)
    print(f"\nFinal GRU test accuracy: {acc:.3f}")
    assert acc > 0.9, "GRU should also solve the long-range task"
    print("OK - GRU solved a long-range task with fewer params than an LSTM.")
