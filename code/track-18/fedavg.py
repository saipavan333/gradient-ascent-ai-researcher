"""
Track 17 (Federated & Privacy) · Lesson 0 — Federated Averaging (FedAvg)
=======================================================================
Sometimes data can't be pooled — it lives on phones, in hospitals, across banks,
and privacy or law forbids sending it to one server. FEDERATED LEARNING trains a
shared model anyway: each client trains locally on its OWN data, sends only the
updated WEIGHTS to a server, and the server AVERAGES them into a new global model.
Data never leaves the device. Here FedAvg reaches almost the same accuracy as
centralised training that saw all the data at once.

Run:  python fedavg.py
"""
import numpy as np
rng = np.random.default_rng(0)

def make_data(n=2000):
    X = rng.standard_normal((n, 4))
    w = np.array([1.5, -2.0, 1.0, 0.5])
    y = (X @ w + 0.3 * rng.standard_normal(n) > 0).astype(int)
    return X, y

def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

def train_local(X, y, w, b, steps=15, lr=0.3):
    w, b = w.copy(), b
    for _ in range(steps):
        e = sigmoid(X @ w + b) - y
        w -= lr * X.T @ e / len(y); b -= lr * e.mean()
    return w, b

def accuracy(X, y, w, b): return ((sigmoid(X @ w + b) > 0.5).astype(int) == y).mean()

if __name__ == "__main__":
    X, y = make_data(); Xte, yte = make_data(4000)
    K = 10
    idx = rng.permutation(len(X)); shards = np.array_split(idx, K)   # each client's private data

    # --- centralised baseline (sees ALL data) ---
    wc, bc = train_local(X, y, np.zeros(4), 0.0, steps=300)
    print(f"centralised accuracy (all data pooled): {accuracy(Xte,yte,wc,bc):.3f}")

    # --- FedAvg: clients train locally, server averages weights ---
    w, b = np.zeros(4), 0.0
    for rnd_ in range(30):                          # communication rounds
        ws, bs = [], []
        for s in shards:
            lw, lb = train_local(X[s], y[s], w, b)  # local training on private shard
            ws.append(lw); bs.append(lb)
        w = np.mean(ws, 0); b = float(np.mean(bs))  # AVERAGE the client models
    fa = accuracy(Xte, yte, w, b)
    print(f"FedAvg accuracy (data never pooled):    {fa:.3f}")
    assert fa > accuracy(Xte, yte, wc, bc) - 0.03
    print("OK — averaging locally-trained models matches central training, without sharing data.")
