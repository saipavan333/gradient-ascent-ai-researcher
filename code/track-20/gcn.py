"""
Track 14 · Lesson 2 — Graph Convolutional Network (GCN) from scratch
====================================================================
A GCN layer is message passing with learnable weights: H' = σ(Â H W). Â mixes each
node with its neighbours; W and σ (a nonlinearity) turn the mixture into features.
Stack two layers and train on just a FEW labelled nodes — the graph structure spreads
the signal so the model labels every node. This is semi-supervised learning on a graph.

Run:  python gcn.py
"""
import numpy as np
rng = np.random.default_rng(0)

def sbm(n_per=25, p_in=0.30, p_out=0.03):
    N = 2 * n_per; comm = np.array([0] * n_per + [1] * n_per); A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            p = p_in if comm[i] == comm[j] else p_out
            if rng.random() < p: A[i, j] = A[j, i] = 1
    return A, comm

def norm_adj(A):
    A = A + np.eye(len(A)); Dinv = np.diag(1.0 / np.sqrt(A.sum(1)))
    return Dinv @ A @ Dinv

def relu(x): return np.maximum(0, x)
def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

if __name__ == "__main__":
    A, y = sbm(); N = len(A); Ah = norm_adj(A)
    X = np.eye(N)                                   # featureless: identity features
    # label just 4 nodes per community (8 of 50)
    sd = np.random.default_rng(2)
    tr = np.concatenate([sd.choice(np.where(y == c)[0], 4, replace=False) for c in (0, 1)])
    mask = np.zeros(N, bool); mask[tr] = True

    W1 = rng.standard_normal((N, 16)) * 0.1
    W2 = rng.standard_normal((16, 2)) * 0.1
    AhX = Ah @ X
    for ep in range(300):
        pre1 = AhX @ W1; H1 = relu(pre1)
        AhH1 = Ah @ H1; scores = AhH1 @ W2; Pr = softmax(scores)
        G = Pr.copy(); G[np.arange(N), y] -= 1        # dscores = P - onehot
        G[~mask] = 0; G /= mask.sum()                 # only labelled nodes drive the loss
        dW2 = AhH1.T @ G
        dH1 = Ah @ G @ W2.T
        dpre1 = dH1 * (pre1 > 0)
        dW1 = AhX.T @ dpre1
        W2 -= 0.5 * dW2; W1 -= 0.5 * dW1
    pred = softmax(Ah @ relu(AhX @ W1) @ W2).argmax(1)
    acc = max((pred == y).mean(), (pred != y).mean())     # label permutation-invariant
    print(f"labelled nodes: {mask.sum()} of {N}")
    print(f"2-layer GCN node-classification accuracy: {acc*100:.0f}%")
    assert acc > 0.9
    print("OK — 8 labels + graph structure classified all", N, "nodes. That's a GCN.")
