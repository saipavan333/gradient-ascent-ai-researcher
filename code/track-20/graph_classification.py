"""
Track 14 · Lesson 4 — Graph classification: labelling a WHOLE graph
===================================================================
Node tasks label each node; GRAPH tasks label the entire graph (is this molecule
toxic? is this network community-structured or random?). The recipe: summarise the
graph into a fixed vector — a "readout" of structural features every graph shares —
then classify that vector. Here we tell two-community graphs from random ones, with
the degree matched so the model must use real structure (clustering, triangles, the
spectral gap), not just count edges.

Run:  python graph_classification.py
"""
import numpy as np
rng = np.random.default_rng(0)

def sbm(n_per=15, p_in=0.5, p_out=0.02):
    N = 2 * n_per; comm = np.array([0] * n_per + [1] * n_per); A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            p = p_in if comm[i] == comm[j] else p_out
            if rng.random() < p: A[i, j] = A[j, i] = 1
    return A

def erdos(N=30, p=0.25):
    A = (rng.random((N, N)) < p).astype(float); A = np.triu(A, 1); return A + A.T

def clustering_coef(A):
    """avg over nodes of (edges among its neighbours)/(possible such edges)."""
    N = len(A); cs = []
    for i in range(N):
        nb = np.where(A[i] > 0)[0]; k = len(nb)
        if k < 2: cs.append(0.0); continue
        cs.append(A[np.ix_(nb, nb)].sum() / (k * (k - 1)))
    return float(np.mean(cs))

def spectral_gap(A):
    """2nd-smallest eigenvalue of the normalised Laplacian — small when the graph splits."""
    N = len(A); Aself = A + np.eye(N); Dinv = np.diag(1.0 / np.sqrt(Aself.sum(1)))
    L = np.eye(N) - Dinv @ Aself @ Dinv
    return float(np.sort(np.linalg.eigvalsh(L))[1])

def readout(A):
    """summarise a whole graph as one fixed-length vector."""
    deg = A.sum(1); triangles = np.trace(A @ A @ A) / 6
    return np.array([deg.mean(), deg.std(), clustering_coef(A), triangles / len(A), spectral_gap(A)])

if __name__ == "__main__":
    X, y = [], []
    for _ in range(60): X.append(readout(sbm())); y.append(0)      # community-structured
    for _ in range(60): X.append(readout(erdos())); y.append(1)    # random
    X = np.array(X); y = np.array(y)
    print(f"avg degree — structured {X[y==0,0].mean():.1f}, random {X[y==1,0].mean():.1f}  (matched, so no cheating on size)")
    print(f"clustering — structured {X[y==0,2].mean():.2f}, random {X[y==1,2].mean():.2f}")
    print(f"spectral gap — structured {X[y==0,4].mean():.2f}, random {X[y==1,4].mean():.2f}")
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(500):
        p = 1 / (1 + np.exp(-(X @ w + b))); e = p - y
        w -= 0.1 * X.T @ e / len(y); b -= 0.1 * e.mean()
    acc = (((X @ w + b) > 0).astype(int) == y).mean()
    print(f"telling community-structured graphs from random ones: {acc*100:.0f}% accuracy")
    assert acc > 0.9
    print("OK — readout the whole graph into one vector, then classify it.")
