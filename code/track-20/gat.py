"""
Track 14 · Lesson 3 — Graph Attention Network (GAT) from scratch
================================================================
A GCN weights every neighbour by a fixed degree factor. A GAT LEARNS a weight
(an "attention") for each neighbour, so it can lean on the informative ones and
ignore the rest. We build one GAT layer with additive attention + a masked softmax,
train it with hand-derived backprop (gradient-checked), and confirm it (a) classifies
well and (b) learns to attend more to same-community neighbours than to cross ones.

Run:  python gat.py
"""
import numpy as np
rng = np.random.default_rng(0)

def sbm(n_per=20, p_in=0.30, p_out=0.05):
    N = 2 * n_per; comm = np.array([0] * n_per + [1] * n_per); A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            p = p_in if comm[i] == comm[j] else p_out
            if rng.random() < p: A[i, j] = A[j, i] = 1
    return A, comm

def leaky(x, s=0.2): return np.where(x > 0, x, s * x)
def dleaky(x, s=0.2): return np.where(x > 0, 1.0, s)
def softmax_rows(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def forward(X, W, a1, a2, mask, NEG=-1e9):
    M = X @ W                                   # projected features / messages  (N,F)
    e1 = M @ a1; e2 = M @ a2                     # (N,1) each
    Zraw = e1 + e2.T                             # (N,N) additive scores
    Zl = leaky(Zraw)
    Zm = np.where(mask, Zl, NEG)                 # only real edges compete
    alpha = softmax_rows(Zm)                     # attention weights per node
    O = alpha @ M                                # aggregated logits (N,F=2)
    cache = (X, W, a1, a2, mask, M, Zraw, alpha, O)
    return O, alpha, cache

def loss_and_grad(cache, y, lab):
    X, W, a1, a2, mask, M, Zraw, alpha, O = cache
    P = softmax_rows(O)
    L = -np.log(P[lab, y[lab]] + 1e-12).mean()
    dO = P.copy(); dO[np.arange(len(y)), y] -= 1
    m = np.zeros(len(y), bool); m[lab] = True
    dO[~m] = 0; dO /= m.sum()
    # O = alpha @ M
    dAlpha = dO @ M.T
    dM = alpha.T @ dO
    # alpha = softmax_rows(Zm)
    dZm = alpha * (dAlpha - (alpha * dAlpha).sum(1, keepdims=True))
    dZl = dZm * mask                            # gradient blocked at non-edges
    dZraw = dZl * dleaky(Zraw)
    de1 = dZraw.sum(1, keepdims=True)           # (N,1)
    de2 = dZraw.sum(0).reshape(-1, 1)           # (N,1)
    da1 = M.T @ de1; da2 = M.T @ de2
    dM = dM + de1 @ a1.T + de2 @ a2.T
    dW = X.T @ dM
    return L, dW, da1, da2

if __name__ == "__main__":
    A, y = sbm(); N = len(A); mask = (A + np.eye(N)) > 0
    mu = rng.standard_normal(6)
    sign = np.where(y == 0, 1.0, -1.0)[:, None]
    X = sign * mu[None, :] + 1.6 * rng.standard_normal((N, 6))   # noisy community signal
    W = rng.standard_normal((6, 2)) * 0.3; a1 = rng.standard_normal((2, 1)) * 0.3; a2 = rng.standard_normal((2, 1)) * 0.3
    sd = np.random.default_rng(2)
    lab = np.concatenate([sd.choice(np.where(y == c)[0], 5, replace=False) for c in (0, 1)])

    # ---- gradient check on a1 before training ----
    O, _, cache = forward(X, W, a1, a2, mask)
    L0, dW, da1, da2 = loss_and_grad(cache, y, lab)
    eps = 1e-5; i = 0; a1p = a1.copy(); a1p[i] += eps
    Lp = loss_and_grad(forward(X, W, a1p, a2, mask)[2], y, lab)[0]
    num = (Lp - L0) / eps
    print(f"gradient check (a1[0]): analytic {da1[i,0]:+.5f}  numeric {num:+.5f}")
    assert abs(da1[i, 0] - num) < 1e-3

    # ---- train ----
    for ep in range(300):
        O, alpha, cache = forward(X, W, a1, a2, mask)
        L, dW, da1, da2 = loss_and_grad(cache, y, lab)
        W -= 0.5 * dW; a1 -= 0.5 * da1; a2 -= 0.5 * da2
    O, alpha, _ = forward(X, W, a1, a2, mask)
    pred = O.argmax(1); acc = max((pred == y).mean(), (pred != y).mean())

    # same-community vs cross-community average attention (over real edges)
    same = cross = 0.0; ns = nc = 0
    for i in range(N):
        for j in range(N):
            if A[i, j] and i != j:
                if y[i] == y[j]: same += alpha[i, j]; ns += 1
                else: cross += alpha[i, j]; nc += 1
    same /= ns; cross /= nc
    print(f"GAT node accuracy: {acc*100:.0f}%")
    print(f"avg attention on same-community edges {same:.3f}  vs cross-community {cross:.3f}")
    assert acc > 0.9 and same > cross
    print("OK — attention learned to focus on the neighbours that actually help.")
