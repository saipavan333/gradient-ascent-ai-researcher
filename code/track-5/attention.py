"""
Track 5 · Lessons 1-2 — Self-attention & multi-head attention
Run:  python attention.py
Attention(Q,K,V) = softmax(QK^T / sqrt(d)) V.  Every query compares to every key
(dot products, Track 1), the scores are normalized, and the values are blended.
"""
import numpy as np


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q, K, V, causal=False):
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)            # (T, T): query i vs key j
    if causal:                               # GPT: position i can't see j > i
        T = scores.shape[0]
        scores = scores + np.triu(np.full((T, T), -1e9), 1)
    weights = softmax(scores, axis=1)        # rows sum to 1
    return weights @ V, weights              # blended values, and the weights


def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, causal=False):
    """Split into heads, attend in each, concatenate, project."""
    T, d = X.shape
    dh = d // n_heads
    Q, K, V = X @ Wq, X @ Wk, X @ Wv         # (T, d)
    heads = []
    for h in range(n_heads):
        sl = slice(h*dh, (h+1)*dh)
        out, _ = attention(Q[:, sl], K[:, sl], V[:, sl], causal)
        heads.append(out)
    concat = np.concatenate(heads, axis=1)   # (T, d)
    return concat @ Wo                       # final linear projection


def main():
    rng = np.random.default_rng(0)
    T, d = 4, 8                               # 4 tokens, dim 8
    X = rng.normal(size=(T, d))
    Wq, Wk, Wv = (rng.normal(0, 1/np.sqrt(d), (d, d)) for _ in range(3))

    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    out, weights = attention(Q, K, V, causal=True)
    print("attention weights (causal — lower triangular):")
    print(np.round(weights, 2))
    print("each row sums to 1:", np.allclose(weights.sum(axis=1), 1))
    print("output shape:", out.shape)        # (4, 8)

    # Multi-head: 2 heads of size 4
    Wo = rng.normal(0, 1/np.sqrt(d), (d, d))
    mh = multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads=2, causal=True)
    print("multi-head output shape:", mh.shape)   # (4, 8)

    # --- Try it yourself -----------------------------------------------------
    # 1) Set causal=False and watch each token attend to ALL tokens (full rows).


if __name__ == "__main__":
    main()
