"""
Track 6 · Lesson 1 — An autoregressive model from scratch, in NumPy
Run:  python autoregressive.py

Every joint distribution factorizes by the chain rule of probability:
    p(x1, x2) = p(x1) * p(x2 | x1).
An autoregressive model learns each factor in order, then GENERATES by
"ancestral sampling": draw x1 from p(x1), then draw x2 from p(x2 | x1). Here we
make the factors simple histograms over a grid so you can see the whole thing
with no neural network at all — the same principle scales up to pixels (PixelCNN)
and text (GPT, which you built in Track 5).
"""
import numpy as np

rng = np.random.default_rng(1)

def two_moons(n):
    t = rng.uniform(0, np.pi, n)
    a = np.c_[np.cos(t),       np.sin(t)]     + rng.normal(0, 0.05, (n, 2))
    b = np.c_[1 - np.cos(t), 0.4 - np.sin(t)] + rng.normal(0, 0.05, (n, 2))
    return np.vstack([a, b])

K = 24                                          # number of bins per axis

def fit(X):
    lo, hi = X.min(0) - 0.1, X.max(0) + 0.1
    def binize(col, c): return np.clip(((col - lo[c]) / (hi[c]-lo[c]) * K).astype(int), 0, K-1)
    b1, b2 = binize(X[:,0], 0), binize(X[:,1], 1)
    p1 = np.bincount(b1, minlength=K).astype(float); p1 /= p1.sum()      # p(x1)
    joint = np.zeros((K, K))
    for i, j in zip(b1, b2): joint[i, j] += 1
    cond = joint / np.clip(joint.sum(1, keepdims=True), 1, None)         # p(x2|x1)
    return lo, hi, p1, cond, joint

def sample(n, lo, hi, p1, cond, joint):
    cols = rng.choice(K, size=n, p=p1)                                   # x1 ~ p(x1)
    rows = np.array([rng.choice(K, p=cond[c]) if joint[c].sum() > 0 else 0
                     for c in cols])                                     # x2 ~ p(x2|x1)
    gx = lo[0] + (cols + rng.uniform(0, 1, n)) / K * (hi[0]-lo[0])       # de-quantize
    gy = lo[1] + (rows + rng.uniform(0, 1, n)) / K * (hi[1]-lo[1])
    return np.c_[gx, gy]

def main():
    X = two_moons(4000)
    lo, hi, p1, cond, joint = fit(X)
    gen = sample(4000, lo, hi, p1, cond, joint)
    print(f"Fit p(x1) over {K} bins and p(x2|x1) as a {K}x{K} table.")
    print("Ancestral-sampled", len(gen), "points.")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
        ax[0].scatter(X[:,0], X[:,1], s=5, c="#2563eb"); ax[0].set_title("real")
        ax[1].scatter(gen[:,0], gen[:,1], s=5, c="#b45309"); ax[1].set_title("generated (autoregressive)")
        for a in ax: a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        fig.savefig("autoregressive_samples.png", dpi=110, bbox_inches="tight")
        print("Saved autoregressive_samples.png")
    except Exception as e:
        print("(plotting skipped:", e, ")")

    # --- Try it yourself -----------------------------------------------------
    # 1) Raise K to 60. Sharper moons, but some bins go empty — why is that a problem?
    # 2) Swap the order: model p(x2) p(x1|x2). Same distribution, different factors.

if __name__ == "__main__":
    main()
