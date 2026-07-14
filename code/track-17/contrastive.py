"""
Track 13 · Lesson 2 — Contrastive learning (SimCLR-style)
========================================================
Contrastive learning builds a representation by pulling together two augmented
"views" of the SAME example, and pushing apart views of DIFFERENT examples. No
labels — the only supervision is "these two came from the same source." We use the
NT-Xent (normalised temperature-scaled cross-entropy) idea on toy points: after
training, augmentations of the same point sit close in embedding space.

Run:  python contrastive.py
"""
import numpy as np
rng = np.random.default_rng(2)

def normalise(v): return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)

if __name__ == "__main__":
    N, D = 40, 2                       # 40 anchor points in 2-D
    anchors = rng.standard_normal((N, D)) * 2
    W = rng.standard_normal((D, D)) * 0.3       # linear encoder we will train
    def aug(x): return x + 0.15 * rng.standard_normal(x.shape)   # augmentation = jitter
    tau = 0.3
    for step in range(400):
        i = rng.integers(N)
        a, b = aug(anchors[i]), aug(anchors[i])           # two views of the same point
        negs = np.array([aug(anchors[j]) for j in range(N) if j != i])
        za = normalise(W @ a); zb = normalise(W @ b)
        zn = normalise((W @ negs.T).T)
        pos = np.dot(za, zb) / tau
        neg = zn @ za / tau
        # softmax cross-entropy: make the positive score highest
        logits = np.concatenate([[pos], neg])
        p = np.exp(logits - logits.max()); p /= p.sum()
        # gradient of -log p[0] wrt za (treat zb, zn as fixed targets)
        grad = (p[0] - 1) * (zb / tau) + sum(p[k+1] * (zn[k] / tau) for k in range(len(zn)))
        W -= 0.05 * np.outer(grad, a)
    # evaluate: distance between same-point views vs different-point views
    emb = normalise((W @ anchors.T).T)
    same = np.mean([np.linalg.norm(normalise(W@aug(anchors[i])) - normalise(W@aug(anchors[i]))) for i in range(N)])
    diff = np.mean([np.linalg.norm(emb[i] - emb[j]) for i in range(N) for j in range(N) if i != j])
    print(f"avg embedding distance — same point's views: {same:.3f}")
    print(f"avg embedding distance — different points:   {diff:.3f}")
    assert same < diff
    print("OK — contrastive training put augmentations of the same point close, others far.")
