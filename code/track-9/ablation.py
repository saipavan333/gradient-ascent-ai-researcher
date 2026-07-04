"""
Track 9 · Lesson 1 — Ablation study: measure each component's contribution
Run:  python ablation.py

An ablation removes one component at a time and measures the drop in performance;
the drop is that component's contribution. We classify CONCENTRIC CIRCLES (an inner
disk vs an outer ring) with a logistic model whose feature pipeline has several
parts, then ablate each. The axes are on very different scales on purpose.
"""
import numpy as np
rng = np.random.default_rng(0)

def circles(n):
    # inner disk (class 0) and outer ring (class 1); x-axis stretched x10
    r0 = rng.uniform(0, 1.0, n);   t0 = rng.uniform(0, 2*np.pi, n)
    r1 = rng.uniform(2.0, 3.0, n); t1 = rng.uniform(0, 2*np.pi, n)
    A = np.c_[10*r0*np.cos(t0), r0*np.sin(t0)]
    B = np.c_[10*r1*np.cos(t1), r1*np.sin(t1)]
    X = np.vstack([A, B]); y = np.r_[np.zeros(n), np.ones(n)]
    return X, y

def features(X, use_square=True, use_interaction=True, standardize=True):
    cols = [np.ones(len(X)), X[:, 0], X[:, 1]]            # bias + linear (always on)
    if use_square:      cols += [X[:, 0]**2, X[:, 1]**2]   # quadratic -> radius^2
    if use_interaction: cols += [X[:, 0]*X[:, 1]]          # interaction term
    F = np.c_[tuple(cols)]
    if standardize:
        mu, sd = F.mean(0), F.std(0) + 1e-9; F = (F - mu) / sd; F[:, 0] = 1.0
    return F

def accuracy(cfg, Xtr, ytr, Xte, yte):
    F = features(Xtr, **cfg); w = np.zeros(F.shape[1])
    for _ in range(500):
        p = 1/(1+np.exp(-F @ w)); w -= 0.1 * F.T @ (p - ytr) / len(ytr)
    Fte = features(Xte, **cfg)
    return float(((Fte @ w > 0) == yte).mean())

if __name__ == "__main__":
    Xtr, ytr = circles(300); Xte, yte = circles(300)
    full = dict(use_square=True, use_interaction=True, standardize=True)
    base = accuracy(full, Xtr, ytr, Xte, yte)
    print(f"FULL model accuracy: {base:.3f}\n")
    print(f"{'ablation':>22} {'accuracy':>9} {'drop':>7}")
    rows = []
    for name, change in [("- squared terms", dict(use_square=False)),
                         ("- interaction term", dict(use_interaction=False)),
                         ("- standardization", dict(standardize=False))]:
        cfg = dict(full); cfg.update(change)
        acc = accuracy(cfg, Xtr, ytr, Xte, yte)
        rows.append((name, acc, base-acc))
        print(f"{name:>22} {acc:>9.3f} {base-acc:>+7.3f}")
    print("\nThe larger the drop, the more that component contributed.")
