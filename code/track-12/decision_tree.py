"""
Track 12 · Lesson 1 — A decision tree (CART) from scratch
Run:  python decision_tree.py

A decision tree asks a sequence of yes/no questions ("is feature j < threshold?"),
splitting the data to make each region as PURE as possible. We pick splits that most
reduce Gini impurity, recursively, up to a max depth. Trees are the building block of
random forests and gradient boosting (the next two lessons).
"""
import numpy as np
rng = np.random.default_rng(0)

def two_moons(n):
    t = rng.uniform(0, np.pi, n)
    a = np.c_[np.cos(t), np.sin(t)] + rng.normal(0, 0.25, (n, 2))
    b = np.c_[1-np.cos(t), 0.4-np.sin(t)] + rng.normal(0, 0.25, (n, 2))
    X = np.vstack([a, b]); y = np.r_[np.zeros(n), np.ones(n)]; return X, y.astype(int)

def gini(y):
    if len(y) == 0: return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    return 1.0 - (p**2).sum()

class Node:
    __slots__ = ("feat", "thr", "left", "right", "label")
    def __init__(self): self.feat=self.thr=self.left=self.right=self.label=None

def build(X, y, depth, max_depth, max_features=None, rng=np.random.default_rng(0)):
    node = Node()
    if depth == max_depth or gini(y) == 0 or len(y) < 5:
        node.label = np.bincount(y, minlength=2).argmax(); return node
    best = (1e9, None, None)                       # (impurity, feature, threshold)
    feats = range(X.shape[1]) if max_features is None else rng.choice(X.shape[1], max_features, replace=False)
    for j in feats:
        for thr in np.percentile(X[:, j], np.arange(10, 100, 10)):
            L = y[X[:, j] < thr]; R = y[X[:, j] >= thr]
            if len(L) == 0 or len(R) == 0: continue
            imp = (len(L)*gini(L) + len(R)*gini(R)) / len(y)   # weighted child impurity
            if imp < best[0]: best = (imp, j, thr)
    if best[1] is None:
        node.label = np.bincount(y, minlength=2).argmax(); return node
    _, j, thr = best; node.feat, node.thr = j, thr
    m = X[:, j] < thr
    node.left  = build(X[m],  y[m],  depth+1, max_depth, max_features, rng)
    node.right = build(X[~m], y[~m], depth+1, max_depth, max_features, rng)
    return node

def predict1(node, x):
    while node.label is None:
        node = node.left if x[node.feat] < node.thr else node.right
    return node.label

def fit_tree(X, y, max_depth=4, max_features=None, rng=np.random.default_rng(0)):
    return build(X, y, 0, max_depth, max_features, rng)
def predict(tree, X): return np.array([predict1(tree, x) for x in X])

def main():
    Xtr, ytr = two_moons(200); Xte, yte = two_moons(200)
    print("Decision tree on two-moons — depth controls fit:\n")
    print(f"{'max_depth':>10} {'train acc':>10} {'test acc':>10}")
    for d in [1, 2, 4, 8, 20]:
        tree = fit_tree(Xtr, ytr, d)
        tr = (predict(tree, Xtr) == ytr).mean(); te = (predict(tree, Xte) == yte).mean()
        print(f"{d:>10} {tr:>10.3f} {te:>10.3f}")
    print("\nDeep trees fit training data perfectly but overfit (train↑, test plateaus/↓).")

if __name__ == "__main__":
    main()
