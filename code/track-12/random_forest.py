"""
Track 12 · Lesson 2 — Random Forest (bagging + feature subsampling) from scratch
Run:  python random_forest.py

A single deep tree overfits (high variance). A RANDOM FOREST averages many trees,
each trained on a bootstrap resample of the data and choosing splits from a random
subset of features. Averaging decorrelated trees cancels their errors — the same
variance-reduction idea as averaging seeds (Track 9), applied to models.
"""
import numpy as np
from decision_tree import two_moons, fit_tree, predict

def fit_forest(X, y, n_trees=25, max_depth=8, seed=0):
    rng = np.random.default_rng(seed); trees = []
    for b in range(n_trees):
        idx = rng.integers(0, len(X), len(X))               # bootstrap resample
        tr = fit_tree(X[idx], y[idx], max_depth=max_depth,
                      max_features=1, rng=np.random.default_rng(seed + b))  # random feature per split
        trees.append(tr)
    return trees

def forest_predict(trees, X):
    votes = np.array([predict(t, X) for t in trees])         # (n_trees, n_samples)
    return (votes.mean(axis=0) >= 0.5).astype(int)           # majority vote

def main():
    Xtr, ytr = two_moons(200); Xte, yte = two_moons(200)
    single = fit_tree(Xtr, ytr, max_depth=20)                # one deep (overfit) tree
    s_acc = (predict(single, Xte) == yte).mean()
    print("Single deep tree vs Random Forest on two-moons:\n")
    print(f"  single deep tree : test acc {s_acc:.3f}")
    for n in [5, 25, 100]:
        forest = fit_forest(Xtr, ytr, n_trees=n)
        f_acc = (forest_predict(forest, Xte) == yte).mean()
        print(f"  forest ({n:>3} trees): test acc {f_acc:.3f}")
    print("\nAveraging many decorrelated trees beats one deep tree — lower variance,")
    print("smoother boundary, and it's robust without careful depth tuning.")

if __name__ == "__main__":
    main()
