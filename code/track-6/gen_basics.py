"""
Track 6 · Lesson 0 — Generative vs discriminative, from scratch in NumPy
Run:  python gen_basics.py

A DISCRIMINATIVE model learns a boundary: p(y | x), "which class is this point?"
A GENERATIVE model learns the data itself: p(x | y), so it can not only classify
(via Bayes' rule) but also SAMPLE brand-new points. We fit both to the two-moons
data and contrast them: logistic regression draws a boundary; class-conditional
Gaussians let us generate. (The Gaussian is a crude fit to a moon — which is
exactly why the rest of this track builds far better generative models.)
"""
import numpy as np

rng = np.random.default_rng(2)

def two_moons(n):
    t = rng.uniform(0, np.pi, n)
    a = np.c_[np.cos(t),       np.sin(t)]     + rng.normal(0, 0.08, (n, 2))
    b = np.c_[1 - np.cos(t), 0.4 - np.sin(t)] + rng.normal(0, 0.08, (n, 2))
    X = np.vstack([a, b]); y = np.r_[np.zeros(n), np.ones(n)]
    return X, y

def feats(Z):                                   # quadratic features for a curved boundary
    return np.c_[np.ones(len(Z)), Z, Z[:,0]**2, Z[:,1]**2, Z[:,0]*Z[:,1]]

def main():
    X, y = two_moons(400)
    # ---- DISCRIMINATIVE: logistic regression learns p(y|x) ------------------
    F = feats(X); w = np.zeros(F.shape[1])
    for _ in range(4000):
        p = 1/(1+np.exp(-F @ w)); w -= 0.05 * F.T @ (p - y) / len(y)
    acc = ((F @ w > 0) == y).mean()
    print(f"Discriminative logistic regression: train accuracy {acc:.2f}")
    # ---- GENERATIVE: one Gaussian per class, p(x|y); then SAMPLE ------------
    gens = []
    for c in (0, 1):
        Xc = X[y == c]; mu = Xc.mean(0); cov = np.cov(Xc.T)
        gens.append(rng.multivariate_normal(mu, cov, 200))
    gen = np.vstack(gens); gy = np.r_[np.zeros(200), np.ones(200)]
    print("Generative model sampled", len(gen), "new points from p(x|y).")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        xx, yy = np.meshgrid(np.linspace(-2,3,300), np.linspace(-2,2,300))
        grid = np.c_[xx.ravel(), yy.ravel()]
        pp = (1/(1+np.exp(-feats(grid) @ w))).reshape(xx.shape)
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        ax[0].contourf(xx, yy, pp, levels=20, cmap="RdBu_r", alpha=.7)
        ax[0].contour(xx, yy, pp, levels=[.5], colors="k", linewidths=2)
        ax[0].scatter(X[:,0], X[:,1], c=y, cmap="RdBu_r", s=10, edgecolors="k", linewidths=.3)
        ax[0].set_title("discriminative: boundary  p(y|x)")
        ax[1].scatter(gen[:,0], gen[:,1], c=gy, cmap="RdBu_r", s=12, edgecolors="k", linewidths=.3)
        ax[1].set_title("generative: sampled from p(x|y)")
        for a in ax: a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.set_xlim(-2,3); a.set_ylim(-2,2)
        fig.savefig("gen_vs_disc.png", dpi=110, bbox_inches="tight")
        print("Saved gen_vs_disc.png")
    except Exception as e:
        print("(plotting skipped:", e, ")")

    # --- Try it yourself -----------------------------------------------------
    # 1) Use the generative model to classify via Bayes' rule. Compare its accuracy.
    # 2) Replace each class's single Gaussian with a 2-component mixture. Better samples?

if __name__ == "__main__":
    main()
