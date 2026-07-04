"""
Track 12 · Lesson 3 — Gradient Boosting from scratch (regression)
Run:  python gradient_boosting.py

Boosting builds an ensemble SEQUENTIALLY: each new weak learner is trained to fix the
mistakes (the residuals) of the current ensemble. For squared-error regression the
residual IS the negative gradient of the loss — so fitting residuals is gradient
descent in function space. We boost tiny regression stumps to fit a curve.
"""
import numpy as np
rng = np.random.default_rng(0)

def data(n=120):
    x = np.sort(rng.uniform(-3, 3, n))
    y = np.sin(x) + 0.15 * rng.normal(size=n)
    return x.reshape(-1, 1), y

def fit_stump(x, r):
    """A depth-1 regression tree: one threshold, predict the mean residual each side."""
    best = (1e9, None, 0, 0)
    for thr in np.percentile(x[:, 0], np.arange(5, 100, 5)):
        L = r[x[:, 0] < thr]; R = r[x[:, 0] >= thr]
        if len(L) == 0 or len(R) == 0: continue
        err = ((L - L.mean())**2).sum() + ((R - R.mean())**2).sum()   # squared error
        if err < best[0]: best = (err, thr, L.mean(), R.mean())
    _, thr, lv, rv = best
    return lambda X: np.where(X[:, 0] < thr, lv, rv)

def gradient_boost(x, y, n_stages=40, lr=0.3):
    pred = np.full(len(y), y.mean()); stumps = [(None, y.mean())]; curve = []
    for _ in range(n_stages):
        resid = y - pred                        # = negative gradient of ½(y-pred)²
        h = fit_stump(x, resid)                 # weak learner fits the residual
        pred = pred + lr * h(x)                 # shrink by learning rate, add on
        stumps.append((h, lr)); curve.append(np.sqrt(np.mean((y - pred)**2)))
    return stumps, curve

def main():
    x, y = data()
    stumps, curve = gradient_boost(x, y)
    print("Gradient boosting stumps to fit a noisy sine:\n")
    print(f"{'stage':>6} {'train RMSE':>11}")
    for s in [1, 5, 10, 20, 40]:
        print(f"{s:>6} {curve[s-1]:>11.3f}")
    print(f"\nRMSE falls stage by stage ({curve[0]:.2f} -> {curve[-1]:.2f}) as each stump")
    print("corrects the previous ensemble's residual — gradient descent in function space.")

if __name__ == "__main__":
    main()
