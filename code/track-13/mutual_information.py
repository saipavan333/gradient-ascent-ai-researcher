"""
Track 13 · Lesson 3 — Mutual information: how much X tells you about Y
Run:  python mutual_information.py

Mutual information  I(X;Y) = Σ p(x,y) log₂( p(x,y) / (p(x)p(y)) )  measures how much
knowing X reduces uncertainty about Y. It's zero iff X and Y are independent, and —
unlike correlation — it detects NONLINEAR dependence. We estimate it from samples by
binning, on three relationships.
"""
import numpy as np
rng = np.random.default_rng(0)

def mutual_information(x, y, bins=12):
    c, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = c / c.sum()
    px = pxy.sum(1, keepdims=True); py = pxy.sum(0, keepdims=True)
    m = pxy > 0
    return float((pxy[m] * np.log2(pxy[m] / (px @ py)[m])).sum())

def main():
    n = 4000
    x = rng.normal(size=n)
    datasets = {
        "independent   (Y random)":      rng.normal(size=n),
        "linear        (Y = X + noise)": x + 0.5*rng.normal(size=n),
        "nonlinear     (Y = X² + noise)": x**2 + 0.5*rng.normal(size=n),
    }
    print(f"{'relationship':>28} {'correlation':>12} {'mutual info':>12}")
    for name, y in datasets.items():
        corr = np.corrcoef(x, y)[0, 1]
        mi = mutual_information(x, y)
        print(f"{name:>28} {corr:>12.2f} {mi:>12.3f} bits")
    print("\nNote the nonlinear case: correlation ≈ 0 (misses it!) but MI is clearly > 0.")
    print("MI catches ANY dependence — which is why it's used for feature selection.")

if __name__ == "__main__":
    main()
