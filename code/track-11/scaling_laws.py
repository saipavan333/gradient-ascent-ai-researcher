"""
Track 11 · Lesson 1 — Neural scaling laws from scratch
Run:  python scaling_laws.py

One of the most consequential empirical findings in modern AI: as you grow model
size N, the loss falls as a PREDICTABLE power law,  L(N) = E + A * N^(-alpha),
where E is the irreducible loss. We generate noisy loss-vs-size data from a known
law, fit it, and show the fit EXTRAPOLATES to larger models we never measured —
which is exactly how labs decide to scale up before spending the compute.
"""
import numpy as np
rng = np.random.default_rng(0)

E_TRUE, A_TRUE, ALPHA_TRUE = 1.8, 12.0, 0.32     # the hidden scaling law

def measure_loss(N):
    return E_TRUE + A_TRUE * N**(-ALPHA_TRUE) + rng.normal(0, 0.02, np.shape(N))

def fit_power_law(N, L):
    """Fit L = E + A N^-alpha. Grid-search E, then linear fit in log space for A, alpha."""
    best = None
    for E in np.linspace(0.0, L.min() - 1e-3, 200):     # irreducible loss < min observed
        y = np.log(L - E); x = np.log(N)                 # log(L-E) = logA - alpha*logN
        alpha_neg, logA = np.polyfit(x, y, 1)
        pred = E + np.exp(logA) * N**(alpha_neg)
        sse = np.sum((pred - L)**2)
        if best is None or sse < best[0]:
            best = (sse, E, np.exp(logA), -alpha_neg)
    return best[1], best[2], best[3]                     # E, A, alpha

def main():
    N_train = np.array([1e2, 3e2, 1e3, 3e3, 1e4, 3e4], dtype=float)  # sizes we can afford
    L_train = measure_loss(N_train)
    E, A, alpha = fit_power_law(N_train, L_train)
    print("Fitted scaling law   L(N) = E + A·N^(−α)")
    print(f"  E (irreducible) = {E:.3f}   (true {E_TRUE})")
    print(f"  A               = {A:.3f}   (true {A_TRUE})")
    print(f"  alpha           = {alpha:.3f}   (true {ALPHA_TRUE})")
    # extrapolate to a 100x larger model we never trained
    N_big = 3e6
    pred = E + A * N_big**(-alpha); true = E_TRUE + A_TRUE * N_big**(-ALPHA_TRUE)
    print(f"\nExtrapolate to N = {N_big:.0e} (100x beyond training data):")
    print(f"  predicted loss {pred:.3f}  vs  true {true:.3f}   (error {abs(pred-true):.3f})")
    print("\nThe law lets you forecast a big model's loss from small, cheap runs.")

if __name__ == "__main__":
    main()
