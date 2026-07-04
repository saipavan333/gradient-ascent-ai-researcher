"""
Track 9 · Lesson 0 — Statistical power: how many seeds do you need?
Run:  python power_analysis.py

Before running an experiment, ask: if my method truly helps by some amount, how
likely am I to DETECT it? That probability is "power". It grows with the number of
seeds (runs) and the effect size, and shrinks with noise. We estimate power by
simulation so you can size an experiment honestly instead of guessing.
"""
import numpy as np
rng = np.random.default_rng(0)

def power(effect, sigma=0.03, n_seeds=10, trials=2000, alpha=0.05):
    """Fraction of experiments that correctly find a significant A-vs-B difference."""
    crit = 1.96 if alpha == 0.05 else 2.58
    hits = 0
    for _ in range(trials):
        A = rng.normal(0.0,    sigma, n_seeds)        # baseline runs
        B = rng.normal(effect, sigma, n_seeds)        # method runs (true gain = effect)
        se = np.sqrt(A.var(ddof=1)/n_seeds + B.var(ddof=1)/n_seeds)
        t = (B.mean() - A.mean()) / se
        hits += abs(t) > crit
    return hits / trials

if __name__ == "__main__":
    print("Power to detect an improvement, by #seeds and true effect size (sigma=0.03):\n")
    print(f"{'seeds':>6} | " + " ".join(f"d={e:<5}" for e in [0.01, 0.02, 0.05]))
    for n in [3, 5, 10, 20, 40, 80]:
        row = " ".join(f"{power(e, n_seeds=n):>6.2f}" for e in [0.01, 0.02, 0.05])
        print(f"{n:>6} | {row}")
    print("\nRule of thumb: small effects need many seeds. If power is < 0.8, your")
    print("experiment may miss a real improvement — or report noise as a result.")
