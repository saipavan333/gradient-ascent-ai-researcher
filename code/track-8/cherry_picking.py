"""
Track 8 · Lesson 2 — How results mislead: cherry-picking & missing error bars
Run:  python cherry_picking.py

Two methods, A (baseline) and B (baseline + a useless tweak), have the SAME true
mean performance. Run-to-run randomness (the seed) makes any single comparison
noisy. We show how reporting a single lucky run for B vs an unlucky run for A
manufactures a "win" that vanishes under honest multi-seed evaluation.
"""
import numpy as np
rng = np.random.default_rng(1)

# Each "run" returns a noisy accuracy. A and B have identical true mean = 0.80.
def run(method, seed):
    r = np.random.default_rng(seed)
    base = 0.80 + 0.03 * r.normal()                 # seed-to-seed variation
    return base                                      # B's "tweak" adds nothing real

def evaluate(method, seeds):
    return np.array([run(method, s) for s in seeds])

if __name__ == "__main__":
    seeds = list(range(20))
    A = evaluate("A", seeds); B = evaluate("B", [s+100 for s in seeds])
    # --- the dishonest comparison: cherry-pick best B vs worst A ---
    print("CHERRY-PICKED (one run each):")
    print(f"  best B = {B.max():.3f}   worst A = {A.min():.3f}   'B beats A by {B.max()-A.min():+.3f}' (!)")
    # --- the honest comparison: mean +/- 95% CI over all seeds ---
    def mean_ci(x):
        m = x.mean(); se = x.std(ddof=1)/np.sqrt(len(x)); return m, 1.96*se
    mA, cA = mean_ci(A); mB, cB = mean_ci(B)
    print("\nHONEST (mean over 20 seeds, 95% CI):")
    print(f"  A = {mA:.3f} ± {cA:.3f}")
    print(f"  B = {mB:.3f} ± {cB:.3f}")
    # Welch's t-test (are the means distinguishable?)
    diff = mB - mA
    se = np.sqrt(A.var(ddof=1)/len(A) + B.var(ddof=1)/len(B))
    t = diff/se
    print(f"\n  difference of means: {diff:+.3f}   t-stat: {t:+.2f}")
    print("  |t| < 2  -> the difference is NOT statistically significant.")
    print("  The 'improvement' was an artifact of cherry-picking, not a real effect.")
