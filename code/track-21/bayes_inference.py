"""
Track 15 (Probabilistic & Bayesian ML) · Lesson 0 — Bayesian inference
======================================================================
Bayes' rule updates a BELIEF with EVIDENCE: posterior ∝ prior × likelihood. Start
with a prior over a coin's bias, flip the coin, and the posterior sharpens around
the truth. For a Beta prior + coin flips the math is exact (conjugacy): the
posterior is just Beta(a+heads, b+tails).

Run:  python bayes_inference.py
"""
import numpy as np
rng = np.random.default_rng(0)

def beta_mean(a, b): return a / (a + b)

def credible_interval(a, b, mass=0.95, n=200000):
    s = rng.beta(a, b, n)
    lo, hi = np.percentile(s, [100*(1-mass)/2, 100*(1+mass)/2])
    return lo, hi

if __name__ == "__main__":
    true_p = 0.7
    a0, b0 = 2.0, 2.0                       # prior Beta(2,2): "probably a fairish coin"
    flips = (rng.random(40) < true_p).astype(int)
    heads, tails = flips.sum(), len(flips) - flips.sum()
    a, b = a0 + heads, b0 + tails            # conjugate update -> posterior Beta(a,b)

    prior_m = beta_mean(a0, b0)             # 0.5
    mle = heads / len(flips)                # data-only estimate
    post_m = beta_mean(a, b)               # Bayesian estimate
    lo, hi = credible_interval(a, b)
    print(f"{heads} heads / {len(flips)} flips.  true p = {true_p}")
    print(f"prior mean {prior_m:.2f}  →  posterior mean {post_m:.3f}   (MLE {mle:.3f})")
    print(f"95% credible interval for p: [{lo:.2f}, {hi:.2f}]")
    # posterior mean sits between the prior and the data (a compromise), and the CI covers the truth
    assert min(prior_m, mle) - 1e-9 <= post_m <= max(prior_m, mle) + 1e-9
    assert lo < true_p < hi
    print("OK — prior + evidence → posterior: a belief updated by data, with honest uncertainty.")
