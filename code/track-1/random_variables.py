"""
Track 1 · Lesson 10 — Random variables & distributions
Run:  python random_variables.py
A random variable maps outcomes to numbers; a distribution describes how those
numbers spread. Discrete -> PMF (sums to 1); continuous -> PDF (area = 1).
"""
import numpy as np


def empirical_cdf(samples, x):
    return np.mean(samples <= x)


def main():
    rng = np.random.default_rng(0)

    # X = number of heads in 2 coin flips -> empirical PMF
    flips = rng.integers(0, 2, size=(100_000, 2))      # 0=tails, 1=heads
    X = flips.sum(axis=1)
    values, counts = np.unique(X, return_counts=True)
    pmf = counts / counts.sum()
    print("PMF:", dict(zip(values.tolist(), np.round(pmf, 3).tolist())))
    # ~ {0: 0.25, 1: 0.50, 2: 0.25}

    # Continuous: samples form a histogram approximating the PDF
    heights = rng.normal(170, 7, size=100_000)         # cm
    frac = np.mean((heights >= 165) & (heights <= 175))
    print("P(165<=h<=175) ~", round(frac, 3))

    # CDF: P(X <= x) is the fraction of samples at or below x
    print("F(170) ~", round(empirical_cdf(heights, 170), 3))   # ~0.5

    # A density CAN exceed 1: Uniform(0, 0.5) has height 2, yet integrates to 1
    u = rng.uniform(0, 0.5, size=100_000)
    hist, edges = np.histogram(u, bins=10, density=True)
    print("max density ~", round(hist.max(), 2))       # ~2.0

    # Sampling from a Categorical (like a softmax output)
    probs = np.array([0.1, 0.2, 0.7])                  # 3-class distribution
    draws = rng.choice(len(probs), size=100_000, p=probs)
    print("empirical class freqs:", np.round(np.bincount(draws)/draws.size, 3))

    # --- Try it yourself -----------------------------------------------------
    # 1) Simulate a fair die (1..6); build its PMF and check P(X<=2) ~ 1/3.


if __name__ == "__main__":
    main()
