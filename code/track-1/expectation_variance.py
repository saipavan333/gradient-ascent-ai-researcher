"""
Track 1 · Lesson 11 — Expectation, variance & covariance
Run:  python expectation_variance.py
Three summaries of a distribution: where it's centered (mean), how spread out
(variance), and -- for two variables -- whether they move together (covariance).
"""
import numpy as np


def main():
    # Expectation & variance of a discrete distribution, by definition
    values = np.array([1, 2, 3, 4])
    probs = np.array([0.4, 0.3, 0.2, 0.1])
    mean = np.sum(values * probs)
    var = np.sum((values - mean)**2 * probs)
    print("E[X] =", mean, " Var =", round(var, 3))     # 2.0, 1.0

    # Shortcut: Var(X) = E[X^2] - (E[X])^2
    E_X2 = np.sum(values**2 * probs)
    print("shortcut matches:", np.isclose(var, E_X2 - mean**2))   # True

    # From samples (how you'll do it on real data)
    rng = np.random.default_rng(0)
    X = rng.normal(5, 2, size=100_000)
    print("sample mean ~", round(X.mean(), 2),
          " std ~", round(X.std(), 2))                 # ~5.0, ~2.0

    # Covariance & correlation of two related variables
    Y = 0.8 * X + rng.normal(0, 1, size=X.size)
    cov = np.mean((X - X.mean()) * (Y - Y.mean()))
    corr = cov / (X.std() * Y.std())
    print("Cov ~", round(cov, 2), " corr ~", round(corr, 2))   # positive, ~0.85

    # The full covariance matrix (its eigenvectors are PCA axes, Lesson 8)
    Sigma = np.cov(np.vstack([X, Y]))
    print("covariance matrix:\n", np.round(Sigma, 2))

    # Var(aX) = a^2 Var(X)
    a = 3.0
    print("Var(aX)==a^2 Var(X):",
          np.isclose(np.var(a*X), a**2 * np.var(X)))   # True

    # --- Try it yourself -----------------------------------------------------
    # 1) Make Y = -1.5 X + small noise; confirm Cov < 0 and corr ~ -1.
    # 2) Verify the best constant predictor under MSE is E[Y] (the mean).


if __name__ == "__main__":
    main()
