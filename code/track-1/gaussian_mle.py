"""
Track 1 · Lesson 12 — Gaussian, MLE & the bridge to ML
Run:  python gaussian_mle.py
Maximum likelihood turns data into parameters. For a Gaussian, the MLE of the
mean is the sample mean -- and maximizing its log-likelihood is exactly
minimizing squared error. That equivalence is the heart of ML.
"""
import numpy as np


def log_likelihood(mu, x, sigma=2.0):
    return np.sum(-0.5*np.log(2*np.pi*sigma**2) - (x - mu)**2 / (2*sigma**2))


def main():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=5.0, scale=2.0, size=10_000)

    # MLE for a Gaussian: the estimates ARE the sample mean and variance
    mu_hat = data.mean()
    var_hat = data.var()                  # MLE variance (divides by n)
    print("mu_hat ~", round(mu_hat, 3),
          " sigma_hat ~", round(np.sqrt(var_hat), 3))   # ~5.0, ~2.0

    # The MLE maximizes the log-likelihood: it beats any other mu
    print("log-likelihood at several mu:")
    for mu in [3.0, 4.0, mu_hat, 6.0]:
        print(f"  mu={mu:>5.2f} -> {log_likelihood(mu, data):.1f}")

    # THE BRIDGE: maximizing Gaussian log-likelihood == minimizing squared error
    grid = np.linspace(4, 6, 2001)
    sse = [np.sum((data - mu)**2) for mu in grid]
    mu_min_sse = grid[int(np.argmin(sse))]
    print("argmin SSE matches sample mean:",
          np.isclose(mu_hat, mu_min_sse, atol=1e-3))    # True

    # Central Limit Theorem: averages of a SKEWED variable become Gaussian
    skewed_means = np.array([rng.exponential(1.0, size=30).mean()
                             for _ in range(5000)])
    mu, sd = skewed_means.mean(), skewed_means.std()
    within_1sd = np.mean(np.abs(skewed_means - mu) <= sd)
    print("CLT: mean of averages ~", round(mu, 3),       # ~1.0
          " within 1 sigma ~", round(within_1sd, 3))     # ~0.68 -> Gaussian-like

    # --- Try it yourself -----------------------------------------------------
    # 1) Derive and check the MLE variance: (1/n) sum (x_i - mu)^2.
    # 2) Average n Uniform(0,1) samples; watch it become Gaussian as n grows.


if __name__ == "__main__":
    main()
