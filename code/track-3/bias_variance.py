"""
Track 3 · Lesson 1 — Bias-variance decomposition
Run:  python bias_variance.py
Expected test error = bias^2 + variance + noise. We MEASURE bias and variance by
training many models on many resampled datasets and averaging over the domain.
"""
import numpy as np


def main():
    rng = np.random.default_rng(0)
    true_f = lambda x: np.sin(1.3*x) + 0.3*x
    sigma = 0.3
    grid = np.linspace(1.0, 4.0, 21)        # average the decomposition over the domain

    def predictions(degree, n=25, reps=400):
        P = np.zeros((reps, grid.size))
        for r in range(reps):
            x = rng.uniform(0, 5, n)
            y = true_f(x) + rng.normal(0, sigma, n)
            P[r] = np.polyval(np.polyfit(x, y, degree), grid)
        return P

    print(f"{'degree':>6} {'bias^2':>8} {'variance':>9} {'total':>8}")
    for degree in [1, 3, 7]:
        P = predictions(degree)
        mean_pred = P.mean(axis=0)
        bias2 = np.mean((mean_pred - true_f(grid))**2)
        variance = np.mean(P.var(axis=0))
        total = bias2 + variance + sigma**2
        print(f"{degree:>6} {bias2:>8.3f} {variance:>9.3f} {total:>8.3f}")


if __name__ == "__main__":
    main()
