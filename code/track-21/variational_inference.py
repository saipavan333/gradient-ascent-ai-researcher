"""
Track 15 · Lesson 2 — Variational inference: fit, don't sample
==============================================================
MCMC samples the posterior; variational inference (VI) instead APPROXIMATES it with
a simple distribution (here a Gaussian q) by tuning q to be as close as possible —
maximising the ELBO, equivalently minimising KL(q‖posterior). It's optimisation, not
sampling, so it's fast. We fit a Gaussian to a smooth 1-D posterior and check it
matches the true posterior mean (from a grid).

Run:  python variational_inference.py
"""
import numpy as np
rng = np.random.default_rng(0)

def log_joint(theta, heads=8, tails=2):
    # posterior (up to constant) over log-odds theta of a coin, with a N(0,1.5^2) prior
    p = 1.0 / (1.0 + np.exp(-theta))
    loglik = heads * np.log(p + 1e-12) + tails * np.log(1 - p + 1e-12)
    logprior = -0.5 * (theta / 1.5)**2
    return loglik + logprior

if __name__ == "__main__":
    m, logs = 0.0, 0.0                        # q = Normal(m, exp(logs)^2), start N(0,1)
    lr, S = 0.02, 200
    for it in range(4000):                    # maximise the ELBO by gradient ascent
        s = np.exp(logs)
        eps = rng.standard_normal(S)
        theta = m + s * eps                   # reparameterisation trick
        # grad of E_q[log_joint] via pathwise derivative (numerical d/dtheta)
        h = 1e-4
        g = (log_joint(theta + h) - log_joint(theta - h)) / (2*h)
        gm = g.mean()                         # dELBO/dm
        gs = (g * eps).mean() + 1.0/s         # dELBO/ds includes entropy term (+1/s)
        m += lr * gm
        logs += lr * gs * s                   # chain rule ds/dlogs = s
    # true posterior via grid
    grid = np.linspace(-6, 6, 6000); w = np.exp(log_joint(grid)); w /= w.sum()
    true_mean = (grid * w).sum(); true_sd = np.sqrt(((grid-true_mean)**2 * w).sum())
    print(f"variational q ≈ Normal(mean {m:.3f}, sd {np.exp(logs):.3f})")
    print(f"true posterior:      mean {true_mean:.3f}, sd {true_sd:.3f}")
    assert abs(m - true_mean) < 0.2
    print("OK — VI turned posterior inference into optimisation; q matched the true posterior.")
