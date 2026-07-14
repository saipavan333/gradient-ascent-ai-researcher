"""
Track 15 · Lesson 1 — MCMC: sampling a distribution you can't solve
==================================================================
Most posteriors have no tidy formula. Markov Chain Monte Carlo (MCMC) still lets
us draw samples from them, using only the ability to evaluate the (unnormalised)
density. Metropolis–Hastings: propose a small random step, accept it if the density
goes up, and accept it with some probability if it goes down. The chain's samples
end up distributed like the target — here a tricky bimodal shape.

Run:  python mcmc.py
"""
import numpy as np
rng = np.random.default_rng(0)

def log_target(x):
    # unnormalised bimodal density  p(x) ∝ exp(-(x^2 - 4)^2 / 8)  (two humps near ±2)
    return -((x**2 - 4.0)**2) / 8.0

def metropolis(log_p, x0=0.0, steps=200000, step=1.0):
    x = x0; lp = log_p(x); samples = np.empty(steps); acc = 0
    for t in range(steps):
        xp = x + step * rng.standard_normal()
        lpp = log_p(xp)
        if np.log(rng.random()) < lpp - lp:       # accept with prob min(1, p'/p)
            x, lp = xp, lpp; acc += 1
        samples[t] = x
    return samples, acc / steps

if __name__ == "__main__":
    s, acc = metropolis(log_target)
    burn = s[20000:]                              # discard warm-up
    # ground truth via a fine grid (numeric integration of the normalised density)
    grid = np.linspace(-6, 6, 4000); w = np.exp(log_target(grid)); w /= w.sum()
    true_mean = (grid * w).sum(); true_ex2 = (grid**2 * w).sum()
    print(f"acceptance rate {acc:.2f}")
    print(f"E[x]:   MCMC {burn.mean():+.3f}   grid-truth {true_mean:+.3f}")
    print(f"E[x^2]: MCMC {(burn**2).mean():.3f}   grid-truth {true_ex2:.3f}")
    assert abs((burn**2).mean() - true_ex2) < 0.3
    assert (burn > 0).mean() > 0.3 and (burn < 0).mean() > 0.3   # explored BOTH modes
    print("OK — MCMC sampled a distribution with no closed form, visiting both modes.")
