"""
Track 15 · Lesson 3 — Gaussian processes: uncertainty that knows what it doesn't
================================================================================
A Gaussian process (GP) fits a smooth function AND its uncertainty. Near data the
prediction is confident (narrow band); far from data it widens honestly. The whole
model is a kernel that says "nearby inputs have similar outputs" plus exact Gaussian
algebra for the posterior mean and variance.

Run:  python gaussian_process.py
"""
import numpy as np
rng = np.random.default_rng(0)

def rbf(A, B, l=1.0):
    d = A[:, None] - B[None, :]
    return np.exp(-(d**2) / (2 * l**2))

if __name__ == "__main__":
    Xtr = np.array([-4.0, -3.0, -1.0, 0.0, 2.0, 3.5])
    ytr = np.sin(Xtr)
    noise = 1e-4
    K = rbf(Xtr, Xtr) + noise * np.eye(len(Xtr))
    Kinv = np.linalg.inv(K)
    Xte = np.linspace(-6, 6, 200)
    Ks = rbf(Xte, Xtr)
    mean = Ks @ Kinv @ ytr                              # posterior mean
    var = 1.0 - np.einsum('ij,jk,ik->i', Ks, Kinv, Ks)  # posterior variance
    var = np.clip(var, 0, None)

    # variance is tiny AT the training inputs, large in the gap far from data (x≈6)
    at_data = var[np.argmin(np.abs(Xte - 0.0))]
    far = var[np.argmin(np.abs(Xte - 6.0))]
    fit_err = np.abs(mean[np.argmin(np.abs(Xte - 0.0))] - np.sin(0.0))
    print(f"posterior variance at a data point ≈ {at_data:.4f}")
    print(f"posterior variance far from data (x=6) ≈ {far:.3f}")
    print(f"fit error at x=0: {fit_err:.4f}")
    assert at_data < 0.01 and far > 0.5 and fit_err < 0.05
    print("OK — a GP interpolates the data and grows its error bars where it has no data.")
