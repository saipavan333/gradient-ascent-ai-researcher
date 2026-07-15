"""
Track 18 · Lesson 3 — Backdoor adjustment
=========================================
We usually can't run the experiment (do(T)). Backdoor adjustment recovers the causal
effect from OBSERVATIONAL data by adjusting for the confounders that open a "backdoor
path" between T and Y:  E[Y | do(T=t)] = Σ_z E[Y | T=t, Z=z] P(Z=z). Average the
within-stratum effects, weighted by how common each stratum is.

Run:  python backdoor.py
"""
import numpy as np
rng = np.random.default_rng(0)

if __name__ == "__main__":
    n = 40000
    Z = (rng.random(n) < 0.5).astype(int)
    T = (rng.random(n) < np.where(Z == 1, 0.8, 0.2)).astype(int)
    Y = 20.0*Z + 5.0*T + rng.standard_normal(n)*2               # true effect of T is 5

    naive = Y[T==1].mean() - Y[T==0].mean()
    # backdoor: adjust for Z (average within-stratum effects, weighted by P(Z))
    adj = 0.0
    for z in (0, 1):
        pz = (Z == z).mean()
        eff_z = Y[(T==1)&(Z==z)].mean() - Y[(T==0)&(Z==z)].mean()
        adj += pz * eff_z
    print(f"naive (confounded) estimate:  {naive:+.2f}")
    print(f"backdoor-adjusted estimate:   {adj:+.2f}   (true effect = 5)")
    assert abs(adj - 5.0) < 0.4 and abs(naive - 5.0) > 2
    print("OK — adjusting for the confounder recovers the true causal effect from observation alone.")
