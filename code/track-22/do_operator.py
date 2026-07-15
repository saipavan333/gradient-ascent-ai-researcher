"""
Track 18 · Lesson 2 — The do-operator: seeing vs doing
======================================================
P(Y | X=x) is SEEING: "among those who happen to have X=x, what's Y?" P(Y | do(X=x))
is DOING: "if we INTERVENE and set X=x for everyone, what's Y?" They differ whenever
a confounder is present. The do-operator is defined by a surgery on the causal model:
delete the arrows INTO X (cut its normal causes) and set X by fiat, then read off Y.

Run:  python do_operator.py
"""
import numpy as np
rng = np.random.default_rng(0)

def scm(n, force_T=None):
    """Structural causal model:  Z -> T,  Z -> Y,  T -> Y."""
    Z = (rng.random(n) < 0.5).astype(int)
    if force_T is None:
        T = (rng.random(n) < np.where(Z == 1, 0.8, 0.2)).astype(int)   # observational: Z drives T
    else:
        T = np.full(n, force_T)                                        # do(T): cut Z->T, set T
    Y = 20.0*Z + 5.0*T + rng.standard_normal(n)*2
    return Z, T, Y

if __name__ == "__main__":
    n = 40000
    # observational: P(Y | T=1) — condition (SEE)
    Z, T, Y = scm(n)
    see_1 = Y[T == 1].mean(); see_0 = Y[T == 0].mean()
    # interventional: P(Y | do(T=1)) — force T for everyone (DO)
    _, _, Ydo1 = scm(n, force_T=1); _, _, Ydo0 = scm(n, force_T=0)
    do_1 = Ydo1.mean(); do_0 = Ydo0.mean()
    print(f"SEEING:  E[Y | T=1] - E[Y | T=0] = {see_1 - see_0:+.2f}   (mixes in the confounder)")
    print(f"DOING:   E[Y | do T=1] - E[Y | do T=0] = {do_1 - do_0:+.2f}   (the true causal effect ≈ 5)")
    assert abs((do_1 - do_0) - 5.0) < 0.5 and (see_1 - see_0) > (do_1 - do_0) + 3
    print("OK — seeing ≠ doing under confounding; the do-operator gives the real intervention effect.")
