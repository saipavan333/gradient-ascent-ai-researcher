"""
Track 18 · Lesson 1 — Confounding (and Simpson's paradox)
========================================================
A confounder distorts the apparent effect of a treatment. Worse, it can FLIP the
sign: a treatment that helps everyone can look harmful overall. Here a drug helps
patients in BOTH severity groups, but because sicker patients both get the drug more
AND have worse outcomes, the naive "treated vs untreated" comparison says the drug
hurts. That's Simpson's paradox — confounding at its most dramatic.

Run:  python confounding.py
"""
import numpy as np
rng = np.random.default_rng(0)

if __name__ == "__main__":
    n = 20000
    sick = (rng.random(n) < 0.5).astype(int)          # confounder: 1 = sick, 0 = healthy
    # sicker patients are much more likely to get the drug
    p_treat = np.where(sick == 1, 0.85, 0.15)
    T = (rng.random(n) < p_treat).astype(int)
    # TRUE effect: the drug ADDS 10 to recovery for everyone; sick patients start much lower
    base = np.where(sick == 1, 30.0, 70.0)
    Y = base + 10.0 * T + rng.standard_normal(n) * 3

    naive = Y[T == 1].mean() - Y[T == 0].mean()        # ignore the confounder
    within_sick = Y[(T==1)&(sick==1)].mean() - Y[(T==0)&(sick==1)].mean()
    within_healthy = Y[(T==1)&(sick==0)].mean() - Y[(T==0)&(sick==0)].mean()
    print(f"naive treated - untreated overall: {naive:+.1f}   (drug looks HARMFUL!)")
    print(f"effect within sick patients:       {within_sick:+.1f}   (drug helps)")
    print(f"effect within healthy patients:    {within_healthy:+.1f}   (drug helps)")
    assert naive < 0 and within_sick > 8 and within_healthy > 8
    print("OK — Simpson's paradox: the drug helps every subgroup, yet looks harmful overall.")
    print("   The confounder (severity) drove both who got the drug and the outcome.")
