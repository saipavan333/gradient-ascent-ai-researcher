"""
Track 18 (Causal Inference) · Lesson 0 — Correlation vs causation
=================================================================
Ice-cream sales and drownings rise together — but ice cream doesn't cause drowning;
hot weather causes both. A CONFOUNDER (a common cause) makes two unrelated things
move together. Here Z drives both X and Y; X and Y are strongly correlated even
though neither causes the other. Controlling for Z makes the correlation vanish.

Run:  python correlation_vs_causation.py
"""
import numpy as np
rng = np.random.default_rng(0)

def corr(a, b): return np.corrcoef(a, b)[0, 1]

if __name__ == "__main__":
    n = 5000
    Z = rng.standard_normal(n)                 # hidden common cause (e.g., temperature)
    X = 2.0 * Z + rng.standard_normal(n)       # X caused by Z only (ice-cream sales)
    Y = 2.0 * Z + rng.standard_normal(n)       # Y caused by Z only (drownings) — NO X->Y link

    print(f"correlation(X, Y) = {corr(X, Y):.3f}   (looks like a strong relationship!)")
    # control for Z: regress each on Z, correlate the residuals (partial correlation)
    bx = np.polyfit(Z, X, 1); rx = X - (bx[0]*Z + bx[1])
    by = np.polyfit(Z, Y, 1); ry = Y - (by[0]*Z + by[1])
    print(f"partial correlation given Z = {corr(rx, ry):.3f}   (vanishes — it was all Z)")
    assert abs(corr(X, Y)) > 0.6 and abs(corr(rx, ry)) < 0.1
    print("OK — a common cause fakes a correlation; controlling for it reveals there's no real link.")
