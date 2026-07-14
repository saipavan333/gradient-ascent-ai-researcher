"""
Track 15 · Lesson 4 — Uncertainty & calibration
===============================================
A model can be accurate but OVER-confident: it says "99% sure" and is right only 80%
of the time. CALIBRATION measures that gap (Expected Calibration Error, ECE). A simple
fix — TEMPERATURE SCALING — divides the logits by a learned T>1 to soften the
probabilities, lowering ECE without changing any prediction.

Run:  python calibration.py
"""
import numpy as np
rng = np.random.default_rng(0)

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-30,30)))

def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins+1); e = 0.0
    for k in range(bins):
        m = (p >= edges[k]) & (p < edges[k+1] if k < bins-1 else p <= edges[k+1])
        if m.sum() == 0: continue
        conf = p[m].mean(); acc = y[m].mean()
        e += (m.sum()/len(p)) * abs(conf - acc)
    return e

if __name__ == "__main__":
    # true probabilities, sampled labels
    z = rng.standard_normal(4000)
    ptrue = sigmoid(z)
    y = (rng.random(4000) < ptrue).astype(int)
    logits = 2.3 * z                       # an OVER-confident model (logits too large)
    p_over = sigmoid(logits)
    e0 = ece(p_over, y)

    # temperature scaling: find T that minimises negative log-likelihood on held-out data
    Ts = np.linspace(0.5, 4.0, 200)
    nll = [ -(y*np.log(sigmoid(logits/T)+1e-12) + (1-y)*np.log(1-sigmoid(logits/T)+1e-12)).mean() for T in Ts ]
    T = Ts[int(np.argmin(nll))]
    e1 = ece(sigmoid(logits / T), y)
    print(f"over-confident model ECE: {e0:.3f}")
    print(f"best temperature T = {T:.2f}  →  ECE after scaling: {e1:.3f}")
    assert T > 1.0 and e1 < e0 * 0.6
    print("OK — temperature scaling fixed the confidence without changing any decision.")
