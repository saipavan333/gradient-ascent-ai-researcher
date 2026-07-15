"""
Track 20 · Lesson 1 — State-space models (Mamba-style)
======================================================
Transformers are powerful but cost O(n^2) in sequence length. STATE-SPACE MODELS
(SSMs) process a sequence with a linear recurrence — h_t = a·h_{t-1} + b·x_t,
y_t = c·h_t — in O(n) time and O(1) memory per step, while still capturing very
long-range dependencies (a≈1 means the state barely forgets). Mamba adds
"selectivity" (a, b depend on the input). Here an SSM remembers a signal from the
very start of a long sequence.

Run:  python state_space.py
"""
import numpy as np
rng = np.random.default_rng(0)

def ssm_scan(x, a, b, c):
    """Linear recurrence — the O(n) 'scan'. Returns y_t for all t."""
    h = 0.0; ys = []
    for t in range(len(x)):
        h = a * h + b * x[t]       # update state (a<1 forgets slowly)
        ys.append(c * h)
    return np.array(ys)

if __name__ == "__main__":
    T = 200
    # long-memory test: a pulse at t=0, then silence. Does y_T still 'see' it?
    corr = []
    for _ in range(300):
        x = np.zeros(T); x[0] = rng.choice([-1.0, 1.0])      # information only at the start
        y = ssm_scan(x, a=0.995, b=1.0, c=1.0)
        corr.append(np.sign(y[-1]) == np.sign(x[0]))         # is the sign preserved 200 steps later?
    keep = np.mean(corr)
    print(f"with a=0.995: the sign of the t=0 pulse survives to t={T} in {keep*100:.0f}% of runs")
    # a forgetful SSM loses it fast
    corr2 = []
    for _ in range(300):
        x = np.zeros(T); x[0] = rng.choice([-1.0,1.0]); x[1:] = 0.3*rng.standard_normal(T-1)
        y = ssm_scan(x, a=0.5, b=1.0, c=1.0)                  # a=0.5 forgets quickly
        corr2.append(np.sign(y[-1]) == np.sign(x[0]))
    forget = np.mean(corr2)
    print(f"with a=0.5 (forgetful): only {forget*100:.0f}% — the early signal has decayed away")
    print(f"cost: O(n) time, O(1) state — vs a Transformer's O(n^2) attention")
    assert keep > 0.95 and keep > forget + 0.3
    print("OK — a near-1 decay lets an SSM carry information across a very long sequence, cheaply.")
