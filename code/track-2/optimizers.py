"""
Track 2 · Lesson 3 — Momentum, RMSProp & Adam
Run:  python optimizers.py
On an ill-conditioned 'ravine', plain GD stalls. Momentum, RMSProp, and Adam
each fix it; Adam (momentum + RMSProp + bias correction) is the deep-learning default.
"""
import numpy as np

# Ill-conditioned bowl: very steep in y, shallow in x. Minimum at (0,0).
A = np.array([1.0, 100.0])
grad = lambda p: 2 * A * p
loss = lambda p: float(np.sum(A * p**2))


def optimize(kind, steps=80):
    p = np.array([-4.0, 1.0]); v = np.zeros(2); s = np.zeros(2); m = np.zeros(2)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        g = grad(p)
        if kind == "gd":
            p = p - 0.009 * g
        elif kind == "momentum":
            v = 0.9 * v + g
            p = p - 0.0018 * v
        elif kind == "rmsprop":
            s = 0.9 * s + 0.1 * g * g
            p = p - 0.1 * g / (np.sqrt(s) + eps)
        elif kind == "adam":
            m = b1*m + (1-b1)*g
            v = b2*v + (1-b2)*g*g
            mh = m / (1 - b1**t)
            vh = v / (1 - b2**t)
            p = p - 0.25 * mh / (np.sqrt(vh) + eps)
    return loss(p)


def main():
    results = {k: optimize(k) for k in ["gd", "momentum", "rmsprop", "adam"]}
    for k, v in results.items():
        print(f"{k:9s} final loss: {v:.5f}")
    best = sorted(results, key=results.get)
    print("ranking (best first):", best)
    assert results["gd"] > 10 * results["adam"], "GD should stall far above Adam"
    print("confirmed: plain GD stalls far above the adaptive methods")

    # --- Try it yourself -----------------------------------------------------
    # 1) With beta=0.9 the momentum window is ~1/(1-beta)=10 steps. Try beta=0.99.


if __name__ == "__main__":
    main()
