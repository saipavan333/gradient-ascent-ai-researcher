"""
Track 17 · Lesson 4 — Secure aggregation
========================================
In FedAvg the server sees each client's update — which can leak information. SECURE
AGGREGATION lets the server learn ONLY the SUM of updates, never any individual one.
The trick: every pair of clients shares a random mask that is +s for one and −s for
the other. Each client adds all its masks to its update. The masks cancel in the
total, so the sum is exact — but any single masked update looks like random noise.

Run:  python secure_aggregation.py
"""
import numpy as np
rng = np.random.default_rng(0)

if __name__ == "__main__":
    N, d = 6, 5                                   # 6 clients, update dimension 5
    updates = rng.standard_normal((N, d))         # each client's private update

    # pairwise masks: S[i,j] = -S[j,i], drawn at random
    S = np.zeros((N, N, d))
    for i in range(N):
        for j in range(i+1, N):
            s = rng.standard_normal(d) * 5
            S[i, j] = s; S[j, i] = -s

    masked = np.array([updates[i] + S[i].sum(0) for i in range(N)])   # what each client SENDS

    true_sum = updates.sum(0)
    masked_sum = masked.sum(0)                     # server adds up the masked messages
    print("true sum of updates :", np.round(true_sum, 3))
    print("sum of masked msgs  :", np.round(masked_sum, 3))
    assert np.allclose(true_sum, masked_sum)       # masks cancel -> exact sum

    # a single masked message reveals nothing: it's far from the real update
    leak = np.linalg.norm(masked[0] - updates[0]) / np.linalg.norm(updates[0])
    print(f"a single masked message differs from the real update by {leak*100:.0f}% (looks like noise)")
    assert leak > 1.0
    print("OK — the server recovers the exact SUM but never sees any client's individual update.")
