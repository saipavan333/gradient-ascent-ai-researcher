"""
Track 14 · Lesson 1 — Message passing: how GNNs move information
================================================================
The one idea behind every GNN: each node updates itself by AGGREGATING messages
from its neighbours. Do it once and a node knows about its 1-hop neighbourhood;
repeat and information spreads across the graph. Here we start a signal on ONE
node per community and let message passing carry each community's label outward —
pure diffusion classifies the whole graph, no training at all.

Run:  python message_passing.py
"""
import numpy as np
rng = np.random.default_rng(0)

def sbm(n_per=20, p_in=0.35, p_out=0.03):
    N = 2 * n_per; comm = np.array([0] * n_per + [1] * n_per)
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            p = p_in if comm[i] == comm[j] else p_out
            if rng.random() < p: A[i, j] = A[j, i] = 1
    return A, comm

if __name__ == "__main__":
    A, comm = sbm()
    N = len(A)
    Aself = A + np.eye(N)
    Dinv = np.diag(1.0 / Aself.sum(1))
    P = Dinv @ Aself                       # one message-passing step = average of neighbours

    # put +1 on one seed in community 0, -1 on one seed in community 1
    h = np.zeros(N)
    h[np.where(comm == 0)[0][0]] = 1.0
    h[np.where(comm == 1)[0][0]] = -1.0

    for step in range(8):                  # spread the signal outward
        h = P @ h
    pred = (h < 0).astype(int)             # sign of the diffused signal = predicted community
    acc = max((pred == comm).mean(), (pred != comm).mean())
    print(f"after 8 message-passing steps, sign of the signal matches community {acc*100:.0f}% of the time")
    assert acc > 0.8
    print("neighbours end up similar: message passing alone spread 2 labels across all", N, "nodes.")
    print("OK — aggregate-from-neighbours, repeated, is the engine inside every GNN.")
