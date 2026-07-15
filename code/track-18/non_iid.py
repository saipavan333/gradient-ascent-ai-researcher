"""
Track 17 · Lesson 1 — The non-IID challenge
===========================================
FedAvg assumes each client's data looks like everyone else's. In reality it doesn't:
your phone's photos aren't like mine; one hospital sees different patients than another.
When clients hold SKEWED, conflicting slices of the data and train a lot locally before
syncing, their models drift to different local solutions, and averaging them clashes.
Here the true rule is XOR (needs a non-linear model). IID clients each see the whole
pattern and FedAvg works; non-IID clients each see only one half — where the local rule
is the OPPOSITE — so their averaged model is far worse.

Run:  python non_iid.py
"""
import numpy as np
rng = np.random.default_rng(0)

def xor(n=4000):
    X = rng.uniform(-3, 3, (n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)      # checkerboard: needs a non-linear boundary
    return X, y

def relu(x): return np.maximum(0, x)
def softmax(z): z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
def init():
    r = np.random.default_rng(3)
    return [r.standard_normal((2,24))*0.7, np.zeros(24), r.standard_normal((24,2))*0.7, np.zeros(2)]
def fwd(P, X): h = relu(X@P[0]+P[1]); return softmax(h@P[2]+P[3]), h
def train_local(P, X, y, steps=200, lr=0.15):
    P = [w.copy() for w in P]
    for _ in range(steps):
        Pr, h = fwd(P, X); G = Pr.copy(); G[np.arange(len(y)), y] -= 1; G /= len(y)
        gW2 = h.T@G; gb2 = G.sum(0); dh = (G@P[2].T)*(h>0); gW1 = X.T@dh; gb1 = dh.sum(0)
        P[0]-=lr*gW1; P[1]-=lr*gb1; P[2]-=lr*gW2; P[3]-=lr*gb2
    return P
def acc(P, X, y): return (fwd(P, X)[0].argmax(1) == y).mean()
def fedavg(shards, X, y, rounds=2, steps=200):
    P = init()
    for _ in range(rounds):
        Ps = [train_local(P, X[s], y[s], steps) for s in shards if len(s) > 3]
        P = [np.mean([Q[i] for Q in Ps], 0) for i in range(4)]     # average the client models
    return P

if __name__ == "__main__":
    X, y = xor(4000); Xte, yte = xor(3000)
    iid    = np.array_split(rng.permutation(len(X)), 2)            # each client: the whole pattern
    noniid = [np.where(X[:,0] > 0)[0], np.where(X[:,0] < 0)[0]]    # each client: one half (opposite local rule)
    ai = acc(fedavg(iid, X, y), Xte, yte)
    an = acc(fedavg(noniid, X, y), Xte, yte)
    print(f"FedAvg accuracy — IID clients (balanced):        {ai:.3f}")
    print(f"FedAvg accuracy — non-IID clients (skewed halves): {an:.3f}")
    print(f"the non-IID penalty from client drift: {ai-an:.3f}")
    assert ai > an + 0.08
    print("OK — skewed client data makes locally-trained models clash; averaging them loses accuracy.")
    print("   (mitigations: sync more often, or correct the drift — FedProx, SCAFFOLD.)")
