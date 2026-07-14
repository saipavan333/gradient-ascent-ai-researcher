"""
Track 16 · Lesson 1 — MAML: learning an initialisation that adapts fast
=======================================================================
Meta-learning is "learning to learn". MAML finds a network INITIALISATION such
that, for any new task, a few gradient steps from that init already solve it. We
meta-train on many sine-wave regression tasks (random amplitude & phase) with the
first-order Reptile algorithm, then show the meta-init adapts to a brand-new sine
in a handful of steps — far better than starting from a random init.

Run:  python maml.py
"""
import numpy as np
rng = np.random.default_rng(0)

def net_init():
    s = 0.3
    return [rng.standard_normal((1,40))*s, np.zeros(40),
            rng.standard_normal((40,40))*s, np.zeros(40),
            rng.standard_normal((40,1))*s, np.zeros(1)]

def forward(P, x):
    W1,b1,W2,b2,W3,b3 = P
    h1 = np.tanh(x@W1+b1); h2 = np.tanh(h1@W2+b2); return h2@W3+b3, (x,h1,h2)

def grads(P, x, y):
    W1,b1,W2,b2,W3,b3 = P
    out,(x_,h1,h2) = forward(P,x); d = 2*(out-y)/len(x)
    gW3 = h2.T@d; gb3 = d.sum(0)
    dh2 = (d@W3.T)*(1-h2**2); gW2 = h1.T@dh2; gb2 = dh2.sum(0)
    dh1 = (dh2@W2.T)*(1-h1**2); gW1 = x_.T@dh1; gb1 = dh1.sum(0)
    return [gW1,gb1,gW2,gb2,gW3,gb3]

def sgd_steps(P, x, y, steps, lr):
    P = [w.copy() for w in P]
    for _ in range(steps):
        g = grads(P,x,y)
        for i in range(len(P)): P[i] -= lr*g[i]
    return P

def task():
    A = rng.uniform(0.5,3.0); ph = rng.uniform(0,np.pi)
    return A, ph
def sample(A, ph, n):
    x = rng.uniform(-5,5,(n,1)); return x, A*np.sin(x+ph)

def loss(P, x, y): out,_ = forward(P,x); return ((out-y)**2).mean()

if __name__ == "__main__":
    P = net_init()
    # --- Reptile meta-training: nudge init toward each task's fine-tuned weights ---
    for it in range(3000):
        A,ph = task(); x,y = sample(A,ph,20)
        Pk = sgd_steps(P, x, y, steps=8, lr=0.02)
        for i in range(len(P)): P[i] += 0.05*(Pk[i]-P[i])       # meta-update

    # --- test on a NEW task: adapt from meta-init vs from random init ---
    meta_before = meta_after = rand_after = 0.0; T = 40
    for _ in range(T):
        A,ph = task(); xs,ys = sample(A,ph,10); xq,yq = sample(A,ph,100)
        meta_before += loss(P, xq, yq)
        Pm = sgd_steps(P, xs, ys, steps=5, lr=0.02); meta_after += loss(Pm, xq, yq)
        Pr = sgd_steps(net_init(), xs, ys, steps=5, lr=0.02); rand_after += loss(Pr, xq, yq)
    meta_before/=T; meta_after/=T; rand_after/=T
    print(f"query loss BEFORE adapting (meta-init):      {meta_before:.3f}")
    print(f"query loss after 5 steps from META init:     {meta_after:.3f}")
    print(f"query loss after 5 steps from RANDOM init:   {rand_after:.3f}")
    assert meta_after < rand_after * 0.6
    print("OK — the meta-learned initialisation adapts to a new task far faster than random.")
