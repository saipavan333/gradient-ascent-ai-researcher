"""
Track 20 (Frontier Architectures & Agents) · Lesson 0 — Mixture of Experts
==========================================================================
A dense network runs ALL its weights on EVERY input. A Mixture of Experts (MoE)
instead has many specialist "expert" sub-networks and a "gating" network that
routes each input to just a few of them. So the model can be huge (many experts)
but only a small part runs per input — big capacity, small compute. Here a gate
learns to send each region of the data to its own expert.

Run:  python mixture_of_experts.py
"""
import numpy as np
rng = np.random.default_rng(0)

def data(n=1200):
    C = np.array([[3,3],[-3,3],[-3,-3],[3,-3]], float)     # 4 regions
    X = np.vstack([c + rng.standard_normal((n//4,2))*0.8 for c in C])
    y = np.repeat(range(4), n//4); return X, y
def softmax(z): z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)

if __name__ == "__main__":
    X, y = data(); Xte, yte = data(2000); E = 4
    # gate: routes input -> distribution over experts;  experts: linear classifiers
    Wg = rng.standard_normal((2, E))*0.1; bg = np.zeros(E)
    We = rng.standard_normal((E, 2, 4))*0.1; be = np.zeros((E, 4))
    for ep in range(400):
        g = softmax(X @ Wg + bg)                            # (n,E) routing weights
        outs = np.stack([softmax(X @ We[e] + be[e]) for e in range(E)], 1)  # (n,E,4)
        P = (g[:, :, None] * outs).sum(1)                   # mixture output
        d = (P - np.eye(4)[y]) / len(X)
        for e in range(E):
            ge = g[:, e][:, None]
            We[e] -= 0.5 * X.T @ (ge * d); be[e] -= 0.5 * (ge * d).sum(0)
        # gate gradient (which expert helped?)
        adv = np.stack([( -(np.log(outs[:,e]+1e-9)*np.eye(4)[y]).sum(1)) for e in range(E)],1)
        dg = g * (adv - (g*adv).sum(1,keepdims=True)); Wg -= 0.3 * X.T @ dg / len(X); bg -= 0.3 * dg.mean(0)
    # top-1 routing at test: only the single best expert runs per input
    gate = softmax(Xte @ Wg + bg); pick = gate.argmax(1)
    pred = np.array([(Xte[i] @ We[pick[i]] + be[pick[i]]).argmax() for i in range(len(Xte))])
    acc = (pred == yte).mean()
    # did experts specialise? each region should mostly route to one expert
    routing = [np.bincount(pick[yte==c], minlength=E).argmax() for c in range(4)]
    print(f"top-1 MoE accuracy (only 1 of {E} experts runs per input): {acc:.3f}")
    print(f"each class routes to expert: {routing}  (specialised = {len(set(routing))} distinct experts)")
    assert acc > 0.9 and len(set(routing)) >= 3
    print("OK — the gate learned to route each region to its own expert: big model, small per-input compute.")
