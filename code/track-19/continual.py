"""
Track 16 · Lesson 4 — Continual learning: replay beats forgetting
=================================================================
The fix for catastrophic forgetting (Lesson 3): don't let the network see only the
new task. REPLAY keeps a small memory buffer of old-task examples and mixes them
into training on the new task, so the weights that solve A are rehearsed while B is
learned. We compare naive sequential training (forgets A) with replay (keeps A).

Run:  python continual.py
"""
import numpy as np
rng = np.random.default_rng(0)

def blobs(centres, n=200):
    X,y=[],[]
    for c,mu in enumerate(centres): X.append(mu+rng.standard_normal((n,2))); y+=[c]*n
    return np.vstack(X), np.array(y)
def relu(x): return np.maximum(0,x)
def softmax(z): z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)

class MLP:
    def __init__(s,d=2,h=64,k=2):
        s.W1=rng.standard_normal((d,h))*0.3; s.b1=np.zeros(h)
        s.W2=rng.standard_normal((h,k))*0.3; s.b2=np.zeros(k)
    def fwd(s,X): s.h=relu(X@s.W1+s.b1); return softmax(s.h@s.W2+s.b2)
    def step(s,X,y,lr=0.1):
        P=s.fwd(X); G=P.copy(); G[np.arange(len(y)),y]-=1; G/=len(y)
        gW2=s.h.T@G; gb2=G.sum(0); dh=(G@s.W2.T)*(s.h>0); gW1=X.T@dh; gb1=dh.sum(0)
        s.W1-=lr*gW1; s.b1-=lr*gb1; s.W2-=lr*gW2; s.b2-=lr*gb2
    def train(s,X,y,epochs=250):
        for _ in range(epochs): s.step(X,y)
    def acc(s,X,y): return (s.fwd(X).argmax(1)==y).mean()

if __name__ == "__main__":
    XA,yA = blobs([[-2,-2.5],[-2,2.5]]); XB,yB = blobs([[2,2.5],[2,-2.5]])  # reversed rule, overlapping-ish
    # --- naive: A then B ---
    n1=MLP(); n1.train(XA,yA,epochs=400); n1.train(XB,yB,epochs=400)
    naive_A = n1.acc(XA,yA)
    # --- replay: keep 40 A-examples, interleave while training B ---
    idx = rng.choice(len(XA), 40, replace=False); bufX,bufY = XA[idx], yA[idx]
    n2=MLP(); n2.train(XA,yA,epochs=400)
    for _ in range(400):
        n2.step(np.vstack([XB,bufX]), np.concatenate([yB,bufY]))   # B + replayed A
    replay_A = n2.acc(XA,yA); replay_B = n2.acc(XB,yB)
    print(f"task A accuracy after B — naive:  {naive_A:.3f}   (forgotten)")
    print(f"task A accuracy after B — replay: {replay_A:.3f}   (retained)")
    print(f"task B accuracy after B — replay: {replay_B:.3f}")
    assert replay_A > naive_A + 0.2 and replay_A > 0.85 and replay_B > 0.85
    print("OK — rehearsing a few old examples keeps task A alive while learning B.")
