"""
Track 16 · Lesson 3 — Catastrophic forgetting
=============================================
Train a network on task A, then on task B, and something alarming happens: it
FORGETS task A almost entirely. Neural nets overwrite the weights that solved the
old task while learning the new one. This "catastrophic forgetting" is the central
obstacle to continual (lifelong) learning. Here we watch a classifier's task-A
accuracy collapse as it learns task B.

Run:  python forgetting.py
"""
import numpy as np
rng = np.random.default_rng(0)

def blobs(centres, n=200):
    X, y = [], []
    for c, mu in enumerate(centres):
        X.append(mu + rng.standard_normal((n, 2))); y += [c]*n
    return np.vstack(X), np.array(y)

def relu(x): return np.maximum(0, x)
def softmax(z): z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)

class MLP:
    def __init__(s, d=2, h=64, k=2):
        s.W1=rng.standard_normal((d,h))*0.3; s.b1=np.zeros(h)
        s.W2=rng.standard_normal((h,k))*0.3; s.b2=np.zeros(k)
    def fwd(s,X): s.h=relu(X@s.W1+s.b1); return softmax(s.h@s.W2+s.b2)
    def train(s,X,y,epochs=250,lr=0.1):
        for _ in range(epochs):
            P=s.fwd(X); G=P.copy(); G[np.arange(len(y)),y]-=1; G/=len(y)
            gW2=s.h.T@G; gb2=G.sum(0); dh=(G@s.W2.T)*(s.h>0)
            gW1=X.T@dh; gb1=dh.sum(0)
            s.W1-=lr*gW1; s.b1-=lr*gb1; s.W2-=lr*gW2; s.b2-=lr*gb2
    def acc(s,X,y): return (s.fwd(X).argmax(1)==y).mean()

if __name__ == "__main__":
    # Task A (left region): "up = class 1".   Task B (right region): "up = class 0" (reversed).
    # Both are solvable together (different x regions), but learning B-only drifts the shared
    # weights and erases A unless A is rehearsed -> a clean catastrophic-forgetting demo.
    XA,yA = blobs([[-2,-2.5],[-2,2.5]])      # class0 = down-left, class1 = up-left
    XB,yB = blobs([[2,2.5],[2,-2.5]])        # class0 = up-right, class1 = down-right (reversed)
    net = MLP()
    net.train(XA,yA,epochs=400); a_before = net.acc(XA,yA)
    print(f"task A accuracy after learning A: {a_before:.3f}")
    net.train(XB,yB,epochs=400)               # now learn B, with NO access to A
    a_after = net.acc(XA,yA); b_after = net.acc(XB,yB)
    print(f"task B accuracy after learning B: {b_after:.3f}")
    print(f"task A accuracy AFTER learning B: {a_after:.3f}   <-- forgotten!")
    assert a_before > 0.95 and a_after < 0.5 and b_after > 0.95
    print("OK — learning B overwrote the weights that solved A. That's catastrophic forgetting.")
