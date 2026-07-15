"""
Track 19 · Lesson 1 — Pruning: most weights don't matter
========================================================
Trained networks are hugely over-parameterised: many weights are near zero and can
be DELETED with little accuracy loss. Magnitude pruning zeroes out the smallest
weights, making the network sparse (and smaller/faster). Here we prune increasing
fractions and watch accuracy hold up until very high sparsity.

Run:  python pruning.py
"""
import numpy as np
rng = np.random.default_rng(0)

def data(n):
    C=np.array([[2,2],[-2,2],[0,-2.2]],float); X=np.vstack([c+rng.standard_normal((n//3,2)) for c in C])
    return X, np.repeat(range(3),n//3)
def relu(x): return np.maximum(0,x)
def softmax(z): z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)

class MLP:
    def __init__(s,h=64): s.W1=rng.standard_normal((2,h))*0.4;s.b1=np.zeros(h);s.W2=rng.standard_normal((h,3))*0.4;s.b2=np.zeros(3)
    def fwd(s,X): s.h=relu(X@s.W1+s.b1); return softmax(s.h@s.W2+s.b2)
    def fit(s,X,y,epochs=400,lr=0.2):
        for _ in range(epochs):
            P=s.fwd(X); G=P-np.eye(3)[y]; G/=len(X)
            gW2=s.h.T@G;gb2=G.sum(0);dh=(G@s.W2.T)*(s.h>0);gW1=X.T@dh;gb1=dh.sum(0)
            s.W1-=lr*gW1;s.b1-=lr*gb1;s.W2-=lr*gW2;s.b2-=lr*gb2
    def acc(s,X,y): return (s.fwd(X).argmax(1)==y).mean()

def prune(W, frac):
    thr = np.quantile(np.abs(W), frac)
    return W * (np.abs(W) >= thr)

if __name__ == "__main__":
    Xtr,ytr=data(900); Xte,yte=data(3000)
    net=MLP(); net.fit(Xtr,ytr)
    full=net.acc(Xte,yte); print(f"dense network accuracy: {full:.3f}")
    W1o,W2o=net.W1.copy(),net.W2.copy()
    for frac in (0.5,0.7,0.9):
        net.W1=prune(W1o,frac); net.W2=prune(W2o,frac)
        print(f"  pruned {int(frac*100)}% of weights → accuracy {net.acc(Xte,yte):.3f}")
    net.W1=prune(W1o,0.7); net.W2=prune(W2o,0.7)
    assert net.acc(Xte,yte) > full - 0.05
    print("OK — deleting 70% of the weights barely dented accuracy. Trained nets are very redundant.")
