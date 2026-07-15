"""
Track 17 · Lesson 3 — DP-SGD: training with a privacy guarantee
===============================================================
DP-SGD makes the WHOLE trained model differentially private. Two changes to normal
SGD: (1) CLIP each example's gradient to a max norm C (so no single example can
dominate), then (2) add Gaussian noise to the summed gradients before the step. The
model still learns, but with a formal guarantee that it didn't memorise any
individual. We compare plain SGD to DP-SGD.

Run:  python dp_sgd.py
"""
import numpy as np
rng = np.random.default_rng(0)

def data(n=1500):
    X=rng.standard_normal((n,4)); w=np.array([1.5,-2.,1.,.5])
    y=(X@w+0.3*rng.standard_normal(n)>0).astype(int); return X,y
def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-30,30)))
def acc(X,y,w,b): return ((sigmoid(X@w+b)>0.5).astype(int)==y).mean()

def train(X,y,dp=False,C=1.0,sigma=1.0,steps=300,lr=0.2):
    w=np.zeros(X.shape[1]); b=0.0
    for _ in range(steps):
        p=sigmoid(X@w+b); e=(p-y)
        gw=X*e[:,None]; gb=e                               # per-example gradients
        if dp:
            norms=np.sqrt((gw**2).sum(1)+gb**2)+1e-9
            clip=np.minimum(1.0, C/norms)                  # (1) clip each example's gradient
            gw=gw*clip[:,None]; gb=gb*clip
            noise_w=rng.normal(0, sigma*C, gw.shape[1])    # (2) add Gaussian noise to the sum
            noise_b=rng.normal(0, sigma*C)
            gW=(gw.sum(0)+noise_w)/len(y); gB=(gb.sum()+noise_b)/len(y)
        else:
            gW=gw.mean(0); gB=gb.mean()
        w-=lr*gW; b-=lr*gB
    return w,b

if __name__ == "__main__":
    X,y=data(); Xte,yte=data(3000)
    w0,b0=train(X,y,dp=False)
    w1,b1=train(X,y,dp=True,C=1.0,sigma=1.0)
    a0,a1=acc(Xte,yte,w0,b0), acc(Xte,yte,w1,b1)
    print(f"plain SGD accuracy:  {a0:.3f}")
    print(f"DP-SGD accuracy:     {a1:.3f}   (clip + noise -> privacy, slight accuracy cost)")
    assert a1 > 0.75 and a0 >= a1 - 0.02
    print("OK — DP-SGD still learns a good model while provably limiting what it memorises.")
