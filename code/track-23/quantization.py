"""
Track 19 · Lesson 2 — Quantization: fewer bits per weight
=========================================================
Weights are usually stored as 32-bit floats, but that precision is overkill. QUANTIZATION
rounds each weight to one of a small number of levels — e.g., 256 levels for 8-bit — cutting
memory 4x (32→8 bits) and enabling fast integer arithmetic, with tiny accuracy loss.

Run:  python quantization.py
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

def quantize(W, bits):
    lo, hi = W.min(), W.max(); L = 2**bits - 1
    q = np.round((W - lo) / (hi - lo) * L)          # to integer levels
    return q / L * (hi - lo) + lo                    # back to float (dequantize)

if __name__ == "__main__":
    Xtr,ytr=data(900); Xte,yte=data(3000)
    net=MLP(); net.fit(Xtr,ytr)
    full=net.acc(Xte,yte); print(f"32-bit float accuracy: {full:.3f}   (memory 32 bits/weight)")
    W1o,W2o=net.W1.copy(),net.W2.copy()
    for bits in (8,4,2):
        net.W1=quantize(W1o,bits); net.W2=quantize(W2o,bits)
        print(f"  {bits}-bit weights → accuracy {net.acc(Xte,yte):.3f}   (memory {bits} bits/weight, {32//bits}x smaller)")
    net.W1=quantize(W1o,8); net.W2=quantize(W2o,8)
    assert abs(net.acc(Xte,yte) - full) < 0.02
    print("OK — 8-bit weights match 32-bit accuracy at 1/4 the memory. Precision was overkill.")
