"""
Track 19 (Efficient & Scalable ML) · Lesson 0 — Knowledge distillation
======================================================================
A big, accurate "teacher" model is expensive to run. DISTILLATION trains a small,
cheap "student" to imitate the teacher — not just its final answer, but its full
probability distribution (the "soft targets"), which carry rich information about how
classes relate. The student learns more from "80% cat, 15% dog, 5% fox" than from a
bare "cat". Here a distilled student beats a student trained on hard labels alone.

Run:  python distillation.py
"""
import numpy as np
rng = np.random.default_rng(0)

def data(n):
    C = np.array([[2,2],[-2,2],[0,-2.2]], float)
    X = np.vstack([c + rng.standard_normal((n//3,2))*1.3 for c in C])
    y = np.repeat(range(3), n//3); return X, y
def relu(x): return np.maximum(0,x)
def softmax(z, T=1.0): z=(z/T); z-=z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)

class MLP:
    def __init__(s, h): s.W1=rng.standard_normal((2,h))*0.4; s.b1=np.zeros(h); s.W2=rng.standard_normal((h,3))*0.4; s.b2=np.zeros(3)
    def logits(s,X): s.h=relu(X@s.W1+s.b1); return s.h@s.W2+s.b2
    def prob(s,X,T=1.0): return softmax(s.logits(X),T)
    def fit(s, X, target, epochs=300, lr=0.2, soft=False):
        for _ in range(epochs):
            L=s.logits(X); P=softmax(L)
            G = P - (target if soft else np.eye(3)[target]); G/=len(X)
            gW2=s.h.T@G; gb2=G.sum(0); dh=(G@s.W2.T)*(s.h>0); gW1=X.T@dh; gb1=dh.sum(0)
            s.W1-=lr*gW1; s.b1-=lr*gb1; s.W2-=lr*gW2; s.b2-=lr*gb2
    def acc(s,X,y): return (s.prob(X).argmax(1)==y).mean()

if __name__ == "__main__":
    Xtr,ytr = data(90)                       # LIMITED training data
    Xbig,ybig = data(3000)                   # teacher sees lots
    Xte,yte = data(3000)
    teacher = MLP(64); teacher.fit(Xbig, ybig, epochs=500)
    print(f"teacher (big, lots of data) accuracy: {teacher.acc(Xte,yte):.3f}")
    # student on HARD labels only (limited data)
    s_hard = MLP(8); s_hard.fit(Xtr, ytr, epochs=400)
    # student DISTILLED: match teacher's soft probabilities on the same limited data
    soft = teacher.prob(Xtr, T=3.0)
    s_dist = MLP(8); s_dist.fit(Xtr, soft, epochs=400, soft=True)
    ah, ad = s_hard.acc(Xte,yte), s_dist.acc(Xte,yte)
    print(f"small student — hard labels only: {ah:.3f}")
    print(f"small student — distilled:        {ad:.3f}")
    assert ad > ah
    print("OK — the student learned more from the teacher's soft targets than from hard labels alone.")
