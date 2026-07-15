"""
Track 19 · Lesson 3 — Data parallelism: many workers, one model
===============================================================
To train on huge data fast, split each batch across many WORKERS (GPUs/machines).
Each worker computes gradients on its slice; the gradients are AVERAGED and every
worker applies the same update. The maths is exact: the average of the shard
gradients equals the gradient on the full batch — so N workers give an N-times
bigger effective batch with identical results, just faster.

Run:  python parallelism.py
"""
import numpy as np
rng = np.random.default_rng(0)

def data(n=1200):
    X=rng.standard_normal((n,4)); w=np.array([1.5,-2.,1.,.5]); y=(X@w+0.3*rng.standard_normal(n)>0).astype(int); return X,y
def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-30,30)))
def grad(X,y,w,b): e=sigmoid(X@w+b)-y; return X.T@e/len(y), e.mean()

if __name__ == "__main__":
    X,y=data(); W=4
    shards=np.array_split(np.arange(len(X)), W)      # split the batch across 4 workers

    # single machine: gradient on the FULL batch
    gW_single, gb_single = grad(X, y, np.zeros(4), 0.0)

    # data-parallel: each worker's gradient on its shard, then AVERAGE
    gWs, gbs = [], []
    for s in shards:
        gW, gb = grad(X[s], y[s], np.zeros(4), 0.0); gWs.append(gW*len(s)); gbs.append(gb*len(s))
    gW_par = sum(gWs)/len(X); gb_par = sum(gbs)/len(X)

    print(f"single-machine gradient[0]: {gW_single[0]:.6f}")
    print(f"data-parallel gradient[0] : {gW_par[0]:.6f}   (averaged over {W} workers)")
    assert np.allclose(gW_single, gW_par) and np.isclose(gb_single, gb_par)
    print("OK — averaging worker gradients exactly equals the full-batch gradient. That's data parallelism.")
    print("   (Model parallelism instead splits ONE model across devices when it's too big to fit.)")
