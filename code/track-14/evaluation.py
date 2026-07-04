"""
Track 14 · Lesson 3 — Why average metrics hide failures (evaluation & red-teaming)
Run:  python evaluation.py

A single headline number can look great while the model fails catastrophically on an
important slice. Safety evaluation means DISAGGREGATING by subgroup and actively
RED-TEAMING for failure cases — not trusting the average. We build a classifier with a
high overall accuracy that nonetheless fails a minority subgroup.
"""
import numpy as np
rng = np.random.default_rng(0)

def make(n):
    # group A (90%): easy, label depends on x1.  group B (10%): rule is REVERSED.
    gA = rng.random(n) < 0.9
    x = rng.normal(0, 1, (n, 2)); group = np.where(gA, 0, 1)
    y = (x[:, 0] > 0).astype(int)
    y[~gA] = (x[~gA, 0] < 0).astype(int)          # minority follows the opposite rule
    return x, y, group

def sigmoid(z): return 1/(1+np.exp(-z))
def train(X, y):
    w = np.zeros(2); b = 0.0
    for _ in range(400):
        p = sigmoid(X@w+b); g = p-y; w -= 0.1*X.T@g/len(X); b -= 0.1*g.mean()
    return w, b

def main():
    X, y, group = make(4000); w, b = train(X, y)
    Xt, yt, gt = make(4000); pred = (sigmoid(Xt@w+b) > 0.5).astype(int)
    overall = (pred == yt).mean()
    print(f"Headline metric — overall test accuracy: {overall:.3f}   (looks great!)\n")
    print("But disaggregate by subgroup (what a safety eval does):")
    for g, name in [(0, "majority group A (90% of data)"), (1, "minority group B (10%)")]:
        m = gt == g
        print(f"   {name:32s} accuracy {(pred[m]==yt[m]).mean():.3f}")
    print("\nThe model is ~100% on the majority and ~0% on the minority — the average")
    print("(0.9) hides a total failure for group B. Averages conceal; slices and")
    print("red-teaming reveal. Evaluate the worst case, not just the mean (Track 9).")

if __name__ == "__main__":
    main()
