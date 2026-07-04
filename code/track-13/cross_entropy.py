"""
Track 13 · Lesson 1 — Cross-entropy: the loss you've been using all along
Run:  python cross_entropy.py

Cross-entropy H(p,q) = -Σ p(x) log q(x) measures the average surprise of using the
WRONG distribution q when the truth is p. It is THE loss for classification: your
model outputs q (predicted class probabilities), the label gives p, and minimizing
cross-entropy is exactly maximum likelihood (Track 1). Here we see why.
"""
import numpy as np

def cross_entropy(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float(-(p * np.log2(q + 1e-12)).sum())

def entropy(p):
    p = np.asarray(p, float); p = p[p > 0]; return float(-(p*np.log2(p)).sum())

def main():
    true = [1, 0, 0]                                   # one-hot: the answer is class 0
    print("One example, true class = 0.  Cross-entropy of different predictions:\n")
    for name, q in {"confident & right [.9,.05,.05]":[.9,.05,.05],
                    "unsure           [.34,.33,.33]":[.34,.33,.33],
                    "confident & WRONG [.05,.9,.05]":[.05,.9,.05]}.items():
        print(f"  {name:32s} CE = {cross_entropy(true, q):.3f} bits")
    print("\n  -> confident-correct is cheap; confident-wrong is heavily penalized.")

    # Cross-entropy is minimized when q = p, and its minimum value is H(p).
    p = np.array([0.6, 0.3, 0.1])
    print(f"\nTrue distribution p = {p}")
    print(f"  CE(p, q=p) = {cross_entropy(p, p):.3f}  equals  H(p) = {entropy(p):.3f}  (the floor)")
    for q in [np.array([0.33,0.33,0.34]), np.array([0.5,0.3,0.2])]:
        print(f"  CE(p, q={q}) = {cross_entropy(p, q):.3f}  > H(p)   (using the wrong q always costs more)")
    print("\nMinimizing cross-entropy drives your model's q toward the true p — that's training.")

if __name__ == "__main__":
    main()
