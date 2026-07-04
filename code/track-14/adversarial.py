"""
Track 14 · Lesson 4 — Adversarial examples from scratch (FGSM)
Run:  python adversarial.py

A tiny, carefully-chosen perturbation — far too small to notice — can flip a confident
classifier to the opposite class. This is a core AI-safety/robustness finding. We train
a logistic-regression classifier, then use the Fast Gradient Sign Method: nudge the
input along the SIGN of the loss gradient. In high dimensions, many tiny nudges add up.
"""
import numpy as np
rng = np.random.default_rng(0)

d = 40
def data(n):
    a = rng.normal(-0.4, 1, (n, d)); b = rng.normal(0.4, 1, (n, d))
    X = np.vstack([a, b]); y = np.r_[np.zeros(n), np.ones(n)]; return X, y

def sigmoid(z): return 1/(1+np.exp(-z))

def train(X, y, epochs=300, lr=0.1):
    w = np.zeros(d); b = 0.0
    for _ in range(epochs):
        p = sigmoid(X@w + b); g = p - y
        w -= lr * X.T@g / len(X); b -= lr * g.mean()
    return w, b

def main():
    X, y = data(400); w, b = train(X, y)
    acc = ((sigmoid(X@w+b) > 0.5) == y).mean()
    print(f"Clean test accuracy: {acc:.3f}\n")
    # pick a CORRECTLY and confidently (~0.9) classified point of class 1
    idx = np.where(y == 1)[0]; probs = sigmoid(X[idx]@w+b)
    x = X[idx[np.argmin(np.abs(probs - 0.92))]].copy()
    conf0 = sigmoid(x@w+b)
    print(f"A test point classified as class 1 with confidence {conf0:.2f}. Attack it:\n")
    # FGSM: to LOWER the prob of class 1, step against the gradient => x - eps*sign(w)
    for eps in [0.0, 0.1, 0.2, 0.35]:
        x_adv = x - eps*np.sign(w)                       # ≤ eps change per feature
        conf = sigmoid(x_adv@w+b); L2 = np.linalg.norm(x_adv - x)
        flip = "  <-- FLIPPED to class 0!" if conf < 0.5 else ""
        print(f"  ε={eps:.2f}/feature (‖δ‖₂={L2:4.1f} vs ‖x‖₂={np.linalg.norm(x):4.1f})  ->  P(class 1)={conf:.3f}{flip}")
    print(f"\nEach feature moves by ≤0.35 (small vs its spread ~1.0), yet the confident")
    print(f"prediction collapses to the WRONG class. In high dimensions, many tiny")
    print(f"aligned nudges sum to a decisive push across the boundary.")

if __name__ == "__main__":
    main()
