"""
Track 12 · Lesson 4 — Support Vector Machine + the kernel trick, from scratch
Run:  python svm.py

An SVM finds the separating line with the widest MARGIN — the biggest gap to the
nearest points (the 'support vectors'). We train a soft-margin linear SVM by
subgradient descent on the hinge loss, then show the KERNEL TRICK: mapping data to a
higher dimension lets a linear SVM separate data that isn't linearly separable.
"""
import numpy as np
rng = np.random.default_rng(0)

def blobs(n):                              # two linearly-separable-ish clusters
    a = rng.normal([-1.2, -1.2], 0.6, (n, 2)); b = rng.normal([1.2, 1.2], 0.6, (n, 2))
    X = np.vstack([a, b]); y = np.r_[-np.ones(n), np.ones(n)]; return X, y

def circles(n):                            # NOT linearly separable
    r0 = rng.uniform(0, 1, n); t0 = rng.uniform(0, 2*np.pi, n)
    r1 = rng.uniform(2, 3, n); t1 = rng.uniform(0, 2*np.pi, n)
    A = np.c_[r0*np.cos(t0), r0*np.sin(t0)]; B = np.c_[r1*np.cos(t1), r1*np.sin(t1)]
    X = np.vstack([A, B]); y = np.r_[-np.ones(n), np.ones(n)]; return X, y

def train_svm(X, y, C=1.0, lr=0.01, epochs=300):
    """Soft-margin linear SVM: minimize ½‖w‖² + C·Σ hinge(y(w·x+b))."""
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(epochs):
        margin = y * (X @ w + b)
        mask = margin < 1                              # points inside the margin / misclassified
        grad_w = w - C * (X[mask] * y[mask, None]).sum(0)   # subgradient of hinge
        grad_b = -C * y[mask].sum()
        w -= lr * grad_w / len(X); b -= lr * grad_b / len(X)
    return w, b

def acc(X, y, w, b): return (np.sign(X @ w + b) == y).mean()

def phi(X):                                # kernel-trick feature map: add radius^2
    return np.c_[X, (X**2).sum(1)]

def main():
    X, y = blobs(100)
    w, b = train_svm(X, y)
    print(f"Linear SVM on separable blobs: accuracy {acc(X, y, w, b):.3f}, margin width {2/np.linalg.norm(w):.2f}\n")
    Xc, yc = circles(100)
    wl, bl = train_svm(Xc, yc)
    print(f"Linear SVM on concentric circles: accuracy {acc(Xc, yc, wl, bl):.3f}  (linear fails — not separable)")
    Xp = phi(Xc); wp, bp = train_svm(Xp, yc)
    print(f"SVM on kernel-mapped circles (add x²+y²): accuracy {acc(Xp, yc, wp, bp):.3f}  (now separable!)")
    print("\nThe kernel trick: a linear boundary in a lifted feature space is a")
    print("curved boundary in the original space — power without leaving linear algebra.")

if __name__ == "__main__":
    main()
