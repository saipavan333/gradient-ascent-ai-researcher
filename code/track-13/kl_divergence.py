"""
Track 13 · Lesson 2 — KL divergence: distance between distributions
Run:  python kl_divergence.py

KL divergence  KL(p‖q) = Σ p(x) log₂(p(x)/q(x))  measures how many EXTRA bits you pay
by encoding data from p using a code optimized for q. It's ≥ 0, zero only when p=q,
and NOT symmetric. You've already used it: the VAE's regularizer (Track 6) and RLHF's
penalty (Track 7) are both KL terms. Here we build the intuition.
"""
import numpy as np

def kl(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = p > 0
    return float((p[m] * np.log2(p[m] / (q[m] + 1e-12))).sum())

def entropy(p):
    p = np.asarray(p, float); p = p[p > 0]; return float(-(p*np.log2(p)).sum())
def cross_entropy(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float); return float(-(p*np.log2(q+1e-12)).sum())

def main():
    p = np.array([0.7, 0.2, 0.1]); q = np.array([0.4, 0.4, 0.2])
    print(f"p = {p}\nq = {q}\n")
    print(f"  KL(p‖q) = {kl(p,q):.3f} bits")
    print(f"  KL(q‖p) = {kl(q,p):.3f} bits   <- different!  KL is NOT symmetric")
    print(f"  KL(p‖p) = {kl(p,p):.3f} bits   <- zero only when the distributions match")

    # The fundamental identity: cross-entropy = entropy + KL
    print("\nIdentity:  cross-entropy(p,q) = H(p) + KL(p‖q)")
    print(f"  H(p) + KL(p‖q) = {entropy(p):.3f} + {kl(p,q):.3f} = {entropy(p)+kl(p,q):.3f}")
    print(f"  cross_entropy(p,q)                    = {cross_entropy(p,q):.3f}")
    print("\nSo minimizing cross-entropy (training) = minimizing KL to the true distribution,")
    print("since H(p) is a constant you can't change. Training pushes your model onto the data.")

if __name__ == "__main__":
    main()
