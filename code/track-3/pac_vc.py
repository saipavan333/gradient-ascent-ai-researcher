"""
Track 3 · Lesson 3 — PAC learning & VC dimension
Run:  python pac_vc.py
The generalization gap shrinks like 1/sqrt(n). A 2-D linear classifier has VC
dimension 3: it shatters 3 points (all labelings separable) but not 4 (XOR).
"""
import itertools
import numpy as np


def gap_for_n(n, d=20, trials=40, rng=None):
    rng = rng or np.random.default_rng(0)
    gaps = []
    w_true = rng.normal(size=d)
    for _ in range(trials):
        Xtr = rng.normal(size=(n, d)); ytr = np.sign(Xtr @ w_true)
        Xte = rng.normal(size=(2000, d)); yte = np.sign(Xte @ w_true)
        w = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
        tr = np.mean(np.sign(Xtr @ w) != ytr)
        te = np.mean(np.sign(Xte @ w) != yte)
        gaps.append(te - tr)
    return float(np.mean(gaps))


def shatters(points):
    """Can a linear threshold (with bias) realize ALL labelings of these points?"""
    P = np.c_[points, np.ones(len(points))]            # add bias column
    for labels in itertools.product([0, 1], repeat=len(points)):
        y = np.array(labels) * 2 - 1                     # {0,1} -> {-1,+1}
        w, *_ = np.linalg.lstsq(P, y, rcond=None)
        if not np.all(np.sign(P @ w) == y):
            return False
    return True


def main():
    rng = np.random.default_rng(0)
    print(f"{'n':>6} {'gap':>8}")
    for n in [25, 50, 100, 400, 1600]:
        print(f"{n:>6} {gap_for_n(n, rng=rng):>8.3f}")   # gap ~ 1/sqrt(n)

    three = np.array([[0, 0], [1, 0], [0, 1]])           # generic 3 points
    xor4 = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])    # XOR square
    print("shatters 3 points:", shatters(three))         # True
    print("shatters XOR 4   :", shatters(xor4))          # False -> VC dim = 3

    # --- Try it yourself -----------------------------------------------------
    # 1) Compare gap at n=100 vs n=400; 4x data should roughly halve the gap.


if __name__ == "__main__":
    main()
