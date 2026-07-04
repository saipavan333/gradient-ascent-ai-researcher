"""
Track 13 · Lesson 0 — Entropy: measuring uncertainty, from scratch
Run:  python entropy.py

Information theory is the mathematical glue under most of ML — cross-entropy loss,
KL divergence, mutual information all start here. ENTROPY measures the average
'surprise' of a distribution:  H(X) = -Σ p(x) log₂ p(x)  bits. A fair coin is 1 bit
of uncertainty; a predictable one is less. It's the number of yes/no questions you'd
need, on average, to pin down the outcome.
"""
import numpy as np

def entropy(p):
    p = np.asarray(p, float); p = p[p > 0]        # 0·log0 = 0
    return float(-(p * np.log2(p)).sum())          # bits

def main():
    print("Entropy H = average surprise, in bits:\n")
    cases = {
        "fair coin  [0.5, 0.5]":            [0.5, 0.5],
        "biased coin [0.9, 0.1]":           [0.9, 0.1],
        "almost sure [0.99, 0.01]":         [0.99, 0.01],
        "fair die   [1/6]*6":               [1/6]*6,
        "uniform 8  [1/8]*8":               [1/8]*8,
    }
    for name, p in cases.items():
        print(f"  {name:28s} H = {entropy(p):.3f} bits")
    # English letters are far from uniform -> less than log2(26)=4.70 bits
    freq = np.array([8.2,1.5,2.8,4.3,12.7,2.2,2.0,6.1,7.0,0.15,0.77,4.0,2.4,
                     6.7,7.5,1.9,0.095,6.0,6.3,9.1,2.8,0.98,2.4,0.15,2.0,0.074])
    freq = freq / freq.sum()
    print(f"\n  Uniform over 26 letters : H = {np.log2(26):.3f} bits (max possible)")
    print(f"  Actual English letters  : H = {entropy(freq):.3f} bits  (structure = lower entropy)")
    print("\nLower entropy = more predictable = more compressible (Lesson 4).")

if __name__ == "__main__":
    main()
