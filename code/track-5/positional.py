"""
Track 5 · Lesson 3 — Positional encodings
Run:  python positional.py
Attention is permutation-invariant: by itself it can't tell word order. We fix
this by ADDING a position-dependent vector to each token embedding.
"""
import numpy as np


def sinusoidal_encoding(T, d):
    """The original transformer's positional encoding: sines & cosines of
    geometrically-spaced frequencies."""
    pos = np.arange(T)[:, None]
    i = np.arange(d)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d)
    PE = np.where(i % 2 == 0, np.sin(angle), np.cos(angle))
    return PE                                 # (T, d)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True); e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def attention_out(X):
    d = X.shape[1]
    W = softmax(X @ X.T / np.sqrt(d), axis=1)
    return W @ X


def main():
    PE = sinusoidal_encoding(T=6, d=8)
    print("positional encoding (6 positions x 8 dims):")
    print(np.round(PE, 2))

    # Demonstrate permutation-invariance of bare attention:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 8))
    perm = [2, 0, 1]
    out1 = attention_out(X)
    out2 = attention_out(X[perm])
    # without positions, permuting inputs just permutes the SAME outputs
    print("\nbare attention is permutation-equivariant:",
          np.allclose(out1[perm], out2))     # True -> it ignores order!

    # Adding positions breaks the symmetry, so order now matters:
    Xp = X + sinusoidal_encoding(3, 8)
    Xp_perm = X[perm] + sinusoidal_encoding(3, 8)
    print("with positions added, order matters:",
          not np.allclose(attention_out(Xp)[perm], attention_out(Xp_perm)))  # True

    # --- Try it yourself -----------------------------------------------------
    # 1) Modern models often use LEARNED position embeddings instead -- just a
    #    (T, d) table trained like any other weights (that's what minigpt.py uses).


if __name__ == "__main__":
    main()
