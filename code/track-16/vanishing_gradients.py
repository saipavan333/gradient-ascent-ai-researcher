"""
Track 16 · Lesson 1 — Why RNNs forget: vanishing & exploding gradients
======================================================================
A vanilla RNN multiplies by the same recurrent weight matrix at every time
step. Backpropagation-through-time therefore multiplies many Jacobians together
— and a product of many similar factors either shrinks toward zero (vanishing)
or blows up (exploding), exponentially in the sequence length. Vanishing
gradients are why plain RNNs can't learn long-range dependencies: the error
signal from step T never reaches step 0 with any strength.

We demonstrate it numerically: send a unit gradient back through a tanh RNN and
watch its norm as a function of how many steps it has travelled.

Run:  python vanishing_gradients.py
"""
import numpy as np

rng = np.random.default_rng(0)


def backprop_signal(Whh, T, H):
    """Propagate a unit gradient backward T steps through a tanh RNN and record
    the gradient norm arriving at each earlier step. Returns list of norms."""
    # a random hidden trajectory (values in (-1,1), as after tanh)
    hs = [rng.uniform(-0.5, 0.5, H) for _ in range(T)]
    dh = rng.standard_normal(H)
    dh /= np.linalg.norm(dh)                 # start with a unit gradient at step T
    norms = [1.0]
    for t in reversed(range(T)):
        da = (1 - hs[t] ** 2) * dh           # back through tanh
        dh = Whh.T @ da                      # back one recurrent step
        norms.append(np.linalg.norm(dh))
    return norms                              # norms[k] = signal after k steps back


if __name__ == "__main__":
    H, T = 20, 40
    print(f"Propagating a unit gradient back through {T} steps of a tanh RNN.\n")

    for scale, label in [(0.5, "small weights (spectral radius < 1)"),
                         (1.0, "moderate weights"),
                         (1.6, "large weights (spectral radius > 1)")]:
        Whh = rng.standard_normal((H, H)) * (scale / np.sqrt(H))
        norms = backprop_signal(Whh, T, H)
        print(f"{label}:")
        print(f"   gradient norm after  1 step : {norms[1]:.3e}")
        print(f"   gradient norm after 20 steps: {norms[20]:.3e}")
        print(f"   gradient norm after 40 steps: {norms[40]:.3e}")
        ratio = norms[40] / (norms[1] + 1e-30)
        print(f"   → shrank/grew by factor {ratio:.2e} over 40 steps\n")

    # Verify the core claim: with small recurrent weights the signal vanishes.
    Whh = rng.standard_normal((H, H)) * (0.5 / np.sqrt(H))
    norms = backprop_signal(Whh, T, H)
    assert norms[40] < 0.01 * norms[1], "gradient should vanish over 40 steps"
    print("OK — the learning signal decays exponentially with distance: that is why")
    print("     plain RNNs cannot connect events far apart in a sequence.")
