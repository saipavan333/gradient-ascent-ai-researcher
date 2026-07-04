"""
Track 4 · Lesson 3 — Activation functions
Run:  python activations.py
Nonlinearities are what let a deep network bend space (Track 1, Lesson 6). Here
are the common ones and their derivatives — note where the gradient vanishes.
"""
import numpy as np


def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def relu(x): return np.maximum(0, x)
def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715*x**3)))


def main():
    xs = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    print("x      :", xs)
    print("sigmoid:", np.round(sigmoid(xs), 3))
    print("tanh   :", np.round(tanh(xs), 3))
    print("relu   :", np.round(relu(xs), 3))
    print("gelu   :", np.round(gelu(xs), 3))

    # Derivatives, and the vanishing-gradient problem
    dsigmoid = lambda x: sigmoid(x) * (1 - sigmoid(x))
    print("\nsigmoid'(0) =", round(dsigmoid(0.0), 3),      # 0.25 (max)
          "  sigmoid'(10) =", round(dsigmoid(10.0), 6))    # ~0.00005 -> vanishes!
    print("tanh'(0) =", round(1 - tanh(0.0)**2, 3),        # 1.0
          "   tanh'(10) =", round(1 - tanh(10.0)**2, 8))   # ~0 -> vanishes
    print("relu'(x>0) = 1, relu'(x<0) = 0  (no saturation on the positive side)")

    # Why ReLU is popular: its gradient is exactly 1 for active units, so it
    # doesn't shrink gradients in deep stacks the way sigmoid/tanh do.

    # --- Try it yourself -----------------------------------------------------
    # 1) Plot these with matplotlib; confirm sigmoid/tanh flatten (saturate) at the ends.


if __name__ == "__main__":
    main()
