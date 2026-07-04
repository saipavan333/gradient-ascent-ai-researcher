"""
Track 4 · Lesson 0 — The neuron & the computational graph
Run:  python neuron_forward.py
A neuron computes a weighted sum of its inputs, adds a bias, and applies an
activation. Stringing such operations together forms a computational graph.
"""
import numpy as np


def neuron(x, w, b, activation=np.tanh):
    z = np.dot(w, x) + b         # weighted sum + bias  (a dot product, Track 1)
    return activation(z)


def main():
    x = np.array([2.0, 0.0, -1.0])      # inputs
    w = np.array([-3.0, 1.0, 0.5])      # weights
    b = 6.0                              # bias
    z = np.dot(w, x) + b
    y = np.tanh(z)
    print("weighted sum z =", z)         # -3*2 + 1*0 + 0.5*-1 + 6 = -0.5
    print("activation y  =", round(y, 4))  # tanh(-0.5) ~ -0.462

    # A tiny computational graph, built by hand: e = a*b, then d = e + c
    a, b2, c = 2.0, -3.0, 10.0
    e = a * b2          # node: multiply
    d = e + c           # node: add
    print("graph: e =", e, " d =", d)    # -6, 4

    # ReLU vs tanh on the same neuron
    print("relu neuron:", round(neuron(x, w, b, lambda z: np.maximum(0, z)), 4))

    # --- Try it yourself -----------------------------------------------------
    # 1) Change one weight and predict whether y rises or falls before running.


if __name__ == "__main__":
    main()
