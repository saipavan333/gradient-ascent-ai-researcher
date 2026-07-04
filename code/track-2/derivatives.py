"""
Track 2 · Lesson 0 — Derivatives & gradients, geometrically
Run:  python derivatives.py
The derivative is a slope; the gradient stacks the partial slopes into a vector
that points uphill. -gradient is the downhill direction we follow to minimize.
"""
import numpy as np


def numerical_derivative(f, x, h=1e-5):
    """Centered finite difference: error ~ h^2 (better than the raw definition)."""
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(f, x, h=1e-5):
    """Gradient of a multivariable f at vector x, one partial at a time."""
    x = np.asarray(x, float)
    grad = np.zeros_like(x)
    for i in range(x.size):
        step = np.zeros_like(x); step[i] = h
        grad[i] = (f(x + step) - f(x - step)) / (2 * h)   # vary x_i, hold others fixed
    return grad


def main():
    f = lambda x: x**2                       # f'(x) = 2x
    print("f'(3) ~", numerical_derivative(f, 3.0))   # ~6.0

    g = lambda v: v[0]**2 + 2*v[1]**2         # grad = (2x, 4y)
    print("grad g at (3,1) ~", np.round(numerical_gradient(g, [3.0, 1.0]), 4))  # [6. 4.]

    x = np.array([3.0, 1.0])
    print("downhill direction:", -numerical_gradient(g, x))   # toward min (0,0)

    # Gradient check: compare analytic gradient to finite differences
    analytic = np.array([2*x[0], 4*x[1]])
    numeric = numerical_gradient(g, x)
    print("gradient check passes:", np.allclose(analytic, numeric, atol=1e-4))  # True

    # h too small makes it WORSE (round-off): demonstrate the sweet spot
    for h in [1e-1, 1e-5, 1e-12]:
        print(f"  h={h:.0e} -> f'(3) ~ {numerical_derivative(f, 3.0, h):.6f}")

    # --- Try it yourself -----------------------------------------------------
    # 1) For f(x,y)=x^2 y + 3y, check grad at (2,1) equals (4, 7).


if __name__ == "__main__":
    main()
