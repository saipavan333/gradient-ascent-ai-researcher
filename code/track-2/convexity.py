"""
Track 2 · Lesson 2 — Convexity & why it matters
Run:  python convexity.py
Convex = every chord lies above the graph => one bowl => gradient descent finds
the global minimum. On non-convex functions, where you land depends on the start.
"""
import numpy as np


def chord_test(f, a, b, n=50):
    """Definition check: does the chord stay above the curve between a and b?"""
    for lam in np.linspace(0, 1, n):
        mid = lam*a + (1-lam)*b
        if f(mid) > lam*f(a) + (1-lam)*f(b) + 1e-9:   # curve poked above the chord
            return False
    return True


def gd_1d(grad, x0, lr=0.05, steps=200):
    x = x0
    for _ in range(steps):
        x = x - lr*grad(x)
    return x


def main():
    convex = lambda x: x**2
    nonconvex = lambda x: x**4 - 4*x**2
    print("x^2 convex on [-2,3]   :", chord_test(convex, -2, 3))       # True
    print("x^4-4x^2 convex on[-2,2]:", chord_test(nonconvex, -2, 2))   # False
    print("|x| convex on [-3,3]    :", chord_test(np.abs, -3, 3))      # True (non-smooth!)

    # Consequence: GD's destination on a non-convex function depends on the start
    grad_nc = lambda x: 4*x**3 - 8*x         # minima at +-sqrt(2) ~ +-1.414
    print("non-convex from +0.1 ->", round(gd_1d(grad_nc,  0.1), 3))   # 1.414
    print("non-convex from -0.1 ->", round(gd_1d(grad_nc, -0.1), 3))   # -1.414

    # On a convex function the destination is the SAME from anywhere
    grad_c = lambda x: 2*x                   # x^2, min at 0
    print("convex from +5 ->", round(gd_1d(grad_c,  5.0), 3),
          " from -5 ->", round(gd_1d(grad_c, -5.0), 3))                # 0.0  0.0

    # Second-derivative test (numerical): f'' >= 0 everywhere?  (for x^2 vs x^4-4x^2)
    def second_deriv(f, x, h=1e-3): return (f(x+h) - 2*f(x) + f(x-h)) / h**2
    xs = np.linspace(-2, 2, 9)
    print("x^2 f'' >=0 :", all(second_deriv(convex, x) >= -1e-6 for x in xs))      # True
    print("x^4-4x^2 f'':", np.round([second_deriv(nonconvex, x) for x in [-1.5,0,1.5]], 1))  # neg at 0

    # --- Try it yourself -----------------------------------------------------
    # 1) Is x^4 convex? Check with chord_test and the second-derivative test.


if __name__ == "__main__":
    main()
