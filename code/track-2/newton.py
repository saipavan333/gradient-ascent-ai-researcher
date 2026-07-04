"""
Track 2 · Lesson 4 — Newton's method & second-order ideas
Run:  python newton.py
Newton uses curvature (the Hessian) to take a smarter step: on a quadratic it
reaches the minimum in ONE step. The Hessian's eigenvalues give the condition
number that predicts how slow plain gradient descent will be.
"""
import numpy as np


def newton_1d(fp, fpp, x0, steps=30):
    """1D Newton: x <- x - f'/f''."""
    x = x0
    for _ in range(steps):
        x = x - fp(x) / fpp(x)
    return x


def main():
    # Minimize g(x) = (x-3)^4 (min at 3)
    gp = lambda x: 4*(x-3)**3
    gpp = lambda x: 12*(x-3)**2 + 1e-12      # guard the vanishing curvature at the min
    print("Newton 1D ->", round(newton_1d(gp, gpp, 0.0, 60), 4))   # ~3.0

    # Multivariable Newton on a quadratic: ONE step reaches the minimum
    A = np.array([[3.0, 1.0], [1.0, 2.0]])    # f = 0.5 x^T A x ; min at 0
    grad = lambda x: A @ x
    H = A                                      # Hessian of a quadratic is constant
    x = np.array([5.0, -4.0])
    x_newton = x - np.linalg.solve(H, grad(x))     # solve H d = grad (don't invert!)
    print("Newton in ONE step:", np.round(x_newton, 6))    # [0. 0.]

    # Compare: gradient descent needs many steps and a tuned lr
    x_gd = np.array([5.0, -4.0])
    for _ in range(50):
        x_gd = x_gd - 0.2 * grad(x_gd)
    print("GD after 50 steps :", np.round(x_gd, 4))         # close, not exact

    # Condition number of the Hessian (Track 1 eigenvalues) predicts GD's difficulty
    eigs = np.linalg.eigvalsh(A)
    print("eigenvalues:", np.round(eigs, 3),
          " condition number:", round(eigs.max()/eigs.min(), 2))

    # An ill-conditioned example: kappa = 4
    H2 = np.array([[4.0, 0.0], [0.0, 16.0]])
    g2 = lambda p: np.array([4*p[0], 16*p[1]])
    p = np.array([3.0, 2.0])
    print("Newton on 2x^2+8y^2:", np.round(p - np.linalg.solve(H2, g2(p)), 6))  # [0. 0.]
    e2 = np.linalg.eigvalsh(H2)
    print("its condition number:", round(e2.max()/e2.min(), 2))                 # 4.0

    # --- Try it yourself -----------------------------------------------------
    # 1) One Newton step on f(x)=x^2-4x+7 from x0=0 should land exactly at x=2.


if __name__ == "__main__":
    main()
