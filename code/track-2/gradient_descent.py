"""
Track 2 · Lesson 1 — Gradient descent from scratch
Run:  python gradient_descent.py
theta <- theta - alpha * grad(theta), repeated. This is how essentially every
model in modern AI is trained. We also fit a real linear regression with it.
"""
import numpy as np


def gradient_descent(grad, theta0, lr=0.1, steps=100):
    """Follow -gradient downhill. `grad` returns the gradient at a point."""
    theta = np.array(theta0, float)
    history = [theta.copy()]
    for _ in range(steps):
        theta = theta - lr * grad(theta)      # the entire algorithm
        history.append(theta.copy())
    return theta, np.array(history)


def main():
    # Minimize f(x,y) = x^2 + 2y^2 ; gradient (2x, 4y) ; minimum at (0,0)
    grad_f = lambda t: np.array([2*t[0], 4*t[1]])
    theta, hist = gradient_descent(grad_f, [3.0, 1.0], lr=0.1, steps=60)
    print("converged to:", np.round(theta, 4))   # ~[0. 0.]

    # The payoff: fit y = w*x + b by gradient descent on Mean Squared Error
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=200)
    y = 2.0 * X - 1.0 + rng.normal(0, 0.5, size=200)   # true: w=2, b=-1, + noise

    def mse_grad(p):
        w, b = p
        err = (w * X + b) - y
        return np.array([2*np.mean(err * X), 2*np.mean(err)])

    params, _ = gradient_descent(mse_grad, [0.0, 0.0], lr=0.05, steps=400)
    print("learned w, b:", np.round(params, 3))        # ~[2. -1.]

    # Learning-rate sweep: too small / good / too large
    def fit(lr, steps=400):
        w, b = 0.0, 0.0
        for _ in range(steps):
            err = (w*X + b) - y
            w -= lr * 2*np.mean(err*X); b -= lr * 2*np.mean(err)
        return w, b, np.mean(((w*X+b)-y)**2)
    for lr in [0.01, 0.05, 0.5]:
        with np.errstate(over='ignore', invalid='ignore'):
            print(f"  lr={lr}: (w,b,mse) =", np.round(fit(lr), 3))   # 0.5 -> diverges

    # --- Try it yourself -----------------------------------------------------
    # 1) Minimize f(x)=(x-3)^2 from x0=0 with alpha=0.25; print first 3 iterates.


if __name__ == "__main__":
    main()
