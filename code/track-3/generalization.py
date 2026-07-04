"""
Track 3 · Lesson 0 — The learning problem & generalization
Run:  python generalization.py
Training error always drops with complexity; TEST error is U-shaped. We fit
polynomials of growing degree and watch overfitting appear in the numbers.
"""
import numpy as np


def main():
    rng = np.random.default_rng(0)
    true_f = lambda x: np.sin(1.3*x) + 0.3*x

    def make_data(n):
        x = rng.uniform(0, 5, n)
        return x, true_f(x) + rng.normal(0, 0.3, n)

    x_train, y_train = make_data(20)
    x_test, y_test = make_data(500)            # large, unseen test set

    def error(deg):
        coef = np.polyfit(x_train, y_train, deg)            # fit on TRAIN only
        tr = np.mean((np.polyval(coef, x_train) - y_train)**2)
        te = np.mean((np.polyval(coef, x_test) - y_test)**2)
        return tr, te

    print(f"{'degree':>6} {'train':>8} {'test':>8}")
    for deg in [1, 2, 3, 5, 9, 15]:
        tr, te = error(deg)
        print(f"{deg:>6} {tr:>8.3f} {te:>8.3f}")

    best = min(range(1, 16), key=lambda d: error(d)[1])
    print("best degree by test error:", best)   # small/medium, not 15

    # More data lets you safely use more complex models:
    x_train, y_train = make_data(200)
    best200 = min(range(1, 16), key=lambda d: error(d)[1])
    print("best degree with 200 training points:", best200)

    # --- Try it yourself -----------------------------------------------------
    # 1) Print the degree-15 train error on 16 points -> it can be ~0 (memorize).


if __name__ == "__main__":
    main()
