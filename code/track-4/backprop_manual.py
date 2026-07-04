"""
Track 4 · Lesson 1 — Backpropagation by hand
Run:  python backprop_manual.py
Before automating it, do backprop manually on  e = a*b,  d = e + c.  Start with
dL/dd = 1 and apply the chain rule backward, then verify with finite differences.
"""


def main():
    # forward
    a, b, c = 2.0, -3.0, 10.0
    e = a * b          # e = -6
    d = e + c          # d = 4  (this is our "loss" L)
    print("forward:  e =", e, " d =", d)

    # backward, by hand (chain rule), starting from dL/dd = 1
    dd = 1.0
    de = dd * 1.0      # d = e + c  ->  dd/de = 1
    dc = dd * 1.0      # d = e + c  ->  dd/dc = 1
    da = de * b        # e = a * b  ->  de/da = b
    db = de * a        # e = a * b  ->  de/db = a
    print("manual grads:  da =", da, " db =", db, " dc =", dc)   # -3, 2, 1

    # verify each gradient with a finite-difference check
    def L(a, b, c):
        return a * b + c
    h = 1e-6
    num_da = (L(a+h, b, c) - L(a-h, b, c)) / (2*h)
    num_db = (L(a, b+h, c) - L(a, b-h, c)) / (2*h)
    num_dc = (L(a, b, c+h) - L(a, b, c-h)) / (2*h)
    print("finite-diff :  da =", round(num_da, 4),
          " db =", round(num_db, 4), " dc =", round(num_dc, 4))
    assert abs(num_da - da) < 1e-3 and abs(num_db - db) < 1e-3
    print("manual backprop matches finite differences ✓")

    # --- Try it yourself -----------------------------------------------------
    # 1) Add a step  f = d * 2,  treat f as the loss, and redo backprop by hand.


if __name__ == "__main__":
    main()
