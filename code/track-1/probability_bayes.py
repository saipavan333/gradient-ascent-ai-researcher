"""
Track 1 · Lesson 9 — Probability foundations & Bayes' rule
Run:  python probability_bayes.py
When a probability feels counterintuitive, simulate it and count. Here we
confirm the famous "positive test" result and the law of large numbers.
"""
import numpy as np


def main():
    rng = np.random.default_rng(0)
    N = 1_000_000

    # 1% have a disease; test is 99% sensitive, 5% false-positive rate
    diseased = rng.random(N) < 0.01
    test_pos = np.where(diseased,
                        rng.random(N) < 0.99,     # P(+ | disease)
                        rng.random(N) < 0.05)     # P(+ | healthy)

    # P(disease | positive) = (diseased AND positive) / (positive)
    p_d_given_pos = np.sum(diseased & test_pos) / np.sum(test_pos)
    print("P(disease | positive) ~", round(p_d_given_pos, 3))   # ~0.167

    # Compare to Bayes' rule computed directly
    bayes = (0.99*0.01) / (0.99*0.01 + 0.05*0.99)
    print("Bayes' rule        =", round(bayes, 3))              # 0.167

    # Law of large numbers: empirical frequency -> true probability
    flips = rng.random(100_000) < 0.5
    print("estimate of P(heads):", round(flips.mean(), 4))     # ~0.5

    # Conditional probability from a simulation: P(2 heads | >=1 head)
    c1 = rng.random(N) < 0.5
    c2 = rng.random(N) < 0.5
    at_least_one, two_heads = c1 | c2, c1 & c2
    print("P(2 heads | >=1 head) ~",
          round(np.sum(two_heads & at_least_one) / np.sum(at_least_one), 3))  # ~0.333

    # --- Try it yourself -----------------------------------------------------
    # 1) Spam: 40% of mail is spam; filter flags 95% of spam, 8% of ham.
    #    Simulate P(spam | flagged) and compare to Bayes (~0.888).


if __name__ == "__main__":
    main()
