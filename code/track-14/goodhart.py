"""
Track 14 · Lesson 1 — Goodhart's law: over-optimizing a proxy
Run:  python goodhart.py

"When a measure becomes a target, it ceases to be a good measure." We can't specify
what we truly want, so we optimize a PROXY (e.g., a learned reward model). Push the
proxy gently and the true objective improves too; push it too HARD and you exploit the
proxy's flaws — the measured score keeps rising while TRUE quality peaks, then FALLS.
This is exactly the reward-model over-optimization seen in RLHF (Track 7).
"""
import numpy as np

def proxy_reward(x):   return 5.0 * (1 - np.exp(-2.0 * x))     # the measured score: keeps rising
def true_reward(x):    return proxy_reward(x) - 6.0 * x**2      # true value: rise, then over-optimization tax

def main():
    print("Optimizing a proxy reward harder and harder (x = optimization strength):\n")
    print(f"{'optimization x':>15} {'proxy score':>12} {'TRUE value':>11}")
    xs = [0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.00]
    for x in xs:
        note = ""
        print(f"{x:>15.2f} {proxy_reward(x):>12.2f} {true_reward(x):>11.2f}")
    # find the peak of true value
    grid = np.linspace(0, 1, 1000); xstar = grid[np.argmax(true_reward(grid))]
    print(f"\n  proxy score rises monotonically — the metric always looks better.")
    print(f"  TRUE value peaks at x ≈ {xstar:.2f}, then DECLINES as you over-optimize.")
    print(f"  The gap past the peak is reward hacking: chasing the measure, not the goal.")
    print(f"  Fix: stop at the peak (early stopping) or constrain optimization (RLHF's KL penalty).")

if __name__ == "__main__":
    main()
