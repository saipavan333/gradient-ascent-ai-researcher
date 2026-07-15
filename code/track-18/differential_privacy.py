"""
Track 17 · Lesson 2 — Differential privacy
==========================================
Federated learning keeps raw data on-device, but the shared UPDATES can still leak
private facts. DIFFERENTIAL PRIVACY (DP) gives a mathematical guarantee: the output
looks almost the same whether or not any single person is in the data, so no one can
tell if you participated. It works by adding calibrated random noise. The privacy
level epsilon (ε) trades off: smaller ε = more noise = more privacy, less accuracy.

Run:  python differential_privacy.py
"""
import numpy as np
rng = np.random.default_rng(0)

def private_mean(data, eps):
    """Release the mean of values in [0,1] with Laplace noise (sensitivity 1/n)."""
    n = len(data); true = data.mean()
    scale = (1.0 / n) / eps                      # Laplace scale = sensitivity / epsilon
    return true + rng.laplace(0, scale)

if __name__ == "__main__":
    data = rng.random(1000)                      # 1000 people's values in [0,1]
    neighbour = data.copy(); neighbour[0] = 1 - neighbour[0]   # change ONE person

    # --- indistinguishability: the output barely shifts when one person changes ---
    eps = 1.0
    a = np.array([private_mean(data, eps) for _ in range(20000)])
    b = np.array([private_mean(neighbour, eps) for _ in range(20000)])
    print(f"true mean {data.mean():.3f};  with vs without person 0, released means differ by only {abs(a.mean()-b.mean()):.4f}")

    # --- privacy-utility tradeoff: more privacy (small eps) -> more error ---
    for eps in (0.1, 1.0, 10.0):
        err = np.mean([abs(private_mean(data, eps) - data.mean()) for _ in range(5000)])
        print(f"  eps={eps:>4}: average error of released mean = {err:.4f}")
    e_small = np.mean([abs(private_mean(data,0.1)-data.mean()) for _ in range(5000)])
    e_large = np.mean([abs(private_mean(data,10.0)-data.mean()) for _ in range(5000)])
    assert e_small > e_large                     # stronger privacy costs accuracy
    print("OK — noise hides any single person while keeping the answer useful; eps tunes the trade-off.")
