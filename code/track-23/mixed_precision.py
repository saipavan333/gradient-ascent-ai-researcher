"""
Track 19 · Lesson 4 — Mixed precision & loss scaling
====================================================
Training in 16-bit floats (fp16) instead of 32-bit halves memory and speeds up the
math on modern GPUs. The catch: fp16 can't represent very small numbers — anything
below ~6e-5 rounds to ZERO (underflow). Deep-net gradients are often that small, so
they silently vanish and learning stalls. LOSS SCALING fixes it: multiply the loss by
a big constant S before backprop (lifting every gradient above the underflow floor),
then divide the gradients by S before the update. We show underflow and the rescue.

Run:  python mixed_precision.py
"""
import numpy as np
rng = np.random.default_rng(0)

FP16_MIN = 6.1e-5                                  # smallest positive fp16 "normal" number
def to_fp16(x):
    """Simulate fp16 storage: values below the min representable magnitude underflow to 0."""
    return np.where(np.abs(x) < FP16_MIN, 0.0, x)

if __name__ == "__main__":
    # a batch of realistic, tiny gradients (deep-net scale)
    grads = rng.standard_normal(10000) * 2e-5     # magnitudes around 2e-5 — in the danger zone
    true_mean_grad = grads.mean()

    # --- naive fp16: store gradients directly ---
    lost = (np.abs(to_fp16(grads)) == 0).mean()
    naive_recovered = to_fp16(grads).mean()
    print(f"fp16 (no scaling): {lost*100:.0f}% of gradients underflowed to ZERO")
    print(f"   → the update uses {naive_recovered:.2e} instead of the true {true_mean_grad:.2e}")

    # --- loss scaling: scale up, store in fp16, scale back down ---
    S = 1024.0
    scaled = to_fp16(grads * S) / S               # multiply before fp16, divide after
    lost_scaled = (np.abs(to_fp16(grads * S)) == 0).mean()
    scaled_recovered = scaled.mean()
    print(f"loss scaling (S={int(S)}): {lost_scaled*100:.0f}% underflowed")
    print(f"   → recovers {scaled_recovered:.2e}  (true {true_mean_grad:.2e})")

    naive_err  = abs(naive_recovered - true_mean_grad) / abs(true_mean_grad)
    scaled_err = abs(scaled_recovered - true_mean_grad) / abs(true_mean_grad)
    print(f"relative error — naive: {naive_err*100:.0f}%   loss-scaled: {scaled_err*100:.1f}%")
    assert lost > 0.3 and scaled_err < 0.05 and naive_err > scaled_err
    print("OK — loss scaling lifts tiny gradients above the fp16 floor, so they survive instead of vanishing.")
