"""
Track 11 · Lesson 2 — LoRA: low-rank fine-tuning from scratch
Run:  python lora.py

Fine-tuning a big model by updating ALL its weights is expensive. LoRA (Low-Rank
Adaptation) freezes the pretrained weight W0 and learns a tiny low-rank update
ΔW = B·A (B is d×r, A is r×d, with rank r ≪ d). Because fine-tuning updates are
empirically low-rank, this matches full fine-tuning while training a fraction of
the parameters. We show it on a linear layer with a genuinely low-rank target.
"""
import numpy as np
rng = np.random.default_rng(0)

d, r_true, r_lora = 32, 2, 2
W0 = rng.normal(0, 1/np.sqrt(d), (d, d))                 # frozen "pretrained" weights
dW_true = rng.normal(size=(d, r_true)) @ rng.normal(size=(r_true, d)) * 0.3  # low-rank task shift
Wstar = W0 + dW_true                                     # the fine-tuned target

X = rng.normal(size=(400, d))
Y = X @ Wstar.T                                          # target task outputs

def loss(W): 
    return float(np.mean((X @ W.T - Y)**2))

def full_finetune(steps=400, lr=0.1):
    dW = np.zeros((d, d))                                # update ALL d*d weights
    for _ in range(steps):
        G = (X @ (W0 + dW).T - Y).T @ X * (2/len(X))
        dW -= lr * G
    return W0 + dW, d*d

def lora_finetune(steps=1500, lr=0.02):
    A = rng.normal(0, 1e-2, (r_lora, d)); B = np.zeros((d, r_lora))   # B·A starts at 0
    for _ in range(steps):
        W = W0 + B @ A
        E = (X @ W.T - Y)                               # (n,d) error
        gW = E.T @ X * (2/len(X))                        # grad wrt the effective weight
        gB = gW @ A.T; gA = B.T @ gW                     # chain rule through B·A
        B -= lr * gB; A -= lr * gA
    return W0 + B @ A, r_lora*d + d*r_lora

def main():
    print(f"Layer size: {d}x{d} = {d*d} weights.  Task shift is rank {r_true}.\n")
    Wf, nf = full_finetune(); Wl, nl = lora_finetune()
    print(f"{'method':>14} {'final loss':>12} {'trained params':>16}")
    print(f"{'full fine-tune':>14} {loss(Wf):>12.5f} {nf:>16}")
    print(f"{'LoRA (r=%d)'%r_lora:>14} {loss(Wl):>12.5f} {nl:>16}")
    print(f"\nLoRA matches full fine-tuning using {100*nl/nf:.0f}% of the trainable parameters.")
    print("On real LLMs (d in the thousands) the saving is even more dramatic.")

if __name__ == "__main__":
    main()
