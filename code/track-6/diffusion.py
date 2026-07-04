"""
Track 6 · Lesson 4 — A diffusion model (DDPM) from scratch, in NumPy
Run:  python diffusion.py

The idea in one breath: take clean data, gradually add Gaussian noise over T
steps until it is pure noise (the "forward process"), then train a small neural
network to *undo* one step of noising. To generate, start from noise and apply
the learned denoiser T times. No PyTorch — we hand-derive every gradient so you
can see that a diffusion model is just an MLP trained on a clever objective.
"""
import numpy as np

rng = np.random.default_rng(0)

# ---- data: the two-moons distribution, standardized to mean 0 / std 1 --------
def two_moons(n):
    t = rng.uniform(0, np.pi, n)
    a = np.c_[np.cos(t),       np.sin(t)]        + rng.normal(0, 0.05, (n, 2))
    b = np.c_[1 - np.cos(t), 0.4 - np.sin(t)]    + rng.normal(0, 0.05, (n, 2))
    X = np.vstack([a, b])
    return (X - X.mean(0)) / X.std(0)

# ---- the forward (noising) schedule -----------------------------------------
T = 50
betas = np.linspace(1e-4, 0.02, T)          # how much noise we add at each step
alphas = 1 - betas
acp = np.cumprod(alphas)                     # \bar\alpha_t  = prod of alphas up to t
# Closed form: x_t = sqrt(acp[t]) * x0 + sqrt(1-acp[t]) * eps,  eps ~ N(0, I)

# ---- the denoiser: a 2-hidden-layer MLP that PREDICTS THE NOISE eps ----------
# Input features: position (x,y) plus a small embedding of the timestep t.
H = 128
def time_feats(t_idx):                       # t_idx: (B,) integer timesteps
    tn = t_idx / T
    return np.stack([np.sin(2*np.pi*tn), np.cos(2*np.pi*tn), np.sin(4*np.pi*tn)], 1)

def init(shape, fan_in):
    return rng.normal(0, np.sqrt(2.0/fan_in), shape)

P = {
    'W1': init((5, H), 5),  'b1': np.zeros(H),     # input is [x, y, 3 time feats] = 5
    'W2': init((H, H), H),  'b2': np.zeros(H),
    'W3': init((H, 2), H),  'b3': np.zeros(2),     # output: predicted noise (2D)
}
opt = {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in P.items()}

def relu(z): return np.maximum(0.0, z)

def denoiser(x, t_idx):
    """Forward pass. Returns predicted noise and a cache for backprop."""
    inp = np.concatenate([x, time_feats(t_idx)], axis=1)
    z1 = inp @ P['W1'] + P['b1']; h1 = relu(z1)
    z2 = h1 @ P['W2'] + P['b2'];  h2 = relu(z2)
    out = h2 @ P['W3'] + P['b3']
    return out, (inp, z1, h1, z2, h2)

def adam(grads, step, lr=2e-3):
    for k in P:
        m, v = opt[k]
        m[...] = 0.9*m + 0.1*grads[k]
        v[...] = 0.999*v + 0.001*grads[k]**2
        P[k] -= lr * (m/(1-0.9**step)) / (np.sqrt(v/(1-0.999**step)) + 1e-8)

# ---- training: the DDPM objective is just MSE between true and predicted noise
def train(data, steps=12000, bs=256):
    for step in range(1, steps+1):
        x0 = data[rng.integers(0, len(data), bs)]
        t_idx = rng.integers(0, T, bs)                 # random timestep per sample
        eps = rng.normal(size=x0.shape)                # the noise we will add
        sa = np.sqrt(acp[t_idx])[:, None]
        sb = np.sqrt(1 - acp[t_idx])[:, None]
        x_t = sa * x0 + sb * eps                       # noised sample (closed form)

        pred, (inp, z1, h1, z2, h2) = denoiser(x_t, t_idx)
        # loss = mean ||pred - eps||^2  -> dL/dpred:
        d_out = (pred - eps) * (2.0 / bs)
        g = {}
        g['W3'] = h2.T @ d_out; g['b3'] = d_out.sum(0)
        dh2 = d_out @ P['W3'].T * (z2 > 0)
        g['W2'] = h1.T @ dh2;  g['b2'] = dh2.sum(0)
        dh1 = dh2 @ P['W2'].T * (z1 > 0)
        g['W1'] = inp.T @ dh1; g['b1'] = dh1.sum(0)
        adam(g, step)
        if step % 3000 == 0:
            print(f"step {step:5d}   noise-MSE {np.mean((pred-eps)**2):.4f}")

# ---- sampling: start from noise, walk the reverse process back to data -------
def sample(n):
    x = rng.normal(size=(n, 2))                        # x_T ~ pure noise
    for t in reversed(range(T)):
        t_idx = np.full(n, t)
        pred, _ = denoiser(x, t_idx)                   # predicted noise
        a_t, ac_t, beta_t = alphas[t], acp[t], betas[t]
        mean = (x - (1-a_t)/np.sqrt(1-ac_t) * pred) / np.sqrt(a_t)
        x = mean + (np.sqrt(beta_t) * rng.normal(size=x.shape) if t > 0 else 0)
    return x

def main():
    data = two_moons(2000)
    print("Training a from-scratch DDPM on two-moons ...")
    train(data)
    gen = sample(2000)
    print("Generated", len(gen), "samples. mean:", gen.mean(0).round(2),
          " std:", gen.std(0).round(2), " (data is ~0 mean, ~1 std)")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
        ax[0].scatter(data[:,0], data[:,1], s=6, c="#2563eb"); ax[0].set_title("real")
        ax[1].scatter(gen[:,0],  gen[:,1],  s=6, c="#0f766e"); ax[1].set_title("generated (diffusion)")
        for a in ax: a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        fig.savefig("diffusion_samples.png", dpi=110, bbox_inches="tight")
        print("Saved diffusion_samples.png")
    except Exception as e:
        print("(plotting skipped:", e, ")")

    # --- Try it yourself -----------------------------------------------------
    # 1) Raise T to 200 with the same beta range — do samples get cleaner?
    # 2) Remove the time features (feed only x,y). Why does training collapse?

if __name__ == "__main__":
    main()
