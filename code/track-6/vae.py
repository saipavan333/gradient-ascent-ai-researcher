"""
Track 6 · Lesson 2 — A Variational Autoencoder (VAE) from scratch, in NumPy
Run:  python vae.py

An autoencoder squeezes data through a narrow "latent" bottleneck and rebuilds
it. A *variational* autoencoder makes that bottleneck PROBABILISTIC: the encoder
outputs a mean and variance, we sample a latent z from that Gaussian (the
"reparameterization trick" keeps it differentiable), and the decoder maps z back
to data. Training maximizes the ELBO = reconstruction quality minus a KL term
that pulls the latent code toward a standard normal. Afterwards we can GENERATE
by sampling z ~ N(0, I) and decoding. Every gradient is hand-derived.
"""
import numpy as np

rng = np.random.default_rng(0)

def two_moons(n):
    t = rng.uniform(0, np.pi, n)
    a = np.c_[np.cos(t),       np.sin(t)]     + rng.normal(0, 0.05, (n, 2))
    b = np.c_[1 - np.cos(t), 0.4 - np.sin(t)] + rng.normal(0, 0.05, (n, 2))
    X = np.vstack([a, b])
    return (X - X.mean(0)) / X.std(0)

def relu(z): return np.maximum(0.0, z)

H, Zd = 64, 2                                  # hidden width, latent dimension
P = {
    'We':  rng.normal(0, 1.0, (2, H)),  'be':  np.zeros(H),    # encoder hidden
    'Wmu': rng.normal(0, 0.1, (H, Zd)), 'bmu': np.zeros(Zd),   # -> mean
    'Wlv': rng.normal(0, 0.1, (H, Zd)), 'blv': np.zeros(Zd),   # -> log-variance
    'Wd':  rng.normal(0, 1.0, (Zd, H)), 'bd':  np.zeros(H),    # decoder hidden
    'Wo':  rng.normal(0, 0.3, (H, 2)),  'bo':  np.zeros(2),    # -> reconstruction
}
opt = {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in P.items()}
def adam(g, step, lr=2e-3):
    for k in P:
        m, v = opt[k]
        m[...] = 0.9*m + 0.1*g[k]
        v[...] = 0.999*v + 0.001*g[k]**2
        P[k] -= lr * (m/(1-0.9**step)) / (np.sqrt(v/(1-0.999**step)) + 1e-8)

def train(data, steps=8000, bs=256):
    for step in range(1, steps+1):
        x = data[rng.integers(0, len(data), bs)]
        he_ = relu(x @ P['We'] + P['be'])
        mu  = he_ @ P['Wmu'] + P['bmu']
        lv  = he_ @ P['Wlv'] + P['blv']            # log variance
        eps = rng.normal(size=mu.shape)
        z   = mu + np.exp(0.5*lv) * eps            # reparameterization trick
        hd  = relu(z @ P['Wd'] + P['bd'])
        xo  = hd @ P['Wo'] + P['bo']               # reconstruction
        # ---- backprop: loss = recon MSE + KL(q(z|x) || N(0,I)) --------------
        drec = (xo - x) * (2.0 / bs)
        g = {}
        g['Wo'] = hd.T @ drec; g['bo'] = drec.sum(0)
        dhd = drec @ P['Wo'].T * (hd > 0)
        g['Wd'] = z.T @ dhd; g['bd'] = dhd.sum(0)
        dz = dhd @ P['Wd'].T
        # KL gradients (per-sample, averaged over batch):
        dmu = dz + mu / bs                          # dKL/dmu = mu
        dlv = dz * np.exp(0.5*lv) * eps * 0.5 + 0.5*(np.exp(lv) - 1) / bs
        g['Wmu'] = he_.T @ dmu; g['bmu'] = dmu.sum(0)
        g['Wlv'] = he_.T @ dlv; g['blv'] = dlv.sum(0)
        dhe = (dmu @ P['Wmu'].T + dlv @ P['Wlv'].T) * (he_ > 0)
        g['We'] = x.T @ dhe; g['be'] = dhe.sum(0)
        adam(g, step)
        if step % 2000 == 0:
            recon = np.mean((xo - x)**2)
            kl = np.mean(-0.5*np.sum(1 + lv - mu**2 - np.exp(lv), 1))
            print(f"step {step:5d}   recon {recon:.3f}   KL {kl:.3f}")

def generate(n):
    z = rng.normal(size=(n, Zd))                    # sample latent from the prior
    hd = relu(z @ P['Wd'] + P['bd'])
    return hd @ P['Wo'] + P['bo']

def main():
    data = two_moons(2000)
    print("Training a from-scratch VAE on two-moons ...")
    train(data)
    gen = generate(2000)
    print("Generated", len(gen), "samples by decoding z ~ N(0, I).")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
        ax[0].scatter(data[:,0], data[:,1], s=6, c="#2563eb"); ax[0].set_title("real")
        ax[1].scatter(gen[:,0],  gen[:,1],  s=6, c="#6d28d9"); ax[1].set_title("generated (VAE)")
        for a in ax: a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        fig.savefig("vae_samples.png", dpi=110, bbox_inches="tight")
        print("Saved vae_samples.png")
    except Exception as e:
        print("(plotting skipped:", e, ")")

    # --- Try it yourself -----------------------------------------------------
    # 1) Drop the KL term. The reconstructions sharpen but sampling z~N(0,I) breaks. Why?
    # 2) Set the latent to 1-D. Can a single number still trace the two moons?

if __name__ == "__main__":
    main()
