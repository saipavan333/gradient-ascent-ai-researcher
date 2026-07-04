"""
Track 6 · Lesson 3 — A GAN from scratch, in NumPy
Run:  python gan.py

Two networks play a game. The GENERATOR turns random noise z into fake samples;
the DISCRIMINATOR tries to tell real data from fakes. We train them together:
the discriminator improves at spotting fakes, which pushes the generator to make
better fakes. At the ideal equilibrium the discriminator is reduced to guessing
(accuracy ~ 0.5). Both networks and all gradients are hand-derived here.
"""
import numpy as np

rng = np.random.default_rng(0)

def two_moons(n):
    t = rng.uniform(0, np.pi, n)
    a = np.c_[np.cos(t),       np.sin(t)]     + rng.normal(0, 0.05, (n, 2))
    b = np.c_[1 - np.cos(t), 0.4 - np.sin(t)] + rng.normal(0, 0.05, (n, 2))
    X = np.vstack([a, b])
    return (X - X.mean(0)) / X.std(0)

def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
def relu(z):    return np.maximum(0.0, z)
def he(shape, fan_in): return rng.normal(0, np.sqrt(2.0/fan_in), shape)

Z, H = 2, 128                                  # latent dim, hidden width
G = {'W1': he((Z, H), Z), 'b1': np.zeros(H),
     'W2': he((H, H), H), 'b2': np.zeros(H),
     'W3': he((H, 2), H), 'b3': np.zeros(2)}
D = {'W1': he((2, H), 2), 'b1': np.zeros(H),
     'W2': he((H, H), H), 'b2': np.zeros(H),
     'W3': he((H, 1), H), 'b3': np.zeros(1)}

def make_adam(P): return {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in P.items()}
oG, oD = make_adam(G), make_adam(D)
def adam(P, opt, grads, step, lr=2e-4):
    for k in P:
        m, v = opt[k]
        m[...] = 0.9*m + 0.1*grads[k]
        v[...] = 0.999*v + 0.001*grads[k]**2
        P[k] -= lr * (m/(1-0.9**step)) / (np.sqrt(v/(1-0.999**step)) + 1e-8)

def gen_forward(z):
    z1 = z  @ G['W1'] + G['b1']; h1 = relu(z1)
    z2 = h1 @ G['W2'] + G['b2']; h2 = relu(z2)
    x  = h2 @ G['W3'] + G['b3']
    return x, (z, z1, h1, z2, h2)

def disc_forward(x):
    z1 = x  @ D['W1'] + D['b1']; h1 = relu(z1)
    z2 = h1 @ D['W2'] + D['b2']; h2 = relu(z2)
    logit = h2 @ D['W3'] + D['b3']
    return logit, (x, z1, h1, z2, h2)

def disc_backward(cache, d_logit):
    x, z1, h1, z2, h2 = cache
    g = {}
    g['W3'] = h2.T @ d_logit; g['b3'] = d_logit.sum(0)
    dh2 = d_logit @ D['W3'].T * (z2 > 0)
    g['W2'] = h1.T @ dh2; g['b2'] = dh2.sum(0)
    dh1 = dh2 @ D['W2'].T * (z1 > 0)
    g['W1'] = x.T @ dh1; g['b1'] = dh1.sum(0)
    dx = dh1 @ D['W1'].T                       # gradient w.r.t. D's input (the fake)
    return g, dx

def gen_backward(cache, dx):
    z, z1, h1, z2, h2 = cache
    g = {}
    g['W3'] = h2.T @ dx; g['b3'] = dx.sum(0)
    dh2 = dx @ G['W3'].T * (z2 > 0)
    g['W2'] = h1.T @ dh2; g['b2'] = dh2.sum(0)
    dh1 = dh2 @ G['W2'].T * (z1 > 0)
    g['W1'] = z.T @ dh1; g['b1'] = dh1.sum(0)
    return g

def train(data, steps=8000, bs=256):
    for step in range(1, steps+1):
        # ---- train discriminator on real (label 1) and fake (label 0) -------
        real = data[rng.integers(0, len(data), bs)]
        z = rng.normal(size=(bs, Z))
        fake, gcache = gen_forward(z)
        lr_, cr = disc_forward(real)
        lf_, cf = disc_forward(fake)
        # BCE grads w.r.t logits:  d/dlogit = sigmoid(logit) - label
        gr, _ = disc_backward(cr, (sigmoid(lr_) - 1.0) / bs)
        gf, _ = disc_backward(cf, (sigmoid(lf_) - 0.0) / bs)
        adam(D, oD, {k: gr[k] + gf[k] for k in D}, step)
        # ---- train generator: push fakes toward "real" (non-saturating) -----
        z = rng.normal(size=(bs, Z))
        fake, gcache = gen_forward(z)
        lf_, cf = disc_forward(fake)
        _, dx = disc_backward(cf, (sigmoid(lf_) - 1.0) / bs)   # want D to say "real"
        adam(G, oG, gen_backward(gcache, dx), step)
        if step % 2000 == 0:
            acc = 0.5*((lr_ > 0).mean() + (lf_ < 0).mean())
            print(f"step {step:5d}   D-accuracy {acc:.2f}  (0.5 = perfectly fooled)")

def main():
    data = two_moons(2000)
    print("Training a from-scratch GAN on two-moons ...")
    train(data)
    z = rng.normal(size=(2000, Z)); gen, _ = gen_forward(z)
    print("Generated", len(gen), "samples. mean:", gen.mean(0).round(2),
          " std:", gen.std(0).round(2))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
        ax[0].scatter(data[:,0], data[:,1], s=6, c="#2563eb"); ax[0].set_title("real")
        ax[1].scatter(gen[:,0],  gen[:,1],  s=6, c="#b91c1c"); ax[1].set_title("generated (GAN)")
        for a in ax: a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        fig.savefig("gan_samples.png", dpi=110, bbox_inches="tight")
        print("Saved gan_samples.png")
    except Exception as e:
        print("(plotting skipped:", e, ")")

    # --- Try it yourself -----------------------------------------------------
    # 1) Train D 2x per G step. Does the generator collapse to one blob?
    # 2) Switch G's loss to the saturating form -log(1 - D(fake)). Watch it stall.

if __name__ == "__main__":
    main()
