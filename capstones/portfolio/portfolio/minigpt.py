"""
A minimal char-level Transformer (single causal self-attention head) in pure NumPy,
with hand-derived backprop. Trained on a list of names, it generates plausible new
ones. This is the same architecture you built in Track 5 of the AI Researcher Path,
packaged for serving. No deep-learning framework required.
"""
import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

class MiniGPT:
    """One embedding + one causal self-attention head + an output projection."""
    def __init__(self, vocab, T, d, seed=0):
        r = np.random.default_rng(seed); s = 1.0 / np.sqrt(d)
        self.V, self.T, self.d = vocab, T, d
        self.Wemb = r.normal(0, s, (vocab, d)); self.Wpos = r.normal(0, s, (T, d))
        self.Wq = r.normal(0, s, (d, d)); self.Wk = r.normal(0, s, (d, d)); self.Wv = r.normal(0, s, (d, d))
        self.Wo = r.normal(0, s, (d, vocab)); self.mask = np.triu(np.full((T, T), -1e9), 1)

    def forward(self, x, y=None):
        T, d = len(x), self.d
        E = self.Wemb[x] + self.Wpos[:T]
        Q, K, Vv = E @ self.Wq, E @ self.Wk, E @ self.Wv
        A = softmax((Q @ K.T) / np.sqrt(d) + self.mask[:T, :T], axis=1)
        C = A @ Vv; logits = C @ self.Wo; P = softmax(logits, axis=1)
        cache = (x, E, Q, K, Vv, A, C, P)
        if y is None:
            return logits, None, cache
        return logits, -np.mean(np.log(P[np.arange(T), y] + 1e-12)), cache

    def backward(self, cache, y):
        x, E, Q, K, Vv, A, C, P = cache; T, d = len(x), self.d
        dlogits = P.copy(); dlogits[np.arange(T), y] -= 1; dlogits /= T
        dWo = C.T @ dlogits; dC = dlogits @ self.Wo.T
        dA = dC @ Vv.T; dVv = A.T @ dC
        dS = A * (dA - (dA * A).sum(axis=1, keepdims=True)); dS /= np.sqrt(d)
        dQ = dS @ K; dK = dS.T @ Q
        dWq = E.T @ dQ; dWk = E.T @ dK; dWv = E.T @ dVv
        dE = dQ @ self.Wq.T + dK @ self.Wk.T + dVv @ self.Wv.T
        dWpos = np.zeros_like(self.Wpos); dWpos[:T] = dE
        dWemb = np.zeros_like(self.Wemb); np.add.at(dWemb, x, dE)
        return {"Wemb": dWemb, "Wpos": dWpos, "Wq": dWq, "Wk": dWk, "Wv": dWv, "Wo": dWo}

PARAMS = ["Wemb", "Wpos", "Wq", "Wk", "Wv", "Wo"]

def fit(text, T=16, d=48, steps=4000, lr=3e-3, seed=0, log=print):
    """Train on a raw text stream (names separated by newlines)."""
    chars = sorted(set(text)); V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
    data = np.array([stoi[c] for c in text]); rng = np.random.default_rng(seed)
    m = MiniGPT(V, T, d, seed)
    mom = {n: np.zeros_like(getattr(m, n)) for n in PARAMS}
    vel = {n: np.zeros_like(getattr(m, n)) for n in PARAMS}
    b1, b2, eps = 0.9, 0.999, 1e-8
    for step in range(1, steps + 1):
        i = rng.integers(0, len(data) - T - 1); x = data[i:i+T]; y = data[i+1:i+T+1]
        _, loss, cache = m.forward(x, y); grads = m.backward(cache, y)
        for n in PARAMS:
            g = grads[n]; mom[n] = b1*mom[n] + (1-b1)*g; vel[n] = b2*vel[n] + (1-b2)*g*g
            mh = mom[n]/(1-b1**step); vh = vel[n]/(1-b2**step)
            getattr(m, n)[...] -= lr * mh / (np.sqrt(vh) + eps)
        if log and (step % 1000 == 0 or step == 1):
            log(f"step {step:5d}  loss {loss:.3f}")
    return m, stoi, itos

def save(path, m, stoi, itos):
    np.savez(path, T=m.T, d=m.d, V=m.V, chars="".join(itos[i] for i in range(len(itos))),
             **{n: getattr(m, n) for n in PARAMS})

def load(path):
    z = np.load(path, allow_pickle=True)
    chars = str(z["chars"]); stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for i, c in enumerate(chars)}
    m = MiniGPT(int(z["V"]), int(z["T"]), int(z["d"]))
    for n in PARAMS: setattr(m, n, z[n])
    return m, stoi, itos

def sample_name(m, stoi, itos, temperature=0.9, max_len=18, seed=None, prefix=""):
    """Generate one name. Starts from the newline boundary; stops at the next newline."""
    rng = np.random.default_rng(seed)
    nl = stoi["\n"]
    ids = [nl] + [stoi[c] for c in prefix.lower() if c in stoi]
    out = list(prefix.lower())
    for _ in range(max_len):
        ctx = ids[-m.T:]
        logits, _, _ = m.forward(np.array(ctx))
        p = softmax(logits[-1] / max(1e-3, temperature))
        nxt = int(rng.choice(len(p), p=p))
        if nxt == nl:
            break
        ids.append(nxt); out.append(itos[nxt])
    return "".join(out).capitalize()
