"""Minimal char-level transformer (single causal self-attention head) in NumPy,
with hand-derived backprop (gradient-checked). Trains on a tiny corpus and
generates text — a 'nano-GPT' you can read end to end."""
import numpy as np

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True); e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

class MiniGPT:
    def __init__(self, vocab, T, d, seed=0):
        r = np.random.default_rng(seed); s = 1.0/np.sqrt(d)
        self.V, self.T, self.d = vocab, T, d
        self.Wemb = r.normal(0, s, (vocab, d)); self.Wpos = r.normal(0, s, (T, d))
        self.Wq = r.normal(0, s, (d, d)); self.Wk = r.normal(0, s, (d, d)); self.Wv = r.normal(0, s, (d, d))
        self.Wo = r.normal(0, s, (d, vocab)); self.mask = np.triu(np.full((T, T), -1e9), 1)
    def forward(self, x, y=None):
        T, d = len(x), self.d
        E = self.Wemb[x] + self.Wpos[:T]
        Q, K, Vv = E@self.Wq, E@self.Wk, E@self.Wv
        A = softmax((Q@K.T)/np.sqrt(d) + self.mask[:T,:T], axis=1)
        C = A@Vv; logits = C@self.Wo; P = softmax(logits, axis=1)
        cache = (x, E, Q, K, Vv, A, C, P)
        if y is None: return logits, None, cache
        return logits, -np.mean(np.log(P[np.arange(T), y] + 1e-12)), cache
    def backward(self, cache, y):
        x, E, Q, K, Vv, A, C, P = cache; T, d = len(x), self.d
        dlogits = P.copy(); dlogits[np.arange(T), y] -= 1; dlogits /= T
        dWo = C.T@dlogits; dC = dlogits@self.Wo.T
        dA = dC@Vv.T; dVv = A.T@dC
        dS = A*(dA - (dA*A).sum(axis=1, keepdims=True)); dS /= np.sqrt(d)
        dQ = dS@K; dK = dS.T@Q
        dWq = E.T@dQ; dWk = E.T@dK; dWv = E.T@dVv
        dE = dQ@self.Wq.T + dK@self.Wk.T + dVv@self.Wv.T
        dWpos = np.zeros_like(self.Wpos); dWpos[:T] = dE
        dWemb = np.zeros_like(self.Wemb); np.add.at(dWemb, x, dE)
        return {"Wemb":dWemb,"Wpos":dWpos,"Wq":dWq,"Wk":dWk,"Wv":dWv,"Wo":dWo}

def train(text, T=16, d=48, steps=3000, lr=3e-3, seed=0):
    chars = sorted(set(text)); V = len(chars)
    stoi = {c:i for i,c in enumerate(chars)}; itos = {i:c for c,i in stoi.items()}
    data = np.array([stoi[c] for c in text]); rng = np.random.default_rng(seed)
    m = MiniGPT(V, T, d, seed)
    names = ["Wemb","Wpos","Wq","Wk","Wv","Wo"]
    mom = {n: np.zeros_like(getattr(m,n)) for n in names}; vel = {n: np.zeros_like(getattr(m,n)) for n in names}
    b1,b2,eps = 0.9,0.999,1e-8
    for step in range(1, steps+1):
        i = rng.integers(0, len(data)-T-1); x = data[i:i+T]; y = data[i+1:i+T+1]
        _, loss, cache = m.forward(x, y); grads = m.backward(cache, y)
        for n in names:
            g = grads[n]; mom[n] = b1*mom[n]+(1-b1)*g; vel[n] = b2*vel[n]+(1-b2)*g*g
            mh = mom[n]/(1-b1**step); vh = vel[n]/(1-b2**step)
            getattr(m,n)[...] -= lr*mh/(np.sqrt(vh)+eps)
        if step % 600 == 0 or step == 1: print(f"step {step:4d}  loss {loss:.3f}")
    return m, stoi, itos

def generate(m, stoi, itos, prompt, n=80, seed=0):
    rng = np.random.default_rng(seed); ids = [stoi[c] for c in prompt]
    for _ in range(n):
        ctx = ids[-m.T:]; logits,_,_ = m.forward(np.array(ctx))
        p = softmax(logits[-1]); nxt = rng.choice(len(p), p=p); ids.append(nxt)
    return "".join(itos[i] for i in ids)

if __name__ == "__main__":
    text = ("to be or not to be that is the question. " * 20)
    m, stoi, itos = train(text, steps=3000)
    print("sample:", repr(generate(m, stoi, itos, "to be", n=60)))
