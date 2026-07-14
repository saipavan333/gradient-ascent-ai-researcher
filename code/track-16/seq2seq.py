"""
Track 16 · Lesson 4 — Sequence-to-sequence: encoder → context → decoder
=======================================================================
Classification reads a sequence and emits one label. But translation,
summarization, and speech emit a whole SEQUENCE. The classic architecture is
encoder–decoder: an encoder RNN compresses the input into a single fixed-length
"context" vector, and a decoder RNN unrolls that vector into the output sequence,
one token at a time (feeding its previous output back in).

We train a tiny seq2seq to REVERSE a short sequence of symbols. It works — and it
exposes the flaw that led to modern NLP: everything the decoder knows must squeeze
through one fixed vector, the "context bottleneck." Relaxing that bottleneck by
letting the decoder look back at all encoder states is exactly attention (Track 5)
— which is why Transformers replaced RNNs.

Run:  python seq2seq.py
"""
import numpy as np

rng = np.random.default_rng(3)

V = 6          # symbol vocabulary 0..5
L = 4          # sequence length
GO = V         # start-of-decode token (input dim is V+1)
H = 48


def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def onehot(i, n):
    v = np.zeros(n); v[i] = 1.0; return v


def make_pair():
    src = rng.integers(0, V, size=L)
    tgt = src[::-1].copy()          # the target is the reversed source
    return src, tgt


class Seq2Seq:
    def __init__(self):
        s = 0.3
        # encoder (input one-hot dim V)
        self.Wxh_e = rng.standard_normal((H, V)) * s
        self.Whh_e = np.eye(H) * 0.5 + rng.standard_normal((H, H)) * 0.05
        self.b_e = np.zeros(H)
        # decoder (input one-hot dim V+1, includes GO)
        self.Wxh_d = rng.standard_normal((H, V + 1)) * s
        self.Whh_d = np.eye(H) * 0.5 + rng.standard_normal((H, H)) * 0.05
        self.b_d = np.zeros(H)
        self.Wout = rng.standard_normal((V, H)) * s
        self.b_out = np.zeros(V)
        self._names = ["Wxh_e", "Whh_e", "b_e", "Wxh_d", "Whh_d", "b_d", "Wout", "b_out"]

    def _encode(self, src):
        h = np.zeros(H); enc = []
        for t in range(L):
            x = onehot(src[t], V)
            hp = h
            h = np.tanh(self.Wxh_e @ x + self.Whh_e @ h + self.b_e)
            enc.append((x, hp, h))
        return h, enc

    def loss_and_grad(self, src, tgt):
        context, enc = self._encode(src)
        # decoder with teacher forcing
        h = context; dec = []; loss = 0.0
        for k in range(L):
            din = onehot(GO if k == 0 else tgt[k - 1], V + 1)
            hp = h
            h = np.tanh(self.Wxh_d @ din + self.Whh_d @ h + self.b_d)
            p = softmax(self.Wout @ h + self.b_out)
            loss += -np.log(p[tgt[k]] + 1e-9)
            dec.append((din, hp, h, p))

        g = {n: np.zeros_like(getattr(self, n)) for n in self._names}
        dh_next = np.zeros(H)
        for k in reversed(range(L)):
            din, hp, h, p = dec[k]
            dlogits = p.copy(); dlogits[tgt[k]] -= 1.0
            g["Wout"] += np.outer(dlogits, h); g["b_out"] += dlogits
            dh = self.Wout.T @ dlogits + dh_next
            da = (1 - h ** 2) * dh
            g["b_d"] += da
            g["Wxh_d"] += np.outer(da, din)
            g["Whh_d"] += np.outer(da, hp)
            dh_next = self.Whh_d.T @ da
        dctx = dh_next                          # gradient into the context vector
        dh = dctx
        for t in reversed(range(L)):
            x, hp, h = enc[t]
            da = (1 - h ** 2) * dh
            g["b_e"] += da
            g["Wxh_e"] += np.outer(da, x)
            g["Whh_e"] += np.outer(da, hp)
            dh = self.Whh_e.T @ da
        return loss, g

    def step(self, g, lr, clip=5.0):
        total = np.sqrt(sum(np.sum(v * v) for v in g.values()))
        scale = min(1.0, clip / (total + 1e-9))
        for n in self._names:
            setattr(self, n, getattr(self, n) - lr * scale * g[n])

    def greedy_decode(self, src):
        context, _ = self._encode(src)
        h = context; out = []; prev = GO
        for k in range(L):
            din = onehot(prev, V + 1)
            h = np.tanh(self.Wxh_d @ din + self.Whh_d @ h + self.b_d)
            p = softmax(self.Wout @ h + self.b_out)
            prev = int(np.argmax(p)); out.append(prev)
        return out


def evaluate(model, n=300):
    seq_ok = tok_ok = tot_tok = 0
    for _ in range(n):
        src, tgt = make_pair()
        pred = model.greedy_decode(src)
        seq_ok += int(list(pred) == list(tgt))
        tok_ok += sum(int(a == b) for a, b in zip(pred, tgt)); tot_tok += L
    return seq_ok / n, tok_ok / tot_tok


if __name__ == "__main__":
    model = Seq2Seq()
    print(f"Task: reverse a length-{L} sequence over {V} symbols (context vector = {H} dims).")
    s0, t0 = make_pair()
    print(f"example: {list(s0)} -> target {list(t0)}, model {model.greedy_decode(s0)} (untrained)\n")
    lr = 0.08
    for ep in range(40):
        for _ in range(300):
            src, tgt = make_pair()
            _, g = model.loss_and_grad(src, tgt); model.step(g, lr)
        if (ep + 1) % 10 == 0:
            sa, ta = evaluate(model, 200)
            print(f"epoch {ep+1:2d}: exact-sequence acc = {sa:.3f}, token acc = {ta:.3f}")
    sa, ta = evaluate(model, 400)
    print(f"\nFinal: exact-sequence acc = {sa:.3f}, token acc = {ta:.3f}")
    src, tgt = make_pair()
    print(f"demo: {list(src)} -> {model.greedy_decode(src)} (target {list(tgt)})")
    assert ta > 0.9, "seq2seq should learn to reverse short sequences"
    print("OK — encoder compressed the input to one vector; the decoder unrolled it back, reversed.")
