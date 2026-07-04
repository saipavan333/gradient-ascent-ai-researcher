"""
Track 11 · Lesson 3 — KV-cache: why LLM generation is fast, from scratch
Run:  python kv_cache.py

Generating token t with self-attention needs the keys/values of ALL previous tokens.
The NAIVE way recomputes every past key/value at every step — O(T^2) work to produce
T tokens. The KV-CACHE stores past keys/values and computes only the NEW token's k,v
each step — O(T) work. We implement both, verify identical output, and count the work.
"""
import numpy as np
rng = np.random.default_rng(0)

d = 16
Wq, Wk, Wv = [rng.normal(0, 1/np.sqrt(d), (d, d)) for _ in range(3)]
def softmax(z): z = z - z.max(); e = np.exp(z); return e / e.sum()

def attn_step(q, K, V):
    w = softmax(K @ q / np.sqrt(d))          # attend query to all cached keys
    return w @ V

def generate_naive(embs):
    """At each step recompute K,V for the whole prefix (wasteful)."""
    out, work = [], 0
    for t in range(1, len(embs) + 1):
        pref = embs[:t]
        K = pref @ Wk; V = pref @ Wv; q = embs[t-1] @ Wq     # recomputes t keys/values
        work += t                                            # t token-projections
        out.append(attn_step(q, K, V))
    return np.array(out), work

def generate_cached(embs):
    """Keep past K,V; compute only the new token's k,v each step."""
    K_cache, V_cache, out, work = [], [], [], 0
    for t in range(len(embs)):
        x = embs[t]
        K_cache.append(x @ Wk); V_cache.append(x @ Wv); q = x @ Wq   # ONE new k,v
        work += 1                                                    # one projection
        out.append(attn_step(q, np.array(K_cache), np.array(V_cache)))
    return np.array(out), work


def quantize_int8(W):
    """Quantize float weights to 8-bit integers (per-tensor). Returns (q, scale)."""
    scale = np.abs(W).max() / 127.0
    q = np.round(W / scale).astype(np.int8)      # 1 byte/weight instead of 4
    return q, scale

def dequantize(q, scale):
    return q.astype(np.float32) * scale

def demo_quantization():
    W = rng.normal(size=(512, 512)).astype(np.float32)
    q, scale = quantize_int8(W); Wq = dequantize(q, scale)
    err = np.abs(W - Wq).mean()
    print("\nInt8 quantization of a 512x512 weight matrix:")
    print(f"  memory: {W.nbytes/1024:.0f} KB (float32) -> {q.nbytes/1024:.0f} KB (int8)  = 4x smaller")
    print(f"  mean abs error introduced: {err:.5f}  (tiny relative to the weights)")

def main():
    T = 64
    embs = rng.normal(size=(T, d))
    o1, w1 = generate_naive(embs)
    o2, w2 = generate_cached(embs)
    print(f"Generating {T} tokens with single-head self-attention:\n")
    print(f"  identical output: {np.allclose(o1, o2)}")
    print(f"  naive  key/value projections: {w1}   (~T^2/2)")
    print(f"  cached key/value projections: {w2}   (= T)")
    print(f"  speedup in this work metric : {w1/w2:.1f}x  (grows with sequence length)")
    demo_quantization()
    print("\nCache to avoid recompute; quantize to shrink memory — the two pillars of fast, cheap inference.")

if __name__ == "__main__":
    main()
