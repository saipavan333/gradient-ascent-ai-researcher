"""
Track 13 · Lesson 4 — Compression: entropy is the limit, Huffman reaches it
Run:  python compression.py

Shannon's source coding theorem: you cannot compress data below its entropy (bits per
symbol), and you can get arbitrarily close. HUFFMAN CODING builds an optimal prefix
code — frequent symbols get short codes — landing within 1 bit of the entropy. A
language model is, in this exact sense, a compressor: better predictions = fewer bits.
"""
import heapq
from collections import Counter
import numpy as np

def entropy_bits(freqs):
    total = sum(freqs.values()); p = np.array([f/total for f in freqs.values()])
    return float(-(p*np.log2(p)).sum())

def huffman(freqs):
    # each node: [frequency, list of (symbol, code-so-far)]; merge two lowest repeatedly
    nodes = [[f, [(sym, "")]] for sym, f in freqs.items()]
    heapq.heapify(nodes)
    while len(nodes) > 1:
        lo = heapq.heappop(nodes); hi = heapq.heappop(nodes)
        merged = [[s, "0"+c] for s, c in lo[1]] + [[s, "1"+c] for s, c in hi[1]]
        heapq.heappush(nodes, [lo[0]+hi[0], [(s, c) for s, c in merged]])
    return dict(nodes[0][1])

def main():
    text = ("the quick brown fox jumps over the lazy dog. " * 12 +
            "information theory tells us the true cost of a message. ") * 3
    freqs = Counter(text)
    codes = huffman(freqs)
    avg_len = sum(freqs[s]*len(codes[s]) for s in freqs) / sum(freqs.values())
    H = entropy_bits(freqs)
    print(f"Text: {len(text)} characters, {len(freqs)} distinct symbols\n")
    print(f"  entropy (theoretical limit) : {H:.3f} bits/char")
    print(f"  Huffman average code length : {avg_len:.3f} bits/char   (within 1 bit of H)")
    print(f"  fixed ASCII                 : 8.000 bits/char")
    enc_bits = int(avg_len*len(text))
    print(f"\n  compressed: {enc_bits} bits vs {8*len(text)} bits (ASCII)"
          f"  ->  {100*(1-enc_bits/(8*len(text))):.0f}% smaller")
    print(f"  a few short codes: " + ", ".join(f"'{s}'={codes[s]}" for s in [' ','e','t']))
    print("\nBetter probability estimates -> shorter codes. Prediction and compression are one.")

if __name__ == "__main__":
    main()
