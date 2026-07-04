"""
Track 5 · Lesson 0 — From sequences to attention (tokenization & embeddings)
Run:  python tokenize.py
A model can't read text — only numbers. Tokenization maps text to integer ids;
an embedding table maps each id to a learned vector (Track 1) the model works with.
"""
import numpy as np


def main():
    text = "the cat sat"

    # Character-level tokenization: build a vocabulary, map chars <-> ids
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}     # string -> id
    itos = {i: c for c, i in stoi.items()}          # id -> string
    ids = [stoi[c] for c in text]
    print("vocab:", chars)
    print("ids  :", ids)
    print("back :", "".join(itos[i] for i in ids))   # round-trips to the text

    # Embedding table: each token id -> a learned vector in R^d (here random)
    vocab_size, d = len(chars), 6
    embed = np.random.default_rng(0).normal(0, 1, (vocab_size, d))
    seq = embed[ids]                                 # (sequence_length, d)
    print("embedded sequence shape:", seq.shape)     # (11, 6)
    print("vector for first token:", np.round(seq[0], 2))

    # Real LLMs tokenize into 'subwords' (e.g. byte-pair encoding), not characters,
    # so common words are one token and rare words split into pieces -- but the
    # idea is identical: text -> ids -> vectors.

    # --- Try it yourself -----------------------------------------------------
    # 1) Tokenize a sentence of your own; confirm encode then decode round-trips.


if __name__ == "__main__":
    main()
