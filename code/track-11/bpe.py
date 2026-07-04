"""
Track 11 · Lesson 0 — Byte-Pair Encoding (BPE) tokenizer from scratch
Run:  python bpe.py

Real language models don't use characters or whole words — they use SUBWORD tokens
learned by Byte-Pair Encoding. Start from characters, then repeatedly merge the most
frequent adjacent pair into a new token. Common chunks ("th", "the", "ing") become
single tokens, so text compresses and rare words still decompose into known pieces.
"""
from collections import Counter

def get_stats(words):
    """Count adjacent symbol pairs across the corpus (weighted by word frequency)."""
    pairs = Counter()
    for word, freq in words.items():
        syms = word.split()
        for a, b in zip(syms[:-1], syms[1:]):
            pairs[(a, b)] += freq
    return pairs

def merge(words, pair):
    """Merge every occurrence of `pair` into a single symbol."""
    bigram = " ".join(pair); joined = "".join(pair)
    return {w.replace(bigram, joined): f for w, f in words.items()}

def train_bpe(corpus, num_merges=30):
    # each word -> space-separated characters, with an end-of-word marker '_'
    words = Counter(corpus.split())
    words = {" ".join(list(w) + ["_"]): f for w, f in words.items()}
    merges = []
    for _ in range(num_merges):
        stats = get_stats(words)
        if not stats: break
        best = max(stats, key=stats.get)
        words = merge(words, best); merges.append(best)
    return merges

def encode_word(word, merges):
    syms = list(word) + ["_"]
    for a, b in merges:                      # apply learned merges in order
        i = 0
        while i < len(syms) - 1:
            if syms[i] == a and syms[i+1] == b:
                syms[i:i+2] = [a + b]
            else:
                i += 1
    return syms

def main():
    corpus = ("the theory of the theme is the thesis theme " * 40 +
              "learning the deeper meaning meanwhile leaning ") * 3
    merges = train_bpe(corpus, num_merges=20)
    print("First 12 learned merges (most frequent pairs, in order):")
    for i, (a, b) in enumerate(merges[:12]):
        print(f"  {i+1:2d}. '{a}' + '{b}'  ->  '{a+b}'")
    # compression: characters vs BPE tokens
    words = corpus.split()
    char_len = sum(len(w) + 1 for w in words)
    bpe_len = sum(len(encode_word(w, merges)) for w in words)
    print(f"\nTokens to represent the corpus:")
    print(f"  character-level : {char_len}")
    print(f"  BPE ({len(merges)} merges) : {bpe_len}   ({100*(1-bpe_len/char_len):.0f}% fewer)")
    # a rare word still decomposes into known pieces (no [UNK])
    print("\nRare word 'themelessness' ->", encode_word("themelessness", merges))
    print("Round-trip check:", "".join(encode_word("theory", merges)).replace("_","") == "theory")

if __name__ == "__main__":
    main()
