"""
Track 11 · Lesson 4 — Retrieval-Augmented Generation (RAG) from scratch
Run:  python rag.py

An LLM only knows what was in its training data. RAG gives it an external memory:
embed a set of documents as vectors, embed the user's question, retrieve the closest
documents by cosine similarity (the dot product from Track 1), and condition the
answer on them. We build the retrieval core with bag-of-words embeddings.
"""
import numpy as np, re
rng = np.random.default_rng(0)

DOCS = [
    "The Eiffel Tower is located in Paris, the capital of France.",
    "Mount Everest is the highest mountain on Earth, in the Himalayas.",
    "Photosynthesis lets plants convert sunlight into chemical energy.",
    "The Python programming language was created by Guido van Rossum.",
    "Water boils at 100 degrees Celsius at sea-level atmospheric pressure.",
    "The transformer architecture was introduced in the paper 'Attention Is All You Need'.",
]

def tokenize(s): return re.findall(r"[a-z]+", s.lower())
VOCAB = sorted(set(t for d in DOCS for t in tokenize(d)))
IDX = {w: i for i, w in enumerate(VOCAB)}

def embed(text):
    v = np.zeros(len(VOCAB))
    for t in tokenize(text):
        if t in IDX: v[IDX[t]] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v                    # unit vector -> dot product = cosine

DOC_VECS = np.array([embed(d) for d in DOCS])

def retrieve(query, k=1):
    q = embed(query)
    sims = DOC_VECS @ q                          # cosine similarity to every document
    order = np.argsort(-sims)[:k]
    return [(DOCS[i], float(sims[i])) for i in order]

def answer(query):
    doc, sim = retrieve(query, k=1)[0]
    return f"[retrieved · sim={sim:.2f}] {doc}"

def main():
    queries = ["What is the capital of France?",
               "Who created Python?",
               "Where is the tallest mountain?",
               "What paper introduced transformers?"]
    print("Tiny RAG: retrieve the most relevant fact for each question.\n")
    for q in queries:
        print(f"Q: {q}\n   -> {answer(q)}\n")
    print("The LLM would then generate an answer CONDITIONED on the retrieved text,")
    print("grounding it in facts instead of relying on memorized (possibly wrong) knowledge.")

if __name__ == "__main__":
    main()
