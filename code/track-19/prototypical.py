"""
Track 16 · Lesson 2 — Prototypical networks
===========================================
Prototypical networks are the cleanest few-shot method: embed the support examples,
take the MEAN embedding of each class as its "prototype", and classify a query by
which prototype it's nearest to (turning distances into class probabilities with a
softmax). Simple, fast, and a very strong baseline.

Run:  python prototypical.py
"""
import numpy as np
rng = np.random.default_rng(0)

def episode(n_way=5, k_shot=5, n_query=15, dim=16, sep=2.0):
    C = rng.standard_normal((n_way, dim)) * sep
    S = np.vstack([C[c] + rng.standard_normal((k_shot, dim)) for c in range(n_way)])
    Sy = np.repeat(range(n_way), k_shot)
    Q = np.vstack([C[c] + rng.standard_normal((n_query, dim)) for c in range(n_way)])
    Qy = np.repeat(range(n_way), n_query)
    return S, Sy, Q, Qy

def proto_classify(S, Sy, Q, n_way):
    protos = np.array([S[Sy == c].mean(0) for c in range(n_way)])   # class prototypes
    d = ((Q[:, None] - protos[None]) ** 2).sum(2)                   # squared distances
    logits = -d                                                    # nearer = higher score
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)                                    # softmax over classes
    return p.argmax(1), p

if __name__ == "__main__":
    accs = []
    for _ in range(300):
        S, Sy, Q, Qy = episode()
        pred, _ = proto_classify(S, Sy, Q, 5)
        accs.append((pred == Qy).mean())
    print(f"5-way 5-shot prototypical accuracy: {np.mean(accs):.3f}")
    # confidence: the softmax puts most mass on the right class
    S, Sy, Q, Qy = episode()
    _, p = proto_classify(S, Sy, Q, 5)
    print(f"average probability assigned to the true class: {p[np.arange(len(Qy)), Qy].mean():.3f}")
    assert np.mean(accs) > 0.8
    print("OK — embed, average into prototypes, classify by nearest prototype. That's a protonet.")
