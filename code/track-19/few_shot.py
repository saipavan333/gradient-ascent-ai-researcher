"""
Track 16 (Meta / Few-Shot / Continual) · Lesson 0 — Few-shot learning
=====================================================================
Humans recognise a new animal from ONE picture. Standard deep learning needs
thousands of labels per class. FEW-SHOT learning is the setting where each new
class comes with only a handful (K) of examples. The episodic setup: an "N-way
K-shot" task gives K labelled support examples for each of N new classes, and asks
you to label fresh query examples. A strong baseline: classify a query by the
nearest class centroid in a good feature space.

Run:  python few_shot.py
"""
import numpy as np
rng = np.random.default_rng(0)

def make_episode(n_way=5, k_shot=1, n_query=15, dim=8, sep=1.3):
    centres = rng.standard_normal((n_way, dim)) * sep      # each class = a cluster centre
    S, Sy, Q, Qy = [], [], [], []
    for c in range(n_way):
        S.append(centres[c] + rng.standard_normal((k_shot, dim)))
        Sy += [c] * k_shot
        Q.append(centres[c] + rng.standard_normal((n_query, dim)))
        Qy += [c] * n_query
    return np.vstack(S), np.array(Sy), np.vstack(Q), np.array(Qy)

def nearest_centroid_accuracy(S, Sy, Q, Qy, n_way):
    protos = np.array([S[Sy == c].mean(0) for c in range(n_way)])   # centroid per class
    d = ((Q[:, None, :] - protos[None, :, :]) ** 2).sum(2)          # dist to each centroid
    pred = d.argmin(1)
    return (pred == Qy).mean()

if __name__ == "__main__":
    for k in (1, 5):
        accs = [nearest_centroid_accuracy(*make_episode(k_shot=k), n_way=5) for _ in range(200)]
        print(f"5-way {k}-shot accuracy (nearest centroid): {np.mean(accs):.3f}")
    a1 = np.mean([nearest_centroid_accuracy(*make_episode(k_shot=1), n_way=5) for _ in range(200)])
    a5 = np.mean([nearest_centroid_accuracy(*make_episode(k_shot=5), n_way=5) for _ in range(200)])
    assert a5 > a1 + 0.05 and a1 > 0.6
    print("OK — more shots -> better centroids -> higher accuracy. Few-shot works with a good representation.")
