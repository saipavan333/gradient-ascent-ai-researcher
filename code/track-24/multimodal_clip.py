"""
Track 20 · Lesson 2 — Multimodal models (CLIP)
==============================================
CLIP learns images and text in ONE shared space by contrastive training: matching
image-caption pairs are pulled together, mismatched pairs pushed apart. The payoff is
ZERO-SHOT classification — to label a new image, compare it to text prompts like
"a photo of a cat" and pick the nearest, with no task-specific training. Here we align
toy image and text vectors and classify images by their nearest text prompt.

Run:  python multimodal_clip.py
"""
import numpy as np
rng = np.random.default_rng(0)

def normalise(v): return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
def softmax_rows(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

if __name__ == "__main__":
    K, d, m = 6, 12, 8                              # concepts, raw dim, shared-space dim
    concept_img = rng.standard_normal((K, d))       # each concept's image "look"
    concept_txt = rng.standard_normal((K, d))       # each concept's text "description"
    Wi = rng.standard_normal((d, m)) * 0.3          # image encoder
    Wt = rng.standard_normal((d, m)) * 0.3          # text encoder
    tau = 0.1; I = np.eye(K)
    for step in range(4000):
        imgs = concept_img + 0.15 * rng.standard_normal((K, d))   # a batch: one image+text per concept
        txts = concept_txt + 0.15 * rng.standard_normal((K, d))
        Ei = normalise(imgs @ Wi); Et = normalise(txts @ Wt)
        S = Ei @ Et.T / tau                          # (K,K) similarities; diagonal = matched pairs
        Gi = (softmax_rows(S) - I) / K               # image→text: image i should match text i
        Gt = (softmax_rows(S.T) - I) / K             # text→image (symmetric)
        Wi -= 0.05 * imgs.T @ (Gi @ Et) / tau
        Wt -= 0.05 * txts.T @ (Gt @ Ei) / tau
    # zero-shot: label each image by its nearest text prompt (the concept "class names")
    txts = normalise(concept_txt @ Wt); correct = 0; sm = []; mm = []
    for c in range(K):
        for _ in range(60):
            img = normalise((concept_img[c] + 0.15 * rng.standard_normal(d)) @ Wi)
            s = txts @ img; correct += (s.argmax() == c)
            sm.append(s[c]); mm.append(np.delete(s, c).mean())
    acc = correct / (K * 60)
    print(f"zero-shot accuracy (label image by nearest text prompt): {acc:.3f}")
    print(f"avg similarity — matched pair {np.mean(sm):.2f}  vs mismatched {np.mean(mm):.2f}")
    assert acc > 0.9 and np.mean(sm) > np.mean(mm) + 0.2
    print("OK — images and text share one space; new images are classified by text prompts, zero-shot.")
