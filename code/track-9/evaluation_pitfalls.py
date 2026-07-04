"""
Track 9 · Lesson 2 — Why "best on the test set" lies (adaptive overfitting)
Run:  python evaluation_pitfalls.py

If you try many models and KEEP the one with the best TEST accuracy, that number is
optimistically biased — you've used the test set to choose, so it no longer measures
generalization. We show the inflation grows with the number of models tried, and that
a held-out FINAL test set restores an honest estimate.
"""
import numpy as np
rng = np.random.default_rng(0)

def experiment(K, n_test=200, n_final=200):
    """Try K random 'models'; pick the best on the test set; report both estimates."""
    truth = 0.5                                   # all models are pure chance
    best_acc, best_i = -1, -1
    test_labels  = rng.integers(0, 2, n_test)
    final_labels = rng.integers(0, 2, n_final)
    best_final = None
    for i in range(K):
        preds_test  = rng.integers(0, 2, n_test)   # a random 'model'
        acc = (preds_test == test_labels).mean()
        if acc > best_acc:
            best_acc, best_i = acc, i
            best_final = (rng.integers(0, 2, n_final) == final_labels).mean()
    return best_acc, best_final                    # test-selected acc, fresh-data acc

if __name__ == "__main__":
    print("Pick the best of K random models BY TEST ACCURACY (true accuracy = 0.50):\n")
    print(f"{'K models tried':>15} {'best test acc':>14} {'same model, fresh test':>24}")
    for K in [1, 5, 20, 100, 500, 2000]:
        picks = [experiment(K) for _ in range(40)]
        sel = np.mean([p[0] for p in picks]); fresh = np.mean([p[1] for p in picks])
        print(f"{K:>15} {sel:>14.3f} {fresh:>24.3f}")
    print("\nThe selected 'best test accuracy' inflates far above 0.50 as K grows —")
    print("pure selection noise. On a FRESH test set the same model is ~0.50 again.")
    print("Lesson: never select models on the set you report; keep a final hold-out.")
