"""Unit tests for the MiniGPT name generator. Run: pytest -q"""
import numpy as np
from minigpt.model import MiniGPT, fit, save, load, sample_name, softmax

TINY = "\n" + "\n".join(["ana", "ben", "cara", "dan", "eve", "finn"]) + "\n"

def test_softmax_normalizes():
    p = softmax(np.array([1.0, 2.0, 3.0]))
    assert abs(p.sum() - 1.0) < 1e-9 and (p > 0).all()

def test_forward_shapes():
    m = MiniGPT(vocab=10, T=8, d=16)
    logits, loss, cache = m.forward(np.array([1, 2, 3]), y=np.array([2, 3, 4]))
    assert logits.shape == (3, 10) and loss > 0

def test_training_reduces_loss():
    m, stoi, itos = fit(TINY, T=8, d=16, steps=400, log=None)
    _, loss0, _ = MiniGPT(len(stoi), 8, 16).forward(
        np.array([stoi[c] for c in "ana"]), y=np.array([stoi[c] for c in "na\n"]))
    _, loss1, _ = m.forward(np.array([stoi[c] for c in "ana"]), y=np.array([stoi[c] for c in "na\n"]))
    assert loss1 < loss0          # trained model fits the data better than init

def test_sample_name_returns_clean_string():
    m, stoi, itos = fit(TINY, T=8, d=16, steps=200, log=None)
    name = sample_name(m, stoi, itos, seed=0)
    assert isinstance(name, str) and name == name.capitalize() and "\n" not in name

def test_save_load_roundtrip(tmp_path):
    m, stoi, itos = fit(TINY, T=8, d=16, steps=100, log=None)
    p = str(tmp_path / "m.npz"); save(p, m, stoi, itos)
    m2, stoi2, itos2 = load(p)
    x = np.array([1, 2, 3])
    assert np.allclose(m.forward(x)[0], m2.forward(x)[0]) and stoi == stoi2
