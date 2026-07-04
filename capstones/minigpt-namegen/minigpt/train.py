"""Train the name generator and save weights to model.npz.  Run:  python -m minigpt.train"""
import os
from .model import fit, save, sample_name

def main(data_path="data/names.txt", out="model.npz", steps=4000):
    names = [n.strip().lower() for n in open(data_path) if n.strip()]
    text = "\n" + "\n".join(names) + "\n"
    print(f"Training on {len(names)} names ({len(text)} chars)...")
    m, stoi, itos = fit(text, steps=steps)
    save(out, m, stoi, itos)
    print(f"Saved -> {out}")
    print("Samples:", ", ".join(sample_name(m, stoi, itos, seed=k) for k in range(8)))

if __name__ == "__main__":
    main()
