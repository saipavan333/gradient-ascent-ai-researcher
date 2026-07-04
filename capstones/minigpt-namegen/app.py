"""
Flask web app that serves the from-scratch MiniGPT name generator in real time.
Run locally:  python app.py   then open http://localhost:8000
The model is trained once (python -m minigpt.train) and loaded from model.npz.
"""
import os
from flask import Flask, jsonify, request, render_template
from minigpt.model import load, sample_name, fit, save

app = Flask(__name__)
MODEL_PATH = os.environ.get("MODEL_PATH", "model.npz")

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        names = [n.strip() for n in open("data/names.txt") if n.strip()]
        m, stoi, itos = fit("\n" + "\n".join(names) + "\n", steps=4000, log=None)
        save(MODEL_PATH, m, stoi, itos)
    return load(MODEL_PATH)

MODEL, STOI, ITOS = _ensure_model()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/generate")
def generate():
    count = max(1, min(20, int(request.args.get("count", 6))))
    temp = max(0.1, min(1.5, float(request.args.get("temperature", 0.9))))
    prefix = "".join(c for c in request.args.get("prefix", "") if c.isalpha())[:8]
    names = []
    for _ in range(count):
        names.append(sample_name(MODEL, STOI, ITOS, temperature=temp, prefix=prefix))
    return jsonify(names=names, temperature=temp, prefix=prefix)

@app.route("/health")
def health():
    return jsonify(status="ok", vocab=MODEL.V, context=MODEL.T, dim=MODEL.d)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
