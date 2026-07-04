"""
AI Researcher Path — Portfolio.
One Flask app that showcases three from-scratch, NumPy-only ML projects and serves
all three as live interactive demos under a single URL:
  /namegen   a character-level Transformer that invents names   (Tracks 4-5)
  /ddlab     an interactive double-descent reproduction         (Tracks 3, 8)
  /gridworld a Q-learning agent that learns in your browser     (Track 7)
Deploy once, share one link. See README.md for hosting steps.
"""
import os, io, base64
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request, render_template

from portfolio.minigpt import load as gpt_load, fit as gpt_fit, save as gpt_save, sample_name
from portfolio.ddlab import double_descent_curve
from portfolio.rlgrid import GridWorld, QAgent

app = Flask(__name__)

# ---- Capstone A: name generator (train once, then load) ---------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "model.npz")
def _ensure_gpt():
    if not os.path.exists(MODEL_PATH):
        names = [n.strip() for n in open("data/names.txt") if n.strip()]
        m, stoi, itos = gpt_fit("\n" + "\n".join(names) + "\n", steps=4000, log=None)
        gpt_save(MODEL_PATH, m, stoi, itos)
    return gpt_load(MODEL_PATH)
GPT, STOI, ITOS = _ensure_gpt()

# ---- Capstone C: gridworld agent (stateful, single worker) ------------------
ENV = GridWorld(); AGENT = QAgent(ENV)

# ---- pages ------------------------------------------------------------------
@app.route("/")            
def index():     return render_template("index.html")
@app.route("/namegen")     
def p_namegen(): return render_template("namegen.html")
@app.route("/ddlab")       
def p_ddlab():   return render_template("ddlab.html")
@app.route("/gridworld")   
def p_grid():    return render_template("gridworld.html")

# ---- Capstone A API ---------------------------------------------------------
@app.route("/namegen/api/generate")
def gen():
    count = max(1, min(20, int(request.args.get("count", 6))))
    temp  = max(0.1, min(1.5, float(request.args.get("temperature", 0.9))))
    prefix = "".join(c for c in request.args.get("prefix", "") if c.isalpha())[:8]
    return jsonify(names=[sample_name(GPT, STOI, ITOS, temperature=temp, prefix=prefix) for _ in range(count)])

# ---- Capstone B API ---------------------------------------------------------
@app.route("/ddlab/api/curve")
def curve():
    noise = max(0.0, min(2.0, float(request.args.get("noise", 0.5))))
    ridge = max(1e-7, min(10.0, float(request.args.get("ridge", 1e-6))))
    ratios, errs = double_descent_curve(noise=noise, ridge=ridge, repeats=15)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(ratios, errs, "o-", color="#2563eb", lw=1.8, ms=4)
    ax.axvline(1.0, ls="--", color="#b91c1c", lw=1.3)
    ax.set_yscale("log"); ax.set_xlabel("model size  P / n"); ax.set_ylabel("test error (log)")
    ax.set_title(f"noise={noise:.2f}, ridge={ridge:.0e}"); ax.grid(alpha=.3, which="both")
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight"); plt.close(fig)
    peak = max(range(len(errs)), key=lambda i: errs[i])
    return jsonify(image=base64.b64encode(buf.getvalue()).decode(),
                   peak_ratio=round(ratios[peak], 2), peak_error=round(errs[peak], 2))

# ---- Capstone C API ---------------------------------------------------------
@app.route("/gridworld/api/train")
def train():
    AGENT.train(episodes=max(1, min(2000, int(request.args.get("episodes", 200)))))
    return jsonify(AGENT.state())
@app.route("/gridworld/api/reset")
def reset():
    global AGENT; AGENT = QAgent(ENV, seed=int(request.args.get("seed", 0)))
    return jsonify(AGENT.state())
@app.route("/gridworld/api/state")
def state():   return jsonify(AGENT.state())

@app.route("/health")
def health():  return jsonify(status="ok", demos=["namegen", "ddlab", "gridworld"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
