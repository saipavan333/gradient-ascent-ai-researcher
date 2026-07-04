# 🎓 AI Researcher Path — Portfolio

**One deployable app. One live link. Three from-scratch ML demos.**

This is a single Flask application that showcases three interactive machine-learning projects — a Transformer, a research reproduction, and a reinforcement-learning agent — all written **from scratch in pure NumPy** (no PyTorch, no TensorFlow) as the capstones of a from-scratch AI curriculum. Deploy it once and share a single URL.

| Demo | What it is | Built from |
|------|------------|-----------|
| **🧬 `/namegen`** | A character-level **Transformer** that invents new names in real time, with hand-derived backprop. | Tracks 4–5 |
| **📉 `/ddlab`** | An interactive reproduction of **double descent** — drag sliders, watch the curve. | Tracks 3, 8 |
| **🤖 `/gridworld`** | A **Q-learning** agent that learns to solve a maze live in your browser. | Track 7 |

The landing page (`/`) presents all three as a portfolio; each links to its live demo.

---

## Run it locally

```bash
git clone https://github.com/<your-username>/ai-portfolio.git
cd ai-portfolio
pip install -r requirements.txt
python app.py            # trains the tiny model on first run, then serves
# open http://localhost:8000
```

Run the tests:

```bash
pip install pytest && pytest -q      # 5 tests: pages load, all 3 APIs work, the RL agent learns
```

---

## Deploy it — get your one live link

All three demos ship in this single app, so you deploy **once** and get **one** shareable URL. Two free options:

### Option A — Render (recommended, free tier)

1. Push this folder to a **public GitHub repo** (see below).
2. Go to **[render.com](https://render.com)** → **New → Web Service** → connect the repo.
3. Set:
   - **Build Command:** `pip install -r requirements.txt && python -c "import app"`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1`
   - *(one worker is important — the gridworld agent keeps its learning in memory)*
4. **Create Web Service.** In ~2 minutes you get `https://your-portfolio.onrender.com` — **that's your live link.** Share it anywhere.

### Option B — Hugging Face Spaces (Docker)

1. Create a **Docker Space** at **[huggingface.co/new-space](https://huggingface.co/new-space)**.
2. Push this repo to it — the included `Dockerfile` builds and serves everything.
3. Your live link is `https://huggingface.co/spaces/<user>/<space>`.

### Push to GitHub first

```bash
git init && git add . && git commit -m "AI portfolio: three from-scratch ML demos"
git branch -M main
git remote add origin https://github.com/<your-username>/ai-portfolio.git
git push -u origin main
```

CI (`.github/workflows/ci.yml`) runs the test suite on every push.

### Run the Docker image anywhere

```bash
docker build -t ai-portfolio . && docker run -p 8000:8000 ai-portfolio
# http://localhost:8000
```

---

## Using it as a portfolio piece

Once deployed, put the live link on your CV / LinkedIn / GitHub profile with a line like:

> **AI Researcher Path — [live demo](https://your-portfolio.onrender.com)** · Built a Transformer, a research reproduction, and an RL agent **from scratch in NumPy** (no ML frameworks) and deployed them as interactive web demos. [Source](https://github.com/<you>/ai-portfolio).

Because every model is your own code, you can explain any part in an interview — the architecture (attention, sampling), the science (double descent, the interpolation threshold), the algorithm (Q-learning's TD update), and the engineering (Flask API, tests, Docker, CI).

---

## Project structure

```
portfolio/
├── app.py                    # one Flask app: landing + 3 demos + their APIs
├── portfolio/
│   ├── minigpt.py            # the Transformer (NumPy)
│   ├── ddlab.py              # the double-descent experiment (NumPy)
│   └── rlgrid.py             # the Q-learning agent (NumPy)
├── templates/                # landing + 3 demo pages
├── static/portfolio.css
├── data/names.txt
├── tests/test_app.py         # pages load · APIs respond · the agent learns
├── Dockerfile  Procfile  requirements.txt
└── .github/workflows/ci.yml
```

The three projects also exist as standalone repos (`../minigpt-namegen`, `../double-descent-lab`, `../rl-gridworld-live`) if you prefer to deploy them separately.

## License

MIT — see [LICENSE](LICENSE).
