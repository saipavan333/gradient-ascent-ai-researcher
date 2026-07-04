# Gradient Ascent — AI Researcher Path

A complete, from-scratch path from linear algebra to AI safety — **14 tracks, 72 lessons**, every model built in pure NumPy (no PyTorch/TensorFlow). This one repo holds everything you need to **read**, **run**, and **build**:

- **`lessons/`** + `researcher-path.html` — the interactive course site (lessons, diagrams, quizzes, progress). Publish it with GitHub Pages and read it anywhere.
- **`notebooks/track-*/`** — a runnable Jupyter/Colab notebook for each lesson's code.
- **`code/track-*/`** — the same code as plain `.py` files.
- **`capstones/`** — four deployable web apps (three demos + one unified portfolio).

## How to use this repo

1. **Read** a lesson on the live site (see below), or open the `.html` files locally.
2. **Run** that lesson's notebook in [Google Colab](https://colab.research.google.com): `File → Open notebook → GitHub`, paste this repo's URL, pick `notebooks/track-N/...`.
3. **Practice** — do the lesson's exercises by editing the code yourself.
4. **Commit** your changes (GitHub Desktop → *Commit* → *Push*). Your history is your progress log.

Work the tracks **in order, 1 → 14** — each builds on the last.

## Make the site live (GitHub Pages)

Repo **Settings → Pages → Build and deployment → Deploy from a branch → `main` / `/ (root)` → Save**.
Your site appears at `https://YOUR-USERNAME.github.io/REPO-NAME/` in ~1–2 minutes. The root `index.html` auto-redirects into the home page.

## The capstones

`capstones/portfolio/` is a single Flask app that serves all three interactive demos under one URL — deploy it to **Render** or **Hugging Face Spaces** (steps in `capstones/portfolio/README.md`) when you reach Track 10. Static GitHub Pages can't run the apps; they need a server host.

## Requirements

Only `numpy` and `matplotlib` (both preinstalled in Colab). Locally: `pip install -r requirements.txt`.

---
Built to be rigorous, runnable, and honest. Bookmark your Pages URL and start with Track 1, Lesson 0.
