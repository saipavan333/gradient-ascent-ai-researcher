# Gradient Ascent — AI Researcher Path

A complete, from-scratch path from the geometry of a vector to the frontier of AI — **24 tracks** taking you from first principles to a capable AI researcher, with **every model built by hand in pure NumPy** (no PyTorch/TensorFlow). All 24 tracks are built and live — 123 lessons, every model built by hand.

Every lesson is taught to a **masterclass standard**, in plain English a motivated beginner can follow:

- **Intuition first** — an everyday analogy before any math, every symbol defined the moment it appears.
- **A worked example** — one concrete case walked through number-by-number.
- **An interactive widget** — drag a vector, slide a learning rate until it diverges, run k-means by hand, fit a line by gradient descent (117 interactive widgets across the course).
- **Common mistakes**, a **quiz** (every wrong answer explained), **graded practice** with full solutions, a **frontier note**, and **primary sources** for going deeper.

This one repo holds everything you need to **read**, **run**, and **build**:

- **`lessons/`** + `researcher-path.html` — the interactive course site (lessons, diagrams, widgets, quizzes, progress). Publish it with GitHub Pages and read it anywhere.
- **`notebooks/track-*/`** — a runnable Jupyter/Colab notebook for each lesson's code.
- **`code/track-*/`** — the same code as plain `.py` files (all validated to run).
- **`capstones/`** — deployable web apps (three interactive demos + one unified portfolio).

## Learn like a pro — the guided on-ramp & study tools

New to all this? The site has a full beginner on-ramp and built-in retention tools, all linked from the sidebar:

- **Start Here** (`start-here.html`) — what you need, how the lessons work, a **clickable visual map** of all 24 tracks by phase, and **curated learning paths** (complete, fast-track to a GPT, LLM engineer, theory-first, RL, researcher).
- **Math & Notation primer** (`math-primer.html`) — every symbol in the course decoded in plain English.
- **Run Python in your browser** — a **▶ Run** button on every code block (Pyodide/WASM, numpy included); edit and re-run in place, no install.
- **Progress dashboard** (`progress.html`) — completion %, per-track bars, a day streak, your next lesson, and a certificate at 100% (stored privately in your browser).
- **Interactive Labs** (`labs.html`) — 70 flagship, standalone sandboxes (train a neural net, race optimizers, attention, diffusion, GANs, Bayes, and more) plus a full index of every in-lesson interactive.
- **Glossary** (`glossary.html`) — 82 key terms, searchable · **Flashcards** (`flashcards.html`) — spaced-repetition review of every concept.

## The curriculum (24 tracks, in learning order)

**Foundations** — 1. Linear Algebra & Probability · 2. Optimization · 3. Information Theory · 4. ML Theory
**Classical & core ML** — 5. Classical ML (incl. linear/logistic regression) · 6. Deep Learning from Scratch
**Deep architectures** — 7. Convolutional Networks & Vision · 8. Sequence Models (RNN/LSTM/GRU) · 9. Transformers · 10. Large Language Models
**Beyond supervised learning** — 11. Generative Models · 12. Reinforcement Learning · 13. Learning Paradigms · 14. Graph Neural Networks · 15. Probabilistic & Bayesian ML · 16. Meta, Few-Shot & Continual Learning · 17. Federated & Privacy-Preserving ML · 18. Causal Inference
**Systems & frontier** — 19. Efficient & Scalable ML · 20. Frontier Architectures & Agents
**Research practice** — 21. Reading & Reproducing Papers · 22. Research Methods · 23. AI Safety & Interpretability · 24. Research Capstones

 Every track builds only on earlier ones — it never depends on something taught later. All 24 are complete.

## How to use this repo

1. **Read** a lesson on the live site (see below), or open the `.html` files locally in a browser.
2. **Run** that lesson's notebook in [Google Colab](https://colab.research.google.com): `File → Open notebook → GitHub`, paste this repo's URL, pick `notebooks/track-N/...`.
3. **Play** with the interactive widget in each lesson — the fastest way to build intuition.
4. **Practice** — do the graded exercises by editing the code yourself.
5. **Commit** your changes (`git add -A && git commit && git push`). Your history is your progress log.

Work the tracks **in order** — start at Track 1, Lesson 0.

## Make the site live (GitHub Pages)

Repo **Settings → Pages → Build and deployment → Deploy from a branch → `main` / `/ (root)` → Save**.
Your site appears at `https://YOUR-USERNAME.github.io/REPO-NAME/` in ~1–2 minutes. The root `index.html` auto-redirects into the home page.

## The capstones (Track 24 — the finale)

`capstones/portfolio/` is a single Flask app that serves three interactive, from-scratch demos (a NumPy Transformer that invents names, a live double-descent reproduction, and a Q-learning agent solving a maze) under one URL. Deploy it to **Render** or **Hugging Face Spaces** (steps in `capstones/portfolio/README.md`). Static GitHub Pages can't run the apps — they need a server host.

## Requirements

Only `numpy` and `matplotlib` (both preinstalled in Colab). Locally: `pip install -r requirements.txt`.

---
Built to be rigorous, runnable, honest, and genuinely fun to learn from. Bookmark your Pages URL and start with Track 1, Lesson 0.
