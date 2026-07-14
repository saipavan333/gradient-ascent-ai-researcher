"""Generate Track 16 diagrams (Sequence Models). PNGs -> assets/img/track-16/."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "assets", "img", "track-16"))
os.makedirs(OUT, exist_ok=True)

BLUE = "#2563eb"; SLATE = "#334155"; LIGHT = "#eff6ff"; GREEN = "#059669"
AMBER = "#d97706"; RED = "#dc2626"; GREY = "#94a3b8"; INK = "#0f172a"
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans",
                     "axes.edgecolor": SLATE, "savefig.dpi": 140})


def box(ax, x, y, w, h, text, fc=LIGHT, ec=BLUE, tc=INK, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc,
            fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, color=SLATE, rad=0.0, lw=2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                                 color=color, lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}"))


def opnode(ax, x, y, sym, color):
    ax.add_patch(Circle((x, y), 0.22, fc="white", ec=color, lw=2))
    ax.text(x, y, sym, ha="center", va="center", fontsize=13, fontweight="bold", color=color)


# 1) rnn_unrolled.png -------------------------------------------------------
def rnn_unrolled():
    fig, ax = plt.subplots(figsize=(11, 3.8)); ax.set_xlim(0, 11); ax.set_ylim(0, 3.8); ax.axis("off")
    xs = [1.2, 4.2, 7.2]
    for k, x in enumerate(xs):
        box(ax, x, 1.5, 1.6, 0.9, f"cell\nh$_{{{k}}}$", fc="#dbeafe", ec=BLUE, bold=True)
        box(ax, x, 0.2, 1.6, 0.6, f"x$_{{{k}}}$", fc="#f1f5f9", ec=GREY, fs=10)
        arrow(ax, x + 0.8, 0.8, x + 0.8, 1.5)                       # input up
    for k in range(len(xs) - 1):
        arrow(ax, xs[k] + 1.6, 1.95, xs[k + 1], 1.95, color=BLUE)   # recurrence
        ax.text((xs[k] + 1.6 + xs[k + 1]) / 2, 2.2, "same\nweights", ha="center", fontsize=8, color=BLUE)
    box(ax, 9.2, 1.5, 1.4, 0.9, "output\nŷ", fc="#fee2e2", ec=RED, fs=9, bold=True)
    arrow(ax, 8.8, 1.95, 9.2, 1.95, color=RED)
    ax.text(5.5, 3.35, "An RNN unrolled in time: one cell reused at every step, passing a hidden state (memory) forward",
            ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.text(5.5, 1.0, "h$_t$ = tanh( W$_{xh}$ x$_t$ + W$_{hh}$ h$_{t-1}$ + b )", ha="center", fontsize=10, color=SLATE)
    fig.savefig(os.path.join(OUT, "rnn_unrolled.png"), bbox_inches="tight"); plt.close(fig)


# 2) vanishing_plot.png -----------------------------------------------------
def vanishing_plot():
    rng = np.random.default_rng(0); H, T = 20, 40
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for scale, col, lab in [(0.5, GREEN, "small weights → vanish"),
                            (1.0, BLUE, "moderate"),
                            (1.6, RED, "large weights → explode")]:
        Whh = rng.standard_normal((H, H)) * (scale / np.sqrt(H))
        hs = [rng.uniform(-0.5, 0.5, H) for _ in range(T)]
        dh = rng.standard_normal(H); dh /= np.linalg.norm(dh)
        norms = [1.0]
        for t in reversed(range(T)):
            da = (1 - hs[t] ** 2) * dh; dh = Whh.T @ da; norms.append(np.linalg.norm(dh))
        ax.semilogy(range(len(norms)), norms, color=col, lw=2.4, label=lab)
    ax.axhline(1.0, color=GREY, ls=":", lw=1)
    ax.set_xlabel("time steps the gradient has travelled backward")
    ax.set_ylabel("gradient norm (log scale)")
    ax.set_title("Why plain RNNs forget: the learning signal vanishes or explodes exponentially",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="center right", fontsize=9); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "vanishing_plot.png"), bbox_inches="tight"); plt.close(fig)


# 3) lstm_cell.png ----------------------------------------------------------
def lstm_cell():
    fig, ax = plt.subplots(figsize=(10.5, 4.6)); ax.set_xlim(0, 10.5); ax.set_ylim(0, 4.6); ax.axis("off")
    # cell-state highway (top)
    ax.text(0.55, 3.85, "c$_{t-1}$", ha="center", fontsize=11, color=GREEN, fontweight="bold")
    arrow(ax, 0.9, 3.7, 2.8, 3.7, color=GREEN, lw=3)
    opnode(ax, 3.1, 3.7, "×", GREEN)                       # forget
    arrow(ax, 3.4, 3.7, 4.8, 3.7, color=GREEN, lw=3)
    opnode(ax, 5.1, 3.7, "+", GREEN)                       # input add
    arrow(ax, 5.4, 3.7, 9.6, 3.7, color=GREEN, lw=3)
    ax.text(9.9, 3.85, "c$_t$", ha="center", fontsize=11, color=GREEN, fontweight="bold")
    ax.text(5.2, 4.25, "cell state = protected memory highway (updated by × and +, so gradients survive)",
            ha="center", fontsize=9.5, color=GREEN)
    # gates row
    box(ax, 2.4, 1.7, 1.3, 0.8, "f\nσ  forget", fc="#fee2e2", ec=RED, fs=9)
    box(ax, 4.0, 1.7, 1.3, 0.8, "i\nσ  input", fc="#dbeafe", ec=BLUE, fs=9)
    box(ax, 5.5, 1.7, 1.3, 0.8, "g\ntanh  cand.", fc="#dbeafe", ec=BLUE, fs=9)
    box(ax, 7.6, 1.7, 1.3, 0.8, "o\nσ  output", fc="#dcfce7", ec=GREEN, fs=9)
    arrow(ax, 3.05, 2.5, 3.1, 3.48, color=RED)
    arrow(ax, 4.65, 2.5, 5.0, 3.48, color=BLUE)
    arrow(ax, 6.15, 2.5, 5.2, 3.48, color=BLUE)
    # output path
    opnode(ax, 8.25, 2.55, "×", GREEN)
    arrow(ax, 8.25, 3.48, 8.25, 2.77, color=GREEN)         # tanh(c_t) down into xo
    ax.text(8.95, 3.1, "tanh", ha="center", fontsize=8.5, color=GREEN)
    arrow(ax, 8.25, 2.33, 8.25, 1.2, color=SLATE)
    ax.text(8.25, 0.95, "h$_t$", ha="center", fontsize=11, fontweight="bold", color=INK)
    # inputs
    box(ax, 0.4, 0.4, 1.5, 0.7, "x$_t$ , h$_{t-1}$", fc="#f1f5f9", ec=GREY, fs=9)
    for gx in [3.05, 4.65, 6.15, 8.25]:
        arrow(ax, 1.9, 0.75, gx - 0.2, 1.68 if gx < 7 else 1.68, color=GREY, rad=-0.15)
    ax.text(5.25, 0.2, "LSTM cell: three σ gates and a tanh candidate control what to forget, write, and read",
            ha="center", fontsize=11, fontweight="bold", color=INK)
    fig.savefig(os.path.join(OUT, "lstm_cell.png"), bbox_inches="tight"); plt.close(fig)


# 4) gru_cell.png -----------------------------------------------------------
def gru_cell():
    fig, ax = plt.subplots(figsize=(10, 4.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")
    ax.text(0.5, 3.4, "h$_{t-1}$", ha="center", fontsize=11, color=BLUE, fontweight="bold")
    arrow(ax, 0.9, 3.25, 7.4, 3.25, color=BLUE, lw=2.5)            # carry path
    opnode(ax, 7.7, 3.25, "+", BLUE)
    arrow(ax, 8.0, 3.25, 9.4, 3.25, color=BLUE, lw=2.5)
    ax.text(9.7, 3.4, "h$_t$", ha="center", fontsize=11, color=BLUE, fontweight="bold")
    # gates
    box(ax, 2.3, 1.5, 1.4, 0.8, "z\nupdate gate", fc="#fef9c3", ec=AMBER, fs=9)
    box(ax, 4.1, 1.5, 1.4, 0.8, "r\nreset gate", fc="#fee2e2", ec=RED, fs=9)
    box(ax, 5.9, 1.5, 1.6, 0.8, "n = tanh\ncandidate", fc="#dbeafe", ec=BLUE, fs=9)
    arrow(ax, 3.0, 2.3, 5.5, 3.15, color=AMBER, rad=-0.1)          # z controls blend
    arrow(ax, 4.8, 2.3, 6.4, 1.5 + 0.8, color=RED, rad=0)          # r into candidate
    arrow(ax, 6.7, 2.3, 7.6, 3.05, color=BLUE, rad=0.1)            # candidate into +
    ax.text(5.5, 3.55, "h$_t$ = (1−z)·n + z·h$_{t-1}$   —  z≈1 copies memory forward",
            ha="center", fontsize=9.5, color=SLATE)
    box(ax, 0.4, 0.4, 1.6, 0.7, "x$_t$ , h$_{t-1}$", fc="#f1f5f9", ec=GREY, fs=9)
    for gx in [3.0, 4.8, 6.7]:
        arrow(ax, 2.0, 0.75, gx - 0.1, 1.48, color=GREY, rad=-0.15)
    ax.text(5.0, 0.15, "GRU cell: two gates, one state — simpler than an LSTM, often as good",
            ha="center", fontsize=11, fontweight="bold", color=INK)
    fig.savefig(os.path.join(OUT, "gru_cell.png"), bbox_inches="tight"); plt.close(fig)


# 5) seq2seq_arch.png -------------------------------------------------------
def seq2seq():
    fig, ax = plt.subplots(figsize=(11.5, 4.4)); ax.set_xlim(0, 11.5); ax.set_ylim(0, 4.4); ax.axis("off")
    # encoder
    for k, x in enumerate([0.6, 2.0, 3.4]):
        box(ax, x, 2.3, 1.2, 0.8, f"enc\nh$_{{{k}}}$", fc="#dbeafe", ec=BLUE, fs=9)
        box(ax, x, 1.1, 1.2, 0.6, f"x$_{{{k}}}$", fc="#f1f5f9", ec=GREY, fs=9)
        arrow(ax, x + 0.6, 1.7, x + 0.6, 2.3, color=GREY)
        if k < 2: arrow(ax, x + 1.2, 2.7, x + 1.4, 2.7, color=BLUE)
    # context bottleneck
    arrow(ax, 4.6, 2.7, 5.3, 2.7, color=INK, lw=2.5)
    box(ax, 5.3, 2.3, 1.5, 0.9, "context\nvector", fc="#fde68a", ec=AMBER, fs=9, bold=True)
    ax.text(6.05, 1.95, "ONE fixed vector\n(the bottleneck)", ha="center", fontsize=8.5, color=AMBER)
    arrow(ax, 6.8, 2.7, 7.5, 2.7, color=INK, lw=2.5)
    # decoder
    for k, x in enumerate([7.5, 8.9, 10.3]):
        box(ax, x, 2.3, 1.1, 0.8, f"dec\nh$_{{{k}}}$", fc="#dcfce7", ec=GREEN, fs=9)
        box(ax, x, 3.35, 1.1, 0.55, f"ŷ$_{{{k}}}$", fc="#fee2e2", ec=RED, fs=9)
        arrow(ax, x + 0.55, 3.1, x + 0.55, 3.35, color=RED)
        if k < 2: arrow(ax, x + 1.1, 2.7, x + 1.3, 2.7, color=GREEN)
    ax.text(5.75, 4.05, "Seq2seq: an encoder compresses the whole input into one context vector; a decoder unrolls it into the output",
            ha="center", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(5.75, 0.35, "Everything must pass through that single vector — the limit that attention (Track 5) was invented to remove.",
            ha="center", fontsize=9.5, color=SLATE, style="italic")
    fig.savefig(os.path.join(OUT, "seq2seq_arch.png"), bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    rnn_unrolled(); vanishing_plot(); lstm_cell(); gru_cell(); seq2seq()
    print("wrote Track 16 diagrams to", OUT)
    for f in sorted(os.listdir(OUT)):
        if not f.startswith("_"): print("  ", f)
