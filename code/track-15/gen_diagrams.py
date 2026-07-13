"""Generate Track 15 diagrams (CNNs & Computer Vision).
Outputs PNGs into assets/img/track-15/. Style matches the site: clean, labeled,
light background, blue/slate palette. Run from anywhere; paths are resolved
relative to the repo root two levels up from this file.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "assets", "img", "track-15"))
os.makedirs(OUT, exist_ok=True)

BLUE = "#2563eb"; SLATE = "#334155"; LIGHT = "#eff6ff"; GREEN = "#059669"
AMBER = "#d97706"; GREY = "#94a3b8"; INK = "#0f172a"
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans",
                     "axes.edgecolor": SLATE, "savefig.dpi": 140})


def box(ax, x, y, w, h, text, fc=LIGHT, ec=BLUE, tc=INK, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc,
            fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, color=SLATE, rad=0.0, lw=2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


# ---------------------------------------------------------------------------
# 1) conv_slide.png — a kernel sliding over an image -> feature map
# ---------------------------------------------------------------------------
def conv_slide():
    img = np.zeros((9, 9)); img[:, 5:] = 1.0
    sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    # feature map (valid)
    H = W = 7
    fmap = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            fmap[i, j] = np.sum(img[i:i+3, j:j+3] * sobel)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    # input with a highlighted 3x3 receptive field
    ax = axes[0]
    ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
    ax.add_patch(plt.Rectangle((3.5, 1.5), 3, 3, fill=False, ec=BLUE, lw=3))
    ax.set_title("1. Input image\n(a 3×3 window = receptive field)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # kernel
    ax = axes[1]
    ax.imshow(sobel, cmap="coolwarm", vmin=-2, vmax=2)
    for (i, j), v in np.ndenumerate(sobel):
        ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.set_title("2. Kernel (shared weights)\nvertical-edge detector", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # feature map
    ax = axes[2]
    im = ax.imshow(fmap, cmap="magma")
    ax.set_title("3. Feature map\nbright = strong edge response", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Convolution: slide one small shared kernel over the image, take a weighted sum at each spot",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "conv_slide.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2) maxpool.png — 4x4 map -> 2x2 max-pooled, windows colored
# ---------------------------------------------------------------------------
def maxpool():
    rng = np.random.default_rng(3)
    x = np.round(rng.random((4, 4)), 1)
    pooled = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            pooled[i, j] = x[2*i:2*i+2, 2*j:2*j+2].max()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    colors = ["#bfdbfe", "#bbf7d0", "#fde68a", "#fecaca"]
    ax = axes[0]
    ax.set_xlim(0, 4); ax.set_ylim(0, 4); ax.invert_yaxis(); ax.set_aspect("equal")
    for bi in range(2):
        for bj in range(2):
            ax.add_patch(plt.Rectangle((2*bj, 2*bi), 2, 2, fc=colors[bi*2+bj], ec=SLATE, lw=2))
    for (i, j), v in np.ndenumerate(x):
        mx = (x[2*(i//2):2*(i//2)+2, 2*(j//2):2*(j//2)+2].max() == v)
        ax.text(j+0.5, i+0.5, f"{v:.1f}", ha="center", va="center",
                fontsize=12, fontweight="bold" if mx else "normal",
                color=INK if mx else SLATE)
    ax.set_title("Feature map (4×4)\neach colored block = one 2×2 window", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.invert_yaxis(); ax.set_aspect("equal")
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fc=colors[i*2+j], ec=SLATE, lw=2))
            ax.text(j+0.5, i+0.5, f"{pooled[i, j]:.1f}", ha="center", va="center",
                    fontsize=15, fontweight="bold", color=INK)
    ax.set_title("After max-pool 2×2\nkeep only the strongest response", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Max-pooling: shrink the map 2× and keep the strongest activation in each window",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "maxpool.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3) cnn_arch.png — the tiny CNN pipeline
# ---------------------------------------------------------------------------
def cnn_arch():
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.6); ax.axis("off")
    y = 1.3; h = 1.0
    stages = [
        (0.2, 1.7, "Input\n8×8 image", LIGHT, BLUE),
        (2.3, 1.7, "Conv\n6 × 3×3 filters\n→ 6×6×6", "#dbeafe", BLUE),
        (4.4, 1.7, "ReLU\nkeep positives", "#e0f2fe", "#0284c7"),
        (6.5, 1.7, "Max-pool 2×2\n→ 3×3×6", "#dcfce7", GREEN),
        (8.6, 1.4, "Flatten\n→ 54", "#fef9c3", AMBER),
        (10.3, 1.6, "Linear + Softmax\n→ 2 classes", "#fee2e2", "#dc2626"),
    ]
    for (x, w, t, fc, ec) in stages:
        box(ax, x, y, w, h, t, fc=fc, ec=ec, fs=9)
    xs = [0.2+1.7, 2.3+1.7, 4.4+1.7, 6.5+1.7, 8.6+1.4]
    nxt = [2.3, 4.4, 6.5, 8.6, 10.3]
    for a, b in zip(xs, nxt):
        arrow(ax, a, y+h/2, b, y+h/2)
    ax.text(6, 3.2, "A ConvNet = feature extractor (conv→relu→pool, repeated) + a small classifier head",
            ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.text(6, 0.5, "Gradients flow back through every stage — the filters are learned, not designed.",
            ha="center", fontsize=9.5, color=SLATE, style="italic")
    fig.savefig(os.path.join(OUT, "cnn_arch.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4) learned_filters.png — actual filters after training the tiny CNN
# ---------------------------------------------------------------------------
def learned_filters():
    import importlib.util
    spec = importlib.util.spec_from_file_location("convnet", os.path.join(HERE, "convnet.py"))
    cn = importlib.util.module_from_spec(spec); spec.loader.exec_module(cn)
    m = cn.TinyCNN(n_filters=6)
    Xtr, ytr = cn.make_dataset(400)
    for _ in range(6):
        for idx in cn.rng.permutation(len(Xtr)):
            p, c = m.forward(Xtr[idx]); m.step(m.backward(p, ytr[idx], c), 0.02)
    fig, axes = plt.subplots(1, 6, figsize=(11, 2.2))
    for k, ax in enumerate(axes):
        ax.imshow(m.W[k], cmap="coolwarm", vmin=-abs(m.W).max(), vmax=abs(m.W).max())
        ax.set_title(f"filter {k+1}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("The 6 3×3 filters the CNN learned by itself — several became oriented-edge detectors",
                 fontsize=11, fontweight="bold", y=1.12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "learned_filters.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5) augment_grid.png — one image and label-preserving variants
# ---------------------------------------------------------------------------
def augment_grid():
    rng = np.random.default_rng(7)
    base = np.zeros((8, 8)); base[:, 3] = 1.0   # a vertical bar
    def aug(x):
        o = x.copy()
        if rng.random() < 0.5: o = o[:, ::-1]
        if rng.random() < 0.5: o = o[::-1, :]
        o = np.roll(o, rng.integers(-1, 2), axis=rng.integers(0, 2))
        return o + 0.12 * rng.standard_normal(o.shape)
    fig, axes = plt.subplots(1, 6, figsize=(11, 2.2))
    axes[0].imshow(base, cmap="gray_r"); axes[0].set_title("original", fontsize=9, color=BLUE)
    for k in range(1, 6):
        axes[k].imshow(aug(base), cmap="gray_r"); axes[k].set_title(f"aug {k}", fontsize=9)
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Data augmentation: same label, new pixels — flips, shifts and noise multiply your data for free",
                 fontsize=11, fontweight="bold", y=1.12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "augment_grid.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6) transfer_schematic.png — freeze features, train a new head
# ---------------------------------------------------------------------------
def transfer_schematic():
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.4); ax.axis("off")
    # pretrained extractor (frozen)
    box(ax, 0.3, 2.4, 5.2, 1.3, "Pretrained feature extractor  (conv → relu → pool)",
        fc="#dbeafe", ec=BLUE, fs=10, bold=True)
    ax.text(2.9, 2.15, "[ FROZEN ] — reuse as-is, no gradients", ha="center", fontsize=9.5, color=BLUE)
    box(ax, 6.2, 2.4, 4.4, 1.3, "NEW small head\n(train on few labels)", fc="#fee2e2", ec="#dc2626", fs=10, bold=True)
    arrow(ax, 5.5, 3.05, 6.2, 3.05, color=SLATE)
    ax.text(8.4, 2.15, "[ TRAINED ] on your labels", ha="center", fontsize=9.5, color="#dc2626")

    box(ax, 0.3, 0.4, 5.2, 1.1, "trained once on ABUNDANT source data", fc="#f1f5f9", ec=GREY, fs=9)
    box(ax, 6.2, 0.4, 4.4, 1.1, "your task: SCARCE target labels", fc="#f1f5f9", ec=GREY, fs=9)
    arrow(ax, 2.9, 2.4, 2.9, 1.5, color=GREY, rad=0)
    arrow(ax, 8.4, 2.4, 8.4, 1.5, color=GREY, rad=0)
    ax.text(5.5, 4.1, "Transfer learning: generic features learned on big data + a tiny head learned on your small data",
            ha="center", fontsize=11, fontweight="bold", color=INK)
    fig.savefig(os.path.join(OUT, "transfer_schematic.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    conv_slide(); maxpool(); cnn_arch(); learned_filters(); augment_grid(); transfer_schematic()
    print("Track 15 diagrams written to", OUT)
    for f in sorted(os.listdir(OUT)):
        pr