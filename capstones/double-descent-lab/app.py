"""
Interactive double-descent lab. Move the noise and ridge sliders and watch the
test-error curve re-render live — see the interpolation-threshold spike appear
and disappear. Built from the Track 8 reproduction.
"""
import io, base64, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request, render_template
from ddlab.experiment import double_descent_curve

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/curve")
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
    img = base64.b64encode(buf.getvalue()).decode()
    peak = max(range(len(errs)), key=lambda i: errs[i])
    return jsonify(image=img, peak_ratio=round(ratios[peak], 2), peak_error=round(errs[peak], 2))

@app.route("/health")
def health():
    return jsonify(status="ok")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
