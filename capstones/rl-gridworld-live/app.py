"""
Watch a Q-learning agent learn to solve a GridWorld, live in the browser. Click
"Train" to run more episodes and see the value heatmap and policy arrows update.
Built from Track 7 of the AI Researcher Path.
"""
import os
from flask import Flask, jsonify, request, render_template
from rlgrid.agent import GridWorld, QAgent

app = Flask(__name__)
ENV = GridWorld()
AGENT = QAgent(ENV)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/train")
def train():
    episodes = max(1, min(2000, int(request.args.get("episodes", 200))))
    AGENT.train(episodes=episodes)
    return jsonify(AGENT.state())

@app.route("/api/reset")
def reset():
    global AGENT
    AGENT = QAgent(ENV, seed=int(request.args.get("seed", 0)))
    return jsonify(AGENT.state())

@app.route("/api/state")
def state():
    return jsonify(AGENT.state())

@app.route("/health")
def health():
    return jsonify(status="ok", states=len(ENV.states))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
