"""
Track 7 · Lesson 2 — REINFORCE (Monte-Carlo policy gradient) on GridWorld
Run:  python reinforce.py

Instead of learning values and acting greedily, policy-gradient methods learn the
policy DIRECTLY. We parameterize a softmax policy pi(a|s) by preferences theta[s,a]
and push up the log-probability of actions that led to high return:
    theta <- theta + alpha * G_t * grad log pi(a_t | s_t),
where G_t is the (discounted) return following step t. This is REINFORCE.
(We use a deterministic GridWorld here so the signal is clean.)
"""
import numpy as np
from gridworld import ROWS, COLS, WALL, GOAL, PIT, GAMMA, A_LIST, ACTIONS, reward, is_terminal, in_grid

rng = np.random.default_rng(0)
STATES = [(r, c) for r in range(ROWS) for c in range(COLS) if (r, c) != WALL]

def det_step(s, a):
    dr, dc = ACTIONS[a]; nr, nc = s[0] + dr, s[1] + dc
    ns = (nr, nc) if in_grid(nr, nc) else s
    return ns, reward(ns)

def softmax(x):
    x = x - x.max(); e = np.exp(x); return e / e.sum()

def run_episode(theta, max_t=50):
    s = (0, 0); traj = []
    for t in range(max_t):
        if is_terminal(s): break
        p = softmax(theta[s]); a_i = rng.choice(len(A_LIST), p=p)
        ns, r = det_step(s, A_LIST[a_i]); traj.append((s, a_i, r)); s = ns
    return traj

def reinforce(episodes=3000, alpha=0.05, baseline=False):
    theta = {s: np.zeros(len(A_LIST)) for s in STATES}
    curve = []; run_b = 0.0
    for ep in range(episodes):
        traj = run_episode(theta)
        # discounted returns G_t
        G = 0.0; returns = []
        for (s, a_i, r) in reversed(traj):
            G = r + GAMMA * G; returns.append(G)
        returns.reverse()
        b = run_b if baseline else 0.0
        for (s, a_i, _), Gt in zip(traj, returns):
            p = softmax(theta[s]); grad = -p; grad[a_i] += 1.0   # grad log pi
            theta[s] += alpha * (Gt - b) * grad
        ep_ret = sum(r for _, _, r in traj)
        run_b = 0.95 * run_b + 0.05 * (returns[0] if returns else 0.0)
        if ep % 20 == 0: curve.append((ep, ep_ret))
    return theta, curve

def main():
    theta, curve = reinforce()
    arrows = {"up": "^", "down": "v", "left": "<", "right": ">"}
    print("Learned policy (argmax pi):")
    for r in range(ROWS):
        line = []
        for c in range(COLS):
            s = (r, c)
            if s == WALL: line.append("#")
            elif s == GOAL: line.append("G")
            elif s == PIT: line.append("X")
            else: line.append(arrows[A_LIST[int(np.argmax(theta[s]))]])
        print(" ".join(line))
    print(f"\nFinal episode return (smoothed): {np.mean([r for _,r in curve[-20:]]):+.3f}")
    return theta, curve

if __name__ == "__main__":
    main()
