"""
Track 7 · Lesson 1 — Q-learning (model-free control) on the GridWorld
Run:  python q_learning.py

Value Iteration needed the full model (P and R). Q-learning does NOT: it learns
purely from experience via the temporal-difference update
    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ].
We explore with a decaying epsilon-greedy policy and use "exploring starts" (each
episode begins in a random state) so every state gets visited and learned.
"""
import numpy as np
from gridworld import (ROWS, COLS, WALL, GOAL, PIT, GAMMA, A_LIST,
                       reward, is_terminal, transitions)

rng = np.random.default_rng(0)
STATES = [(r, c) for r in range(ROWS) for c in range(COLS) if (r, c) != WALL]
NONTERM = [s for s in STATES if not is_terminal(s)]

def step(s, a):
    """Sample a next state from the (slippery) environment, and the reward of s."""
    probs = [p for p, _ in transitions(s, a)]
    nss   = [ns for _, ns in transitions(s, a)]
    ns = nss[rng.choice(len(nss), p=probs)]
    return ns, reward(ns)   # reward for ENTERING ns

def greedy_return(Q, start=(0, 0), rollouts=200):
    """Average return of the current greedy policy from `start` (evaluation only)."""
    tot = 0.0
    for _ in range(rollouts):
        s = start; g = 0.0
        for t in range(100):
            if is_terminal(s): break
            a = max(A_LIST, key=lambda a: Q[(s, a)])
            s, r = step(s, a); g += r
        tot += g
    return tot / rollouts

def q_learning(episodes=8000, alpha=0.2):
    Q = {(s, a): 0.0 for s in STATES for a in A_LIST}
    curve = []
    for ep in range(episodes):
        eps = max(0.02, 0.3 * (1 - ep / episodes))      # decay exploration
        s = NONTERM[rng.integers(len(NONTERM))]         # exploring start
        for t in range(100):
            if is_terminal(s): break
            a = rng.choice(A_LIST) if rng.random() < eps else \
                max(A_LIST, key=lambda a: Q[(s, a)])
            ns, r = step(s, a)
            best_next = 0.0 if is_terminal(ns) else max(Q[(ns, a2)] for a2 in A_LIST)
            Q[(s, a)] += alpha * (r + GAMMA * best_next - Q[(s, a)])   # TD update
            s = ns
        if ep % 100 == 0:
            curve.append((ep, greedy_return(Q)))
    return Q, curve

def policy_grid(Q):
    arrows = {"up": "^", "down": "v", "left": "<", "right": ">"}
    rows = []
    for r in range(ROWS):
        line = []
        for c in range(COLS):
            s = (r, c)
            if s == WALL: line.append("#")
            elif s == GOAL: line.append("G")
            elif s == PIT: line.append("X")
            else: line.append(arrows[max(A_LIST, key=lambda a: Q[(s, a)])])
        rows.append(" ".join(line))
    return "\n".join(rows)

def main():
    Q, curve = q_learning()
    print(f"Trained Q-learning ({len(curve)*100} episodes, exploring starts).")
    print("\nLearned policy (greedy w.r.t. Q):")
    print(policy_grid(Q))
    print(f"\nGreedy return from (0,0): {greedy_return(Q):+.3f}  (undiscounted; ~+0.86 is near-optimal)")
    return Q, curve

if __name__ == "__main__":
    main()
