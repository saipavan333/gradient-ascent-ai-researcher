"""
Track 7 · Lesson 1 — GridWorld MDP + Value Iteration (dynamic programming)
Run:  python gridworld.py

A Markov Decision Process is (States, Actions, Transitions P, Rewards R, discount gamma).
Here: a 4x4 grid. The agent starts top-left, wants the goal (+1) bottom-right, and
must avoid a pit (-1). Moves are "slippery": with prob 0.8 you go where you intend,
else you slip sideways. Value Iteration computes the optimal value of every state by
repeatedly applying the Bellman optimality backup until it stops changing.
"""
import numpy as np

ROWS, COLS = 4, 4
GOAL, PIT, WALL = (3, 3), (1, 3), (1, 1)
GAMMA = 0.9
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
A_LIST = list(ACTIONS)

def in_grid(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS and (r, c) != WALL

def reward(s):
    """Reward for ENTERING state s."""
    if s == GOAL: return 1.0
    if s == PIT:  return -1.0
    return -0.02                                   # small step cost: hurry up

def is_terminal(s):
    return s == GOAL or s == PIT

def transitions(s, a):
    """Slippery: 0.8 intended, 0.1 each perpendicular. Returns list of (prob, s')."""
    if is_terminal(s): return [(1.0, s)]
    intended = ACTIONS[a]
    perp = [(-intended[1], intended[0]), (intended[1], -intended[0])]
    out = []
    for prob, (dr, dc) in [(0.8, intended), (0.1, perp[0]), (0.1, perp[1])]:
        nr, nc = s[0] + dr, s[1] + dc
        ns = (nr, nc) if in_grid(nr, nc) else s     # bump into wall/edge -> stay
        out.append((prob, ns))
    return out

def value_iteration(theta=1e-6):
    V = {(r, c): 0.0 for r in range(ROWS) for c in range(COLS) if (r, c) != WALL}
    while True:
        delta = 0.0
        for s in V:
            if is_terminal(s):
                V[s] = 0.0; continue            # terminal: no future reward
            best = max(sum(p * (reward(ns) + GAMMA * (0.0 if is_terminal(ns) else V[ns]))
                           for p, ns in transitions(s, a)) for a in A_LIST)
            delta = max(delta, abs(best - V[s])); V[s] = best
        if delta < theta: break
    # greedy policy from V
    policy = {}
    for s in V:
        if is_terminal(s): policy[s] = None; continue
        policy[s] = max(A_LIST, key=lambda a: sum(p * (reward(ns) + GAMMA * (0.0 if is_terminal(ns) else V[ns]))
                                                   for p, ns in transitions(s, a)))
    return V, policy

def main():
    V, policy = value_iteration()
    print("Optimal state values (Value Iteration):")
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            s = (r, c)
            row.append(" WALL " if s == WALL else f"{V[s]:+.2f}")
        print("  ".join(row))
    print("\nOptimal policy (arrows):")
    arrows = {"up": "^", "down": "v", "left": "<", "right": ">"}
    for r in range(ROWS):
        line = []
        for c in range(COLS):
            s = (r, c)
            if s == WALL: line.append("#")
            elif s == GOAL: line.append("G")
            elif s == PIT: line.append("X")
            else: line.append(arrows[policy[s]])
        print(" ".join(line))
    return V, policy

if __name__ == "__main__":
    main()
