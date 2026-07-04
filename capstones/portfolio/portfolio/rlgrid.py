"""
A GridWorld MDP and a tabular Q-learning agent that trains incrementally, so a web
UI can watch the policy improve in real time. This is the Track 7 algorithm,
packaged for serving. Pure NumPy.
"""
import numpy as np

ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
A_LIST = list(ACTIONS)

class GridWorld:
    def __init__(self, rows=4, cols=4, goal=(3, 3), pit=(1, 3), wall=(1, 1),
                 gamma=0.9, slip=0.1, step_cost=-0.02):
        self.rows, self.cols, self.goal, self.pit, self.wall = rows, cols, goal, pit, wall
        self.gamma, self.slip, self.step_cost = gamma, slip, step_cost
        self.states = [(r, c) for r in range(rows) for c in range(cols) if (r, c) != wall]

    def in_grid(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols and (r, c) != self.wall

    def terminal(self, s):
        return s == self.goal or s == self.pit

    def reward(self, s):                                  # reward for ENTERING s
        if s == self.goal: return 1.0
        if s == self.pit:  return -1.0
        return self.step_cost

    def step(self, s, a, rng):
        intended = ACTIONS[a]
        perp = [(-intended[1], intended[0]), (intended[1], -intended[0])]
        moves = [(1 - 2*self.slip, intended), (self.slip, perp[0]), (self.slip, perp[1])]
        probs = [p for p, _ in moves]
        dr, dc = moves[rng.choice(len(moves), p=probs)][1]
        ns = (s[0]+dr, s[1]+dc) if self.in_grid(s[0]+dr, s[1]+dc) else s
        return ns, self.reward(ns)

class QAgent:
    def __init__(self, env, alpha=0.2, seed=0):
        self.env, self.alpha = env, alpha
        self.rng = np.random.default_rng(seed)
        self.Q = {(s, a): 0.0 for s in env.states for a in A_LIST}
        self.episodes = 0
        self.nonterm = [s for s in env.states if not env.terminal(s)]

    def train(self, episodes=200, eps=0.15):
        e = self.env
        for _ in range(episodes):
            s = self.nonterm[self.rng.integers(len(self.nonterm))]   # exploring start
            for _ in range(100):
                if e.terminal(s): break
                a = (self.rng.choice(A_LIST) if self.rng.random() < eps
                     else max(A_LIST, key=lambda a: self.Q[(s, a)]))
                ns, r = e.step(s, a, self.rng)
                best = 0.0 if e.terminal(ns) else max(self.Q[(ns, b)] for b in A_LIST)
                self.Q[(s, a)] += self.alpha * (r + e.gamma * best - self.Q[(s, a)])
                s = ns
            self.episodes += 1
        return self

    def greedy_return(self, start=(0, 0), rollouts=200):
        e = self.env; tot = 0.0
        for _ in range(rollouts):
            s, g = start, 0.0
            for _ in range(100):
                if e.terminal(s): break
                a = max(A_LIST, key=lambda a: self.Q[(s, a)])
                s, r = e.step(s, a, self.rng); g += r
            tot += g
        return tot / rollouts

    def state(self):
        """JSON-serializable snapshot of values + greedy policy for the UI."""
        e = self.env; cells = []
        for r in range(e.rows):
            for c in range(e.cols):
                s = (r, c)
                kind = ("wall" if s == e.wall else "goal" if s == e.goal
                        else "pit" if s == e.pit else "cell")
                v = 0.0 if e.terminal(s) or s == e.wall else max(self.Q[(s, a)] for a in A_LIST)
                pol = None if kind != "cell" else max(A_LIST, key=lambda a: self.Q[(s, a)])
                cells.append({"r": r, "c": c, "kind": kind, "value": round(float(v), 3), "policy": pol})
        return {"rows": e.rows, "cols": e.cols, "cells": cells,
                "episodes": self.episodes, "return": round(self.greedy_return(), 3)}
