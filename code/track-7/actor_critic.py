"""
Track 7 · Lesson 3 — Actor–Critic on GridWorld
Run:  python actor_critic.py

REINFORCE works but its updates are noisy: a whole episode's return G_t is a high-
variance estimate. Actor–Critic adds a learned value baseline V(s) (the "critic")
and updates the policy (the "actor") with the lower-variance ADVANTAGE
    A_t = r + gamma * V(s') - V(s)     (the one-step TD error),
which says "was this action better or worse than the state's average?"
"""
import numpy as np
from reinforce import det_step, softmax, STATES
from gridworld import ROWS, COLS, WALL, GOAL, PIT, GAMMA, A_LIST, is_terminal

rng = np.random.default_rng(0)

def actor_critic(episodes=3000, a_actor=0.1, a_critic=0.1):
    theta = {s: np.zeros(len(A_LIST)) for s in STATES}      # actor
    V = {s: 0.0 for s in STATES}                            # critic
    curve = []
    for ep in range(episodes):
        s = (0, 0); ep_ret = 0.0
        for t in range(50):
            if is_terminal(s): break
            p = softmax(theta[s]); a_i = rng.choice(len(A_LIST), p=p)
            ns, r = det_step(s, A_LIST[a_i]); ep_ret += r
            v_ns = 0.0 if is_terminal(ns) else V[ns]
            adv = r + GAMMA * v_ns - V[s]                   # TD advantage
            V[s] += a_critic * adv                          # critic update
            grad = -p; grad[a_i] += 1.0
            theta[s] += a_actor * adv * grad               # actor update (uses advantage)
            s = ns
        if ep % 20 == 0: curve.append((ep, ep_ret))
    return theta, V, curve

def main():
    theta, V, curve = actor_critic()
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
    return theta, V, curve

if __name__ == "__main__":
    main()
