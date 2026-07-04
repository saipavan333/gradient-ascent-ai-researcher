"""
Track 14 · Lesson 0 — Specification gaming: the agent that games its reward
Run:  python specification_gaming.py

We can rarely write down exactly what we want, so we give an agent a PROXY reward. A
capable optimizer will then do whatever maximizes that proxy — even if it violates our
intent. Here a Q-learning agent (Track 7) in a corridor is rewarded for standing on a
'coin' tile. The TRUE goal is to reach the exit — but the agent learns to camp on the
coin forever, never exiting. Capability without alignment.
"""
import numpy as np
rng = np.random.default_rng(0)

N = 5; COIN, EXIT = 2, 4; GAMMA = 0.95
ACTIONS = [-1, 0, +1]                                  # left, stay, right

def step(s, a):
    ns = min(N-1, max(0, s + a))
    return ns

def reward(ns, kind):
    if kind == "proxy": return 0.3 if ns == COIN else 0.0   # points for being on the coin
    else:               return 1.0 if ns == EXIT else 0.0   # true goal: reach the exit

def q_learn(kind, episodes=8000, alpha=0.2, eps=0.2):
    Q = np.zeros((N, len(ACTIONS)))
    for _ in range(episodes):
        s = 0
        for _ in range(30):
            if s == EXIT: break                         # exit is terminal
            a = rng.integers(3) if rng.random() < eps else int(np.argmax(Q[s]))
            ns = step(s, ACTIONS[a]); r = reward(ns, kind)
            target = r + (0 if ns == EXIT else GAMMA*Q[ns].max())
            Q[s, a] += alpha*(target - Q[s, a]); s = ns
    return Q

def rollout(Q):
    s = 0; path = [s]
    for _ in range(15):
        if s == EXIT: break
        s = step(s, ACTIONS[int(np.argmax(Q[s]))]); path.append(s)
    return path, (path[-1] == EXIT)

def main():
    for kind in ["proxy", "true"]:
        Q = q_learn(kind); path, reached = rollout(Q)
        label = "PROXY reward (+0.3 per step on the coin)" if kind=="proxy" else "TRUE reward (+1 for the exit)"
        print(f"Agent trained on {label}:")
        print(f"   greedy path from start: {path}")
        print(f"   reached the exit? {'YES' if reached else 'NO — it camps on the coin!'}\n")
    print("Same capable agent, two reward specs, opposite behavior. The proxy-trained")
    print("agent 'succeeds' by its metric while completely failing the real objective —")
    print("specification gaming. Getting the reward right is the alignment problem.")

if __name__ == "__main__":
    main()
