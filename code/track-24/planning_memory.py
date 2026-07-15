"""
Track 20 · Lesson 4 — Planning & memory
=======================================
A capable agent doesn't just react step-by-step — it PLANS a sequence of actions
toward a goal, and uses MEMORY to track where it has been so it doesn't loop forever.
Here an agent must reach a target number from a start using the operations +3 and ×2.
A greedy agent (always take the step that looks closest now) gets stuck; a planner
that searches with memory of visited states finds a correct plan.

Run:  python planning_memory.py
"""
from collections import deque

def greedy(start, target, ops, max_steps=20):
    """React step-by-step: always pick the op that gets closest to the target."""
    x = start; path = []
    for _ in range(max_steps):
        if x == target: return path
        best = min(ops, key=lambda o: abs(o(x) - target))   # no lookahead, no memory
        path.append(best.__name__); x = best(x)
        if abs(x) > 10 * abs(target) + 100: break
    return None                                             # failed / stuck

def planner(start, target, ops):
    """Breadth-first search: PLAN ahead, and use MEMORY (visited) to avoid revisiting."""
    q = deque([(start, [])]); seen = {start}               # 'seen' = the agent's memory
    while q:
        x, path = q.popleft()
        if x == target: return path
        if len(path) > 15: continue
        for o in ops:
            nx = o(x)
            if nx not in seen and abs(nx) <= 10*abs(target)+100:
                seen.add(nx); q.append((nx, path + [o.__name__]))
    return None

if __name__ == "__main__":
    def add3(x): return x + 3
    def mul2(x): return x * 2
    ops = [add3, mul2]
    start, target = 1, 23                                  # needs a non-greedy sequence

    g = greedy(start, target, ops)
    p = planner(start, target, ops)
    print(f"goal: turn {start} into {target} using +3 and ×2")
    print(f"greedy agent (no planning/memory): {g}")
    print(f"planner (search + memory):         {p}")
    # verify the plan actually works
    x = start
    for step in p: x = add3(x) if step == "add3" else mul2(x)
    print(f"executing the plan reaches: {x}")
    assert g is None and p is not None and x == target
    print("OK — planning ahead with memory solves what greedy step-by-step reaction cannot.")
