"""
Track 20 · Lesson 3 — LLM agents & tool use
===========================================
A language model alone can only *predict text* — it can't reliably do exact
arithmetic or look up live facts. An AGENT wraps the model in a loop: it REASONS
about what to do, calls a TOOL (calculator, search, code), OBSERVES the result, and
repeats until it can answer. This "ReAct" loop turns a text predictor into something
that can act. Here a tool-using agent solves problems the model-alone gets wrong.

Run:  python llm_agents.py
"""
import re

# --- tools the agent can call ---
def calculator(expr):
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", expr): return "error"
    return eval(expr)                                    # safe: only arithmetic chars allowed
FACTS = {"capital of france": "Paris", "author of hamlet": "Shakespeare"}
def lookup(q):
    for k, v in FACTS.items():
        if k in q.lower(): return v
    return "unknown"

# --- the "LLM" without tools: guesses arithmetic (rounds operands — unreliable) ---
def llm_alone(q):
    m = re.search(r"(\d+)\s*([\*\+\-])\s*(\d+)", q)
    if m:
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        a2, b2 = round(a, -1), round(b, -1)             # "mental math": round, so it's approximate
        return {"*": a2*b2, "+": a2+b2, "-": a2-b2}[op]
    return "I don't know"

# --- the AGENT: reason -> choose tool -> act -> observe -> answer ---
def agent(q, verbose=False):
    m = re.search(r"(\d+\s*[\*\+\-]\s*\d+)", q)
    if m:                                               # reasoning: "this needs arithmetic"
        if verbose: print(f"   [agent] thought: needs math → call calculator('{m.group(1)}')")
        return calculator(m.group(1))
    if verbose: print("   [agent] thought: needs a fact → call lookup()")
    return lookup(q)

if __name__ == "__main__":
    tasks = [("What is 47 * 89?", 47*89),
             ("Compute 1234 + 5678", 1234+5678),
             ("What is the capital of France?", "Paris"),
             ("Who is the author of Hamlet?", "Shakespeare")]
    agent_ok = alone_ok = 0
    print("task-by-task:")
    for q, truth in tasks:
        a = agent(q); b = llm_alone(q)
        agent_ok += (a == truth); alone_ok += (b == truth)
        print(f"  Q: {q:34s} truth={truth}   agent={a}   LLM-alone={b}")
    print(f"\nagent (with tools): {agent_ok}/{len(tasks)} correct")
    print(f"LLM alone (no tools): {alone_ok}/{len(tasks)} correct")
    assert agent_ok == len(tasks) and alone_ok < len(tasks)
    print("OK — the reason→act→observe loop lets the agent use tools to get exact answers.")
