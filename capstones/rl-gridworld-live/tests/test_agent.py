"""Tests for the GridWorld Q-learning agent. Run: pytest -q"""
from rlgrid.agent import GridWorld, QAgent, A_LIST

def test_env_rewards():
    e = GridWorld()
    assert e.reward(e.goal) == 1.0 and e.reward(e.pit) == -1.0 and e.reward((0,0)) < 0
    assert e.terminal(e.goal) and not e.terminal((0,0))

def test_state_serializable():
    a = QAgent(GridWorld())
    s = a.state()
    assert set(s) >= {"rows","cols","cells","episodes","return"}
    assert len(s["cells"]) == s["rows"]*s["cols"]
    assert all(set(c) >= {"r","c","kind","value","policy"} for c in s["cells"])

def test_learning_improves_return():
    a = QAgent(GridWorld(), seed=0)
    before = a.greedy_return()
    a.train(episodes=1500)
    after = a.greedy_return()
    assert after > before + 0.3        # learns a clearly better policy

def test_learned_policy_reaches_goal():
    a = QAgent(GridWorld(), seed=0); a.train(episodes=2000)
    # from the start, the greedy return should be close to optimal (~+0.86)
    assert a.greedy_return() > 0.5
