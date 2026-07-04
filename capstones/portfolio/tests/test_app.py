"""Smoke + API tests for the unified portfolio app. Run: pytest -q"""
import app as A
c = A.app.test_client()

def test_pages_load():
    for path in ["/", "/namegen", "/ddlab", "/gridworld"]:
        r = c.get(path); assert r.status_code == 200

def test_health():
    j = c.get("/health").get_json()
    assert j["status"] == "ok" and set(j["demos"]) == {"namegen", "ddlab", "gridworld"}

def test_namegen_api():
    j = c.get("/namegen/api/generate?count=5&prefix=al").get_json()
    assert len(j["names"]) == 5 and all(isinstance(n, str) for n in j["names"])

def test_ddlab_api():
    j = c.get("/ddlab/api/curve?noise=0.5&ridge=1e-6").get_json()
    assert 0.8 <= j["peak_ratio"] <= 1.3 and len(j["image"]) > 1000

def test_gridworld_learns():
    c.get("/gridworld/api/reset")
    s = c.get("/gridworld/api/train?episodes=1500").get_json()
    assert s["episodes"] == 1500 and s["return"] > 0.5      # agent actually learns
