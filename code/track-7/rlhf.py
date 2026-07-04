"""
Track 7 · Lesson 4 — RLHF, in miniature, from scratch
Run:  python rlhf.py

RLHF aligns a model with human preferences in two stages:
  (1) REWARD MODEL: humans compare pairs of outputs; we fit a reward r(x) to those
      preferences with the Bradley–Terry model (logistic regression on differences).
  (2) POLICY OPTIMIZATION: improve a policy to maximize the learned reward, but with
      a KL penalty toward a reference policy so it doesn't drift into nonsense
      (reward hacking). Objective:  maximize  E_pi[r(x)] - beta * KL(pi || pi_ref).
This toy uses a handful of discrete outputs so every step is transparent.
"""
import numpy as np
rng = np.random.default_rng(0)

N = 8                                              # number of possible outputs
true_q = rng.normal(0, 1, N)                       # hidden "human quality" of each

def sample_pref(i, j):
    """A human prefers i over j with Bradley–Terry prob sigmoid(q_i - q_j)."""
    return 1 if rng.random() < 1/(1+np.exp(-(true_q[i]-true_q[j]))) else 0

def train_reward_model(n_pairs=4000, lr=0.1):
    r = np.zeros(N)                                # learned reward per output
    for _ in range(n_pairs):
        i, j = rng.choice(N, 2, replace=False)
        y = sample_pref(i, j)                      # 1 if human preferred i
        p = 1/(1+np.exp(-(r[i]-r[j])))             # model's predicted preference
        g = (y - p)                                # gradient of BT log-likelihood
        r[i] += lr*g; r[j] -= lr*g
    return r - r.mean()                            # identifiable up to a constant

def optimize_policy(r, beta):
    """Closed-form KL-regularized optimum: pi(x) ∝ pi_ref(x) * exp(r(x)/beta)."""
    ref = np.ones(N)/N                             # uniform reference policy
    logits = np.log(ref) + r/beta
    pi = np.exp(logits - logits.max()); pi /= pi.sum()
    exp_reward = float((pi*r).sum())
    kl = float((pi*np.log(pi/ref)).sum())
    return pi, exp_reward, kl

def main():
    r = train_reward_model()
    corr = np.corrcoef(r, true_q)[0,1]
    print(f"Reward model recovers human preferences: corr(r, true_q) = {corr:.3f}")
    print("\n  beta   E[reward]    KL(pi||ref)   (smaller beta = chase reward harder)")
    for beta in [2.0, 1.0, 0.5, 0.25, 0.1]:
        pi, er, kl = optimize_policy(r, beta)
        print(f"  {beta:>4}   {er:+.3f}       {kl:.3f}")
    print("\nLower beta -> higher reward but larger KL drift from the reference.")
    return r

if __name__ == "__main__":
    main()
