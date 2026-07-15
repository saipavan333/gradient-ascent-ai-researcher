"""
Track 18 · Lesson 4 — Estimating treatment effects (ATE via IPW)
================================================================
The Average Treatment Effect (ATE) is E[Y | do(T=1)] - E[Y | do(T=0)]. With a
continuous confounder we can't just stratify. INVERSE PROPENSITY WEIGHTING (IPW)
fixes the confounding by re-weighting: estimate each unit's propensity e(Z)=P(T=1|Z),
then up-weight under-represented treatment choices so the weighted groups are balanced.

Run:  python treatment_effects.py
"""
import numpy as np
rng = np.random.default_rng(0)

def sigmoid(z): return 1/(1+np.exp(-z))

if __name__ == "__main__":
    n = 20000
    Z = rng.standard_normal(n)                          # continuous confounder
    T = (rng.random(n) < sigmoid(1.5*Z)).astype(int)    # confounded assignment
    Y = 3.0*Z + 4.0*T + rng.standard_normal(n)          # TRUE ATE of T is 4

    naive = Y[T==1].mean() - Y[T==0].mean()
    # estimate propensity e(Z)=P(T=1|Z) by logistic regression on Z
    w = 0.0; b = 0.0
    for _ in range(500):
        p = sigmoid(w*Z + b); e = p - T
        w -= 0.1*(e*Z).mean(); b -= 0.1*e.mean()
    e_hat = np.clip(sigmoid(w*Z + b), 0.05, 0.95)
    # IPW estimator
    ate_ipw = (T*Y/e_hat).sum()/(T/e_hat).sum() - ((1-T)*Y/(1-e_hat)).sum()/((1-T)/(1-e_hat)).sum()
    print(f"naive difference in means: {naive:+.2f}   (confounded)")
    print(f"IPW estimate of the ATE:   {ate_ipw:+.2f}   (true ATE = 4)")
    assert abs(ate_ipw - 4.0) < 0.5 and abs(naive - 4.0) > 1
    print("OK — inverse propensity weighting rebalances the groups and recovers the true ATE.")
