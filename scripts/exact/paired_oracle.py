#!/usr/bin/env python3
"""Paired-difference bootstrap for the oracle proxy audit.

The paper claimed "margin dominates confidence" from two NON-OVERLAPPING marginal CIs.
On paired data (same 355 candidate sets scored by every proxy) that is the wrong test:
it answers "could these two correlations have come from unrelated samples", not "is
margin better on THIS sample". The correct statistic is a paired bootstrap of the
DIFFERENCE, which is also strictly more powerful.
"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tcaudit.stats import spearman, kendall_tau

d = json.load(open(os.path.join(ROOT, "results", "e3_proxy_audit.json")))
sc = d["natural_scatter"]
C = np.asarray(sc["C"], float)
S = {k: np.asarray(v, float) for k, v in sc["scores"].items()}
n = len(C)
print(f"natural set: n={n}, proxies={list(S)}")

def paired_diff(a, b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    obs = spearman(S[a], C) - spearman(S[b], C)
    out = np.empty(n_boot)
    for i in range(n_boot):
        ii = rng.integers(0, n, n)
        out[i] = spearman(S[a][ii], C[ii]) - spearman(S[b][ii], C[ii])
    lo, hi = np.quantile(out, [0.025, 0.975])
    return obs, lo, hi

res = {"n": n, "spearman": {k: spearman(v, C) for k, v in S.items()},
       "kendall": {k: kendall_tau(v, C) for k, v in S.items()}, "paired_differences": {}}
pairs = [("margin_min", "confidence_min"), ("margin_min", "neg_entropy_sum"),
         ("neg_entropy_sum", "confidence_min"), ("margin_min", "confidence_mean")]
print("\npaired bootstrap of rho(A,C) - rho(B,C)   [negative => A ranks C better]")
for a, b in pairs:
    o, lo, hi = paired_diff(a, b)
    sig = "SIGNIFICANT" if hi < 0 or lo > 0 else "not significant"
    res["paired_differences"][f"{a}__minus__{b}"] = {
        "diff": float(o), "ci95": [float(lo), float(hi)],
        "significant": bool(hi < 0 or lo > 0)}
    print(f"  {a:18s} - {b:18s} = {o:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {sig}")

with open(os.path.join(ROOT, "results", "e3_paired_tests.json"), "w") as f:
    json.dump(res, f, indent=1)
print("\nwrote results/e3_paired_tests.json")
