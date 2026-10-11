#!/usr/bin/env python3
"""
Does the headline residual depend on the meter's setting of M?

The convergence study (e9) shows the estimator's own standard error falls as
1/sqrt(M). That is necessary but not sufficient: what a reviewer actually needs is that
the *reported result* -- the residual left by the best marginal-only predictor -- does not
move when the estimator gets four times more precise. The noise correction is supposed
to handle exactly that, but "our correction is validated on synthetic data" is a weaker
statement than "we ran it at 4x M and the number did not move".

This re-runs explains_C on the same checkpoint at M=64 and M=256 and compares:
  * the measured C itself (mean, sd),
  * the estimator's standard error,
  * the recovered fraction and the residual in bits,
  * agreement block-by-block, since both runs use the same blocks and seed.

Usage: python3 scripts/analysis/m_sensitivity.py [tag]
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
_s = importlib.util.spec_from_file_location(
    "av3", os.path.join(ROOT, "scripts", "analysis", "analyze.py"))
av3 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(av3)

TAG = sys.argv[1] if len(sys.argv) > 1 else "llada_base"
lo_p = os.path.join(RES, f"e5v3_{TAG}.json")
hi_p = os.path.join(RES, f"e5v3_{TAG}_m256.json")
for f in (lo_p, hi_p):
    if not os.path.exists(f):
        sys.exit(f"missing {f}")

LO = json.load(open(lo_p))
HI = json.load(open(hi_p))
m_lo = LO["config"]["m_meter"]
m_hi = HI["config"]["m_meter"]
print(f"{TAG}:  M={m_lo}  vs  M={m_hi}\n")

out = {"tag": TAG, "m_low": m_lo, "m_high": m_hi, "by_regime": {}}
hdr = (f"{'regime':8s} {'n':>4} {'C(M='+str(m_lo)+')':>10} {'C(M='+str(m_hi)+')':>11} "
       f"{'SE lo':>7} {'SE hi':>7} {'R2 lo':>6} {'R2 hi':>6} {'res lo':>7} {'res hi':>7} "
       f"{'|dC| med':>9}")
print(hdr)
print("-" * len(hdr))

for rg in ("suffix", "infill"):
    a = [r for r in LO["records"] if r["regime"] == rg]
    b = [r for r in HI["records"] if r["regime"] == rg]
    if not a or not b:
        continue
    # same corpus, same seed, same block construction -> align on (doc_id, start)
    key = lambda r: (r["doc_id"], r["start"])          # noqa: E731
    ib = {key(r): r for r in b}
    pairs = [(x, ib[key(x)]) for x in a if key(x) in ib]
    ca = np.array([x["C_hat_bits"] for x, _ in pairs])
    cb = np.array([y["C_hat_bits"] for _, y in pairs])
    ea = av3.explains_C([x for x, _ in pairs])
    eb = av3.explains_C([y for _, y in pairs])
    row = {
        "n_matched": len(pairs),
        "C_mean": [float(ca.mean()), float(cb.mean())],
        "C_sd_true": [ea["C_sd_true_bits"], eb["C_sd_true_bits"]],
        "meter_se_median": [ea["meter_se_median_bits"], eb["meter_se_median_bits"]],
        "r2_true": [ea["best_marginal_predictor"]["r2_true"],
                    eb["best_marginal_predictor"]["r2_true"]],
        "residual_bits": [ea["unexplained_sd_bits"], eb["unexplained_sd_bits"]],
        "median_abs_delta_C": float(np.median(np.abs(ca - cb))),
        "spearman_C_lo_hi": av3.spearman(ca, cb),
    }
    out["by_regime"][rg] = row
    print(f"{rg:8s} {len(pairs):4d} {ca.mean():10.2f} {cb.mean():11.2f} "
          f"{row['meter_se_median'][0]:7.2f} {row['meter_se_median'][1]:7.2f} "
          f"{100*row['r2_true'][0]:6.0f} {100*row['r2_true'][1]:6.0f} "
          f"{row['residual_bits'][0]:7.2f} {row['residual_bits'][1]:7.2f} "
          f"{row['median_abs_delta_C']:9.2f}")

print()
for rg, r in out["by_regime"].items():
    d = abs(r["residual_bits"][1] - r["residual_bits"][0])
    print(f"  {rg}: residual moves {d:.2f} bits ({100*d/max(r['residual_bits'][0],1e-9):.0f}%) "
          f"when M goes {m_lo} -> {m_hi}; rank agreement of C rho={r['spearman_C_lo_hi']:.3f}")

p = os.path.join(RES, f"e10_m_sensitivity_{TAG}.json")
json.dump(out, open(p, "w"), indent=1)
print(f"\nwrote {p}")
