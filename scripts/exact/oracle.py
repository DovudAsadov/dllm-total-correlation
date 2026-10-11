#!/usr/bin/env python3
"""
Produce the CPU-only (oracle) results for the paper: contributions 1, 2, 4.

Writes JSON to results/ and figures to figures/. Everything here is exact or
validated against exact ground truth -- no neural network. This is the spine of
the paper and it stands alone even if the GPU confirmation never runs.

Experiments:
  E0  ground-truth table         : analytic vs enumerated C for every task
  E1  meter recovery             : meter C vs analytic C on the oracle (+ M sweep)
  E2  matched-marginal blindness : the decisive separation, n = 4 and 6
  E3  proxy-fidelity audit       : Spearman/Kendall/AUROC of every proxy vs C
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from tcaudit import tasks  # noqa: E402
from tcaudit.audit import matched_marginal_separation, run_audit  # noqa: E402
from tcaudit.backends import OracleBackend  # noqa: E402
from tcaudit.information import entropy_bits, total_correlation  # noqa: E402
from tcaudit.meter import meter_C_curve_over_orders, meter_total_correlation  # noqa: E402
from tcaudit.samples import (  # noqa: E402
    graded_samples,
    natural_audit_samples,
    pooled_audit_samples,
)

RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)


def dump(name, obj):
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote {path}")


def e0_ground_truth():
    print("[E0] ground-truth table (analytic vs enumerated)")
    out = {"replace_random": {}, "shuffle": {}, "shuffle_no_identity": {}}
    for n in (3, 4, 5, 6):
        out["replace_random"][n] = {
            "analytic": tasks.analytic_C_replace_random(n),
            "enumerated": total_correlation(tasks.replace_random(n)),
        }
        out["shuffle"][n] = {
            "analytic": tasks.analytic_C_shuffle(n),
            "enumerated": total_correlation(tasks.shuffle(n)),
        }
        out["shuffle_no_identity"][n] = {
            "analytic": tasks.analytic_C_shuffle(n, exclude_identity=True),
            "enumerated": total_correlation(tasks.shuffle(n, exclude_identity=True)),
        }
    out["matched_pair_n4"] = {
        "C_free": 0.0,
        "C_constrained": tasks.analytic_C_constrained_coins(4, 2),
        "note": "4 - log2(6) = 1.41504",
    }
    out["replace_random_limit"] = {
        "value": tasks.analytic_C_replace_random(100000),
        "target_log2_e": math.log2(math.e),
    }
    # assert exact agreement
    max_gap = 0.0
    for task in ("replace_random", "shuffle", "shuffle_no_identity"):
        for n, d in out[task].items():
            max_gap = max(max_gap, abs(d["analytic"] - d["enumerated"]))
    out["max_analytic_vs_enum_gap"] = max_gap
    print(f"  max |analytic - enumerated| = {max_gap:.2e}")
    dump("e0_ground_truth.json", out)
    return out


def e1_meter_recovery():
    print("[E1] meter recovery on the oracle")
    rng = np.random.default_rng(12345)
    recovery = {}
    for name, joint, S in [
        ("replace_random_n3", tasks.replace_random(3), (0, 1, 2)),
        ("replace_random_n4", tasks.replace_random(4), (0, 1, 2, 3)),
        ("replace_random_n5", tasks.replace_random(5), tuple(range(5))),
        ("shuffle_n4", tasks.shuffle(4), (0, 1, 2, 3)),
        ("constrained_coins_4_2", tasks.constrained_coins(4, 2), (0, 1, 2, 3)),
        ("free_coins_4", tasks.free_coins(4), (0, 1, 2, 3)),
    ]:
        b = OracleBackend(joint)
        res = meter_total_correlation(b, S, {}, n_orders=6000, rng=rng)
        analytic = total_correlation(joint, list(S))
        recovery[name] = {
            "analytic_C": analytic,
            "meter_C": res.C_bits,
            "abs_error": abs(res.C_bits - analytic),
            "meter_nfe": res.meter_nfe,
            "order_std_bits": res.order_std_bits,
        }
        print(f"  {name:24s} analytic={analytic:.4f} meter={res.C_bits:.4f} "
              f"err={abs(res.C_bits-analytic):.4f}")

    # M-vs-variance sweep (fixes the M used elsewhere): repeat at each M, report std
    print("  M-vs-variance sweep (replace_random_n4)")
    joint = tasks.replace_random(4)
    b = OracleBackend(joint)
    analytic = total_correlation(joint)
    sweep = {}
    for M in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        ests = []
        for rep in range(40):
            r = meter_total_correlation(b, (0, 1, 2, 3), {}, n_orders=M,
                                        rng=np.random.default_rng(1000 + rep))
            ests.append(r.C_bits)
        sweep[M] = {
            "mean": float(np.mean(ests)),
            "std": float(np.std(ests)),
            "rmse": float(np.sqrt(np.mean((np.array(ests) - analytic) ** 2))),
        }
    out = {"recovery": recovery, "M_sweep_replace_random_n4": sweep,
           "M_sweep_analytic": analytic}
    dump("e1_meter_recovery.json", out)
    return out


def e2_matched_marginal():
    print("[E2] matched-marginal blindness")
    out = {}
    for n in (4, 6):
        out[f"n{n}"] = matched_marginal_separation(None, n=n)
        print(f"  n={n}: {out[f'n{n}']['verdict']}")
    dump("e2_matched_marginal.json", out)
    return out


def _audit_to_dict(res):
    return {
        "n_safe": res.n_safe,
        "n_unsafe": res.n_unsafe,
        "safe_threshold": res.safe_threshold,
        "proxies": [
            {
                "name": r.name,
                "spearman_vs_C": r.spearman_vs_C,
                "kendall_vs_C": r.kendall_vs_C,
                "spearman_ci_lo": r.spearman_ci[0],
                "spearman_ci_hi": r.spearman_ci[1],
                "auroc_safe": r.auroc_safe,
                "n_samples": r.n_samples,
            }
            for r in res.proxy_reports
        ],
    }


def _iff_structure(prepared, safe_threshold=1e-6):
    """On a sample set, check the claim 'C = 0 iff the summed marginal entropy
    is 0' and report the separation in summed marginal entropy (the 'budget')
    between the C=0 and C>0 groups. This is what makes AUROC_safe an identity
    rather than an empirical finding: every marginal proxy is a monotone
    function of the per-position entropies, so if C=0 coincides exactly with
    zero entropy, every proxy is pinned at its maximum on exactly the safe sets.
    """
    C = np.array([p.C_true for p in prepared], float)
    budget = np.array([sum(entropy_bits(d.values()) for d in p.dists)
                       for p in prepared], float)
    safe = C <= safe_threshold
    zero_budget = budget <= 1e-12
    return {
        "n": int(len(C)),
        "n_safe": int(safe.sum()),
        "n_unsafe": int((~safe).sum()),
        "n_zero_budget": int(zero_budget.sum()),
        "n_safe_and_zero_budget": int((safe & zero_budget).sum()),
        "n_safe_but_positive_budget": int((safe & ~zero_budget).sum()),
        "n_unsafe_but_zero_budget": int((~safe & zero_budget).sum()),
        "iff_holds": bool((safe == zero_budget).all()),
        "max_budget_on_safe": float(budget[safe].max()) if safe.any() else float("nan"),
        "min_budget_on_unsafe": float(budget[~safe].min()) if (~safe).any() else float("nan"),
        "budget_separation_bits": (
            float(budget[~safe].min() - budget[safe].max())
            if safe.any() and (~safe).any() else float("nan")
        ),
        "C_range": [float(C.min()), float(C.max())],
    }


def e3_proxy_audit():
    print("[E3] proxy-fidelity audit")
    out = {}

    # --- (a) NATURAL: semantic tasks only. This is the observational population
    #         and the ONLY set whose numbers may be labelled "natural".
    natural = natural_audit_samples(n_perm=4)
    res_nat = run_audit(natural, bootstrap=2000)
    out["natural"] = _audit_to_dict(res_nat)
    out["natural"]["tasks"] = list(("replace_random", "shuffle", "shuffle_no_identity"))
    out["natural_iff"] = _iff_structure(natural)
    print(f"  natural: {out['natural']['n_safe']} safe / "
          f"{out['natural']['n_unsafe']} unsafe, {len(natural)} samples")
    for r in res_nat.proxy_reports:
        print(f"    {r.name:18s} spearman={r.spearman_vs_C:+.4f} "
              f"[{r.spearman_ci[0]:+.4f}, {r.spearman_ci[1]:+.4f}] "
              f"kendall={r.kendall_vs_C:+.4f} auroc_safe={r.auroc_safe:.4f}")
    ii = out["natural_iff"]
    print(f"    iff(C=0 <=> sum_i H(Y_i)=0): {ii['iff_holds']} "
          f"({ii['n_safe_and_zero_budget']}/{ii['n_safe']} safe, "
          f"0 counterexamples); max budget on safe = {ii['max_budget_on_safe']:.6g} bits, "
          f"min budget on unsafe = {ii['min_budget_on_unsafe']:.6g} bits")

    # --- (b) GRADED: the confound-removed control (marginals pinned).
    graded = graded_samples()
    out["graded"] = _audit_to_dict(run_audit(graded, bootstrap=2000))
    out["graded_iff"] = _iff_structure(graded)
    print(f"  graded: {out['graded']['n_safe']} safe / {out['graded']['n_unsafe']} unsafe")

    # --- (c) POOLED: natural + graded together. Kept for continuity, but it is
    #         NOT a natural-task statistic and must not be reported as one.
    pooled = pooled_audit_samples(n_perm=4)
    res = run_audit(pooled, bootstrap=2000)
    out["pooled"] = _audit_to_dict(res)
    out["pooled_iff"] = _iff_structure(pooled)
    print(f"  pooled: {out['pooled']['n_safe']} safe / {out['pooled']['n_unsafe']} unsafe, "
          f"{len(pooled)} samples  [natural + graded control -- not 'natural']")
    for r in res.proxy_reports:
        print(f"    {r.name:18s} spearman={r.spearman_vs_C:+.3f} "
              f"kendall={r.kendall_vs_C:+.3f} auroc_safe={r.auroc_safe:.3f}")

    # also stash the raw scatter data (proxy score vs C) for the figures
    out["natural_scatter"] = {
        "C": res_nat.C_values,
        "scores": res_nat.proxy_scores,
    }
    out["pooled_scatter"] = {
        "C": res.C_values,
        "scores": res.proxy_scores,
    }
    dump("e3_proxy_audit.json", out)
    return out


def main():
    print("=" * 70)
    print("Oracle (CPU-only) experiments -- contributions 1, 2, 4")
    print("=" * 70)
    e0_ground_truth()
    e1_meter_recovery()
    e2_matched_marginal()
    e3_proxy_audit()
    print("\nAll oracle experiments complete. Results in results/.")


if __name__ == "__main__":
    main()
