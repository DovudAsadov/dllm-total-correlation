#!/usr/bin/env python3
"""
E8 -- NON-TIGHTNESS of marginal-only proxies (the paper's central theoretical result).

The old framing ("marginal proxies are provably blind to C") is FALSE as stated.
Since H(Y_S | c) >= 0, the definition

    C(S | c) = sum_{i in S} H(Y_i | c) - H(Y_S | c)                        (1)

immediately gives the *certified upper bound*

    C(S | c) <= sum_{i in S} H(Y_i | c) =: B(S | c)      ("the entropy budget")

and B is exactly minus the paper's own `neg_entropy_sum` proxy. So a marginal
proxy is not blind to C: it computes a valid, certified upper bound on C.

The defensible claim is NON-TIGHTNESS. Write M for the map that sends a joint
distribution P over the k candidate positions to its tuple of per-position
marginals (q_1, ..., q_k). A marginal-only proxy is by construction a function of
M(P) alone, so it cannot distinguish any two joints in the same *fibre*
M^{-1}(q_1,...,q_k). This script computes the achievable range of C over a fibre
-- the "fibre diameter" -- exactly.

STRUCTURE OF THE FIBRE (all of this is proved, then verified numerically here):

  F(q) := { P : P has marginals q_1, ..., q_k } is a transportation polytope:
  non-empty (it contains the product measure), convex and compact.

  (a) LOWER END.  The product measure prod_i q_i lies in F(q) and has C = 0.
      Since C >= 0 always, min_{P in F(q)} C(P) = 0 exactly, for every q.

  (b) THE RANGE IS AN INTERVAL.  C is continuous on F(q) and F(q) is convex hence
      connected, so { C(P) : P in F(q) } is an interval. With (a) it is exactly
      [0, C_max(q)].  Every value in between is achieved by some joint.

  (c) UPPER END.  B(q) = sum_i H(q_i) is fixed on the fibre, so
          C_max(q) = B(q) - min_{P in F(q)} H(P),
      i.e. maximising C is exactly the MINIMUM-ENTROPY COUPLING problem.
      H is concave, so its minimum over the polytope F(q) is attained at a
      VERTEX -- which is what makes the exact enumeration below possible.

  (d) CLOSED FORM FOR AN EQUAL-MARGINAL PROFILE.  For any coupling,
      H(P) >= max_i H(q_i) (a joint entropy dominates each of its marginals).
      If all k marginals are the SAME distribution q, the "diagonal" coupling
      Y_1 = Y_2 = ... = Y_k lies in F(q) and attains H(P) = H(q). Hence
          min_{P in F(q)} H(P) = H(q)  and  C_max = (k - 1) H(q),
      exactly, for every k and every q. In particular for k FAIR BITS the range
      is exactly [0, k - 1] bits, so the conjecture "k-1 bits" is TRUE and is a
      theorem, not a numerical accident.

  (e) NO SUCH CLOSED FORM FOR UNEQUAL MARGINALS.  C_max <= B(q) - max_i H(q_i)
      always, but this is generally NOT attained when the q_i differ; the script
      exhibits counterexamples so the paper does not overclaim (again).

  (f) MARGINAL-ONLY CERTIFICATE THAT SURVIVES UNEQUAL MARGINALS. Since
      max_i H(q_i) >= B(q)/k, we get the always-valid, budget-only bound
          C <= (1 - 1/k) B(q),
      with equality iff the profile is equal-entropy and perfectly couplable.
      This is what we can state for the real models, where E5 logs the budget
      B = -neg_entropy_sum per block but not the individual H(q_i).

Outputs
  results/e8_nontightness.json
  paper/tables/nontightness.tex
"""
from __future__ import annotations

import itertools
import json
import math
import os
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from tcaudit.information import entropy_bits  # noqa: E402
from tcaudit.samples import natural_audit_samples, pooled_audit_samples  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
PAPER = os.path.join(ROOT, "paper")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(PAPER, exist_ok=True)

TOL = 1e-9


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def h2(p: float) -> float:
    """Binary entropy in bits."""
    return entropy_bits([p, 1.0 - p])


def budget_of(dists) -> float:
    """B = sum_i H(q_i) in bits: the entropy budget of a candidate set.
    Exactly -neg_entropy_sum, the marginal proxy of the same name."""
    return sum(entropy_bits(d.values()) for d in dists)


def max_marginal_entropy(dists) -> float:
    return max(entropy_bits(d.values()) for d in dists)


# --------------------------------------------------------------------------
# PART 1 + 2 -- the certified bound on the real audit samples
# --------------------------------------------------------------------------

def check_bound(prepared, label):
    """Verify C_true <= sum_i H(q_i) on every sample, plus the refined bound
    C_true <= sum_i H(q_i) - max_i H(q_i), plus the zero-budget collapse."""
    n = len(prepared)
    slacks, slacks_refined = [], []
    slacks_pos, slacks_refined_pos = [], []   # restricted to samples with B > 0
    viol, viol_refined = 0, 0
    worst = None
    zero_budget_n, zero_budget_maxC = 0, 0.0
    C_max_seen = 0.0
    refined_tight = []
    for s in prepared:
        B = budget_of(s.dists)
        Hmax = max_marginal_entropy(s.dists)
        C = s.C_true
        C_max_seen = max(C_max_seen, C)
        sl = B - C
        slr = (B - Hmax) - C
        slacks.append(sl)
        slacks_refined.append(slr)
        if B > TOL:
            slacks_pos.append(sl)
            slacks_refined_pos.append(slr)
            if slr <= TOL:
                refined_tight.append(s.task)
        if sl < -TOL:
            viol += 1
        if slr < -TOL:
            viol_refined += 1
        if B > TOL and (worst is None or sl < worst[0]):
            worst = (sl, s.task, B, C)
        if B <= TOL:
            zero_budget_n += 1
            zero_budget_maxC = max(zero_budget_maxC, C)
    return {
        "label": label,
        "n_samples": n,
        "violations_budget_bound": viol,
        "violations_refined_bound": viol_refined,
        "min_slack_bits": float(min(slacks)),
        "median_slack_bits": float(statistics.median(slacks)),
        "max_slack_bits": float(max(slacks)),
        "min_slack_refined_bits": float(min(slacks_refined)),
        "median_slack_refined_bits": float(statistics.median(slacks_refined)),
        "n_positive_budget": len(slacks_pos),
        "min_slack_positive_budget_bits": float(min(slacks_pos)),
        "median_slack_positive_budget_bits": float(statistics.median(slacks_pos)),
        "min_slack_refined_positive_budget_bits": float(min(slacks_refined_pos)),
        "n_refined_bound_tight_positive_budget": len(refined_tight),
        "refined_bound_tight_tasks": sorted(set(refined_tight)),
        "tightest_sample": {"slack_bits": float(worst[0]), "task": worst[1],
                            "budget_bits": float(worst[2]), "C_bits": float(worst[3])},
        "n_zero_budget": zero_budget_n,
        "max_C_among_zero_budget": float(zero_budget_maxC),
        "max_C_bits": float(C_max_seen),
    }


# --------------------------------------------------------------------------
# PART 3a -- EXACT min-entropy coupling by vertex enumeration
# --------------------------------------------------------------------------

def _binary_constraints(ps):
    """A x = b for the transportation polytope of k binary positions with
    P(Y_i = 1) = ps[i]. Columns index the 2^k joint configs."""
    k = len(ps)
    cfgs = list(itertools.product((0, 1), repeat=k))
    A = np.zeros((k + 1, len(cfgs)))
    for j, cfg in enumerate(cfgs):
        for i in range(k):
            A[i, j] = 1.0 if cfg[i] == 1 else 0.0
        A[k, j] = 1.0
    b = np.array(list(ps) + [1.0])
    return A, b, cfgs


def min_entropy_coupling_binary_exact(ps):
    """EXACT minimum joint entropy over all couplings of k independent-alphabet
    binary marginals, by enumerating the vertices of the transportation polytope.

    Valid because H is concave, so min H over a compact convex polytope is
    attained at an extreme point, and every extreme point of {x >= 0 : Ax = b}
    has support of size <= rank(A) <= k+1.
    """
    k = len(ps)
    A, b, cfgs = _binary_constraints(ps)
    m = A.shape[0]  # k+1
    ncol = A.shape[1]
    best_H = math.inf
    best_x = None
    for size in range(1, m + 1):
        for sup in itertools.combinations(range(ncol), size):
            As = A[:, sup]
            xs, *_ = np.linalg.lstsq(As, b, rcond=None)
            if np.linalg.norm(As @ xs - b) > 1e-9:
                continue
            if np.any(xs < -1e-12):
                continue
            xs = np.clip(xs, 0.0, None)
            H = entropy_bits(xs.tolist())
            if H < best_H - 1e-12:
                best_H = H
                x = np.zeros(ncol)
                x[list(sup)] = xs
                best_x = x
    return float(best_H), best_x, cfgs


def random_couplings_check(ps, n_draws, rng, C_max):
    """Independent sanity check: draw many random joints with EXACTLY the given
    marginals (random seed matrix + iterative proportional fitting) and confirm
    none exceeds C_max, and that C values populate the interior of [0, C_max]."""
    k = len(ps)
    shape = (2,) * k
    B = sum(h2(p) for p in ps)
    seen = []
    for _ in range(n_draws):
        P = rng.random(shape) ** rng.uniform(1.0, 12.0) + 1e-12
        P /= P.sum()
        for _ in range(300):
            for i in range(k):
                axes = tuple(a for a in range(k) if a != i)
                cur = P.sum(axis=axes)
                target = np.array([1.0 - ps[i], ps[i]])
                scale = target / np.maximum(cur, 1e-300)
                sh = [1] * k
                sh[i] = 2
                P = P * scale.reshape(sh)
            P /= P.sum()
        # verify marginals converged
        ok = all(abs(P.sum(axis=tuple(a for a in range(k) if a != i))[1] - ps[i]) < 1e-8
                 for i in range(k))
        if not ok:
            continue
        C = B - entropy_bits(P.ravel().tolist())
        seen.append(C)
    if not seen:
        return {"n_valid": 0}
    return {
        "n_valid": len(seen),
        "max_C_seen": float(max(seen)),
        "min_C_seen": float(min(seen)),
        "n_exceeding_C_max": int(sum(1 for c in seen if c > C_max + 1e-9)),
    }


def interval_filling_check(ps, x_star, cfgs, n_steps=2001):
    """Verify (b): C sweeps the whole interval [0, C_max]. Walk the segment
    (1-t)*product + t*argmin, which stays in the fibre (it is convex), and
    report the largest gap between consecutive C values -- by the intermediate
    value theorem a small max gap certifies the interval is filled."""
    k = len(ps)
    B = sum(h2(p) for p in ps)
    prod = np.array([np.prod([ps[i] if cfg[i] == 1 else 1.0 - ps[i] for i in range(k)])
                     for cfg in cfgs])
    Cs = []
    for t in np.linspace(0.0, 1.0, n_steps):
        P = (1.0 - t) * prod + t * x_star
        Cs.append(B - entropy_bits(P.tolist()))
    Cs = np.array(Cs)
    gaps = np.diff(Cs)
    return {
        "n_steps": n_steps,
        "C_at_t0": float(Cs[0]),
        "C_at_t1": float(Cs[-1]),
        "max_consecutive_gap_bits": float(np.max(np.abs(gaps))),
        "monotone_increasing": bool(np.all(gaps >= -1e-12)),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

CONF_GRID = [0.5, 0.75, 0.9, 0.95, 0.99]   # trimmed for the paper table
P_GRID = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
K_TABLE = 4


def main():
    rng = np.random.default_rng(20260818)
    out = {}

    # ---- PART 1 & 2 : certified bound + zero-budget collapse -------------
    print("[E8.1] certified upper bound C <= sum_i H(Y_i) on the audit samples")
    pooled = pooled_audit_samples(4)
    natural = natural_audit_samples(4)
    out["bound_check"] = {
        "pooled": check_bound(pooled, "pooled (graded + natural)"),
        "natural": check_bound(natural, "natural only"),
    }
    for key, r in out["bound_check"].items():
        print(f"  {key:8s} n={r['n_samples']:5d}  violations={r['violations_budget_bound']}"
              f"  min slack={r['min_slack_bits']:.6f} (B>0: {r['min_slack_positive_budget_bits']:.6f}) bits"
              f"  refined violations={r['violations_refined_bound']}"
              f"  zero-budget n={r['n_zero_budget']} maxC={r['max_C_among_zero_budget']:.3e}")

    # ---- PART 3a : exact fibre diameter, binary, k = 2,3,4 ---------------
    print("[E8.2] exact min-entropy coupling by vertex enumeration (binary)")
    exact = []
    max_dev = 0.0
    for k in (2, 3, 4):
        for p in P_GRID:
            ps = [p] * k
            Hmin, xstar, cfgs = min_entropy_coupling_binary_exact(ps)
            B = k * h2(p)
            C_max = B - Hmin
            closed = (k - 1) * h2(p)
            dev = abs(C_max - closed)
            max_dev = max(max_dev, dev)
            exact.append({
                "k": k, "p": p, "per_position_H_bits": h2(p), "budget_bits": B,
                "H_min_exact_bits": Hmin, "C_min_bits": 0.0, "C_max_exact_bits": C_max,
                "C_max_closed_form_bits": closed, "abs_deviation_bits": dev,
                "diameter_bits": C_max,
            })
            print(f"    k={k} p={p:<5} B={B:7.4f}  Hmin={Hmin:7.4f}  "
                  f"C_max={C_max:7.4f}  closed=(k-1)h(p)={closed:7.4f}  dev={dev:.2e}")
    out["exact_binary_equal_marginals"] = {
        "rows": exact,
        "max_abs_deviation_from_closed_form_bits": float(max_dev),
        "closed_form": "C_max = (k-1) * h(p);  C_min = 0",
        "n_cases": len(exact),
    }
    print(f"  max |exact - closed form| over {len(exact)} cases = {max_dev:.3e} bits")

    # fair bits headline: k = 2,3,4 at p = 0.5
    fair = [r for r in exact if abs(r["p"] - 0.5) < 1e-12]
    out["fair_bits"] = {
        "claim": "for k fair bits the achievable range of C is exactly [0, k-1] bits",
        "verified_k": [r["k"] for r in fair],
        "C_max_bits": {str(r["k"]): r["C_max_exact_bits"] for r in fair},
        "predicted_bits": {str(r["k"]): float(r["k"] - 1) for r in fair},
        "max_abs_error_bits": float(max(abs(r["C_max_exact_bits"] - (r["k"] - 1))
                                        for r in fair)),
        "status": "CONFIRMED (and proved: H(P) >= max_i H(q_i) = 1, attained by "
                  "the perfectly-correlated coupling)",
    }

    # ---- independent random-coupling + interval-filling checks -----------
    print("[E8.3] random-coupling sanity check and interval-filling check (k=4)")
    ps4 = [0.5] * 4
    Hmin4, x4, cfgs4 = min_entropy_coupling_binary_exact(ps4)
    C_max4 = 4 * h2(0.5) - Hmin4
    out["argmin_coupling_k4_fair"] = {
        "H_min_bits": float(Hmin4), "C_max_bits": float(C_max4),
        "support": {"".join(str(b) for b in cfgs4[j]): float(x4[j])
                    for j in range(len(cfgs4)) if x4[j] > 1e-12},
        "comment": "the minimum-entropy coupling of 4 fair bits is the "
                   "perfectly-correlated one (all bits equal), H = 1 bit",
    }
    out["random_coupling_check_k4_fair"] = random_couplings_check(ps4, 400, rng, C_max4)
    out["interval_filling_k4_fair"] = interval_filling_check(ps4, x4, cfgs4)
    print("   ", json.dumps(out["random_coupling_check_k4_fair"]))
    print("   ", json.dumps(out["interval_filling_k4_fair"]))

    # ---- PART 3b : does the bound B - max_i H_i hold with UNEQUAL marginals?
    print("[E8.4] unequal marginals: is C_max = B - max_i H_i attained? (expect NO)")
    unequal = []
    n_attained = 0
    for trial in range(12):
        k = int(rng.integers(2, 5))
        ps = [float(rng.uniform(0.5, 0.98)) for _ in range(k)]
        Hmin, _, _ = min_entropy_coupling_binary_exact(ps)
        B = sum(h2(p) for p in ps)
        C_max = B - Hmin
        naive = B - max(h2(p) for p in ps)
        att = abs(C_max - naive) < 1e-8
        n_attained += int(att)
        unequal.append({"k": k, "ps": ps, "budget_bits": B,
                        "C_max_exact_bits": C_max,
                        "upper_bound_B_minus_maxH_bits": naive,
                        "gap_bits": naive - C_max, "attained": att})
    out["unequal_marginals"] = {
        "rows": unequal,
        "n_trials": len(unequal),
        "n_attained": n_attained,
        "max_gap_bits": float(max(r["gap_bits"] for r in unequal)),
        "conclusion": ("C_max <= B - max_i H(q_i) always holds, but is NOT attained "
                       "in general when the marginals differ; equality is guaranteed "
                       "only for an equal-marginal profile."),
    }
    print(f"    attained in {n_attained}/{len(unequal)} unequal-marginal trials; "
          f"max slack {out['unequal_marginals']['max_gap_bits']:.4f} bits")

    # ---- the headline table rows: k = 4, confidence grid -----------------
    table_rows = []
    for k in (2, 3, 4):
        p = 0.5
        r = next(x for x in exact if x["k"] == k and abs(x["p"] - p) < 1e-12)
        table_rows.append({"panel": "A", "k": k, "c": p, "H_bits": r["per_position_H_bits"],
                           "budget_bits": r["budget_bits"], "C_min": 0.0,
                           "C_max": r["C_max_exact_bits"], "diameter": r["C_max_exact_bits"],
                           "exact": True})
    for c in CONF_GRID:
        if any(abs(c - p) < 1e-12 for p in P_GRID):
            r = next(x for x in exact if x["k"] == K_TABLE and abs(x["p"] - c) < 1e-12)
            Cmax, ex = r["C_max_exact_bits"], True
            Hb = r["per_position_H_bits"]
            Bb = r["budget_bits"]
        else:
            Hb = h2(c)
            Bb = K_TABLE * Hb
            Cmax, ex = (K_TABLE - 1) * Hb, False
        table_rows.append({"panel": "B", "k": K_TABLE, "c": c, "H_bits": Hb,
                           "budget_bits": Bb, "C_min": 0.0, "C_max": Cmax,
                           "diameter": Cmax, "exact": ex})
    out["table_rows"] = table_rows

    # ---- PART 4 : the real models ---------------------------------------
    print("[E8.5] real-model operating points (k=4 blocks from E5)")
    # Prefer the v2 measurement (real corpus, non-overlapping blocks, both regimes)
    # and fall back to v1 only if v2 has not been run. Within v2 we use the DEPLOYED
    # `suffix` regime, since that is the operating point a decoder actually sees.
    v2 = {"Dream-Coder-7B (base)": "e5v2_dream_base.json",
          "Dream-Coder-7B (instruct)": "e5v2_dream_instruct.json",
          "DiffusionGemma-26B-A4B": "e5v2_diffusiongemma.json"}
    v1 = {"Dream-Coder-7B (base)": "e5_model_confirmation.json",
          "Dream-Coder-7B (instruct)": "e5_dream_instruct.json",
          "DiffusionGemma-26B-A4B": "e5_diffusiongemma.json"}
    model_files = {k: (v2[k] if os.path.exists(os.path.join(RESULTS, v2[k])) else v1[k])
                   for k in v2}
    model_files = {k: v for k, v in model_files.items()
                   if os.path.exists(os.path.join(RESULTS, v))}
    models = []
    for pretty, fname in model_files.items():
        d = json.load(open(os.path.join(RESULTS, fname)))
        recs = d["records"]
        if "config" in d:                      # v2 schema
            k = d["config"]["block_size"]
            recs = [r for r in recs if r.get("regime") == "suffix"] or recs
        else:                                  # v1 schema
            k = d["analysis"]["block_size"]
        conf = [r["confidence_min"] for r in recs]
        budg = [-r["neg_entropy_sum"] for r in recs]
        c_med = float(statistics.median(conf))
        B_med = float(statistics.median(budg))
        # always-valid certified bound (part (f)); equals the equal-marginal
        # closed form (k-1)*H per position, with H = B/k.
        C_max_cert = (1.0 - 1.0 / k) * B_med
        models.append({
            "model": pretty, "source": fname, "k": k, "n_blocks": len(recs),
            "median_confidence_min": c_med,
            "median_budget_bits": B_med,
            "min_budget_bits": float(min(budg)),
            "max_budget_bits": float(max(budg)),
            "q1_budget_bits": float(np.percentile(budg, 25)),
            "q3_budget_bits": float(np.percentile(budg, 75)),
            "C_max_q1_bits": float((1.0 - 1.0 / k) * np.percentile(budg, 25)),
            "C_max_q3_bits": float((1.0 - 1.0 / k) * np.percentile(budg, 75)),
            "n_blocks_diameter_ge_1bit": int(sum(1 for b in budg
                                                 if (1.0 - 1.0 / k) * b >= 1.0)),
            "n_blocks_diameter_gt_0": int(sum(1 for b in budg if b > 1e-12)),
            "median_blockwise_diameter_bits": float(
                statistics.median([(1.0 - 1.0 / k) * b for b in budg])),
            "certified_upper_bound_bits": B_med,          # C <= sum_i H_i
            "C_max_bits": C_max_cert,                     # C <= (1-1/k) sum_i H_i
            "C_min_bits": 0.0,
            "diameter_bits": C_max_cert,
            "binary_profile_C_max_bits": ((k - 1) * h2(c_med)) if c_med >= 0.5 else None,
        })
        print(f"    {pretty:28s} c_med={c_med:.4f}  B_med={B_med:8.4f} bits  "
              f"C in [0, {C_max_cert:.4f}] bits")
    out["models"] = models

    out["meta"] = {
        "units": "bits (log base 2)",
        "k_table": K_TABLE,
        "confidence_grid": CONF_GRID,
        "p_grid_exact": P_GRID,
        "notes": [
            "C_min = 0 on every fibre: the product measure realises it exactly.",
            "The achievable set is the full interval [0, C_max] (C continuous on a "
            "convex, hence connected, polytope).",
            "C_max = B - (min-entropy coupling); for an equal-marginal profile "
            "C_max = (k-1) H(q) exactly.",
            "For k fair bits C_max = k-1 bits exactly: PROVED and verified for k=2,3,4.",
            "For unequal marginals B - max_i H(q_i) is an upper bound on C_max but is "
            "generally NOT attained.",
        ],
    }

    path = os.path.join(RESULTS, "e8_nontightness.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {path}")

    write_table(out)
    return out



def write_table(out):
    rows = out["table_rows"]
    models = out["models"]
    nb_lo = min(m["n_blocks"] for m in models)
    nb_hi = max(m["n_blocks"] for m in models)
    tex = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Marginal profile & $k$ & $B=\sum_i H(Y_i)$ & $\mathrm{C}_{\min}$ & $\mathrm{C}_{\max}$ & diameter \\",
        r"\midrule",
        r"\multicolumn{6}{l}{\emph{(a) fair bits: exact vertex enumeration of the fibre}} \\",
    ]
    for r in rows:
        if r["panel"] != "A":
            continue
        tex.append(f"{r['k']} fair bits ($c=0.5$) & {r['k']} & {r['budget_bits']:.3f} & "
                   f"{r['C_min']:.3f} & {r['C_max']:.3f} & {r['diameter']:.3f} \\\\")
    tex += [
        r"\midrule",
        r"\multicolumn{6}{l}{\emph{(b) $k=4$, every position Bernoulli($c$) at confidence $c$}} \\",
    ]
    for r in rows:
        if r["panel"] != "B":
            continue
        tex.append(f"$c={r['c']:g}$ & {r['k']} & {r['budget_bits']:.3f} & "
                   f"{r['C_min']:.3f} & {r['C_max']:.3f} & {r['diameter']:.3f} \\\\")
    if os.environ.get("MODEL_ROWS") == "1":
        tex += [
            r"\midrule",
            rf"\multicolumn{{6}}{{l}}{{\emph{{(c) certified upper bound at observed "
            rf"operating points (median over {nb_lo} to {nb_hi} blocks)}}}} \\",
        ]
        for m in models:
            tex.append(f"{m['model']} ($c={m['median_confidence_min']:.2f}$) & {m['k']} & "
                       f"{m['median_budget_bits']:.3f} & {m['C_min_bits']:.3f} & "
                       f"{m['C_max_bits']:.3f} & {m['diameter_bits']:.3f} \\\\")
    frac = "; ".join(
        f"{m['model']}: {m['n_blocks_diameter_ge_1bit']}/{m['n_blocks']}"
        for m in models)
    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\textbf{What fixing the marginals leaves undetermined.} Since"
        r" $H(Y_S\mid c)\ge 0$, Eq.~\eqref{eq:tc} certifies $\mathrm{C}\le B:=\sum_{i\in S}H(Y_i\mid c)$,"
        r" which is exactly minus the \texttt{neg\_entropy\_sum} proxy. Marginal rules"
        r" bound $\mathrm{C}$, they do not locate it. The fibre of joints with these marginals is"
        r" convex, so $\mathrm{C}$ attains every value in $[\mathrm{C}_{\min},\mathrm{C}_{\max}]$: the product measure"
        r" gives $\mathrm{C}_{\min}=0$, and a minimum-entropy coupling gives"
        r" $\mathrm{C}_{\max}=B-\min_P H(P)$, which for equal marginals is $(k-1)H(Y_1)$, so $k$"
        r" fair bits span exactly $[0,k-1]$ bits (rows (a), by exact vertex enumeration)."
        r" Every entry is a statement about the whole fibre of joints compatible with a"
        r" marginal profile, so it bounds what the marginals fail to say and not what any"
        r" particular trained model does. What a real checkpoint actually does is measured"
        r" in \S\ref{sec:explains}.}",
        r"\label{tab:nontightness}",
        r"\end{table}",
    ]
    path = os.path.join(PAPER, "tables/nontightness.tex")
    with open(path, "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
