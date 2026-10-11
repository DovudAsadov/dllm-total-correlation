#!/usr/bin/env python3
"""
Camera-ready corrections (DiffuLM 2026, submission 6). CPU only, reads results/ only.

Each analysis answers a specific review point and is written to
results/e5v3_corrections.json. Every number the camera-ready quotes from that file is
asserted in scripts/report/check_claims.py.

  window        the tight certified ceiling W = B - max_i H_i. For any genuine
                joint H(Y_S) >= max_i H(Y_i), so C <= W <= (1 - 1/k) B.
  impossible    both impossibility rates, C < 0 and C > W, beyond 2 s.e., split into
                hairline and substantive (> MIN_BITS) cases.
  null          a coupling-agnostic reference: C ~ Uniform[0, W] per block. Its best
                predictor is W/2 with residual variance W^2/12. A reference, not a model.
  fraction      a one-parameter predictor C = f * W, f fitted out-of-fold. How much of
                the predictability is just the width of the certified window.
  partial       Spearman of each rule with C given W. What the rules see beyond
                certainty (TODO 1.4, replaced: the wrong-readout control mixed regimes).
  pooled        paired tests for the pooled rules the paper recommends, and for
                pooled minus min on all 24 comparisons.
  domain        residual by domain under one pooled predictor (the Dream-instruct
                diagnosis a reviewer asked for).
  k8            the null and the tight window at k=8.

Usage:  python3 scripts/analysis/corrections.py
"""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
_s = importlib.util.spec_from_file_location(
    "av3", os.path.join(ROOT, "scripts", "analysis", "analyze.py"))
av3 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(av3)

TAGS = ("dream_base", "dream_instruct", "llada_base", "llada_instruct")
REGIMES = ("infill", "suffix")
MIN_BITS = 0.2           # an impossibility beyond this is substantive, not a hairline
N_BOOT = 2000


# ------------------------------------------------------------------ pure functions

def tight_window(B, H):
    """W = B - max_i H_i, the tight certified ceiling on C. B [n], H [n, K]."""
    B = np.asarray(B, float)
    H = np.asarray(H, float)
    return B - H.max(axis=1)


def impossibility(C, se, W, min_bits=MIN_BITS):
    """Rates of estimates that no genuine joint admits: C < 0 or C > W, beyond 2 s.e."""
    C, se, W = (np.asarray(a, float) for a in (C, se, W))
    below = C < -2 * se
    above = C > W + 2 * se
    return {
        "n": int(len(C)),
        "frac_below_zero": float(below.mean()),
        "frac_above_window": float(above.mean()),
        "frac_either": float((below | above).mean()),
        "frac_below_substantive": float((below & (C < -min_bits)).mean()),
        "frac_above_substantive": float((above & (C - W > min_bits)).mean()),
        "median_excess_above_bits": float(np.median((C - W)[above])) if above.any() else 0.0,
        "median_C_below_bits": float(np.median(C[below])) if below.any() else 0.0,
    }


def coupling_agnostic_null(W):
    """C ~ Uniform[0, W]: oracle predictor W/2, residual variance W^2/12."""
    W = np.maximum(np.asarray(W, float), 0.0)
    m, v = W / 2.0, W ** 2 / 12.0
    var_c = m.var() + v.mean()
    return {"r2": float(m.var() / var_c) if var_c > 0 else 0.0,
            "resid_bits": float(np.sqrt(v.mean()))}


def fold_ids(groups, n_folds=5, seed=0):
    """Document-grouped folds, built exactly as analyze builds them."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    fog = {g: i % n_folds for i, g in enumerate(rng.permutation(uniq))}
    return np.array([fog[g] for g in groups])


def fraction_predictor_oof(C, W, groups, n_folds=5, seed=0, min_w=0.5):
    """Out-of-fold C_hat = f * W, f = median(C / W) over training blocks with W > min_w."""
    C, W = np.asarray(C, float), np.asarray(W, float)
    fold = fold_ids(np.asarray(groups), n_folds, seed)
    pred = np.empty_like(C)
    fs = []
    for f in range(n_folds):
        tr = (fold != f) & (W > min_w)
        fr = float(np.median(C[tr] / W[tr])) if tr.any() else 0.0
        fs.append(fr)
        pred[fold == f] = fr * W[fold == f]
    return pred, fs


def noise_corrected_resid(C, pred, se):
    C, pred, se = (np.asarray(a, float) for a in (C, pred, se))
    return float(np.sqrt(max(np.mean((C - pred) ** 2) - np.mean(se ** 2), 0.0)))


def partial_spearman(x, y, z):
    """Spearman of x and y after linearly removing rank(z) from both ranks."""
    rx, ry, rz = (av3._ranks(np.asarray(a, float)) for a in (x, y, z))
    Z = np.c_[np.ones_like(rz), rz]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    den = np.sqrt((ex ** 2).sum() * (ey ** 2).sum())
    return float((ex * ey).sum() / den) if den > 0 else 0.0


def cluster_boot(stat, groups, n_boot=N_BOOT, seed=0):
    """95% interval of stat(idx) under a document-clustered bootstrap."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    vals = [stat(np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq))]))
            for _ in range(n_boot)]
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


# ------------------------------------------------------------------ per setting

PAIRS = {    # (x1, x2): rho(x1, C) - rho(x2, C). Faithful rules are negative, so a
             # positive entry means x2 is the better rule (the paper's marg-conf column).
    "marg_min_minus_conf_min": ("margin_min", "confidence_min"),
    "marg_mean_minus_conf_mean": ("margin_mean", "confidence_mean"),
    "negH_sum_minus_conf_mean": ("neg_entropy_sum", "confidence_mean"),
    "conf_mean_minus_conf_min": ("confidence_mean", "confidence_min"),
    "marg_mean_minus_marg_min": ("margin_mean", "margin_min"),
    "negH_sum_minus_negH_max": ("neg_entropy_sum", "neg_entropy_max"),
}
PARTIAL_RULES = ("confidence_min", "confidence_mean", "margin_min", "margin_mean",
                 "neg_entropy_max", "neg_entropy_sum")


def arrays(recs):
    g = lambda k: np.array([r[k] for r in recs], float)  # noqa: E731
    B = g("budget_bits")
    H = np.array([r["profile_entropy_bits"] for r in recs], float)
    return {"C": g("C_hat_bits"), "se": g("C_hat_se_bits"), "B": B,
            "W": tight_window(B, H), "doc": np.array([r["doc_id"] for r in recs]),
            "dom": np.array([r["domain"] for r in recs])}


def domain_split(recs, lam=10.0):
    """One pooled ridge predictor (profile + confidence + B), residual per domain."""
    a = arrays(recs)
    F = np.hstack([av3.profile_features(recs),
                   np.array([[r["confidence_min"]] for r in recs], float),
                   a["B"][:, None]])
    pred = av3._ridge_oof(F, a["C"], a["doc"], lam=lam)
    crd = np.array([r["order_logprob_std_bits"] for r in recs], float)
    out = {"pooled_resid_bits": noise_corrected_resid(a["C"], pred, a["se"])}
    for dm in ("prose", "code"):
        m = a["dom"] == dm
        ii = np.where(m)[0]
        st = lambda j: noise_corrected_resid(a["C"][j], pred[j], a["se"][j])  # noqa: E731
        lo, hi = cluster_boot(lambda j: st(ii[j]), a["doc"][m], n_boot=1000)
        out[dm] = {"n": int(m.sum()), "resid_bits": st(ii), "ci95": [lo, hi],
                   "chain_rule_dev_median_bits": float(np.median(crd[m])),
                   "impossible": impossibility(a["C"][m], a["se"][m], a["W"][m])}
    return out


def setting(recs):
    a = arrays(recs)
    C, se, W, doc = a["C"], a["se"], a["W"], a["doc"]
    K = len(recs[0]["profile_entropy_bits"])
    pred, fs = fraction_predictor_oof(C, W, doc)
    big = W > 1.0
    out = {
        "n": len(recs),
        "window_tight_median_bits": float(np.median(W)),
        "window_loose_median_bits": float(np.median((1 - 1 / K) * a["B"])),
        "impossible": impossibility(C, se, W),
        "null_uniform": coupling_agnostic_null(W),
        "fraction": {"resid_bits": noise_corrected_resid(C, pred, se),
                     "f_by_fold": fs,
                     "f_median_W_gt_1": float(np.median(C[big] / W[big])),
                     "f_iqr_W_gt_1": [float(q) for q in np.quantile(C[big] / W[big], [.25, .75])]},
        "partial_given_W": {},
        "pooled": {},
        "domain": domain_split(recs),
    }
    for p in PARTIAL_RULES:
        x = np.array([r[p] for r in recs], float)
        o = partial_spearman(x, C, W)
        lo, hi = cluster_boot(lambda j: partial_spearman(x[j], C[j], W[j]), doc)
        out["partial_given_W"][p] = [o, lo, hi]
    for nm, (p1, p2) in PAIRS.items():
        o, lo, hi = av3.cluster_boot_spearman_diff(
            [r[p1] for r in recs], [r[p2] for r in recs], C, doc, n_boot=N_BOOT, seed=0)[:3]
        out["pooled"][nm] = [float(o), float(lo), float(hi)]
    return out


def main():
    out = {"min_bits": MIN_BITS, "n_boot": N_BOOT, "k4": {}, "k8": {}}
    A = json.load(open(os.path.join(RES, "e5v3_analysis.json")))["explains_C"]
    for t in TAGS:
        recs = json.load(open(os.path.join(RES, f"e5v3_{t}.json")))["records"]
        for rg in REGIMES:
            key = f"{t}/{rg}"
            out["k4"][key] = setting([r for r in recs if r["regime"] == rg])
            out["k4"][key]["best_resid_bits"] = A[key]["unexplained_sd_bits"]
            print(f"  {key:22s} done")
    A8 = json.load(open(os.path.join(RES, "e5v3_analysis_k8.json")))["explains_C"]
    for t in ("dream_base", "llada_base"):
        recs = json.load(open(os.path.join(RES, f"e5v3_{t}_k8.json")))["records"]
        for rg in REGIMES:
            a = arrays([r for r in recs if r["regime"] == rg])
            key = f"{t}/{rg}"
            out["k8"][key] = {"null_uniform": coupling_agnostic_null(a["W"]),
                              "window_tight_median_bits": float(np.median(a["W"])),
                              "best_resid_bits": A8[key]["unexplained_sd_bits"]}
    path = os.path.join(RES, "e5v3_corrections.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
