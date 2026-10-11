#!/usr/bin/env python3
"""
Nested document-grouped CV, to price the winner's curse in the reported residual.

Why this exists. explains_C() picks the winning (feature set, model) over ~40 candidates
using the same out-of-fold scores it then reports. Every prediction is out-of-fold, but
the *selection* is not, so the reported residual is optimistically biased. That matters
because the paper calls the residual an upper bound on what a marginal rule leaves
behind, and an optimistic bias runs the wrong way for an upper bound. (It runs the right
way for the "fraction of variance the marginals recover" statement, which is a different
claim; the two got conflated in an earlier draft.)

This measures the bias instead of arguing about it. Outer folds evaluate, inner folds
select, no outcome is ever used to choose the model that predicts it. CPU only.

Usage:  python3 scripts/analysis/nested_cv.py
"""
from __future__ import annotations

import importlib.util
import json
import os
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
_s = importlib.util.spec_from_file_location(
    "av3", os.path.join(ROOT, "scripts", "analysis", "analyze.py"))
av3 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(av3)

TAGS = ["dream_base", "dream_instruct", "llada_base", "llada_instruct"]
N_FOLDS = 5
MODELS = [("knn", k) for k in (5, 10, 20, 40)] + \
         [("ridge", l) for l in (0.1, 1.0, 10.0, 100.0)]


def folds_of(groups, n_folds, seed):
    """Document-grouped folds: every block of a document lands in the same fold."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    fo = {g: i % n_folds for i, g in enumerate(rng.permutation(uniq))}
    return np.array([fo[g] for g in groups])


def _knn(Xtr, ytr, Xte, k):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    k = min(k, len(ytr))
    d = ((Zte[:, None, :] - Ztr[None, :, :]) ** 2).sum(-1)
    return ytr[np.argsort(d, axis=1)[:, :k]].mean(axis=1)


def _ridge(Xtr, ytr, Xte, lam):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
    Zte = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))])
    P = np.eye(Ztr.shape[1])
    P[-1, -1] = 0.0                      # never penalise the intercept
    w = np.linalg.solve(Ztr.T @ Ztr + lam * P, Ztr.T @ ytr)
    return Zte @ w


def predict(kind, hp, Xtr, ytr, Xte):
    return _knn(Xtr, ytr, Xte, hp) if kind == "knn" else _ridge(Xtr, ytr, Xte, hp)


def feature_sets(recs):
    """The same ladder explains_C() sweeps, so the comparison is like for like."""
    X = av3.profile_features(recs)
    conf = np.array([r["confidence_min"] for r in recs], float)[:, None]
    B = np.array([r["budget_bits"] for r in recs], float)[:, None]
    pp = np.hstack([
        np.asarray([np.sort(np.asarray(r["profile_top8"], float)[:, 0])[::-1] for r in recs]),
        np.asarray([np.sort(np.asarray(r["profile_entropy_bits"], float))[::-1] for r in recs])])
    return {"confidence_only": conf, "budget_only": B,
            "per_position_conf_entropy": pp, "full_profile": X,
            "profile_plus_scalars": np.hstack([X, conf, B])}


def nested(recs):
    y = np.array([r["C_hat_bits"] for r in recs], float)
    se = np.array([r["C_hat_se_bits"] for r in recs], float)
    g = np.array([r["doc_id"] for r in recs])
    FS = feature_sets(recs)
    noise = float((se ** 2).mean())
    outer = folds_of(g, N_FOLDS, seed=0)

    pred = np.empty(len(y))
    picks = []
    for f in range(N_FOLDS):
        te, tr = outer == f, outer != f
        inner = folds_of(g[tr], N_FOLDS, seed=1)
        best, best_mse = None, np.inf
        for fname, F in FS.items():
            Ftr = F[tr]
            for kind, hp in MODELS:
                ip = np.empty(int(tr.sum()))
                for j in range(N_FOLDS):
                    ite, itr = inner == j, inner != j
                    if ite.sum() == 0 or itr.sum() < 5:
                        ip[ite] = y[tr][itr].mean() if itr.sum() else y.mean()
                        continue
                    ip[ite] = predict(kind, hp, Ftr[itr], y[tr][itr], Ftr[ite])
                m = float(((y[tr] - ip) ** 2).mean())
                if m < best_mse:
                    best_mse, best = m, (fname, kind, hp)
        fname, kind, hp = best
        picks.append(f"{fname}/{kind}{hp:g}")
        pred[te] = predict(kind, hp, FS[fname][tr], y[tr], FS[fname][te])

    mse = float(((y - pred) ** 2).mean())
    var_true = max(float(y.var()) - noise, 1e-12)
    mse_true = max(mse - noise, 0.0)
    return {
        "n": len(recs),
        "residual_sd_bits": float(np.sqrt(mse_true)),
        "r2_true": float(1.0 - mse_true / var_true),
        "selected_most_often": Counter(picks).most_common(1)[0][0],
        "n_distinct_selections": len(set(picks)),
    }


def main() -> int:
    A = json.load(open(os.path.join(RES, "e5v3_analysis.json")))["explains_C"]
    out, deltas = {}, []
    print(f"{'setting':30s} {'reported':>9s} {'nested':>8s} {'delta':>7s}  selected")
    for t in TAGS:
        p = os.path.join(RES, f"e5v3_{t}.json")
        if not os.path.exists(p):
            continue
        R = json.load(open(p))["records"]
        for rg in ("infill", "suffix"):
            recs = [r for r in R if r["regime"] == rg]
            if not recs:
                continue
            k = f"{t}/{rg}"
            nv = nested(recs)
            rep = A[k]["unexplained_sd_bits"]
            nv["reported_residual_sd_bits"] = rep
            nv["delta_bits"] = nv["residual_sd_bits"] - rep
            nv["reported_r2_true"] = A[k]["best_marginal_predictor"]["r2_true"]
            out[k] = nv
            deltas.append(nv["delta_bits"])
            print(f"{k:30s} {rep:9.3f} {nv['residual_sd_bits']:8.3f} "
                  f"{nv['delta_bits']:+7.3f}  {nv['selected_most_often']}")

    summary = {
        "n_folds_outer": N_FOLDS, "n_folds_inner": N_FOLDS,
        "n_candidates": len(MODELS) * 5,
        "max_delta_bits": float(max(deltas)),
        "min_delta_bits": float(min(deltas)),
        "nested_residual_range_bits": [
            float(min(v["residual_sd_bits"] for v in out.values())),
            float(max(v["residual_sd_bits"] for v in out.values()))],
    }
    print(f"\nselection bias costs at most {summary['max_delta_bits']:+.3f} bits")
    json.dump({"summary": summary, "by_setting": out},
              open(os.path.join(RES, "e11_nested_cv.json"), "w"), indent=1)
    print(f"wrote {os.path.join(RES, 'e11_nested_cv.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
