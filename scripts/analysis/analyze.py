#!/usr/bin/env python3
"""
Analysis for E5 v3.

Produces five things:

  A. gates          the reproducibility table: measured readout offset per checkpoint,
                    with the evidence, plus determinism and fluency.

  B. bug_impact     what the v2 off-by-one actually did, by re-measuring the same
                    quantities on the same corpus with the offset fixed.

  C. fidelity       all SIX marginal proxies against THREE outcomes (the meter's C_hat,
                    disagreement, and the log-prob deficit), per model and regime, with
                    document-clustered bootstrap CIs. This is the honest version of the
                    paper's one prescriptive claim, which v2 tested only on synthetic
                    sets and reported for confidence alone.

  D. explains_C     THE HEADLINE. How much of the variance in C does the marginal
                    profile actually explain on a real model? We fit the best
                    marginal-only predictor of C_hat we can (k-NN and ridge over the
                    per-position top-8 probability profile), score it out-of-fold with
                    document-grouped folds, and correct for the meter's own variance.
                    The residual is what NO marginal rule can ever recover. That is the
                    empirical analogue of the fibre width, and unlike the worst-case
                    bound it is a number a practitioner can use.

  E. controls       reference-order sensitivity (is "disagreement" well defined?),
                    conditional sensitivity (does the model react to what you fill in,
                    or is it degenerate?), and the chain-rule-deviation diagnostic.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from tcaudit.stats import spearman as _spearman_ref  # noqa: E402


def _ranks(a):
    """Average ranks with ties, fully vectorised.

    tcaudit.stats._avg_ranks resolves ties with a Python while-loop, which is fine for
    one call and ruinous inside a 10^4-resample bootstrap (~10^8 interpreter steps).
    Verified against the reference implementation in tests."""
    a = np.asarray(a, float)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    is_start = np.empty(n, bool)
    is_start[0] = True
    np.not_equal(sa[1:], sa[:-1], out=is_start[1:])
    starts = np.flatnonzero(is_start)
    ends = np.append(starts[1:], n)
    avg = (starts + ends - 1) / 2.0 + 1.0
    grp = np.cumsum(is_start) - 1
    out = np.empty(n, float)
    out[order] = avg[grp]
    return out


def spearman(x, y):
    rx, ry = _ranks(x), _ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")

RESULTS = os.path.join(ROOT, "results")
PROXIES = ["confidence_min", "confidence_mean", "margin_min", "margin_mean",
           "neg_entropy_max", "neg_entropy_sum"]
OUTCOMES = ["C_hat_bits", "disagreement", "logprob_deficit_bits",
            "ref_order_disagree_mean"]
TAGS = ["dream_base", "dream_instruct", "llada_base", "llada_instruct"]
VARIANT = os.environ.get("VARIANT", "")   # e.g. "_k8" for the block-size robustness
LABEL = {"dream_base": "Dream-Coder-7B (base)",
         "dream_instruct": "Dream-Coder-7B (instruct)",
         "llada_base": "LLaDA-8B (base)",
         "llada_instruct": "LLaDA-8B (instruct)"}


# ------------------------------------------------------------------ resampling

def cluster_boot_spearman(x, y, clusters, n_boot=10000, seed=0):
    """Document-clustered bootstrap CI for Spearman. Resample DOCUMENTS, not blocks."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    idx_by_c = [np.where(inv == i)[0] for i in range(len(uniq))]
    obs = spearman(x, y)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        ii = np.concatenate([idx_by_c[p] for p in pick])
        out[b] = spearman(x[ii], y[ii])
    lo, hi = np.nanquantile(out, [0.025, 0.975])
    return float(obs), float(lo), float(hi)


def cluster_boot_spearman_diff(x1, x2, y, clusters, n_boot=10000, seed=0):
    """Paired CI for rho(x1,y) - rho(x2,y). This is the correct test for 'proxy A beats
    proxy B'; comparing two marginal CIs for overlap is a weaker, different question."""
    rng = np.random.default_rng(seed)
    x1, x2, y = map(lambda v: np.asarray(v, float), (x1, x2, y))
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    idx_by_c = [np.where(inv == i)[0] for i in range(len(uniq))]
    obs = spearman(x1, y) - spearman(x2, y)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        ii = np.concatenate([idx_by_c[p] for p in pick])
        out[b] = spearman(x1[ii], y[ii]) - spearman(x2[ii], y[ii])
    lo, hi = np.nanquantile(out, [0.025, 0.975])
    return float(obs), float(lo), float(hi)


# ------------------------------------------------------------------ the headline

def profile_features(recs):
    """The statistic a marginal rule can see, made permutation-invariant over positions
    by sorting positions by confidence. Per position: top-8 probabilities + entropy."""
    F = []
    for r in recs:
        top = np.asarray(r["profile_top8"], float)          # [K, 8]
        ent = np.asarray(r["profile_entropy_bits"], float)  # [K]
        order = np.argsort(-top[:, 0])
        F.append(np.concatenate([top[order].ravel(), ent[order]]))
    return np.asarray(F)


def _knn_oof(X, y, groups, k=10, n_folds=5, seed=0):
    """k-NN regression, document-grouped out-of-fold. Standardised Euclidean distance."""
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    uniq = np.unique(groups)
    fold_of_group = {g: i % n_folds for i, g in enumerate(rng.permutation(uniq))}
    fold = np.array([fold_of_group[g] for g in groups])
    pred = np.empty(len(y))
    for f in range(n_folds):
        te, tr = fold == f, fold != f
        if te.sum() == 0 or tr.sum() < k:
            pred[te] = y[tr].mean() if tr.sum() else y.mean()
            continue
        d = ((Z[te][:, None, :] - Z[tr][None, :, :]) ** 2).sum(-1)
        nn = np.argsort(d, axis=1)[:, :k]
        pred[te] = y[tr][nn].mean(axis=1)
    return pred


def _ridge_oof(X, y, groups, lam=1.0, n_folds=5, seed=0):
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    uniq = np.unique(groups)
    fold_of_group = {g: i % n_folds for i, g in enumerate(rng.permutation(uniq))}
    fold = np.array([fold_of_group[g] for g in groups])
    pred = np.empty(len(y))
    P = np.eye(Z.shape[1])
    P[-1, -1] = 0.0                      # never penalise the intercept: shrinking it
    for f in range(n_folds):             # toward 0 biases predictions by mean(y)
        te, tr = fold == f, fold != f
        A = Z[tr].T @ Z[tr] + lam * P
        w = np.linalg.solve(A, Z[tr].T @ y[tr])
        pred[te] = Z[te] @ w
    return pred


def explains_C(recs):
    """How much of Var[C] is recoverable from the marginals, and how much is not.

    Var[C_hat] = Var[C] + E[SE^2]; likewise for the out-of-fold residual. Subtracting
    the meter's own variance turns both into statements about the true C.
    """
    y = np.array([r["C_hat_bits"] for r in recs], float)
    se = np.array([r["C_hat_se_bits"] for r in recs], float)
    groups = np.array([r["doc_id"] for r in recs])
    X = profile_features(recs)
    noise = float((se ** 2).mean())

    var_obs = float(y.var())
    var_true = max(var_obs - noise, 1e-12)

    def score(pred):
        mse = float(((y - pred) ** 2).mean())
        mse_true = max(mse - noise, 0.0)
        return {"mse_obs": mse, "mse_true": mse_true,
                "r2_naive": 1.0 - mse / var_obs,
                "r2_true": 1.0 - mse_true / var_true,
                "residual_sd_bits": float(np.sqrt(mse_true))}

    def best_of(feats):
        """Strongest marginal-only predictor we can fit, over a small sweep. Choosing
        the winner on out-of-fold error can only FLATTER the marginals, which makes the
        reported invisible fraction conservative -- the bias runs in the safe direction
        for the claim we make."""
        n = len(y)
        cands = {}
        for kk in (5, 10, 20, 40):
            if kk < n // 6:
                cands[f"knn{kk}"] = _knn_oof(feats, y, groups, k=kk)
        for lam in (0.1, 1.0, 10.0, 100.0):
            cands[f"ridge{lam:g}"] = _ridge_oof(feats, y, groups, lam=lam)
        scored = {nm: score(pr) for nm, pr in cands.items()}
        nm = max(scored, key=lambda k: scored[k]["r2_true"])
        return dict(scored[nm], model=nm)

    # A single R^2 conflates "the marginals are informative" with "the crude entropy
    # budget is informative". Decompose it into what a deployed scalar rule gets, what
    # the summed-entropy budget gets, and what the full profile gets.
    conf = np.array([r["confidence_min"] for r in recs], float)[:, None]
    Bcol = np.array([r["budget_bits"] for r in recs], float)[:, None]
    # per-position confidence + entropy: exactly the statistic deployed rules read,
    # one rung below the full top-8 profile. It turns out to be all that matters.
    K = len(recs[0]["profile_entropy_bits"])
    per_pos = np.hstack([X[:, :K * 1 * 0 + 0].reshape(len(X), 0),
                         np.asarray([np.sort(np.asarray(r["profile_top8"], float)[:, 0])[::-1]
                                     for r in recs]),
                         np.asarray([np.sort(np.asarray(r["profile_entropy_bits"], float))[::-1]
                                     for r in recs])])
    ladder = {
        "confidence_only": best_of(conf),
        "budget_only": best_of(Bcol),
        "per_position_conf_entropy": best_of(per_pos),
        "full_profile": best_of(X),
        "profile_plus_scalars": best_of(np.hstack([X, conf, Bcol])),
    }
    best_key = max(ladder, key=lambda k: ladder[k]["r2_true"])
    best = dict(ladder[best_key], feature_set=best_key)

    # A reviewer will object that B is literally a term in C = B - H(Y_S|c), so "the
    # marginals explain X% of Var[C]" is partly definitional. The objection is worth
    # answering rather than dodging. B is a DETERMINISTIC function of the marginals, so
    # a marginal rule knows it exactly and the only thing left to predict is the joint
    # entropy. Predicting C and predicting H(Y_S|c) are therefore the same problem with
    # the SAME residual; only the R^2 denominator changes. We quote the residual in
    # BITS, which is identical under both framings and is not definitional at all.
    Hj = np.array([r["budget_bits"] - r["C_hat_bits"] for r in recs], float)
    var_H_true = max(float(Hj.var()) - noise, 1e-12)
    for nm, v in ladder.items():
        v["r2_true_vs_joint_entropy"] = 1.0 - v["mse_true"] / var_H_true
    best["r2_true_vs_joint_entropy"] = 1.0 - best["mse_true"] / var_H_true
    joint_entropy_view = {
        "sd_H_joint_true_bits": float(np.sqrt(var_H_true)),
        "sd_C_true_bits": float(np.sqrt(var_true)),
        "residual_sd_bits": best["residual_sd_bits"],
        "r2_vs_C": best["r2_true"],
        "r2_vs_joint_entropy": best["r2_true_vs_joint_entropy"],
        "note": ("B is exactly known from the marginals, so predicting C and predicting "
                 "H(Y_S|c) share one residual. The bits figure is identical; only the "
                 "R^2 denominator differs. Quote the bits."),
    }

    cert = np.array([r["C_max_certified_bits"] for r in recs], float)
    return {
        "joint_entropy_view": joint_entropy_view,
        "n": len(recs),
        "meter_noise_var_bits2": noise,
        "meter_se_median_bits": float(np.median(se)),
        "C_hat_mean_bits": float(y.mean()),
        "C_hat_sd_observed_bits": float(np.sqrt(var_obs)),
        "C_sd_true_bits": float(np.sqrt(var_true)),
        "predictor_ladder": ladder,
        "best_marginal_predictor": best,
        "unexplained_sd_bits": best["residual_sd_bits"],
        "frac_variance_invisible_to_marginals": float(1.0 - best["r2_true"]),
        "frac_variance_invisible_to_confidence": float(
            1.0 - ladder["confidence_only"]["r2_true"]),
        "profile_gain_over_confidence": float(
            best["r2_true"] - ladder["confidence_only"]["r2_true"]),
        "certified_worstcase_width_median_bits": float(np.median(cert)),
        "implied_k_from_certificate": float(np.median(
            1.0 / np.maximum(1.0 - cert / np.maximum(
                np.array([r["budget_bits"] for r in recs], float), 1e-9), 1e-9))),
        "ratio_measured_residual_to_certified": float(
            best["residual_sd_bits"] / max(np.median(cert), 1e-9)),
    }


def matched_profile_pairs(recs, n_pairs_frac=0.1, seed=0):
    """Real-data analogue of the matched-marginal construction: for each block find the
    nearest block in marginal-profile space from a DIFFERENT document, and report how
    far apart their C values are. Compared against random pairs."""
    X = profile_features(recs)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    y = np.array([r["C_hat_bits"] for r in recs], float)
    se = np.array([r["C_hat_se_bits"] for r in recs], float)
    doc = np.array([r["doc_id"] for r in recs])
    n = len(y)
    if n < 20:
        return {}
    D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    D[doc[:, None] == doc[None, :]] = np.inf
    nn = np.argmin(D, axis=1)
    dist = np.sqrt(D[np.arange(n), nn])
    dC = np.abs(y - y[nn])
    pair_noise = se ** 2 + se[nn] ** 2

    keep = dist <= np.quantile(dist, n_pairs_frac)      # the closest-matched decile
    rms_obs = float(np.sqrt((dC[keep] ** 2).mean()))
    rms_true = float(np.sqrt(max((dC[keep] ** 2).mean() - pair_noise[keep].mean(), 0.0)))

    rng = np.random.default_rng(seed)
    j = rng.permutation(n)
    rnd = np.abs(y - y[j])
    return {
        "n_matched_pairs": int(keep.sum()),
        "match_distance_quantile": n_pairs_frac,
        "median_profile_distance_matched": float(np.median(dist[keep])),
        "rms_delta_C_matched_observed_bits": rms_obs,
        "rms_delta_C_matched_noise_corrected_bits": rms_true,
        "rms_delta_C_random_pairs_bits": float(np.sqrt((rnd ** 2).mean())),
        "median_abs_delta_C_matched_bits": float(np.median(dC[keep])),
    }


# ------------------------------------------------------------------ driver

def main():
    out = {"gates": {}, "bug_impact": {}, "fidelity": {}, "explains_C": {},
           "controls": {}}

    for tag in TAGS:
        p = os.path.join(RESULTS, f"e5v3_{tag}{VARIANT}.json")
        if not os.path.exists(p):
            print(f"  (missing {p}, skipping)")
            continue
        d = json.load(open(p))
        recs = d["records"]
        g = d["gates"]
        out["gates"][tag] = {
            "label": LABEL[tag], "repo": d["config"]["repo"],
            "measured_offset": g["offset"],
            "offset_evidence_bits": g["offset_evidence"][
                "mean_log2_prob_of_hidden_true_token_by_delta"],
            "offset_gap_bits": g["offset_evidence"]["min_gap_bits_over_runner_up"],
            "determinism": g["determinism"],
            "fluency_sample": g["fluency_sample"],
            "gates_passed": g["gates_passed"],
            "total_forward_passes": d["config"]["total_forward_passes"],
            "invariant_violations": {k: v["invariant_violations"]
                                     for k, v in d["run_stats"].items()},
        }

        for regime in sorted({r["regime"] for r in recs}):
            sub = [r for r in recs if r["regime"] == regime]
            key = f"{tag}/{regime}"
            doc = [r["doc_id"] for r in sub]

            # ---- C. fidelity: every proxy against every outcome ----
            fid = {}
            for oc in OUTCOMES:
                yv = [r[oc] for r in sub]
                row = {}
                for px in PROXIES:
                    o, lo, hi = cluster_boot_spearman([r[px] for r in sub], yv, doc)
                    row[px] = {"spearman": o, "ci95": [lo, hi],
                               "excludes_zero": (lo > 0) or (hi < 0)}
                # the paper's prescription, tested properly: margin vs confidence
                dobs, dlo, dhi = cluster_boot_spearman_diff(
                    [r["margin_min"] for r in sub],
                    [r["confidence_min"] for r in sub], yv, doc)
                row["_margin_minus_confidence"] = {
                    "diff": dobs, "ci95": [dlo, dhi],
                    "margin_significantly_better": dhi < 0}
                fid[oc] = row
            out["fidelity"][key] = fid

            # ---- D. the headline ----
            out["explains_C"][key] = explains_C(sub)
            out["explains_C"][key]["matched_pairs"] = matched_profile_pairs(sub)
            # robustness: drop blocks whose C estimate is significantly negative, i.e.
            # where the conditionals demonstrably fail to compose and C is not
            # well-defined. The headline should not hinge on those.
            ok = [r for r in sub
                  if r["C_hat_bits"] >= -2 * r["C_hat_se_bits"]]
            if 20 < len(ok) < len(sub):
                e_ok = explains_C(ok)
                out["explains_C"][key]["excluding_negative_C"] = {
                    "n": e_ok["n"],
                    "r2_true": e_ok["best_marginal_predictor"]["r2_true"],
                    "unexplained_sd_bits": e_ok["unexplained_sd_bits"],
                }

            # ---- E. controls ----
            arr = lambda k: np.array([r[k] for r in sub], float)  # noqa: E731
            out["controls"][key] = {
                "n_blocks": len(sub),
                "n_documents": len(set(doc)),
                "median_confidence_min": float(np.median(arr("confidence_min"))),
                "median_budget_bits": float(np.median(arr("budget_bits"))),
                "median_C_hat_bits": float(np.median(arr("C_hat_bits"))),
                "median_C_hat_over_budget": float(np.median(
                    arr("C_hat_bits") / np.maximum(arr("budget_bits"), 1e-9))),
                "mean_disagreement": float(arr("disagreement").mean()),
                "frac_zero_disagreement": float((arr("disagreement") == 0).mean()),
                "chain_rule_deviation_median_bits": float(
                    np.median(arr("order_logprob_std_bits"))),
                "cond_sensitivity_tv_median": float(np.median(arr("cond_sensitivity_tv"))),
                "ref_order_pairwise_disagree_median": float(
                    np.median(arr("ref_order_pairwise_disagree"))),
                "ref_order_n_distinct_median": float(
                    np.median(arr("ref_order_n_distinct_assignments"))),
                "ref_order_disagree_spread_median": float(np.median(
                    arr("ref_order_disagree_max") - arr("ref_order_disagree_min"))),
                "invariant_ok_frac": float(np.mean([r["invariant_ok"] for r in sub])),
                # C >= 0 holds for any genuine joint. Negative estimates therefore flag
                # blocks where the model's any-order conditionals do not compose: the
                # chain-rule entropy exceeds the summed marginal entropy. Report the
                # rate and the magnitude rather than clipping, which would hide it.
                "frac_C_hat_below_minus_2se": float(np.mean(
                    arr("C_hat_bits") < -2 * arr("C_hat_se_bits"))),
                "median_C_hat_when_negative": (float(np.median(
                    arr("C_hat_bits")[arr("C_hat_bits") < -2 * arr("C_hat_se_bits")]))
                    if (arr("C_hat_bits") < -2 * arr("C_hat_se_bits")).any() else 0.0),
            }

        # ---- B(i). the CLEAN comparison: identical blocks, identical code path,
        # identical seed, only the readout offset changed.
        abp = os.path.join(RESULTS, f"e5v3_{tag}_ablate0.json")
        if os.path.exists(abp):
            ab = json.load(open(abp))["records"]
            abl = {}
            for regime in sorted({r["regime"] for r in recs}):
                a = [r for r in ab if r["regime"] == regime]
                b = [r for r in recs if r["regime"] == regime]
                if not a or not b:
                    continue
                da, db = [r["doc_id"] for r in a], [r["doc_id"] for r in b]
                arr = lambda rs, k: np.array([r[k] for r in rs], float)  # noqa: E731
                abl[regime] = {
                    "n_wrong": len(a), "n_right": len(b),
                    "chain_rule_deviation_bits": {
                        "wrong_offset": float(np.median(arr(a, "order_logprob_std_bits"))),
                        "right_offset": float(np.median(arr(b, "order_logprob_std_bits")))},
                    "median_confidence": {
                        "wrong_offset": float(np.median(arr(a, "confidence_min"))),
                        "right_offset": float(np.median(arr(b, "confidence_min")))},
                    "median_C_hat_bits": {
                        "wrong_offset": float(np.median(arr(a, "C_hat_bits"))),
                        "right_offset": float(np.median(arr(b, "C_hat_bits")))},
                    "mean_disagreement": {
                        "wrong_offset": float(arr(a, "disagreement").mean()),
                        "right_offset": float(arr(b, "disagreement").mean())},
                    "rho_conf_disagree": {
                        "wrong_offset": list(cluster_boot_spearman(
                            arr(a, "confidence_min"), arr(a, "disagreement"), da, n_boot=2000)),
                        "right_offset": list(cluster_boot_spearman(
                            arr(b, "confidence_min"), arr(b, "disagreement"), db, n_boot=2000))},
                    "rho_conf_C_hat": {
                        "wrong_offset": list(cluster_boot_spearman(
                            arr(a, "confidence_min"), arr(a, "C_hat_bits"), da, n_boot=2000)),
                        "right_offset": list(cluster_boot_spearman(
                            arr(b, "confidence_min"), arr(b, "C_hat_bits"), db, n_boot=2000))},
                }
            out.setdefault("offset_ablation", {})[tag] = abl

        # ---- B(ii). corroboration against the earlier (v2) run ----
        v2p = os.path.join(RESULTS, f"e5v2_{tag}.json")
        if os.path.exists(v2p):
            v2 = json.load(open(v2p))["records"]
            imp = {}
            for regime in sorted({r["regime"] for r in recs}):
                a = [r for r in v2 if r["regime"] == regime]
                b = [r for r in recs if r["regime"] == regime]
                if not a or not b:
                    continue
                da = [r["doc_id"] for r in a]
                db = [r["doc_id"] for r in b]
                imp[regime] = {
                    "chain_rule_deviation_bits": {
                        "v2_buggy": float(np.median([r["order_logprob_std_bits"] for r in a])),
                        "v3_fixed": float(np.median([r["order_logprob_std_bits"] for r in b])),
                    },
                    "median_confidence": {
                        "v2_buggy": float(np.median([r["confidence_min"] for r in a])),
                        "v3_fixed": float(np.median([r["confidence_min"] for r in b])),
                    },
                    "mean_disagreement": {
                        "v2_buggy": float(np.mean([r["disagreement"] for r in a])),
                        "v3_fixed": float(np.mean([r["disagreement"] for r in b])),
                    },
                    "rho_conf_disagree": {
                        "v2_buggy": list(cluster_boot_spearman(
                            [r["confidence_min"] for r in a],
                            [r["disagreement"] for r in a], da, n_boot=2000)),
                        "v3_fixed": list(cluster_boot_spearman(
                            [r["confidence_min"] for r in b],
                            [r["disagreement"] for r in b], db, n_boot=2000)),
                    },
                }
            out["bug_impact"][tag] = imp

    path = os.path.join(RESULTS, f"e5v3_analysis{VARIANT}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")

    # ---------------- human-readable summary ----------------
    print("\n" + "=" * 78 + "\nA. GATES\n" + "=" * 78)
    for t, g in out["gates"].items():
        ev = g["offset_evidence_bits"].get("k=4", {})
        print(f"{LABEL[t]:28s} offset={g['measured_offset']:+d}  "
              f"gap={g['offset_gap_bits']:.1f} bits  "
              f"deltas(k=4)=" + " ".join(f"{k}:{v:+.1f}" for k, v in sorted(ev.items())))
        print(f"{'':28s} argmax flips={g['determinism']['repeat_call_argmax_flips']}, "
              f"invariant violations={g['invariant_violations']}")

    if out.get("offset_ablation"):
        print("\n" + "=" * 78 +
              "\nB(i). OFFSET ABLATION (identical blocks/seed/code; only the readout differs)"
              "\n" + "=" * 78)
        for t, imp in out["offset_ablation"].items():
            for rg, v in imp.items():
                c = v["chain_rule_deviation_bits"]
                rd, rc = v["rho_conf_disagree"], v["rho_conf_C_hat"]
                print(f"{LABEL[t]:26s} {rg:7s} chain-rule dev "
                      f"{c['wrong_offset']:6.2f} -> {c['right_offset']:5.2f} bits | "
                      f"rho(conf,disagree) {rd['wrong_offset'][0]:+.3f} -> "
                      f"{rd['right_offset'][0]:+.3f} | rho(conf,C) "
                      f"{rc['wrong_offset'][0]:+.3f} -> {rc['right_offset'][0]:+.3f}")

    print("\n" + "=" * 78 + "\nB(ii). VERSUS THE EARLIER v2 RUN\n" + "=" * 78)
    for t, imp in out["bug_impact"].items():
        for rg, v in imp.items():
            c = v["chain_rule_deviation_bits"]
            r2, r3 = v["rho_conf_disagree"]["v2_buggy"], v["rho_conf_disagree"]["v3_fixed"]
            print(f"{LABEL[t]:28s} {rg:7s} chain-rule dev {c['v2_buggy']:5.2f} -> "
                  f"{c['v3_fixed']:5.2f} bits | rho(conf,disagree) "
                  f"{r2[0]:+.3f} -> {r3[0]:+.3f}")

    print("\n" + "=" * 78 + "\nD. HOW MUCH OF C DO THE MARGINALS EXPLAIN?\n" + "=" * 78)
    for k, e in out["explains_C"].items():
        mp = e.get("matched_pairs", {})
        print(f"{k:26s} n={e['n']:4d}  C={e['C_hat_mean_bits']:5.2f}+-"
              f"{e['C_sd_true_bits']:4.2f} bits  "
              f"marginals explain {100*(1-e['frac_variance_invisible_to_marginals']):5.1f}%"
              f"  residual={e['unexplained_sd_bits']:5.2f} bits"
              f"  (worst-case bound {e['certified_worstcase_width_median_bits']:5.1f})")
        if mp:
            print(f"{'':26s} matched-profile pairs: |dC| rms "
                  f"{mp['rms_delta_C_matched_noise_corrected_bits']:.2f} bits "
                  f"(random pairs {mp['rms_delta_C_random_pairs_bits']:.2f})")

    print("\n" + "=" * 78 + "\nE. CONTROLS\n" + "=" * 78)
    for k, c in out["controls"].items():
        print(f"{k:26s} chain-rule dev {c['chain_rule_deviation_median_bits']:5.2f} b | "
              f"cond-sens TV {c['cond_sensitivity_tv_median']:.3f} | "
              f"ref-order pairwise disagree {c['ref_order_pairwise_disagree_median']:.3f} | "
              f"distinct assignments {c['ref_order_n_distinct_median']:.1f}/8")


if __name__ == "__main__":
    main()
