#!/usr/bin/env python3
"""Figures for the v3 paper. Run after scripts/analysis/analyze.py."""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 160})

TAGS = ["dream_base", "dream_instruct", "llada_base", "llada_instruct"]
SHORT = {"dream_base": "Dream base", "dream_instruct": "Dream instr.",
         "llada_base": "LLaDA base", "llada_instruct": "LLaDA instr."}
COL = {"dream_base": "#1f77b4", "dream_instruct": "#7fb3d5",
       "llada_base": "#d95f02", "llada_instruct": "#f0a860"}
PROXIES = ["confidence_min", "confidence_mean", "margin_min", "margin_mean",
           "neg_entropy_max", "neg_entropy_sum"]
PLABEL = {"confidence_min": "conf. min", "confidence_mean": "conf. mean",
          "margin_min": "margin min", "margin_mean": "margin mean",
          "neg_entropy_max": "neg-ent. max", "neg_entropy_sum": "neg-ent. sum"}


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


A = load("e5v3_analysis.json")
G = load("gates.json")
if A is None:
    sys.exit("run scripts/analysis/analyze.py first")
present = [t for t in TAGS if t in A["gates"]]


# ---------------------------------------------------------------- Figure 1
def fig_gates_and_bug():
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 2.65))

    # (a) offset gate evidence
    deltas = ["-1", "0", "1"]
    w, x = 0.26, np.arange(3)
    for i, t in enumerate(present):
        ev = A["gates"][t]["offset_evidence_bits"]["k=4"]
        vals = [ev[d] for d in deltas]
        off = (i - (len(present) - 1) / 2) * w
        ax[0].bar(x + off, vals, w, label=SHORT[t], color=COL[t],
                  edgecolor="white", linewidth=0.4)
        best = int(np.argmax(vals))
        ax[0].scatter([x[best] + off], [vals[best] + 0.55], marker="*", s=95,
                      color=COL[t], edgecolor="k", linewidth=0.4, zorder=6,
                      label="_nolegend_")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([r"$\delta=-1$", r"$\delta=0$", r"$\delta=+1$"])
    ax[0].set_ylabel(r"$\log_2 p(\mathrm{hidden\ true\ token})$  (bits)")
    ax[0].set_title("(a) the readout offset is measured, not assumed")
    ax[0].legend(fontsize=7.5, loc="lower left", framealpha=0.95)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_ylim(top=2.2)
    ax[0].annotate("$\\star$ = convention selected by the gate", (0.02, 0.965),
                   xycoords="axes fraction", fontsize=7, va="top")

    # (b) the offset ablation -- the controlled comparison the paper reports.
    # (bug_impact, the version-to-version comparison, is confounded by other changes.)
    ab = A.get("offset_ablation", {})
    labels, wrong, right, rw, rr = [], [], [], [], []
    for t, imp in ab.items():
        for rgm, v in sorted(imp.items()):
            labels.append(f"{SHORT.get(t, t)}\n{rgm}")
            c = v["chain_rule_deviation_bits"]
            wrong.append(c["wrong_offset"]); right.append(c["right_offset"])
            d = v["rho_conf_disagree"]
            rw.append(d["wrong_offset"][0]); rr.append(d["right_offset"][0])
    if labels:
        xx = np.arange(len(labels))
        ax[1].bar(xx - 0.19, wrong, 0.38, label="misaligned readout", color="#c0392b")
        ax[1].bar(xx + 0.19, right, 0.38, label="validated readout", color="#27ae60")
        ax[1].set_xticks(xx); ax[1].set_xticklabels(labels, fontsize=7.5)
        ax[1].set_ylabel("chain-rule deviation (bits)")
        ax[1].set_title("(b) offset ablation: only the readout differs")
        ax[1].set_ylim(top=max(wrong) * 1.42)
        ax[1].legend(fontsize=7.5, loc="upper center", ncol=2, framealpha=0.95)
        for i, (a_, b_) in enumerate(zip(rw, rr)):
            flip = (a_ > 0) != (b_ > 0)
            ax[1].annotate(rf"$\rho$ {a_:+.2f}$\to${b_:+.2f}" + ("  (flip)" if flip else ""),
                           (i - 0.19, wrong[i]), textcoords="offset points",
                           xytext=(0, 4 + 11 * (i % 2)),   # stagger: labels collide otherwise
                           ha="center", fontsize=6.4,
                           fontweight="bold" if flip else "normal",
                           color="#c0392b" if flip else "#333333")
    else:
        ax[1].set_visible(False)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig1_gates.{e}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_gates")


# ---------------------------------------------------------------- Figure 2
def fig_explains():
    keys = [k for k in A["explains_C"] if k.endswith("/suffix")] or list(A["explains_C"])
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 2.75))

    # (a) where real blocks sit inside the tight certified window W = B - max_i H_i
    allf = []
    for k in keys:
        t = k.split("/")[0]
        raw = load(f"e5v3_{t}.json")
        if raw is None:
            continue
        rg = k.split("/")[1]
        sub = [r for r in raw["records"] if r["regime"] == rg]
        B = np.array([r["budget_bits"] for r in sub])
        W = B - np.array([r["profile_entropy_bits"] for r in sub]).max(1)
        C = np.array([r["C_hat_bits"] for r in sub])
        ax[0].scatter(W, C, s=5, alpha=0.28, color=COL[t], lw=0,
                      label=f"{SHORT[t]} {rg}")
        if t != "dream_instruct":
            allf.append(C[W > 1] / W[W > 1])
    hi = ax[0].get_xlim()[1]
    xs = np.linspace(0, hi, 50)
    ax[0].plot(xs, xs, "k--", lw=1.1, label=r"certified ceiling $\mathrm{C}=W$")
    fmed = float(np.median(np.concatenate(allf)))
    ax[0].plot(xs, fmed * xs, "k:", lw=1.1, label=rf"$\mathrm{{C}}={fmed:.2f}\,W$ (median)")
    ax[0].set_xlabel(r"certified window $W=B-\max_i H(Y_i)$ (bits)")
    ax[0].set_ylabel(r"measured $\widehat{\mathrm{C}}$ (bits)")
    ax[0].set_title("(a) real blocks inside the certified window")
    ax[0].legend(fontsize=7.0, loc="upper left")

    # (b) the headline: best marginal-only prediction of C against the truth
    import importlib.util
    _s = importlib.util.spec_from_file_location(
        "av3", os.path.join(ROOT, "scripts", "analysis", "analyze.py"))
    av3 = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(av3)

    for k in keys:
        t, rg = k.split("/")
        raw = load(f"e5v3_{t}.json")
        if raw is None:
            continue
        sub = [r for r in raw["records"] if r["regime"] == rg]
        y = np.array([r["C_hat_bits"] for r in sub])
        g = np.array([r["doc_id"] for r in sub])
        X = av3.profile_features(sub)
        Bc = np.array([r["budget_bits"] for r in sub])[:, None]
        cf = np.array([r["confidence_min"] for r in sub])[:, None]
        # mirror the ladder: use whichever feature set the analysis reported as best
        fs = A["explains_C"][k]["best_marginal_predictor"].get("feature_set", "full_profile")
        K = len(sub[0]["profile_entropy_bits"])
        pp = np.hstack([
            np.asarray([np.sort(np.asarray(r["profile_top8"], float)[:, 0])[::-1] for r in sub]),
            np.asarray([np.sort(np.asarray(r["profile_entropy_bits"], float))[::-1] for r in sub])])
        feats = {"confidence_only": cf, "budget_only": Bc, "full_profile": X,
                 "per_position_conf_entropy": pp,
                 "profile_plus_scalars": np.hstack([X, cf, Bc])}[fs]
        best = None
        for kk in (5, 10, 20, 40):
            if kk < len(y) // 6:
                p = av3._knn_oof(feats, y, g, k=kk)
                m = ((y - p) ** 2).mean()
                if best is None or m < best[0]:
                    best = (m, p)
        for lam in (0.1, 1.0, 10.0, 100.0):
            p = av3._ridge_oof(feats, y, g, lam=lam)
            m = ((y - p) ** 2).mean()
            if best is None or m < best[0]:
                best = (m, p)
        e = A["explains_C"][k]
        ax[1].scatter(best[1], y, s=6, alpha=0.3, color=COL[t], lw=0,
                      label=f"{SHORT[t]} {rg}: "
                            f"{100*(1-e['frac_variance_invisible_to_marginals']):.0f}%, "
                            f"$\\pm${e['unexplained_sd_bits']:.2f} b")
    lim = max(ax[1].get_xlim()[1], ax[1].get_ylim()[1])
    ax[1].plot([0, lim], [0, lim], "k--", lw=1.0)
    ax[1].set_xlabel(r"predicted $\mathrm{C}$ from marginals alone (bits)")
    ax[1].set_ylabel(r"measured $\widehat{\mathrm{C}}$ (bits)")
    ax[1].set_title("(b) out-of-fold, document-grouped")
    ax[1].legend(fontsize=6.6, loc="upper left")
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig2_explains.{e}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote fig2_explains")


# ---------------------------------------------------------------- Figure 3
def fig_fidelity():
    keys = sorted(A["fidelity"])
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(2.6 * len(keys), 3.2), sharex=True)
    if len(keys) == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        row = A["fidelity"][k]["C_hat_bits"]
        ys = np.arange(len(PROXIES))
        for j, px in enumerate(PROXIES):
            r = row[px]
            lo, hi = r["ci95"]
            ax.plot([lo, hi], [j, j], color="#4c72b0", lw=1.6)
            ax.plot([r["spearman"]], [j], "o", ms=4, color="#20415f")
        ax.axvline(0, color="k", lw=0.7)
        ax.set_yticks(ys)
        ax.set_yticklabels([PLABEL[p] for p in PROXIES] if k == keys[0] else [],
                           fontsize=7)
        ax.set_title(k.replace("/", "\n"), fontsize=7.5)
        ax.set_xlabel(r"Spearman $\rho$ vs $\widehat{\mathrm{C}}$")
        ax.invert_yaxis()
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig3_fidelity.{e}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote fig3_fidelity")


if __name__ == "__main__":
    fig_gates_and_bug()
    fig_explains()
    fig_fidelity()
