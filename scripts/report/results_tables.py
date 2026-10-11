#!/usr/bin/env python3
"""Result tables for the v3 paper, generated from results/e5v3_analysis.json."""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A = json.load(open(os.path.join(ROOT, "results", "e5v3_analysis.json")))
PAPER = os.path.join(ROOT, "paper")

LAB = {"dream_base": "Dream base", "dream_instruct": "Dream instr.",
       "llada_base": "LLaDA base", "llada_instruct": "LLaDA instr."}
PROXIES = ["confidence_min", "confidence_mean", "margin_min", "margin_mean",
           "neg_entropy_max", "neg_entropy_sum"]
PLAB = {"confidence_min": r"confidence (min)", "confidence_mean": r"confidence (mean)",
        "margin_min": r"margin (min)", "margin_mean": r"margin (mean)",
        "neg_entropy_max": r"neg-entropy (max)", "neg_entropy_sum": r"neg-entropy (sum)"}


def key_label(k):
    t, rg = k.split("/")
    return f"{LAB.get(t, t)} / {rg}"


# ---------------------------------------------------------------- Table: explains C
XC = json.load(open(os.path.join(ROOT, "results", "e5v3_corrections.json")))["k4"]
rows = []
for k in sorted(A["explains_C"]):
    e = A["explains_C"][k]
    x = XC[k]
    lad = e["predictor_ladder"]
    rows.append(" & ".join([
        key_label(k),
        f"${e['n']}$",
        f"${e['C_hat_mean_bits']:.2f}$",
        f"${e['C_sd_true_bits']:.2f}$",
        f"${100*lad['budget_only']['r2_true']:.0f}$",
        f"${100*e['best_marginal_predictor']['r2_true']:.0f}$",
        f"${e['unexplained_sd_bits']:.2f}$",
        f"${x['fraction']['resid_bits']:.2f}$",
        f"${x['null_uniform']['resid_bits']:.2f}$",
        f"${x['window_tight_median_bits']:.1f}$",
    ]) + r" \\")

tex = r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrrrr}
\toprule
 & & \multicolumn{2}{c}{measured $\mathrm{C}$} & \multicolumn{2}{c}{\% Var$[\mathrm{C}]$} & \multicolumn{3}{c}{residual s.d. (bits)} & window \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-9}
Model / regime & $n$ & mean & s.d. & $B$ & best & best & $fW$ & ref. & $W$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{\textbf{How well the marginals predict $\mathrm{C}$.} $\mathrm{C}$ is estimated
per candidate set with the order-averaged chain-rule estimator, in bits. ``s.d.'' is the
block-to-block standard deviation after subtracting the estimator's own variance.
``\% Var'' is out-of-fold $R^2$ with document-grouped folds, corrected for estimator
variance, for a predictor built from the summed marginal entropy $B$ alone and for the
best of $40$ $k$-NN and ridge predictors over five feature sets up to the full top-$8$
profile, which subsumes every \emph{single-step} rule we score but not the cross-step
stability rules of \cite{less,stability}. The residual columns are noise-corrected
residual standard deviations: \emph{best} for that predictor (nested selection moves it
by at most $0.035$ bits, App.~\ref{app:nested}), \emph{$fW$} for the one-parameter
predictor $f\cdot W$ with $f$ fitted out-of-fold, and \emph{ref.} for a coupling-agnostic
reference that knows the marginal entropies, hence $W$, with $\mathrm{C}$ uniform on $[0,W]$.
$W=B-\max_i H(Y_i)$ is the certified window of \eqref{eq:width}, at the median block.}
\label{tab:explains}
\end{table}
"""
open(os.path.join(PAPER, "tables/explains.tex"), "w").write(tex)
print("wrote paper/tables/explains.tex")

# ---------------------------------------------------------------- Table: fidelity
# transposed: with 8 settings the proxies-as-columns layout overflows the text block.
keys = sorted(A["fidelity"])
SHORT = {"confidence_min": "conf (min)", "confidence_mean": "conf (mean)",
         "margin_min": "marg (min)", "margin_mean": "marg (mean)",
         "neg_entropy_max": "negH (max)", "neg_entropy_sum": "negH (sum)"}
hdr = " & ".join(["Model / regime"] + [SHORT[p] for p in PROXIES] + [r"marg$-$conf"])
body = []
for k in keys:
    f = A["fidelity"][k]["C_hat_bits"]
    cells = []
    for p in PROXIES:
        r = f[p]
        cells.append(f"${r['spearman']:+.2f}$" + (r"$^{*}$" if r["excludes_zero"] else ""))
    d = f["_margin_minus_confidence"]
    cells.append(f"${d['diff']:+.3f}$" + (r"$^{*}$" if d["ci95"][0] > 0 or d["ci95"][1] < 0 else ""))
    body.append(" & ".join([key_label(k)] + cells) + r" \\")

tex2 = r"""\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{l""" + "r" * (len(PROXIES) + 1) + r"""}
\toprule
""" + hdr + r""" \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\caption{\textbf{Every marginal rule against measured $\mathrm{C}$, on real
checkpoints.} Spearman $\rho$ between each proxy's safety score and the meter's
$\widehat{\mathrm{C}}$, so a faithful proxy is negative. $^{*}$ marks intervals
excluding zero under a document-clustered bootstrap ($10^4$ resamples), and every one of
the
""" + str(len(PROXIES) * len(keys)) + r""" correlations does. Pooling across the block
(mean, sum) beats gating on its weakest position (min, max) in every setting. The last column is a \emph{paired} bootstrap of the difference in $\rho$ between
the min versions of top-2 margin and confidence on the same blocks, positive where
confidence is the better rule. The pooled comparisons are in Table~\ref{tab:paired}. This is
the correct test for ``does margin beat confidence'' and is strictly stronger than asking
whether two marginal intervals overlap. Our exact audit (Table~\ref{tab:proxy}) favours
margin, and on real checkpoints that reverses.}
\label{tab:fidelity}
\end{table}
"""
open(os.path.join(PAPER, "tables/fidelity.tex"), "w").write(tex2)
print("wrote paper/tables/fidelity.tex")

# ---------------------------------------------------------------- Table: controls
crows = []
for k in sorted(A["controls"]):
    c = A["controls"][k]
    crows.append(" & ".join([
        key_label(k),
        f"${c['median_confidence_min']:.3f}$",
        f"${c['median_budget_bits']:.1f}$",
        f"${c['median_C_hat_bits']:.2f}$",
        f"${c['chain_rule_deviation_median_bits']:.2f}$",
        f"${c['cond_sensitivity_tv_median']:.3f}$",
        f"${c['ref_order_n_distinct_median']:.1f}$",
        f"${c['ref_order_pairwise_disagree_median']:.2f}$",
        f"${100*c['invariant_ok_frac']:.1f}$",
    ]) + r" \\")

tex3 = r"""\begin{table}[!ht]
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lrrrrrrrr}
\toprule
Model / regime & med.\ conf. & $B$ & $\widehat{\mathrm{C}}$ & chain-rule dev. & cond.\ sens. & distinct $y$ & pairwise dis. & inv.\ ok \% \\
\midrule
""" + "\n".join(crows) + r"""
\bottomrule
\end{tabular}
\caption{\textbf{Controls.} \emph{chain-rule dev.}: spread over reveal orders of the
chain-rule log-probability at a fixed assignment, exactly zero for conditionals that
compose into one joint \cite{chainruledev}. \emph{cond.\ sens.}: mean total-variation
distance between $p(Y_j\given c)$ and $p(Y_j\given y_i,c)$. A model that ignores what
has been filled in sits near zero, would trivially score as ``consistent'', and would
trivially have $\mathrm{C}=0$, so this separates a genuinely composable joint from a
degenerate one. \emph{distinct $y$}: number of distinct greedy assignments obtained
under $8$ different reveal orders, out of $8$. When this exceeds one, the sequential
``reference'' decode against which parallel-commit error is usually measured is itself
order-dependent, and \emph{pairwise dis.} reports how far apart those references are.
\emph{inv.\ ok}: fraction of blocks satisfying the structural invariant that the first
sequentially decoded position, drawn from the same distribution as the parallel commit,
agrees with it. Anything below $100$ is nondeterminism in the forward pass.}
\label{tab:controls}
\end{table}
"""
open(os.path.join(PAPER, "tables/controls.tex"), "w").write(tex3)
print("wrote paper/tables/controls.tex")
