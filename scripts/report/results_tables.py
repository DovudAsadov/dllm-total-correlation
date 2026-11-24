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
 & & \multicolumn{2}{c}{measured $\widehat{\mathrm{C}}$} & \multicolumn{2}{c}{\% Var explained} & \multicolumn{3}{c}{residual (bits)} & bound \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-9}
Model / regime & $n$ & mean & s.d. & $B$ & best & best & $f\cdot W$ & reference & $W$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{\textbf{How well the marginals predict $\widehat{\mathrm{C}}$.}
\emph{measured $\widehat{\mathrm{C}}$}: mean and noise-corrected s.d.\ in bits.
\emph{\% Var explained}: out-of-fold $R^2$ of $B$ alone and of the best of $40$
predictors (\S\ref{sec:explains}). \emph{residual}: noise-corrected out-of-fold error of
that predictor, of the one-parameter $f\cdot W$, and of a reference that knows the
marginal entropies, with $\mathrm{C}$ uniform on $[0,W]$.
$W$: median bound.}
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
hdr = " & ".join(["Model / regime"] + [SHORT[p] for p in PROXIES])
body = []
for k in keys:
    f = A["fidelity"][k]["C_hat_bits"]
    cells = []
    for p in PROXIES:
        r = f[p]
        cells.append(f"${r['spearman']:+.2f}$" + (r"$^{*}$" if r["excludes_zero"] else ""))
    body.append(" & ".join([key_label(k)] + cells) + r" \\")

tex2 = r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l""" + "r" * len(PROXIES) + r"""}
\toprule
""" + hdr + r""" \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\caption{\textbf{Every marginal rule against measured $\mathrm{C}$.} Spearman $\rho$
between each rule's safety score and $\widehat{\mathrm{C}}$, so a faithful rule is
negative. $^{*}$: the document-clustered bootstrap interval excludes zero. Paired
comparisons between rules are in Table~\ref{tab:paired}.}
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
        f"${c['ref_order_pairwise_disagree_median']:.2f}$",
    ]) + r" \\")

tex3 = r"""\begin{table}[!ht]
\centering
\footnotesize
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrrrr}
\toprule
Model / regime & med.\ conf. & $B$ & $\widehat{\mathrm{C}}$ & chain-rule dev. & cond.\ sens. & pairwise dis. \\
\midrule
""" + "\n".join(crows) + r"""
\bottomrule
\end{tabular}
\caption{\textbf{Controls.} Medians over blocks.
\emph{chain-rule dev.}: as in \S\ref{sec:bugimpact}. \emph{cond.\ sens.}: total-variation
distance between $p(Y_j\given c)$ and $p(Y_j\given y_i,c)$. \emph{pairwise dis.}:
disagreement between the greedy assignments obtained under $8$ reveal orders.}
\label{tab:controls}
\end{table}
"""
open(os.path.join(PAPER, "tables/controls.tex"), "w").write(tex3)
print("wrote paper/tables/controls.tex")
