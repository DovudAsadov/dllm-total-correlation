#!/usr/bin/env python3
"""Appendix tables for the camera-ready corrections, from results/e5v3_corrections.json.

  tables/domain.tex   residual by domain under one pooled predictor, with the
                       chain-rule deviation and both impossibility rates
  tables/paired.tex   paired tests for the pooled rules and the partial rank
                       correlation of each rule with C given the window W
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
X = json.load(open(os.path.join(ROOT, "results", "e5v3_corrections.json")))["k4"]
PAPER = os.path.join(ROOT, "paper")
LAB = {"dream_base": "Dream base", "dream_instruct": "Dream instr.",
       "llada_base": "LLaDA base", "llada_instruct": "LLaDA instr."}


def key_label(k):
    t, rg = k.split("/")
    return f"{LAB[t]} / {rg}"


def ci(v, fmt="+.3f"):
    """[obs, lo, hi] -> '$obs$' with a star when the interval excludes zero."""
    o, lo, hi = v
    star = r"$^{*}$" if lo > 0 or hi < 0 else ""
    return f"${o:{fmt}}$" + star


def write(name, tex):
    open(os.path.join(PAPER, name), "w").write(tex)
    print(f"wrote paper/{name}")


# ---------------------------------------------------------------- domain table
rows = []
for k in sorted(X):
    d = X[k]["domain"]
    im = X[k]["impossible"]
    rows.append(" & ".join([
        key_label(k),
        f"${d['prose']['resid_bits']:.2f}$ \\scriptsize$[{d['prose']['ci95'][0]:.2f},{d['prose']['ci95'][1]:.2f}]$",
        f"${d['code']['resid_bits']:.2f}$ \\scriptsize$[{d['code']['ci95'][0]:.2f},{d['code']['ci95'][1]:.2f}]$",
        f"${d['prose']['chain_rule_dev_median_bits']:.2f}$",
        f"${d['code']['chain_rule_dev_median_bits']:.2f}$",
        f"${100*im['frac_below_zero']:.1f}$",
        f"${100*im['frac_above_window']:.1f}$",
        f"${100*(im['frac_below_substantive']+im['frac_above_substantive']):.1f}$",
    ]) + r" \\")
write("tables/domain.tex", r"""\begin{table}[!ht]
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\caption{\textbf{Where the residual and the ill-defined blocks are.} \emph{residual}:
noise-corrected residual of one pooled ridge predictor (profile, confidence and $B$,
$\lambda{=}10$) fitted on both domains and scored separately on each, with a
document-clustered 95\% interval. \emph{chain-rule dev.}: median spread over reveal orders
of the chain-rule log-probability at a fixed assignment \cite{chainruledev}.
\emph{$\widehat{\mathrm{C}}<0$} and \emph{$\widehat{\mathrm{C}}>W$}: percentage of blocks
beyond two standard errors outside $[0,W]$, $W=B-\max_i H(Y_i)$, which no joint admits.
\emph{subst.}: the part of those beyond $0.2$ bits. The rest are hairlines on
near-deterministic blocks.}
\label{tab:domain}
\begin{tabular}{lrrrrrrr}
\toprule
 & \multicolumn{2}{c}{residual (bits)} & \multicolumn{2}{c}{chain-rule dev.} & \multicolumn{3}{c}{impossible (\% of blocks)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}
Model / regime & prose & code & prose & code & $\widehat{\mathrm{C}}<0$ & $\widehat{\mathrm{C}}>W$ & subst. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- paired table
rows = []
for k in sorted(X):
    p = X[k]["pooled"]
    q = X[k]["partial_given_W"]
    worst = max(p[n][2] for n in ("conf_mean_minus_conf_min", "marg_mean_minus_marg_min",
                                  "negH_sum_minus_negH_max"))
    rows.append(" & ".join([
        key_label(k),
        ci(p["marg_min_minus_conf_min"]),
        ci(p["marg_mean_minus_conf_mean"]),
        ci(p["negH_sum_minus_conf_mean"]),
        f"${worst:+.3f}$",
        f"${min(v[0] for v in q.values()):+.2f}$",
        f"${max(v[0] for v in q.values()):+.2f}$",
    ]) + r" \\")
write("tables/paired.tex", r"""\begin{table}[!ht]
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\caption{\textbf{Paired comparisons, and what the rules see beyond certainty.} The first
three columns are paired, document-clustered bootstraps ($2{,}000$ resamples) of
$\rho(\text{first rule},\widehat{\mathrm{C}})-\rho(\text{second rule},\widehat{\mathrm{C}})$
on the same blocks, positive where the second rule is the better one. $^{*}$ marks
intervals excluding zero. \emph{pooled vs min}: the largest upper end, over the three
statistics, of the interval for pooled minus min. Negative throughout means pooling
wins every one of the $24$ paired comparisons. The last two columns are the smallest and
largest partial Spearman correlation, over the six rules, of a rule with
$\widehat{\mathrm{C}}$ given the window $W$ (per-rule values and intervals are in
\texttt{results/e5v3\_corrections.json}). Without conditioning the pooled rules reach $\rho$
of $-0.83$ to $-0.94$ outside Dream-instruct (Table~\ref{tab:fidelity}), so most of what a
rule knows about $\mathrm{C}$ is the width of the window it certifies.}
\label{tab:paired}
\begin{tabular}{lrrrrrr}
\toprule
 & marg$-$conf & marg$-$conf & negH$-$conf & pooled vs & \multicolumn{2}{c}{partial $\rho$ given $W$} \\
\cmidrule(lr){6-7}
Model / regime & (min) & (mean) & (sum vs mean) & min (max hi) & \multicolumn{2}{c}{over the six rules} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
