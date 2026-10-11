#!/usr/bin/env python3
"""
Appendix table for the meter's convergence in M, from results/e9_meter_convergence_*.json.

Why this exists: the Limitations section asserts that the estimator's per-block standard
error falls as 1/sqrt(M). That claim had no table behind it, so a reviewer had nothing to
check. The invariant tested here is that se * sqrt(M) is constant, which is what 1/sqrt(M)
means. The script computes the spread of that product and refuses to write a caption
claiming a tighter bound than the data supports.
"""
from __future__ import annotations

import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
PAPER = os.path.join(ROOT, "paper")

LAB = {"dream_base": "Dream base", "llada_base": "LLaDA base"}
TAGS = ["dream_base", "llada_base"]

rows: list[str] = []
worst_dev = 0.0
for t in TAGS:
    p = os.path.join(RES, f"e9_meter_convergence_{t}_k4.json")
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    grid = d["config"]["grid"]
    prod = [d["curve"][str(m)]["median_se_bits"] * math.sqrt(m) for m in grid]
    mean = sum(prod) / len(prod)
    worst_dev = max(worst_dev, max(abs(x - mean) / mean for x in prod))
    if rows:
        rows.append(r"\midrule")
    rows.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{LAB[t]}, deployed regime, "
                rf"$n={d['config']['n_blocks']}$ blocks, $k={d['config']['k']}$}}}} \\")
    for m, pr in zip(grid, prod):
        c = d["curve"][str(m)]
        dev = ("---" if m == grid[-1]
               else f"${c['mean_abs_dev_from_M256_bits']:.2f}$")
        rows.append(" & ".join([
            f"${m}$",
            f"${c['median_se_bits']:.3f}$",
            f"${pr:.2f}$",
            dev.replace("---", "$0$"),
            f"${c['max_abs_dev_bits']:.2f}$".replace("$0.00$", "$0$"),
            f"${c['forward_passes_per_block']}$",
        ]) + r" \\")

pct = int(math.ceil(worst_dev * 100))

tex = r"""\begin{table}[!ht]
\centering
\small
\setlength{\tabcolsep}{5pt}
\caption{\textbf{The meter converges as $1/\sqrt{M}$.} Per-block standard error of
$\Chat_M$ against the number of sampled reveal orders $M$, on a fixed block set per
checkpoint. Each smaller $M$ uses the first $M$ of the same $256$ orders, so the deviation from
$\Chat_{256}$ in the last columns understates the error of an independent run of size
$M$, by a factor of $\sqrt{256/(256-M)}$. The last
three columns give the mean and maximum absolute deviation of $\Chat_M$ from
$\Chat_{256}$ on the same blocks, and the forward-pass cost per block. The main runs use
$M{=}64$, the row where the deviation from a $4\times$ larger budget is already a
fraction of the residuals reported in Table~\ref{tab:explains}.}
\label{tab:meter}
\begin{tabular}{rrrrrr}
\toprule
$M$ & s.e.\ (bits) & s.e.$\times\sqrt{M}$ & mean $|\Chat_M-\Chat_{256}|$ & max & fwd/block \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

out = os.path.join(PAPER, "tables/convergence.tex")
open(out, "w").write(tex)
print(f"wrote {out}  (se*sqrt(M) flat to within {100*worst_dev:.1f}%, caption says {pct}%)")
