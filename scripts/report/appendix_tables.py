#!/usr/bin/env python3
"""
Appendix tables that back claims the main text makes in prose only.

A reviewer found four places where the paper asserted something the reader could not
check: the full output of gates 2 and 3, the per-setting k=8 replication, the M=64 vs 256
ablation, and the cost of selecting the predictor on the same scores we report. All four
already existed in results/; none of them were tabulated. This writes them out.

Usage:  python3 scripts/report/appendix_tables.py
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
PAPER = os.path.join(ROOT, "paper")

LAB = {"dream_base": "Dream base", "dream_instruct": "Dream instr.",
       "llada_base": "LLaDA base", "llada_instruct": "LLaDA instr."}
TAGS = ["dream_base", "dream_instruct", "llada_base", "llada_instruct"]


def key_label(k):
    t, rg = k.split("/")
    return f"{LAB.get(t, t)} / {rg}"


def write(name, tex):
    p = os.path.join(PAPER, name)
    open(p, "w").write(tex)
    print(f"wrote {p}")


# ------------------------------------------------------- Gate 2 and Gate 3 outputs
def gate_records():
    """gates.json holds the first three checkpoints; LLaDA-instruct was gated in its own
    measurement run, so its gate block lives in results/e5v3_llada_instruct.json."""
    G = dict(json.load(open(os.path.join(RES, "gates.json")))["models"])
    for t in TAGS:
        if t in G:
            continue
        p = os.path.join(RES, f"e5v3_{t}.json")
        if os.path.exists(p):
            g = json.load(open(p)).get("gates")
            if g:
                G[t] = g
    return G


def gate_detail():
    G = gate_records()
    rows = []
    for t in TAGS:
        m = G.get(t)
        if not m or "error" in m:
            continue
        d = m["determinism"]
        sample = " ".join(m["fluency_sample"].split())[:38].replace("&", r"\&") + r"\,\ldots"
        rows.append(" & ".join([
            LAB[t],
            f"${d['repeat_call_max_abs_logprob_diff']:.3g}$",
            f"${d['repeat_call_argmax_flips']}$",
            f"${d['batch1_vs_batchN_max_abs_logprob_diff']:.3f}$",
            f"${d['batch_size_argmax_flips']}$",
            r"\texttt{\footnotesize " + sample.replace("_", r"\_").replace("#", r"\#") + r"}",
        ]) + r" \\")
    return r"""\begin{table}[!ht]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{\textbf{Gates 2 and 3, as run for the main measurement.} \emph{repeat call}:
largest log-probability difference between two identical forward calls over $8$
sequences, and the argmax flips it causes. \emph{alone vs batch}: the same for one
sequence scored alone and inside a batch. \emph{gate 3}: the start of a greedy decode
into an all-masked canvas.}
\label{tab:gatedetail}
\begin{tabular}{lrrrrl}
\toprule
 & \multicolumn{2}{c}{repeat call} & \multicolumn{2}{c}{alone vs batch} & gate 3 \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Model & max $\Delta$ & flips & max $\Delta$ & flips & greedy decode \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


# ------------------------------------------------------------- the k=8 replication
def k8():
    K = json.load(open(os.path.join(RES, "e5v3_analysis_k8.json")))["explains_C"]
    A = json.load(open(os.path.join(RES, "e5v3_analysis.json")))["explains_C"]
    XC8 = json.load(open(os.path.join(RES, "e5v3_corrections.json")))["k8"]
    rows = []
    for k in sorted(K):
        e, base = K[k], A.get(k)
        rows.append(" & ".join([
            key_label(k),
            f"${e['n']}$",
            f"${e['C_hat_mean_bits']:.2f}$",
            f"${100*e['best_marginal_predictor']['r2_true']:.0f}$",
            f"${e['unexplained_sd_bits']:.2f}$",
            (f"${base['unexplained_sd_bits']:.2f}$" if base else "--"),
            f"${XC8[k]['window_tight_median_bits']:.1f}$",
            f"${XC8[k]['null_uniform']['resid_bits']:.2f}$",
        ]) + r" \\")
    return r"""\begin{table}[!ht]
\centering
\small
\setlength{\tabcolsep}{5pt}
\caption{\textbf{The $k{=}8$ replication, per setting.} The protocol of
Table~\ref{tab:explains} with blocks of eight positions on the two base checkpoints, next
to the $k{=}4$ residual. $W$ and \emph{reference} as in Table~\ref{tab:explains}.}
\label{tab:k8}
\begin{tabular}{lrrrrrrr}
\toprule
 & & \multicolumn{1}{c}{$\Chat$} & \% Var & \multicolumn{2}{c}{residual (bits)} & bound & reference \\
\cmidrule(lr){5-6}
Model / regime & $n$ & mean & expl. & $k{=}8$ & $k{=}4$ & $W$ & (bits) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


# ------------------------------------------------------- M = 64 against M = 256
def msens():
    p = os.path.join(RES, "e10_m_sensitivity_llada_base.json")
    d = json.load(open(p))
    lo, hi = d["m_low"], d["m_high"]
    rows = []
    for rg in ("suffix", "infill"):
        r = d["by_regime"][rg]
        rows.append(" & ".join([
            f"LLaDA base / {rg}",
            f"${r['n_matched']}$",
            f"${r['meter_se_median'][0]:.3f}$", f"${r['meter_se_median'][1]:.3f}$",
            f"${r['residual_bits'][0]:.3f}$", f"${r['residual_bits'][1]:.3f}$",
            f"${abs(r['residual_bits'][1]-r['residual_bits'][0]):.3f}$",
            f"${r['spearman_C_lo_hi']:.3f}$",
        ]) + r" \\")
    return r"""\begin{table}[!ht]
\centering
\small
\setlength{\tabcolsep}{4.5pt}
\caption{\textbf{The estimator budget does not carry the headline.} The same blocks at
$M{=}""" + str(lo) + r"""$ and $M{=}""" + str(hi) + r"""$ reveal orders. \emph{rank agr.}:
Spearman correlation of the two estimates across blocks. The runs also differ in batch
size, $8$ and $4$ blocks, so this tests budget and batch shape together.}
\label{tab:msens}
\begin{tabular}{lrrrrrrr}
\toprule
 & & \multicolumn{2}{c}{estimator s.e.\ (bits)} & \multicolumn{2}{c}{residual (bits)} & & \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Model / regime & $n$ & $M{=}""" + str(lo) + r"""$ & $M{=}""" + str(hi) + r"""$ & $M{=}""" + str(lo) + r"""$ & $M{=}""" + str(hi) + r"""$ & $|\Delta|$ & rank agr. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


# ------------------------------------------------- selection bias, priced by nested CV
def nested():
    d = json.load(open(os.path.join(RES, "e11_nested_cv.json")))
    s, by = d["summary"], d["by_setting"]
    rows = []
    for k in sorted(by):
        v = by[k]
        rows.append(" & ".join([
            key_label(k),
            f"${v['n']}$",
            f"${v['reported_residual_sd_bits']:.3f}$",
            f"${v['residual_sd_bits']:.3f}$",
            f"${v['delta_bits']:+.3f}$",
            r"\texttt{\footnotesize " + v["selected_most_often"].replace("_", r"\_") + r"}",
        ]) + r" \\")
    return r"""\begin{table}[!ht]
\centering
\small
\setlength{\tabcolsep}{5pt}
\caption{\textbf{What selecting the predictor costs.} The residual of
Table~\ref{tab:explains} (\emph{reported}) against the same procedure with the selection
nested in $""" + str(s["n_folds_inner"]) + r"""$ inner folds within $""" + str(s["n_folds_outer"]) + r"""$ outer
document-grouped folds (\emph{nested}). \emph{selected}: the feature set and model chosen
most often.}
\label{tab:nested}
\begin{tabular}{lrrrrl}
\toprule
 & & \multicolumn{2}{c}{residual (bits)} & & \\
\cmidrule(lr){3-4}
Model / regime & $n$ & reported & nested & $\Delta$ & selected \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


if __name__ == "__main__":
    write("tables/gate_detail.tex", gate_detail())
    write("tables/k8.tex", k8())
    write("tables/m_sensitivity.tex", msens())
    write("tables/nested.tex", nested())
