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
\caption{\textbf{Gate 2 and gate 3, in full.} Table~\ref{tab:gates} reports gate 1.
\emph{repeat}: maximum absolute difference in log-probability between two identical
forward calls, and the argmax flips that difference causes. \emph{batch}: the same
comparison between scoring a sequence alone and scoring it inside a batch. Exact
reproducibility on repeat calls is what the proxy audit needs, because a nondeterministic
forward flips argmaxes preferentially where confidence is low and so manufactures the
negative confidence-error correlation the audit is trying to detect. The gate requires
identical repeat calls and no alone-versus-batch argmax flip. It does not bound the
alone-versus-batch change in log-probability, which is nonzero, and that change reaches
the entropies the estimator uses (App.~\ref{app:probe}). Repeat calls compare $8$
sequences ($32$ positions), the alone-versus-batch column one sequence ($4$ positions).
Gate 3 is the first tokens greedy-decoded into an all-masked canvas using the calibrated
offset, truncated for width.}
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
\caption{\textbf{The $k{=}8$ replication, per setting.} \S\ref{sec:explains} quotes these
as ranges. Same corpus, same protocol, same estimator, blocks of eight positions instead
of four, on the two base checkpoints. The last two columns give the median certified
window $W=B-\max_i H(Y_i)$ and the residual of the coupling-agnostic reference of
Table~\ref{tab:explains} at $k{=}8$, which knows the marginal entropies but not the
coupling. The $k{=}4$ column repeats Table~\ref{tab:explains} for comparison on the same
checkpoint and regime.}
\label{tab:k8}
\begin{tabular}{lrrrrrrr}
\toprule
 & & \multicolumn{1}{c}{$\Chat$} & \% Var & \multicolumn{2}{c}{unexpl.\ (bits)} & window & ref. \\
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
\caption{\textbf{The estimator budget does not carry the headline.} Quadrupling the
number of reveal orders from $M{=}""" + str(lo) + r"""$ to $M{=}""" + str(hi) + r"""$ on
the \emph{same} blocks. The per-block standard error halves, as $1/\sqrt{M}$ predicts,
and the reported residual moves by at most $0.034$ bits. Note this is a different
experiment from Table~\ref{tab:meter}, which traces the convergence curve on a smaller
fixed set of $48$ blocks, so the standard errors are not directly comparable between the
two tables. \emph{rank agr.}: Spearman correlation between the two estimates of
$\mathrm{C}$ across blocks, i.e.\ whether the two budgets order the blocks the same way.
The two runs also differ in batch composition, $8$ blocks per batch at $M{=}64$ and $4$ at
$M{=}256$, so this tests the budget and the batch shape together.}
\label{tab:msens}
\begin{tabular}{lrrrrrrr}
\toprule
 & & \multicolumn{2}{c}{meter s.e.\ (bits)} & \multicolumn{2}{c}{residual (bits)} & & \\
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
\caption{\textbf{What selecting the predictor costs.} Every prediction behind
Table~\ref{tab:explains} is out-of-fold, but the winning feature set and model are chosen
over $""" + str(s["n_candidates"]) + r"""$ candidates using those same out-of-fold
scores, which biases the residual downward. That bias runs the wrong way for an upper
bound, so we price it rather than assume it away: \emph{nested} repeats the whole
procedure with $""" + str(s["n_folds_outer"]) + r"""$ outer document-grouped folds for
evaluation and $""" + str(s["n_folds_inner"]) + r"""$ inner folds for selection, so no
block's outcome ever influences the model that predicts it. The largest movement across
the eight settings is $""" + f"{s['max_delta_bits']:+.3f}" + r"""$ bits, and the range of
residuals is unchanged at two decimal places. \emph{selected}: the feature set and model
the inner folds chose most often.}
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
