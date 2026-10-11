#!/usr/bin/env python3
"""
Emit the proxy-fidelity table (Table 1) as LaTeX and Markdown from the audit JSON.

Everything in this table is computed on the NATURAL sample set
(replace_random + shuffle + shuffle_no_identity), i.e. the observational
population of semantic tasks. It is emphatically NOT the pooled set: the pooled
set additionally contains the 21 graded blocks, which are the confound-REMOVED
control on which every marginal proxy is constant by construction. Reporting a
pooled number under a "(natural)" heading mixes the control into the population
and drags the safe-set AUROC down from its true value.

Columns:
  Proxy family | aggregations | Spearman(score, C) with 95% bootstrap CI
               | Kendall | AUROC_safe (natural) | matched-marginal gap

Row collapsing: several proxies induce *identical* rankings of the natural
samples and therefore identical rank statistics. Which ones do is verified
empirically here from the stored per-sample scores (not assumed), and one row is
emitted per distinct ranking.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, "results")
PAPER = os.path.join(ROOT, "paper")
os.makedirs(PAPER, exist_ok=True)

audit = json.load(open(os.path.join(RESULTS, "e3_proxy_audit.json")))
matched = json.load(open(os.path.join(RESULTS, "e2_matched_marginal.json")))["n4"]
paired_path = os.path.join(RESULTS, "e3_paired_tests.json")
PAIRED = json.load(open(paired_path)) if os.path.exists(paired_path) else None

if "natural" not in audit:
    raise SystemExit(
        "results/e3_proxy_audit.json has no 'natural' audit -- rerun "
        "scripts/exact/oracle.py (it now stores natural/graded/pooled)."
    )

NAT = audit["natural"]
IFF = audit["natural_iff"]

BASE = {
    "confidence_min": ("confidence", "min"),
    "confidence_mean": ("confidence", "mean"),
    "margin_min": ("margin", "min"),
    "margin_mean": ("margin", "mean"),
    "neg_entropy_max": ("neg-entropy", "max"),
    "neg_entropy_sum": ("neg-entropy", "sum"),
}


# ---------- formatting helpers ----------

def mnum(x: float, digits: int = 3) -> str:
    """A number in LaTeX math mode. The point is the wrapping: outside math mode
    a leading ASCII hyphen sets as a short text hyphen, not a minus sign."""
    return f"${x:.{digits}f}$"


def mest(x: float, lo: float, hi: float, digits: int = 3) -> str:
    """Point estimate and CI inside ONE math group, so every minus is a math
    minus and the bracket spacing is consistent."""
    return (f"${x:.{digits}f}\\ [{lo:.{digits}f},\\, {hi:.{digits}f}]$")


def mgap(x: float) -> str:
    r"""Matched-marginal gap. An exact zero must print as `$0$`: `$0e+00$`
    typesets the `+` as a binary operator and renders as "0e + 00"."""
    if x == 0.0:
        return "$0$"
    mant, exp = f"{x:.1e}".split("e")
    exp = int(exp)
    return f"${mant}{{\\times}}10^{{{exp}}}$"


def md_num(x: float, digits: int = 3) -> str:
    """Markdown mirror of mnum (also a real minus, U+2212, for readability)."""
    return f"{x:.{digits}f}".replace("-", "−")


def md_gap(x: float) -> str:
    if x == 0.0:
        return "0"
    mant, exp = f"{x:.1e}".split("e")
    return f"{mant}e{int(exp)}"


# ---------- empirically collapse proxies with identical rankings ----------

def avg_ranks(a: Sequence[float]) -> np.ndarray:
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    sa = a[order]
    i, n = 0, len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


scores: Dict[str, List[float]] = audit["natural_scatter"]["scores"]
reports = {r["name"]: r for r in NAT["proxies"]}
order = [r["name"] for r in NAT["proxies"]]

ranks = {name: avg_ranks(scores[name]) for name in order}
groups: List[List[str]] = []
for name in order:
    for g in groups:
        if np.array_equal(ranks[name], ranks[g[0]]):
            g.append(name)
            break
    else:
        groups.append([name])

# A distinct ranking must imply identical rank statistics; verify, don't assume.
for g in groups:
    for k in ("spearman_vs_C", "kendall_vs_C", "auroc_safe",
              "spearman_ci_lo", "spearman_ci_hi"):
        vals = [reports[n][k] for n in g]
        if max(vals) - min(vals) > 1e-9:
            raise SystemExit(
                f"identical rankings but differing {k} across {g}: {vals}")

N_DISTINCT = len(groups)
print(f"natural set: {NAT['proxies'][0]['n_samples']} samples, "
      f"{NAT['n_safe']} safe / {NAT['n_unsafe']} unsafe")
print(f"{len(order)} proxies collapse to {N_DISTINCT} distinct rankings:")
for g in groups:
    print("  " + ", ".join(g))


def label(group: Sequence[str]) -> tuple[str, str]:
    """(proxy-family label, aggregations label) for a group of proxy names."""
    bases: List[str] = []
    aggs: Dict[str, List[str]] = {}
    for name in group:
        b, a = BASE[name]
        if b not in aggs:
            bases.append(b)
            aggs[b] = []
        aggs[b].append(a)
    family = " / ".join(bases)
    agg = ", ".join(f"{b} ({'/'.join(aggs[b])})" for b in bases)
    return family, agg


rows = []
for g in groups:
    rep = reports[g[0]]
    gaps = [matched["proxy_gap"][n] for n in g]
    family, agg = label(g)
    rows.append({
        "family": family,
        "aggs": agg,
        "spearman": rep["spearman_vs_C"],
        "ci_lo": rep["spearman_ci_lo"],
        "ci_hi": rep["spearman_ci_hi"],
        "kendall": rep["kendall_vs_C"],
        "auroc": rep["auroc_safe"],
        "gap": max(gaps),
    })

# ---------- the identity behind AUROC = 1 ----------
if not IFF["iff_holds"]:
    raise SystemExit("the C=0 <-> zero-marginal-entropy identity does NOT hold; "
                     "the caption below would be false.")
assert all(abs(r["auroc"] - 1.0) < 1e-12 for r in rows), \
    "iff holds but AUROC is not exactly 1 -- check the proxies"

N_SAFE, N_UNSAFE = IFF["n_safe"], IFF["n_unsafe"]
N_TOT = IFF["n"]
H_SAFE_MAX = IFF["max_budget_on_safe"]
H_UNSAFE_MIN = IFF["min_budget_on_unsafe"]
DC = matched["C_constrained"] - matched["C_free"]

print(f"iff check: C=0 <=> sum_i H(Y_i)=0 on all {N_TOT} natural samples "
      f"({N_SAFE} safe / {N_UNSAFE} unsafe); max budget on safe = {H_SAFE_MAX:g} bits, "
      f"min budget on unsafe = {H_UNSAFE_MIN:.5f} bits")

PAIRED_TXT = ""
if PAIRED:
    def _d(a, b):
        d = PAIRED["paired_differences"][f"{a}__minus__{b}"]
        return (f"${-d['diff']:.3f}$ $[{-d['ci95'][1]:.3f},\\, {-d['ci95'][0]:.3f}]$")
    PAIRED_TXT = (" Proxy comparisons use a paired bootstrap of the "
                  r"\emph{difference} in Spearman on the same sets, which is the "
                  "correct test on paired data and is strictly more powerful than "
                  "asking whether two marginal CIs overlap: margin beats confidence "
                  "by " + _d("margin_min", "confidence_min") + " and beats "
                  "neg-entropy-sum by " + _d("margin_min", "neg_entropy_sum") +
                  ", both excluding zero.")

PAIRED_SHORT = ""
if PAIRED:
    d1 = PAIRED["paired_differences"]["margin_min__minus__confidence_min"]
    PAIRED_SHORT = (rf" margin beats confidence by ${-d1['diff']:.3f}$ "
                    rf"$[{-d1['ci95'][1]:.3f},\, {-d1['ci95'][0]:.3f}]$. "
                    r"\S\ref{sec:fidelity} finds this reverses on the four checkpoints.")

# ---------- Markdown ----------
md = ["| Proxy family | aggregations | Spearman$(\\cdot,\\mathrm{C})$ [95% CI] | Kendall "
      "| AUROC safe (natural) | matched gap |",
      "|---|---|---|---|---|---|"]
for r in rows:
    md.append(
        f"| {r['family']} | {r['aggs']} | {md_num(r['spearman'])} "
        f"[{md_num(r['ci_lo'])}, {md_num(r['ci_hi'])}] | {md_num(r['kendall'])} "
        f"| {r['auroc']:.4f} | {md_gap(r['gap'])} |"
    )
md.append("")
md.append(
    f"Natural set only ({N_TOT} candidate sets from replace_random / shuffle / "
    f"shuffle_no_identity; {N_SAFE} with C = 0, {N_UNSAFE} with C > 0). "
    f"{len(order)} proxy rules collapse to {N_DISTINCT} distinct rankings. "
    f"AUROC = 1.0000 is an identity, not a finding: on these tasks C = 0 iff the "
    f"summed marginal entropy is 0 ({N_SAFE}/{N_SAFE} safe sets at exactly "
    f"{H_SAFE_MAX:g} bits, all {N_UNSAFE} unsafe sets at >= {H_UNSAFE_MIN:.2f} bits, "
    f"no overlap)."
)
md_txt = "\n".join(md)
open(os.path.join(PAPER, "tables/oracle_audit.md"), "w").write(md_txt + "\n")

# ---------- LaTeX ----------
tex = [
    r"\begin{table}[!ht]",
    r"\centering",
    r"\small",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{lrrrr}",
    r"\toprule",
    r"Rule & Spearman$(\cdot,\mathrm{C})$ [95\% CI] & Kendall & "
    r"AUROC$_{\mathrm{safe}}$ & gap \\",
    r"\midrule",
]
for r in rows:
    tex.append(
        f"{r['aggs']} & {mest(r['spearman'], r['ci_lo'], r['ci_hi'])}"
        f" & {mnum(r['kendall'])} & "
        f"${r['auroc']:.4f}$ & {mgap(r['gap'])} \\\\"
    )
tex += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\caption{\textbf{The exact audit.} Six marginal rules scored against "
rf"analytic $\mathrm{{C}}$ on all ${N_TOT}$ candidate sets from \textsc{{replace\_random}} and "
rf"\textsc{{shuffle}}, of which ${N_SAFE}$ have $\mathrm{{C}}=0$. The rules collapse to "
rf"${N_DISTINCT}$ distinct rankings, so one row is shown per ranking. The perfect "
r"safe-set AUROC is an identity, not a finding: here $\mathrm{C}=0$ \emph{if and only if} "
rf"the summed marginal entropy is $0$ (all ${N_SAFE}$ at ${H_SAFE_MAX:g}$ bits, all "
rf"${N_UNSAFE}$ others at $\geq {H_UNSAFE_MIN:.2f}$), so any certainty detector scores "
r"$1.000$ while measuring no dependency. \emph{gap} is the matched-marginal score "
rf"difference, exactly $0$ while $\mathrm{{C}}$ differs by ${DC:.2f}$ bits. Comparisons use a "
r"paired bootstrap of the \emph{difference} in Spearman on the same sets:" + PAIRED_SHORT + r"}",
    r"\label{tab:proxy}",
    r"\end{table}",
]
open(os.path.join(PAPER, "tables/oracle_audit.tex"), "w").write("\n".join(tex) + "\n")

print()
print(md_txt)
print("\nwrote paper/tables/oracle_audit.{md,tex}")
