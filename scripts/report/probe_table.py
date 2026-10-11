#!/usr/bin/env python3
"""Verification-probe appendix table and summary, from results/probe_<tag>.json and
results/probe_gate3.json (scripts/verify/probe.py, run on a DGX Spark GB10). These numbers
come from a different GPU than every other table and are never merged into them.

  paper/tables/probe.tex     per model, regime and domain, medians over probed blocks of
                              C_hat (B from the estimator's own first forward pass), the KL
                              version E_pi KL(q_pi || prod p_i), the order information
                              I(pi; Y_S), and the mixture cost KL(q_bar || prod p_i)
  results/probe_summary.json  every number the paper quotes from the probe

The probe oversamples blocks the main run flagged as outside [0, W]. Statements about
typical blocks use the random controls only (`controls`).
"""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
_s = importlib.util.spec_from_file_location("av3", os.path.join(ROOT, "scripts", "analysis", "analyze.py"))
av3 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(av3)
TAGS = ("dream_base", "dream_instruct", "llada_base")
LAB = {"dream_base": "Dream base", "dream_instruct": "Dream instr.", "llada_base": "LLaDA base"}
G3 = json.load(open(os.path.join(RES, "probe_gate3.json")))


def load(tag):
    d = json.load(open(os.path.join(RES, f"probe_{tag}.json")))
    main = json.load(open(os.path.join(RES, f"e5v3_{tag}.json")))["records"]
    for r in d["records"]:                       # same blocks as the A100 run, or stop
        m = [x for x in main if x["regime"] == r["regime"]][r["idx"]]
        assert m["doc_id"] == r["doc_id"], (tag, r["regime"], r["idx"], m["doc_id"], r["doc_id"])
        assert abs(m["budget_bits"] - r["B_main"]) < 1e-9 and abs(m["C_hat_bits"] - r["C_main"]) < 1e-9, \
            (tag, r["regime"], r["idx"], "B or C_hat differs from the A100 record")
    return d


def med(rs, f):
    return float(np.median([f(r) for r in rs]))


summary, rows = {}, []
for t in TAGS:
    d = load(t)
    R = d["records"]
    H8, H1, H0 = (np.array([r[k] for r in R]) for k in ("H8", "H1", "H0"))
    s = {"gate1_gap_bits": {dom: g.get("gap") for dom, g in d["gate1"].items()},
         "alone_vs_batch_max_dH_bits": float(np.abs(H8 - H1).max()),
         "hw_dB_max_bits": float(np.abs(H8.sum(1) - np.array([r["B_main"] for r in R])).max()),
         "hw_dB_mean_bits": {}, "settings": {}, "controls": {}}
    # impossible values on the A100 that are impossible again here, B from the estimator
    fl = np.array([r["flag_main"] for r in R])
    C0 = np.array([r["C0"] for r in R]); se = np.array([r["Hsum_se"] for r in R])
    W0 = H0.sum(1) - H0.max(1)
    again = (C0 < -2 * se) | (C0 > W0 + 2 * se)
    s["flagged_reappear"] = [int((again & fl).sum()), int(fl.sum())]
    # hairlines: near-deterministic blocks flagged below zero. KL >= 0 in expectation for
    # any conditionals, so KL below -2 s.e. there is noise the s.e. does not contain.
    hl = [r for r in R if r["flag_main"] and r["C_main"] < 0 and abs(r["C_main"]) < 0.2]
    neg = [r["KL_mean"] for r in hl if r["KL_mean"] < -2 * r["KL_se"]]
    s["hairline_KL_below_2se"] = [len(neg), len(hl), float(max(map(abs, neg))) if neg else 0.0]
    # rank agreement of C_hat with the KL cost, over every probed block outside DI prose
    rk = [r for r in R if not (t == "dream_instruct" and r["domain"] == "prose")]
    s["rho_C_KL"] = av3.spearman(np.array([r["C0"] for r in rk]), np.array([r["KL_mean"] for r in rk]))
    rc = [r for r in rk if not r["flag_main"]]
    s["rho_C_KL_controls"] = av3.spearman(np.array([r["C0"] for r in rc]), np.array([r["KL_mean"] for r in rc]))
    for rg in ("infill", "suffix"):
        rr = [r for r in R if r["regime"] == rg]
        s["hw_dB_mean_bits"][rg] = float(np.mean([abs(sum(r["H8"]) - r["B_main"]) for r in rr]))
        for dom in ("prose", "code"):
            rs = [r for r in rr if r["domain"] == dom]
            if not rs:
                continue
            cell = {"n": len(rs), "C0": med(rs, lambda r: r["C0"]), "KL": med(rs, lambda r: r["KL_mean"]),
                    "I": med(rs, lambda r: r["I_mean"]), "mix": med(rs, lambda r: r["mix_mean"])}
            s["settings"][f"{rg}/{dom}"] = cell
            cs = [r for r in rs if not r["flag_main"]]
            if len(cs) >= 3:
                s["controls"][f"{rg}/{dom}"] = {"n": len(cs), "gap": med(cs, lambda r: r["KL_mean"] - r["C0"]),
                                                 "I": med(cs, lambda r: r["I_mean"])}
            ctl = s["controls"].get(f"{rg}/{dom}")
            rows.append(" & ".join([f"{LAB[t]} / {rg} / {dom}", f"${cell['n']}$", f"${cell['C0']:+.2f}$",
                                    f"${cell['KL']:+.2f}$", f"${cell['I']:.2f}$", f"${cell['mix']:+.2f}$",
                                    (f"${ctl['n']}$" if ctl else "--"),
                                    (f"${ctl['gap']:+.2f}$" if ctl else "--"),
                                    (f"${ctl['I']:.2f}$" if ctl else "--")]) + r" \\")
    summary[t] = s

# Gate 3 at the wrong offset, token-level, from scripts/verify/probe.py decodes
runs_bad = [x["longest_run"] for t in TAGS for x in G3[t] if x["which"] == "wrong" and x["domain"] == "prose"]
runs_good = [x["longest_run"] for t in TAGS for x in G3[t] if x["which"] == "calibrated"]
summary["gate3"] = {"wrong_prose_run_min": min(runs_bad), "wrong_prose_run_max": max(runs_bad),
                    "calibrated_run_max": max(runs_good)}
# headline ranges the text quotes
cells = [(t, k) for t in ("llada_base", "dream_base") for k in ("infill/prose", "suffix/prose", "suffix/code")]
gaps = [summary[t]["controls"][k]["gap"] for t, k in cells if k in summary[t]["controls"]]
summary["typical_gap_bits"] = [min(gaps), max(gaps)]
summary["di_prose_gap_bits"] = [summary["dream_instruct"]["controls"][k]["gap"] for k in ("infill/prose", "suffix/prose")]
summary["max_control_I_bits"] = max(c["I"] for t in TAGS for c in summary[t]["controls"].values())
summary["flagged_reappear_total"] = [sum(summary[t]["flagged_reappear"][i] for t in TAGS) for i in (0, 1)]
hb = [summary[t]["hairline_KL_below_2se"] for t in TAGS]
summary["hairline_total"] = [sum(h[0] for h in hb), sum(h[1] for h in hb), max(h[2] for h in hb)]
json.dump(summary, open(os.path.join(RES, "probe_summary.json"), "w"), indent=1)

tex = r"""\begin{table}[!ht]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{\textbf{Both terms of \eqref{eq:orderinfo}, measured on a subset.} Medians over
probed blocks, in bits, on a second GPU (NVIDIA GB10, same checkpoints and blocks as the
main run). Per model and regime the probe takes $24$ blocks the main run flagged as
outside $[0,W]$, where there are that many, and $16$ random others.
$\widehat{\mathrm{C}}$ is the order-averaged estimate with $B$ from the estimator's own first
forward pass. \emph{KL} is $\mathbb{E}_\pi\mathrm{KL}(q_\pi\,\|\,\prod_i p_i)$, estimated
from the same samples by scoring them under the parallel marginals. $I$ is the order
information $I(\pi;Y_S)$, from all $4!$ orders evaluated at $32$ sampled assignments per
block. \emph{KL$-I$} is the cost relative to a random-order sampler. Where the
conditionals compose the four columns agree. The last three columns use the random blocks
only (cells with at least $3$), which the text quotes: their number, the median of
KL$-\widehat{\mathrm{C}}$, and the median $I$.}
\label{tab:probe}
\begin{tabular}{lrrrrrrrr}
\toprule
 & \multicolumn{5}{c}{all probed blocks} & \multicolumn{3}{c}{random blocks only} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-9}
Model / regime / domain & $n$ & $\widehat{\mathrm{C}}$ & KL & $I(\pi;Y_S)$ & KL$-I$ & $n$ & KL$-\widehat{\mathrm{C}}$ & $I$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(ROOT, "paper", "tables/probe.tex"), "w").write(tex)
print("wrote paper/tables/probe.tex and results/probe_summary.json")
for k in ("typical_gap_bits", "di_prose_gap_bits", "max_control_I_bits", "flagged_reappear_total", "hairline_total", "gate3"):
    print(f"  {k}: {summary[k]}")
for t in TAGS:
    print(f"  {t}: rho_C_KL={summary[t]['rho_C_KL']:.2f} gate1={ {k: round(v, 1) for k, v in summary[t]['gate1_gap_bits'].items()} } "
          f"alone_vs_batch={summary[t]['alone_vs_batch_max_dH_bits']:.2f} hw_mean={ {k: round(v, 2) for k, v in summary[t]['hw_dB_mean_bits'].items()} } hw_max={summary[t]['hw_dB_max_bits']:.2f}")
