#!/usr/bin/env python3
"""
Cross-check the numbers asserted in the paper against the data they come from.

Motivation: a self-audit of the first draft found three numeric claims that did not
match the analysis -- an aggregation-gain range quoted as 0.10-0.21 when it was
0.066-0.235, a block count of 4,500 when it was 2,256, and a correlation range that held
only for a subset of checkpoints. Those are precisely the errors this project exists to
criticise, and eyeballing does not catch them. So: assert them in code, and re-run after
every analysis refresh.

Each check states the claim as it appears in the paper and recomputes it. Exit code is
nonzero if any check fails, so it can gate a rebuild.

Usage:  python3 scripts/report/check_claims.py
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
_s = importlib.util.spec_from_file_location(
    "av3", os.path.join(ROOT, "scripts", "analysis", "analyze.py"))
av3 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(av3)

A = json.load(open(os.path.join(RES, "e5v3_analysis.json")))
TEX = open(os.path.join(ROOT, "paper", "main.tex")).read()
TAGS = [t for t in ("dream_base", "dream_instruct", "llada_base", "llada_instruct")
        if os.path.exists(os.path.join(RES, f"e5v3_{t}.json"))]
RECS = {t: json.load(open(os.path.join(RES, f"e5v3_{t}.json")))["records"] for t in TAGS}

fails: list[str] = []
checks = 0


def claim(desc: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if ok:
        print(f"  ok   {desc}")
    else:
        print(f"  FAIL {desc}   {detail}")
        fails.append(f"{desc} {detail}")


def in_tex(needle: str) -> bool:
    return needle in TEX


_TEX_WS = re.sub(r"\s+", " ", TEX)


def in_tex_ws(needle: str) -> bool:
    """in_tex, but blind to line wrapping: the source wraps sentences anywhere."""
    return re.sub(r"\s+", " ", needle) in _TEX_WS


_MAIN_WS = re.sub(r"\s+", " ", TEX.split("\\appendix")[0])


def in_main_ws(needle: str) -> bool:
    """in_tex_ws restricted to the main text, for numbers the appendix also quotes."""
    return re.sub(r"\s+", " ", needle) in _MAIN_WS


def rng(vals):
    return float(min(vals)), float(max(vals))


# ---------------------------------------------------------------- the headline
resid = [e["unexplained_sd_bits"] for k, e in A["explains_C"].items()
         if not k.startswith("dream_instruct")]   # Dream-instruct is quoted separately
lo, hi = rng(resid)
claim(f"residual range {lo:.2f}-{hi:.2f} bits appears in the paper",
      in_tex(f"${lo:.2f}$ to ${hi:.2f}$ bits"), f"(computed {lo:.2f}-{hi:.2f})")
claim("Dream-instruct residuals (1.62 and 1.84) are quoted separately",
      in_tex_ws("($1.62$ and $1.84$ bits)"))

# The "N of eight settings agree closely" sentence. An earlier draft said seven, which
# double-counted: Dream-instruct contributes TWO settings to the exception, so 7+2=9>8.
_tight = [k for k, e in A["explains_C"].items() if 0.74 <= e["unexplained_sd_bits"] <= 1.18]
_loose = sorted(set(A["explains_C"]) - set(_tight))
_W = {5: "five", 6: "six", 7: "seven", 8: "eight"}
claim(f"{len(_tight)} of {len(A['explains_C'])} settings sit in the tight residual band, "
      f"and the paper says so",
      in_tex_ws(f"{_W[len(_tight)].capitalize()} of the eight settings"),
      f"(tight={len(_tight)}, exceptions={_loose})")
claim("the tight band and the exceptions together account for all eight settings",
      len(_tight) + len(_loose) == len(A["explains_C"]))
claim("the quoted upper residual (1.84) is the max over all settings",
      abs(max(e["unexplained_sd_bits"] for e in A["explains_C"].values()) - 1.84) < 0.005)

bound = [e["certified_worstcase_width_median_bits"] for e in A["explains_C"].values()]
blo, bhi = rng(bound)
pass

pass

r2h = [e["joint_entropy_view"]["r2_vs_joint_entropy"] for e in A["explains_C"].values()]
pass

# ---------------------------------------------------------------- block counts
total = sum(len(v) for v in RECS.values())
claim(f"total block count {total:,} appears in the paper",
      in_tex(f"${total:,}$".replace(",", "{,}")), f"(computed {total})")

per = sorted({len([r for r in v if r["regime"] == rg])
              for v in RECS.values() for rg in ("suffix", "infill")})
claim(f"per-setting block range {per[0]}-{per[-1]} appears",
      in_tex(f"${per[0]}$ to ${per[-1]}$ blocks"), f"(computed {per[0]}-{per[-1]})")

# ---------------------------------------------------------------- invariant
inv = [c["invariant_ok_frac"] for c in A["controls"].values()]
claim("structural invariant holds on 100% of blocks", all(abs(v - 1.0) < 1e-12 for v in inv),
      f"(min {min(inv):.4f})")

# ---------------------------------------------------------------- aggregation
gains = []
for k, f in A["fidelity"].items():
    g = f["C_hat_bits"]
    for a, b in (("confidence_min", "confidence_mean"),
                 ("margin_min", "margin_mean"),
                 ("neg_entropy_max", "neg_entropy_sum")):
        gains.append(abs(g[b]["spearman"]) - abs(g[a]["spearman"]))
glo, ghi = rng(gains)
claim("pooled aggregation beats worst-position in every comparison", all(x > 0 for x in gains),
      f"({sum(1 for x in gains if x <= 0)} exceptions of {len(gains)})")
claim(f"aggregation gain range {glo:.3f}-{ghi:.3f} appears",
      in_tex(f"${glo:.3f}$ to ${ghi:.3f}$"), f"(computed {glo:.3f}-{ghi:.3f})")
claim(f"number of aggregation comparisons ({len(gains)}) appears",
      in_tex(f"all ${len(gains)}$ comparisons"), f"(computed {len(gains)})")

# ---------------------------------------------------------------- margin vs confidence
diffs = {k: f["C_hat_bits"]["_margin_minus_confidence"] for k, f in A["fidelity"].items()}
conf_better = sum(1 for d in diffs.values() if d["diff"] > 0)
conf_sig = sum(1 for d in diffs.values() if d["ci95"][0] > 0)
WORD = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
claim(f"confidence beats margin in {conf_better} of {len(diffs)} settings, "
      f"significantly in {conf_sig}",
      in_tex_ws("the min versions") and in_tex(f"better in {WORD[conf_better]} of {WORD[len(diffs)]} settings, "
             f"significantly so in {WORD[conf_sig]}"),
      f"(computed {conf_better}/{len(diffs)}, sig {conf_sig})")

# ---------------------------------------------------------------- all correlations signed
allrho = [f["C_hat_bits"][p]["spearman"] for f in A["fidelity"].values()
          for p in av3.PROXIES]
allexc = [f["C_hat_bits"][p]["excludes_zero"] for f in A["fidelity"].values()
          for p in av3.PROXIES]
claim(f"all {len(allrho)} proxy correlations are negative and exclude zero",
      all(r < 0 for r in allrho) and all(allexc))
claim(f"count of proxy correlations ({len(allrho)}) appears",
      in_tex(f"${len(allrho)}$"), f"(computed {len(allrho)})")

# ---------------------------------------------------------------- disagreement vs C
dr = []
for t, v in RECS.items():
    for rg in ("suffix", "infill"):
        s = [x for x in v if x["regime"] == rg]
        if s:
            dr.append(av3.spearman([x["disagreement"] for x in s],
                                   [x["C_hat_bits"] for x in s]))
dlo, dhi = rng(dr)
claim(f"rho(disagreement, C) range {dlo:+.2f}-{dhi:+.2f} appears",
      in_tex(f"$+{dlo:.2f}$ to $+{dhi:.2f}$"), f"(computed {dlo:+.3f}-{dhi:+.3f})")

# ---------------------------------------------------------------- bug impact
bi = A.get("bug_impact", {})
ratios, flips = [], []
for t, imp in bi.items():
    for rgm, v in imp.items():
        c = v["chain_rule_deviation_bits"]
        ratios.append(c["v2_buggy"] / c["v3_fixed"])
        r = v["rho_conf_disagree"]
        flips.append((r["v2_buggy"][0] > 0) != (r["v3_fixed"][0] > 0))
ab = A.get("offset_ablation", {})
abr = [v["chain_rule_deviation_bits"]["wrong_offset"] / v["chain_rule_deviation_bits"]["right_offset"]
       for imp in ab.values() for v in imp.values()]
if abr:
    claim(f"ablation chain-rule inflation {min(abr):.0f}-{max(abr):.0f}x appears",
          in_tex(f"a factor of ${min(abr):.0f}$ to ${max(abr):.0f}$"),
          f"(computed {min(abr):.1f}-{max(abr):.1f})")
    negC = [v["median_C_hat_bits"]["wrong_offset"] for imp in ab.values() for v in imp.values()]
    claim("the misaligned readout drives median C negative (impossible for a joint)",
          min(negC) < 0, f"(min median C_hat = {min(negC):.2f})")
if ratios:
    claim("exactly the two deployed-regime correlations flip sign",
          sum(flips) == 2, f"(computed {sum(flips)} flips)")

# ---------------------------------------------------------------- M sensitivity
msp = os.path.join(RES, "e10_m_sensitivity_llada_base.json")
if os.path.exists(msp):
    ms = json.load(open(msp))
    moves = {rg: abs(r["residual_bits"][1] - r["residual_bits"][0])
             for rg, r in ms["by_regime"].items()}
    claim(f"residual is insensitive to M ({ms['m_low']}->{ms['m_high']}): "
          f"moves {min(moves.values()):.2f}-{max(moves.values()):.2f} bits",
          max(moves.values()) < 0.10, f"(computed {moves})")
    ses = {rg: r["meter_se_median"] for rg, r in ms["by_regime"].items()}
    claim("quadrupling M roughly halves the estimator standard error",
          all(0.35 < hi / lo < 0.65 for lo, hi in ses.values()), f"(computed {ses})")
    claim("the two M settings agree on the ranking of C (rho >= 0.98)",
          all(r["spearman_C_lo_hi"] >= 0.98 for r in ms["by_regime"].values()),
          f"({[round(r['spearman_C_lo_hi'],3) for r in ms['by_regime'].values()]})")

# ---------------------------------------------------------------- checkpoint count
words = {3: "three", 4: "four", 5: "five"}
claim(f"checkpoint count ({len(TAGS)}) stated consistently",
      TEX.lower().count(f"{words[len(TAGS)]} masked diffusion lms") >= 1
      and f"{words[len(TAGS)].capitalize()} checkpoints" in TEX,
      f"(have {len(TAGS)}: {', '.join(TAGS)})")

# ---------------------------------------------------------------- gates
G = json.load(open(os.path.join(RES, "gates.json")))["models"]
gaps = [m["offset_evidence"]["min_gap_bits_over_runner_up"] for m in G.values()
        if "offset" in m and m.get("offset") is not None]
claim("every gate passed and matched its documented convention",
      all(all(m["gates_passed"].values()) and m.get("measured_matches_docs", False)
          for m in G.values() if "error" not in m))
if gaps:
    claim(f"offset gap range {min(gaps):.1f}-{max(gaps):.1f} bits appears",
          in_tex(f"${min(gaps):.1f}$ to ${max(gaps):.1f}$ bits"),
          f"(computed {min(gaps):.1f}-{max(gaps):.1f})")

# ---------------------------------------------------------------- no impossibility claims
# The residual is an upper bound over the predictor class we tested, never a statement
# that no marginal rule can do better. That distinction leaked into three separate
# places (conclusion, a table caption, a figure caption) and was caught by reviewers
# each time rather than by us, so it is asserted here instead.
BANNED = [r"no marginal rule recover", r"no rule reading these marginals recover",
          r"invisible to any rule", r"cannot be recovered by any", r"impossible for any rule"]
hits = [b for b in BANNED if re.search(b, TEX, re.I)]
claim("no impossibility phrasing about the residual anywhere in the paper",
      not hits, f"(found: {hits})")

# ---------------------------------------------------------------- house style
# The author asked for no prose semicolons and no em dashes anywhere in the paper.
# Table captions are generated, so a regenerated table can silently reintroduce them.
_files = [os.path.join(ROOT, "paper", "main.tex"),
          os.path.join(ROOT, "paper", "references.tex")]
_files += sorted(glob.glob(os.path.join(ROOT, "paper", "tables", "*.tex")))
_style = []
for _f in _files:
    if os.path.basename(_f) == "table2_models.tex":   # superseded, not \input by the paper
        continue
    _txt = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}",      # keep line numbers
                  lambda m: "\n" * m.group(0).count("\n"), open(_f).read(), flags=re.S)
    for _i, _l in enumerate(_txt.split("\n"), 1):
        if "Kullback" in _l:                          # Kullback--Leibler is a proper en dash
            continue
        _l = re.sub(r"\$[^$]*\$", "", _l)              # math is notation, not prose
        if re.search(r"(?<!\\);", _l) or "---" in _l or re.search(r"(?<!-)--(?!-)", _l):
            _style.append(f"{os.path.basename(_f)}:{_i}")
claim("no prose semicolons or em dashes in the paper sources", not _style,
      f"(found at {_style})")

# ---------------------------------------------------------------- meter convergence
_cv = []
for _t in ("dream_base", "llada_base"):
    _p = os.path.join(RES, f"e9_meter_convergence_{_t}_k4.json")
    if not os.path.exists(_p):
        continue
    _d = json.load(open(_p))
    _pr = [_d["curve"][str(_m)]["median_se_bits"] * _m ** 0.5 for _m in _d["config"]["grid"]]
    _mu = sum(_pr) / len(_pr)
    _cv.append(max(abs(_x - _mu) / _mu for _x in _pr))
if _cv:
    pass

# ---------------------------------------------------------------- reviewer round 3
# The feature set is NOT a superset of every deployed rule: cross-step stability reads
# more than one decoding step and cannot be rebuilt from one step's profile. The old
# phrasing contradicted the list of deployed rules in the introduction.
claim("the discredited 'superset of every deployed marginal rule' phrasing is gone",
      not re.search(r"superset of every deployed", TEX, re.I)
      and all(not re.search(r"superset of every deployed", open(_f).read(), re.I)
              for _f in glob.glob(os.path.join(ROOT, "paper", "tables", "*.tex"))))

_lad = next(iter(A["explains_C"].values()))["predictor_ladder"]
_NW = {4: "four", 5: "five", 6: "six"}
claim(f"the paper says how many feature sets are swept ({len(_lad)}) and it matches",
      in_tex(f"ridge over {_NW[len(_lad)]} feature sets"), f"(ladder has {len(_lad)})")

_nst = os.path.join(RES, "e11_nested_cv.json")
if os.path.exists(_nst):
    _n = json.load(open(_nst))
    _md = _n["summary"]["max_delta_bits"]
    claim(f"nested-CV selection bias ({_md:+.3f} bits) is stated in the paper",
          in_tex(f"${_md:.3f}$ bits") or in_tex(f"{_md:.3f}$ bits"), f"(computed {_md:+.4f})")
    claim("nested selection never lowers the residual range below the reported one",
          _n["summary"]["nested_residual_range_bits"][1] >= max(
              e["unexplained_sd_bits"] for e in A["explains_C"].values()) - 0.01)

# Corpus, as described in the experimental-details appendix.
_cp = os.path.join(ROOT, "data", "corpus.jsonl")
if os.path.exists(_cp):
    _docs = [json.loads(_l) for _l in open(_cp)]
    _pro = sum(1 for d in _docs if d["id"].startswith("wikitext"))
    _cod = sum(1 for d in _docs if d["id"].startswith("humaneval"))
    claim(f"corpus composition ({len(_docs)} docs = {_pro} wikitext + {_cod} humaneval) "
          f"is stated correctly",
          in_tex(f"${len(_docs)}$ documents") and in_tex(f"${_pro}$ WikiText-103")
          and in_tex(f"all ${_cod}$ OpenAI"), f"(computed {len(_docs)}/{_pro}/{_cod})")

# Protocol constants must match the config the runs actually used.
_cfg = json.load(open(os.path.join(RES, "e5v3_llada_base.json")))["config"]
claim(f"window ({_cfg['window']}), M ({_cfg['m_meter']}) and top-m ({_cfg['topm']}) "
      f"match the appendix",
      in_tex(f"first ${_cfg['window']}$ tokens") and in_tex(f"$M{{=}}{_cfg['m_meter']}$ reveal orders")
      and in_tex(f"top ${_cfg['topm']}$ probabilities"), f"(config {_cfg['window']}/{_cfg['m_meter']}/{_cfg['topm']})")

# The k=8 replication now has a table, so its ranges must match it.
_k8p = os.path.join(RES, "e5v3_analysis_k8.json")
if os.path.exists(_k8p):
    _k8 = json.load(open(_k8p))["explains_C"]
    _r = [e["unexplained_sd_bits"] for e in _k8.values()]
    claim(f"k=8 residual range {min(_r):.2f}-{max(_r):.2f} appears",
          in_tex(f"${min(_r):.2f}$"), f"(computed {min(_r):.2f}-{max(_r):.2f})")

# ---------------------------------------------------------------- camera-ready corrections
# Appended to scripts/report/check_claims.py. Every number below is quoted in the
# camera-ready text, so each check recomputes it from results/e5v3_corrections.json and
# asserts the exact string the paper prints.
XC = json.load(open(os.path.join(RES, "e5v3_corrections.json")))
K4 = XC["k4"]
GOOD = [k for k in K4 if not k.startswith("dream_instruct/")]

wt = [K4[k]["window_tight_median_bits"] for k in K4]
claim(f"tight window range {min(wt):.1f}-{max(wt):.1f} bits appears",
      in_tex_ws(f"${min(wt):.1f}$ to ${max(wt):.1f}$ bits"), f"(computed {min(wt):.1f}-{max(wt):.1f})")
claim("the loose (1-1/k)B window is no longer quoted as the worst case",
      not in_tex_ws("$2.6$ to $13.0$ bits"))
claim("residual is below the tight window in every setting except Dream-instruct/infill",
      all(K4[k]["best_resid_bits"] < K4[k]["window_tight_median_bits"] for k in GOOD)
      and K4["dream_instruct/infill"]["best_resid_bits"] > K4["dream_instruct/infill"]["window_tight_median_bits"])

ratio = [K4[k]["null_uniform"]["resid_bits"] / K4[k]["best_resid_bits"] for k in GOOD]
claim(f"ratio to the coupling-agnostic reference {min(ratio):.1f}-{max(ratio):.1f} appears",
      in_tex_ws(f"${min(ratio):.1f}$ to ${max(ratio):.1f}$ times"), f"(computed {min(ratio):.2f}-{max(ratio):.2f})")
r8 = [v["null_uniform"]["resid_bits"] / v["best_resid_bits"] for v in XC["k8"].values()]
claim(f"k=8 ratio to the reference {min(r8):.1f}-{max(r8):.1f} appears",
      in_tex_ws(f"${min(r8):.1f}$ to ${max(r8):.1f}$ times"), f"(computed {min(r8):.2f}-{max(r8):.2f})")

fm = [K4[k]["fraction"]["f_median_W_gt_1"] for k in GOOD]
claim(f"median fraction of the window {min(fm):.2f}-{max(fm):.2f} appears",
      in_tex_ws(f"${min(fm):.2f}$ to ${max(fm):.2f}$ of"), f"(computed {min(fm):.2f}-{max(fm):.2f})")

pv = [K4[k]["partial_given_W"][p][0] for k in GOOD for p in K4[k]["partial_given_W"]]
claim(f"partial rho given W range {min(pv):+.2f} to {max(pv):+.2f} appears",
      in_tex_ws(f"${min(pv):.2f}$ and ${max(pv):+.2f}$"), f"(computed {min(pv):+.2f}..{max(pv):+.2f})")
claim("the coupling-agnostic comparison is called a reference, not a model",
      in_tex_ws("reference that knows the marginal entropies"))

hi_pool = max(K4[k]["pooled"][n][2] for k in K4 for n in
              ("conf_mean_minus_conf_min", "marg_mean_minus_marg_min", "negH_sum_minus_negH_max"))
claim("pooling beats min in all 24 PAIRED comparisons (every interval below zero)", hi_pool < 0,
      f"(largest upper end {hi_pool:+.3f})")

mm = {k: K4[k]["pooled"]["marg_mean_minus_conf_mean"] for k in K4}
conf_wins = sum(v[1] > 0 for v in mm.values())
marg_wins = sum(v[2] < 0 for v in mm.values())
claim(f"pooled confidence beats pooled margin in {conf_wins}, margin wins in {marg_wins}",
      (conf_wins, marg_wins) == (5, 2) and mm["dream_base/infill"][1] < 0 < mm["dream_base/infill"][2],
      f"(computed conf {conf_wins}, margin {marg_wins})")

dp = {k: K4[k]["domain"] for k in K4}
di_prose = [dp[k]["prose"]["resid_bits"] for k in K4 if k.startswith("dream_instruct/")]
other_prose = [dp[k]["prose"]["resid_bits"] for k in GOOD]
claim(f"Dream-instruct prose residual {min(di_prose):.2f}-{max(di_prose):.2f} exceeds every other "
      f"prose residual ({max(other_prose):.2f})",
      min(di_prose) > max(other_prose) and in_tex_ws(f"${min(di_prose):.2f}$ and ${max(di_prose):.2f}$ bits on prose"))
di_code = [dp[k]["code"]["resid_bits"] for k in K4 if k.startswith("dream_instruct/")]
claim(f"Dream-instruct code residual {min(di_code):.2f}-{max(di_code):.2f} is quoted",
      in_tex_ws(f"${min(di_code):.2f}$ and ${max(di_code):.2f}$ bits on code"))

# the sign-flip wording: a CI that includes zero is never called a flip
abl = A["offset_ablation"]["dream_base"]["suffix"]["rho_conf_disagree"]["wrong_offset"]
claim("Dream-base wrong-readout correlation includes zero, and the paper says 'erases'",
      abl[1] < 0 < abl[2] and in_tex_ws("erases") and not in_tex_ws("Both checkpoints flip"))


# ---------------------------------------------------------------- verification appendix (Task 12)
_PS = json.load(open(os.path.join(RES, "probe_summary.json")))
_T = ("dream_base", "dream_instruct", "llada_base")
_code = [_PS[t]["gate1_gap_bits"]["code"] for t in _T]
_prose = [_PS[t]["gate1_gap_bits"]["prose"] for t in _T]
claim("appendix: Gate 1 code and prose gaps are quoted",
      in_tex_ws(f"${min(_code):.1f}$ to ${max(_code):.1f}$ bits, against ${min(_prose):.1f}$ to ${max(_prose):.1f}$ on prose"))
_fr = _PS["flagged_reappear_total"]
claim("appendix: flagged blocks that reappear are quoted", in_tex_ws(f"${_fr[0]}$ of the ${_fr[1]}$ blocks the main run"))
_hl = _PS["hairline_total"]
claim("appendix: hairline KL counts are quoted", in_tex_ws(f"on ${_hl[0]}$ of ${_hl[1]}$, by up to ${_hl[2]:.2f}$ bits"))
claim("appendix: Dream alone-vs-batch entropy change is quoted",
      in_tex_ws(f"by up to ${max(_PS[t]['alone_vs_batch_max_dH_bits'] for t in _T):.1f}$ bits at a single position")
      and _PS["llada_base"]["alone_vs_batch_max_dH_bits"] < 0.005)   # "unchanged" at quoted precision
_hw = [v for t in _T for v in _PS[t]["hw_dB_mean_bits"].values()]
claim("appendix: A100 vs GB10 differences are quoted",
      in_tex_ws(f"by a mean of ${min(_hw):.2f}$ to ${max(_hw):.2f}$ bits per setting and by up to ${max(_PS[t]['hw_dB_max_bits'] for t in _T):.1f}$ bits"))
claim("appendix: order information bound is quoted", in_tex_ws(f"at most ${_PS['max_control_I_bits']:.2f}$ bits"))

claim("the estimand identity is stated", in_tex(r"\mathrm{KL}(\bar q\,\|\,\textstyle\prod_i p_i)") and in_tex("I(\\pi;Y_S)"))
claim("a reviewer's two-bit example is in the paper", in_tex("$0.531$ bits"))
claim("the assumption boundary is stated", in_tex("assume that the conditionals\ncompose into one joint") or in_tex("assume that the conditionals compose into one joint"))
PS = json.load(open(os.path.join(RES, "probe_summary.json")))
_g0, _g1 = PS["typical_gap_bits"]
claim(f"typical KL-minus-C gap {_g0:.1f}-{_g1:.1f} bits is quoted", in_main_ws(f"${_g0:.1f}$ to\n${_g1:.1f}$ bits"))
claim("rank agreement of C_hat with the KL cost is quoted",
      in_main_ws(f"Spearman ${PS['llada_base']['rho_C_KL_controls']:.2f}$ and ${PS['dream_base']['rho_C_KL_controls']:.2f}$"))
_d0, _d1 = PS["di_prose_gap_bits"]
claim("Dream-instruct prose KL gap is quoted", in_main_ws(f"the gap is ${_d0:.1f}$\nand ${_d1:.1f}$ bits"))

claim("no claim that the residual measures room for dependency-aware decoders",
      not in_tex_ws("how much room a marginal rule leaves") and not in_tex_ws("far smaller\nin practice") and not in_tex_ws("cuts both ways"))
claim("the decoder question is named as unmeasured", in_tex_ws("matched-compute"))


# ---------------------------------------------------------------- Task 6: impossible values
_ab = [100 * K4[k]["impossible"]["frac_above_window"] for k in K4]
claim(f"share of blocks above W ({min(_ab):.1f}-{max(_ab):.1f}%) is quoted",
      in_tex_ws(f"${min(_ab):.1f}\\%$ to ${max(_ab):.1f}\\%$ above $W$"))
_sb = [100 * (K4[k]["impossible"]["frac_below_substantive"] + K4[k]["impossible"]["frac_above_substantive"]) for k in GOOD]
claim(f"substantive impossible share outside Dream-instruct ({min(_sb):.1f}-{max(_sb):.1f}%) is quoted",
      in_tex_ws(f"${min(_sb):.1f}\\%$ to ${max(_sb):.1f}\\%$ of blocks outside Dream-instruct"))


# ---------------------------------------------------------------- Task 7: pooled rules
_pr = [A["fidelity"][k]["C_hat_bits"][p]["spearman"] for k in A["fidelity"] if not k.startswith("dream_instruct")
       for p in ("confidence_mean", "margin_mean", "neg_entropy_sum")]
claim(f"pooled rules reach rho {max(_pr):.2f} to {min(_pr):.2f} outside Dream-instruct, and the paper says so",
      in_main_ws(f"$\\rho={max(_pr):.2f}$ to ${min(_pr):.2f}$"))
claim("the 24 pooled-vs-min comparisons are described as paired",
      hi_pool < 0 and in_main_ws("a paired bootstrap excludes zero in all $24$ comparisons"))

claim("no sentence says a wrong offset produces no garbage", not in_tex_ws("no obvious garbage") and not in_tex_ws("nor garbles"))
claim("Gate 2 is described as implemented (argmax, not logits)", in_tex_ws("argmax") and not in_tex_ws("must return\nidentical logits, and a sequence must get the same logits alone as in a batch"))
claim("the ablation no longer claims an identical seed", not in_tex_ws("same corpus, blocks, seed and\ncode path") and not in_tex_ws("identical corpus,\nblocks, seed and code path"))
_g3 = json.load(open(os.path.join(RES, "probe_summary.json")))["gate3"]
claim("Gate 3 evidence is quoted from the probe",
      in_tex_ws(f"${_g3['wrong_prose_run_min']}$ to ${_g3['wrong_prose_run_max']}$ of $32$ prose positions")
      and in_tex_ws(f"of at most ${_g3['calibrated_run_max']}$ at the calibrated offset"))


# ---------------------------------------------------------------- Task 8: ablation intervals
for _t in ("dream_base", "dream_instruct"):
    _o, _lo, _hi = A["offset_ablation"][_t]["suffix"]["rho_conf_disagree"]["wrong_offset"]
    claim(f"{_t} wrong-readout correlation {_o:+.2f} [{_lo:.2f},{_hi:.2f}] is quoted with its interval",
          in_main_ws(f"${_o:+.2f}$ $[{_lo:.2f},{_hi:.2f}]$"))

claim("Eq. (3) states the tight window", in_tex_ws(r"\C_{\max} \;\le\; W \;=\; B-\max_i H(Y_i) \;\le\; (1-1/k)\,B"))
claim("the exact certification statement is present", in_tex_ws("if and only if at most one position"))
claim("Huang Thm 1 and ParallelBench Sec 4 are cited precisely", in_tex_ws(r"\cite[Thm.~1]{huang}") and in_tex_ws(r"\cite[\S4]{parallelbench}"))

claim("App. A no longer says the 600 cap explains the counts", not in_tex_ws("which is why the per-setting counts run"))
claim("App. A states the per-family document counts", in_tex_ws("$126$ of the code problems") and in_tex_ws("$137$"))
claim("no 'same text' claim for the two families", not in_tex_ws("see the same text"))

for phrase in ("entirely spurious", "the failure mode is\nsilent", "cuts both ways", "is dangerous in proportion",
               "toward a publishable claim", "wildly incompatible"):
    claim(f"tone: '{phrase.split(chr(10))[0]}' removed", not in_tex_ws(phrase))


# ---------------------------------------------------------------- final-review fixes
_TABLES = " ".join(re.sub(r"\s+", " ", open(_f).read()) for _f in sorted(glob.glob(os.path.join(ROOT, "paper", "tables", "*.tex"))))
in_tables = lambda s: re.sub(r"\s+", " ", s) in _TABLES  # noqa: E731
# I1: W is not tight, the reference knows the entropies only
claim("W is never called tight, and no 'nothing tighter' claim",
      not any(in_tex_ws(x) for x in ("nothing tighter", "The tight form", "tight certified window", "The tight worst case"))
      and not in_tables("tight certified window"))
claim("the reference is described as knowing the marginal entropies",
      in_tex_ws("knows the marginal entropies") and in_tables("knows the marginal entropies"))
# I2: the partial range is scoped, and Table 11 shows all six rules
claim("partial-correlation range is scoped to outside Dream-instruct in both places",
      in_main_ws("outside Dream-instruct, given $W$, the rank correlation")
      and in_main_ws("given $W$, the correlations outside Dream-instruct fall to between"))
claim("Table 11 shows the range over all six rules", in_tables("over the six rules"))
# I3: probe numbers name their population, Table 12 shows the random blocks
_PS2 = json.load(open(os.path.join(RES, "probe_summary.json")))
_nc = [_PS2[t]["controls"][k]["n"] for t in ("llada_base", "dream_base") for k in ("infill/prose", "suffix/prose", "suffix/code")]
claim("appendix gives the random-block cell sizes and says the Spearman is pooled",
      in_tex_ws(f"(${min(_nc)}$ to ${max(_nc)}$ blocks per cell)") and in_tex_ws("pooled across settings"))
claim("Table 12 reports the random blocks separately", in_tables("random blocks only"))
# I4: hairlines are Monte Carlo under-coverage, with their shares
_tot = _hl = _hlc = 0
for _t, _rs in RECS.items():
    if _t == "dream_instruct":
        continue
    for _r in _rs:
        _H = np.asarray(_r["profile_entropy_bits"]); _Wr = _r["budget_bits"] - _H.max()
        _b, _a = _r["C_hat_bits"] < -2 * _r["C_hat_se_bits"], _r["C_hat_bits"] > _Wr + 2 * _r["C_hat_se_bits"]
        if _b or _a:
            _tot += 1
            if (-_r["C_hat_bits"] if _b else _r["C_hat_bits"] - _Wr) <= 0.2:
                _hl += 1; _hlc += _r["domain"] == "code"
claim("hairline share and its code share are quoted, and no 'numerical noise' explanation",
      in_main_ws(f"${100*_hl/_tot:.0f}\\%$ of these are hairlines within $0.2$ bits, ${100*_hlc/_hl:.0f}\\%$ of them on")
      and not in_tex_ws("numerical noise") and in_tex_ws("under-covers"))
# I5: flags that change when B comes from the estimator's own forward pass
_chg = _N = 0
for _t in ("dream_base", "dream_instruct", "llada_base"):
    for _r in json.load(open(os.path.join(RES, f"probe_{_t}.json")))["records"]:
        _f = []
        for _C, _Hk in ((_r["C8"], "H8"), (_r["C0"], "H0")):
            _h = np.asarray(_r[_Hk]); _f.append(_C < -2 * _r["Hsum_se"] or _C > _h.sum() - _h.max() + 2 * _r["Hsum_se"])
        _chg += _f[0] != _f[1]; _N += 1
claim(f"flag changes ({_chg} of {_N}) are quoted, not 'no flag'",
      in_tex_ws(f"changes ${_chg}$ of the ${_N}$ impossibility flags") and not in_tex_ws("changes no impossibility flag"))
claim("reappearing flags outside Dream-instruct are quoted", in_tex_ws("($39$ of $53$ outside Dream-instruct)"))
# I6: numbers that had no check
_fw = lambda k: K4[k]["fraction"]["resid_bits"]  # noqa: E731
_bb = lambda k: K4[k]["best_resid_bits"]  # noqa: E731
claim("fW vs best on LLaDA-base is quoted",
      in_main_ws(f"(${_fw('llada_base/infill'):.2f}$ and ${_fw('llada_base/suffix'):.2f}$ bits against ${_bb('llada_base/infill'):.2f}$ and ${_bb('llada_base/suffix'):.2f}$)"))
claim("Dream-base suffix fW-to-best drop is quoted",
      in_main_ws(f"from ${_fw('dream_base/suffix'):.2f}$ to ${_bb('dream_base/suffix'):.2f}$ bits"))
_bz = [100 * K4[k]["impossible"]["frac_below_zero"] for k in K4]
claim("below-zero range is quoted", in_main_ws(f"${min(_bz):.0f}\\%$ to ${max(_bz):.0f}\\%$ of blocks fall below zero"))
_ds = [100 * (K4[k]["impossible"]["frac_below_substantive"] + K4[k]["impossible"]["frac_above_substantive"]) for k in ("dream_instruct/suffix", "dream_instruct/infill")]
claim("Dream-instruct substantive shares are quoted", in_main_ws(f"${_ds[0]:.1f}\\%$ and\n${_ds[1]:.1f}\\%$ on it"))
_ns = [K4[k]["pooled"]["negH_sum_minus_conf_mean"] for k in K4]
_nb, _cb = sum(v[2] < 0 for v in _ns), sum(v[1] > 0 for v in _ns)
claim("summed entropy vs mean confidence split is quoted",
      in_main_ws(f"splits {WORD[_nb]}, {WORD[_cb]} and {WORD[len(_ns) - _nb - _cb]}"))
claim("pooled confidence vs margin counts are quoted in words",
      in_main_ws(f"significantly better in {WORD[conf_wins]} settings, mean margin in the {WORD[marg_wins]} Dream-instruct settings")
      and all(mm[k][2] < 0 for k in mm if k.startswith("dream_instruct/")))
_g = json.load(open(os.path.join(RES, "gates.json")))["models"]
claim("Gate 2 alone-vs-batch maximum is quoted",
      in_main_ws(f"by up to ${max(m['determinism']['batch1_vs_batchN_max_abs_logprob_diff'] for m in _g.values()):.2f}$"))
claim("Gate 2 text names the argmax condition on the measurement shapes", in_main_ws("must agree on every argmax"))
_ms = json.load(open(os.path.join(RES, "e10_m_sensitivity_llada_base.json")))
_md = max(abs(r["residual_bits"][1] - r["residual_bits"][0]) for r in _ms["by_regime"].values())
claim(f"M sensitivity quoted as at most {_md:.3f} bits in text and Table 8",
      in_main_ws(f"at most ${_md:.3f}$ bits") and in_tables(f"at most ${_md:.3f}$ bits"))
# I7: the k=8 table uses W and the reference
claim("Table 7 reports W and the reference residual, not the loose bound",
      in_tables("$k{=}8$ replication") and not in_tables("worst-case width $(1-1/k)B$ grows faster") and in_main_ws("Table~\\ref{tab:k8}"))
# I8: abstract and conclusion say what is predicted and how it relates to the KL cost
claim("abstract and conclusion name the estimate and its gap to the KL cost",
      in_main_ws("\\emph{predict} an order-averaged estimate") and in_main_ws("predict the estimate $\\Chat$")
      and in_main_ws("sits a median $0.4$ to $1.1$ bits below it") and not in_tex_ws("only where the diagnostics"))
# upgraded minors
claim("Dream-instruct anomaly described as concentrated, not confined, on prose",
      in_tex_ws("concentrated on prose") and in_tex_ws("mainly on prose")
      and not in_tex_ws("confined to prose") and not in_tex_ws("on prose only"))
claim("instruction tuning claim is limited", in_tex_ws("not a generic instruction-tuning effect"))
claim("Limitations keep the condition of the order identity", in_tex_ws("When each reveal order keeps the parallel marginals"))
claim("no 'follows from C<=W' overstatement", not in_tex_ws("follows from $\\C\\le W$"))
claim("median fraction names its block set", in_main_ws("with $W$ above one bit"))
claim("no 'ordinary-looking statistics' after the impossible C_hat", not in_tex_ws("ordinary-looking"))
claim("conclusion restricts the collapse to prose", in_main_ws("collapses greedy prose"))
claim("proxy table no longer says 'real models'", not in_tables("real models"))


# ---------------------------------------------------------------- release
claim("code availability points at the public repository and the tag",
      in_tex_ws(r"\url{https://github.com/DovudAsadov/dllm-total-correlation}")
      and in_tex_ws(r"\texttt{paper-numbers}") and not in_tex_ws("upon acceptance"))


# ---------------------------------------------------------------- Gate 2 with the entropy tolerance
_G2 = {t: json.load(open(os.path.join(RES, f"gate2_entropy_{t}.json"))) for t in
       ("dream_base", "dream_instruct", "llada_base", "llada_instruct")}
_d2 = {t: g["determinism"] for t, g in _G2.items()}
claim("Gate 2 tolerance is quoted as a tenth of the smallest residual",
      all(abs(d["entropy_rms_tol_bits"] - 0.074) < 1e-9 for d in _d2.values())
      and abs(0.1 * min(rng(resid)) - 0.074) < 0.001 and in_main_ws("$0.074$ bits RMS"))
claim("all four checkpoints pass Gate 2, with the estimator shapes agreeing exactly",
      all(g["passed"] for g in _G2.values())
      and all(d["estimator_batch_max_abs_dH_bits"] == 0.0 and d["estimator_batch_argmax_flips"] == 0 for d in _d2.values())
      and in_main_ws("all four checkpoints pass with the two shapes agreeing exactly"))
_df = {t: _d2[t]["batch_size_argmax_flips"] for t in _d2}
claim("alone-vs-batch flips are quoted (1 of 64 on each Dream, none on LLaDA)",
      _df["dream_base"] == _df["dream_instruct"] == 1 and _df["llada_base"] == _df["llada_instruct"] == 0
      and all(d["n_positions_compared_alone_vs_batch"] == 64 for d in _d2.values())
      and in_main_ws("flips $1$ of $64$ argmaxes on each Dream checkpoint"))
claim("appendix quotes the largest alone-vs-batch entropy change",
      in_tex_ws(f"up to ${max(d['batch1_vs_batchN_max_abs_entropy_diff_bits'] for d in _d2.values()):.1f}$ bits of entropy at one position"))


claim("author emails are on the title page, in author order",
      in_tex_ws(r"\texttt{\{d.asadov, r.urokov, q.toshnazarov, a.kayumov, m.hazratov, h.mansuraliyev\}@newuu.uz}"))

print(f"\n{checks - len(fails)}/{checks} claim checks passed")
if fails:
    print("\nFAILURES:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
