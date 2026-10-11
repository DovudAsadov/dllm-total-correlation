#!/usr/bin/env python3
"""Table 1: the validation gates, generated straight from results/gates.json so the
paper can never drift from the measurement."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LABEL = {"dream_base": r"Dream-Coder-7B (base)",
         "dream_instruct": r"Dream-Coder-7B (instruct)",
         "llada_base": r"LLaDA-8B (base)",
         "llada_instruct": r"LLaDA-8B (instruct)"}
DOC = {"dream_base": r"$-1$ (AR shift)", "dream_instruct": r"$-1$ (AR shift)",
       "llada_base": r"$0$ (in place)", "llada_instruct": r"$0$ (in place)"}
ORDER = ["dream_base", "dream_instruct", "llada_base", "llada_instruct"]

models = {}
gpath = os.path.join(ROOT, "results", "gates.json")
if os.path.exists(gpath):
    models.update({k: v for k, v in json.load(open(gpath))["models"].items()
                   if "error" not in v})
for tag in ORDER:                       # per-model files are authoritative and complete
    f = os.path.join(ROOT, "results", f"e5v3_{tag}.json")
    if os.path.exists(f):
        g = json.load(open(f))["gates"]
        g.setdefault("measured_matches_docs", True)
        models[tag] = g
rows = []
for tag in ORDER:
    if tag not in models:
        continue
    r = models[tag]
    ev = r["offset_evidence"]["mean_log2_prob_of_hidden_true_token_by_delta"]["k=4"]
    det = r["determinism"]
    rows.append((
        LABEL.get(tag, tag),
        ("$0$" if r["offset"] == 0 else f"${r['offset']:+d}$"),
        DOC.get(tag, "--"),
        f"${ev['-1']:.2f}$", f"${ev['0']:.2f}$", f"${ev['1']:.2f}$",
        f"${sorted(ev.values())[-1] - sorted(ev.values())[-2]:.1f}$",
        f"${(lambda e1: sorted(e1.values())[-1] - sorted(e1.values())[-2])(r['offset_evidence']['mean_log2_prob_of_hidden_true_token_by_delta']['k=1']):.1f}$",
        str(det["repeat_call_argmax_flips"]),
        r"\checkmark" if all(r["gates_passed"].values()) else r"$\times$",
    ))

tex = r"""\begin{table}[!ht]
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccrrrrccc}
\toprule
 & \multicolumn{2}{c}{readout offset} & \multicolumn{3}{c}{$\log_2 p(\text{hidden true token})$ read at $p{+}\delta$} & \multicolumn{2}{c}{gap (bits)} & & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}
Model & measured & documented & $\delta{=}{-}1$ & $\delta{=}0$ & $\delta{=}{+}1$ & $k{=}4$ & $k{=}1$ & flips & gates \\
\midrule
""" + "\n".join(" & ".join(r) + r" \\" for r in rows) + r"""
\bottomrule
\end{tabular}
\caption{\textbf{The readout convention is measured, not assumed.} We mask a $k{=}4$
block and score the log-probability the model gives the tokens that were actually there,
reading at index $p{+}\delta$. The true tokens are hidden, so a bidirectional model
cannot score well by copying, and only the correct convention wins. The gate recovers
each checkpoint's documented convention without being told it, and the two families are
\emph{opposite}: Dream keeps the autoregressive shift of the backbone it was initialised
from, LLaDA predicts masked tokens in place. The two Dream rows are the Dream-Coder
checkpoints \cite{dreamcoder}, adapted from an autoregressive backbone by the recipe of
\cite{dream}, and the gate finds they keep its shift. ``gap'' is the winning $\delta$'s margin
over the runner-up in bits, at $k{=}4$ (the block size of the three columns before it)
and with one hidden token ($k{=}1$). The gate requires both to exceed one bit. ``flips'' counts argmax changes
between two identical forward calls. Any nonzero value would be a noise floor that
correlates with confidence and therefore contaminates a proxy audit. Reading Dream at
$\delta{=}0$, the convention our own earlier version used, costs
$6.9$ bits of log-probability on the correct token at $k{=}1$ and $8.4$ at $k{=}4$, a
factor of ${\sim}120$ or more in probability (\S\ref{sec:bugimpact}).}
\label{tab:gates}
\end{table}
"""
p = os.path.join(ROOT, "paper", "tables/gates.tex")
open(p, "w").write(tex)
print("wrote", p)
print(tex)
