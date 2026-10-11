"""
The selection proxies under audit.

Every proxy maps (per-position predictive distributions over a candidate set S)
to a scalar SAFETY score: higher = "the model thinks this set is safer to commit
in parallel". The whole point of the paper is to check these scalars against the
true C(S | context).

The families here are the ones the ~40 adaptive-parallelism papers actually use.
Critically, ALL of the ones computable without a denoising trajectory or attention
are functions of the per-position MARGINALS ONLY:

    confidence, margin, negative-entropy, entropy-sum

so on the matched-marginal pair (identical marginals, different C) they are
analytically forced to return identical scores. That is the decisive result and
it needs no model.

Two proxies are intrinsically real-model-only and cannot be run on the oracle:
    - stability_jsd : needs consecutive denoising-step predictions
    - attention_dependency : needs the backbone's attention maps
They are declared here with `requires="trajectory"/"attention"` so the audit can
report them as N/A on the oracle and only compute them on LLaDA/Dream.

Sign convention: we return SAFETY (higher = safer). A faithful proxy should be a
DECREASING function of C, i.e. correlate negatively with C. The audit reports the
correlation of each safety score with C and interprets a strong NEGATIVE
correlation as "faithful". We also expose the raw underlying quantity for clarity.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence

from .information import entropy_bits

Dist = Dict[int, float]  # {symbol: prob} for one position


def _max_prob(d: Dist) -> float:
    return max(d.values())


def _top2_margin(d: Dist) -> float:
    vals = sorted(d.values(), reverse=True)
    if len(vals) == 1:
        return vals[0]
    return vals[0] - vals[1]


def _entropy(d: Dist) -> float:
    return entropy_bits(d.values())


# ---- marginal-only proxies (auditable on the oracle) ----

def confidence_min(dists: Sequence[Dist]) -> float:
    """min_i max_v p(Y_i=v). The most common rule: commit the set only if the
    least-confident member is confident. Higher = safer."""
    return min(_max_prob(d) for d in dists)


def confidence_mean(dists: Sequence[Dist]) -> float:
    return sum(_max_prob(d) for d in dists) / len(dists)


def margin_min(dists: Sequence[Dist]) -> float:
    """min_i (p_top1 - p_top2). Higher = safer."""
    return min(_top2_margin(d) for d in dists)


def margin_mean(dists: Sequence[Dist]) -> float:
    return sum(_top2_margin(d) for d in dists) / len(dists)


def neg_entropy_max(dists: Sequence[Dist]) -> float:
    """-max_i H(Y_i). Low max marginal entropy assumed safe. Higher = safer."""
    return -max(_entropy(d) for d in dists)


def neg_entropy_sum(dists: Sequence[Dist]) -> float:
    """-sum_i H(Y_i). This is the negative of the FIRST term of C itself -- the
    'different currency' proxy: it tracks the marginal-entropy budget, never the
    subtracted joint term. Included precisely to show it correlates with sum-H
    and not with C. Higher = safer."""
    return -sum(_entropy(d) for d in dists)


# ---- controls ----

def random_score(dists: Sequence[Dist], rng=None) -> float:
    import numpy as np
    rng = rng or np.random.default_rng()
    return float(rng.random())


# Registry of proxies that depend only on marginals (oracle-auditable).
MARGINAL_PROXIES: Dict[str, Callable[[Sequence[Dist]], float]] = {
    "confidence_min": confidence_min,
    "confidence_mean": confidence_mean,
    "margin_min": margin_min,
    "margin_mean": margin_mean,
    "neg_entropy_max": neg_entropy_max,
    "neg_entropy_sum": neg_entropy_sum,
}

# Declared-but-model-only proxies (documented; computed in the model backend).
MODEL_ONLY_PROXIES = {
    "stability_jsd": "trajectory",       # LESS, arXiv:2606.16908
    "attention_dependency": "attention", # DAPD 2603.12996 / GeoBlock 2603.26675
}


def score_all_marginal(dists: Sequence[Dist]) -> Dict[str, float]:
    """Compute every marginal-only proxy for a candidate set's per-position
    distributions."""
    return {name: fn(dists) for name, fn in MARGINAL_PROXIES.items()}
