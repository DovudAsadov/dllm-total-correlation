"""
The proxy-fidelity audit.

The unit of the audit is a PreparedSample: the per-position predictive
distributions over a candidate set, together with the ground-truth C of
committing that set. Preparing samples this way decouples the audit from any
single global joint (so we never build a 2^22 Kronecker product) and makes the
oracle path and the real-model path identical downstream:

  * oracle path : dists come from conditioning an explicit task joint; C is exact.
  * graded path : each sample is its own tiny 2-bit block; C is exact.
  * model path  : dists come from a forward pass with the candidate positions
                  masked; C comes from the meter.

For every proxy we report how well its safety score tracks C:
  - Spearman / Kendall vs C   (faithful => strongly NEGATIVE: higher safety, lower C)
  - AUROC for safe-set detection (C ~ 0 vs C > 0)
and the decisive matched-marginal separation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .information import (
    Joint,
    condition_on,
    marginal,
    total_correlation,
)
from .proxies import MARGINAL_PROXIES
from .stats import auroc, kendall_tau, paired_bootstrap_corr, spearman

Assignment = Dict[int, int]
Dist = Dict[int, float]


@dataclass
class Sample:
    """A ground-truth query: commit `candidate_set` given `context`."""
    context: Assignment
    candidate_set: Tuple[int, ...]
    C_true: float
    task: str = ""


@dataclass
class PreparedSample:
    """A sample with its per-position distributions materialised."""
    dists: List[Dist]
    C_true: float
    task: str = ""


@dataclass
class ProxyReport:
    name: str
    spearman_vs_C: float
    kendall_vs_C: float
    spearman_ci: Tuple[float, float]
    auroc_safe: float
    n_samples: int


@dataclass
class AuditResult:
    proxy_reports: List[ProxyReport]
    C_values: List[float]
    proxy_scores: Dict[str, List[float]]
    safe_threshold: float
    n_safe: int
    n_unsafe: int


# ---- preparing samples ----

def prepare_from_joint(joint: Joint, samples: Sequence[Sample]) -> List[PreparedSample]:
    """Materialise per-position distributions for oracle-task samples by
    conditioning the joint on each sample's context."""
    prepared: List[PreparedSample] = []
    for s in samples:
        post = condition_on(joint, s.context) if s.context else joint
        dists = [marginal(post, i) for i in s.candidate_set]
        prepared.append(PreparedSample(dists=dists, C_true=s.C_true, task=s.task))
    return prepared


# ---- running the audit ----

def run_audit(
    prepared: Sequence[PreparedSample],
    safe_threshold: float = 1e-6,
    proxies: Optional[Dict[str, Callable]] = None,
    bootstrap: int = 1000,
) -> AuditResult:
    proxies = proxies or MARGINAL_PROXIES
    C_values = [p.C_true for p in prepared]
    scores: Dict[str, List[float]] = {name: [] for name in proxies}
    for p in prepared:
        for name, fn in proxies.items():
            scores[name].append(float(fn(p.dists)))

    labels_safe = [1 if c <= safe_threshold else 0 for c in C_values]
    n_safe = sum(labels_safe)
    n_unsafe = len(labels_safe) - n_safe

    reports: List[ProxyReport] = []
    for name in proxies:
        sc = scores[name]
        rho = spearman(sc, C_values)
        tau = kendall_tau(sc, C_values)
        ci = paired_bootstrap_corr(sc, C_values, spearman, n_boot=bootstrap) \
            if bootstrap else (float("nan"), float("nan"))
        au = auroc(sc, labels_safe)
        reports.append(ProxyReport(name, rho, tau, ci, au, len(sc)))

    return AuditResult(reports, C_values, scores, safe_threshold, n_safe, n_unsafe)


# ---- the decisive experiment ----

def matched_marginal_separation(backend_unused=None, n: int = 4) -> Dict:
    """Build free vs constrained coins, verify identical marginals, and show
    every marginal proxy returns identical scores while true C differs.

    (The positional `backend_unused` argument is kept for call-site symmetry; the
    experiment constructs its own exact distributions.)"""
    from .tasks import matched_marginal_pair

    free, cons, info = matched_marginal_pair(n)
    S = list(range(n))
    d_free = [marginal(free, i) for i in S]
    d_cons = [marginal(cons, i) for i in S]

    p1_free = [d.get(1, 0.0) for d in d_free]
    p1_cons = [d.get(1, 0.0) for d in d_cons]
    max_marginal_gap = max(abs(a - b) for a, b in zip(p1_free, p1_cons))

    proxy_free = {name: fn(d_free) for name, fn in MARGINAL_PROXIES.items()}
    proxy_cons = {name: fn(d_cons) for name, fn in MARGINAL_PROXIES.items()}
    proxy_gap = {name: abs(proxy_free[name] - proxy_cons[name])
                 for name in MARGINAL_PROXIES}

    return {
        "n": n,
        "C_free": info["C_free"],
        "C_constrained": info["C_constrained"],
        "max_marginal_gap": max_marginal_gap,
        "proxy_free": proxy_free,
        "proxy_constrained": proxy_cons,
        "proxy_gap": proxy_gap,
        "verdict": (
            f"All marginal proxies identical across the pair (max gap "
            f"{max(proxy_gap.values()):.2e}) while C differs by "
            f"{info['C_constrained'] - info['C_free']:.4f} bits."
        ),
    }
