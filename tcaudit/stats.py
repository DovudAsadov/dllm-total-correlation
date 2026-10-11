"""
Rank statistics, implemented in pure numpy so nothing depends on SciPy
(which is not reliably installable on the target aarch64 box).

  - spearman(x, y)     : Pearson correlation of ranks (ties -> average ranks)
  - kendall_tau(x, y)  : tau-b, O(n^2), fine for our sample sizes
  - auroc(scores, labels): area under ROC via the Mann-Whitney U statistic,
                           with proper handling of tied scores (0.5 credit)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _avg_ranks(a: np.ndarray) -> np.ndarray:
    """Average ranks (1..n), ties averaged -- matches scipy.rankdata('average')."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2:
        return float("nan")
    sx, sy = x.std(), y.std()
    if sx <= 1e-15 or sy <= 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2:
        return float("nan")
    return pearson(_avg_ranks(x), _avg_ranks(y))


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall tau-b (handles ties)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 2:
        return float("nan")
    concordant = discordant = 0
    tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            s = dx * dy
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
            else:
                if dx == 0:
                    tx += 1
                if dy == 0:
                    ty += 1
    n0 = n * (n - 1) / 2.0
    denom = np.sqrt((n0 - tx) * (n0 - ty))
    if denom <= 1e-15:
        return float("nan")
    return float((concordant - discordant) / denom)


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUROC that a higher score indicates label==1, via Mann-Whitney U with
    tie-aware average ranks. Returns nan if only one class present."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _avg_ranks(scores)
    sum_ranks_pos = ranks[pos].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def bootstrap_ci(values: Sequence[float], stat_fn, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0):
    """Generic bootstrap CI over paired samples is done at the call site; this
    helper bootstraps a 1-D array statistic. Returns (lo, hi)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = stat_fn(values[idx])
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return (lo, hi)


def paired_bootstrap_corr(x: Sequence[float], y: Sequence[float], corr_fn,
                          n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Bootstrap CI for a correlation between paired x, y."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = corr_fn(x[idx], y[idx])
    stats = stats[~np.isnan(stats)]
    if len(stats) == 0:
        return (float("nan"), float("nan"))
    return (float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2)))
