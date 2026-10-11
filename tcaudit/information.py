"""
Information-theoretic primitives over *explicit* joint distributions.

Everything here operates on a fully materialised joint distribution represented
as a dict {config: prob}, where `config` is a tuple of ints (one symbol per
position). This is only tractable for the small synthetic tasks (support up to a
few thousand configs) but it is *exact* -- no estimation, no model -- which is
exactly what we need for ground truth.

Units: all entropies and total correlations are in BITS (log base 2).

The one quantity the whole paper is about:

    C(S | context) = sum_{i in S} H(Y_i | context)  -  H(Y_S | context)

the *conditional total correlation* of a set of positions S. It is >= 0, and it
is 0 iff the positions in S are mutually independent given the context. It is the
exact KL cost of committing the set S in a single parallel step
(ParallelBench, arXiv:2510.04767, Thm 1).

No SciPy: pure Python + numpy so it runs anywhere, including an aarch64 box with
a fragile wheel situation.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

Config = Tuple[int, ...]
Joint = Dict[Config, float]

_LOG2 = math.log(2.0)
_EPS = 1e-15


def _xlogx_bits(p: float) -> float:
    """Return -p * log2(p), with the usual convention 0*log0 = 0."""
    if p <= _EPS:
        return 0.0
    return -p * math.log(p) / _LOG2


def entropy_bits(probs: Iterable[float]) -> float:
    """Shannon entropy in bits of an iterable of probabilities.

    Does not assume the probabilities are normalised to exactly 1; it uses them
    as given (callers are expected to pass a valid distribution). Values <= 0 are
    treated as 0 under the 0log0 convention.
    """
    return sum(_xlogx_bits(float(p)) for p in probs)


def normalise(joint: Joint) -> Joint:
    """Return a copy of `joint` renormalised to sum to 1."""
    z = sum(joint.values())
    if z <= _EPS:
        raise ValueError("joint has zero total mass")
    return {k: v / z for k, v in joint.items()}


def joint_entropy(joint: Joint) -> float:
    """H(Y) in bits of an explicit joint over full configs."""
    return entropy_bits(joint.values())


def marginal(joint: Joint, i: int) -> Dict[int, float]:
    """Marginal distribution of position i as {symbol: prob}."""
    m: Dict[int, float] = defaultdict(float)
    for cfg, p in joint.items():
        m[cfg[i]] += p
    return dict(m)


def marginal_entropy(joint: Joint, i: int) -> float:
    """H(Y_i) in bits."""
    return entropy_bits(marginal(joint, i).values())


def sub_joint(joint: Joint, positions: Sequence[int]) -> Joint:
    """Marginalise the joint down to only the positions in `positions`,
    preserving their order. Returns a joint over |positions|-tuples."""
    out: Joint = defaultdict(float)
    for cfg, p in joint.items():
        key = tuple(cfg[i] for i in positions)
        out[key] += p
    return dict(out)


def total_correlation(joint: Joint, positions: Sequence[int] | None = None) -> float:
    """Conditional total correlation C of the given positions under `joint`.

    If `positions` is None, uses all positions of the joint. `joint` is treated
    as already conditioned on whatever context/committed values it encodes, so
    the returned value is C(positions | that context).

    C = sum_i H(Y_i) - H(Y_{positions}).
    """
    joint = normalise(joint)
    any_cfg = next(iter(joint))
    if positions is None:
        positions = list(range(len(any_cfg)))
    if len(positions) == 0:
        return 0.0
    if len(positions) == 1:
        return 0.0  # a single variable has zero total correlation by definition
    sub = sub_joint(joint, positions)
    sum_marg = sum(marginal_entropy(sub, k) for k in range(len(positions)))
    return sum_marg - joint_entropy(sub)


def condition_on(joint: Joint, assignment: Dict[int, int]) -> Joint:
    """Condition the joint on position->value assignments, returning the
    renormalised posterior over the *same* config length (assigned positions
    become deterministic). Raises if the assignment has zero probability."""
    kept = {cfg: p for cfg, p in joint.items()
            if all(cfg[i] == v for i, v in assignment.items())}
    if not kept:
        raise ValueError(f"assignment {assignment} has zero probability")
    return normalise(kept)


def conditional_distribution(joint: Joint, i: int,
                             given: Dict[int, int]) -> Dict[int, float]:
    """p(Y_i = . | Y_given = given_values) as {symbol: prob}.

    This is the exact any-order conditional an oracle backend exposes. `given`
    must not include i.
    """
    if i in given:
        raise ValueError("cannot condition position i on itself")
    num: Dict[int, float] = defaultdict(float)
    denom = 0.0
    for cfg, p in joint.items():
        if all(cfg[j] == v for j, v in given.items()):
            num[cfg[i]] += p
            denom += p
    if denom <= _EPS:
        raise ValueError(f"conditioning set {given} has zero probability")
    return {s: v / denom for s, v in num.items()}


def conditional_total_correlation(joint: Joint, positions: Sequence[int],
                                  given: Dict[int, int]) -> float:
    """C(positions | Y_given = values), computed exactly by first conditioning
    the explicit joint on `given` and then taking the total correlation of
    `positions` in the posterior."""
    post = condition_on(joint, given)
    return total_correlation(post, positions)


# ---- combinatorial helpers (avoid math.comb edge issues, keep it explicit) ----

def log2_factorial(n: int) -> float:
    """log2(n!) computed via lgamma for numerical stability at larger n."""
    return math.lgamma(n + 1) / _LOG2


def log2_binom(n: int, k: int) -> float:
    """log2(C(n, k))."""
    if k < 0 or k > n:
        return float("-inf")
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / _LOG2
