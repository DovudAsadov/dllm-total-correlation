"""
Synthetic task posteriors, as *explicit* joint distributions over the answer.

Each task exposes:
  - build(n, ...) -> Joint          the exact posterior p(Y | prompt) over n positions
  - analytic_C(n, ...) -> float     the closed-form total correlation, in bits

The closed forms are what the ground-truth engine and the meter are validated
against. They were derived from scratch and cross-checked against the values
ParallelBench (arXiv:2510.04767) reports; where the shipped scorer differs from
the paper's Eq. 4 (Shuffle excluding the identity permutation) we implement BOTH
and expose both analytic values.

IMPORTANT (the correction that reshaped the project): the Replace-Random total
correlation is

    C = (n-1) * (log2 n - log2 (n-1))  ->  log2 e ~= 1.4427 bits

NOT ((n-1)/n)^(n-1) -> 1/e. The latter is the one-step parallel-decoding
*accuracy*, a different quantity. Everything is validated against the bits value.

The star of the paper is `matched_marginal_pair`: two tasks with provably
identical per-position marginals but C = 0 vs C > 0, so any proxy that is a
function of the marginals alone is analytically forced to score them identically.
"""
from __future__ import annotations

import itertools
from typing import Dict, Tuple

from .information import (
    Joint,
    log2_binom,
    log2_factorial,
)

# ---------------------------------------------------------------------------
# Permutation / assignment family
# ---------------------------------------------------------------------------


def shuffle(n: int, exclude_identity: bool = False) -> Joint:
    """Answer is a uniform random permutation of n distinct symbols {0..n-1}.

    C = n*log2(n) - log2(n!).  If exclude_identity, the identity permutation is
    removed from the support (this is what ParallelBench's shipped
    list_shuffle_score does, and it changes C).
    """
    perms = list(itertools.permutations(range(n)))
    if exclude_identity:
        identity = tuple(range(n))
        perms = [p for p in perms if p != identity]
    p = 1.0 / len(perms)
    return {perm: p for perm in perms}


def analytic_C_shuffle(n: int, exclude_identity: bool = False) -> float:
    if not exclude_identity:
        return n * (__import__("math").log2(n)) - log2_factorial(n)
    # exclude identity: compute marginals exactly from counts.
    import math

    total = math.factorial(n) - 1
    # count of permutations (excluding identity) with Y_i = j:
    #   off-diagonal (i != j): (n-1)!         (identity never has Y_i=j for i!=j)
    #   diagonal   (i == j):   (n-1)! - 1     (identity removed)
    fac_nm1 = math.factorial(n - 1)
    # marginal entropy of one position (all positions symmetric)
    p_diag = (fac_nm1 - 1) / total
    p_off = fac_nm1 / total
    def h(x):
        return 0.0 if x <= 0 else -x * math.log2(x)
    # one diagonal value + (n-1) off-diagonal values
    H_marg = h(p_diag) + (n - 1) * h(p_off)
    sum_marg = n * H_marg
    H_joint = math.log2(total)
    return sum_marg - H_joint


def replace_random(n: int) -> Joint:
    """Exactly-one-replace: a length-n binary vector with exactly one 1 (the
    replaced position), position uniform over {0..n-1}. One-hot, uniform.

    C = (n-1)*(log2 n - log2(n-1)) -> log2 e.
    """
    configs: Joint = {}
    p = 1.0 / n
    for r in range(n):
        cfg = tuple(1 if k == r else 0 for k in range(n))
        configs[cfg] = p
    return configs


def analytic_C_replace_random(n: int) -> float:
    import math

    return (n - 1) * (math.log2(n) - math.log2(n - 1))


def replace_index(n: int, index: int = 0) -> Joint:
    """Replace a *specified* index -> deterministic answer -> point mass.

    C = 0, but the task is non-trivial (it has a prompt-dependent structure),
    so it is a second zero-dependency control distinct from Copy.
    """
    cfg = tuple(1 if k == index else 0 for k in range(n))
    return {cfg: 1.0}


def copy(n: int) -> Joint:
    """Copy the input -> deterministic -> point mass. C = 0."""
    return {tuple(range(n)): 1.0}


# ---------------------------------------------------------------------------
# The matched-marginal pair -- the decisive construction
# ---------------------------------------------------------------------------


def free_coins(n: int) -> Joint:
    """n independent fair coins. Marginals all Bernoulli(1/2). C = 0."""
    p = 1.0 / (2 ** n)
    return {cfg: p for cfg in itertools.product((0, 1), repeat=n)}


def constrained_coins(n: int, k: int) -> Joint:
    """n coins conditioned on exactly k heads, uniform over the C(n,k) configs.

    For k = n/2 the marginals are EXACTLY Bernoulli(1/2) -- identical to
    free_coins -- yet C = n*Hb(k/n) - log2 C(n,k) > 0.

    For n=4, k=2:  C = 4 - log2(6) = 1.41504 bits, marginals all 1/2.
    """
    configs = [cfg for cfg in itertools.product((0, 1), repeat=n) if sum(cfg) == k]
    p = 1.0 / len(configs)
    return {cfg: p for cfg in configs}


def analytic_C_constrained_coins(n: int, k: int) -> float:
    import math

    frac = k / n
    def hb(x):
        if x <= 0 or x >= 1:
            return 0.0
        return -x * math.log2(x) - (1 - x) * math.log2(1 - x)
    return n * hb(frac) - log2_binom(n, k)


def matched_marginal_pair(n: int = 4) -> Tuple[Joint, Joint, Dict[str, float]]:
    """Return (free, constrained, info) for the matched-marginal experiment.

    Requires n even so that k = n/2 gives marginals of exactly 1/2 in both.
    `info` records the analytic C of each and the per-position marginal of each
    (which must be identical) so the 'provably identical marginals' claim is
    checkable in code, not just on paper.
    """
    if n % 2 != 0:
        raise ValueError("matched_marginal_pair needs even n so k=n/2 gives 1/2 marginals")
    k = n // 2
    free = free_coins(n)
    cons = constrained_coins(n, k)
    info = {
        "n": n,
        "k": k,
        "C_free": 0.0,
        "C_constrained": analytic_C_constrained_coins(n, k),
        "marginal_free_p1": 0.5,
        "marginal_constrained_p1": k / n,  # == 0.5 when k = n/2
    }
    return free, cons, info


# ---------------------------------------------------------------------------
# Graded family -- a continuum of C for the correlation study
# ---------------------------------------------------------------------------


def graded_pair_block(rho: float) -> Joint:
    """A 2-bit block with a tunable dependency.

    Two symmetric bits with P(00)=P(11) = rho/2 and P(01)=P(10) = (1-rho)/2.
    Marginals are exactly (1/2, 1/2) for ALL rho, so this block's marginal
    entropy is fixed at 1 bit per position regardless of dependency -- another
    matched-marginal object, now with a *continuous* C from 0 (rho=1/2) up to
    1 bit (rho in {0,1}).

    C(rho) = 2*1 - H(joint) = 2 - [Hb(rho)] where the joint has two outcomes of
    mass rho/2,rho/2 and two of (1-rho)/2 ... (see analytic_C_graded_block).
    """
    if not (0.0 <= rho <= 1.0):
        raise ValueError("rho must be in [0,1]")
    return {
        (0, 0): rho / 2,
        (1, 1): rho / 2,
        (0, 1): (1 - rho) / 2,
        (1, 0): (1 - rho) / 2,
    }


def analytic_C_graded_block(rho: float) -> float:
    import math

    def h(x):
        return 0.0 if x <= 0 else -x * math.log2(x)
    # joint entropy: two masses rho/2, two masses (1-rho)/2
    H_joint = 2 * h(rho / 2) + 2 * h((1 - rho) / 2)
    sum_marg = 2 * 1.0  # each position is fair -> 1 bit
    return sum_marg - H_joint


# Registry for convenience / iteration in experiments.
TASKS = {
    "copy": copy,
    "replace_index": replace_index,
    "replace_random": replace_random,
    "shuffle": shuffle,
    "shuffle_no_identity": lambda n: shuffle(n, exclude_identity=True),
    "free_coins": free_coins,
}
