"""
Build audit samples from tasks, WITHOUT ever materialising a giant joint.

Sources:
  1. from_task(...)     : semantic ParallelBench-style task -> enumerate reachable
                          partial states and candidate subsets, attach exact C,
                          prepare per-position distributions by conditioning.
  2. graded_samples(...): a controlled continuum of C in [0,1] built from many
                          INDEPENDENT tiny 2-bit blocks (each its own small joint),
                          every position fair, so marginal proxies are starved of
                          signal by construction. No Kronecker product.

Three sample sets are exported, and they must not be confused:
  natural_audit_samples() : (1) only  -- the observational population.
  graded_samples()        : (2) only  -- the confound-removed control.
  pooled_audit_samples()  : (1) + (2) -- both together; NOT "natural".
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .audit import PreparedSample, Sample, prepare_from_joint
from .groundtruth import enumerate_ground_truth
from .information import Joint, marginal, total_correlation
from . import tasks


def from_task(name: str, n: int, max_set_size: int | None = None,
              max_samples: int | None = None) -> Tuple[Joint, List[PreparedSample]]:
    builder = tasks.TASKS[name]
    joint = builder(n)
    rows = enumerate_ground_truth(joint, max_set_size=max_set_size)
    samples = [
        Sample(context=dict(r.committed), candidate_set=r.candidate_set,
               C_true=r.C_bits, task=name)
        for r in rows if r.set_size >= 2
    ]
    if max_samples is not None and len(samples) > max_samples:
        idx = np.linspace(0, len(samples) - 1, max_samples).astype(int)
        samples = [samples[i] for i in idx]
    return joint, prepare_from_joint(joint, samples)


def graded_samples(rhos: Sequence[float] | None = None) -> List[PreparedSample]:
    """One PreparedSample per rho: a 2-bit block whose two positions are both
    fair (marginal entropy pinned at 1 bit) but whose total correlation sweeps
    from 0 (rho=0.5) up to 1 bit (rho->0). Each block is a standalone tiny joint,
    so there is no exponential blow-up."""
    if rhos is None:
        rhos = list(np.linspace(0.0, 0.5, 21))
    prepared: List[PreparedSample] = []
    for r in rhos:
        j = tasks.graded_pair_block(float(r))
        dists = [marginal(j, 0), marginal(j, 1)]
        C = total_correlation(j)
        prepared.append(PreparedSample(dists=dists, C_true=C,
                                       task=f"graded[rho={r:.3f}]"))
    return prepared


NATURAL_TASKS = ("replace_random", "shuffle", "shuffle_no_identity")


def natural_audit_samples(n_perm: int = 4) -> List[PreparedSample]:
    """The NATURAL sample set: semantic ParallelBench-style tasks only
    (replace_random + shuffle, with and without the identity permutation).

    This is deliberately WITHOUT the graded family. The graded blocks are the
    confound-REMOVED control -- every position is pinned to a fair bit, so every
    marginal proxy is constant on them by construction. Pooling them with the
    natural tasks mixes a control into the very population whose statistics are
    being reported, so any "natural-task" number must be computed here, not on
    pooled_audit_samples()."""
    prepared: List[PreparedSample] = []
    for name in NATURAL_TASKS:
        _, ps = from_task(name, n_perm, max_set_size=n_perm)
        prepared += ps
    return prepared


def pooled_audit_samples(n_perm: int = 4) -> List[PreparedSample]:
    """Headline pooled sample set: graded continuum + replace_random + shuffle
    (both identity variants). Gives a wide C range with strong external validity.

    NOTE: this pool mixes the graded control into the natural tasks. Anything
    reported as a property of *natural* tasks must use natural_audit_samples()."""
    prepared: List[PreparedSample] = []
    prepared += graded_samples()
    prepared += natural_audit_samples(n_perm=n_perm)
    return prepared
