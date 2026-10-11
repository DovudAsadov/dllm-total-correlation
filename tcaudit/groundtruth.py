"""
Exact ground-truth total-correlation engine.

Given an explicit task posterior, compute C(S | committed) for EVERY candidate
subset S at EVERY partial decode state, by exact enumeration. Pure CPU, no model,
no network. This is the safety asset of the whole project: it alone guarantees a
defensible artifact and the ground-truth axis of every figure.

Tractable because the synthetic tasks have small support (n <= 6 -> <= 720
configs). Everything is exact; the only approximation anywhere in the paper is
the *meter*, which is validated against this engine.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .information import (
    Joint,
    condition_on,
    conditional_total_correlation,
    total_correlation,
)

Assignment = Dict[int, int]


@dataclass(frozen=True)
class GTRow:
    """One ground-truth record: the true C of committing set S given a partial
    state, plus enough metadata to join against proxy scores."""
    committed: Tuple[Tuple[int, int], ...]  # sorted ((pos,val),...) already decoded
    candidate_set: Tuple[int, ...]          # positions we consider committing now
    C_bits: float                           # true conditional total correlation
    set_size: int


def _remaining_positions(n: int, committed: Assignment) -> List[int]:
    return [i for i in range(n) if i not in committed]


def enumerate_ground_truth(
    joint: Joint,
    max_set_size: int | None = None,
    max_states: int | None = None,
    include_singletons: bool = True,
) -> List[GTRow]:
    """Enumerate C(S | committed) for candidate subsets S of the remaining
    positions, across reachable partial states.

    A "state" is an assignment of a subset of positions to values that has
    non-zero probability under the joint. For each state we consider candidate
    subsets S of the still-masked positions (up to max_set_size).

    To keep the enumeration bounded we walk states in increasing number of
    committed positions and, at each, sample candidate subsets of the remainder.
    For the small tasks here the full walk is cheap; `max_states` caps it for
    safety on larger supports.
    """
    joint = {k: v for k, v in joint.items()}
    n = len(next(iter(joint)))
    rows: List[GTRow] = []
    seen_states = 0

    # Reachable partial states: every prefix-agnostic assignment that some config
    # satisfies. We enumerate by choosing a set of committed positions and a
    # config, then reading off that config's values on those positions.
    # Deduplicate identical (committed positions, values) states.
    state_keys = set()

    committed_counts = range(0, n)  # commit 0..n-1 positions, leave >=1 masked
    for c in committed_counts:
        for committed_positions in itertools.combinations(range(n), c):
            # gather the distinct value-assignments seen on these positions
            distinct = set()
            for cfg in joint:
                distinct.add(tuple(cfg[i] for i in committed_positions))
            for vals in distinct:
                assignment: Assignment = dict(zip(committed_positions, vals))
                key = tuple(sorted(assignment.items()))
                if key in state_keys:
                    continue
                # must be a reachable (nonzero) state
                try:
                    post = condition_on(joint, assignment)
                except ValueError:
                    continue
                state_keys.add(key)
                seen_states += 1

                remaining = _remaining_positions(n, assignment)
                if not remaining:
                    continue
                upper = len(remaining) if max_set_size is None else min(max_set_size, len(remaining))
                lo = 1 if include_singletons else 2
                for size in range(lo, upper + 1):
                    for S in itertools.combinations(remaining, size):
                        C = total_correlation(post, list(S))
                        rows.append(
                            GTRow(
                                committed=key,
                                candidate_set=tuple(S),
                                C_bits=C,
                                set_size=size,
                            )
                        )
                if max_states is not None and seen_states >= max_states:
                    return rows
    return rows


def full_answer_C(joint: Joint) -> float:
    """C of committing ALL positions at once from the empty state -- the single
    number the task's closed form predicts."""
    return total_correlation(joint)


def C_given(joint: Joint, positions: Sequence[int], given: Assignment) -> float:
    """Convenience passthrough for a single (S, committed) query."""
    if not given:
        return total_correlation(joint, list(positions))
    return conditional_total_correlation(joint, list(positions), given)
