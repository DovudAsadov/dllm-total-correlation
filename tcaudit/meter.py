"""
The total-correlation meter.

Estimates C(S | context) = sum_{i in S} H(Y_i | context) - H(Y_S | context)
for a candidate set S, using ONLY a backend's any-order conditionals.

The marginal term is easy (one distribution per position). The hard term is the
joint entropy H(Y_S). We estimate it with an order-averaged chain-rule Monte
Carlo estimator:

    H(Y_S) = E_pi [ sum_t H_theta( Y_{pi(t)} | y_{pi(<t)} ) ]

For a *consistent* joint, the chain rule makes the inner sum exactly H(Y_S) for
ANY fixed order pi, so averaging over orders only reduces variance. For an
any-order model whose conditionals may be mutually inconsistent (a real dLLM),
the order-average is the natural symmetric estimand, and its order-variance is
itself a reportable diagnostic of conditional inconsistency.

Per draw m with order pi_m:
  1. walk the positions of S in order pi_m
  2. at step t, query the backend for p(Y_{pi(t)} | already-sampled prefix)
  3. add the EXACT entropy of that predicted categorical (we have the full
     distribution -- we do not Monte-Carlo the entropy itself)
  4. sample a value for Y_{pi(t)} from that categorical to extend the prefix

Two NFE counters are threaded through from the start (decode vs meter) because on
a real model every conditional query costs forward passes and the paper's headline
metric must never be ambiguous about which forward passes were decoding and which
were measurement.

Validation contract: with an OracleBackend, meter_C must equal analytic C within
Monte-Carlo error. That test runs on CPU with no neural network.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .backends import Backend
from .information import entropy_bits


@dataclass
class NFECounter:
    decode: int = 0
    meter: int = 0

    def add_meter(self, k: int = 1):
        self.meter += k

    def add_decode(self, k: int = 1):
        self.decode += k


@dataclass
class MeterResult:
    C_bits: float
    sum_marginal_bits: float
    joint_entropy_bits: float
    order_std_bits: float           # std of the per-order joint-entropy estimates
    n_orders: int
    meter_nfe: int
    per_position_marginal_entropy: Dict[int, float] = field(default_factory=dict)


def _entropy_of(dist: Dict[int, float]) -> float:
    return entropy_bits(dist.values())


def _sample_from(dist: Dict[int, float], rng: np.random.Generator) -> int:
    syms = list(dist.keys())
    probs = np.array([dist[s] for s in syms], dtype=float)
    probs = probs / probs.sum()
    return int(syms[rng.choice(len(syms), p=probs)])


def meter_total_correlation(
    backend: Backend,
    candidate_set: Sequence[int],
    context: Dict[int, int] | None = None,
    n_orders: int = 16,
    rng: np.random.Generator | None = None,
    nfe: NFECounter | None = None,
) -> MeterResult:
    """Estimate C(candidate_set | context) via order-averaged chain-rule MC.

    `context` is the already-committed assignment (empty for the top state).
    Each conditional query is counted as one meter NFE on the shared counter
    (on the oracle this is bookkeeping; on a real model it is a real cost).
    """
    S = list(candidate_set)
    context = dict(context or {})
    rng = rng if rng is not None else np.random.default_rng(0)
    nfe = nfe if nfe is not None else NFECounter()

    if len(S) == 0:
        return MeterResult(0.0, 0.0, 0.0, 0.0, n_orders, 0)
    if len(S) == 1:
        d = backend.conditional(S[0], context)
        nfe.add_meter(1)
        h = _entropy_of(d)
        return MeterResult(0.0, h, h, 0.0, n_orders, 1,
                           per_position_marginal_entropy={S[0]: h})

    # --- marginal term: H(Y_i | context) for each i in S ---
    marg_H: Dict[int, float] = {}
    for i in S:
        d = backend.conditional(i, context)
        nfe.add_meter(1)
        marg_H[i] = _entropy_of(d)
    sum_marg = sum(marg_H.values())

    # --- joint term: order-averaged chain rule ---
    order_estimates: List[float] = []
    for _ in range(n_orders):
        order = list(S)
        rng.shuffle(order)
        prefix = dict(context)
        h_sum = 0.0
        for pos in order:
            d = backend.conditional(pos, prefix)
            nfe.add_meter(1)
            h_sum += _entropy_of(d)
            prefix[pos] = _sample_from(d, rng)
        order_estimates.append(h_sum)

    joint_H = float(np.mean(order_estimates))
    order_std = float(np.std(order_estimates))
    C = sum_marg - joint_H
    # C is >= 0 in theory; small negative values can appear from MC noise or
    # from an inconsistent model. We report the raw value (clipping would hide
    # exactly the inconsistency we want to measure).
    return MeterResult(
        C_bits=C,
        sum_marginal_bits=sum_marg,
        joint_entropy_bits=joint_H,
        order_std_bits=order_std,
        n_orders=n_orders,
        meter_nfe=nfe.meter,
        per_position_marginal_entropy=marg_H,
    )


def meter_C_curve_over_orders(
    backend: Backend,
    candidate_set: Sequence[int],
    context: Dict[int, int] | None = None,
    order_grid: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
    rng: np.random.Generator | None = None,
) -> Dict[int, float]:
    """Return {n_orders: estimated C} to expose the estimator's convergence.
    Used for the M-vs-variance figure that fixes the M used elsewhere."""
    rng = rng if rng is not None else np.random.default_rng(0)
    out = {}
    for m in order_grid:
        res = meter_total_correlation(backend, candidate_set, context,
                                      n_orders=m, rng=rng)
        out[m] = res.C_bits
    return out
