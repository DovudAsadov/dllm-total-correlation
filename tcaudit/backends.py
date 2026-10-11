"""
Conditional backends: the interface the meter talks to.

A backend answers exactly one kind of question:

    conditional(i, given) -> {symbol: prob}

i.e. p(Y_i = . | Y_given = values, context). The meter is written entirely
against this interface, so the SAME meter code runs on:

  * OracleBackend      -- exact any-order conditionals from an explicit joint,
                          zero learning error. Used to validate the meter and to
                          run the whole CPU-only audit (contributions 1, 2, 4).

  * ModelBackend       -- a real masked-diffusion LM (LLaDA / Dream) whose
                          any-order conditionals come from forward passes with
                          the queried positions masked and `given` positions
                          filled. Used for the confirmation study (contribution 3,
                          needs the GPU). Implemented in model_backend.py so this
                          module has no torch dependency.

Keeping the interface this narrow is the point: it makes "does the meter recover
analytic C?" answerable with no neural network, so a broken estimator is caught
on CPU before it ever touches the Spark.
"""
from __future__ import annotations

from typing import Dict, Protocol

from .information import Joint, conditional_distribution


class Backend(Protocol):
    vocab_at: Dict[int, tuple]

    def conditional(self, i: int, given: Dict[int, int]) -> Dict[int, float]:
        ...

    def marginals(self, positions) -> Dict[int, Dict[int, float]]:
        ...


class OracleBackend:
    """Exact conditionals from an explicit joint distribution."""

    def __init__(self, joint: Joint):
        self.joint = joint
        any_cfg = next(iter(joint))
        self.n = len(any_cfg)

    def conditional(self, i: int, given: Dict[int, int]) -> Dict[int, float]:
        """Exact p(Y_i | Y_given). `given` may be empty (returns the marginal)."""
        if not given:
            # marginal of i
            from collections import defaultdict

            m: Dict[int, float] = defaultdict(float)
            for cfg, p in self.joint.items():
                m[cfg[i]] += p
            z = sum(m.values())
            return {s: v / z for s, v in m.items()}
        return conditional_distribution(self.joint, i, given)

    def marginals(self, positions) -> Dict[int, Dict[int, float]]:
        """Per-position marginals for the given positions (empty context)."""
        return {i: self.conditional(i, {}) for i in positions}
