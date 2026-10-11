"""
Dependency-gated decoding: the constructive counterpart to the audit.

The audit shows confidence cannot see dependency. This shows the payoff of seeing
it. A sequence is decoded step by step; at each step a SELECTOR chooses which
masked positions to commit in parallel (each sampled from its marginal given what
is already committed). We compare:

  * confidence_threshold(tau) : commit every masked position with top prob >= tau
                                (Fast-dLLM / Dynamic-dLLM style).
  * confidence_topk(k)        : commit the k highest-confidence masked positions.
  * meter_gated(tauC)         : greedily grow the commit set in confidence order
                                while its total correlation C(S | committed) stays
                                <= tauC. Parallelises independent positions,
                                serialises coupled ones.

Cost is forward passes (NFE): one decode forward per step, plus, for meter-gated,
the meter forwards used to estimate C (charged separately). Quality is the fraction
of decoded samples that are VALID under the task (in the support of the true
joint) -- the operational correctness a parallel commit preserves or destroys.

TWO CORRECTIONS (see MeterCostModel and the `rng` arguments below):

  * TIE-BREAKING. `sorted(..., reverse=True)` is stable, so when several masked
    positions carry the SAME confidence -- the normal case on a matched-marginal
    workload, where every position is exactly fair -- top-k degenerates into
    "commit the k lowest global indices". That is not a property of confidence
    gating, it is a property of Python's sort interacting with how the sequence
    happens to be laid out. Every selector therefore takes an optional `rng`;
    with an rng, ties are broken uniformly at random. The legacy (rng=None)
    behaviour is kept so the published curve stays reproducible next to the
    corrected one.

  * METER COST. Charging `meter_nfe += 1` per step ("amortized") is a placeholder,
    not a measurement: sel_meter_gated calls C_of up to |masked| times per step,
    and each call evaluates one conditional total correlation per block it
    touches. The decoders now INSTRUMENT those calls (MeterInstrumentation),
    memoize repeated (block, subset) evaluations within a step, and convert the
    counts into forward passes through an explicit, stated MeterCostModel. The
    legacy 1-per-step model is retained *by name* so it is obvious when it is
    being used. Reported speedups should be DECODE-NFE only, with the exclusion
    of metering cost stated.

The selector talks to the world through two callables only:
  conf[i]       -> the top probability at masked position i (for ordering/threshold)
  C_of(S)       -> the true conditional total correlation of committing set S now
so the SAME selectors run on a single-task joint and on a block-structured
sequence. On the exact oracle C_of is exact: this is the CEILING result -- if you
could see dependency, gating on it dominates gating on confidence. Estimating C
cheaply on a real model is the open problem the paper names.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .information import (
    Joint,
    conditional_distribution,
    conditional_total_correlation,
    marginal,
    total_correlation,
)

Assignment = Dict[int, int]


# ----------------------------- selectors -----------------------------

def _confidence_order(masked: Sequence[int], conf: Dict[int, float], rng) -> List[int]:
    """Masked positions sorted by decreasing confidence.

    rng is None  -> legacy behaviour: Python's stable sort leaves equal-confidence
                    positions in `masked` order, i.e. ascending global index.
    rng given    -> ties broken uniformly at random (an independent U[0,1) key per
                    position per call), so the selector cannot read the layout.
    """
    if rng is None:
        return sorted(masked, key=lambda i: conf[i], reverse=True)
    keys = rng.random(len(masked))
    idx = sorted(range(len(masked)), key=lambda t: (-conf[masked[t]], keys[t]))
    return [masked[t] for t in idx]


def sel_confidence_threshold(tau: float, rng=None):
    def f(masked, conf, C_of):
        chosen = [i for i in masked if conf[i] >= tau]
        if chosen:
            return chosen
        return [_confidence_order(masked, conf, rng)[0]]
    f.name = f"conf_thresh({tau:.2f})" + ("" if rng is None else "+tb")
    return f


def sel_confidence_topk(k: int, rng=None):
    def f(masked, conf, C_of):
        return _confidence_order(masked, conf, rng)[:max(1, k)]
    f.name = f"conf_topk({k})" + ("" if rng is None else "+tb")
    return f


def sel_meter_gated(tauC: float, rng=None):
    def f(masked, conf, C_of):
        order = _confidence_order(masked, conf, rng)
        chosen = [order[0]]
        for cand in order[1:]:
            if C_of(chosen + [cand]) <= tauC:
                chosen = chosen + [cand]
        return chosen
    f.name = f"meter_gated({tauC:.2f})" + ("" if rng is None else "+tb")
    return f


# ------------------------- meter cost instrumentation -------------------------

@dataclass
class MeterInstrumentation:
    """What the meter-gated selector ACTUALLY asked for, per decoded sequence.

    `raw` counts every underlying conditional-total-correlation evaluation the
    selector triggered; `memo` counts the DISTINCT (block, subset) evaluations
    after per-step memoization (the committed context is fixed within a step, so
    memoizing within a step is exact). `positions_*` sum |subset| over those
    evaluations, which is what a chain-rule meter's cost actually scales with.
    """
    steps: int = 0
    c_of_calls: int = 0
    tc_evals_raw: int = 0
    tc_evals_memo: int = 0
    tc_positions_raw: int = 0
    tc_positions_memo: int = 0

    def add(self, other: "MeterInstrumentation") -> None:
        self.steps += other.steps
        self.c_of_calls += other.c_of_calls
        self.tc_evals_raw += other.tc_evals_raw
        self.tc_evals_memo += other.tc_evals_memo
        self.tc_positions_raw += other.tc_positions_raw
        self.tc_positions_memo += other.tc_positions_memo

    def scaled(self, d: float) -> Dict[str, float]:
        return {"steps": self.steps / d, "c_of_calls": self.c_of_calls / d,
                "tc_evals_raw": self.tc_evals_raw / d,
                "tc_evals_memo": self.tc_evals_memo / d,
                "tc_positions_raw": self.tc_positions_raw / d,
                "tc_positions_memo": self.tc_positions_memo / d}


@dataclass
class MeterCostModel:
    """An EXPLICIT, stated conversion from meter queries to forward passes.

    cost = per_step * steps
         + per_c_call * (#C_of calls)
         + per_tc_eval * (#distinct C evaluations)
         + n_orders * per_order_per_position * (sum of |S| over those evaluations)

    Named instances below; every result that reports a total NFE must also report
    which of these produced it.
    """
    name: str
    per_step: float = 0.0
    per_c_call: float = 0.0
    per_tc_eval: float = 0.0
    per_order_per_position: float = 0.0
    n_orders: int = 0
    memoized: bool = True
    note: str = ""

    def cost(self, instr: MeterInstrumentation) -> float:
        evals = instr.tc_evals_memo if self.memoized else instr.tc_evals_raw
        poss = instr.tc_positions_memo if self.memoized else instr.tc_positions_raw
        return (self.per_step * instr.steps
                + self.per_c_call * instr.c_of_calls
                + self.per_tc_eval * evals
                + self.n_orders * self.per_order_per_position * poss)

    def describe(self) -> Dict:
        return {"name": self.name, "per_step": self.per_step,
                "per_c_call": self.per_c_call, "per_tc_eval": self.per_tc_eval,
                "per_order_per_position": self.per_order_per_position,
                "n_orders": self.n_orders, "memoized": self.memoized,
                "note": self.note}


# The number the published code charged. Kept only so the old figure is
# reproducible; it is a placeholder, not a measurement.
COST_LEGACY_AMORTIZED = MeterCostModel(
    "legacy_amortized_1", per_step=1.0,
    note="one forward per decode step regardless of how many C queries were made; "
         "unjustified -- the selector makes O(|masked|) queries per step")

# One forward pass per distinct C evaluation (optimistic: assumes a single
# forward can answer a whole conditional-total-correlation query).
COST_ONE_FWD_PER_C = MeterCostModel(
    "one_forward_per_distinct_C", per_tc_eval=1.0,
    note="optimistic lower bound: one forward pass per memoized (block, subset) "
         "C evaluation, i.e. C is assumed readable in a single pass")


def cost_chain_rule(n_orders: int, memoized: bool = True) -> MeterCostModel:
    """Cost of estimating each C with tcaudit.meter's order-averaged chain rule:
    one (batched) forward for the |S| marginals plus one forward per position per
    sampled order."""
    return MeterCostModel(
        f"chain_rule_M{n_orders}" + ("" if memoized else "_unmemoized"),
        per_tc_eval=1.0, per_order_per_position=1.0, n_orders=n_orders,
        memoized=memoized,
        note="tcaudit.meter cost: 1 batched forward for the marginals of S plus "
             f"M={n_orders} sampled orders x |S| conditional forwards, per C query")


DEFAULT_COST_MODELS = (COST_LEGACY_AMORTIZED, COST_ONE_FWD_PER_C,
                       cost_chain_rule(16), cost_chain_rule(6000))


# --------------------------- single-task decode ---------------------------

def _sample(dist: Dict[int, float], rng) -> int:
    syms = list(dist.keys())
    p = np.array([dist[s] for s in syms], float)
    p = p / p.sum()
    return int(syms[rng.choice(len(syms), p=p)])


def _cond_or_marginal(joint: Joint, i: int, committed: Assignment) -> Dict[int, float]:
    """Conditional given `committed`, falling back to the unconditional marginal
    when `committed` is off-support. An off-support committed state means an
    earlier parallel commit already produced an invalid sample; we still need to
    fill the rest to return a (correctly-invalid) full sample rather than crash."""
    if not committed:
        return marginal(joint, i)
    try:
        return conditional_distribution(joint, i, committed)
    except ValueError:
        return marginal(joint, i)


@dataclass
class DecodeResult:
    sample: Tuple[int, ...]
    steps: int
    decode_nfe: int
    meter_nfe: float
    instr: MeterInstrumentation = field(default_factory=MeterInstrumentation)


def decode_task(joint: Joint, selector, rng,
                meter_cost_per_eval: int = 1,
                cost_model: MeterCostModel | None = None) -> DecodeResult:
    n = len(next(iter(joint)))
    committed: Assignment = {}
    steps = decode_nfe = 0
    is_meter = selector.name.startswith("meter")
    instr = MeterInstrumentation()
    while len(committed) < n:
        masked = [i for i in range(n) if i not in committed]
        marg = {i: _cond_or_marginal(joint, i, committed) for i in masked}
        conf = {i: max(marg[i].values()) for i in masked}
        decode_nfe += 1

        memo: Dict[Tuple[int, ...], float] = {}   # exact within a step

        def C_of(S, _committed=committed, _memo=memo):
            instr.c_of_calls += 1
            key = tuple(sorted(S))
            if len(key) < 2:
                return 0.0
            instr.tc_evals_raw += 1
            instr.tc_positions_raw += len(key)
            if key in _memo:
                return _memo[key]
            instr.tc_evals_memo += 1
            instr.tc_positions_memo += len(key)
            if not _committed:
                v = total_correlation(joint, list(key))
            else:
                try:
                    v = conditional_total_correlation(joint, list(key), _committed)
                except ValueError:
                    v = float("inf")  # off-support: never parallelize further
            _memo[key] = v
            return v

        chosen = selector(masked, conf, C_of if is_meter else (lambda S: 0.0))
        for i in chosen:
            committed[i] = _sample(marg[i], rng)
        steps += 1
    instr.steps = steps
    if not is_meter:
        meter_nfe: float = 0.0
    elif cost_model is not None:
        meter_nfe = cost_model.cost(instr)
    else:
        meter_nfe = float(meter_cost_per_eval) * steps  # legacy amortized default
    return DecodeResult(tuple(committed[i] for i in range(n)), steps,
                        decode_nfe, meter_nfe, instr)


def valid_fn_for(task_name: str, n: int) -> Callable[[Tuple[int, ...]], bool]:
    if task_name == "constrained_coins":
        k = n // 2
        return lambda s: sum(s) == k
    if task_name == "replace_random":
        return lambda s: sum(s) == 1
    if task_name == "shuffle":
        return lambda s: len(set(s)) == n
    if task_name == "free_coins":
        return lambda s: True
    raise ValueError(task_name)


def _cost_block(instr: MeterInstrumentation, trials: int, mean_decode_nfe: float,
                cost_models) -> Dict:
    """Per-model mean meter NFE and mean total NFE, each tagged with the model."""
    out = {}
    for m in cost_models:
        mn = m.cost(instr) / trials
        out[m.name] = {"mean_meter_nfe": mn,
                       "mean_total_nfe": mean_decode_nfe + mn,
                       "model": m.describe()}
    return out


def evaluate_task(joint: Joint, task_name: str, n: int, selector,
                  trials: int = 4000, seed: int = 0,
                  cost_models=DEFAULT_COST_MODELS) -> Dict:
    rng = np.random.default_rng(seed)
    valid = valid_fn_for(task_name, n)
    ok = dnfe = stp = 0
    mnfe = 0.0
    instr = MeterInstrumentation()
    for _ in range(trials):
        r = decode_task(joint, selector, rng)
        ok += int(valid(r.sample))
        dnfe += r.decode_nfe
        mnfe += r.meter_nfe
        stp += r.steps
        instr.add(r.instr)
    mean_decode = dnfe / trials
    is_meter = selector.name.startswith("meter")
    return {"valid_rate": ok / trials, "mean_decode_nfe": mean_decode,
            "mean_meter_nfe": mnfe / trials,
            "mean_total_nfe": (dnfe + mnfe) / trials, "mean_steps": stp / trials,
            "meter_calls_per_decode": instr.scaled(trials),
            "meter_nfe_by_cost_model": (
                _cost_block(instr, trials, mean_decode, cost_models)
                if is_meter else {})}


# ------------------------- block-structured decode -------------------------
# A sequence of independent blocks with local dependence. Global position i lives
# in exactly one block; C decomposes as a sum over blocks, so no Kronecker product.

@dataclass
class Block:
    joint: Joint                 # local joint over this block's positions
    positions: Tuple[int, ...]   # global indices, in local order
    valid: Callable              # predicate on the block's local sample tuple


def make_free_block(base: int) -> Block:
    from . import tasks
    return Block(tasks.graded_pair_block(0.5), (base, base + 1), lambda s: True)


def make_parity_block(base: int) -> Block:
    # perfectly coupled: support {(0,0),(1,1)}, marginals fair, C = 1 bit
    from . import tasks
    return Block(tasks.graded_pair_block(1.0), (base, base + 1),
                 lambda s: s[0] == s[1])


def decode_blocks(blocks: List[Block], selector, rng,
                  meter_cost_per_eval: int = 1,
                  cost_model: MeterCostModel | None = None) -> DecodeResult:
    is_meter = selector.name.startswith("meter")
    instr = MeterInstrumentation()
    # map global pos -> (block index, local index)
    loc = {}
    for bi, b in enumerate(blocks):
        for li, gp in enumerate(b.positions):
            loc[gp] = (bi, li)
    committed_local: List[Assignment] = [dict() for _ in blocks]
    all_positions = [gp for b in blocks for gp in b.positions]

    steps = decode_nfe = 0
    committed_global = set()
    while len(committed_global) < len(all_positions):
        masked = [gp for gp in all_positions if gp not in committed_global]
        marg = {}
        for gp in masked:
            bi, li = loc[gp]
            b = blocks[bi]
            cl = committed_local[bi]
            marg[gp] = _cond_or_marginal(b.joint, li, cl)
        conf = {gp: max(marg[gp].values()) for gp in masked}
        decode_nfe += 1

        memo: Dict[Tuple[int, Tuple[int, ...]], float] = {}  # exact within a step

        def C_of(S, _memo=memo):
            instr.c_of_calls += 1
            total = 0.0
            byblock: Dict[int, List[int]] = {}
            for gp in S:
                bi, li = loc[gp]
                byblock.setdefault(bi, []).append(li)
            for bi, lis in byblock.items():
                if len(lis) < 2:
                    continue
                key = (bi, tuple(sorted(lis)))
                instr.tc_evals_raw += 1
                instr.tc_positions_raw += len(lis)
                if key in _memo:
                    v = _memo[key]
                else:
                    instr.tc_evals_memo += 1
                    instr.tc_positions_memo += len(lis)
                    cl = committed_local[bi]
                    try:
                        v = (conditional_total_correlation(blocks[bi].joint, lis, cl)
                             if cl else total_correlation(blocks[bi].joint, lis))
                    except ValueError:
                        v = float("inf")
                    _memo[key] = v
                if v == float("inf"):
                    return float("inf")
                total += v
            return total

        chosen = selector(masked, conf, C_of if is_meter else (lambda S: 0.0))
        for gp in chosen:
            bi, li = loc[gp]
            committed_local[bi][li] = _sample(marg[gp], rng)
            committed_global.add(gp)
        steps += 1

    instr.steps = steps
    if not is_meter:
        meter_nfe: float = 0.0
    elif cost_model is not None:
        meter_nfe = cost_model.cost(instr)
    else:
        meter_nfe = float(meter_cost_per_eval) * steps  # legacy amortized default

    # assemble validity
    all_valid = True
    for bi, b in enumerate(blocks):
        s = tuple(committed_local[bi][li] for li in range(len(b.positions)))
        if not b.valid(s):
            all_valid = False
            break
    sample = tuple(committed_local[loc[gp][0]][loc[gp][1]] for gp in all_positions)
    return DecodeResult(sample, steps, decode_nfe, meter_nfe, instr), all_valid


def evaluate_blocks(blocks: List[Block], selector, trials: int = 3000,
                    seed: int = 0, cost_models=DEFAULT_COST_MODELS) -> Dict:
    rng = np.random.default_rng(seed)
    ok = dnfe = stp = 0
    mnfe = 0.0
    instr = MeterInstrumentation()
    for _ in range(trials):
        r, valid = decode_blocks(blocks, selector, rng)
        ok += int(valid)
        dnfe += r.decode_nfe
        mnfe += r.meter_nfe
        stp += r.steps
        instr.add(r.instr)
    mean_decode = dnfe / trials
    is_meter = selector.name.startswith("meter")
    return {"valid_rate": ok / trials, "mean_decode_nfe": mean_decode,
            "mean_meter_nfe": mnfe / trials,
            "mean_total_nfe": (dnfe + mnfe) / trials, "mean_steps": stp / trials,
            "meter_calls_per_decode": instr.scaled(trials),
            "meter_nfe_by_cost_model": (
                _cost_block(instr, trials, mean_decode, cost_models)
                if is_meter else {})}
