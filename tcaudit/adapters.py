"""
Model adapters and the validation gates that must pass before any measurement runs.

WHY THIS FILE EXISTS
--------------------
E5 v1 and v2 read a masked diffusion LM's logits as

    p(token at position p | context)  ==  softmax(raw_logits[:, p])

That contract is FALSE for Dream. Dream is initialised from an autoregressive Qwen and
keeps the AR shift: its own paper (arXiv:2508.15487, Sec. 4.1) says "the model continues
to utilize hidden state h_i to generate predictions for position i + 1", and its shipped
`generation_utils.py` therefore does

    logits = self(x, attention_mask, tok_idx).logits
    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)   # <-- the shift
    mask_logits = logits[mask_index]

outside the model, in the generation loop. `DreamModel.forward` returns UNSHIFTED logits.
LLaDA (arXiv:2502.09992) has no such shift: `generate.py` reads `model(x).logits` and
indexes `mask_index` directly.

So two checkpoints in the same experiment have OPPOSITE readout conventions, and getting
one wrong silently corrupts every downstream number: marginals, confidence, the
sequential decode, the log-probability deficit, and the any-order spread.

This module therefore never assumes the convention. It MEASURES it (`calibrate_offset`)
and refuses to run if the evidence is weak.

THE CONTRACT
------------
    adapter.token_logprobs(sequences, positions) -> [B, P, V] log-probs, where entry
    [b, j, v] is log p(token at positions[b][j] == v | sequences[b]).

Everything downstream talks to that and nothing else.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

LOG2E = 1.4426950408889634


def degenerate(ids: Sequence[int], max_run: int = 8, max_top_share: float = 0.5) -> bool:
    """A greedy decode dominated by one token. At a wrong offset every checkpoint we
    tested repeated one token for 21 to 32 of 32 positions on prose. At the calibrated
    offset none repeated a token more than three times in a row."""
    ids = list(ids)
    if not ids:
        return True
    run = best = 1
    for a, b in zip(ids, ids[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    top = max(ids.count(t) for t in set(ids)) / len(ids)
    return best > max_run or top > max_top_share


# --------------------------------------------------------------------------- loading

@dataclass
class AdapterReport:
    """Everything we learned about a checkpoint before trusting it. Serialised into the
    results so the paper can print a reproducibility table."""
    model_tag: str = ""
    model_path: str = ""
    loaded_via: str = ""
    torch_dtype: str = ""
    transformers_version: str = ""
    mask_id: Optional[int] = None
    vocab_size: Optional[int] = None
    offset: Optional[int] = None
    offset_evidence: Dict = field(default_factory=dict)
    determinism: Dict = field(default_factory=dict)
    batch_invariance: Dict = field(default_factory=dict)
    fluency_sample: str = ""
    fluency_evidence: Dict = field(default_factory=dict)
    gates_passed: Dict[str, bool] = field(default_factory=dict)

    def all_gates_passed(self) -> bool:
        return bool(self.gates_passed) and all(self.gates_passed.values())


class MDMAdapter:
    """A masked diffusion LM behind a single, position-explicit contract."""

    def __init__(self, model_path: str, model_tag: str, mask_id: Optional[int] = None,
                 offset: Optional[int] = None, device: str = "cuda"):
        self.model_path = model_path
        self.model_tag = model_tag
        self.device = device
        self._forced_offset = offset
        self.offset: Optional[int] = None
        self.report = AdapterReport(model_tag=model_tag, model_path=model_path)
        self.tok, self.model, self.mask_id = self._load(mask_id)
        self.report.mask_id = self.mask_id

    # ---- loading -----------------------------------------------------------

    def _load(self, mask_id: Optional[int]):
        import torch
        import transformers as T
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.report.transformers_version = T.__version__
        tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        arch = (getattr(cfg, "architectures", None) or [None])[0]
        named = getattr(T, arch, None) if arch else None

        attempts: List[str] = []
        model = None
        for cand in [c for c in (named, AutoModel, T.AutoModelForCausalLM) if c is not None]:
            for kw in ({"dtype": torch.bfloat16}, {"torch_dtype": torch.bfloat16}):
                try:
                    m = cand.from_pretrained(self.model_path, trust_remote_code=True, **kw)
                    model = m.to(self.device).eval()
                    self.report.loaded_via = cand.__name__
                    break
                except Exception as e:                      # noqa: BLE001
                    msg = str(e).split("\n")[0][:220]
                    attempts.append(f"{cand.__name__}({list(kw)[0]}): {type(e).__name__}: {msg}")
            if model is not None:
                break
        self.report.offset_evidence["load_attempts"] = attempts
        if model is None:
            raise RuntimeError(
                f"no loader worked for {self.model_path}\n  " + "\n  ".join(attempts))

        self.report.torch_dtype = "bfloat16"
        self.report.vocab_size = int(getattr(cfg, "vocab_size", 0)) or None

        if mask_id is None:
            mask_id = self._resolve_mask_id(tok, model.config)
        return tok, model, int(mask_id)

    @staticmethod
    def _resolve_mask_id(tok, cfg) -> int:
        if os.environ.get("MASK_ID"):
            return int(os.environ["MASK_ID"])
        mid = getattr(tok, "mask_token_id", None)
        if mid is not None:
            return int(mid)
        for obj in (cfg, getattr(cfg, "text_config", None)):
            m = getattr(obj, "mask_token_id", None) if obj is not None else None
            if m is not None:
                return int(m)
        raise ValueError("could not resolve mask_token_id; set MASK_ID explicitly")

    # ---- the contract ------------------------------------------------------

    # forward passes are counted from the start so the cost axis is never ambiguous
    nfe: int = 0
    max_rows: int = 64          # sub-batch cap; full [B,L,V] logits are materialised

    def raw_logits(self, batch_ids: Sequence[Sequence[int]], read_idx: Sequence[Sequence[int]]):
        """Gather raw model logits at *raw tensor indices* (no offset applied).
        Returns log-probs [B, P, V] in float32.

        Sub-batched internally: the backbone materialises [B, L, V] before we gather,
        which at B=512, L=128, V=152k would be ~20 GB. The caller may therefore pass a
        logically large batch (the meter wants nb*M rows) without thinking about memory.
        """
        import torch
        outs = []
        with torch.no_grad():
            for s in range(0, len(batch_ids), self.max_rows):
                ids = batch_ids[s:s + self.max_rows]
                idx = read_idx[s:s + self.max_rows]
                t = torch.tensor(ids, device=self.device)
                out = self.model(input_ids=t)
                lg = out.logits if hasattr(out, "logits") else out[0]
                exp_v = self.report.vocab_size
                bad = (lg.size(-1) != exp_v) if exp_v else (lg.size(-1) < 1000)
                if bad:
                    raise RuntimeError(
                        f"{self.model_tag}: model output last dim is {lg.size(-1)}, "
                        f"expected the vocabulary ({exp_v}). This loader most likely "
                        "returned hidden states rather than logits.")
                ix = torch.tensor(idx, device=self.device)
                g = torch.gather(lg, 1, ix.unsqueeze(-1).expand(-1, -1, lg.size(-1)))
                outs.append(torch.log_softmax(g.float(), dim=-1))
                del lg, out
                type(self).nfe += 1
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)

    def token_logprobs(self, batch_ids: Sequence[Sequence[int]],
                       positions: Sequence[Sequence[int]]):
        """log p(token AT positions[b][j]) -- the only contract downstream code uses."""
        if self.offset is None:
            raise RuntimeError(
                f"{self.model_tag}: offset not calibrated. Call calibrate_offset() first. "
                "Refusing to measure with an unverified readout convention.")
        read = [[max(0, p + self.offset) for p in row] for row in positions]
        return self.raw_logits(batch_ids, read)

    # ---- GATE 1: which index predicts position p? --------------------------

    def calibrate_offset(self, windows: Sequence[Sequence[int]], n_probe: int = 96,
                         k_block: int = 4, seed: int = 0,
                         min_gap_bits: float = 1.0) -> int:
        """Determine the readout convention EMPIRICALLY.

        Probe: mask exactly the positions we intend to read, then score the log-prob the
        model assigns to the tokens that were actually there, reading at index p+delta
        for delta in {-1, 0, +1}. The true tokens are hidden, so a bidirectional model
        cannot win by copying -- only the correct convention scores well.

        Run at k=1 (single hidden token) and at k=k_block (the measurement regime). Both
        must select the same delta, and it must beat the runner-up by `min_gap_bits`.
        """
        import numpy as np
        rng = np.random.default_rng(seed)
        deltas = (-1, 0, 1)
        evidence: Dict[str, Dict[str, float]] = {}

        for k in sorted({1, k_block}):
            seqs, pos, truth = [], [], []
            for w in windows[:n_probe]:
                n = len(w)
                lo, hi = max(2, n // 4), n - k - 2
                if hi <= lo:
                    continue
                s = int(rng.integers(lo, hi))
                block = list(range(s, s + k))
                ids = list(w)
                truth.append([ids[p] for p in block])
                for p in block:
                    ids[p] = self.mask_id
                seqs.append(ids)
                pos.append(block)
            if not seqs:
                raise RuntimeError("no usable probe windows")

            scores: Dict[str, float] = {}
            for d in deltas:
                read = [[max(0, p + d) for p in row] for row in pos]
                lp = self.raw_logits(seqs, read)
                tot, cnt = 0.0, 0
                for b in range(len(seqs)):
                    for j in range(k):
                        tot += float(lp[b, j, truth[b][j]]) * LOG2E
                        cnt += 1
                scores[str(d)] = tot / cnt
            evidence[f"k={k}"] = scores

        # winner must agree across k and beat the runner-up decisively
        winners, gaps = [], []
        for _, sc in evidence.items():
            order = sorted(sc.items(), key=lambda kv: -kv[1])
            winners.append(int(order[0][0]))
            gaps.append(order[0][1] - order[1][1])

        agree = len(set(winners)) == 1
        gap = min(gaps)
        passed = agree and gap >= min_gap_bits
        self.report.offset_evidence = {
            "mean_log2_prob_of_hidden_true_token_by_delta": evidence,
            "winner_per_k": winners, "agree_across_k": agree,
            "min_gap_bits_over_runner_up": gap, "min_gap_required": min_gap_bits,
            "n_probe_windows": len(windows[:n_probe]),
            "probe": ("mask the block, score the hidden true tokens; copying impossible"),
        }
        self.report.gates_passed["offset"] = bool(passed)
        if not passed:
            raise RuntimeError(
                f"{self.model_tag}: offset gate FAILED (winners={winners}, gap={gap:.3f} "
                f"bits, need >={min_gap_bits}). Evidence: {json.dumps(evidence, indent=1)}")

        measured = winners[0]
        if self._forced_offset is not None and self._forced_offset != measured:
            raise RuntimeError(
                f"{self.model_tag}: measured offset {measured} contradicts the forced "
                f"offset {self._forced_offset}. Refusing to proceed.")
        self.offset = measured
        self.report.offset = measured
        return measured

    # ---- GATE 2: is the forward reproducible? ------------------------------

    def check_determinism(self, windows: Sequence[Sequence[int]], n: int = 8,
                          k: int = 4, tol: float = 0.0, batch_blocks: int = 8,
                          max_rows: int = 64,
                          entropy_rms_tol_bits: Optional[float] = None) -> Dict:
        """Two identical calls must give identical logits, the two batch shapes the
        measurement uses must agree on every argmax, and, when `entropy_rms_tol_bits` is
        set, on the summed entropy B. Scoring a sequence alone, a shape the measurement
        never uses, is reported but not gated.

        The estimator reads B from a batch of `batch_blocks` canvases and its first
        chain-rule step from a `max_rows`-row batch of one repeated canvas. Their
        disagreement adds noise to C_hat directly, so it is gated as an RMS over canvases
        of the difference in B, in bits, on infill and suffix canvases alike.

        E5 v2 silently violated this: `disagreement == 1.0` is structurally impossible
        (step 0 of the sequential decode reads the identical tensor at the identical
        position as the parallel pass), yet DiffusionGemma produced it 34 times.
        """
        import torch
        seqs, pos = [], []
        for w in windows[:n]:
            s = len(w) // 2
            ids = list(w)
            for p in range(s, s + k):
                ids[p] = self.mask_id
            seqs.append(ids)
            pos.append(list(range(s, s + k)))

        a = self.raw_logits(seqs, pos)
        b = self.raw_logits(seqs, pos)
        repeat_max = float((a - b).abs().max())

        # every sequence alone vs inside the batch (the submitted version compared one)
        solo = torch.cat([self.raw_logits(seqs[i:i + 1], pos[i:i + 1]) for i in range(len(seqs))])
        batch_max = float((a - solo).abs().max())
        ent = lambda lp: -(lp.exp() * lp).sum(-1) * LOG2E          # noqa: E731
        batch_ent_max = float((ent(a) - ent(solo)).abs().max())

        # does the argmax move? that is what actually corrupts a measurement
        argmax_flips = int((a.argmax(-1) != b.argmax(-1)).sum())
        solo_flips = int((a.argmax(-1) != solo.argmax(-1)).sum())

        # the two batch shapes the estimator combines, on infill and suffix canvases
        ecan, epos = [], []
        for w in windows[:n]:
            s0 = len(w) // 2
            for stop in (s0 + k, len(w)):                      # infill, then suffix
                ids = list(w)
                for p in range(s0, stop):
                    ids[p] = self.mask_id
                ecan.append(ids)
                epos.append(list(range(s0, s0 + k)))
        lp_b = torch.cat([self.raw_logits(ecan[c:c + batch_blocks], epos[c:c + batch_blocks])
                          for c in range(0, len(ecan), batch_blocks)])
        lp_m = torch.cat([self.raw_logits([ecan[i]] * max_rows, [epos[i]] * max_rows)[:1]
                          for i in range(len(ecan))])
        h_b, h_m = ent(lp_b), ent(lp_m)
        est_flips = int((lp_b.argmax(-1) != lp_m.argmax(-1)).sum())
        d_b = (h_b - h_m).sum(-1)
        rms_db = float(d_b.pow(2).mean().sqrt())

        res = {
            "repeat_call_max_abs_logprob_diff": repeat_max,
            "batch1_vs_batchN_max_abs_logprob_diff": batch_max,
            "repeat_call_argmax_flips": argmax_flips,
            "batch_size_argmax_flips": solo_flips,
            "n_positions_compared": int(a.shape[0] * a.shape[1]),
            "n_positions_compared_alone_vs_batch": int(solo.shape[0] * solo.shape[1]),
            "batch1_vs_batchN_max_abs_entropy_diff_bits": batch_ent_max,
            "n_canvases_entropy_check": len(ecan),
            "estimator_batch_shapes": [batch_blocks, max_rows],
            "estimator_batch_rms_dB_bits": rms_db,
            "estimator_batch_argmax_flips": est_flips,
            "estimator_batch_max_abs_dB_bits": float(d_b.abs().max()),
            "estimator_batch_max_abs_dH_bits": float((h_b - h_m).abs().max()),
            "entropy_rms_tol_bits": entropy_rms_tol_bits,
            "tolerance": tol,
        }
        # Scoring a sequence alone is reported but not gated: the measurement never does it.
        passed = (repeat_max <= tol and argmax_flips == 0 and est_flips == 0)
        if entropy_rms_tol_bits is not None:
            passed = passed and rms_db <= entropy_rms_tol_bits
        self.report.determinism = res
        self.report.batch_invariance = {"max_abs_logprob_diff": batch_max,
                                        "argmax_flips": solo_flips}
        self.report.gates_passed["determinism"] = bool(passed)
        return res

    # ---- GATE 3: does it actually generate text? ---------------------------

    def fluency_ids(self, window: Sequence[int], n_new: int = 32) -> List[int]:
        """Sequential greedy decode into an all-masked canvas at the current offset. One
        extra masked position keeps a read at p+1 inside the canvas."""
        cut = len(window) // 2
        ids = list(window[:cut]) + [self.mask_id] * (n_new + 1)
        for j in range(n_new):
            p = cut + j
            lp = self.token_logprobs([ids], [[p]])
            ids[p] = int(lp[0, 0].argmax())
        return ids[cut:cut + n_new]

    def check_fluency(self, windows: Sequence[Sequence[int]], n_new: int = 32) -> Dict:
        """Gate 3. Greedy-decode at the calibrated offset and at every other offset in
        {-1, 0, +1}. Pass only if the calibrated decode is not degenerate and at least one
        wrong offset's decode is, so the gate demonstrably tells the conventions apart.
        Any exception fails the gate."""
        good = self.offset
        res: Dict = {"calibrated_offset": good, "wrong_degenerate": {}}
        try:
            ids_good = self.fluency_ids(windows[0], n_new)
            for d in (-1, 0, 1):
                if d == good:
                    continue
                self.offset = d
                res["wrong_degenerate"][str(d)] = degenerate(self.fluency_ids(windows[0], n_new))
            self.offset = good
            res["calibrated_degenerate"] = degenerate(ids_good)
            self.report.fluency_sample = (self.tok.decode(ids_good) if self.tok is not None
                                          else " ".join(map(str, ids_good)))
            passed = (not res["calibrated_degenerate"]) and any(res["wrong_degenerate"].values())
        except Exception as e:                                # noqa: BLE001
            self.offset = good
            self.report.fluency_sample = f"<failed: {type(e).__name__}: {e}>"
            passed = False
        self.report.fluency_evidence = res
        self.report.gates_passed["fluency"] = bool(passed)
        return res

    # ---- run everything ----------------------------------------------------

    def run_gates(self, windows: Sequence[Sequence[int]], k_block: int = 4,
                  entropy_rms_tol_bits: Optional[float] = None, batch_blocks: int = 8,
                  max_rows: int = 64) -> AdapterReport:
        t0 = time.time()
        self.calibrate_offset(windows, k_block=k_block)
        self.check_determinism(windows, k=k_block, batch_blocks=batch_blocks,
                               max_rows=max_rows, entropy_rms_tol_bits=entropy_rms_tol_bits)
        self.check_fluency(windows)
        self.report.offset_evidence["gate_seconds"] = round(time.time() - t0, 1)
        return self.report


# --------------------------------------------------------------------------- registry

# Expected conventions, from each model's own documentation and shipped decoder.
# These are ASSERTIONS to be checked against the measured value, never inputs to it.
EXPECTED_OFFSET = {
    "dream_base": -1,      # Dream paper Sec. 4.1 + generation_utils.py shift
    "dream_instruct": -1,
    "llada_base": 0,       # LLaDA generate.py reads mask_index directly
    "llada_instruct": 0,
    "llada_instruct": 0,
}


def report_to_dict(r: AdapterReport) -> Dict:
    return asdict(r)
