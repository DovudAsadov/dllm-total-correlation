"""
Real masked-diffusion-LM backend (Dream / LLaDA family).

Implements the SAME Backend interface the oracle uses, so the identical meter and
proxy code run on a real model. A conditional query p(Y_i | Y_given, context) is a
single masked forward pass: fill the context and `given` positions with their
tokens, mask every other block position (including i), read the softmax at i.

This is the only module that imports torch; it is meant to run inside the NVIDIA
NGC PyTorch container on the DGX Spark, never on the laptop.

Cost accounting: every forward is one model NFE. A NFECounter is threaded through
so decoding vs. metering forward passes are reported separately, as the paper's
cost axis requires.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class ModelSpec:
    model_id: str
    mask_token_id: int
    device: str = "cuda"
    dtype: str = "bfloat16"
    top_k_keep: int = 0  # if >0, truncate conditional to top-k for speed/entropy


class ModelBackend:
    """Any-order conditional backend over a fixed base token sequence.

    A "sample" is: base_tokens (a full token id list) + block_positions (indices
    into base_tokens that form the candidate region). Context = all non-block
    positions held at their base tokens; block positions are masked unless given.
    """

    def __init__(self, model, tokenizer, spec: ModelSpec):
        import torch  # local import; only available in the container

        self.torch = torch
        self.model = model
        self.tok = tokenizer
        self.spec = spec
        self.mask_id = spec.mask_token_id
        self.model.eval()

        # state set per sample
        self.base_tokens: Optional[List[int]] = None
        self.block_positions: Optional[List[int]] = None
        self._forward_calls = 0

    # ---- sample setup ----
    def set_sample(self, base_tokens: Sequence[int], block_positions: Sequence[int]):
        self.base_tokens = list(base_tokens)
        self.block_positions = list(block_positions)

    @property
    def forward_calls(self) -> int:
        return self._forward_calls

    # ---- the Backend interface ----
    def _build_input(self, given: Dict[int, int], query: Optional[int]):
        torch = self.torch
        ids = list(self.base_tokens)
        for pos in self.block_positions:
            ids[pos] = self.mask_id            # mask the whole block by default
        for pos, val in given.items():
            ids[pos] = val                     # reveal given positions
        if query is not None:
            ids[query] = self.mask_id          # ensure the query is masked
        return torch.tensor([ids], device=self.spec.device)

    def _logits_at(self, given: Dict[int, int], positions: Sequence[int]):
        """Return softmax distributions at `positions` given `given`. One forward."""
        torch = self.torch
        inp = self._build_input(given, query=None)
        # ensure each queried position is masked
        for p in positions:
            inp[0, p] = self.mask_id
        with torch.no_grad():
            out = self.model(input_ids=inp)
            logits = out.logits if hasattr(out, "logits") else out[0]
        self._forward_calls += 1
        dists = {}
        for p in positions:
            lp = logits[0, p].float()
            probs = torch.softmax(lp, dim=-1)
            dists[p] = probs
        return dists

    def conditional(self, i: int, given: Dict[int, int]) -> Dict[int, float]:
        """p(Y_i | Y_given, context) as {token_id: prob}. Optionally top-k truncated
        (renormalised) to keep entropy sums finite and cheap; full vocab if k=0."""
        probs = self._logits_at(given, [i])[i]
        torch = self.torch
        if self.spec.top_k_keep and self.spec.top_k_keep < probs.numel():
            vals, idx = torch.topk(probs, self.spec.top_k_keep)
            vals = (vals / vals.sum()).tolist()
            idx = idx.tolist()
            return {int(t): float(p) for t, p in zip(idx, vals)}
        # full distribution as a dict is large; keep only nonzero-ish mass
        p = probs.tolist()
        return {t: pv for t, pv in enumerate(p) if pv > 1e-6}

    def marginals(self, positions) -> Dict[int, Dict[int, float]]:
        """All block marginals in ONE forward (every block position masked)."""
        torch = self.torch
        dists = self._logits_at({}, list(positions))
        out = {}
        for p, probs in dists.items():
            if self.spec.top_k_keep and self.spec.top_k_keep < probs.numel():
                vals, idx = torch.topk(probs, self.spec.top_k_keep)
                vals = (vals / vals.sum()).tolist()
                idx = idx.tolist()
                out[p] = {int(t): float(pv) for t, pv in zip(idx, vals)}
            else:
                pl = probs.tolist()
                out[p] = {t: pv for t, pv in enumerate(pl) if pv > 1e-6}
        return out


def load_dream(model_id: str, mask_token_id: int = 151666,
               device: str = "cuda", dtype: str = "bfloat16",
               top_k_keep: int = 64):
    """Load a Dream-family masked diffusion LM with trust_remote_code."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True,
        torch_dtype=getattr(torch, dtype),
    ).to(device)
    spec = ModelSpec(model_id=model_id, mask_token_id=mask_token_id,
                     device=device, dtype=dtype, top_k_keep=top_k_keep)
    return ModelBackend(model, tok, spec), tok
