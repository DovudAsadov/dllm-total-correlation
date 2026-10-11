#!/usr/bin/env python3
"""
Does the meter converge on a REAL model, and is M=64 enough?

E1 validated the estimator against analytic C on the oracle, where there is no model
error. That shows the estimator is correct; it does not show that the order-averaged
chain-rule sum has settled at a given M on a real checkpoint, where the conditionals are
not guaranteed to compose into a joint and the per-order spread can be much larger.

This runs the same estimator at M in {4,8,16,32,64,128,256} on a fixed set of real
blocks, sharing draws across M by drawing 256 orders once and taking prefixes, so the
curve is a genuine convergence curve rather than independent noisy runs.

Reports, per M: mean |C_hat(M) - C_hat(256)|, the standard error implied by the
per-order spread, and the forward-pass cost. Justifies (or refutes) the M used in the
main measurement.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from tcaudit.adapters import MDMAdapter, EXPECTED_OFFSET  # noqa: E402

LOG2E = 1.4426950408889634
MODEL_TAG = os.environ["MODEL_TAG"]
K = int(os.environ.get("BLOCK_SIZE", "4"))
M_MAX = int(os.environ.get("M_MAX", "256"))
GRID = [int(x) for x in os.environ.get("M_GRID", "4,8,16,32,64,128,256").split(",")]
N_BLOCKS = int(os.environ.get("N_BLOCKS", "48"))
REGIME = os.environ.get("REGIME", "suffix")
WINDOW = int(os.environ.get("WINDOW", "128"))
MAX_ROWS = int(os.environ.get("MAX_ROWS", "64"))
SEED = int(os.environ.get("SEED", "0"))
CORPUS = os.environ.get("CORPUS", os.path.join(ROOT, "data", "corpus.jsonl"))
RESULTS = os.path.join(ROOT, "results")
HUB = os.path.expanduser("~/.cache/huggingface/hub")
REGISTRY = {"dream_base": "Dream-org/Dream-Coder-v0-Base-7B",
            "dream_instruct": "Dream-org/Dream-Coder-v0-Instruct-7B",
            "llada_base": "GSAI-ML/LLaDA-8B-Base",
            "llada_instruct": "GSAI-ML/LLaDA-8B-Instruct"}


def snapshot(repo):
    d = os.path.join(HUB, "models--" + repo.replace("/", "--"), "snapshots")
    return os.path.join(d, sorted(os.listdir(d))[-1])


def main():
    import torch
    repo = REGISTRY[MODEL_TAG]
    ad = MDMAdapter(snapshot(repo), MODEL_TAG)
    MDMAdapter.max_rows = MAX_ROWS

    docs = [json.loads(l) for l in open(CORPUS)]
    wins = []
    for d in docs:
        ids = ad.tok(d["text"], return_tensors=None)["input_ids"]
        if len(ids) >= WINDOW:
            wins.append(ids[:WINDOW])
    ad.run_gates(wins, k_block=K)
    assert ad.report.all_gates_passed(), ad.report.gates_passed
    assert EXPECTED_OFFSET.get(MODEL_TAG) in (None, ad.offset)
    print(f"gates OK, offset={ad.offset}", flush=True)

    rng = np.random.default_rng(SEED)
    blocks = []
    for w in wins[:N_BLOCKS]:
        s = len(w) // 2
        ids = list(w)
        stop = len(ids) if REGIME == "suffix" else s + K
        for p in range(s, stop):
            ids[p] = ad.mask_id
        blocks.append((ids, list(range(s, s + K))))

    # marginal term
    B_bits = []
    for i in range(0, len(blocks), 8):
        ch = blocks[i:i + 8]
        lp = ad.token_logprobs([b[0] for b in ch], [b[1] for b in ch])
        pr = lp.exp()
        B_bits += list((-(pr * lp).sum(-1) * LOG2E).sum(-1).cpu().numpy())
    B_bits = np.asarray(B_bits)

    # one pool of M_MAX orders; prefixes give nested estimates
    orders = [rng.permutation(K) for _ in range(M_MAX)]
    nb = len(blocks)
    Hsum = np.zeros((nb, M_MAX))
    t0 = time.time()
    CH = 2                                    # blocks per chunk (M_MAX rows each)
    for c0 in range(0, nb, CH):
        ch = blocks[c0:c0 + CH]
        n = len(ch)
        state = [[list(ch[i][0]) for _ in range(M_MAX)] for i in range(n)]
        for t in range(K):
            flat, want = [], []
            for i in range(n):
                for m in range(M_MAX):
                    flat.append(state[i][m])
                    want.append([ch[i][1][int(orders[m][t])]])
            d = ad.token_logprobs(flat, want)
            dp = d[:, 0].exp()
            h = (-(dp * d[:, 0]).sum(-1) * LOG2E).cpu().numpy()
            smp = torch.multinomial(dp, 1).squeeze(-1).cpu().numpy()
            r = 0
            for i in range(n):
                for m in range(M_MAX):
                    j = int(orders[m][t])
                    Hsum[c0 + i, m] += h[r]
                    state[i][m][ch[i][1][j]] = int(smp[r])
                    r += 1
        print(f"  {min(c0+CH, nb)}/{nb} blocks  {time.time()-t0:.0f}s "
              f"nfe={MDMAdapter.nfe}", flush=True)

    ref = B_bits - Hsum.mean(axis=1)
    curve = {}
    for M in GRID:
        if M > M_MAX:
            continue
        ch = B_bits - Hsum[:, :M].mean(axis=1)
        se = Hsum[:, :M].std(axis=1, ddof=1) / np.sqrt(M)
        curve[str(M)] = {
            "mean_abs_dev_from_M%d_bits" % M_MAX: float(np.abs(ch - ref).mean()),
            "max_abs_dev_bits": float(np.abs(ch - ref).max()),
            "median_se_bits": float(np.median(se)),
            "mean_C_hat_bits": float(ch.mean()),
            "forward_passes_per_block": M * K,
        }
        print(f"M={M:4d}  mean|dev|={curve[str(M)]['mean_abs_dev_from_M%d_bits'%M_MAX]:.4f} "
              f"median SE={curve[str(M)]['median_se_bits']:.4f} bits", flush=True)

    out = {"config": {"model_tag": MODEL_TAG, "repo": repo, "k": K, "regime": REGIME,
                      "n_blocks": nb, "m_max": M_MAX, "grid": GRID, "seed": SEED,
                      "offset": ad.offset},
           "reference_C_hat_bits": ref.tolist(),
           "budget_bits": B_bits.tolist(),
           "curve": curve}
    p = os.path.join(RESULTS, f"e9_meter_convergence_{MODEL_TAG}_k{K}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
