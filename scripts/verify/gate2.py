#!/usr/bin/env python3
"""Gate 2 with the entropy tolerance, on a real checkpoint (GPU).

Runs the offset gate and then Gate 2 on canvases from both domains (prose and code
windows interleaved), in both regimes, at the batch shapes the main run used (8 blocks
for B, 64-row estimator batches). Writes results/gate2_entropy_<tag>.json.

Env: MODEL_TAG (required), N_WINDOWS (default 16), GATE2_ENTROPY_TOL (default 0.074),
     plus whatever scripts/measure/estimate.py needs to find the checkpoint (HF_HUB).
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "measure"))
import estimate as rm                                   # noqa: E402
from tcaudit.adapters import MDMAdapter, EXPECTED_OFFSET      # noqa: E402

TAG = rm.MODEL_TAG
N = int(os.environ.get("N_WINDOWS", "16"))
t0 = time.time()
ad = MDMAdapter(rm.snapshot(rm.REGISTRY[TAG]), TAG)
MDMAdapter.max_rows = rm.MAX_ROWS
wins = rm.load_windows(ad.tok)
prose = [w["ids"] for w in wins if w["domain"] == "prose"]
code = [w["ids"] for w in wins if w["domain"] == "code"]
mixed = [w for pair in zip(prose, code) for w in pair]       # prose, code, prose, ...
ad.calibrate_offset(mixed, k_block=rm.K)
assert ad.offset == EXPECTED_OFFSET[TAG], (ad.offset, EXPECTED_OFFSET[TAG])
res = ad.check_determinism(mixed, n=N, k=rm.K, batch_blocks=rm.BATCH_BLOCKS,
                           max_rows=rm.MAX_ROWS, entropy_rms_tol_bits=rm.GATE2_ENTROPY_TOL)
out = {"model": TAG, "device": __import__("torch").cuda.get_device_name(0),
       "n_windows": N, "passed": ad.report.gates_passed["determinism"],
       "seconds": round(time.time() - t0, 1), "determinism": res}
path = os.path.join(rm.RESULTS, f"gate2_entropy_{TAG}.json")
json.dump(out, open(path, "w"), indent=1)
print(f"{TAG}: passed={out['passed']} rms_dB={res['estimator_batch_rms_dB_bits']:.4f} "
      f"max_dB={res['estimator_batch_max_abs_dB_bits']:.4f} max_dH={res['estimator_batch_max_abs_dH_bits']:.4f} "
      f"tol={res['entropy_rms_tol_bits']} -> {path}")
