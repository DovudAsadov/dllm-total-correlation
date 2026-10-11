#!/usr/bin/env python3
"""
Phase 1: the validation gates. NOTHING measures until these pass.

For each checkpoint we establish, by measurement rather than assumption:

  GATE 1 (offset)       which raw logit index carries the distribution for position p.
                        Probe: mask the block, score the log-prob of the tokens that
                        were actually there, reading at p+delta for delta in {-1,0,+1}.
                        The truth is hidden, so a bidirectional model cannot win by
                        copying. Must agree at k=1 and k=4 and beat the runner-up by
                        >=1 bit.

  GATE 2 (determinism)  two identical forward calls must give identical logits, and a
                        sequence must get the same logits alone as in a batch. Any
                        violation is a noise floor that has to be reported, because it
                        is anti-correlated with confidence and therefore manufactures
                        exactly the proxy-fidelity signal this paper measures.

  GATE 3 (fluency)      greedy-decode into an all-masked canvas with the calibrated
                        offset and read the text. A wrong offset produces garbage.

Writes results/gates.json.

Usage:
  python scripts/measure/gates.py                 # all registered checkpoints
  MODELS=dream_base python scripts/measure/gates.py
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from tcaudit.adapters import MDMAdapter, EXPECTED_OFFSET, report_to_dict  # noqa: E402

WINDOW = int(os.environ.get("WINDOW", "128"))
K = int(os.environ.get("BLOCK_SIZE", "4"))
N_PROBE = int(os.environ.get("N_PROBE", "96"))
CORPUS = os.environ.get("CORPUS", os.path.join(ROOT, "data", "corpus.jsonl"))
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

HUB = os.path.expanduser(os.environ.get("HF_HUB", "~/.cache/huggingface/hub"))


def snapshot(repo: str) -> str:
    d = os.path.join(HUB, "models--" + repo.replace("/", "--"), "snapshots")
    snaps = sorted(os.listdir(d))
    return os.path.join(d, snaps[-1])


REGISTRY = {
    "dream_base":     "Dream-org/Dream-Coder-v0-Base-7B",
    "dream_instruct": "Dream-org/Dream-Coder-v0-Instruct-7B",
    "llada_base":     "GSAI-ML/LLaDA-8B-Base",
    "llada_instruct": "GSAI-ML/LLaDA-8B-Instruct",
}


def load_windows(tok, n_max: int = 400):
    docs = [json.loads(l) for l in open(CORPUS)]
    wins = []
    for d in docs:
        ids = tok(d["text"], return_tensors=None)["input_ids"]
        if len(ids) < WINDOW:
            continue
        wins.append(ids[:WINDOW])
        if len(wins) >= n_max:
            break
    return wins


def main():
    tags = os.environ.get("MODELS", ",".join(REGISTRY)).split(",")
    out = {"config": {"window": WINDOW, "k": K, "n_probe": N_PROBE,
                      "corpus": os.path.basename(CORPUS)}, "models": {}}
    path = os.path.join(RESULTS, "gates.json")

    for tag in tags:
        tag = tag.strip()
        if not tag:
            continue
        repo = REGISTRY[tag]
        print(f"\n{'='*72}\n{tag}  ({repo})\n{'='*72}", flush=True)
        rec = {"repo": repo}
        t0 = time.time()
        try:
            ad = MDMAdapter(snapshot(repo), tag)
            print(f"  loaded via {ad.report.loaded_via}, mask_id={ad.mask_id}, "
                  f"transformers={ad.report.transformers_version}", flush=True)
            wins = load_windows(ad.tok, n_max=N_PROBE + 16)
            print(f"  {len(wins)} probe windows of {WINDOW} tokens", flush=True)

            rep = ad.run_gates(wins, k_block=K)
            rec.update(report_to_dict(rep))

            exp = EXPECTED_OFFSET.get(tag)
            rec["expected_offset_from_docs"] = exp
            rec["measured_matches_docs"] = (exp is None or exp == rep.offset)

            print(f"  OFFSET      measured={rep.offset}  docs={exp}  "
                  f"match={rec['measured_matches_docs']}", flush=True)
            for kk, sc in rep.offset_evidence[
                    "mean_log2_prob_of_hidden_true_token_by_delta"].items():
                pretty = "  ".join(f"d={d}:{v:+7.3f}" for d, v in sorted(sc.items()))
                print(f"    {kk:6s} mean log2 p(hidden true token):  {pretty}", flush=True)
            print(f"  DETERMINISM {rep.determinism}", flush=True)
            print(f"  FLUENCY     {rep.fluency_sample!r}", flush=True)
            print(f"  GATES       {rep.gates_passed}", flush=True)
        except Exception as e:                                # noqa: BLE001
            import traceback
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()[-3000:]
            print(f"  FAILED: {rec['error']}", flush=True)
            print(rec["traceback"], flush=True)
        rec["seconds"] = round(time.time() - t0, 1)
        out["models"][tag] = rec

        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        # free the GPU before the next checkpoint
        try:
            del ad
            import gc, torch
            gc.collect(); torch.cuda.empty_cache()
        except Exception:                                     # noqa: BLE001
            pass

    print(f"\nwrote {path}")
    ok = {t: r.get("gates_passed", {}) for t, r in out["models"].items()}
    print("SUMMARY:", json.dumps(ok, indent=1))


if __name__ == "__main__":
    main()
