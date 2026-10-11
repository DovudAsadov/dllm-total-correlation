"""Token-level degeneracy of the Gate 3 decodes, to calibrate an automatic criterion."""
import json, os, sys
from transformers import AutoTokenizer
sys.path.insert(0, os.path.join(os.environ["REPO_ROOT"], "scripts", "measure"))
REG = {"dream_instruct": "Dream-org/Dream-Coder-v0-Instruct-7B", "llada_base": "GSAI-ML/LLaDA-8B-Base",
       "dream_base": "Dream-org/Dream-Coder-v0-Base-7B"}
def snap(repo):
    d = os.path.join(os.environ["HF_HUB_CACHE"], "models--" + repo.replace("/", "--"), "snapshots")
    return os.path.join(d, sorted(os.listdir(d))[-1])
def stats(ids):
    n = len(ids); run = best = 1
    for a, b in zip(ids, ids[1:]):
        run = run + 1 if a == b else 1; best = max(best, run)
    top = max(ids.count(t) for t in set(ids)) / n
    return len(set(ids)) / n, best, top
out = {}
for path in sys.argv[1:]:
    d = json.load(open(path)); tok = AutoTokenizer.from_pretrained(snap(REG[d["model"]]), trust_remote_code=True)
    print("##", d["model"]); rows = []
    for row in d["gate3"]:
        for k in ("calibrated", "wrong"):
            ids = tok(row[k], add_special_tokens=False)["input_ids"]
            dr, run, top = stats(ids)
            rows.append({"domain": row["domain"], "which": k, "n": len(ids), "distinct": dr, "longest_run": run, "top_share": top})
            print(f"  {row['domain']:5s} {k:10s} n={len(ids):2d} distinct={dr:.2f} longest_run={run:2d} top_share={top:.2f}")
    out[d["model"]] = rows
json.dump(out, open("gate3_stats.json", "w"), indent=1)
