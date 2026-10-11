"""GPU verification probes for the camera-ready plan. NOT part of the paper pipeline.

Env: MODEL_TAG, REPO_ROOT (copy of the paper repo), N_FLAG, N_CTRL, M, OUT, HF_HUB.

Q1  Gate 1 on code windows as well as prose (the paper's gate probed prose only).
Q2  Gate 3 at the calibrated vs a wrong offset: does a wrong offset garble greedy text?
Q3  Batch shape: per-position entropy for the same block alone, inside its original
    batch of 8 (how B was computed), and in the meter's 64-row t=0 layout.
Q4  Do the hairline 'impossible' values (C<0, C>B-maxH) vanish when B is taken from the
    meter's own t=0 forward instead of a differently shaped batch?
Q5  Eq.(5) under inconsistency: per-order KL version  E_pi KL(q_pi || prod p_i)  and the
    order information  I(pi; Y_S),  from all K! orders evaluated at the meter's samples
    (sum_{|A|<K} C(K,|A|) canvases per sample).
"""
import itertools, json, os, sys, time
import numpy as np

ROOT = os.environ["REPO_ROOT"]
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts", "measure"))
os.environ.setdefault("BLOCK_SIZE", "4"); os.environ.setdefault("MAX_BLOCKS", "600")
import estimate as rm                                   # noqa: E402
from tcaudit.adapters import MDMAdapter, EXPECTED_OFFSET      # noqa: E402
import torch                                                  # noqa: E402

LOG2E = 1.4426950408889634
TAG = os.environ["MODEL_TAG"]; K = rm.K
N_FLAG = int(os.environ.get("N_FLAG", "40")); N_CTRL = int(os.environ.get("N_CTRL", "24"))
M = int(os.environ.get("M", "64")); OUT = os.environ["OUT"]
M_I = int(os.environ.get("M_I", "32"))   # samples used for the K!-order evaluation
REGIMES = os.environ.get("REGIMES", "infill,suffix").split(",")
t_start = time.time()
def log(msg): print(f"[{time.time()-t_start:7.1f}s] {msg}", flush=True)

ad = MDMAdapter(rm.snapshot(rm.REGISTRY[TAG]), TAG)
MDMAdapter.max_rows = 64
rm.MASK = ad.mask_id          # estimate.main() sets this global
wins = rm.load_windows(ad.tok)
out = {"model": TAG, "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "M": M, "K": K}

if os.environ.get("TIME_ONLY"):
    ad.offset = EXPECTED_OFFSET[TAG]
    seq = [rm.base_sequence(rm.make_blocks(wins, "suffix")[0])] * 64
    for rows in (1, 8, 64):
        ad.token_logprobs(seq[:rows], [[64, 65, 66, 67]] * rows); torch.cuda.synchronize()
        t0 = time.time(); ad.token_logprobs(seq[:rows], [[64, 65, 66, 67]] * rows); torch.cuda.synchronize()
        log(f"forward rows={rows}: {time.time()-t0:.2f}s  attn={getattr(ad.model.config, '_attn_implementation', '?')} dtype={next(ad.model.parameters()).dtype}")
    sys.exit(0)
# ---------------------------------------------------------------- Q1: gate 1, prose vs code
gate1 = {}
for dom in ["prose", "code"]:
    ws = [w["ids"] for w in wins if w["domain"] == dom]
    try:
        ad.calibrate_offset(ws, k_block=K)
        gate1[dom] = {"winner": ad.report.offset_evidence["winner_per_k"],
                      "gap": ad.report.offset_evidence["min_gap_bits_over_runner_up"],
                      "scores": ad.report.offset_evidence["mean_log2_prob_of_hidden_true_token_by_delta"]}
    except RuntimeError as e:
        gate1[dom] = {"failed": str(e)[:400]}
out["gate1"] = gate1
log(f"gate1: { {d: (v.get('winner'), round(v.get('gap', -1), 2)) for d, v in gate1.items()} }")
good = EXPECTED_OFFSET[TAG]; ad.offset = good

# ---------------------------------------------------------------- timing
seq = [rm.base_sequence(rm.make_blocks(wins, "suffix")[0])] * 64
torch.cuda.synchronize(); t0 = time.time()
for _ in range(3): ad.token_logprobs(seq, [[64, 65, 66, 67]] * 64)
torch.cuda.synchronize(); out["sec_per_64row_forward"] = (time.time() - t0) / 3
log(f"forward of 64 rows: {out['sec_per_64row_forward']:.2f}s")

# ---------------------------------------------------------------- Q2: gate 3, right vs wrong offset
wrong = 0 if good == -1 else -1
g3 = []
for dom in ["prose", "code"]:
    ws = [w["ids"] for w in wins if w["domain"] == dom][:3]
    for w in ws:
        row = {"domain": dom}
        for nm, off in [("calibrated", good), ("wrong", wrong)]:
            ad.offset = off
            txt = ad.tok.decode(ad.fluency_ids(w, n_new=32))   # fluency_sample before f393b2f
            row[nm] = txt
        ad.offset = good
        g3.append(row)
out["gate3"] = g3
log("gate3 done")

# ---------------------------------------------------------------- Q3-Q5 per block
main = json.load(open(os.path.join(ROOT, "results", f"e5v3_{TAG}.json")))
BB = 8   # remote/run_all.sh and llada_instruct_run.sh: BATCH_BLOCKS=8
gen = torch.Generator(device="cuda"); gen.manual_seed(1234)
subsets = [A for r in range(1, K) for A in itertools.combinations(range(K), r)]
perms = list(itertools.permutations(range(K)))

def ent_bits(lp):            # lp [..., V] natural-log probs -> entropy bits
    return (-(lp.exp() * lp).sum(-1) * LOG2E)

records = []
for rg in REGIMES:
    blocks = rm.make_blocks(wins, rg)
    recs = [r for r in main["records"] if r["regime"] == rg]
    assert [(b["doc_id"], b["start"]) for b in blocks] == [(r["doc_id"], r["start"]) for r in recs], \
        "block order does not reproduce the main run"
    C = np.array([r["C_hat_bits"] for r in recs]); se = np.array([r["C_hat_se_bits"] for r in recs])
    Bm = np.array([r["budget_bits"] for r in recs])
    W = Bm - np.array([r["profile_entropy_bits"] for r in recs]).max(1)
    flag = (C < -2 * se) | (C > W + 2 * se)
    rng = np.random.default_rng(7)
    fi = np.where(flag)[0]; ci = np.where(~flag)[0]
    fi = rng.choice(fi, min(N_FLAG, len(fi)), replace=False) if len(fi) else fi
    ci = rng.choice(ci, min(N_CTRL, len(ci)), replace=False)
    sel = sorted(list(fi) + list(ci))
    log(f"{rg}: {flag.sum()} flagged in main run; probing {len(fi)} flagged + {len(ci)} controls")
    chunk_cache = {}
    for n_done, idx in enumerate(sel):
        b = blocks[idx]; base = rm.base_sequence(b); pos = list(range(b["start"], b["start"] + K))
        # ---- Q3: three batch shapes
        c0 = (idx // BB) * BB
        if c0 not in chunk_cache:
            ch = blocks[c0:c0 + BB]
            lp8 = ad.token_logprobs([rm.base_sequence(x) for x in ch],
                                    [list(range(x["start"], x["start"] + K)) for x in ch])
            chunk_cache = {c0: ent_bits(lp8).cpu().numpy()}
        H8 = chunk_cache[c0][idx - c0]
        H1 = ent_bits(ad.token_logprobs([base], [pos])[0]).cpu().numpy()
        lp0 = ad.token_logprobs([base] * M, [pos] * M)          # meter t=0 layout, all K read
        H0rows = ent_bits(lp0).cpu().numpy()                    # [M, K]
        H0 = H0rows[0]; lpm = lp0[0]                            # marginals used for X and A=()
        # ---- meter, as in estimate but with a seeded sampler and stored samples
        orders = [np.random.default_rng(10_000 * idx + m).permutation(K) for m in range(M)]
        state = [list(base) for _ in range(M)]
        Hsum = np.zeros(M); pathlq = np.zeros(M); y = np.zeros((M, K), dtype=np.int64)
        for t in range(K):
            if t == 0:
                d = torch.stack([lp0[m, int(orders[m][0])] for m in range(M)])     # [M, V]
            else:
                d = ad.token_logprobs(state, [[pos[int(orders[m][t])]] for m in range(M)])[:, 0]
            h = ent_bits(d).cpu().numpy()
            smp = torch.multinomial(d.exp(), 1, generator=gen).squeeze(-1)
            lq = d.gather(1, smp[:, None])[:, 0].cpu().numpy() * LOG2E
            smp = smp.cpu().numpy()
            for m in range(M):
                j = int(orders[m][t]); Hsum[m] += h[m]; pathlq[m] += lq[m]
                state[m][pos[j]] = int(smp[m]); y[m, j] = int(smp[m])
        # ---- Q5: all subset canvases at the sampled assignments
        need = {}                                               # (A, yA) -> canvas row
        for m in range(M_I):
            for A in subsets:
                key = (A, tuple(int(y[m, a]) for a in A))
                if key not in need:
                    cv = list(base)
                    for a in A: cv[pos[a]] = int(y[m, a])
                    need[key] = cv
        keys = list(need)
        lpA = {}
        for s in range(0, len(keys), 64):
            ks = keys[s:s + 64]
            lpb = ad.token_logprobs([need[k] for k in ks], [pos] * len(ks))       # [n, K, V]
            for r, k in enumerate(ks):
                lpA[k] = lpb[r]
        logq = np.zeros((M_I, len(perms)))                     # log2 q_pi(y_m), all pi
        cond = {}
        for m in range(M):
            ym = torch.as_tensor(y[m], device=lpm.device)
            cm = {(): (lpm.gather(1, ym[:, None])[:, 0] * LOG2E).cpu().numpy()}
            cond[m] = cm
            if m >= M_I:
                continue
            for A in subsets:
                key = (A, tuple(int(y[m, a]) for a in A))
                cm[A] = (lpA[key].gather(1, ym[:, None])[:, 0] * LOG2E).cpu().numpy()
            for pi_i, pi in enumerate(perms):
                tot = 0.0
                for t in range(K):
                    A = tuple(sorted(pi[:t])); tot += cm[A][pi[t]]
                logq[m, pi_i] = tot
        own = np.array([perms.index(tuple(int(v) for v in orders[m])) for m in range(M_I)])
        lq_own = logq[np.arange(M_I), own]
        lq_bar = np.log2(np.mean(2.0 ** (logq - logq.max(1, keepdims=True)), axis=1)) + logq.max(1)
        I_m = lq_own - lq_bar
        X = np.array([-cond[m][()].sum() for m in range(M)])  # sum_j -log2 p_j(y_mj)
        KL_m = X - Hsum
        rec = {
            "regime": rg, "idx": int(idx), "doc_id": b["doc_id"], "domain": b["domain"],
            "flag_main": bool(flag[idx]), "C_main": float(C[idx]), "se_main": float(se[idx]),
            "B_main": float(Bm[idx]), "W_main": float(W[idx]),
            "H8": H8.tolist(), "H1": H1.tolist(), "H0": H0.tolist(),
            "H0_row_spread": float((H0rows.max(0) - H0rows.min(0)).max()),
            "C8": float(H8.sum() - Hsum.mean()), "C0": float(H0.sum() - Hsum.mean()),
            "Hsum_mean": float(Hsum.mean()), "Hsum_se": float(Hsum.std(ddof=1) / np.sqrt(M)),
            "KL_mean": float(KL_m.mean()), "KL_se": float(KL_m.std(ddof=1) / np.sqrt(M)),
            "I_mean": float(I_m.mean()),
            "I_se": float(I_m.std(ddof=1) / np.sqrt(M_I)),
            "KL_I_mean": float(KL_m[:M_I].mean()),
            "mix_mean": float((KL_m[:M_I] - I_m).mean()), "mix_se": float((KL_m[:M_I] - I_m).std(ddof=1) / np.sqrt(M_I)),
            "path_vs_subset_logq_maxabs": float(np.abs(pathlq[:M_I] - lq_own).max()),
            "path_vs_subset_logq_meanabs": float(np.abs(pathlq[:M_I] - lq_own).mean()),
            "n_canvases": len(keys),
        }
        records.append(rec)
        if n_done % 8 == 0:
            log(f"  {rg} {n_done+1}/{len(sel)}  idx={idx} flag={rec['flag_main']} "
                f"C8={sum(H8)-rec['Hsum_mean']:+.3f} C0={sum(H0)-rec['Hsum_mean']:+.3f} "
                f"KL={rec['KL_mean']:+.3f} I={rec['I_mean']:.3f} canv={len(keys)}")
        out["records"] = records
        with open(OUT, "w") as f:
            json.dump(out, f)
log(f"wrote {OUT}: {len(records)} blocks")
