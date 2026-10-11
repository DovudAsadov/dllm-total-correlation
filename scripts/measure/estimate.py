#!/usr/bin/env python3
"""
E5 v3: the real-model measurement, rebuilt on a validated readout.

WHAT v2 GOT WRONG (all verified on hardware, see results/gates.json)
-------------------------------------------------------------------
 1. It read logits[p] as the distribution for position p. That is FALSE for Dream,
    which keeps the AR shift from its Qwen initialisation. Measured on the A100:
    reading at delta=-1 assigns the hidden true token -0.84 bits, reading at delta=0
    assigns it -7.76 bits. v2 read delta=0, i.e. a distribution that gives the correct
    token ~120x less probability. Every Dream number in v2 is void. LLaDA needs the
    OPPOSITE convention (delta=0 wins by 11 bits), so no single hard-coded rule is
    safe: the offset must be measured per checkpoint.
 2. DiffusionGemma is not a masked diffusion model at all. Its technical report says
    "our denoising iteration uses multinomial (or uniform) diffusion rather than masked
    diffusion", it starts from a canvas of uniformly random tokens, has no mask token in
    its config, and its decoder needs an encoder KV cache plus a self-conditioning
    signal. Feeding it a <mask> token through a plain model(input_ids=...) measures
    nothing. It is replaced here by LLaDA-8B, a genuine masked diffusion LM.
 3. It never applied its own total-correlation meter to a real model, so the paper's
    central quantity C was never measured where the question is open. It is here.
 4. It had no control for whether "disagreement" is even well defined (the reference
    decode is one arbitrary order), and no control separating "the conditionals compose
    into a joint" from "the conditionals ignore what you fill in". Both are here.

WHAT THIS SCRIPT RECORDS PER BLOCK
----------------------------------
  proxies      all six marginal rules (confidence min/mean, margin min/mean,
               neg-entropy max/sum) so the paper's prescription can be checked on
               real models instead of only on synthetic sets
  profile      per-position top-8 marginal probabilities + entropy + confidence, the
               statistic a marginal rule actually sees. Used offline for the
               matched-profile analysis (how much does C move when the marginals agree?)
  C_hat        the METER: order-averaged chain-rule estimate of C(S|c), with the
               per-order values kept so the estimator's own standard error can be
               separated from real block-to-block variation
  disagree     parallel argmax vs sequential greedy, against the L2R reference
  deficit      sum_t [log p(y_t|y_<t) - log p_marg(y_t)], the pointwise C along the path
  inconsist    fixed-assignment order sweep (spread of the chain-rule log-prob over
               orders). NOT introduced here: this is the chain-rule-deviation
               diagnostic of Yao (arXiv:2606.00997), and MLM conditional inconsistency
               goes back to Young et al. (arXiv:2301.00068). Reported as a covariate.
  ref_order    reference-order control: decode greedily under R different orders and
               report how much the assignment, and hence "disagreement", moves
  cond_sens    conditional-sensitivity control: mean total-variation distance between
               p(Y_j|c) and p(Y_j|y_i,c). ~0 means the model ignores conditioning, which
               would make it trivially "consistent" and trivially C=0
  invariant    seq_tok[0] must equal par_argmax[0] (identical input, identical position).
               Any violation is nondeterminism and is counted, not ignored.

Env: MODEL_TAG (required), REGIME, BLOCK_SIZE, M_METER, M_DIAG, R_ORDERS,
     MAX_BLOCKS, BATCH_BLOCKS, WINDOW, MAX_ROWS, CORPUS, SEED
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

from tcaudit.adapters import MDMAdapter, EXPECTED_OFFSET, report_to_dict  # noqa: E402

LOG2E = 1.4426950408889634

MODEL_TAG = os.environ["MODEL_TAG"]
REGIME = os.environ.get("REGIME", "both")
K = int(os.environ.get("BLOCK_SIZE", "4"))
M_METER = int(os.environ.get("M_METER", "32"))
M_DIAG = int(os.environ.get("M_DIAG", "8"))
R_ORDERS = int(os.environ.get("R_ORDERS", "8"))
MAX_BLOCKS = int(os.environ.get("MAX_BLOCKS", "600"))
BATCH_BLOCKS = int(os.environ.get("BATCH_BLOCKS", "4"))
WINDOW = int(os.environ.get("WINDOW", "128"))
MAX_ROWS = int(os.environ.get("MAX_ROWS", "64"))
SEED = int(os.environ.get("SEED", "0"))
# Gate 2: RMS change in B between the two batch shapes the estimator combines, in bits.
# A tenth of the smallest residual the paper reports (0.74 bits): noise of that size adds
# in quadrature and moves that residual by under 0.5%.
GATE2_ENTROPY_TOL = float(os.environ.get("GATE2_ENTROPY_TOL", "0.074"))
TOPM = 8
CORPUS = os.environ.get("CORPUS", os.path.join(ROOT, "data", "corpus.jsonl"))
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)
HUB = os.path.expanduser(os.environ.get("HF_HUB", "~/.cache/huggingface/hub"))

REGISTRY = {
    "dream_base":     "Dream-org/Dream-Coder-v0-Base-7B",
    "dream_instruct": "Dream-org/Dream-Coder-v0-Instruct-7B",
    "llada_base":     "GSAI-ML/LLaDA-8B-Base",
    "llada_instruct": "GSAI-ML/LLaDA-8B-Instruct",
}
PROGRESS = os.path.join(RESULTS, f"e5v3_{MODEL_TAG}.progress")


def note(msg: str) -> None:
    print(msg, flush=True)
    try:
        with open(PROGRESS, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except OSError:
        pass


def snapshot(repo: str) -> str:
    d = os.path.join(HUB, "models--" + repo.replace("/", "--"), "snapshots")
    return os.path.join(d, sorted(os.listdir(d))[-1])


# ------------------------------------------------------------------ corpus / blocks

def load_windows(tok):
    docs = [json.loads(l) for l in open(CORPUS)]
    wins = []
    for d in docs:
        ids = tok(d["text"], return_tensors=None)["input_ids"]
        if len(ids) < WINDOW:
            continue
        wins.append({"doc_id": d["id"], "domain": d["domain"], "ids": ids[:WINDOW]})
    note(f"corpus: {len(docs)} documents -> {len(wins)} windows of {WINDOW} tokens")
    return wins


def make_blocks(wins, regime):
    out = []
    rng = np.random.default_rng(SEED)
    for w in wins:
        n = len(w["ids"])
        if regime == "suffix":
            starts = [n // 2]
        else:
            lo, hi = n // 4, n - K - 1
            starts = list(range(lo, hi, 4 * K))
            rng.shuffle(starts)
            starts = starts[:2]
        for s in starts:
            if s + K >= n:
                continue
            out.append({"doc_id": w["doc_id"], "domain": w["domain"],
                        "ids": w["ids"], "start": s, "regime": regime})
    rng.shuffle(out)
    return out[:MAX_BLOCKS]


def base_sequence(b):
    """Masked canvas. In `suffix` everything from the block onward is masked (the
    deployed regime); in `infill` only the block is."""
    ids = list(b["ids"])
    stop = len(ids) if b["regime"] == "suffix" else b["start"] + K
    for p in range(b["start"], stop):
        ids[p] = MASK
    return ids


# ------------------------------------------------------------------ measurement

def measure(ad, blocks):
    import torch
    recs, stats = [], {"invariant_violations": 0, "n_blocks": 0}
    rng = np.random.default_rng(SEED + 1)
    torch.manual_seed(SEED)             # the meter samples with torch.multinomial
    t0 = time.time()

    for c0 in range(0, len(blocks), BATCH_BLOCKS):
        chunk = blocks[c0:c0 + BATCH_BLOCKS]
        nb = len(chunk)
        pos = [list(range(b["start"], b["start"] + K)) for b in chunk]
        base = [base_sequence(b) for b in chunk]

        # ---- (1) marginals: one forward -----------------------------------
        lp = ad.token_logprobs(base, pos)                       # [nb, K, V]
        pr = lp.exp()
        ent = (-(pr * lp).sum(-1) * LOG2E)                      # [nb, K] bits
        top = torch.topk(pr, TOPM, dim=-1)
        conf = top.values[:, :, 0]                              # [nb, K]
        margin = top.values[:, :, 0] - top.values[:, :, 1]
        # Take the parallel commit from argmax over the SAME tensor the sequential
        # decode reads (log-probs), not from topk over exp(log-probs). At a near tie
        # float32 exp() can map two distinct log-probs to the same value, and topk then
        # breaks the tie by index while argmax breaks it the other way -- which shows up
        # downstream as a spurious violation of the step-0 invariant.
        par_argmax = lp.argmax(-1)                              # [nb, K]
        ent_np, conf_np = ent.cpu().numpy(), conf.cpu().numpy()
        marg_np = margin.cpu().numpy()
        topv_np = top.values.cpu().numpy()
        marg_lp = lp                                            # keep for the deficit

        # ---- (2) sequential greedy decode, L2R reference -------------------
        cur = [list(s) for s in base]
        seq_tok = np.zeros((nb, K), dtype=np.int64)
        deficit = np.zeros(nb)
        for step in range(K):
            d = ad.token_logprobs(cur, [[pos[i][step]] for i in range(nb)])
            tk = d[:, 0].argmax(-1)                             # [nb]
            for i in range(nb):
                t_i = int(tk[i])
                seq_tok[i, step] = t_i
                deficit[i] += (float(d[i, 0, t_i]) - float(marg_lp[i, step, t_i])) * LOG2E
                cur[i][pos[i][step]] = t_i
        par_np = par_argmax.cpu().numpy()
        disagree = (par_np != seq_tok).mean(axis=1)

        # invariant: step 0 reads the identical tensor at the identical position
        viol = int((seq_tok[:, 0] != par_np[:, 0]).sum())
        stats["invariant_violations"] += viol

        # ---- (3) THE METER: order-averaged chain-rule estimate of C --------
        # per order m: walk S, add the exact entropy of each predicted categorical,
        # sample to extend. Matches tcaudit.meter.meter_total_correlation exactly.
        orders_meter = [rng.permutation(K) for _ in range(M_METER)]
        state = [[list(base[i]) for _ in range(M_METER)] for i in range(nb)]
        Hsum = np.zeros((nb, M_METER))
        for t in range(K):
            flat, want = [], []
            for i in range(nb):
                for m in range(M_METER):
                    flat.append(state[i][m])
                    want.append([pos[i][int(orders_meter[m][t])]])
            d = ad.token_logprobs(flat, want)                   # [nb*M, 1, V]
            dp = d[:, 0].exp()
            h = (-(dp * d[:, 0]).sum(-1) * LOG2E).cpu().numpy()  # [nb*M] bits
            smp = torch.multinomial(dp, 1).squeeze(-1).cpu().numpy()
            r = 0
            for i in range(nb):
                for m in range(M_METER):
                    j = int(orders_meter[m][t])
                    Hsum[i, m] += h[r]
                    state[i][m][pos[i][j]] = int(smp[r])
                    r += 1
        B_bits = ent_np.sum(axis=1)                             # summed marginal entropy
        C_hat = B_bits - Hsum.mean(axis=1)
        C_se = Hsum.std(axis=1, ddof=1) / np.sqrt(M_METER)

        # ---- (4) inconsistency diagnostic (fixed assignment, vary order) ----
        # NOT introduced here: chain-rule deviation, Yao arXiv:2606.00997.
        orders_diag = [rng.permutation(K) for _ in range(M_DIAG)]
        state = [[list(base[i]) for _ in range(M_DIAG)] for i in range(nb)]
        L = np.zeros((nb, M_DIAG))
        for t in range(K):
            flat, want = [], []
            for i in range(nb):
                for m in range(M_DIAG):
                    flat.append(state[i][m])
                    want.append([pos[i][int(orders_diag[m][t])]])
            d = ad.token_logprobs(flat, want)
            r = 0
            for i in range(nb):
                for m in range(M_DIAG):
                    j = int(orders_diag[m][t])
                    y = int(seq_tok[i, j])                      # FIXED assignment
                    L[i, m] += float(d[r, 0, y]) * LOG2E
                    state[i][m][pos[i][j]] = y
                    r += 1

        # ---- (5) reference-order control: is "disagreement" well defined? ---
        orders_ref = [rng.permutation(K) for _ in range(R_ORDERS)]
        state = [[list(base[i]) for _ in range(R_ORDERS)] for i in range(nb)]
        assign = np.zeros((nb, R_ORDERS, K), dtype=np.int64)
        for t in range(K):
            flat, want = [], []
            for i in range(nb):
                for m in range(R_ORDERS):
                    flat.append(state[i][m])
                    want.append([pos[i][int(orders_ref[m][t])]])
            d = ad.token_logprobs(flat, want)
            tk = d[:, 0].argmax(-1).cpu().numpy()
            r = 0
            for i in range(nb):
                for m in range(R_ORDERS):
                    j = int(orders_ref[m][t])
                    assign[i, m, j] = int(tk[r])
                    state[i][m][pos[i][j]] = int(tk[r])
                    r += 1
        dis_by_order = (assign != par_np[:, None, :]).mean(axis=2)      # [nb, R]
        pair_dis = np.zeros(nb)
        for i in range(nb):
            dd = [(assign[i, a] != assign[i, b]).mean()
                  for a in range(R_ORDERS) for b in range(a + 1, R_ORDERS)]
            pair_dis[i] = float(np.mean(dd)) if dd else 0.0
        n_distinct = [len({tuple(assign[i, m]) for m in range(R_ORDERS)}) for i in range(nb)]

        # ---- (6) conditional sensitivity: does filling a position matter? ---
        # TV( p(Y_j | c), p(Y_j | y_i, c) ) averaged over ordered pairs i != j.
        tv_tot = np.zeros(nb)
        for i_fill in range(K):
            filled = []
            for i in range(nb):
                s = list(base[i])
                s[pos[i][i_fill]] = int(par_np[i, i_fill])
                filled.append(s)
            others = [[pos[i][j] for j in range(K) if j != i_fill] for i in range(nb)]
            d = ad.token_logprobs(filled, others)               # [nb, K-1, V]
            q = d.exp()
            keep = [j for j in range(K) if j != i_fill]
            p0 = pr[:, keep, :]
            tv = 0.5 * (q - p0).abs().sum(-1)                   # [nb, K-1]
            tv_tot += tv.mean(-1).cpu().numpy()
        cond_sens = tv_tot / K

        # ---- record --------------------------------------------------------
        for i, b in enumerate(chunk):
            recs.append({
                "doc_id": b["doc_id"], "domain": b["domain"], "regime": b["regime"],
                "start": b["start"], "block": pos[i],
                # all six marginal proxies
                "confidence_min": float(conf_np[i].min()),
                "confidence_mean": float(conf_np[i].mean()),
                "margin_min": float(marg_np[i].min()),
                "margin_mean": float(marg_np[i].mean()),
                "neg_entropy_max": float(-ent_np[i].max()),
                "neg_entropy_sum": float(-ent_np[i].sum()),
                "budget_bits": float(B_bits[i]),
                # the quantity that actually matters
                "C_hat_bits": float(C_hat[i]),
                "C_hat_se_bits": float(C_se[i]),
                "C_max_certified_bits": float((1.0 - 1.0 / K) * B_bits[i]),
                # outcomes
                "disagreement": float(disagree[i]),
                "logprob_deficit_bits": float(deficit[i]),
                # diagnostics and controls
                "order_logprob_std_bits": float(L[i].std(ddof=1)),
                "order_logprob_mean_bits": float(L[i].mean()),
                "ref_order_disagree_mean": float(dis_by_order[i].mean()),
                "ref_order_disagree_min": float(dis_by_order[i].min()),
                "ref_order_disagree_max": float(dis_by_order[i].max()),
                "ref_order_pairwise_disagree": float(pair_dis[i]),
                "ref_order_n_distinct_assignments": int(n_distinct[i]),
                "cond_sensitivity_tv": float(cond_sens[i]),
                "invariant_ok": bool(seq_tok[i, 0] == par_np[i, 0]),
                # the marginal profile a proxy can see (for matched-profile analysis)
                "profile_top%d" % TOPM: topv_np[i].tolist(),
                "profile_entropy_bits": ent_np[i].tolist(),
            })
        stats["n_blocks"] += nb

        done = c0 + nb
        if (c0 // BATCH_BLOCKS) % 10 == 0 or done == len(blocks):
            el = time.time() - t0
            rate = el / max(done, 1)
            note(f"  {done}/{len(blocks)} blocks  {el:.0f}s  ({rate:.2f}s/blk, "
                 f"eta {rate*(len(blocks)-done)/60:.1f}min, nfe={MDMAdapter.nfe}, "
                 f"viol={stats['invariant_violations']})")
    return recs, stats


def main():
    global MASK
    repo = REGISTRY[MODEL_TAG]
    ad = MDMAdapter(snapshot(repo), MODEL_TAG)
    MDMAdapter.max_rows = MAX_ROWS
    MASK = ad.mask_id

    wins_probe = load_windows(ad.tok)
    ad.run_gates([w["ids"] for w in wins_probe], k_block=K, entropy_rms_tol_bits=GATE2_ENTROPY_TOL,
                 batch_blocks=BATCH_BLOCKS, max_rows=MAX_ROWS)
    exp = EXPECTED_OFFSET.get(MODEL_TAG)
    if not ad.report.all_gates_passed():
        raise SystemExit(f"GATES FAILED: {ad.report.gates_passed}")
    if exp is not None and exp != ad.offset:
        raise SystemExit(f"measured offset {ad.offset} != documented {exp}")
    note(f"gates OK: offset={ad.offset} (docs {exp}), "
         f"determinism={ad.report.determinism['repeat_call_argmax_flips']} flips")

    # Deliberate misalignment, for the ablation in the paper ONLY. Everything else is
    # held fixed -- same blocks, same seed, same code path -- so the only difference
    # between this run and the validated one is the readout convention. Needs two
    # explicit env vars so it can never happen by accident.
    if os.environ.get("ABLATE_OFFSET") is not None:
        if os.environ.get("I_KNOW_THIS_IS_WRONG") != "yes":
            raise SystemExit("ABLATE_OFFSET requires I_KNOW_THIS_IS_WRONG=yes")
        bad = int(os.environ["ABLATE_OFFSET"])
        note(f"*** ABLATION: forcing offset {ad.offset} -> {bad} (deliberately wrong) ***")
        ad.offset = bad
        ad.report.offset = bad
        ad.report.gates_passed["offset"] = False
    MDMAdapter.nfe = 0

    regimes = ["suffix", "infill"] if REGIME == "both" else [REGIME]
    all_recs, all_stats = [], {}
    for rg in regimes:
        blocks = make_blocks(wins_probe, rg)
        note(f"=== regime={rg}: {len(blocks)} blocks from "
             f"{len({b['doc_id'] for b in blocks})} documents ===")
        r, s = measure(ad, blocks)
        all_recs += r
        all_stats[rg] = s

    out = {
        "config": {
            "model_tag": MODEL_TAG, "repo": repo, "model_path": snapshot(repo),
            "block_size": K, "m_meter": M_METER, "m_diag": M_DIAG,
            "r_orders": R_ORDERS, "window": WINDOW, "regimes": regimes,
            "max_blocks_per_regime": MAX_BLOCKS, "seed": SEED,
            "corpus": os.path.basename(CORPUS), "topm": TOPM,
            "ablated_offset": os.environ.get("ABLATE_OFFSET"),
            "total_forward_passes": MDMAdapter.nfe,
        },
        "gates": report_to_dict(ad.report),
        "run_stats": all_stats,
        "records": all_recs,
    }
    suffix = os.environ.get("OUT_SUFFIX", "")
    path = os.path.join(RESULTS, f"e5v3_{MODEL_TAG}{suffix}.json")
    with open(path, "w") as f:
        json.dump(out, f)
    note(f"wrote {path} ({len(all_recs)} records, {MDMAdapter.nfe} forwards)")


if __name__ == "__main__":
    main()
