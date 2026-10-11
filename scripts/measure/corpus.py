#!/usr/bin/env python3
"""
Build data/corpus.jsonl for E5 v2: real documents from two public sources, one JSON
object per document, so each measured block is attributable to a distinct document
(the cluster unit for the bootstrap).

v1 used eight hand-written strings such as "The mitochondria is the powerhouse of the
cell", which is why its 48 blocks were really 8 clusters and its intervals were wide.

Sources (both public, plain HTTPS, no `datasets`/parquet dependency):
  * prose : WikiText-103 raw, test + validation splits, pulled through the HF
            datasets-server JSON rows API and reassembled from lines into articles.
  * code  : OpenAI HumanEval, 164 real Python functions -- the standard code benchmark
            and the natural domain for Dream-Coder.

Writes data/corpus.jsonl with fields {id, domain, text}.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

ROWS_API = ("https://datasets-server.huggingface.co/rows"
            "?dataset=Salesforce%2Fwikitext&config=wikitext-103-raw-v1"
            "&split={split}&offset={off}&length=100")
HUMANEVAL = ("https://raw.githubusercontent.com/openai/human-eval/master/data/"
             "HumanEval.jsonl.gz")

N_PROSE = int(os.environ.get("N_PROSE", "400"))
# test+validation together yield only ~120 usable articles, which caps the deployed
# (suffix) regime at one block per document. The train split lifts that ceiling;
# MAX_TRAIN_LINES bounds the fetch so the run stays a few minutes.
SPLITS = tuple(x for x in os.environ.get("SPLITS", "test,validation").split(",") if x)
MAX_TRAIN_LINES = int(os.environ.get("MAX_TRAIN_LINES", "120000"))
N_CODE = int(os.environ.get("N_CODE", "164"))
MIN_CHARS = int(os.environ.get("MIN_CHARS", "600"))


def get(url: str, tries: int = 6) -> bytes:
    """The datasets-server rate-limits bursts; back off and retry rather than losing
    the whole split to a single 429."""
    delay = 2.0
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503, 504) and attempt < tries - 1:
                print(f"    {e.code}; retrying in {delay:.0f}s", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise
    raise RuntimeError("unreachable")


def _wikitext_lines() -> list:
    lines = []
    for split in SPLITS:
        off, total = 0, None
        while total is None or off < total:
            obj = json.loads(get(ROWS_API.format(split=split, off=off)).decode("utf-8"))
            if total is None:
                total = obj["num_rows_total"]
                print(f"  wikitext/{split}: {total} lines", flush=True)
            rows = obj.get("rows", [])
            if not rows:
                break
            lines += [r["row"]["text"] for r in rows]
            off += len(rows)
            if split == "train" and off >= MAX_TRAIN_LINES:
                print(f"  wikitext/train: stopping at {off} lines", flush=True)
                break
            time.sleep(0.4)   # stay under the burst limit
    return lines


def prose_docs() -> list:
    """A level-1 heading ' = Title = ' starts an article; deeper headings are section
    breaks within it."""
    lines = _wikitext_lines()
    docs, title, buf = [], None, []
    h1 = re.compile(r"^ = ([^=].*?) = $")
    hn = re.compile(r"^ =+ .* =+ $")

    def flush():
        if title and buf:
            body = "\n".join(buf)
            body = (body.replace(" @-@ ", "-").replace(" @,@ ", ",")
                        .replace(" @.@ ", "."))
            body = re.sub(r"[ \t]+", " ", body).strip()
            if len(body) >= MIN_CHARS:
                docs.append({"id": f"wikitext:{title[:60]}", "domain": "prose",
                             "text": body})

    for ln in lines:
        t = ln.rstrip("\n")
        m = h1.match(t)
        if m:
            flush()
            title, buf = m.group(1).strip(), []
            continue
        if hn.match(t):
            continue
        if t.strip():
            buf.append(t.strip())
    flush()
    print(f"  wikitext: {len(docs)} articles >= {MIN_CHARS} chars", flush=True)
    return docs[:N_PROSE]


def code_docs() -> list:
    lines = gzip.decompress(get(HUMANEVAL)).decode("utf-8").strip().split("\n")
    docs = []
    for l in lines:
        o = json.loads(l)
        docs.append({"id": f"humaneval:{o['task_id']}", "domain": "code",
                     "text": o["prompt"] + o["canonical_solution"]})
    print(f"  humaneval: {len(docs)} problems", flush=True)
    return docs[:N_CODE]


def main():
    print("fetching corpus ...", flush=True)
    docs = []
    for fn, name in ((prose_docs, "prose"), (code_docs, "code")):
        try:
            docs += fn()
        except Exception as e:
            print(f"  !! {name} fetch failed: {e}", file=sys.stderr)
    if not docs:
        raise SystemExit("no documents fetched")

    # Never destroy a good corpus with a worse one. A partial fetch (e.g. the
    # datasets-server returning 429 on the prose split) previously overwrote a
    # 285-document mixed corpus with a 164-document code-only one, silently changing
    # what every downstream measurement was computed on. Write atomically, and refuse
    # to replace an existing corpus that is larger or covers more domains.
    path = os.path.join(DATA, "corpus.jsonl")
    doms = {d["domain"] for d in docs}
    if os.path.exists(path) and os.environ.get("FORCE_OVERWRITE") != "1":
        old_docs = [json.loads(l) for l in open(path)]
        old_doms = {d["domain"] for d in old_docs}
        if len(docs) < len(old_docs) or not old_doms <= doms:
            raise SystemExit(
                f"refusing to overwrite {path}: existing corpus has {len(old_docs)} docs "
                f"over {sorted(old_doms)}, new fetch has {len(docs)} over {sorted(doms)}. "
                f"A split probably failed. Set FORCE_OVERWRITE=1 to override.")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    os.replace(tmp, path)
    n_p = sum(1 for d in docs if d["domain"] == "prose")
    n_c = sum(1 for d in docs if d["domain"] == "code")
    print(f"wrote {path}: {len(docs)} documents ({n_p} prose, {n_c} code)")


if __name__ == "__main__":
    main()
