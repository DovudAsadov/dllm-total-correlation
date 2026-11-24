#!/usr/bin/env python3
"""
Place each table's \\caption and \\label below its tabular, just before \\end{table}.

The spacing between a table and its caption is set once in the paper's preamble, so no
table needs its own \\vspace. Idempotent: a table whose caption is already below its
tabular is left alone. Run after the other table scripts.
"""
from __future__ import annotations

import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def brace_span(s: str, start: int) -> int:
    """Index just past the balanced {...} group beginning at s[start] == '{'."""
    assert s[start] == "{"
    depth = 0
    i = start
    while i < len(s):
        if s[i] == "\\":            # skip escaped char
            i += 2
            continue
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced braces")


def move_caption_below(tex: str) -> tuple[str, bool]:
    tab = tex.find(r"\begin{tabular}")
    cap = tex.find(r"\caption{")
    if tab < 0 or cap < 0 or cap > tab:
        return tex, False                     # no table, or already below

    end = brace_span(tex, cap + len(r"\caption"))
    block = tex[cap:end]
    m = re.match(r"\s*\\label\{[^}]*\}", tex[end:])
    if m:
        block += m.group(0).strip()
        end += m.end()

    without = tex[:cap].rstrip() + "\n" + tex[end:].lstrip()
    close = without.rfind(r"\end{table}")
    assert close > 0, "no \\end{table}"
    return without[:close].rstrip() + "\n" + block + "\n" + without[close:], True


def main() -> int:
    files = sorted(glob.glob(os.path.join(ROOT, "paper", "tables", "*.tex")))
    moved = 0
    for f in files:
        src = open(f).read()
        out, did = move_caption_below(src)
        if did:
            open(f, "w").write(out)
            moved += 1
        print(f"  {'moved ' if did else 'already'}  {os.path.basename(f)}")
    print(f"{moved}/{len(files)} table captions moved below the tabular")
    return 0


if __name__ == "__main__":
    sys.exit(main())
