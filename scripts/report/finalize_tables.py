#!/usr/bin/env python3
"""
Move each table's \\caption/\\label above its tabular.

Why this exists. neurips_2026.sty redefines the `table` environment and swaps the two
caption skips inside it:

    \\setlength{\\abovecaptionskip}{\\@neuripsbelowcaptionskip}   % 0pt
    \\setlength{\\belowcaptionskip}{\\@neuripsabovecaptionskip}   % 7pt

i.e. it assumes the NeurIPS convention of captions ABOVE tables. A caption placed below
the tabular is therefore governed by \\abovecaptionskip = 0pt and sits flush against the
bottom rule. The caption package's `skip=` option cannot fix it (verified: no visible
change even at skip=30pt), and \\AtBeginEnvironment cannot either, because it prepends to
`table` and the style's \\setlength then runs afterwards and wins.

Putting the caption above is both the house style and the only fix that needs no hack:
\\belowcaptionskip is already 7pt.

Idempotent: a table whose caption is already above is left alone. Run after the
make_*_table(s) scripts.
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


def move_caption_above(tex: str) -> tuple[str, bool]:
    tab = tex.find(r"\begin{tabular}")
    cap = tex.find(r"\caption{")
    if tab < 0 or cap < 0 or cap < tab:
        return tex, False                     # no table, or already above

    end = brace_span(tex, cap + len(r"\caption"))
    block = tex[cap:end]
    rest = tex[end:]
    m = re.match(r"\s*\\label\{[^}]*\}", rest)
    if m:
        block += m.group(0).strip()
        end += m.end()

    without = tex[:cap].rstrip() + "\n" + tex[end:].lstrip()
    tab = without.find(r"\begin{tabular}")
    # no extra \vspace: with the caption above, the style's \belowcaptionskip (7pt)
    # already supplies the gap, and adding more costs page budget for nothing.
    return without[:tab] + block + "\n" + without[tab:], True


def main() -> int:
    files = sorted(glob.glob(os.path.join(ROOT, "paper", "tables", "*.tex")))
    moved = 0
    for f in files:
        src = open(f).read()
        out, did = move_caption_above(src)
        if did:
            open(f, "w").write(out)
            moved += 1
        print(f"  {'moved ' if did else 'already'}  {os.path.basename(f)}")
    print(f"{moved}/{len(files)} table captions moved above the tabular")
    return 0


if __name__ == "__main__":
    sys.exit(main())
