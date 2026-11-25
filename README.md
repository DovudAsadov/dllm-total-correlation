# How Much Dependency Can a Marginal Rule See?

**Total Correlation in Diffusion LM Parallel Decoding**

Dovud Asadov\*, Rustambek Urokov\*, Qobiljon Toshnazarov, Abduaziz Kayumov, Mardon Hazratov,
Husanboy Mansuraliyev (\*equal contribution)
New Uzbekistan University

NeurIPS 2026 Workshop on Diffusion Language Models (DiffuLM).
[arXiv:2611.31990](https://arxiv.org/abs/2611.31990) | [Paper (PDF)](paper/main.pdf)

## Overview

A masked diffusion language model commits several tokens per step. Each token is sampled
from its own marginal, so the dependency between the tokens is lost. The size of that loss
is the conditional total correlation `C` of the committed set. Most decoders choose the set
from per-position marginals, which can bound `C` but cannot determine it. This repository
holds the code, measurements and paper for a study of how well the marginals predict `C`
on real checkpoints.

## Findings

- We estimate `C` per candidate set on four checkpoints: Dream-Coder 7B and LLaDA 8B, base
  and instruct. On three of the four, the best marginal-only predictor is off by 0.74 to
  1.17 bits, 1.8 to 3.7 times less than a reference that knows only the bound `W`. Most of
  that accuracy comes from the bound itself.
- Dream-Coder 7B instruct is the exception, mainly on prose, where its conditionals are
  least self-consistent.
- Among six common marginal rules, pooling over the set beats gating on its weakest
  position in all 24 paired comparisons.
- Dream and LLaDA place a position's prediction at different logit indices. Three gates
  (readout offset, forward-pass determinism, decode-and-read) measure the convention
  before any number is produced.

## Reproducing the numbers

The analysis needs Python 3.10+, NumPy and Matplotlib. Every number the paper quotes is
recomputed from `results/` and checked against the text by:

```
python scripts/report/check_claims.py
```

The measurements ran on one NVIDIA A100 80GB with PyTorch and `transformers==4.46.2`
(Dream's remote code does not load under 5.x). The verification appendix ran on an NVIDIA
GB10. The tag `paper-numbers` marks the results behind the paper's numbers.

The per-block files `results/e5v3_*.json` were measured before three later changes: Gate 2
now bounds the entropy change across batch shapes, Gate 3 can fail, and the sampler is
seeded. A rerun is reproducible but not bit-identical to those files.

## Citation

```bibtex
@inproceedings{asadov2026marginal,
  title     = {How Much Dependency Can a Marginal Rule See? Total Correlation in
               Diffusion LM Parallel Decoding},
  author    = {Asadov, Dovud and Urokov, Rustambek and Toshnazarov, Qobiljon and
               Kayumov, Abduaziz and Hazratov, Mardon and Mansuraliyev, Husanboy},
  booktitle = {NeurIPS 2026 Workshop on Diffusion Language Models (DiffuLM)},
  year      = {2026},
  eprint    = {2611.31990},
  archivePrefix = {arXiv}
}
```
