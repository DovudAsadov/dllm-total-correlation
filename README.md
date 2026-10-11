# How Much Dependency Can a Marginal Rule See?

**Total Correlation in Diffusion LM Parallel Decoding**

Dovud Asadov, Rustambek Urokov, Qobiljon Toshnazarov, Kayumov Abduaziz, Mardon Hazratov,
Husanboy Mansuraliyev
New Uzbekistan University

NeurIPS 2026 Workshop on Diffusion Language Models (DiffuLM), poster. [Paper (PDF)](paper/main.pdf)

## Overview

Masked diffusion language models decode in parallel, committing several positions per step.
The cost of committing a set at once is its conditional total correlation `C`. Most parallel
decoders choose that set from per-position marginals, which bound `C` but do not determine it.
This work measures how much of `C` a marginal rule can actually see.

## Findings

- We estimate `C` for each candidate set on four checkpoints: Dream-Coder 7B (base and
  instruct) and LLaDA 8B (base and instruct).
- Outside Dream-instruct, the best marginal-only predictor leaves a residual of 0.74 to 1.17
  bits, mostly because blocks use a stable fraction of the window the marginals certify.
- Dream and LLaDA read their logits at opposite positions. Every measurement runs behind
  three gates (readout offset, determinism, fluency) that measure this convention instead
  of assuming it.

## Reproducibility

`results/` holds every per-block measurement and derived analysis behind the paper. The
tables and figures in `paper/` are generated from these files, and every number the paper
quotes is recomputed and checked against them by `scripts/report/check_claims.py`. The tag
`paper-numbers` marks the camera-ready version.

The analysis needs Python 3.10+, NumPy and Matplotlib. The measurements ran on one NVIDIA
A100 80GB with PyTorch and `transformers==4.46.2` (Dream's remote code does not load under
5.x). The verification appendix ran on an NVIDIA GB10.

The per-block files `results/e5v3_*.json` were measured before three later changes: Gate 2
now bounds the entropy change across batch shapes, Gate 3 can fail, and the sampler is
seeded. A rerun is therefore reproducible but not bit-identical to those files.

## Citation

```bibtex
@inproceedings{asadov2026marginal,
  title     = {How Much Dependency Can a Marginal Rule See? Total Correlation in
               Diffusion LM Parallel Decoding},
  author    = {Asadov, Dovud and Urokov, Rustambek and Toshnazarov, Qobiljon and
               Kayumov, Abduaziz and Hazratov, Mardon and Mansuraliyev, Husanboy},
  booktitle = {NeurIPS 2026 Workshop on Diffusion Language Models (DiffuLM)},
  year      = {2026}
}
```
