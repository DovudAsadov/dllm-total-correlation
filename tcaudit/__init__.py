"""
tcaudit -- Auditing commit-set selection proxies in diffusion LLMs against the
ground-truth conditional total correlation they are supposed to approximate.

Contribution map:
  1. proxy-fidelity audit         -> audit.run_audit         (CPU, oracle)
  2. total-correlation meter      -> meter.meter_total_correlation
  3. real-model confirmation      -> model_backend.ModelBackend (GPU)
  4. matched-marginal separation  -> audit.matched_marginal_separation (CPU)

Ground truth engine: groundtruth.enumerate_ground_truth (exact, CPU).
"""
from . import audit, backends, groundtruth, information, meter, proxies, stats, tasks  # noqa: F401

__version__ = "0.1.0"
