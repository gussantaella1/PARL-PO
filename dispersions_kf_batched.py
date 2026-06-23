"""
Batch-friendly KF Monte Carlo dispersions.

This keeps the same initial-state rollout dispersions as dispersions.py and keeps
the fixed KF parameter means, but removes per-trial KF parameter randomness so
evaluate_policy.py can use the batched CUDA/EKF eval path.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

from dispersions import DISPERSIONS as BASE_DISPERSIONS


DISPERSIONS: Dict[str, Any] = copy.deepcopy(BASE_DISPERSIONS)

# Keep fixed KF parameter means, but remove per-trial KF parameter draws. The
# main dispersions.py file currently randomizes action_meas_std from config;
# that blocks batched eval because each trial can get a different KF config.
for _spec in (DISPERSIONS.get("kf", {}).get("parameters", {}) or {}).values():
    if isinstance(_spec, dict):
        _spec["std"] = 0.0


def config_for_eval() -> Dict[str, Any]:
    """Return a copy of the batch-friendly dispersion config."""
    return copy.deepcopy(DISPERSIONS)
