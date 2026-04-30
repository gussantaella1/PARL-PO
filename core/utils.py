import math
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# =============================================================
# Utils
# =============================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def resolve_start_radius_bounds(
    cfg: Dict[str, Any],
    arena_radius: float,
    *,
    who: str,
    default_min_frac: float = 0.0,
    default_max_frac: float = 1.0,
) -> Tuple[float, float]:
    """
    Resolve training/eval shell bounds for the defender or attacker.

    Fractional keys such as `r_def_min` are interpreted relative to the arena
    radius for backward compatibility. Explicit meter overrides like
    `r_def_min_m` take precedence when present.
    """
    who = str(who).strip().lower()
    if who not in {"def", "att"}:
        raise ValueError(f"Unsupported shell owner {who!r}; expected 'def' or 'att'.")

    def _resolve(bound: str, default_frac: float) -> float:
        frac_key = f"r_{who}_{bound}"
        meter_key = f"{frac_key}_m"
        meter_val = cfg.get(meter_key, None)
        if meter_val is not None:
            radius = float(meter_val)
            source_key = meter_key
        else:
            radius = float(cfg.get(frac_key, default_frac)) * float(arena_radius)
            source_key = frac_key
        if radius < 0.0:
            raise ValueError(f"{source_key} must be >= 0, got {radius}.")
        return radius

    return _resolve("min", default_min_frac), _resolve("max", default_max_frac)


def atanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -0.999999, 0.999999)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def squash_action(u_raw: torch.Tensor, act_scale: float) -> torch.Tensor:
    return torch.tanh(u_raw) * act_scale


def logprob_squashed(dist: torch.distributions.Normal, u_raw: torch.Tensor) -> torch.Tensor:
    # Stable tanh-squash correction:
    # log(1 - tanh(u)^2) = 2 * (log 2 - u - softplus(-2u))
    logp = dist.log_prob(u_raw).sum(-1)
    correction = (2.0 * (math.log(2.0) - u_raw - F.softplus(-2.0 * u_raw))).sum(-1)
    return logp - correction
