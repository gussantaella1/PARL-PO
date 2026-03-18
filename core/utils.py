import math
import numpy as np
import torch
import torch.nn.functional as F

# =============================================================
# Utils
# =============================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


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
