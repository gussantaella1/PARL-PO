from __future__ import annotations

import torch

from core.utils import atanh, logprob_squashed, set_seed, squash_action


def _obs_offsets(D: int, Na: int):
    """
    Observation layout without optional trailing extras:
      [p1c, pA0..pA{Na-1}, rel0..rel{Na-1}, v1, vA0..vA{Na-1}]
    """
    off_p1 = 0
    off_pA = D
    off_rel = D + Na * D
    off_v1 = D + 2 * Na * D
    off_vA = off_v1 + D
    return off_p1, off_pA, off_rel, off_v1, off_vA


def permute_obs_for_attacker(
    obs: torch.Tensor,
    idx: int,
    D: int,
    Na: int,
) -> torch.Tensor:
    """
    Reorder attacker-centric blocks so attacker `idx` appears in slot 0.

    Any trailing observation extras, such as fuel scalars, are kept unchanged.
    """
    if Na <= 1 or idx == 0:
        return obs

    base_dim = (2 + 3 * Na) * D
    if obs.shape[-1] < base_dim:
        raise ValueError(
            f"Expected obs dim >= {base_dim} for D={D}, Na={Na}, got {obs.shape[-1]}"
        )

    off_p1, off_pA, off_rel, off_v1, off_vA = _obs_offsets(D, Na)

    p1c = obs[:, off_p1:off_p1 + D]
    pA = [obs[:, off_pA + k * D:off_pA + (k + 1) * D] for k in range(Na)]
    rel = [obs[:, off_rel + k * D:off_rel + (k + 1) * D] for k in range(Na)]
    v1 = obs[:, off_v1:off_v1 + D]
    vA = [obs[:, off_vA + k * D:off_vA + (k + 1) * D] for k in range(Na)]
    extras = obs[:, base_dim:]

    order = [idx] + [k for k in range(Na) if k != idx]
    parts = [p1c]
    parts += [pA[k] for k in order]
    parts += [rel[k] for k in order]
    parts += [v1]
    parts += [vA[k] for k in order]
    if extras.shape[-1] > 0:
        parts.append(extras)
    return torch.cat(parts, dim=-1)


__all__ = [
    "set_seed",
    "atanh",
    "squash_action",
    "logprob_squashed",
    "_obs_offsets",
    "permute_obs_for_attacker",
]
