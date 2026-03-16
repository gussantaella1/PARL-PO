import importlib


import numpy as np
import torch



import torch

from datetime import datetime


from pathlib import Path




# --- Single source of truth for config & dynamics ---
import config_rl
importlib.reload(config_rl)
from config_rl import config_for_train, config_for_eval, build_dyn


from ukf_estimator import AgentUKF, _body_bearing_from_world, _azel_from_body_vec


# =============================================================
# Freeze / verification helpers
# =============================================================

def freeze_module_(m: torch.nn.Module):
    """Hard-freeze: eval() + requires_grad_(False) on all params."""
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)

@torch.no_grad()
def snapshot_state_dict(m: torch.nn.Module) -> dict:
    """CPU clone of state_dict tensors for exact comparison."""
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

@torch.no_grad()
def max_state_dict_diff(snap: dict, m: torch.nn.Module) -> float:
    """Max |param - snap| over all tensors in state_dict."""
    cur = m.state_dict()
    diffs = []
    for k, v0 in snap.items():
        v = cur[k].detach().cpu()
        diffs.append((v - v0).abs().max().item())
    return float(max(diffs) if diffs else 0.0)

@torch.no_grad()
def assert_frozen_unchanged(snap: dict, m: torch.nn.Module, name: str, tol: float = 0.0):
    d = max_state_dict_diff(snap, m)
    if d > tol:
        raise RuntimeError(f"[FREEZE CHECK FAILED] {name} changed! max|Δ|={d:.3e} > tol={tol:.3e}")
    return d

@torch.no_grad()
def assert_deterministic_action(ppo, obs_batch: torch.Tensor, who: str, tol: float = 0.0):
    a1, _, _ = ppo.act(obs_batch, who=who, deterministic=True)
    a2, _, _ = ppo.act(obs_batch, who=who, deterministic=True)
    d = (a1 - a2).abs().max().item()
    if d > tol:
        raise RuntimeError(f"[DETERMINISM CHECK FAILED] {who} deterministic act differs: max|Δa|={d:.3e}")
    return d

