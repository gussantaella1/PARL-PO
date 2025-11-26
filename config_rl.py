# config_rl.py
from __future__ import annotations
from typing import Dict, Any
import numpy as np
from copy import deepcopy

# ---------- helpers ----------
def _dcopy(x):
    if isinstance(x, dict):
        return {k: _dcopy(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x.copy()
    return deepcopy(x)

def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge dict b into a (non-mutating), arrays copied."""
    out = _dcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = _dcopy(v)
    return out

# ---------- COMMON (shared by train & eval/rollout) ----------
COMMON: Dict[str, Any] = {
    "seed": 42,
    "device": "cuda",  # "cuda" if available

    # Dynamics / horizon
    "D": 3,
    "dt": 0.1,
    # "T": 240,
    "T": 600,
    "dynamics": "hcw",
    "hcw": {"mu": 3.986004418e14, "r0": 6_371_000.0 + 400_000.0},

    # Arena
    "arena": {"type": "sphere", "cx": 0.0, "cy": 0.0, "cz": 0.0, "r": 20.0},
    "arena_terminate_margin": 1.0,
    "soft_wall_start": 0.5,
    "wall_penalty": 3.0,

    # Action bounds
    "umax": 2,

    # Initial conditions
    "x0": np.array([
        [  2.5, 0.0, 0.0,  -0.02, 0.00, 0.000],  # defender
        [ -19.9, 0.0, 0.0,  +0.02, 0.00, 0.000], # attacker
    ], dtype=float),
    "x0_jitter": {"pos": 0.5, "vel": 0.01},

    # Reward shaping (used by Env at train *and* eval)
    "dense_coef": 0.03,   # α
    "term_coef": 1.0,     # β
    "step_pos_coef": 0.01,
    "step_rel_coef": 0.005,
    "def_center_coef": 0.05,
    "step_vel_coef": 0.0005,
    "effort_def": 0.01,
    "effort_att": 0.01,

    # DiffNash prior (actor mean blend)
    "prior_ridge": 1e-2,
    "prior_blend_def": 0.5,
    "prior_blend_att": 1.0,

    # Attacker control selection
    #   "rule" => deterministic controller
    #   "rl"   => learned attacker (if your training loop supports it)
    "attacker_mode": "rule",

    # Rule-based attacker parameters
    "att_rule": {
        "ridge": 1e-2,        # one-step ridge in u_center
        "w_center": 1.0,      # weight for center pull
        "w_avoid":  2.0,      # weight for repulsion
        "w_damp":   0.3,      # velocity damping
        "min_sep":  4.0,      # (m) soft floor for repulsion distance
        "repulse_gain": 10.0,  # overall strength of repulsion ~ 1/r^3
    },

    # Filled by build_dyn()
    "dyn": {"Ad": None, "Bd": None},

    "oi": {
        "enabled": True,
        "cx": 0.0, "cy": 0.0, "cz": 0.0,  # omit cz in 2D
        "r":  1,                        # keep-out radius (m)
        "avoid_by": [1],                   # only player 1 must avoid
        "color": "tab:purple",
        "alpha": 0.18,
        "edgecolor": "k"        
    },
}

# ---------- TRAIN (training-only knobs) ----------
TRAIN: Dict[str, Any] = {
    # Vectorized rollout & logging
    # "num_envs": 64,          # was 8
    # "steps_per_env": 512,    # was 256
    # "total_updates": 2000,   # was 300

    "num_envs": 8,          # was 8
    "steps_per_env": 256,    # was 256
    "total_updates": 300,   # was 300

    "log_every": 10,

    # PPO hyperparams
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "policy_lr": 3e-4,
    "value_lr": 1e-3,
    "train_epochs": 10,
    "minibatch_size": 1024,
    "entropy_coef": 0.02,
    "value_coef": 0.5,
    "max_grad_norm": 1.0,

    "lr_schedule": "linear",   # options: "none", "linear"
    "lr_final_factor": 0.1,  # final_lr = lr_final_factor * initial_lr

    # Anneal (training-time only)
    "def_center_min_anneal": 0.5,

    # Training-only initial condition randomization
    # (env uses this; eval config will default back to "fixed")
    "train_ic_mode": "random_shell",  # or "fixed"
    "train_ic_vmax": 0.05,            # max |v| component at t=0
    "train_min_sep": 1.0,             # min defender–attacker separation (m)
}


# ---------- Public getters ----------
def config_for_train(**overrides) -> Dict[str, Any]:
    """
    Training config = deep-merge(COMMON, TRAIN), then apply **overrides.
    """
    cfg = _merge(COMMON, TRAIN)
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg

def config_for_eval(**overrides) -> Dict[str, Any]:
    """
    Evaluation/rollout config from COMMON with deterministic ICs and no training:
      - x0_jitter OFF
      - single env
      - steps_per_env = T
      - total_updates = 0
    """
    cfg = _dcopy(COMMON)
    # Safe rollout defaults
    cfg["x0_jitter"] = {"pos": 0.0, "vel": 0.0}
    cfg["num_envs"] = 1
    cfg["steps_per_env"] = cfg["T"]
    cfg["total_updates"] = 0
    cfg["log_every"] = 0
    if overrides:
        cfg = _merge(cfg, overrides)
    # ensure placeholders exist
    cfg.setdefault("dyn", {}).setdefault("Ad", None)
    cfg.setdefault("dyn", {}).setdefault("Bd", None)
    return cfg

# ---------- Dynamics builder ----------
def build_dyn(cfg: Dict[str, Any]):
    from dyn_models import hcw_mean_motion, hcw_discrete_mats, as_numpy_const
    assert cfg["dynamics"].lower() == "hcw", "Only HCW supported in this helper."
    n = hcw_mean_motion(cfg["hcw"])
    Ad, Bd = hcw_discrete_mats(float(n), float(cfg["dt"]))
    cfg["dyn"]["Ad"] = as_numpy_const(Ad).astype(np.float32)
    cfg["dyn"]["Bd"] = as_numpy_const(Bd).astype(np.float32)
