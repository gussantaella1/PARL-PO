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
    # "att_rule": {
    #     "ridge": 1e-2,        # one-step ridge in u_center
    #     "w_center": 1.0,      # weight for center pull
    #     "w_avoid":  2.0,      # weight for repulsion
    #     "w_damp":   0.3,      # velocity damping
    #     "min_sep":  4.0,      # (m) soft floor for repulsion distance
    #     "repulse_gain": 30.0,  # overall strength of repulsion ~ 1/r^3
    # },

    "att_rule": {
        "ridge": 2e-2,       # slightly smoother center controller
        "w_center": 1.0,     # still wants the center
        "w_avoid":  2.5,     # stronger fear
        "w_damp":   0.3,     # more damping to avoid jitter
        "min_sep":  4.0,     # reacts earlier to defender
        "repulse_gain": 1.0, # tuned so repulsion doesn't instantly saturate
    },

    # "att_rule": {
    #     "ridge": 5e-2,       # slightly smoother center controller
    #     "w_center": 0.8,     # still wants the center
    #     "w_avoid":  3.0,     # stronger fear
    #     "w_damp":   0.5,     # more damping to avoid jitter
    #     "min_sep":  6.0,     # reacts earlier to defender
    #     "repulse_gain": 1.0, # tuned so repulsion doesn't instantly saturate
    # },

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

# ---------- Kalman / measurement subconfig ----------
ARENA_R = float(COMMON["arena"]["r"])

KF_COMMON: Dict[str, Any] = {
    "use_ukf": True,
    "use_meas_reward": True,
    "meas_innov_coef": 0.0,   # start at 0, then slowly crank up
    "meas_cov_coef": 0.0,
    "ukf": {
        "sigma_az": np.deg2rad(0.5),
        "sigma_el": np.deg2rad(0.5),
        "pos_std0": 0.2 * ARENA_R,
        "vel_std0": 0.01,
        "init_pos_std": 0.2 * ARENA_R,
        "init_vel_std": 0.01,
        "Q_scale": 1e-5,
    },
    "reward_from_belief": True,   # NEW: toggle

    "belief_clip_factor": 2.0,
}

# ---------- Visualizer config ----------

VIZ: Dict[str, Any] = {
    "viz": {
        "axis_scale": 1e0,       # labels like x (10^2 m)
        "axis_unit": "m",
        "axis_label_only": True,
        "triad_len": (0.5, 0.5, 0.7),
        "triad_colors": ("tab:red", "tab:green", "tab:blue"),
        "triad_labels": ("x_b (boresight)", "y_b", "z_b"),
        "triad_leg_loc": "lower left",
        "triad_leg_ncol": 3,
        "triad_leg_title": "Body axes",
        "only_est": False,
        "show_meas": True,
        "meas_len": 20.0,
    },
}


# merge KF settings into COMMON so config_for_train/eval see them
COMMON = _merge(COMMON, KF_COMMON)
COMMON = _merge(COMMON,VIZ)

# ---------- TRAIN (training-only knobs) ----------
# TRAIN: Dict[str, Any] = {
#     # Vectorized rollout & logging
#     # "num_envs": 64,          # was 8
#     # "steps_per_env": 512,    # was 256
#     # "total_updates": 2000,   # was 300

#     "num_envs": 8,          # was 8
#     "steps_per_env": 256,    # was 256
#     "total_updates": 300,   # was 300

#     "log_every": 10,

#     # PPO hyperparams
#     "gamma": 0.99,
#     "gae_lambda": 0.95,
#     "clip_eps": 0.2,
#     "policy_lr": 3e-4,
#     "value_lr": 1e-3,
#     "train_epochs": 10,
#     "minibatch_size": 1024,
#     "entropy_coef": 0.02,
#     "value_coef": 0.5,
#     "max_grad_norm": 1.0,

#     "lr_schedule": "linear",   # options: "none", "linear"
#     "lr_final_factor": 0.1,  # final_lr = lr_final_factor * initial_lr

#     # Anneal (training-time only)
#     "def_center_min_anneal": 0.5,

#     # Training-only initial condition randomization
#     # (env uses this; eval config will default back to "fixed")
#     "train_ic_mode": "random_shell",  # or "fixed"
#     "train_ic_vmax": 0.05,            # max |v| component at t=0
#     "train_min_sep": 1.0,             # min defender–attacker separation (m)

#     "def_center_safe_radius": 0.10,   # e.g. keep-out inside 10% of R
#     "def_center_avoid_coef":  50.0,   # crank this up if defender still dips in
#     "def_center_coef":        0.0,    # no attractive center tether
#     "prior_blend_def":        0.0,    # (optional) disable center prior for def


#     "att_target_hit_radius": 0.05,          # attacker within 5% of R hits object
#     "att_target_hit_penalty_def": 5.0,      # big negative for defender
#     "att_target_hit_reward_att": 5.0,       # matching positive for attacker

# }

#Long training config below:

# ---------- TRAIN (long-run, more stable) ----------
TRAIN: Dict[str, Any] = {
    # Vectorized rollout & logging
    # Total samples per update = num_envs * steps_per_env
    # Here: 32 * 512 = 16,384 steps/update (8x your original short run)
    "num_envs": 32,
    "steps_per_env": 512,
    "total_updates": 1200,     # long but not crazy; ~19.7M steps

    "log_every": 10,

    # PPO hyperparams
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,

    # ↓ Smaller LRs than your short-run config (since batch is much larger)
    "policy_lr": 1e-4,         # was 3e-4
    "value_lr": 3e-4,          # was 1e-3

    "train_epochs": 10,
    "minibatch_size": 1024,    # 16 minibatches per update w/ 16,384 steps
    "entropy_coef": 0.02,
    "value_coef": 0.5,
    "max_grad_norm": 1.0,

    "lr_schedule": "linear",   # options: "none", "linear"
    "lr_final_factor": 0.1,    # final_lr = lr_final_factor * initial_lr

    # Anneal (training-time only)
    "def_center_min_anneal": 0.5,

    # Training-only initial condition randomization
    "train_ic_mode": "random_shell",  # or "fixed"
    "train_ic_vmax": 0.05,            # max |v| component at t=0
    "train_min_sep": 1.0,             # min defender–attacker separation (m)

    # Defender vs center behaviour
    "def_center_safe_radius": 0.10,   # keep-out inside 10% of R
    "def_center_avoid_coef":  50.0,   # penalty for entering this zone
    "def_center_coef":        0.0,    # no attractive tether to center
    "prior_blend_def":        0.0,    # disable center prior for def (RL only)

    # Terminal hit logic
    "att_target_hit_radius":       0.05,   # attacker within 5% of R hits target
    "att_target_hit_penalty_def": 10.0,    # was 5.0 → stronger slap to defender
    "att_target_hit_reward_att":   5.0,    # attacker reward unchanged
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
