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
    "num_attackers": 1,   # default
    "seed": 42,
    "device": "cuda",  # "cuda" if available

    # Dynamics / horizon
    "D": 3,
    "dt": 0.1,
    # "T": 240,
    "T": 600,
    "dynamics": "hcw",
    "hcw": {"mu": 3.986004418e14, "r0": 6_371_000.0 + 400_000.0},

    # NEW (chief orbit for elliptical/two-body)
    "chief_orbit": {
        "mu": 3.986004418e14,
        "a":  6_371_000.0 + 400_000.0,  # semi-major axis (m)
        "e":  0.0,                      # eccentricity
        "i":  0.0,                      # rad
        "raan": 0.0,                    # rad
        "argp": 0.0,                    # rad
        "nu0":  0.0,                    # rad (true anomaly at t=0)
    },

    # Dyn container (extended)
    "dyn": {
        "type": None,     # "lti" | "ltv" | "nonlinear"
        "Ad": None, "Bd": None,
        "Ad_seq": None, "Bd_seq": None, # for LTV
        "model": None,                  # "hcw" | "two_body_rtn"
        "chief_cache": None,            # for two_body / elliptic linearization
    },

    # Arena
    "arena": {"type": "sphere", "cx": 0.0, "cy": 0.0, "cz": 0.0, "r": 20.0},
    "arena_terminate_margin": 1.0,
    "soft_wall_start": 0.5,
    "wall_penalty": 3.0,

    # Action bounds
    "umax": 2.0,

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
    "attacker_mode": "rl",

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

    #Smart attacker 

    # "att_reward": {

    #     # --- progress / goal terms ---
    #     # Reward *decreasing* d2 each step (encourages steady approach, less dithering)
    #     "k_prog":  4.0,     # weight on (-delta_d2)

    #     # Small shaping toward being near the center (optional)
    #     # If your r2 already uses "-k_pos * d2", you can keep this 0.
    #     "k_cent":  0.5,

    #     # --- interaction with defender (conservative vs reckless) ---
    #     # Penalty for being too close to defender (prevents "suicide charges")
    #     "min_sep": 2.0,     # meters; MUST exist if your r2 uses self.att_min_sep
    #     "k_close": 1.0,     # how hard to penalize inside min_sep

    #     # --- speed / overshoot control ---
    #     # Stronger velocity penalty makes it stop overshooting center and stop slamming walls
    #     # (this adds to your existing -k_vel*||v2||^2 if you keep that in r2)
    #     # "k_speed": 0.0,     # set >0 only if you add an explicit speed term in 

    #     # Optional: penalize radial velocity near the center to reduce fly-through
    #     "k_vrad":  0.0,     # if you include a vrad2^2 term in r2

    #     # --- wall behavior ---
    #     # Extra "hardening" near boundary; makes wall-bouncing expensive
    #     # Usually you don’t need to exceed wall_penalty unless attacker still suicides.
    #     "k_wall":  0.5,     # extra weight on wall2-type penalty (often = wall_penalty)
    #     "wall_power": 4.0,  # if you implement a (rho2-soft_wall)^wall_power style
    # },

    #Killer attacker
    # "att_reward": {

    #     # --- progress / goal terms ---
    #     # Reward *decreasing* d2 each step (encourages steady approach, less dithering)
    #     "k_prog":  6.0,     # weight on (-delta_d2)

    #     # Small shaping toward being near the center (optional)
    #     # If your r2 already uses "-k_pos * d2", you can keep this 0.
    #     "k_cent":  1.0,

    #     # --- interaction with defender (conservative vs reckless) ---
    #     # Penalty for being too close to defender (prevents "suicide charges")
    #     "min_sep": 1.0,     # meters; MUST exist if your r2 uses self.att_min_sep
    #     "k_close": 0.5,     # how hard to penalize inside min_sep

    #     # --- speed / overshoot control ---
    #     # Stronger velocity penalty makes it stop overshooting center and stop slamming walls
    #     # (this adds to your existing -k_vel*||v2||^2 if you keep that in r2)
    #     # "k_speed": 0.0,     # set >0 only if you add an explicit speed term in r2

    #     # Optional: penalize radial velocity near the center to reduce fly-through
    #     "k_vrad":  0.0,     # if you include a vrad2^2 term in r2

    #     # --- wall behavior ---
    #     # Extra "hardening" near boundary; makes wall-bouncing expensive
    #     # Usually you don’t need to exceed wall_penalty unless attacker still suicides.
    #     "k_wall":  0.5,     # extra weight on wall2-type penalty (often = wall_penalty)
    #     "wall_power": 4.0,  # if you implement a (rho2-soft_wall)^wall_power style
    # },

    #Simple test 
    "att_reward": {
        "k_cent":  0.2,
        "k_prog":  2.0,
        "k_close": 0.0,
        "k_vrad":  0.0,
        "k_wall":  0.0,
        "wall_power": 2.0,
        "min_sep": 3.0
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
    "use_ukf": False,
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
TRAIN: Dict[str, Any] = {

    # Optional checkpoints
    "scale_invariant": True, # Normalizes radii

    "distill": True, #Does policy distillation, True or False

    "def_ckpt_path": None,
    "att_ckpt_path": None,
    "verify_freeze": True,
    "freeze_tol": 0.0,

    # Vectorized rollout & logging
    #Long training
    # "num_envs": 64,   
    # "steps_per_env": 512,    
    # "total_updates": 2000,   
    # "train_epochs": 3,
    # "minibatch_size": 8192,  
    # "log_every": 100,

    #Short training
    "num_envs": 8,          
    "steps_per_env": 256,    
    "total_updates": 300, 
    "train_epochs": 10,
    "minibatch_size": 1024,  
    "log_every": 10,

    #Test training
    "num_envs": 1,          
    "steps_per_env": 256,    
    "total_updates": 100, 
    "train_epochs": 3,
    "minibatch_size": 1024,  
    "log_every": 10,

    # PPO hyperparams
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "policy_lr": 3e-4,
    "value_lr": 1e-3,
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

    "prior_type": "none",                 #ls, nash, none
    "prior_blend_att":        0.0,    # (optional) disable center prior for def
    "prior_blend_def":        0.0,    # (optional) disable center prior for def

    # "prior_blend_def":        0.25,    # (optional) disable center prior for def

    "def_center_safe_radius": 0.3,   # e.g. keep-out inside 10% of R
    "def_center_avoid_coef":  0.0,   # crank this up if defender still dips in
    "def_center_coef":        0.0,    # no attractive center tether




    "att_target_hit_radius": 0.0,          # attacker within % of R hits object
    "att_target_hit_penalty_def": 3.0,     # big negative for defender
    "att_target_hit_reward_att": 5.0,       # matching positive for attacker

    # "def_target_hit_radius": 0.2,          # attacker within 5% of R hits object
    "def_target_hit_penalty_def": 5.0,     # big negative for defender
    "def_target_hit_reward_att": 0.0,       # matching positive for attacker

    "def_keepout_buffer_m": 1.5,        # meters (keepout buffer around object) - Reward function
    "def_oi_safety_buffer": 0.25,       #extra percentage - Termination



    #For collision penalties on both agents

    "collision_radius_m": 0.2,            # meters
    "collision_penalty_def": 3.0,         # penalty applied to defender on collision
    "collision_penalty_att": 3.0,         # penalty applied to attacker(s) on collision


    #Learning stat tracking
    "use_tensorboard": True,
    "tb_logdir": "runs",
    "tb_run_name": "ppo_diffgame_def",  # customize per experiment


    "episodes_per_iter": 8,
    "max_steps": 300,
    "lookahead_H": 15,
    "iters": 100,
    "tbptt_chunk_len": 40,
    "dagger_beta_start": 1.0,
    "dagger_beta_end": 0.0,
    "dagger_decay_iters": 50,
    "lambda_intent": 1.0,
    "reward_mode_for_step": "def",




}

#Long training config below:

# ---------- TRAIN (long-run, more stable) ----------
# TRAIN: Dict[str, Any] = {
#     # Vectorized rollout & logging
#     # Total samples per update = num_envs * steps_per_env
#     # Here: 32 * 512 = 16,384 steps/update (8x your original short run)
#     "num_envs": 32,
#     "steps_per_env": 512,
#     "total_updates": 1200,     # long but not crazy; ~19.7M steps

#     "log_every": 10,

#     # PPO hyperparams
#     "gamma": 0.99,
#     "gae_lambda": 0.95,
#     "clip_eps": 0.2,

#     # ↓ Smaller LRs than your short-run config (since batch is much larger)
#     "policy_lr": 1e-4,         # was 3e-4
#     "value_lr": 3e-4,          # was 1e-3

#     "train_epochs": 10,
#     "minibatch_size": 1024,    # 16 minibatches per update w/ 16,384 steps
#     "entropy_coef": 0.02,
#     "value_coef": 0.5,
#     "max_grad_norm": 1.0,

#     "lr_schedule": "linear",   # options: "none", "linear"
#     "lr_final_factor": 0.1,    # final_lr = lr_final_factor * initial_lr

#     # Anneal (training-time only)
#     "def_center_min_anneal": 0.5,

#     # Training-only initial condition randomization
#     "train_ic_mode": "random_shell",  # or "fixed"
#     "train_ic_vmax": 0.05,            # max |v| component at t=0
#     "train_min_sep": 1.0,             # min defender–attacker separation (m)

#     # Defender vs center behaviour
#     "def_center_safe_radius": 0.10,   # keep-out inside 10% of R
#     "def_center_avoid_coef":  50.0,   # penalty for entering this zone
#     "def_center_coef":        0.0,    # no attractive tether to center
#     "prior_blend_def":        0.0,    # disable center prior for def (RL only)

#     # Terminal hit logic
#     "att_target_hit_radius":       0.05,   # attacker within 5% of R hits target
#     "att_target_hit_penalty_def": 10.0,    # was 5.0 → stronger slap to defender
#     "att_target_hit_reward_att":   5.0,    # attacker reward unchanged
# }



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



    cfg["dispersion"] = {
        "enabled": True,

        # -------- Initial conditions (episode-level) --------
        "ic": {
            "mode": "csv",      # "fixed" | "sample_ball" | "csv"
            "pos_scale": 0.95,    # only used for sample_ball (fraction of arena_r)
            "vel_sigma": 0.0,     # m/s (or whatever units your state uses)
            "min_sep": 0.0,       # meters
        },

        # -------- Small perturbations around chosen IC --------
        "x0_jitter": {
            "pos_sigma": 0.0,     # meters
            "vel_sigma": 0.0,     # m/s
        },

        # -------- Observation / measurement noise (inside runner) --------
        "meas_noise": {
            "enabled": True,
            "pos_sigma": 0.0,
            "vel_sigma": 0.0,
        },

        # -------- Process noise / acceleration disturbance --------
        "proc_noise": {
            "enabled": False,
            "acc_sigma": 0.0,     # m/s^2
        },

        # -------- Optional: parameter randomization --------
        "params": {
            "enabled": False,
            # examples:
            "umax": {"dist": "lognormal", "sigma": 0.10},  # 10% multiplicative
            "dt":   {"dist": "normal",    "sigma": 0.00},
            # "mass": {"dist": "normal", "sigma": 0.05},
        },

        # -------- Random seeds --------
        "seed": {
            "episode_seed_base": 0,     # used for IC + jitter + param draws
            "noise_seed_base": 0,       # used for meas/proc noise (if you separate)
            "use_global_np_seed": True, # because your runner uses np.random.randn
        },
    }

    return cfg

# ---------- Dynamics builder ----------
def build_dyn(cfg: Dict[str, Any]):
    import numpy as np
    from dyn_models import (
        hcw_mean_motion, hcw_discrete_mats, as_numpy_const,
        chief_orbit_cache_rtn, linearize_two_body_rtn_discrete
    )

    dyn_name = cfg["dynamics"].lower()
    dt = float(cfg["dt"])
    N = int(cfg["T"])  # assuming T is number of steps (as your code implies)

    cfg.setdefault("dyn", {})
    cfg["dyn"].setdefault("Ad", None)
    cfg["dyn"].setdefault("Bd", None)
    cfg["dyn"].setdefault("Ad_seq", None)
    cfg["dyn"].setdefault("Bd_seq", None)
    cfg["dyn"].setdefault("chief_cache", None)
    cfg["dyn"].setdefault("model", None)
    cfg["dyn"].setdefault("type", None)

    # ---------------------------
    # 1) HCW (LTI)
    # ---------------------------
    if dyn_name == "hcw":
        from dyn_models import hcw_mean_motion, hcw_discrete_mats, as_numpy_const
        n = hcw_mean_motion(cfg["hcw"])
        Ad, Bd = hcw_discrete_mats(float(n), dt)
        cfg["dyn"]["Ad"] = as_numpy_const(Ad).astype(np.float32)
        cfg["dyn"]["Bd"] = as_numpy_const(Bd).astype(np.float32)
        cfg["dyn"]["type"] = "lti"
        cfg["dyn"]["model"] = "hcw"

    # ---------------------------
    # 2) Two-body nonlinear in RTN/LVLH
    # ---------------------------
    elif dyn_name == "two_body":
        orb = cfg.get("chief_orbit", {})
        cache = chief_orbit_cache_rtn(orb, dt=dt, N=N)
        cfg["dyn"]["chief_cache"] = cache
        cfg["dyn"]["type"] = "nonlinear"
        cfg["dyn"]["model"] = "two_body_rtn"
        # leave Ad/Bd None on purpose

    # ---------------------------
    # 3) Elliptical “non-LTI” (LTV) via per-step linearization of two-body RTN
    # ---------------------------
    elif dyn_name in ("elliptic_ltv", "elliptical_ltv", "th", "tschauner_hempel"):
        orb = cfg.get("chief_orbit", {})
        cache = chief_orbit_cache_rtn(orb, dt=dt, N=N)
        Ad_seq, Bd_seq = linearize_two_body_rtn_discrete(cache, dt=dt, eps=1e-5)
        cfg["dyn"]["chief_cache"] = cache
        cfg["dyn"]["Ad_seq"] = Ad_seq.astype(np.float32)  # (N,6,6)
        cfg["dyn"]["Bd_seq"] = Bd_seq.astype(np.float32)  # (N,6,3)
        cfg["dyn"]["Ad"] = cfg["dyn"]["Ad_seq"][0]
        cfg["dyn"]["Bd"] = cfg["dyn"]["Bd_seq"][0]
        cfg["dyn"]["type"] = "ltv"
        cfg["dyn"]["model"] = "two_body_rtn_ltv"

    else:
        raise ValueError(f"Unknown dynamics='{cfg['dynamics']}'")


