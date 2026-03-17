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
    "num_attackers": 1,   # default is 1
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
        "e":  0.1,                      # eccentricity
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

    # Action bounds
    "umax": 2.0,

    # Initial conditions
    "x0": np.array([
        [  2.5, 0.0, 0.0,  -0.02, 0.00, 0.000],  # defender
        [ -19.9, 0.0, 0.0,  +0.02, 0.00, 0.000], # attacker
    ], dtype=float),
    "x0_jitter": {"pos": 0.5, "vel": 0.01},

    # Attacker control selection
    #   "rule" => deterministic controller
    #   "rl"   => learned attacker (if your training loop supports it)
    "attacker_mode": "rl",

    # Rule-based attacker parameters

    "att_rule": {
        "ridge": 2e-2,       # slightly smoother center controller
        "w_center": 1.0,     # still wants the center
        "w_avoid":  2.5,     # stronger fear
        "w_damp":   0.3,     # more damping to avoid jitter
        "min_sep":  4.0,     # reacts earlier to defender
        "repulse_gain": 1.0, # tuned so repulsion doesn't instantly saturate
    },

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

    "mcp": {
        "solver": "path",
        "executable": "/home/gs34433/Research/path_5/ampl/pathampl",       
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

    "fuel": {
        "enable": False,
        "def": {
            "m0": 500.0,
            "m_dry": 400.0,
            "Tmax": 2.0,
            "Isp": 220.0,
        },
        "att": {
            "m0": 500.0,
            "m_dry": 400.0,
            "Tmax": 2.0,
            "Isp": 220.0,
        },
    },
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

    "save_best_ckpt": True,
    "save_last_ckpt": True,

    # Optional checkpoints
    "scale_invariant": True, # Normalizes radii

    "distill": True, #Does policy distillation, True or False

    "def_ckpt_path": None,
    "att_ckpt_path": None,
    "verify_freeze": True,
    "freeze_tol": 0.0,

    # Vectorized rollout & logging

    #Current config
    "num_envs": 256,   #Was 256
    "steps_per_env": 512, #Was 512  
    "total_updates": 1000,   #Was 2000
    "train_epochs": 3, #Was 3
    "minibatch_size": 4096,  #Was 8192
    "log_every": 10,
    "entropy_coef": 0.01,

    "k_pos": 0.3, #Was 0.05
    "k_dock": 0.5,

    "gamma": 0.992, #Was 0.995
    "target_hit_reward_penalty": 10.0,
    "collision_penalty": 10.0,
    "wall_penalty": 2.5,
    "fuel_depletion_penalty": 5.0,
    "collision_radius_m": 0.25,            # meters

    # "k_pos": 0.05, #Was 0.05
    # "k_dock": 0.5,


    # #Long training (new)
    # "num_envs": 128,   
    # "steps_per_env": 512,   
    # "total_updates": 1000,   
    # "train_epochs": 3,
    # "minibatch_size": 4096,  
    # "log_every": 10,
    # "entropy_coef": 0.01,
    # "k_pos": 0.04, #Was 0.05
    # "gamma": 0.998, #Was 0.999

    # #Amy Zhang email config
    # "num_envs": 128,   #Was 64 
    # "steps_per_env": 256, #Was 512  
    # "total_updates": 1000,   #Was 2000
    # "train_epochs": 5, #Was 3
    # "minibatch_size": 4096,  #Was 8192
    # "entropy_coef": 0.01,
    # "k_pos": 0.04,
    # "gamma": 0.998, #Was 0.999
    # "target_hit_reward_penalty": 5.0,
    # "collision_penalty": 5.0,
    # "wall_penalty": 5.0,



    #Long training (old)
    # "num_envs": 64,   
    # "steps_per_env": 512,   
    # "total_updates": 2000,   
    # "train_epochs": 3,
    # "minibatch_size": 8192,  
    # "log_every": 10,
    # "entropy_coef": 0.02,

    #Short training
    # "num_envs": 8,          
    # "steps_per_env": 256,    
    # "total_updates": 300, 
    # "train_epochs": 10,
    # "minibatch_size": 1024,  
    # "log_every": 10,
    # "entropy_coef": 0.01,
    # "k_pos": 0.04,
    # "gamma": 0.999,

    #Test training
    # "num_envs": 1,          
    # "steps_per_env": 256,    
    # "total_updates": 100, 
    # "train_epochs": 3,
    # "minibatch_size": 1024,  
    # "log_every": 10,
    # "entropy_coef": 0.01,
    # "k_pos": 0.04,  
    # "k_dock": 0.5,
    # "gamma": 0.998,
    # "target_hit_reward_penalty": 5.0,
    # "collision_penalty": 5.0,
    # "wall_penalty": 5.0,

    "record_ic_history": True,
    "max_ic_history": 300000,

    #Effort penalties (Fuel setting)
    "k_eff_def": 0.01,
    "k_eff_att": 0.01,

    #Arena start positions config (relative to radius of arena)
    # "r_def_min": 0.0, #Default: 0.0
    # "r_def_max": 0.85, # Default: 0.5

    # "r_att_min": 0.0, #Default 0.4
    # "r_att_max": 0.95, #Default: 0.95

    "r_def_min": 0.0, #Default: 0.0
    "r_def_max": 0.8, # Default: 0.5

    "r_att_min": 0.3, #Default 0.4
    "r_att_max": 0.95, #Default: 0.95

    "percent_advantage_defender": 1.0, 

    # Other PPO hyperparams
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "policy_lr": 3e-4,
    "value_lr": 1e-3,
    "value_coef": 0.5,
    "max_grad_norm": 1.0,

    # Termination rewards/penalties 

    # "target_hit_reward_penalty": 4.0,
    # "collision_penalty": 2.0,
    # "wall_penalty": 2.0,

    "hit_buffer_att": 0.0,
    "hit_buffer_def": 0.0,

    # Legacy reward shaping (used by Env at train *and* eval)
    "effort_def": 0.01,
    "effort_att": 0.01,

    #Learning schedule
    "lr_schedule": "linear",   # options: "none", "linear"
    "lr_final_factor": 0.1,  # final_lr = lr_final_factor * initial_lr

    # Anneal (training-time only)
    "def_center_min_anneal": 0.5,

    # Training-only initial condition randomization
    # (env uses this; eval config will default back to "fixed")
    "train_ic_mode": "random_shell_advantage",  # or "fixed"
    "train_ic_vmax": 0.5,            # max |v| component at t=0
    "train_min_sep": 2.0,             # min defender–attacker separation (m)

    "prior_type": "ls",                 #ls, nash, none
    "prior_blend_att":        0.0,    # (optional) disable center prior for def
    "prior_blend_def":        0.0,    # (optional) disable center prior for def

    "att_target_hit_radius": 0.0,          # attacker within % of R hits object

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
    cfg = _merge(COMMON,TRAIN)
    
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



# cfg["paper_baseline"] = {
#   "objective": "ppo_oi_minmax",
#   "ppo_obj": {
#     "threat_mode": "idx0",
#     "include_step_walls": False,
#     "include_keepout": False,
#     "include_effort_def": False,
#   }
# }