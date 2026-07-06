# config_rl.py
from __future__ import annotations
from typing import Dict, Any
import hashlib
import math
import numpy as np
from pathlib import Path
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


def _apply_role_specific_curriculum_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optionally deep-merge role-specific arena curriculum overrides.

    `arena_radius_knob` remains the shared/default curriculum. When training the
    attacker, `arena_radius_knob_att` can override only the fields that should
    differ while inheriting everything else from the base setup. The defender
    can likewise opt into `arena_radius_knob_def`, but leaving it unset keeps
    the historical behavior unchanged.
    """
    role = str(cfg.get("train_role", "")).strip().lower()
    role_key_map = {
        "def": ("arena_radius_knob_def", "arena_radius_knob_defender"),
        "att": ("arena_radius_knob_att", "arena_radius_knob_attacker"),
    }
    override_keys = role_key_map.get(role)
    if override_keys is None:
        return cfg

    override_cfg = None
    for key in override_keys:
        candidate = cfg.get(key, None)
        if candidate is None:
            continue
        if not isinstance(candidate, dict):
            raise TypeError(f"{key} must be a dict when provided, got {type(candidate).__name__}.")
        override_cfg = _merge(override_cfg or {}, candidate)

    if override_cfg:
        base_knob = cfg.get("arena_radius_knob", {})
        if base_knob is None:
            base_knob = {}
        if not isinstance(base_knob, dict):
            raise TypeError(
                "arena_radius_knob must be a dict when role-specific curriculum overrides are used."
            )
        cfg["arena_radius_knob"] = _merge(base_knob, override_cfg)

    return cfg

# ---------- COMMON (shared by train & eval/rollout) ----------
COMMON: Dict[str, Any] = {
    "umax": 0.1,

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
        "a":  6_371_000.0 + 1_275_000.0,  # semi-major axis (m); 550 x 2000 km altitudes
        "e":  1_450_000.0 / 15_292_000.0, # eccentricity for 550 km perigee, 2000 km apogee
        "i":  0.0,                      # rad
        "raan": 0.0,                    # rad
        "argp": 0.0,                    # rad
        "nu0":  0.0,                    # rad (true anomaly at t=0)
    },
    "elliptic_ltv": {
        "randomize_nu0": True,           # if True, draw nu0 ~ Uniform(0, 2*pi) per eval trial
        "use_full_orbit_cache": True,    # precompute one orbit and slice per randomized nu0
        "disk_cache_enabled": True,      # share the full-orbit cache across eval processes
        "disk_cache_dir": ".dyn_cache",
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
    # "arena": {"type": "sphere", "cx": 0.0, "cy": 0.0, "cz": 0.0, "r": 20.0},
    "arena": {"type": "sphere", "cx": 0.0, "cy": 0.0, "cz": 0.0, "r": 100.0},

    "arena_terminate_margin": 1.0,
    "soft_wall_start": 0.5,

    # Action / controller bounds
    # "umax": 2.0,
    # "umax": 1.0,
    # "umax": 0.1,
    # "umax": 0.01,
    #
    # Keep the controller speed cap under safety_filter["vmax"] so it does not
    # get confused with the training actuation limit cfg["umax"].

    "safety_filter": {
        "enabled": True,
        "kind": "velocity_cbf_qp",
        "alpha": 5.0,
        # "vmax": 1.5,  # controller speed cap; set to None to disable the cap
        "vmax": 1.0
    },

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

    # Zero-sum step shaping used by the PPO env.
    # "progress_barrier" rewards attacker progress to the objective while only
    # penalizing proximity inside a local safety bubble around the defender.
    # "legacy_dock" reproduces the old global stay-away dock-gap shaping.
    "zero_sum_reward": {
        "mode": "legacy_dock",
        "progress_coef": 2.0,
        "safe_sep_m": 3.0,
        "unsafe_close_coef": 2.0,
        "unsafe_close_power": 2.0,
        "progress_gate_power": 1.0,
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
SENSOR_NOISE_DEFAULT_DEG = 0.5

KF_COMMON: Dict[str, Any] = {
    # "use_kf": True,
    "use_kf": False,

    "estimator_kind": "ekf",
    "meas_innov_coef": 1.0,  
    "meas_cov_coef": 0.02,

    "sensor_noise_default": SENSOR_NOISE_DEFAULT_DEG,
    "ukf": {
        "sigma_az": np.deg2rad(SENSOR_NOISE_DEFAULT_DEG),
        "sigma_el": np.deg2rad(SENSOR_NOISE_DEFAULT_DEG),
        "every": 1,
        "ekf_jacobian_mode": "exact",  # "exact" | "frozen"
        "ekf_use_torch": True,  # when True, EKF runs on cfg["device"] (e.g. "cuda")
        # "pos_std0": 0.2 * ARENA_R,
        "pos_std0": 0.2 * 20.0,
        "vel_std0": 0.01,
        "action_access": "measured",  # ground_truth | measured | none
        "action_meas_std": 0.25 * float(COMMON["umax"]),  # used when action_access == "measured"
        # "init_pos_std": 0.2 * ARENA_R,
        "init_pos_std": 0.2 * 20.0,
        "init_vel_std": 0.01,
        "init_mean_pos_std": 0.0,
        "init_mean_vel_std": 0.0,
        "Q_scale": 1e-5,
        "accel_std0": 2.0,
        "init_mean_accel_std": 0.0,
        "accel_Q_scale": 1e-4,
        "center_sigma_az": np.deg2rad(SENSOR_NOISE_DEFAULT_DEG),
        "center_sigma_el": np.deg2rad(SENSOR_NOISE_DEFAULT_DEG),
        # "center_pos_std0": 0.2 * ARENA_R,
        "center_pos_std0": 0.2 * 20.0,
        "center_vel_std0": 0.01,
        # "center_init_pos_std": 0.2 * ARENA_R,
        "center_init_pos_std": 0.2 * 20.0,
        "center_init_vel_std": 0.01,
        "center_init_mean_pos_std": 0.0,
        "center_init_mean_vel_std": 0.0,
        "center_Q_scale": 1e-5,
    },
    "reward_from_belief": True,   # compatibility field; env now keys belief rewards off the estimator toggle

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
    "reward_type": "zero_sum",
    # "reward_type": "zero_sum_kf",

    "save_best_ckpt": True,
    "save_last_ckpt": True,

    "distill": False, #Does policy distillation, True or False
    "distill_method": "modern",  # "modern" or "paper_recurrent"
    "distill_collection_mode": "dagger",  # modern/paper_recurrent only support "dagger"
    "distill_paper_student_lstm_hidden": 256,
    "distill_paper_tbptt_chunk_len": 40,
    "distill_paper_train_episodes_per_iter": 64,  # None => train on the full aggregated dataset
    "distill_paper_lambda_intent": 1.0,
    "distill_paper_action_loss": "mse",
    "distill_paper_intent_loss": "mse",
    "distill_paper_grad_clip_norm": None,
    "distill_paper_log_every": 10,
    "distill_paper_allow_sim_intent_fallback": True,
    "distill_log_every": 10,

    "def_ckpt_path": None,
    "att_ckpt_path": None,
    "verify_freeze": True,
    "freeze_tol": 0.0,

    # Vectorized rollout & logging

    #Base config
    "num_envs": 256,   #Was 256
    "train_epochs": 3, #Was 3
    "minibatch_size": 4096,  #Was 8192
    "log_every": 10,
    "entropy_coef": 0.01,
    "vec_backend": "torch",  # "sync" | "subproc" | "torch"
    "vec_workers": None,     # only used when vec_backend == "subproc"
    "mp_start_method": None, # auto-selects a multiprocessing start method
    # Keep the torch backend on the legacy per-env reset path by default.
    # Enable this only when intentionally testing the newer fast-reset sampler.
    "torch_fast_reset": False,

    #Higher accelerations setting (2.0 m/s^2)
    # "umax": 2.0,
    # "k_pos": 0.1,
    # "steps_per_env": 512, 

    # Scale for 0.5 umax - 
    "umax": 0.5,
    "k_pos": 0.1/2,
    "steps_per_env": 512*2, 
    # "steps_per_env": 512*4, 
    "k_dock": 0.1/8,

    # "k_dock": 0.00,
    # "k_pos": 0.1/2,
    # "steps_per_env": 512*3, 

    # Scale for 0.1 umax - 
    # "umax": 0.1,
    # "k_pos": 0.1/4.5, #Started at -8
    # "steps_per_env": 512*4.5, 
    # "k_pos": 0.1/4, #Started at -8
    # "steps_per_env": 512*4, 


    "gamma": 0.991, 
    # "train_ic_vmax": 0.5/2,            # max |v| component at t=0
    "train_ic_vmax": 1.0,

    # "train_ic_vmax": 1.0,
    # "train_ic_vmax": 0.5,            # max |v| component at t=0
    "total_updates": 1000,   #Was 2000



    #Lower acceleration setting (0.1 m/s^2)
    # "umax": 0.1,
    # "k_pos": 0.1/4.5,
    # "gamma": 0.991, 
    # "steps_per_env": 512*4.5,  
    # # "steps_per_env": 2048,  
    # "train_ic_vmax": 0.5/4.5,            # max |v| component at t=0
    # "total_updates": 1000,   #Was 2000
    # # "total_updates": 2000,   #Was 2000
    # # "k_dock": 0.05/4.5,  # optional mild defender-approach shaping on normalized gap outside collision radius
    # "k_dock": 0.0, 

    # "k_pos": 0.05, #Was 0.05
    # "k_pos": 0.01, #Was 0.05


    "normalize_reward": True,
    "normalize_reward_geometry_power": 2,
    # "reward_normalize_radius_m": None,  # optional constant radius for reward normalization; None => use active arena radius
    "reward_normalize_radius_m": 20.0,  # optional constant radius for reward normalization; None => use active arena radius
   
   
    "target_hit_reward_penalty": 10.0,
    "collision_penalty": 10.0,
    "wall_penalty": 2.5,
    "fuel_depletion_penalty": 5.0,
    "collision_radius_m": 0.75,            # meters

    #Termination penalties
    # "target_hit_reward_penalty": 10.0*5,
    # "collision_penalty": 10.0*5,
    # "wall_penalty": 2.5*5,
    # "fuel_depletion_penalty": 5.0,
    # "collision_radius_m": 1.0,            # meters

    #Test training
    # "num_envs": 1,          
    # "steps_per_env": 256,    
    # "total_updates": 20, 
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
    # "fuel_depletion_penalty": 5.0,

    "record_ic_history": True,
    "max_ic_history": 300000,

    #Effort penalties (Fuel setting)
    "k_eff_def": 0.01,
    "k_eff_att": 0.01,

    # Arena start positions config.
    # Fractional keys below are interpreted relative to arena["r"].
    # Optional *_m overrides let you pin explicit shell bounds in meters.
    # "r_def_min": 0.0, #Default: 0.0
    # "r_def_max": 0.85, # Default: 0.5
    # "r_def_min_m": 0.0,
    # "r_def_max_m": 80.0,

    # "r_att_min": 0.0, #Default 0.4
    # "r_att_max": 0.95, #Default: 0.95
    "r_att_min_m": 20.0 * 0.3,
    # "r_att_max_m": 99.0,

    "r_def_min": 0.0, #Default: 0.0
    "r_def_max": 0.95, # Default: 0.5

    "r_att_min": 0.3, #Default 0.4
    "r_att_max": 0.95, #Default: 0.95

    "percent_advantage_defender": 1.0, 
    # "percent_advantage_defender": 3.0, 


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
    "train_min_sep": 2.0,             # min defender–attacker separation (m)
    "arena_radius_knob": {
        "enabled": True,
        "start_radius_m": 20.0,
        "final_radius_m": 100.0,
        "schedule": "staged",       # "linear" | "fixed" | "staged"
        "schedule_fraction": 1.0,   # linear only: fraction of training spent reaching final_radius_m
        "sample_mode": "fixed",     # "uniform" | "fixed"; fixed uses the exact active stage radius
        "integer_only": True,       # rounds the sampled radius when sample_mode returns a single value
        "stages": [
            # Use exactly one scheme for the whole list:
            # {"radius_m": 20.0, "fraction": 0.10},
            # {"radius_m": 20.0, "updates": 400},
            #
            # Stage entries may also override shell bounds for that stage using
            # the same keys as the top-level config, for example:
            # {"radius_m": 20.0, "updates": 400, "r_att_min": 0.30, "r_att_max": 0.95},
            # {"radius_m": 40.0, "updates": 400, "r_att_min_m": 12.0, "r_att_max_m": 36.0},
            # {"radius_m": 60.0, "updates": 400, "r_def_max": 0.80},


            #Third attempt setting:
            # {"radius_m": 20.0, "updates": 400, "r_att_min_m": 0.20*30, "r_att_max_m": 0.95*20},
            # {"radius_m": 100.0, "updates": 600, "r_att_min_m": 0.95*20, "r_att_max_m": 0.95*100},


            #Fourth attempt setting:
            # {"radius_m": 20.0, "updates": 1000, "r_att_min_m": 0.30*20, "r_att_max_m": 0.95*20},

            # OG with preliminar actuation limits:
            {"radius_m": 20.0, "updates": 1000, "r_att_min_m": 0.30*20, "r_att_max_m": 0.95*20, "r_def_min_m": 0.0, "r_def_max_m": 0.80*20},

        ],
    },
    "arena_radius_knob_att": {
        # Optional deep-merge override applied only when train_role=="att".
        # Keep `arena_radius_knob` as the default/defender curriculum and set
        # only the attacker-specific differences here.
        #
        # Example:
        "enabled": True,
        "stages": [
            #Setting used in staged learening
            # {"radius_m": 20.0, "updates": 1000, "r_att_min_m": 0.0,  "r_att_max_m": 19.0},

            # OG with preliminar actuation limits:
            {"radius_m": 20.0, "updates": 1000, "r_att_min_m": 0.0,  "r_att_max_m": 0.95*20, "r_def_min_m": 0.0, "r_def_max_m": 0.80*20},
        ],
    },

    "prior_type": "none",          # ls, intercept, or none
    "prior_blend_att": 0.0,
    "prior_blend_def": 0.0,
    "intercept_prior": {
        "lookahead_steps": 10.0,  # attacker coasting lookahead in env steps
        "mix": 1.0,               # 0 -> current attacker position, 1 -> full intercept target
        "gain": 2.0,              # raw-action gain for the geometric intercept direction
    },
    "intercept_prior_train_only": {
        "enabled": False,          # when True, override defender prior blend during PPO updates only
        "start_blend": 0.5,
        "end_blend": 0.0,
        "anneal_fraction": 0.8,   # reach end_blend after this fraction of training progress
    },

    "att_target_hit_radius": 0.0,          # attacker within % of R hits object

    #Learning stat tracking
    "use_tensorboard": True,
    "tb_logdir": "runs",
    "tb_run_name": "ppo_diffgame_def",  # customize per experiment
    "tb_run_prefix": None,


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
    cfg = _apply_role_specific_curriculum_overrides(cfg)
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

    cfg = _apply_role_specific_curriculum_overrides(cfg)
    return cfg


def _slice_chief_cache(cache: Dict[str, Any], start_idx: int, steps: int) -> Dict[str, Any]:
    """Return a contiguous chief-cache slice for a rollout starting at start_idx."""
    out = {
        "mu": cache["mu"],
        "dt": cache["dt"],
        "N": int(steps),
    }
    lo = int(start_idx)
    hi = lo + int(steps) + 1
    for key in ("rC", "vC", "C_RTN2I", "C_I2RTN", "w_rtn"):
        out[key] = cache[key][lo:hi].copy()
    return out


def _mean_anomaly_from_true_anomaly(nu: float, e: float) -> float:
    """Convert true anomaly to mean anomaly in [0, 2*pi)."""
    E = 2.0 * math.atan2(
        math.sqrt(1.0 - e) * math.sin(float(nu) / 2.0),
        math.sqrt(1.0 + e) * math.cos(float(nu) / 2.0),
    )
    M = E - float(e) * math.sin(E)
    return float(M % (2.0 * math.pi))


def _elliptic_ltv_full_cache_path(
    ell_cfg: Dict[str, Any],
    orb: Dict[str, Any],
    dt: float,
    rollout_steps: int,
    full_steps: int,
    period_steps: int,
    eps: float,
) -> Path:
    cache_dir = Path(str(ell_cfg.get("disk_cache_dir", ".dyn_cache")))
    key_parts = [
        f"mu={float(orb['mu']):.17g}",
        f"a={float(orb['a']):.17g}",
        f"e={float(orb['e']):.17g}",
        f"i={float(orb.get('i', 0.0)):.17g}",
        f"raan={float(orb.get('raan', 0.0)):.17g}",
        f"argp={float(orb.get('argp', 0.0)):.17g}",
        f"dt={float(dt):.17g}",
        f"rollout_steps={int(rollout_steps)}",
        f"full_steps={int(full_steps)}",
        f"period_steps={int(period_steps)}",
        f"eps={float(eps):.17g}",
    ]
    digest = hashlib.sha256("|".join(key_parts).encode("utf-8")).hexdigest()[:20]
    return cache_dir / f"elliptic_ltv_full_orbit_{digest}.npz"


def _load_elliptic_ltv_full_cache(path: Path) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=False)
    chief_cache = {
        "mu": float(data["mu"]),
        "dt": float(data["dt"]),
        "N": int(data["full_steps"]),
        "rC": data["rC"],
        "vC": data["vC"],
        "C_RTN2I": data["C_RTN2I"],
        "C_I2RTN": data["C_I2RTN"],
        "w_rtn": data["w_rtn"],
    }
    return {
        "chief_cache": chief_cache,
        "Ad_seq": data["Ad_seq"],
        "Bd_seq": data["Bd_seq"],
        "period_steps": int(data["period_steps"]),
        "period_sec": float(data["period_sec"]),
    }


def _save_elliptic_ltv_full_cache(
    path: Path,
    cache: Dict[str, Any],
    Ad_seq: np.ndarray,
    Bd_seq: np.ndarray,
    period_steps: int,
    period_sec: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez(
            f,
            mu=float(cache["mu"]),
            dt=float(cache["dt"]),
            full_steps=int(cache["N"]),
            period_steps=int(period_steps),
            period_sec=float(period_sec),
            rC=cache["rC"],
            vC=cache["vC"],
            C_RTN2I=cache["C_RTN2I"],
            C_I2RTN=cache["C_I2RTN"],
            w_rtn=cache["w_rtn"],
            Ad_seq=Ad_seq,
            Bd_seq=Bd_seq,
        )
    tmp_path.replace(path)


def set_elliptic_ltv_phase(cfg: Dict[str, Any], nu0: float, steps: int | None = None) -> int:
    """Slice a precomputed full-orbit Elliptic LTV cache for a trial nu0.

    The input nu0 is a true anomaly. The cache is uniformly sampled in time, so
    this maps true anomaly to mean anomaly and then to the nearest cache index.
    """
    dyn = cfg.setdefault("dyn", {})
    Ad_full = dyn.get("full_orbit_Ad_seq", None)
    Bd_full = dyn.get("full_orbit_Bd_seq", None)
    chief_full = dyn.get("full_orbit_chief_cache", None)
    period_steps = int(dyn.get("full_orbit_period_steps", 0) or 0)
    if Ad_full is None or Bd_full is None or chief_full is None or period_steps <= 0:
        raise RuntimeError("Elliptic LTV full-orbit cache is not available for phase slicing.")

    steps = int(steps if steps is not None else cfg.get("T", 0))
    if steps <= 0:
        raise RuntimeError(f"Invalid Elliptic LTV phase-slice steps={steps}.")

    e = float((cfg.get("chief_orbit", {}) or COMMON["chief_orbit"]).get("e", COMMON["chief_orbit"]["e"]))
    M0 = _mean_anomaly_from_true_anomaly(float(nu0), e)
    start_idx = int(round((M0 / (2.0 * math.pi)) * float(period_steps))) % period_steps
    hi = start_idx + steps
    if hi > int(Ad_full.shape[0]):
        raise RuntimeError(
            f"Full-orbit Elliptic LTV cache too short for start_idx={start_idx}, steps={steps}, "
            f"Ad_len={Ad_full.shape[0]}."
        )

    cfg.setdefault("chief_orbit", {})
    cfg["chief_orbit"]["nu0"] = float(nu0)
    dyn["Ad_seq"] = Ad_full[start_idx:hi].copy()
    dyn["Bd_seq"] = Bd_full[start_idx:hi].copy()
    dyn["chief_cache"] = _slice_chief_cache(chief_full, start_idx, steps)
    dyn["Ad"] = dyn["Ad_seq"][0]
    dyn["Bd"] = dyn["Bd_seq"][0]
    dyn["type"] = "ltv"
    dyn["model"] = "two_body_rtn_ltv"
    dyn["phase_start_idx"] = int(start_idx)
    dyn["phase_nu0"] = float(nu0)
    return int(start_idx)

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

    def _seq_len_ok(name: str) -> bool:
        arr = cfg["dyn"].get(name, None)
        if arr is None:
            return False
        try:
            return int(arr.shape[0]) >= N
        except Exception:
            return False

    # Reuse precomputed dynamics when a caller passes a cfg that was already
    # built. This matters for scalar Elliptic LTV Monte Carlos, where rebuilding
    # Ad_seq/Bd_seq inside every policy wrapper dominates runtime.
    if dyn_name == "hcw":
        if (
            cfg["dyn"].get("type") == "lti"
            and cfg["dyn"].get("model") == "hcw"
            and cfg["dyn"].get("Ad", None) is not None
            and cfg["dyn"].get("Bd", None) is not None
        ):
            return
    elif dyn_name in ("elliptic_ltv", "elliptical_ltv", "th", "tschauner_hempel"):
        if (
            cfg["dyn"].get("type") == "ltv"
            and cfg["dyn"].get("model") == "two_body_rtn_ltv"
            and cfg["dyn"].get("chief_cache", None) is not None
            and _seq_len_ok("Ad_seq")
            and _seq_len_ok("Bd_seq")
        ):
            cfg["dyn"]["Ad"] = cfg["dyn"]["Ad_seq"][0]
            cfg["dyn"]["Bd"] = cfg["dyn"]["Bd_seq"][0]
            return
    elif dyn_name == "two_body":
        if (
            cfg["dyn"].get("type") == "nonlinear"
            and cfg["dyn"].get("model") == "two_body_rtn"
            and cfg["dyn"].get("chief_cache", None) is not None
        ):
            return

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
        # Evaluation run manifests can carry older chief-orbit settings from
        # training. Keep the nominal Elliptic LTV comparison tied to the current
        # main config so reruns use the corrected physical orbit. When orbital
        # phase randomization is enabled, preserve the trial-level nu0 draw.
        ell_cfg = cfg.setdefault("elliptic_ltv", _dcopy(COMMON.get("elliptic_ltv", {})))
        orb = _dcopy(COMMON["chief_orbit"])
        randomize_nu0 = bool(ell_cfg.get("randomize_nu0", False))
        use_full_orbit_cache = bool(ell_cfg.get("use_full_orbit_cache", True))
        if bool(ell_cfg.get("randomize_nu0", False)):
            orb["nu0"] = float((cfg.get("chief_orbit", {}) or {}).get("nu0", orb.get("nu0", 0.0)))
        cfg["chief_orbit"] = orb
        if randomize_nu0 and use_full_orbit_cache:
            mu = float(orb["mu"])
            a = float(orb["a"])
            eps = 1e-5
            period_sec = 2.0 * math.pi * math.sqrt(a**3 / mu)
            period_steps = int(math.ceil(period_sec / dt))
            full_steps = int(period_steps + N + 1)
            cache_data = None
            cache_path = None
            if bool(ell_cfg.get("disk_cache_enabled", True)):
                cache_path = _elliptic_ltv_full_cache_path(
                    ell_cfg,
                    orb,
                    dt=dt,
                    rollout_steps=N,
                    full_steps=full_steps,
                    period_steps=period_steps,
                    eps=eps,
                )
                cache_data = _load_elliptic_ltv_full_cache(cache_path)
            if cache_data is None:
                cache_orb = _dcopy(orb)
                cache_orb["nu0"] = 0.0
                cache = chief_orbit_cache_rtn(cache_orb, dt=dt, N=full_steps)
                Ad_seq, Bd_seq = linearize_two_body_rtn_discrete(cache, dt=dt, eps=eps)
                Ad_seq = Ad_seq.astype(np.float32)
                Bd_seq = Bd_seq.astype(np.float32)
                if cache_path is not None:
                    _save_elliptic_ltv_full_cache(cache_path, cache, Ad_seq, Bd_seq, period_steps, period_sec)
            else:
                cache = cache_data["chief_cache"]
                Ad_seq = cache_data["Ad_seq"]
                Bd_seq = cache_data["Bd_seq"]
                period_steps = int(cache_data["period_steps"])
                period_sec = float(cache_data["period_sec"])
            cfg["dyn"]["full_orbit_chief_cache"] = cache
            cfg["dyn"]["full_orbit_Ad_seq"] = Ad_seq
            cfg["dyn"]["full_orbit_Bd_seq"] = Bd_seq
            cfg["dyn"]["full_orbit_period_steps"] = int(period_steps)
            cfg["dyn"]["full_orbit_period_sec"] = float(period_sec)
            if cache_path is not None:
                cfg["dyn"]["full_orbit_cache_path"] = str(cache_path)
                cfg["dyn"]["full_orbit_cache_loaded"] = bool(cache_data is not None)
            set_elliptic_ltv_phase(cfg, float(orb.get("nu0", 0.0)), steps=N)
        else:
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
