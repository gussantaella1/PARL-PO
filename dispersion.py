"""
Initial-condition and parameter-dispersion helpers for Monte Carlo rollouts.
"""

# dispersion.py
from __future__ import annotations
import copy
import numpy as np
from typing import Any, Dict, Optional, Tuple

def get_center_and_radius(cfg: Dict[str, Any], D: int) -> Tuple[np.ndarray, float]:
    """Handle get center and radius for this workflow."""
    ar = cfg.get("arena", {}) or {}
    cx, cy = float(ar.get("cx", 0.0)), float(ar.get("cy", 0.0))
    cz = float(ar.get("cz", 0.0)) if D == 3 else 0.0
    r = float(ar.get("r", 20.0))
    return np.array([cx, cy, cz], dtype=float)[:D], r

def sample_uniform_ball(rng: np.random.Generator, center: np.ndarray, radius: float) -> np.ndarray:
    """Sample uniform ball for training, evaluation, or rollout initialization."""
    D = center.size
    v = rng.normal(size=D)
    v = v / (np.linalg.norm(v) + 1e-12)
    u = rng.random()
    rad = radius * (u ** (1.0 / D))
    return center + rad * v

def apply_x0_jitter(x0: np.ndarray, rng: np.random.Generator, pos_sigma: float, vel_sigma: float, D: int) -> np.ndarray:
    """Apply x0 jitter to the current config, state, or rollout data."""
    out = x0.copy().astype(float)
    if pos_sigma > 0:
        out[:, :D] += rng.normal(size=out[:, :D].shape) * pos_sigma
    if vel_sigma > 0:
        out[:, D:2*D] += rng.normal(size=out[:, D:2*D].shape) * vel_sigma
    return out

def apply_param_dispersion(cfg: Dict[str, Any], rng: np.random.Generator) -> None:
    """Apply param dispersion to the current config, state, or rollout data."""
    disp = cfg.get("dispersion", {})
    p = (disp.get("params", {}) or {})
    if not p.get("enabled", False):
        return

    for key, spec in p.items():
        if key == "enabled":
            continue
        if key not in cfg:
            continue
        dist = spec.get("dist", "normal")
        sigma = float(spec.get("sigma", 0.0))
        if sigma <= 0:
            continue

        base = float(cfg[key])

        if dist == "normal":
            cfg[key] = float(base + rng.normal() * sigma)
        elif dist == "lognormal":
            # multiplicative: base * exp(N(0, sigma))
            cfg[key] = float(base * np.exp(rng.normal() * sigma))
        else:
            raise ValueError(f"Unknown dist '{dist}' for param '{key}'")

def apply_noise_cfg(cfg: Dict[str, Any]) -> None:
    """
    IMPORTANT: this part must match whatever keys your runner actually reads.
    Here’s a generic pattern; rename keys to what your runner uses.
    """
    disp = cfg.get("dispersion", {})
    meas = disp.get("meas_noise", {}) or {}
    proc = disp.get("proc_noise", {}) or {}

    cfg["meas_noise_enabled"] = bool(meas.get("enabled", False))
    cfg["meas_pos_sigma"] = float(meas.get("pos_sigma", 0.0))
    cfg["meas_vel_sigma"] = float(meas.get("vel_sigma", 0.0))

    cfg["proc_noise_enabled"] = bool(proc.get("enabled", False))
    cfg["proc_acc_sigma"] = float(proc.get("acc_sigma", 0.0))

def build_episode_cfg_and_x0(cfg0: Dict[str, Any], episode_idx: int, trials_row: Optional[Dict[str, float]] = None):
    """Build episode cfg and x0 for the current workflow."""
    cfg = copy.deepcopy(cfg0)
    disp = cfg.get("dispersion", {}) or {}
    D = int(cfg.get("D", 3))
    num_attackers = int(cfg.get("num_attackers", 1))

    seed_base = int((disp.get("seed", {}) or {}).get("episode_seed_base", 0))
    ep_seed = seed_base + int(episode_idx)
    rng = np.random.default_rng(ep_seed)

    # Apply param dispersion first (so it affects umax/dt/etc before rollout)
    apply_param_dispersion(cfg, rng)

    # Apply noise cfg keys (runner reads these)
    apply_noise_cfg(cfg)

    ic = disp.get("ic", {}) or {}
    mode = ic.get("mode", "fixed")

    if trials_row is not None:
        # CSV-based
        p_def = np.array([trials_row["def_x"], trials_row["def_y"], trials_row["def_z"]], dtype=float)
        v_def = np.array([trials_row.get("def_vx", 0.0), trials_row.get("def_vy", 0.0), trials_row.get("def_vz", 0.0)], dtype=float)
        p_att = np.array([trials_row["att1_x"], trials_row["att1_y"], trials_row["att1_z"]], dtype=float)
        v_att = np.array([trials_row.get("att1_vx", 0.0), trials_row.get("att1_vy", 0.0), trials_row.get("att1_vz", 0.0)], dtype=float)
        x0 = np.stack([np.concatenate([p_def, v_def]), np.concatenate([p_att, v_att])], axis=0)

    elif mode == "sample_ball":
        center, R = get_center_and_radius(cfg, D)
        sample_r = float(ic.get("pos_scale", 0.95)) * float(R)
        vel_sigma = float(ic.get("vel_sigma", 0.0))
        min_sep = float(ic.get("min_sep", 0.0))

        p_def = sample_uniform_ball(rng, center, sample_r)
        v_def = rng.normal(size=D) * vel_sigma
        xs = [np.concatenate([p_def, v_def])]

        for _ in range(num_attackers):
            for _try in range(200):
                p_att = sample_uniform_ball(rng, center, sample_r)
                if min_sep <= 0 or np.linalg.norm(p_att - p_def) >= min_sep:
                    break
            v_att = rng.normal(size=D) * vel_sigma
            xs.append(np.concatenate([p_att, v_att]))

        x0 = np.asarray(xs, dtype=float)

    else:
        # fixed
        x0 = np.asarray(cfg["x0"], dtype=float)

    # Jitter (always optional)
    jit = disp.get("x0_jitter", {}) or {}
    x0 = apply_x0_jitter(
        x0,
        rng=rng,
        pos_sigma=float(jit.get("pos_sigma", 0.0)),
        vel_sigma=float(jit.get("vel_sigma", 0.0)),
        D=D
    )

    return cfg, x0, ep_seed
