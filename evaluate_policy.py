#!/usr/bin/env python3
"""
evaluate_policy.py

Statistical verification harness for your Diff-Nash RL rollout runner:

  - game_runner_diff.run_rhc_with_rl_and_collect_frames_3d_diff(cfg, steps=...)

This version is aligned with your current game_runner_diff.py.

Outputs (all written under --out_dir):
  - results.json : aggregate stats + Wilson CI on pass rate
  - trials.csv   : per-trial metrics + seed + initial conditions (+ grid indices if cartesian)
  - starts_xy.png, starts_xz.png : start position coverage (def + attacker(s))

Grid options:
  1) --trials_in (paired CSV)                 [grid_mode=paired]
  2) --def_trials_in + --att_trials_in        [grid_mode=cartesian]
  3) --auto_shell_grid                        [paired or cartesian, no CSV]
  4) --sample_ic                              [random sampling, no CSV]
  5) default: use cfg['x0']                   [single IC]

NEW:
  - --auto_shell_grid creates discrete testing points on concentric spherical shells
    inside the arena, without needing CSV files.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

# IMPORTANT: must match your CURRENT runner
from game_runner_diff import run_rhc_with_rl_and_collect_frames_3d_diff
from dispersion import build_episode_cfg_and_x0


# ------------------------- ckpt overrides -------------------------

def _apply_ckpt_overrides(cfg, def_path=None, att_path=None):
    if def_path is not None:
        dp = str(Path(def_path))
        cfg["def_ckpt_path"] = dp
        cfg["def_ckpt"] = dp
        cfg["def_policy_path"] = dp
        cfg["defender_ckpt_path"] = dp
        cfg["defender_ckpt"] = dp

    if att_path is not None:
        ap = str(Path(att_path))
        cfg["att_ckpt_path"] = ap
        cfg["att_ckpt"] = ap
        cfg["att_policy_path"] = ap
        cfg["attacker_ckpt_path"] = ap
        cfg["attacker_ckpt"] = ap


# ------------------------- CSV loaders -------------------------

def _load_trials_csv(path: str, D: int) -> List[Dict[str, float]]:
    """
    Paired mode:
      def_x,def_y,def_z,(optional def_vx,def_vy,def_vz)
      att1_x,att1_y,att1_z,(optional att1_vx,att1_vy,att1_vz)
    """
    rows: List[Dict[str, float]] = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        required = ["def_x", "def_y", "def_z", "att1_x", "att1_y", "att1_z"]
        for k in required:
            if k not in (r.fieldnames or []):
                raise RuntimeError(f"trials_in CSV missing required column: '{k}'")

        for row in r:
            out: Dict[str, float] = {}
            for k, v in row.items():
                if v is None or v == "":
                    continue
                try:
                    out[k] = float(v)
                except Exception:
                    pass
            rows.append(out)

    if D != 3:
        raise RuntimeError(f"_load_trials_csv currently expects D=3, got D={D}")
    return rows


def _load_def_csv(path: str, D: int) -> List[Dict[str, float]]:
    """
    Defender-only grid:
      def_x,def_y,def_z,(optional def_vx,def_vy,def_vz)
    """
    rows: List[Dict[str, float]] = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        required = ["def_x", "def_y", "def_z"]
        for k in required:
            if k not in (r.fieldnames or []):
                raise RuntimeError(f"def_trials_in CSV missing required column: '{k}'")

        for row in r:
            out: Dict[str, float] = {}
            for k, v in row.items():
                if v is None or v == "":
                    continue
                try:
                    out[k] = float(v)
                except Exception:
                    pass
            rows.append(out)

    if D != 3:
        raise RuntimeError(f"_load_def_csv currently expects D=3, got D={D}")
    return rows


def _load_att_csv(path: str, D: int) -> List[Dict[str, float]]:
    """
    Attacker-only grid:
      att1_x,att1_y,att1_z,(optional att1_vx,att1_vy,att1_vz)
    """
    rows: List[Dict[str, float]] = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        required = ["att1_x", "att1_y", "att1_z"]
        for k in required:
            if k not in (r.fieldnames or []):
                raise RuntimeError(f"att_trials_in CSV missing required column: '{k}'")

        for row in r:
            out: Dict[str, float] = {}
            for k, v in row.items():
                if v is None or v == "":
                    continue
                try:
                    out[k] = float(v)
                except Exception:
                    pass
            rows.append(out)

    if D != 3:
        raise RuntimeError(f"_load_att_csv currently expects D=3, got D={D}")
    return rows


# ------------------------- stats -------------------------

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _quantiles(x: np.ndarray, qs=(0.0, 0.25, 0.5, 0.75, 1.0)) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {f"q{int(100*q):02d}": float("nan") for q in qs}
    vals = np.quantile(x, qs)
    return {f"q{int(100*q):02d}": float(v) for q, v in zip(qs, vals)}


# ------------------------- config helpers -------------------------

def _get_center_and_radius(cfg: Dict[str, Any], D: int) -> Tuple[np.ndarray, float]:
    ar = cfg.get("arena", {}) or {}
    cx, cy = float(ar.get("cx", 0.0)), float(ar.get("cy", 0.0))
    cz = float(ar.get("cz", 0.0)) if D == 3 else 0.0
    r = float(ar.get("r", 20.0))
    center = np.array([cx, cy, cz], dtype=float)[:D]
    return center, r


def _radius_from_cfg(val: float, arena_r: float) -> float:
    """
    Interpret radii that may be "fraction of arena R" or meters.

    Rule:
      - if val <= 0: disabled -> 0
      - if 0 < val <= 1.0: treat as fraction of arena_r
      - else: treat as meters
    """
    if val <= 0.0:
        return 0.0
    if val <= 1.0:
        return float(val) * float(arena_r)
    return float(val)


def _apply_x0_jitter(cfg: Dict[str, Any], x0: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    jit = cfg.get("x0_jitter", {}) or {}
    pos_j = float(jit.get("pos", 0.0))
    vel_j = float(jit.get("vel", 0.0))
    D = int(cfg.get("D", x0.shape[1] // 2))
    out = x0.copy().astype(float)
    out[:, :D] += rng.normal(size=out[:, :D].shape) * pos_j
    out[:, D:2*D] += rng.normal(size=out[:, D:2*D].shape) * vel_j
    return out


# ------------------------- auto shell grid (no CSV) -------------------------

def _fibonacci_sphere_points(n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Points on unit sphere via Fibonacci lattice. Returns (n,3).
    If rng is provided, applies a random rotation (scramble) while keeping uniformity.
    """
    n = int(n)
    if n <= 0:
        return np.zeros((0, 3), dtype=float)

    i = np.arange(n, dtype=float)
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    theta = 2.0 * np.pi * i / phi
    z = 1.0 - 2.0 * (i + 0.5) / n
    r_xy = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    x = r_xy * np.cos(theta)
    y = r_xy * np.sin(theta)
    pts = np.stack([x, y, z], axis=1)

    if rng is not None:
        u1, u2, u3 = rng.random(3)
        q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
        q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
        q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
        q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
        xq, yq, zq, wq = q1, q2, q3, q4
        R = np.array([
            [1 - 2*(yq*yq + zq*zq),     2*(xq*yq - zq*wq),     2*(xq*zq + yq*wq)],
            [    2*(xq*yq + zq*wq), 1 - 2*(xq*xq + zq*zq),     2*(yq*zq - xq*wq)],
            [    2*(xq*zq - yq*wq),     2*(yq*zq + xq*wq), 1 - 2*(xq*xq + yq*yq)],
        ], dtype=float)
        pts = pts @ R.T

    return pts


def _generate_shelled_positions(
    center: np.ndarray,
    arena_r: float,
    shell_fracs: List[float],
    points_per_shell: int,
    rng: Optional[np.random.Generator] = None,
    include_center: bool = False,
) -> np.ndarray:
    """
    Generate positions on shells at radii = frac * arena_r. Returns (N,3).
    """
    shell_fracs = [float(s) for s in shell_fracs if float(s) > 0.0]
    pts_all = []
    if include_center:
        pts_all.append(center.reshape(1, 3).copy())

    for frac in shell_fracs:
        rad = frac * float(arena_r)
        unit = _fibonacci_sphere_points(points_per_shell, rng=rng)
        pts_all.append(center[None, :] + rad * unit)

    if not pts_all:
        return np.zeros((0, 3), dtype=float)

    return np.concatenate(pts_all, axis=0)


def _rows_from_positions(prefix: str, positions: np.ndarray) -> List[Dict[str, float]]:
    """
    Convert positions (N,3) to dict rows with zero velocity.
    prefix: "def" or "att1"
    """
    rows: List[Dict[str, float]] = []
    for p in positions:
        rows.append({
            f"{prefix}_x": float(p[0]),
            f"{prefix}_y": float(p[1]),
            f"{prefix}_z": float(p[2]),
            f"{prefix}_vx": 0.0,
            f"{prefix}_vy": 0.0,
            f"{prefix}_vz": 0.0,
        })
    return rows


def _make_paired_rows_from_def_att(def_rows: List[Dict[str, float]],
                                   att_rows: List[Dict[str, float]],
                                   n: int) -> List[Dict[str, float]]:
    n_pair = min(len(def_rows), len(att_rows), int(n))
    out: List[Dict[str, float]] = []
    for i in range(n_pair):
        rd, ra = def_rows[i], att_rows[i]
        out.append({
            "def_x": rd["def_x"], "def_y": rd["def_y"], "def_z": rd["def_z"],
            "def_vx": rd.get("def_vx", 0.0), "def_vy": rd.get("def_vy", 0.0), "def_vz": rd.get("def_vz", 0.0),
            "att1_x": ra["att1_x"], "att1_y": ra["att1_y"], "att1_z": ra["att1_z"],
            "att1_vx": ra.get("att1_vx", 0.0), "att1_vy": ra.get("att1_vy", 0.0), "att1_vz": ra.get("att1_vz", 0.0),
        })
    return out


# ------------------------- scenario generation (random sampling) -------------------------

def _sample_uniform_ball(rng: np.random.Generator, center: np.ndarray, radius: float) -> np.ndarray:
    D = center.size
    v = rng.normal(size=D)
    v = v / (np.linalg.norm(v) + 1e-12)
    u = rng.random()
    rad = radius * (u ** (1.0 / D))
    return center + rad * v


def _sample_x0(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    num_attackers: int,
    pos_scale: float,
    vel_scale: float,
    min_sep: float = 0.0
) -> np.ndarray:
    """
    Returns x0 with shape (1 + num_attackers, 2D).
    Enforces a minimum defender-attacker separation if min_sep > 0.
    """
    D = int(cfg.get("D", 3))
    center, R = _get_center_and_radius(cfg, D)
    nx = 2 * D
    sample_r = float(pos_scale) * float(R)

    # Defender
    p_def = _sample_uniform_ball(rng, center, sample_r)
    v_def = rng.normal(size=D) * float(vel_scale)
    x_def = np.concatenate([p_def, v_def], dtype=float)[:nx]

    xs = [x_def]
    for _ in range(num_attackers):
        for _try in range(200):
            p_att = _sample_uniform_ball(rng, center, sample_r)
            if min_sep <= 0 or np.linalg.norm(p_att - p_def) >= min_sep:
                break
        v_att = rng.normal(size=D) * float(vel_scale)
        x_att = np.concatenate([p_att, v_att], dtype=float)[:nx]
        xs.append(x_att)

    return np.asarray(xs, dtype=float)


# ------------------------- metrics -------------------------

def _extract_positions(out: Dict[str, Any], D: int) -> Tuple[np.ndarray, np.ndarray]:
    p1 = np.asarray(out["exec1_xyz"], dtype=float)
    p2 = np.asarray(out["exec2_xyz"], dtype=float)
    p1 = p1[:, :max(3, D)]
    p2 = p2[:, :max(3, D)]
    return p1, p2


def _compute_trial_metrics(cfg: Dict[str, Any], out: Dict[str, Any]) -> Dict[str, Any]:
    D = int(cfg.get("D", 3))
    center, arena_r = _get_center_and_radius(cfg, D)

    p_def, p_att = _extract_positions(out, D)
    def_pos = p_def[:, :D]
    att_pos = p_att[:, :D]

    rel = np.linalg.norm(def_pos - att_pos, axis=1)
    att_center = np.linalg.norm(att_pos - center[None, :], axis=1)
    def_center = np.linalg.norm(def_pos - center[None, :], axis=1)

    collision_r = float(cfg.get("collision_radius_m", 0.0))
    cap_idx = np.where(rel <= collision_r)[0] if collision_r > 0 else np.array([], dtype=int)
    collided = bool(cap_idx.size > 0)
    t_collide = int(cap_idx[0]) if collided else -1

    att_hit_val = float(cfg.get("att_target_hit_radius", 0.0))
    att_hit_r = _radius_from_cfg(att_hit_val, arena_r)
    hit_idx = np.where(att_center <= att_hit_r)[0] if att_hit_r > 0 else np.array([], dtype=int)
    attacker_hit = bool(hit_idx.size > 0)
    t_att_hit = int(hit_idx[0]) if attacker_hit else -1

    term_margin = float(cfg.get("arena_terminate_margin", 0.0))
    att_oob = bool(np.any(att_center > arena_r))
    def_oob = bool(np.any(def_center > arena_r))
    att_term = bool(np.any(att_center > arena_r + term_margin))
    def_term = bool(np.any(def_center > arena_r + term_margin))

    oi = cfg.get("oi", {}) or {}
    oi_enabled = bool(oi.get("enabled", False))
    oi_r = float(oi.get("r", 0.0))
    avoid_by = oi.get("avoid_by", [])
    avoid_by = list(avoid_by) if isinstance(avoid_by, (list, tuple)) else [avoid_by]

    def_keepout_buffer_m = float(cfg.get("def_keepout_buffer_m", 0.0))
    buf_val = float(cfg.get("def_oi_safety_buffer", 0.0))
    extra = (buf_val * oi_r) if (0.0 < buf_val <= 1.0) else buf_val
    oi_safe_r = oi_r + def_keepout_buffer_m + extra

    oi_viol_def = False
    oi_viol_att = False
    if oi_enabled and oi_safe_r > 0:
        if 0 in avoid_by:
            oi_viol_def = bool(np.any(def_center <= oi_safe_r))
        if 1 in avoid_by:
            oi_viol_att = bool(np.any(att_center <= oi_safe_r))

    u_norms = out.get("u_cmd_norm_all", None)
    udef_mean = uatt_mean = udef_max = uatt_max = float("nan")
    if u_norms is not None and len(u_norms) >= 2:
        udef = np.asarray(u_norms[0], dtype=float)
        uatt = np.asarray(u_norms[1], dtype=float)
        if udef.size:
            udef_mean, udef_max = float(np.mean(udef)), float(np.max(udef))
        if uatt.size:
            uatt_mean, uatt_max = float(np.mean(uatt)), float(np.max(uatt))

    verify_require_capture = bool(cfg.get("verify_require_capture", False))
    pass_flag = (
        (not attacker_hit)
        and (not att_term) and (not def_term)
        and (not oi_viol_def) and (not oi_viol_att)
    )
    if verify_require_capture and collision_r > 0:
        pass_flag = pass_flag and collided

    return {
        "pass": int(pass_flag),

        "min_rel_dist": float(np.min(rel)),
        "t_min_rel": int(np.argmin(rel)),

        "min_att_center": float(np.min(att_center)),
        "t_min_att_center": int(np.argmin(att_center)),

        "collided": int(collided),
        "t_collide": t_collide,

        "attacker_hit": int(attacker_hit),
        "t_att_hit": t_att_hit,

        "att_oob": int(att_oob),
        "def_oob": int(def_oob),
        "att_term": int(att_term),
        "def_term": int(def_term),

        "oi_viol_def": int(oi_viol_def),
        "oi_viol_att": int(oi_viol_att),

        "udef_mean": udef_mean,
        "uatt_mean": uatt_mean,
        "udef_max": udef_max,
        "uatt_max": uatt_max,
    }


# ------------------------- plotting -------------------------

def _save_start_plots(
    out_dir: Path,
    cfg: Dict[str, Any],
    D: int,
    starts_def: np.ndarray,
    starts_atts: List[np.ndarray],
    trial_rows: List[Dict[str, Any]],
    cmap: str = "viridis",
    edge_lw: float = 0.7,
    marker_size: float = 45.0,
) -> None:
    import matplotlib.pyplot as plt

    if D != 3:
        raise RuntimeError("_save_start_plots currently expects D=3.")

    center, arena_r = _get_center_and_radius(cfg, D)

    # Pull OI from cfg (like other functions)
    oi = cfg.get("oi", {}) or {}
    oi_enabled = bool(oi.get("enabled", False))
    oi_r = float(oi.get("r", 0.0)) if oi_enabled else 0.0

    # -------------------------
    # helpers
    # -------------------------
    def _set_equal(ax):
        ax.set_aspect("equal", adjustable="box")

    def _draw_circle(ax, plane: str, r: float, label: str = None, lw: float = 1.2):
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        if plane == "xy":
            circ = plt.Circle((cx, cy), r, fill=False, linewidth=lw)
            ax.add_patch(circ)
            if label:
                ax.text(cx + r, cy, label, fontsize=8, va="center")
        elif plane == "xz":
            circ = plt.Circle((cx, cz), r, fill=False, linewidth=lw)
            ax.add_patch(circ)
            if label:
                ax.text(cx + r, cz, label, fontsize=8, va="center")
        else:
            raise ValueError(f"unknown plane={plane}")

    def _decorate(ax, plane: str, title: str, xlabel: str, ylabel: str):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        # Arena boundary
        _draw_circle(ax, plane, float(arena_r), label="arena", lw=1.2)

        # OI radius
        if oi_enabled and oi_r > 0:
            _draw_circle(ax, plane, float(oi_r), label="OI", lw=1.2)

        _set_equal(ax)

    # -------------------------
    # 1) Plain coverage plots
    # -------------------------
    # XY
    fig, ax = plt.subplots()
    ax.scatter(
        starts_def[:, 0], starts_def[:, 1],
        marker="o", s=marker_size,
        edgecolors="k", linewidths=edge_lw,
        label="def_start"
    )
    for j, sa in enumerate(starts_atts):
        if sa.size == 0:
            continue
        ax.scatter(
            sa[:, 0], sa[:, 1],
            marker="o", s=marker_size,
            edgecolors="k", linewidths=edge_lw,
            label=f"att{j+1}_start"
        )
    _decorate(ax, "xy", "Evaluated Starting Positions (XY)", "x", "y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "starts_xy.png", dpi=180)
    plt.close(fig)

    # XZ
    fig, ax = plt.subplots()
    ax.scatter(
        starts_def[:, 0], starts_def[:, 2],
        marker="o", s=marker_size,
        edgecolors="k", linewidths=edge_lw,
        label="def_start"
    )
    for j, sa in enumerate(starts_atts):
        if sa.size == 0:
            continue
        ax.scatter(
            sa[:, 0], sa[:, 2],
            marker="o", s=marker_size,
            edgecolors="k", linewidths=edge_lw,
            label=f"att{j+1}_start"
        )
    _decorate(ax, "xz", "Evaluated Starting Positions (XZ)", "x", "z")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "starts_xz.png", dpi=180)
    plt.close(fig)

    # -------------------------
    # 2) Success-rate overlays
    # -------------------------
    def _key(p, nd=6):
        return (round(float(p[0]), nd), round(float(p[1]), nd), round(float(p[2]), nd))

    def_counts: Dict[Tuple[float, float, float], List[int]] = {}
    att_counts: Dict[Tuple[float, float, float], List[int]] = {}

    for r in trial_rows:
        p = int(r.get("pass_trial", 0))
        kd = _key([r["def_x"], r["def_y"], r["def_z"]])
        ka = _key([r["att1_x"], r["att1_y"], r["att1_z"]])

        def_counts.setdefault(kd, [0, 0])
        att_counts.setdefault(ka, [0, 0])

        def_counts[kd][0] += p
        def_counts[kd][1] += 1
        att_counts[ka][0] += p
        att_counts[ka][1] += 1

    def_pts = np.array(list(def_counts.keys()), dtype=float)
    def_rates = np.array([def_counts[k][0] / max(1, def_counts[k][1]) for k in def_counts], dtype=float)

    att_pts = np.array(list(att_counts.keys()), dtype=float)
    att_rates = np.array([att_counts[k][0] / max(1, att_counts[k][1]) for k in att_counts], dtype=float)

    # Defender success XY
    fig, ax = plt.subplots()
    sc = ax.scatter(
        def_pts[:, 0], def_pts[:, 1],
        c=def_rates, cmap=cmap,
        marker="o", s=marker_size,
        edgecolors="k", linewidths=edge_lw
    )
    _decorate(ax, "xy", "Defender start: pass rate over attacker starts (XY)", "x", "y")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("pass rate")
    fig.tight_layout()
    fig.savefig(out_dir / "def_success_xy.png", dpi=180)
    plt.close(fig)

    # Defender success XZ
    fig, ax = plt.subplots()
    sc = ax.scatter(
        def_pts[:, 0], def_pts[:, 2],
        c=def_rates, cmap=cmap,
        marker="o", s=marker_size,
        edgecolors="k", linewidths=edge_lw
    )
    _decorate(ax, "xz", "Defender start: pass rate over attacker starts (XZ)", "x", "z")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("pass rate")
    fig.tight_layout()
    fig.savefig(out_dir / "def_success_xz.png", dpi=180)
    plt.close(fig)

    # Attacker success XY
    fig, ax = plt.subplots()
    sc = ax.scatter(
        att_pts[:, 0], att_pts[:, 1],
        c=att_rates, cmap=cmap,
        marker="o", s=marker_size,
        edgecolors="k", linewidths=edge_lw
    )
    _decorate(ax, "xy", "Attacker start: pass rate over defender starts (XY)", "x", "y")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("pass rate")
    fig.tight_layout()
    fig.savefig(out_dir / "att_success_xy.png", dpi=180)
    plt.close(fig)

    # Attacker success XZ
    fig, ax = plt.subplots()
    sc = ax.scatter(
        att_pts[:, 0], att_pts[:, 2],
        c=att_rates, cmap=cmap,
        marker="o", s=marker_size,
        edgecolors="k", linewidths=edge_lw
    )
    _decorate(ax, "xz", "Attacker start: pass rate over defender starts (XZ)", "x", "z")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("pass rate")
    fig.tight_layout()
    fig.savefig(out_dir / "att_success_xz.png", dpi=180)
    plt.close(fig)


# ------------------------- main -------------------------

def main():
    def log(msg: str) -> None:
        print(msg, flush=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config_module", default="config_rl",
                    help="Python module containing config_for_eval and build_dyn (default: config_rl).")
    ap.add_argument("--out_dir", default="eval_out", help="Output directory.")
    ap.add_argument("--num_trials", type=int, default=200, help="Number of trials (paired mode).")
    ap.add_argument("--seed", type=int, default=0, help="Base seed.")
    ap.add_argument("--steps", type=int, default=None,
                    help="Override rollout steps (also sets cfg['T'] for eval and dyn sizing).")

    # sampling option (random ICs)
    ap.add_argument("--sample_ic", action="store_true",
                    help="If set, ignore cfg['x0'] and sample starts in arena.")
    ap.add_argument("--pos_scale", type=float, default=0.95, help="Sample within pos_scale * arena_r.")
    ap.add_argument("--vel_scale", type=float, default=0.0, help="Stddev of sampled initial velocities.")
    ap.add_argument("--min_sep", type=float, default=0.0, help="Min defender-attacker separation when sampling.")

    # multi-attacker aggregation mode
    ap.add_argument("--multi_att_mode", default="worst",
                    choices=["worst", "avg"],
                    help="If num_attackers>1, evaluate defender vs each attacker separately then aggregate.")

    # common overrides
    ap.add_argument("--device", default=None)
    ap.add_argument("--def_ckpt_path", default=None)
    ap.add_argument("--att_ckpt_path", default=None)
    ap.add_argument("--deterministic", action="store_true", help="Force rl_eval_deterministic=True")
    ap.add_argument("--use_ukf", action="store_true", help="Force use_ukf=True")
    ap.add_argument("--x0_pos_jitter", type=float, default=None)
    ap.add_argument("--x0_vel_jitter", type=float, default=None)

    ap.add_argument("--alpha", type=float, default=0.05, help="CI alpha (0.05 => 95% CI).")

    # progress/debug
    ap.add_argument("--log_every", type=int, default=10,
                    help="Print progress every N trials (default: 10).")
    ap.add_argument("--print_first_out_keys", action="store_true",
                    help="Print keys and a couple quick sanity checks from the first rollout.")
    ap.add_argument("--print_errors", action="store_true",
                    help="Print exception messages when a trial fails.")
    ap.add_argument("--trace_errors", action="store_true",
                    help="Also print full traceback for failed trials (implies --print_errors).")

    # paired CSV (existing)
    ap.add_argument("--trials_in", default=None,
                    help="CSV of paired start states (def+att). Overrides cfg['x0'] and --sample_ic (paired mode).")

    # grid mode (existing)
    ap.add_argument("--grid_mode", default="paired", choices=["paired", "cartesian"],
                    help="paired: use --trials_in as (def,att) rows. "
                         "cartesian: use --def_trials_in x --att_trials_in product.")
    ap.add_argument("--def_trials_in", default=None,
                    help="CSV of defender start states (def_x,def_y,def_z[,def_vx,def_vy,def_vz]).")
    ap.add_argument("--att_trials_in", default=None,
                    help="CSV of attacker start states (att1_x,att1_y,att1_z[,att1_vx,att1_vy,att1_vz]).")
    ap.add_argument("--max_pairs", type=int, default=None,
                    help="If set in cartesian mode, cap total evaluated pairs (sampled).")

    # NEW: auto shell grid (no CSV)
    ap.add_argument("--auto_shell_grid", action="store_true",
                    help="Generate discrete defender/attacker start grids on spherical shells (no CSV).")
    ap.add_argument("--shell_fracs", type=str, default="0.2,0.4,0.6,0.8",
                    help="Comma-separated shell radii as fractions of arena radius (0,1].")
    ap.add_argument("--points_per_shell", type=int, default=40,
                    help="Number of points per shell (per agent grid).")
    ap.add_argument("--include_center", action="store_true",
                    help="Also include the arena center as an additional point in the grid.")
    
    ap.add_argument(
        "--dynamics",
        default=None,
        choices=["hcw", "two_body", "elliptic_ltv"],
        help="Override dynamics model used in rollouts (overrides cfg['dynamics'])."
    )
    ap.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Optional override for cfg['dt'] (if you want)."
    )

    args = ap.parse_args()
    if args.trace_errors:
        args.print_errors = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_global0 = time.time()
    log(f"[eval] starting; out_dir={out_dir.resolve()}")
    log(f"[eval] config_module={args.config_module}")

    # Import config module
    log("[eval] importing config module...")
    mod = importlib.import_module(args.config_module)
    if not hasattr(mod, "config_for_eval") or not hasattr(mod, "build_dyn"):
        raise RuntimeError(f"Module '{args.config_module}' must define config_for_eval(...) and build_dyn(cfg).")
    log("[eval] config module imported OK.")

    cfg0: Dict[str, Any] = mod.config_for_eval()

    log("[eval] base cfg created from config_for_eval().")


    # Dynamics override (must happen before build_dyn)
    if args.dynamics is not None:
        cfg0["dynamics"] = str(args.dynamics)

    if args.dt is not None:
        cfg0["dt"] = float(args.dt)

    mod.build_dyn(cfg0)

    # Apply overrides
    if args.device is not None:
        cfg0["device"] = args.device

    _apply_ckpt_overrides(cfg0, args.def_ckpt_path, args.att_ckpt_path)

    if args.deterministic:
        cfg0["rl_eval_deterministic"] = True
    if args.use_ukf:
        cfg0["use_ukf"] = True

    if args.x0_pos_jitter is not None or args.x0_vel_jitter is not None:
        cfg0.setdefault("x0_jitter", {})
        if args.x0_pos_jitter is not None:
            cfg0["x0_jitter"]["pos"] = float(args.x0_pos_jitter)
        if args.x0_vel_jitter is not None:
            cfg0["x0_jitter"]["vel"] = float(args.x0_vel_jitter)

    if args.steps is not None:
        cfg0["T"] = int(args.steps)

    D = int(cfg0.get("D", 3))
    num_attackers = max(1, int(cfg0.get("num_attackers", 1)))
    dyn_name = str(cfg0.get("dynamics", "hcw"))
    log(f"[eval] cfg summary: D={D} num_attackers={num_attackers} dynamics={dyn_name}")
    log(f"[eval] ckpts: def={cfg0.get('def_ckpt_path', None)} att={cfg0.get('att_ckpt_path', None)} "
        f"device={cfg0.get('device', None)} deterministic={cfg0.get('rl_eval_deterministic', None)} use_ukf={cfg0.get('use_ukf', False)}")

    # ---------------------------
    # Build dyn (once, based on cfg0)
    # ---------------------------
    log("[eval] building dynamics via build_dyn(cfg)...")
    t_dyn0 = time.time()
    mod.build_dyn(cfg0)
    log(f"[eval] build_dyn done in {time.time() - t_dyn0:.3f}s.")

    # ---------------------------
    # Prepare trial lists (paired/cartesian/auto/random/default)
    # ---------------------------
    paired_rows: Optional[List[Dict[str, float]]] = None
    def_rows: Optional[List[Dict[str, float]]] = None
    att_rows: Optional[List[Dict[str, float]]] = None

    n_total: int

    # 3) AUTO SHELL GRID (no CSV)
    if args.auto_shell_grid:
        if D != 3:
            raise RuntimeError("--auto_shell_grid currently supports only D=3 (sphere).")

        center, arena_r = _get_center_and_radius(cfg0, D)

        shell_fracs = [float(s.strip()) for s in args.shell_fracs.split(",") if s.strip() != ""]
        shell_fracs = [s for s in shell_fracs if 0.0 < s <= 1.0]
        if not shell_fracs:
            raise RuntimeError("No valid --shell_fracs provided (must be in (0,1]).")

        rng_grid = np.random.default_rng(int(args.seed) + 12345)

        def_pos = _generate_shelled_positions(center, arena_r, shell_fracs, int(args.points_per_shell),
                                              rng=rng_grid, include_center=bool(args.include_center))
        att_pos = _generate_shelled_positions(center, arena_r, shell_fracs, int(args.points_per_shell),
                                              rng=rng_grid, include_center=bool(args.include_center))

        def_rows = _rows_from_positions("def", def_pos)
        att_rows = _rows_from_positions("att1", att_pos)

        # Now choose how to pair them
        if args.grid_mode == "paired":
            # n_total bounded by requested num_trials and available pairs
            n_total = min(int(args.num_trials), min(len(def_rows), len(att_rows)))
            paired_rows = _make_paired_rows_from_def_att(def_rows, att_rows, n_total)
            log(f"[eval] auto_shell_grid paired: shells={shell_fracs} points_per_shell={args.points_per_shell} "
                f"include_center={args.include_center} -> paired_rows={len(paired_rows)}")
        else:
            total_pairs = len(def_rows) * len(att_rows)
            if args.max_pairs is None:
                n_total = total_pairs
                log(f"[eval] auto_shell_grid cartesian: evaluating ALL pairs = {total_pairs}")
            else:
                n_total = min(int(args.max_pairs), total_pairs)
                log(f"[eval] auto_shell_grid cartesian: total pairs={total_pairs}, evaluating={n_total} (sampled)")
            log(f"[eval] auto_shell_grid cartesian grids: def_rows={len(def_rows)} att_rows={len(att_rows)}")

        # Make CSV args irrelevant
        args.trials_in = None
        args.def_trials_in = None
        args.att_trials_in = None

    # 1) PAIRED CSV
    elif args.grid_mode == "paired" and args.trials_in is not None:
        paired_rows = _load_trials_csv(args.trials_in, D)
        log(f"[eval] loaded paired trials_in: {args.trials_in} ({len(paired_rows)} rows)")
        if len(paired_rows) < int(args.num_trials):
            raise RuntimeError(
                f"--trials_in has {len(paired_rows)} rows but --num_trials={args.num_trials}. "
                "Either reduce --num_trials or provide more rows."
            )
        n_total = int(args.num_trials)
        log(f"[eval] paired CSV mode: num_trials={n_total}")

    # 2) CARTESIAN CSV
    elif args.grid_mode == "cartesian":
        if args.def_trials_in is None or args.att_trials_in is None:
            raise RuntimeError("cartesian mode requires --def_trials_in and --att_trials_in (unless using --auto_shell_grid)")
        def_rows = _load_def_csv(args.def_trials_in, D)
        att_rows = _load_att_csv(args.att_trials_in, D)

        total_pairs = len(def_rows) * len(att_rows)
        if total_pairs <= 0:
            raise RuntimeError("cartesian mode: empty def/att grids")

        if args.max_pairs is None:
            n_total = total_pairs
            log(f"[eval] cartesian mode: evaluating ALL pairs = {total_pairs}")
        else:
            n_total = min(int(args.max_pairs), total_pairs)
            log(f"[eval] cartesian mode: total pairs={total_pairs}, evaluating={n_total} (sampled)")

        log(f"[eval] loaded defender grid: {args.def_trials_in} ({len(def_rows)} rows)")
        log(f"[eval] loaded attacker grid: {args.att_trials_in} ({len(att_rows)} rows)")

    # 4) RANDOM SAMPLING
    elif args.sample_ic:
        n_total = int(args.num_trials)
        log(f"[eval] random sampling: num_trials={n_total} pos_scale={args.pos_scale} vel_scale={args.vel_scale} min_sep={args.min_sep}")

    # 5) DEFAULT SINGLE x0
    else:
        n_total = 1
        log("[eval] default mode: using cfg['x0'] once (n_total=1).")

    # Start position logs
    starts_def: List[np.ndarray] = []
    starts_atts: List[List[np.ndarray]] = [[] for _ in range(num_attackers)]

    trial_rows: List[Dict[str, Any]] = []
    passes = 0

    log("[eval] beginning trials...")
    t_loop0 = time.time()

    for i in range(n_total):
        t_trial0 = time.time()
        seed = int(args.seed + i)

        # Runner uses np.random for measurement noise, so set it
        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        # Let dispersion mutate cfg per episode (kept)
        cfg_trial, _x0_unused, ep_seed = build_episode_cfg_and_x0(
            cfg0,
            episode_idx=i,
            trials_row=None,  # we handle our own grids below
        )


        if args.dynamics is not None:
            cfg_trial["dynamics"] = str(args.dynamics)
        if args.dt is not None:
            cfg_trial["dt"] = float(args.dt)
        
        np.random.seed(int(ep_seed))

        # Always keep T as a clean int if it exists
        if "T" in cfg_trial and cfg_trial["T"] is not None:
            cfg_trial["T"] = int(cfg_trial["T"])

        # ---------------------------
        # Build x0 for this trial
        # ---------------------------
        cart_di = cart_ai = None

        if paired_rows is not None:
            r = paired_rows[i]
            p_def = np.array([r["def_x"], r["def_y"], r["def_z"]], dtype=float)
            v_def = np.array([r.get("def_vx", 0.0), r.get("def_vy", 0.0), r.get("def_vz", 0.0)], dtype=float)

            p_att = np.array([r["att1_x"], r["att1_y"], r["att1_z"]], dtype=float)
            v_att = np.array([r.get("att1_vx", 0.0), r.get("att1_vy", 0.0), r.get("att1_vz", 0.0)], dtype=float)

            x0 = np.stack([np.concatenate([p_def, v_def]),
                           np.concatenate([p_att, v_att])], axis=0)

        elif args.grid_mode == "cartesian" and (def_rows is not None and att_rows is not None):
            total_pairs = len(def_rows) * len(att_rows)

            if args.max_pairs is None:
                pair_idx = i
            else:
                # sampled pairs (not guaranteed unique) - fine for MC
                pair_idx = int(rng.integers(0, total_pairs))

            di = pair_idx // len(att_rows)
            ai = pair_idx % len(att_rows)
            cart_di, cart_ai = int(di), int(ai)

            rd = def_rows[di]
            ra = att_rows[ai]

            p_def = np.array([rd["def_x"], rd["def_y"], rd["def_z"]], dtype=float)
            v_def = np.array([rd.get("def_vx", 0.0), rd.get("def_vy", 0.0), rd.get("def_vz", 0.0)], dtype=float)

            p_att = np.array([ra["att1_x"], ra["att1_y"], ra["att1_z"]], dtype=float)
            v_att = np.array([ra.get("att1_vx", 0.0), ra.get("att1_vy", 0.0), ra.get("att1_vz", 0.0)], dtype=float)

            x0 = np.stack([np.concatenate([p_def, v_def]),
                           np.concatenate([p_att, v_att])], axis=0)

        elif args.sample_ic:
            x0 = _sample_x0(
                cfg_trial, rng,
                num_attackers=num_attackers,
                pos_scale=float(args.pos_scale),
                vel_scale=float(args.vel_scale),
                min_sep=float(args.min_sep),
            )

        else:
            x0 = np.asarray(cfg_trial["x0"], dtype=float)

        x0 = _apply_x0_jitter(cfg_trial, x0, rng)

        # log starts (for plots)
        starts_def.append(x0[0, :D].copy())
        for j in range(num_attackers):
            if 1 + j < x0.shape[0]:
                starts_atts[j].append(x0[1 + j, :D].copy())

        per_att_metrics = []
        per_att_errors = 0

        # For multi-attackers, evaluate def vs each attacker separately
        for j in range(num_attackers):
            if 1 + j >= x0.shape[0]:
                continue

            cfg_run = copy.deepcopy(cfg_trial)
            cfg_run["x0"] = np.asarray([x0[0], x0[1 + j]], dtype=float).tolist()

            # force CLI ckpt paths into the rollout cfg
            _apply_ckpt_overrides(cfg_run, args.def_ckpt_path, args.att_ckpt_path)

            # Determine steps for this rollout (never pass None)
            steps_run = int(args.steps) if args.steps is not None else int(cfg_run.get("T", cfg0.get("T", 0)) or 0)
            if steps_run <= 0:
                raise RuntimeError(f"Invalid steps_run={steps_run}. Provide --steps or ensure cfg['T'] is set.")

            # Also keep cfg_run['T'] consistent with steps
            cfg_run["T"] = steps_run

            try:
                out = run_rhc_with_rl_and_collect_frames_3d_diff(cfg_run, steps=steps_run)

                if (i == 0 and j == 0 and args.print_first_out_keys):
                    log("[eval] first rollout returned keys:")
                    log("  " + ", ".join(sorted(list(out.keys()))))
                    p1 = np.asarray(out.get("exec1_xyz", []), dtype=float)
                    p2 = np.asarray(out.get("exec2_xyz", []), dtype=float)
                    log(f"[eval] first rollout sanity: len(exec1_xyz)={len(p1)} len(exec2_xyz)={len(p2)}")
                    if p1.size and p2.size:
                        d0 = float(np.linalg.norm(p1[0, :D] - p2[0, :D]))
                        dT = float(np.linalg.norm(p1[-1, :D] - p2[-1, :D]))
                        log(f"[eval] first rollout sanity: rel_dist start={d0:.3f} end={dT:.3f}")

                m = _compute_trial_metrics(cfg_run, out)

            except Exception as e:
                per_att_errors += 1
                m = {"pass": 0, "error": str(e)}
                if args.print_errors:
                    log(f"[eval][trial {i}/{n_total-1}][att {j+1}] ERROR: {e}")
                    if args.trace_errors:
                        log(traceback.format_exc())

            per_att_metrics.append(m)

        if not per_att_metrics:
            per_att_metrics = [{"pass": 0, "error": "no attackers present"}]
            per_att_errors += 1

        # Aggregate across attackers
        if args.multi_att_mode == "worst":
            pass_trial = int(all(m.get("pass", 0) == 1 for m in per_att_metrics))
        else:
            pass_trial = int(round(np.mean([m.get("pass", 0) for m in per_att_metrics])))

        passes += pass_trial

        # One row per trial; stores attacker #1 metrics by default
        row: Dict[str, Any] = {
            "trial": i,
            "seed": seed,
            "grid_mode": args.grid_mode if not args.auto_shell_grid else f"{args.grid_mode}+auto_shell",
            "pass_trial": pass_trial,
            "num_attackers": num_attackers,
            "num_att_errors": per_att_errors,

            "def_x": float(x0[0, 0]),
            "def_y": float(x0[0, 1]) if D >= 2 else 0.0,
            "def_z": float(x0[0, 2]) if D == 3 else 0.0,

            "att1_x": float(x0[1, 0]) if x0.shape[0] > 1 else float("nan"),
            "att1_y": float(x0[1, 1]) if (x0.shape[0] > 1 and D >= 2) else float("nan"),
            "att1_z": float(x0[1, 2]) if (x0.shape[0] > 1 and D == 3) else float("nan"),
        }

        if (args.grid_mode == "cartesian") and (def_rows is not None) and (att_rows is not None):
            row["def_idx"] = cart_di
            row["att_idx"] = cart_ai

        m0 = per_att_metrics[0]
        for k_, v_ in m0.items():
            row[f"att1_{k_}"] = v_

        trial_rows.append(row)

        # Progress print
        if (i == 0) or ((i + 1) % max(1, int(args.log_every)) == 0) or (i == n_total - 1):
            sr_sofar = passes / float(i + 1)
            dt_trial = time.time() - t_trial0

            last_min_rel = row.get("att1_min_rel_dist", None)
            last_hit = row.get("att1_attacker_hit", None)
            last_def_term = row.get("att1_def_term", None)
            last_att_term = row.get("att1_att_term", None)

            extra = ""
            if args.grid_mode == "cartesian":
                extra = f" | def_idx={row.get('def_idx')} att_idx={row.get('att_idx')}"

            log(f"[eval] progress {i+1}/{n_total} | pass_so_far={passes} ({sr_sofar:.3f}) | "
                f"last_pass={pass_trial} | trial_time={dt_trial:.3f}s | "
                f"min_rel={last_min_rel} hit={last_hit} def_term={last_def_term} att_term={last_att_term} "
                f"errors_this_trial={per_att_errors}{extra}")

    # Aggregate stats
    n = len(trial_rows)
    k = passes
    sr = k / max(1, n)
    lo, hi = wilson_ci(k, n, alpha=float(args.alpha))

    def _col(name: str) -> np.ndarray:
        vals = []
        for r in trial_rows:
            v = r.get(name, float("nan"))
            try:
                fv = float(v)
            except Exception:
                continue
            if not np.isnan(fv):
                vals.append(fv)
        return np.asarray(vals, dtype=float)

    results = {
        "num_trials": n,
        "passes": k,
        "success_rate": float(sr),
        "success_rate_ci_wilson": {"alpha": float(args.alpha), "lo": float(lo), "hi": float(hi)},
        "config_module": args.config_module,
        "grid_mode": args.grid_mode,
        "auto_shell_grid": bool(args.auto_shell_grid),
        "shell_fracs": args.shell_fracs if args.auto_shell_grid else None,
        "points_per_shell": int(args.points_per_shell) if args.auto_shell_grid else None,
        "include_center": bool(args.include_center) if args.auto_shell_grid else None,
        "multi_att_mode": args.multi_att_mode,
        "timing_sec": {
            "total": float(time.time() - t_global0),
            "trial_loop": float(time.time() - t_loop0),
        },
        "notes": [
            "Rollouts use game_runner_diff.run_rhc_with_rl_and_collect_frames_3d_diff.",
            "If num_attackers>1 we run separate episodes def vs att_j and aggregate by --multi_att_mode.",
            "PASS criterion defaults to 'attacker does not hit target and no terminations/keepout violations'.",
            "Set cfg['verify_require_capture']=True to require collision/capture as well.",
            "grid_mode=paired uses --trials_in rows or --auto_shell_grid pairing.",
            "grid_mode=cartesian uses product of --def_trials_in x --att_trials_in or --auto_shell_grid grids.",
            "If --max_pairs is set in cartesian mode, pairs are sampled (not guaranteed unique).",
        ],
        "metrics_att1": {
            "min_rel_dist": {"mean": float(np.nanmean(_col("att1_min_rel_dist"))) if _col("att1_min_rel_dist").size else float("nan"),
                             **_quantiles(_col("att1_min_rel_dist"))},
            "min_att_center": {"mean": float(np.nanmean(_col("att1_min_att_center"))) if _col("att1_min_att_center").size else float("nan"),
                               **_quantiles(_col("att1_min_att_center"))},
            "attacker_hit_rate": float(np.mean(_col("att1_attacker_hit"))) if _col("att1_attacker_hit").size else float("nan"),
            "collision_rate": float(np.mean(_col("att1_collided"))) if _col("att1_collided").size else float("nan"),
            "att_term_rate": float(np.mean(_col("att1_att_term"))) if _col("att1_att_term").size else float("nan"),
            "def_term_rate": float(np.mean(_col("att1_def_term"))) if _col("att1_def_term").size else float("nan"),
            "oi_viol_def_rate": float(np.mean(_col("att1_oi_viol_def"))) if _col("att1_oi_viol_def").size else float("nan"),
            "oi_viol_att_rate": float(np.mean(_col("att1_oi_viol_att"))) if _col("att1_oi_viol_att").size else float("nan"),
            "udef_mean": {"mean": float(np.nanmean(_col("att1_udef_mean"))) if _col("att1_udef_mean").size else float("nan"),
                          **_quantiles(_col("att1_udef_mean"))},
            "uatt_mean": {"mean": float(np.nanmean(_col("att1_uatt_mean"))) if _col("att1_uatt_mean").size else float("nan"),
                          **_quantiles(_col("att1_uatt_mean"))},
        }
    }

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    csv_path = out_dir / "trials.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = sorted({k_ for r in trial_rows for k_ in r.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in trial_rows:
            w.writerow(r)

    starts_def_arr = np.asarray(starts_def, dtype=float)
    starts_att_arrs = [
        np.asarray(sa, dtype=float) if len(sa) else np.zeros((0, D))
        for sa in starts_atts
    ]

    center, arena_r = _get_center_and_radius(cfg0, D)

    _save_start_plots(
        out_dir=out_dir,
        cfg=cfg0,                 # <-- gives plot fn access to oi + arena
        D=D,
        starts_def=starts_def_arr,
        starts_atts=starts_att_arrs,
        trial_rows=trial_rows,
        cmap="coolwarm",          # or whatever you want
    )

    log(f"[eval] DONE | trials={n} passes={k} success_rate={sr:.3f} CI={lo:.3f}..{hi:.3f}")
    log(f"[eval] wrote: {out_dir/'results.json'}")
    log(f"[eval] wrote: {out_dir/'trials.csv'}")
    log(f"[eval] wrote: {out_dir/'starts_xy.png'}")
    log(f"[eval] wrote: {out_dir/'starts_xz.png'}")
    log(f"[eval] total_time={time.time() - t_global0:.3f}s")


if __name__ == "__main__":
    main()

"""

python evaluate_policy.py \
  --def_ckpt_path Training_Policy/def1_def_teacher.pt \
  --att_ckpt_path Training_Policy/att1_att_teacher.pt \
  --out_dir Training_Policy/MC_eval/def1_vs_att1

"""

"""

python evaluate_policy.py \
  --def_ckpt_path Training_Policy_New_Reward/def1_teacher.pt \
  --att_ckpt_path Training_Policy_New_Reward/att1_teacher.pt \
  --out_dir Training_Policy_New_Reward/MC_eval/def1_vs_att1 \
  --trials_in shelled_trials.csv
  

"""

"""
python evaluate_policy.py \
  --def_ckpt_path Training_Policy_New_Reward/def1_teacher.pt \
  --att_ckpt_path Training_Policy_New_Reward/att1_teacher.pt \
  --out_dir Training_Policy_New_Reward/MC_eval/cart_def1_vs_att1 \
  --grid_mode cartesian \
  --def_trials_in shelled_trials.csv \
  --att_trials_in shelled_trials.csv
  --max_pairs 5000
"""


"""
python evaluate_policy.py \
  --def_ckpt_path Training_Policy/def1_teacher.pt \
  --att_ckpt_path Training_Policy/att1_teacher.pt \
  --out_dir Training_Policy/MC_eval/two_body_shell_cart \
  --auto_shell_grid \
  --grid_mode cartesian \
  --shell_fracs 0.25,0.5,0.75,0.9 \
  --points_per_shell 10 \
  --dynamics elliptic_ltv
"""


"""
python evaluate_policy.py   --def_ckpt_path Training_Policy_New_Reward/def1_teacher.pt   --att_ckpt_path Training_Policy_New_Reward/att1_teacher.pt   --out_dir Training_Policy_New_Reward/MC_eval/shell_cart_allpairs_def1_vs_att1_tester   --auto_shell_grid   --grid_mode cartesian   --shell_fracs 0.25,0.5,0.75,0.90   --points_per_shell 10

"""