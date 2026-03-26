#!/usr/bin/env python3
"""
evaluate_policy.py

Statistical verification harness for your Diff-Nash RL rollout runner:

  - game_runner_diff.run_rhc_with_rl_and_collect_frames_3d(cfg, steps=...)

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
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

# IMPORTANT: must match your CURRENT runner
from game_runner import run_rhc_with_rl_and_collect_frames_3d
from matchup_runner import (
    SUPPORTED_BASELINE_OPPONENTS,
    run_rhc_with_policy_vs_baseline_collect_frames_3d,
)
from dispersion import build_episode_cfg_and_x0


# ------------------------- local defaults -------------------------

# Edit these when you want persistent evaluate_policy preferences without
# repeating them on the bash command line. CLI flags still override any entry.
EVALUATE_POLICY_DEFAULTS: Dict[str, Any] = {
    # Eval-harness defaults: these control evaluation flow/output and do not
    # overwrite fields loaded from run_manifest.json.
    "out_dir": "eval_out",
    "num_trials": 200,
    "seed": 0,
    "policy_role": "def",          # "def" or "att"
    "opponent_source": "policy",   # "policy" | "paper" | "game_theory" | "ipopt"
    "sample_ic": False,
    "pos_scale": 0.95,
    "vel_scale": 0.0,
    "min_sep": 0.0,
    "multi_att_mode": "worst",
    "alpha": 0.05,
    "log_every": 10,
    "print_first_out_keys": False,
    "print_errors": False,
    "trace_errors": False,
    "trials_in": None,
    "grid_mode": "paired",
    "def_trials_in": None,
    "att_trials_in": None,
    "max_pairs": None,
    "auto_shell_grid": False,
    "shell_fracs": "0.2,0.4,0.6,0.8",
    "points_per_shell": 40,
    "include_center": False,

    # Intentional config overrides: these can overwrite fields loaded from
    # run_manifest.json when set to a non-None or active value.
    "steps": None,
    "device": None,
    "def_ckpt_path": None,
    "att_ckpt_path": None,
    "deterministic": False,
    "use_ukf": False,
    "x0_pos_jitter": None,
    "x0_vel_jitter": None,
    "dynamics": None,
    "dt": None,
    "collision_radius_m": None,
}


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



def _require_existing_file(path_str: Optional[str], label: str) -> None:
    if path_str is None:
        raise RuntimeError(f"Missing required {label} path.")
    p = Path(path_str).expanduser()
    if not p.is_file():
        raise RuntimeError(f"Required {label} file does not exist: {p}")



def _validate_eval_inputs(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    if args.trials_in is not None:
        _require_existing_file(args.trials_in, "trials_in CSV")
    if args.def_trials_in is not None:
        _require_existing_file(args.def_trials_in, "def_trials_in CSV")
    if args.att_trials_in is not None:
        _require_existing_file(args.att_trials_in, "att_trials_in CSV")

    if args.opponent_source == "policy":
        _require_existing_file(cfg.get("def_ckpt_path"), "defender checkpoint")
        _require_existing_file(cfg.get("att_ckpt_path"), "attacker checkpoint")
    elif args.policy_role == "def":
        _require_existing_file(cfg.get("def_ckpt_path"), "defender checkpoint")
    else:
        _require_existing_file(cfg.get("att_ckpt_path"), "attacker checkpoint")


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


# ------------------------- row helpers -------------------------

def _has_entity_xyz(row: Dict[str, float], prefix: str, D: int) -> bool:
    axes = ["x", "y", "z"][:D]
    return all(f"{prefix}_{ax}" in row for ax in axes)


def _entity_state_from_row(row: Dict[str, float], prefix: str, D: int) -> np.ndarray:
    axes = ["x", "y", "z"][:D]
    missing = [f"{prefix}_{ax}" for ax in axes if f"{prefix}_{ax}" not in row]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(f"Row is missing required columns for {prefix}: {missing_str}")

    vals = [float(row[f"{prefix}_{ax}"]) for ax in axes]
    vals.extend(float(row.get(f"{prefix}_v{ax}", 0.0)) for ax in axes)
    return np.asarray(vals, dtype=float)


def _row_num_attackers(row: Dict[str, float], D: int) -> int:
    n = 0
    while _has_entity_xyz(row, f"att{n + 1}", D):
        n += 1
    return n


def _copy_attacker_state(row: Dict[str, float], src_idx: int, dst_idx: int, D: int) -> Dict[str, float]:
    axes = ["x", "y", "z"][:D]
    src = f"att{src_idx}"
    dst = f"att{dst_idx}"
    if not _has_entity_xyz(row, src, D):
        raise RuntimeError(f"Row is missing required attacker columns for {src}.")

    out: Dict[str, float] = {}
    for ax in axes:
        out[f"{dst}_{ax}"] = float(row[f"{src}_{ax}"])
        out[f"{dst}_v{ax}"] = float(row.get(f"{src}_v{ax}", 0.0))
    return out


def _attacker_states_from_row(row: Dict[str, float], D: int, num_attackers: int) -> List[np.ndarray]:
    available = _row_num_attackers(row, D)
    if available < num_attackers:
        raise RuntimeError(
            f"Expected attacker columns att1..att{num_attackers} but row only provides att1..att{available}."
        )
    return [_entity_state_from_row(row, f"att{j + 1}", D) for j in range(num_attackers)]


def _expand_attacker_team_rows(
    att_rows: List[Dict[str, float]],
    D: int,
    num_attackers: int,
) -> List[Dict[str, float]]:
    if num_attackers <= 1 or not att_rows:
        return att_rows

    counts = [_row_num_attackers(row, D) for row in att_rows]
    min_count = min(counts)
    max_count = max(counts)

    if min_count >= num_attackers:
        return att_rows
    if max_count >= num_attackers and min_count < num_attackers:
        raise RuntimeError(
            f"attacker rows are inconsistent: some rows provide fewer than {num_attackers} attackers."
        )
    if max_count != 1:
        raise RuntimeError(
            "attacker rows must provide either a full attacker team (att1_*, att2_*, ...) "
            "or exactly one attacker state per row."
        )

    n = len(att_rows)
    stride = max(1, n // num_attackers)
    expanded: List[Dict[str, float]] = []
    for i in range(n):
        team_row: Dict[str, float] = {}
        used: set[int] = set()
        for dst_idx in range(1, num_attackers + 1):
            src_idx = (i + (dst_idx - 1) * stride) % n
            if n >= num_attackers:
                for _ in range(n):
                    if src_idx not in used:
                        break
                    src_idx = (src_idx + 1) % n
            used.add(src_idx)
            team_row.update(_copy_attacker_state(att_rows[src_idx], 1, dst_idx, D))
        expanded.append(team_row)
    return expanded


def _build_paired_x0(row: Dict[str, float], D: int, num_attackers: int) -> np.ndarray:
    xs = [_entity_state_from_row(row, "def", D)]
    xs.extend(_attacker_states_from_row(row, D, num_attackers))
    return np.stack(xs, axis=0)


def _build_cartesian_x0(
    def_row: Dict[str, float],
    att_row: Dict[str, float],
    D: int,
    num_attackers: int,
) -> np.ndarray:
    xs = [_entity_state_from_row(def_row, "def", D)]
    xs.extend(_attacker_states_from_row(att_row, D, num_attackers))
    return np.stack(xs, axis=0)


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


def _cli_option_strings(argv: List[str]) -> set[str]:
    present: set[str] = set()
    for tok in argv[1:]:
        if tok == "--":
            break
        if tok.startswith("--"):
            present.add(tok.split("=", 1)[0])
    return present


def _apply_parser_defaults_from_cfg(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    defaults: Dict[str, Any],
    argv: List[str],
) -> List[str]:
    if not defaults:
        return []

    cli_present = _cli_option_strings(argv)
    applied: List[str] = []
    for action in parser._actions:
        if action.dest == "help" or not action.option_strings:
            continue
        if action.dest not in defaults:
            continue
        if any(opt in cli_present for opt in action.option_strings):
            continue
        setattr(args, action.dest, defaults[action.dest])
        applied.append(action.dest)
    return applied


# ------------------------- config helpers -------------------------

def _load_json_dict(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object in {path}, got {type(data).__name__}.")
    return data


def _resolve_run_manifest_path(
    run_manifest: Optional[str],
    run_dir: Optional[str],
    def_ckpt_path: Optional[str],
    att_ckpt_path: Optional[str],
) -> Optional[Path]:
    if run_manifest is not None:
        path = Path(run_manifest).expanduser()
        if not path.is_file():
            raise RuntimeError(f"--run_manifest file does not exist: {path}")
        return path

    if run_dir is not None:
        path = Path(run_dir).expanduser() / "run_manifest.json"
        if not path.is_file():
            raise RuntimeError(f"--run_dir does not contain run_manifest.json: {path}")
        return path

    candidates: List[Path] = []
    for ckpt_path in (def_ckpt_path, att_ckpt_path):
        if ckpt_path is None:
            continue
        path = Path(ckpt_path).expanduser().parent / "run_manifest.json"
        if path.is_file():
            candidates.append(path.resolve())

    unique = sorted({p for p in candidates})
    if not unique:
        return None
    if len(unique) > 1:
        joined = ", ".join(str(p) for p in unique)
        raise RuntimeError(
            "Found multiple candidate run manifests from checkpoint paths. "
            f"Please pass --run_manifest or --run_dir explicitly. Candidates: {joined}"
        )
    return unique[0]


def _extract_manifest_training_cfg(manifest: Dict[str, Any], manifest_path: Path) -> Dict[str, Any]:
    configs = manifest.get("configs") or {}
    if not isinstance(configs, dict):
        raise RuntimeError(f"Manifest {manifest_path} is missing a dict-valued 'configs' section.")

    cfg = configs.get("training config used")
    if isinstance(cfg, dict):
        return copy.deepcopy(cfg)

    dict_cfgs = [(k, v) for k, v in configs.items() if isinstance(v, dict)]
    if len(dict_cfgs) == 1:
        return copy.deepcopy(dict_cfgs[0][1])

    keys = ", ".join(sorted(str(k) for k in configs.keys()))
    raise RuntimeError(
        f"Could not find 'configs[\"training config used\"]' in {manifest_path}. "
        f"Available config keys: {keys}"
    )


def _load_base_cfg(
    args: argparse.Namespace,
    mod: Any,
) -> Tuple[Dict[str, Any], str, Optional[Path]]:
    manifest_path = _resolve_run_manifest_path(
        run_manifest=args.run_manifest,
        run_dir=args.run_dir,
        def_ckpt_path=args.def_ckpt_path,
        att_ckpt_path=args.att_ckpt_path,
    )
    if manifest_path is None:
        raise RuntimeError(
            "Could not find run_manifest.json. Pass --run_manifest, --run_dir, or use checkpoint paths "
            "whose parent directory contains run_manifest.json."
        )

    manifest = _load_json_dict(manifest_path)
    cfg = _extract_manifest_training_cfg(manifest, manifest_path)
    return cfg, f"run_manifest:{manifest_path}", manifest_path


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



def _make_paired_rows_from_def_att(
    def_rows: List[Dict[str, float]],
    att_rows: List[Dict[str, float]],
    n: int,
    D: int,
    num_attackers: int,
) -> List[Dict[str, float]]:
    n_pair = min(len(def_rows), len(att_rows), int(n))
    axes = ["x", "y", "z"][:D]
    out: List[Dict[str, float]] = []
    for i in range(n_pair):
        rd, ra = def_rows[i], att_rows[i]
        row = {}
        for ax in axes:
            row[f"def_{ax}"] = float(rd[f"def_{ax}"])
            row[f"def_v{ax}"] = float(rd.get(f"def_v{ax}", 0.0))
        for j in range(num_attackers):
            row.update(_copy_attacker_state(ra, j + 1, j + 1, D))
        out.append(row)
    return out


def _radial_advantage_margin(cfg: Dict[str, Any]) -> float:
    oi_r = float((cfg.get("oi", {}) or {}).get("r", 0.0))
    percent_advantage_defender = float(cfg.get("percent_advantage_defender", 0.75))
    radial_margin = float(percent_advantage_defender * np.pi * 2.0 * oi_r)
    if radial_margin < 0.0:
        raise ValueError(f"percent_advantage_defender must be >= 0, got {percent_advantage_defender}")
    return radial_margin


def _valid_def_att_pair_indices(
    cfg: Dict[str, Any],
    def_rows: List[Dict[str, float]],
    att_rows: List[Dict[str, float]],
    num_attackers: int,
    require_training_advantage: bool = False,
) -> List[Tuple[int, int]]:
    D = int(cfg.get("D", 3))
    center, _arena_r = _get_center_and_radius(cfg, D)
    radial_margin = _radial_advantage_margin(cfg)
    min_sep = float(cfg.get("train_min_sep", 0.0)) if require_training_advantage else 0.0

    out: List[Tuple[int, int]] = []
    for di, rd in enumerate(def_rows):
        p_def = _entity_state_from_row(rd, "def", D)[:D]
        r_def = float(np.linalg.norm(p_def - center))

        for ai, ra in enumerate(att_rows):
            att_states = _attacker_states_from_row(ra, D, num_attackers)
            p_atts = [x[:D] for x in att_states]
            r_atts = [float(np.linalg.norm(p_att - center)) for p_att in p_atts]

            if require_training_advantage and any(r_def > (r_att - radial_margin) for r_att in r_atts):
                continue
            if min_sep > 0.0 and any(np.linalg.norm(p_def - p_att) < min_sep for p_att in p_atts):
                continue
            if min_sep > 0.0 and len(p_atts) > 1:
                attackers_ok = True
                for j in range(len(p_atts)):
                    for k in range(j + 1, len(p_atts)):
                        if np.linalg.norm(p_atts[j] - p_atts[k]) < min_sep:
                            attackers_ok = False
                            break
                    if not attackers_ok:
                        break
                if not attackers_ok:
                    continue

            out.append((di, ai))

    return out


def _paired_rows_from_pair_indices(
    def_rows: List[Dict[str, float]],
    att_rows: List[Dict[str, float]],
    pair_indices: List[Tuple[int, int]],
    n: int,
    D: int,
    num_attackers: int,
) -> List[Dict[str, float]]:
    axes = ["x", "y", "z"][:D]
    out: List[Dict[str, float]] = []
    for di, ai in pair_indices[:max(0, int(n))]:
        rd, ra = def_rows[di], att_rows[ai]
        row = {}
        for ax in axes:
            row[f"def_{ax}"] = float(rd[f"def_{ax}"])
            row[f"def_v{ax}"] = float(rd.get(f"def_v{ax}", 0.0))
        for j in range(num_attackers):
            row.update(_copy_attacker_state(ra, j + 1, j + 1, D))
        out.append(row)
    return out


# ------------------------- scenario generation (random sampling) -------------------------

def _sample_uniform_ball(rng: np.random.Generator, center: np.ndarray, radius: float) -> np.ndarray:
    D = center.size
    v = rng.normal(size=D)
    v = v / (np.linalg.norm(v) + 1e-12)
    u = rng.random()
    rad = radius * (u ** (1.0 / D))
    return center + rad * v


def _sample_in_shell(
    rng: np.random.Generator,
    center: np.ndarray,
    r_min: float,
    r_max: float,
) -> np.ndarray:
    if r_max < r_min:
        raise ValueError(f"Invalid shell: r_min={r_min} > r_max={r_max}")

    D = center.size
    d = rng.normal(size=D)
    d /= (np.linalg.norm(d) + 1e-9)

    u = rng.random()
    r = (r_min**D + (r_max**D - r_min**D) * u) ** (1.0 / D)
    return center + r * d


def _sample_x0_random_shell_advantage(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    num_attackers: int,
    vel_scale_override: Optional[float] = None,
    min_sep_override: Optional[float] = None,
) -> np.ndarray:
    """
    Mirror core/env.py random_shell_advantage so evaluation can sample the
    same defender-favored radial geometry used during training.
    """
    D = int(cfg.get("D", 3))
    center, R = _get_center_and_radius(cfg, D)
    nx = 2 * D

    v_max = float(cfg.get("train_ic_vmax", 0.0))
    if vel_scale_override is not None:
        v_max = float(vel_scale_override)

    min_sep = float(cfg.get("train_min_sep", 0.0))
    if min_sep_override is not None:
        min_sep = float(min_sep_override)

    oi_r = float((cfg.get("oi", {}) or {}).get("r", 0.0))
    percent_advantage_defender = float(cfg.get("percent_advantage_defender", 0.75))
    radial_margin = float(percent_advantage_defender * np.pi * 2.0 * oi_r)

    r_def_min = float(cfg.get("r_def_min", 0.0)) * R
    r_def_max = float(cfg.get("r_def_max", 1.0)) * R
    r_att_min = float(cfg.get("r_att_min", 0.0)) * R
    r_att_max = float(cfg.get("r_att_max", 1.0)) * R
    r_att_min = max(r_att_min, radial_margin)

    if radial_margin < 0.0:
        raise ValueError(f"percent_advantage_defender must be >= 0, got {radial_margin}")
    if r_def_min > r_def_max:
        raise ValueError(f"Invalid defender shell: [{r_def_min}, {r_def_max}]")
    if r_att_min > r_att_max:
        raise ValueError(f"Invalid attacker shell: [{r_att_min}, {r_att_max}]")
    if num_attackers > 0 and r_def_min > (r_att_max - radial_margin):
        raise ValueError(
            "Infeasible radial shells: defender cannot be at least "
            f"{radial_margin:.3f} m closer to center than attacker. "
            f"Got r_def_min={r_def_min:.3f}, r_att_max={r_att_max:.3f}."
        )

    placed = False
    p_def = np.zeros(D, dtype=float)
    v_def = np.zeros(D, dtype=float)
    p_atts: List[np.ndarray] = []
    v_atts: List[np.ndarray] = []

    for _scene_try in range(2000):
        p_atts = []
        v_atts = []
        r_atts: List[float] = []

        attackers_ok = True
        for _ in range(num_attackers):
            found_att = False
            for _att_try in range(1000):
                p_att = _sample_in_shell(rng, center, r_att_min, r_att_max)
                r_att = float(np.linalg.norm(p_att - center))

                if any(np.linalg.norm(p_att - p_prev) < min_sep for p_prev in p_atts):
                    continue

                p_atts.append(p_att)
                r_atts.append(r_att)
                v_atts.append(rng.uniform(-v_max, v_max, size=D))
                found_att = True
                break

            if not found_att:
                attackers_ok = False
                break

        if not attackers_ok:
            continue

        if num_attackers <= 0:
            p_def = _sample_in_shell(rng, center, r_def_min, r_def_max)
            v_def = rng.uniform(-v_max, v_max, size=D)
            placed = True
            break

        r_att_nearest = min(r_atts)
        r_def_max_eff = min(r_def_max, r_att_nearest - radial_margin)
        if r_def_max_eff < r_def_min:
            continue

        found_def = False
        for _def_try in range(1000):
            cand = _sample_in_shell(rng, center, r_def_min, r_def_max_eff)
            if any(np.linalg.norm(cand - p_att) < min_sep for p_att in p_atts):
                continue
            p_def = cand
            found_def = True
            break

        if not found_def:
            continue

        v_def = rng.uniform(-v_max, v_max, size=D)
        placed = True
        break

    if not placed:
        raise RuntimeError(
            "random_shell_advantage: could not sample a feasible initial condition after many attempts. "
            "Try relaxing r_def_max, increasing r_att_min/r_att_max, reducing train_min_sep, "
            "or reducing percent_advantage_defender."
        )

    xs = [np.concatenate([p_def, v_def], dtype=float)[:nx]]
    for p_att, v_att in zip(p_atts, v_atts):
        xs.append(np.concatenate([p_att, v_att], dtype=float)[:nx])
    return np.asarray(xs, dtype=float)


def _sample_x0(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    num_attackers: int,
    pos_scale: float,
    vel_scale: float,
    min_sep: float = 0.0,
    cli_present: Optional[set[str]] = None,
) -> np.ndarray:
    """
    Returns x0 with shape (1 + num_attackers, 2D).
    Enforces a minimum defender-attacker separation if min_sep > 0.
    """
    train_ic_mode = str(cfg.get("train_ic_mode", "fixed"))
    if train_ic_mode == "random_shell_advantage":
        vel_scale_override = float(vel_scale) if cli_present and "--vel_scale" in cli_present else None
        min_sep_override = float(min_sep) if cli_present and "--min_sep" in cli_present else None
        return _sample_x0_random_shell_advantage(
            cfg,
            rng,
            num_attackers=num_attackers,
            vel_scale_override=vel_scale_override,
            min_sep_override=min_sep_override,
        )

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


def _classify_trial_outcome(row: Dict[str, Any]) -> str:
    if int(row.get("num_att_errors", 0)) > 0 or int(row.get("trial_rollout_error_any", 0)) == 1 or row.get("att1_error"):
        return "rollout_error"

    pass_trial = int(row.get("pass_trial", 0))
    collided = int(row.get("trial_collided_any", row.get("att1_collided", 0))) == 1
    attacker_hit = int(row.get("trial_attacker_hit_any", row.get("att1_attacker_hit", 0))) == 1
    att_term = int(row.get("trial_att_term_any", row.get("att1_att_term", 0))) == 1
    def_term = int(row.get("trial_def_term_any", row.get("att1_def_term", 0))) == 1
    oi_viol_att = int(row.get("trial_oi_viol_att_any", row.get("att1_oi_viol_att", 0))) == 1
    oi_viol_def = int(row.get("trial_oi_viol_def_any", row.get("att1_oi_viol_def", 0))) == 1
    att_oob = int(row.get("trial_att_oob_any", row.get("att1_att_oob", 0))) == 1
    def_oob = int(row.get("trial_def_oob_any", row.get("att1_def_oob", 0))) == 1
    success_mode = str(row.get("trial_success_mode", row.get("att1_success_mode", ""))).strip().lower()

    if pass_trial:
        if collided:
            return "defender_capture"
        return "defender_success"

    if attacker_hit or oi_viol_att:
        return "attacker_hit_oi"
    if def_term or def_oob:
        return "defender_crashed_wall"
    if att_term or att_oob:
        return "attacker_crashed_wall"
    if oi_viol_def:
        return "defender_hit_oi"
    if collided:
        return "collision_but_not_success"
    if success_mode == "zero_sum_capture":
        return "timeout_no_capture"
    if success_mode == "legacy_verify":
        return "capture_required_not_met"
    return "unclassified_failure"


def _save_outcome_histogram(
    out_dir: Path,
    trial_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    labels = [
        "defender_capture",
        "defender_success",
        "attacker_hit_oi",
        "defender_crashed_wall",
        "attacker_crashed_wall",
        "defender_hit_oi",
        "collision_but_not_success",
        "timeout_no_capture",
        "capture_required_not_met",
        "rollout_error",
        "unclassified_failure",
    ]
    pretty = {
        "defender_capture": "Defender capture",
        "defender_success": "Defender success",
        "attacker_hit_oi": "Attacker hit OI",
        "defender_crashed_wall": "Defender crashed wall",
        "attacker_crashed_wall": "Attacker crashed wall",
        "defender_hit_oi": "Defender hit OI",
        "collision_but_not_success": "Collision but not success",
        "timeout_no_capture": "Timeout / no capture",
        "capture_required_not_met": "Capture required not met",
        "rollout_error": "Rollout error",
        "unclassified_failure": "Unclassified failure",
    }

    counts = {k: 0 for k in labels}
    for row in trial_rows:
        counts[_classify_trial_outcome(row)] += 1

    n = max(1, len(trial_rows))
    present = [k for k in labels if counts[k] > 0]
    if not present:
        present = labels

    values = [counts[k] / float(n) for k in present]
    fig_h = max(4.0, 0.55 * len(present) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    bars = ax.barh(range(len(present)), values, color="steelblue", edgecolor="black")
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels([pretty[k] for k in present])
    ax.set_xlim(0.0, max(values) * 1.15 if values else 1.0)
    ax.set_xlabel("Proportion of trials (-)")
    ax.set_title("Trial Outcome Breakdown")
    ax.grid(True, axis="x", alpha=0.3)

    for idx, (bar, key) in enumerate(zip(bars, present)):
        v = values[idx]
        ax.text(v + 0.01 * max(1.0, max(values)), bar.get_y() + bar.get_height() / 2.0,
                f"{counts[key]}/{n} ({v:.1%})", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "outcome_hist.png", dpi=180)
    plt.close(fig)

    return {
        "counts": counts,
        "proportions": {k: counts[k] / float(n) for k in labels},
    }


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
    att_term_idx = np.where(att_center > arena_r + term_margin)[0]
    def_term_idx = np.where(def_center > arena_r + term_margin)[0]
    att_term = bool(att_term_idx.size > 0)
    def_term = bool(def_term_idx.size > 0)
    t_att_term = int(att_term_idx[0]) if att_term else -1
    t_def_term = int(def_term_idx[0]) if def_term else -1

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
    t_oi_viol_def = -1
    t_oi_viol_att = -1
    if oi_enabled and oi_safe_r > 0:
        if 0 in avoid_by:
            def_oi_idx = np.where(def_center <= oi_safe_r)[0]
            oi_viol_def = bool(def_oi_idx.size > 0)
            t_oi_viol_def = int(def_oi_idx[0]) if oi_viol_def else -1
        if 1 in avoid_by:
            att_oi_idx = np.where(att_center <= oi_safe_r)[0]
            oi_viol_att = bool(att_oi_idx.size > 0)
            t_oi_viol_att = int(att_oi_idx[0]) if oi_viol_att else -1

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
    zero_sum_cfg = cfg.get("zero_sum_reward", {}) or {}
    zero_sum_mode = str(zero_sum_cfg.get("mode", "")).strip().lower()
    zero_sum_eval = zero_sum_mode not in ("", "none") and collision_r > 0.0

    if zero_sum_eval:
        fail_times = [
            t for t in (t_att_hit, t_att_term, t_def_term, t_oi_viol_def, t_oi_viol_att)
            if t >= 0
        ]
        first_fail_t = min(fail_times) if fail_times else math.inf
        pass_flag = collided and (t_collide <= first_fail_t)
        success_mode = "zero_sum_capture"
    else:
        pass_flag = (
            (not attacker_hit)
            and (not att_term) and (not def_term)
            and (not oi_viol_def) and (not oi_viol_att)
        )
        if verify_require_capture and collision_r > 0:
            pass_flag = pass_flag and collided
        success_mode = "legacy_verify"

    return {
        "pass": int(pass_flag),
        "success_mode": success_mode,

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
        "t_att_term": t_att_term,
        "t_def_term": t_def_term,

        "oi_viol_def": int(oi_viol_def),
        "oi_viol_att": int(oi_viol_att),
        "t_oi_viol_def": t_oi_viol_def,
        "t_oi_viol_att": t_oi_viol_att,

        "udef_mean": udef_mean,
        "uatt_mean": uatt_mean,
        "udef_max": udef_max,
        "uatt_max": uatt_max,
    }


# ------------------------- plotting -------------------------

def _extract_single_attacker_rollout(out: Dict[str, Any], attacker_idx: int) -> Dict[str, Any]:
    exec_xyz_all = out.get("exec_xyz_all", None)
    u_cmd_norm_all = out.get("u_cmd_norm_all", None)

    if exec_xyz_all is not None:
        if (1 + attacker_idx) >= len(exec_xyz_all):
            raise IndexError(f"Missing attacker index {attacker_idx} in rollout with {len(exec_xyz_all) - 1} attackers.")
        single = {
            "exec1_xyz": exec_xyz_all[0],
            "exec2_xyz": exec_xyz_all[1 + attacker_idx],
            "done_info": out.get("done_info", None),
        }
        if isinstance(u_cmd_norm_all, list) and (1 + attacker_idx) < len(u_cmd_norm_all):
            single["u_cmd_norm_all"] = [u_cmd_norm_all[0], u_cmd_norm_all[1 + attacker_idx]]
        elif u_cmd_norm_all is not None:
            single["u_cmd_norm_all"] = u_cmd_norm_all
        return single

    single = {
        "exec1_xyz": out.get("exec1_xyz", None),
        "exec2_xyz": out.get("exec2_xyz", None),
        "done_info": out.get("done_info", None),
    }
    if u_cmd_norm_all is not None:
        single["u_cmd_norm_all"] = u_cmd_norm_all
    return single


def _rollout_num_steps(out: Dict[str, Any]) -> int:
    p1 = np.asarray(out.get("exec1_xyz", []), dtype=float)
    if p1.ndim == 0 or p1.size == 0:
        return 0
    return max(0, int(len(p1)) - 1)


def _min_non_negative(vals: List[Any]) -> int:
    finite = []
    for v in vals:
        try:
            iv = int(v)
        except Exception:
            continue
        if iv >= 0:
            finite.append(iv)
    return min(finite) if finite else -1


def _mean_metric(per_att_metrics: List[Dict[str, Any]], key: str) -> float:
    vals = []
    for m in per_att_metrics:
        try:
            v = float(m.get(key, float("nan")))
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def _aggregate_trial_metrics(per_att_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not per_att_metrics:
        return {
            "trial_rollout_error_any": 1,
            "trial_success_mode": "",
        }

    success_modes = [str(m.get("success_mode", "")).strip().lower() for m in per_att_metrics if str(m.get("success_mode", "")).strip()]

    return {
        "trial_rollout_error_any": int(any(bool(m.get("error")) for m in per_att_metrics)),
        "trial_success_mode": success_modes[0] if success_modes else "",
        "trial_min_rel_dist": min(float(m.get("min_rel_dist", float("inf"))) for m in per_att_metrics),
        "trial_min_att_center": min(float(m.get("min_att_center", float("inf"))) for m in per_att_metrics),
        "trial_collided_any": int(any(int(m.get("collided", 0)) == 1 for m in per_att_metrics)),
        "trial_attacker_hit_any": int(any(int(m.get("attacker_hit", 0)) == 1 for m in per_att_metrics)),
        "trial_att_oob_any": int(any(int(m.get("att_oob", 0)) == 1 for m in per_att_metrics)),
        "trial_def_oob_any": int(any(int(m.get("def_oob", 0)) == 1 for m in per_att_metrics)),
        "trial_att_term_any": int(any(int(m.get("att_term", 0)) == 1 for m in per_att_metrics)),
        "trial_def_term_any": int(any(int(m.get("def_term", 0)) == 1 for m in per_att_metrics)),
        "trial_oi_viol_def_any": int(any(int(m.get("oi_viol_def", 0)) == 1 for m in per_att_metrics)),
        "trial_oi_viol_att_any": int(any(int(m.get("oi_viol_att", 0)) == 1 for m in per_att_metrics)),
        "trial_t_collide": _min_non_negative([m.get("t_collide", -1) for m in per_att_metrics]),
        "trial_t_att_hit": _min_non_negative([m.get("t_att_hit", -1) for m in per_att_metrics]),
        "trial_t_att_term": _min_non_negative([m.get("t_att_term", -1) for m in per_att_metrics]),
        "trial_t_def_term": _min_non_negative([m.get("t_def_term", -1) for m in per_att_metrics]),
        "trial_t_oi_viol_def": _min_non_negative([m.get("t_oi_viol_def", -1) for m in per_att_metrics]),
        "trial_t_oi_viol_att": _min_non_negative([m.get("t_oi_viol_att", -1) for m in per_att_metrics]),
        "trial_udef_mean": _mean_metric(per_att_metrics, "udef_mean"),
        "trial_uatt_mean": _mean_metric(per_att_metrics, "uatt_mean"),
        "trial_rollout_num_steps": _mean_metric(per_att_metrics, "rollout_num_steps"),
        "trial_rollout_setup_sec": _mean_metric(per_att_metrics, "rollout_setup_sec"),
        "trial_rollout_setup_sec_per_step": _mean_metric(per_att_metrics, "rollout_setup_sec_per_step"),
        "trial_rollout_simulation_sec": _mean_metric(per_att_metrics, "rollout_simulation_sec"),
        "trial_rollout_simulation_sec_per_step": _mean_metric(per_att_metrics, "rollout_simulation_sec_per_step"),
        "trial_rollout_total_sec": _mean_metric(per_att_metrics, "rollout_total_sec"),
        "trial_rollout_total_sec_per_step": _mean_metric(per_att_metrics, "rollout_total_sec_per_step"),
    }


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
        ax.set_xlabel(f"{xlabel} (m)")
        ax.set_ylabel(f"{ylabel} (m)")
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

        def_counts.setdefault(kd, [0, 0])
        def_counts[kd][0] += p
        def_counts[kd][1] += 1

        n_att_row = max(1, int(r.get("num_attackers", 1)))
        for j in range(n_att_row):
            ax = r.get(f"att{j+1}_x", float("nan"))
            ay = r.get(f"att{j+1}_y", float("nan"))
            az = r.get(f"att{j+1}_z", float("nan"))
            if any(np.isnan(float(v)) for v in (ax, ay, az)):
                continue
            ka = _key([ax, ay, az])
            att_counts.setdefault(ka, [0, 0])
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
    ap.add_argument("--run_dir", default=None,
                    help="Training run directory containing run_manifest.json. If set, its saved training config becomes the base eval config.")
    ap.add_argument("--run_manifest", default=None,
                    help="Path to a specific run_manifest.json to use as the base eval config.")
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
                    help="If num_attackers>1, aggregate per-attacker metrics from the same joint rollout.")

    # common overrides
    ap.add_argument("--device", default=None)
    ap.add_argument("--def_ckpt_path", default=None)
    ap.add_argument("--att_ckpt_path", default=None)
    ap.add_argument(
        "--policy_role",
        default="def",
        choices=["def", "att"],
        help="When evaluating against a baseline opponent, choose whether the tested policy is the defender or attacker.",
    )
    ap.add_argument(
        "--opponent_source",
        default="policy",
        choices=["policy", *SUPPORTED_BASELINE_OPPONENTS],
        help="Opponent controller used in evaluation. 'policy' preserves the current policy-vs-policy rollout path.",
    )
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
                    help="CSV of attacker start states. Rows may provide att1_* only or a full attacker team att1_*, att2_*, ....")
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
    ap.add_argument(
        "--collision_radius_m",
        type=float,
        default=None,
        help="Optional override for cfg['collision_radius_m'] used by capture evaluation."
    )

    args = ap.parse_args()
    if args.trace_errors:
        args.print_errors = True

    mod = importlib.import_module(args.config_module)
    if not hasattr(mod, "config_for_eval") or not hasattr(mod, "build_dyn"):
        raise RuntimeError(f"Module '{args.config_module}' must define config_for_eval(...) and build_dyn(cfg).")

    cli_present = _cli_option_strings(sys.argv)
    cfg0, base_cfg_source, manifest_path = _load_base_cfg(args, mod)
    applied_defaults = _apply_parser_defaults_from_cfg(args, ap, EVALUATE_POLICY_DEFAULTS, sys.argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_global0 = time.time()
    log(f"[eval] starting; out_dir={out_dir.resolve()}")
    log(f"[eval] config_module={args.config_module}")
    log("[eval] config module imported OK.")
    if applied_defaults:
        log(f"[eval] applied evaluate_policy defaults for: {', '.join(sorted(applied_defaults))}")
    log(f"[eval] base cfg loaded from {base_cfg_source}.")
    if manifest_path is not None:
        log(f"[eval] run manifest={manifest_path}")


    # Dynamics override (must happen before build_dyn)
    if args.dynamics is not None:
        cfg0["dynamics"] = str(args.dynamics)

    if args.dt is not None:
        cfg0["dt"] = float(args.dt)
    if args.collision_radius_m is not None:
        cfg0["collision_radius_m"] = float(args.collision_radius_m)

    mod.build_dyn(cfg0)

    # Apply overrides
    if args.device is not None:
        cfg0["device"] = args.device

    _apply_ckpt_overrides(cfg0, args.def_ckpt_path, args.att_ckpt_path)
    _validate_eval_inputs(args, cfg0)

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
    log(f"[eval] matchup: policy_role={args.policy_role} opponent_source={args.opponent_source}")

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
    valid_pair_indices: Optional[List[Tuple[int, int]]] = None

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
        att_rows = _expand_attacker_team_rows(_rows_from_positions("att1", att_pos), D, num_attackers)
        valid_pair_indices = _valid_def_att_pair_indices(
            cfg0,
            def_rows,
            att_rows,
            num_attackers=num_attackers,
            require_training_advantage=(str(cfg0.get("train_ic_mode", "fixed")) == "random_shell_advantage"),
        )
        total_pairs_all = len(def_rows) * len(att_rows)
        total_pairs_valid = len(valid_pair_indices)
        if total_pairs_valid <= 0:
            raise RuntimeError(
                "auto_shell_grid produced no defender/attacker pairs that satisfy the configured training constraints."
            )

        # Now choose how to pair them
        if args.grid_mode == "paired":
            n_total = min(int(args.num_trials), total_pairs_valid)
            paired_rows = _paired_rows_from_pair_indices(def_rows, att_rows, valid_pair_indices, n_total, D, num_attackers)
            log(f"[eval] auto_shell_grid paired: shells={shell_fracs} points_per_shell={args.points_per_shell} "
                f"include_center={args.include_center} valid_pairs={total_pairs_valid}/{total_pairs_all} -> paired_rows={len(paired_rows)}")
        else:
            if args.max_pairs is None:
                n_total = total_pairs_valid
                log(f"[eval] auto_shell_grid cartesian: evaluating ALL valid pairs = {total_pairs_valid} / {total_pairs_all}")
            else:
                n_total = min(int(args.max_pairs), total_pairs_valid)
                log(f"[eval] auto_shell_grid cartesian: valid pairs={total_pairs_valid} / {total_pairs_all}, evaluating={n_total} (sampled)")
            log(f"[eval] auto_shell_grid cartesian grids: def_rows={len(def_rows)} att_rows={len(att_rows)}")

        # Make CSV args irrelevant
        args.trials_in = None
        args.def_trials_in = None
        args.att_trials_in = None

    # 1) PAIRED CSV
    elif args.grid_mode == "paired" and args.trials_in is not None:
        paired_rows = _load_trials_csv(args.trials_in, D)
        log(f"[eval] loaded paired trials_in: {args.trials_in} ({len(paired_rows)} rows)")
        if num_attackers > 1 and any(_row_num_attackers(r, D) < num_attackers for r in paired_rows):
            raise RuntimeError(
                f"--trials_in rows must include att1..att{num_attackers} columns when cfg['num_attackers']={num_attackers}."
            )
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
        att_rows = _expand_attacker_team_rows(_load_att_csv(args.att_trials_in, D), D, num_attackers)

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
        log(
            f"[eval] random sampling: num_trials={n_total} train_ic_mode={cfg0.get('train_ic_mode', 'fixed')} "
            f"pos_scale={args.pos_scale} vel_scale={args.vel_scale} min_sep={args.min_sep}"
        )

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
            x0 = _build_paired_x0(r, D, num_attackers)

        elif args.grid_mode == "cartesian" and (def_rows is not None and att_rows is not None):
            if valid_pair_indices is not None:
                total_pairs = len(valid_pair_indices)
                if args.max_pairs is None:
                    di, ai = valid_pair_indices[i]
                else:
                    di, ai = valid_pair_indices[int(rng.integers(0, total_pairs))]
            else:
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

            x0 = _build_cartesian_x0(rd, ra, D, num_attackers)

        elif args.sample_ic:
            x0 = _sample_x0(
                cfg_trial, rng,
                num_attackers=num_attackers,
                pos_scale=float(args.pos_scale),
                vel_scale=float(args.vel_scale),
                min_sep=float(args.min_sep),
                cli_present=cli_present,
            )

        else:
            x0 = np.asarray(cfg_trial["x0"], dtype=float)

        expected_rows = 1 + num_attackers
        if x0.ndim != 2 or x0.shape[0] != expected_rows or x0.shape[1] < (2 * D):
            raise RuntimeError(
                f"Expected x0 shape ({expected_rows}, >= {2 * D}) for num_attackers={num_attackers}, got {tuple(x0.shape)}"
            )

        x0 = _apply_x0_jitter(cfg_trial, x0, rng)

        # log starts (for plots)
        starts_def.append(x0[0, :D].copy())
        for j in range(num_attackers):
            if 1 + j < x0.shape[0]:
                starts_atts[j].append(x0[1 + j, :D].copy())



        per_att_metrics = []
        per_att_errors = 0

        cfg_run = copy.deepcopy(cfg_trial)
        cfg_run["x0"] = np.asarray(x0, dtype=float).tolist()

        # force CLI ckpt paths into the rollout cfg
        _apply_ckpt_overrides(cfg_run, args.def_ckpt_path, args.att_ckpt_path)

        # Determine steps for this rollout (never pass None)
        steps_run = int(args.steps) if args.steps is not None else int(cfg_run.get("T", cfg0.get("T", 0)) or 0)
        if steps_run <= 0:
            raise RuntimeError(f"Invalid steps_run={steps_run}. Provide --steps or ensure cfg['T'] is set.")

        # Also keep cfg_run['T'] consistent with steps
        cfg_run["T"] = steps_run

        try:
            if args.opponent_source == "policy":
                out = run_rhc_with_rl_and_collect_frames_3d(cfg_run, steps=steps_run)
            else:
                out = run_rhc_with_policy_vs_baseline_collect_frames_3d(
                    cfg_run,
                    policy_role=args.policy_role,
                    opponent_baseline=args.opponent_source,
                    steps=steps_run,
                )

            if i == 0 and args.print_first_out_keys:
                log("[eval] first rollout returned keys:")
                log("  " + ", ".join(sorted(list(out.keys()))))
                exec_xyz_all = out.get("exec_xyz_all", None)
                if isinstance(exec_xyz_all, list) and exec_xyz_all:
                    lens = [len(np.asarray(p, dtype=float)) for p in exec_xyz_all]
                    log(f"[eval] first rollout sanity: len(exec_xyz_all)={lens}")
                    if len(exec_xyz_all) >= 2:
                        p1 = np.asarray(exec_xyz_all[0], dtype=float)
                        p2 = np.asarray(exec_xyz_all[1], dtype=float)
                        if p1.size and p2.size:
                            d0 = float(np.linalg.norm(p1[0, :D] - p2[0, :D]))
                            dT = float(np.linalg.norm(p1[-1, :D] - p2[-1, :D]))
                            log(f"[eval] first rollout sanity: rel_dist start={d0:.3f} end={dT:.3f}")
                else:
                    p1 = np.asarray(out.get("exec1_xyz", []), dtype=float)
                    p2 = np.asarray(out.get("exec2_xyz", []), dtype=float)
                    log(f"[eval] first rollout sanity: len(exec1_xyz)={len(p1)} len(exec2_xyz)={len(p2)}")
                    if p1.size and p2.size:
                        d0 = float(np.linalg.norm(p1[0, :D] - p2[0, :D]))
                        dT = float(np.linalg.norm(p1[-1, :D] - p2[-1, :D]))
                        log(f"[eval] first rollout sanity: rel_dist start={d0:.3f} end={dT:.3f}")

            rollout_timing = out.get("rollout_timing_sec", {}) or {}
            for j in range(num_attackers):
                try:
                    out_j = _extract_single_attacker_rollout(out, j)
                    m = _compute_trial_metrics(cfg_run, out_j)
                    rollout_steps = _rollout_num_steps(out_j)
                    m["rollout_num_steps"] = int(rollout_steps)
                    for timing_key in ("setup", "simulation", "total"):
                        timing_val = float(rollout_timing.get(timing_key, float("nan")))
                        m[f"rollout_{timing_key}_sec"] = timing_val
                        if rollout_steps > 0 and np.isfinite(timing_val):
                            m[f"rollout_{timing_key}_sec_per_step"] = timing_val / float(rollout_steps)
                        else:
                            m[f"rollout_{timing_key}_sec_per_step"] = float("nan")
                except Exception as e:
                    per_att_errors += 1
                    m = {"pass": 0, "error": str(e)}
                    if args.print_errors:
                        log(f"[eval][trial {i}/{n_total-1}][att {j+1}] ERROR: {e}")
                        if args.trace_errors:
                            log(traceback.format_exc())
                per_att_metrics.append(m)

        except Exception as e:
            per_att_errors = num_attackers
            per_att_metrics = [{"pass": 0, "error": str(e)} for _ in range(num_attackers)]
            if args.print_errors:
                log(f"[eval][trial {i}/{n_total-1}] ERROR: {e}")
                if args.trace_errors:
                    log(traceback.format_exc())

        if not per_att_metrics:
            per_att_metrics = [{"pass": 0, "error": "no attackers present"}]
            per_att_errors += 1

        # Aggregate across attackers
        if args.multi_att_mode == "worst":
            pass_trial = int(all(m.get("pass", 0) == 1 for m in per_att_metrics))
        else:
            pass_trial = int(round(np.mean([m.get("pass", 0) for m in per_att_metrics])))

        passes += pass_trial

        # One row per trial; stores aggregate trial metrics and per-attacker metrics
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
        }
        for j in range(num_attackers):
            row[f"att{j+1}_x"] = float(x0[1 + j, 0]) if (1 + j) < x0.shape[0] else float("nan")
            row[f"att{j+1}_y"] = float(x0[1 + j, 1]) if ((1 + j) < x0.shape[0] and D >= 2) else float("nan")
            row[f"att{j+1}_z"] = float(x0[1 + j, 2]) if ((1 + j) < x0.shape[0] and D == 3) else float("nan")

        if (args.grid_mode == "cartesian") and (def_rows is not None) and (att_rows is not None):
            row["def_idx"] = cart_di
            row["att_idx"] = cart_ai

        row.update(_aggregate_trial_metrics(per_att_metrics))
        for j, metrics in enumerate(per_att_metrics, start=1):
            for k_, v_ in metrics.items():
                row[f"att{j}_{k_}"] = v_

        trial_rows.append(row)

        # Progress print
        if (i == 0) or ((i + 1) % max(1, int(args.log_every)) == 0) or (i == n_total - 1):
            sr_sofar = passes / float(i + 1)
            dt_trial = time.time() - t_trial0

            last_min_rel = row.get("trial_min_rel_dist", row.get("att1_min_rel_dist", None))
            last_hit = row.get("trial_attacker_hit_any", row.get("att1_attacker_hit", None))
            last_def_term = row.get("trial_def_term_any", row.get("att1_def_term", None))
            last_att_term = row.get("trial_att_term_any", row.get("att1_att_term", None))
            last_rollout_sim = row.get("trial_rollout_simulation_sec", row.get("att1_rollout_simulation_sec", None))

            extra = ""
            if args.grid_mode == "cartesian":
                extra = f" | def_idx={row.get('def_idx')} att_idx={row.get('att_idx')}"

            log(
                f"[eval] progress {i+1}/{n_total} | pass_so_far={passes} ({sr_sofar:.3f}) | "
                f"last_pass={pass_trial} | trial_time={dt_trial:.3f}s | "
                f"rollout_sim={last_rollout_sim} | "
                f"min_rel={last_min_rel} hit={last_hit} def_term={last_def_term} att_term={last_att_term} "
                f"errors_this_trial={per_att_errors}{extra}"
            )

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



    outcome_breakdown = _save_outcome_histogram(out_dir, trial_rows)

    def _metric_summary(name: str) -> Dict[str, float]:
        vals = _col(name)
        return {
            "mean": float(np.nanmean(vals)) if vals.size else float("nan"),
            **_quantiles(vals),
        }


    def _attacker_metric_block(prefix: str) -> Dict[str, Any]:
        return {
            "min_rel_dist": _metric_summary(f"{prefix}_min_rel_dist"),
            "min_att_center": _metric_summary(f"{prefix}_min_att_center"),
            "attacker_hit_rate": float(np.mean(_col(f"{prefix}_attacker_hit"))) if _col(f"{prefix}_attacker_hit").size else float("nan"),
            "collision_rate": float(np.mean(_col(f"{prefix}_collided"))) if _col(f"{prefix}_collided").size else float("nan"),
            "att_term_rate": float(np.mean(_col(f"{prefix}_att_term"))) if _col(f"{prefix}_att_term").size else float("nan"),
            "def_term_rate": float(np.mean(_col(f"{prefix}_def_term"))) if _col(f"{prefix}_def_term").size else float("nan"),
            "oi_viol_def_rate": float(np.mean(_col(f"{prefix}_oi_viol_def"))) if _col(f"{prefix}_oi_viol_def").size else float("nan"),
            "oi_viol_att_rate": float(np.mean(_col(f"{prefix}_oi_viol_att"))) if _col(f"{prefix}_oi_viol_att").size else float("nan"),
            "udef_mean": _metric_summary(f"{prefix}_udef_mean"),
            "uatt_mean": _metric_summary(f"{prefix}_uatt_mean"),
            "rollout_num_steps": _metric_summary(f"{prefix}_rollout_num_steps"),
            "rollout_setup_sec": _metric_summary(f"{prefix}_rollout_setup_sec"),
            "rollout_setup_sec_per_step": _metric_summary(f"{prefix}_rollout_setup_sec_per_step"),
            "rollout_simulation_sec": _metric_summary(f"{prefix}_rollout_simulation_sec"),
            "rollout_simulation_sec_per_step": _metric_summary(f"{prefix}_rollout_simulation_sec_per_step"),
            "rollout_total_sec": _metric_summary(f"{prefix}_rollout_total_sec"),
            "rollout_total_sec_per_step": _metric_summary(f"{prefix}_rollout_total_sec_per_step"),
        }


    trial_attacker_hit_vals = _col("trial_attacker_hit_any")
    primary_metric_name = "defender_capture_pass_rate" if args.policy_role == "def" else (
        "attacker_hit_any_rate" if num_attackers > 1 else "attacker_hit_rate"
    )
    primary_metric_value = float(sr) if args.policy_role == "def" else (
        float(np.mean(trial_attacker_hit_vals)) if trial_attacker_hit_vals.size else float("nan")
    )

    results = {
        "num_trials": n,
        "passes": k,
        "success_rate": float(sr),
        "defender_pass_rate": float(sr),
        "policy_role": args.policy_role,
        "opponent_source": args.opponent_source,
        "policy_primary_metric": {
            "name": primary_metric_name,
            "value": primary_metric_value,
        },
        "success_rate_ci_wilson": {"alpha": float(args.alpha), "lo": float(lo), "hi": float(hi)},
        "config_module": args.config_module,
        "base_cfg_source": base_cfg_source,
        "run_manifest": str(manifest_path) if manifest_path is not None else None,
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
        "outcome_breakdown": outcome_breakdown,
        "notes": [
            "Rollouts use the RL runner for policy-vs-policy and the mixed matchup runner for policy-vs-baseline evaluation.",
            "For num_attackers>1, evaluation runs one joint rollout per trial and extracts attacker-specific metrics from the shared trajectory.",
            "When zero_sum_reward.mode is enabled and collision_radius_m > 0, PASS means the defender captures the attacker before any attacker hit or invalid termination/keepout event.",
            "Otherwise PASS uses the legacy verifier: attacker does not hit target and no terminations/keepout violations, with optional capture gating via cfg['verify_require_capture'].",
            "grid_mode=paired uses --trials_in rows or --auto_shell_grid pairing.",
            "grid_mode=cartesian uses product of --def_trials_in x --att_trials_in or --auto_shell_grid grids.",
            "If attacker grid rows only provide att1_* and num_attackers>1, evaluation deterministically expands them into attacker-team rows.",
            "If --max_pairs is set in cartesian mode, pairs are sampled (not guaranteed unique).",
            "When policy_role=att, success_rate remains the defender-centric pass metric; use policy_primary_metric or metrics_trial.attacker_hit_any_rate for attacker-side comparisons.",
        ],
        "metrics_trial": {
            "min_rel_dist": _metric_summary("trial_min_rel_dist"),
            "min_att_center": _metric_summary("trial_min_att_center"),
            "attacker_hit_any_rate": float(np.mean(_col("trial_attacker_hit_any"))) if _col("trial_attacker_hit_any").size else float("nan"),
            "collision_any_rate": float(np.mean(_col("trial_collided_any"))) if _col("trial_collided_any").size else float("nan"),
            "att_term_any_rate": float(np.mean(_col("trial_att_term_any"))) if _col("trial_att_term_any").size else float("nan"),
            "def_term_any_rate": float(np.mean(_col("trial_def_term_any"))) if _col("trial_def_term_any").size else float("nan"),
            "oi_viol_def_any_rate": float(np.mean(_col("trial_oi_viol_def_any"))) if _col("trial_oi_viol_def_any").size else float("nan"),
            "oi_viol_att_any_rate": float(np.mean(_col("trial_oi_viol_att_any"))) if _col("trial_oi_viol_att_any").size else float("nan"),
            "rollout_error_any_rate": float(np.mean(_col("trial_rollout_error_any"))) if _col("trial_rollout_error_any").size else float("nan"),
            "udef_mean": _metric_summary("trial_udef_mean"),
            "uatt_mean": _metric_summary("trial_uatt_mean"),
            "rollout_num_steps": _metric_summary("trial_rollout_num_steps"),
            "rollout_setup_sec": _metric_summary("trial_rollout_setup_sec"),
            "rollout_setup_sec_per_step": _metric_summary("trial_rollout_setup_sec_per_step"),
            "rollout_simulation_sec": _metric_summary("trial_rollout_simulation_sec"),
            "rollout_simulation_sec_per_step": _metric_summary("trial_rollout_simulation_sec_per_step"),
            "rollout_total_sec": _metric_summary("trial_rollout_total_sec"),
            "rollout_total_sec_per_step": _metric_summary("trial_rollout_total_sec_per_step"),
        },
    }
    for j in range(num_attackers):
        results[f"metrics_att{j+1}"] = _attacker_metric_block(f"att{j+1}")


    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    csv_path = out_dir / "trials.csv"
    fieldnames = sorted({k_ for r in trial_rows for k_ in r.keys()})
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in trial_rows:
            w.writerow(r)

    timeout_rows = [r for r in trial_rows if r.get("outcome_label") == "timeout_no_capture"]
    timeout_csv_path = out_dir / "timeout_no_capture_cases.csv"
    with timeout_csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in timeout_rows:
            w.writerow(r)

    clean_resolution_rows = [
        r for r in trial_rows
        if int(r.get("trial_attacker_hit_any", 0)) == 0
        and int(r.get("trial_def_term_any", 0)) == 0
        and int(r.get("trial_att_term_any", 0)) == 0
        and int(r.get("trial_oi_viol_def_any", 0)) == 0
        and int(r.get("trial_oi_viol_att_any", 0)) == 0
    ]
    clean_csv_path = out_dir / "clean_non_hit_non_term_cases.csv"
    with clean_csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in clean_resolution_rows:
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
    log(f"[eval] wrote: {out_dir/'timeout_no_capture_cases.csv'} ({len(timeout_rows)} rows)")
    log(f"[eval] wrote: {out_dir/'clean_non_hit_non_term_cases.csv'} ({len(clean_resolution_rows)} rows)")
    log(f"[eval] wrote: {out_dir/'starts_xy.png'}")
    log(f"[eval] wrote: {out_dir/'starts_xz.png'}")
    log(f"[eval] total_time={time.time() - t_global0:.3f}s")


if __name__ == "__main__":
    main()


"""
python evaluate_policy.py \
  --run_dir Training_Policy \
  --def_ckpt_path Training_Policy/def1_teacher.pt \
  --att_ckpt_path Training_Policy/att1_teacher.pt \
  --sample_ic \
  --out_dir Training_Policy/MC_eval/def1_vs_att1


"""

"""

python evaluate_policy.py \
  --def_ckpt_path Training_Policy/def1_def_teacher.pt \
  --att_ckpt_path Training_Policy/att1_att_teacher.pt \
  --out_dir Training_Policy/MC_eval/def1_vs_att1

"""

"""

python evaluate_policy.py \
  --def_ckpt_path Training_Policy_0.75_Collision/def1_teacher.pt \
  --att_ckpt_path Training_Policy_0.75_Collision/att1_teacher.pt \
  --out_dir Training_Policy_0.75_Collision/MC_eval/def1_vs_att1 \
  --auto_shell_grid \
  --grid_mode cartesian \
  --shell_fracs 0.2,0.4,0.6,0.8 \
  --points_per_shell 40
  

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
