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
from game_runner import (
    run_batched_rhc_with_rl_and_collect_frames_3d,
    run_rhc_with_rl_and_collect_frames_3d,
)
from matchup_runner import (
    SUPPORTED_BASELINE_OPPONENTS,
    run_rhc_with_policy_vs_baseline_collect_frames_3d,
)
from core.utils import resolve_start_radius_bounds, set_seed
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
    "opponent_source": "policy",   # "policy" | "paper" | "game_theory" | "ipopt" | "rule"
    "sample_ic": False,
    "pos_scale": 0.95,
    "vel_scale": 0.0,
    "min_sep": 0.0,
    "advantage_scale": None,
    "alpha": 0.05,
    "log_every": 10,
    "print_first_out_keys": False,
    "print_errors": False,
    "trace_errors": False,
    "save_rollout_error_cases": False,
    "trials_in": None,
    "grid_mode": "paired",
    "def_trials_in": None,
    "att_trials_in": None,
    "max_pairs": None,
    "auto_shell_grid": False,
    "shell_fracs": "0.2,0.4,0.6,0.8",
    "def_shell_radius": None,
    "att_shell_radius": None,
    "points_per_shell": 40,
    "include_center": False,

    # Intentional config overrides: these can overwrite fields loaded from
    # run_manifest.json when set to a non-None or active value.
    "steps": None,
    "device": None,
    "eval_batch_size": None,
    "def_ckpt_path": None,
    "att_ckpt_path": None,
    "attacker_mode": None,
    "deterministic": None,
    "use_kf": None,
    "estimator_kind": None,
    "kf_action_access": None,
    "kf_action_meas_std": None,
    "vmax": None,
    "velocity_controller_enabled": None,
    "velocity_controller_speed": None,
    "umax": None,
    "arena_radius": None,
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


def _parse_bool(value: str) -> bool:
    key = str(value).strip().lower()
    if key in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if key in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Could not parse boolean value {value!r}. Expected true/false."
    )


def _parse_scalar_or_vector(value: str) -> float | List[float]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Expected a scalar or comma-separated list of floats.")
    vals = [float(p) for p in parts]
    if len(vals) == 1:
        return float(vals[0])
    return [float(v) for v in vals]


def _normalize_attacker_mode(value: Any) -> str:
    key = str(value).strip().lower()
    if key in {"rl", "train"}:
        return "rl"
    if key == "rule":
        return "rule"
    raise RuntimeError(
        f"Unsupported attacker_mode={value!r}; expected 'rl', 'train', or 'rule'."
    )



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

    num_attackers = int(cfg.get("num_attackers", 1))
    if num_attackers != 1:
        raise RuntimeError(
            "evaluate_policy.py now supports only num_attackers=1. "
            f"Loaded cfg['num_attackers']={num_attackers}."
        )

    attacker_mode = _normalize_attacker_mode(cfg.get("attacker_mode", "rl"))
    cfg["attacker_mode"] = attacker_mode

    if args.opponent_source == "rule" and args.policy_role != "def":
        raise RuntimeError(
            "opponent_source='rule' is only supported when evaluating the defender policy "
            "against a rule-based attacker."
        )
    if args.policy_role == "att" and attacker_mode != "rl":
        raise RuntimeError(
            "policy_role='att' requires attacker_mode='rl' so the evaluated attacker policy "
            "is loaded from a checkpoint."
        )

    if args.opponent_source == "policy":
        _require_existing_file(cfg.get("def_ckpt_path"), "defender checkpoint")
        if attacker_mode == "rl":
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


def _numeric_summary_from_values(vals: np.ndarray) -> Dict[str, float]:
    x = np.asarray(vals, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "std_sample": float("nan"),
            "stderr": float("nan"),
            **_quantiles(x),
        }

    std = float(np.std(x, ddof=0))
    std_sample = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    stderr = float(std / np.sqrt(n))
    return {
        "n": n,
        "mean": float(np.mean(x)),
        "std": std,
        "std_sample": std_sample,
        "stderr": stderr,
        **_quantiles(x),
    }


def _binary_summary_from_values(vals: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    x = np.asarray(vals, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size:
        x = np.clip(np.rint(x), 0.0, 1.0)

    n = int(x.size)
    if n == 0:
        nan = float("nan")
        return {
            "n": 0,
            "successes": 0,
            "failures": 0,
            "rate": nan,
            "mean": nan,
            "stderr": nan,
            "ci_wilson": {"alpha": float(alpha), "lo": nan, "hi": nan},
        }

    k = int(np.sum(x))
    p = float(k / n)
    lo, hi = wilson_ci(k, n, alpha=float(alpha))
    stderr = float(np.sqrt(p * (1.0 - p) / n))
    return {
        "n": n,
        "successes": k,
        "failures": n - k,
        "rate": p,
        "mean": p,
        "stderr": stderr,
        "ci_wilson": {"alpha": float(alpha), "lo": float(lo), "hi": float(hi)},
    }


def _binary_ci_errorbars(stats: Dict[str, Any]) -> Tuple[float, float]:
    mean = float(stats.get("mean", float("nan")))
    ci = stats.get("ci_wilson", {}) or {}
    lo = float(ci.get("lo", float("nan")))
    hi = float(ci.get("hi", float("nan")))
    if not (np.isfinite(mean) and np.isfinite(lo) and np.isfinite(hi)):
        return float("nan"), float("nan")
    return max(0.0, mean - lo), max(0.0, hi - mean)


def _event_time_summary_from_values(vals: np.ndarray) -> Dict[str, float]:
    x = np.asarray(vals, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    x = x[x >= 0.0]
    return _numeric_summary_from_values(x)


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


def _resolve_shell_plan_radius(
    raw_val: Optional[float],
    arena_r: float,
    label: str,
) -> float:
    """
    Resolve the base radius used for shell planning.

    If unspecified, shell planning falls back to the arena radius. Otherwise,
    reuse _radius_from_cfg so callers may specify either meters or a fraction
    of the arena radius.
    """
    if raw_val is None:
        return float(arena_r)

    radius = _radius_from_cfg(float(raw_val), arena_r)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(
            f"{label} must resolve to a finite positive radius, got {raw_val!r}."
        )
    return float(radius)


def _apply_x0_jitter(cfg: Dict[str, Any], x0: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    jit = cfg.get("x0_jitter", {}) or {}
    pos_j = float(jit.get("pos", 0.0))
    vel_j = float(jit.get("vel", 0.0))
    D = int(cfg.get("D", x0.shape[1] // 2))
    out = x0.copy().astype(float)
    out[:, :D] += rng.normal(size=out[:, :D].shape) * pos_j
    out[:, D:2*D] += rng.normal(size=out[:, D:2*D].shape) * vel_j
    return out


def _resolve_rollout_vmax(cfg: Dict[str, Any]) -> Optional[float]:
    sf_cfg = cfg.get("safety_filter", {}) or {}
    raw_vmax = sf_cfg.get("vmax", None)
    if raw_vmax is None:
        raw_vmax = cfg.get("vmax", None)
    if raw_vmax is None:
        return None
    vmax = float(raw_vmax)
    if not np.isfinite(vmax) or vmax <= 0.0:
        raise ValueError(f"vmax must be finite and > 0, got {raw_vmax!r}")
    return vmax


def _project_x0_velocities_to_vmax(x0: np.ndarray, D: int, vmax: float) -> np.ndarray:
    out = np.asarray(x0, dtype=float).copy()
    if not np.isfinite(vmax) or vmax <= 0.0:
        return out

    vel = out[:, D:2 * D]
    speeds = np.linalg.norm(vel, axis=1, keepdims=True)
    scale = np.ones_like(speeds, dtype=float)
    mask = speeds > float(vmax)
    scale[mask] = float(vmax) / (speeds[mask] + 1e-12)
    vel *= scale
    out[:, D:2 * D] = vel
    return out


def _combine_reproducibility_seed(seeds: List[int]) -> int:
    """
    Fold a list of episode seeds into one stable 32-bit seed for global torch/NumPy setup.
    """
    acc = 2166136261
    for idx, seed in enumerate(seeds):
        mix = (int(seed) + 0x9E3779B9 + idx * 0x85EBCA6B) & 0xFFFFFFFF
        acc ^= mix
        acc = (acc * 16777619) & 0xFFFFFFFF
    return int(acc or 1)


def _is_cuda_device(device: Any) -> bool:
    if device is None:
        return False
    key = str(device).strip().lower()
    return key == "cuda" or key.startswith("cuda:")


def _supports_batched_cuda_eval(cfg: Dict[str, Any], opponent_source: str) -> bool:
    if not _is_cuda_device(cfg.get("device", None)):
        return False
    if str(opponent_source).strip().lower() != "policy":
        return False
    if int(cfg.get("num_attackers", 1)) != 1:
        return False
    if str(cfg.get("dynamics", "hcw")).strip().lower() != "hcw":
        return False
    disp_params = ((cfg.get("dispersion", {}) or {}).get("params", {}) or {})
    if bool(disp_params.get("enabled", False)):
        return False
    return True


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
    shell_radius: float,
    shell_fracs: List[float],
    points_per_shell: int,
    rng: Optional[np.random.Generator] = None,
    include_center: bool = False,
) -> np.ndarray:
    """
    Generate positions on shells at radii = frac * shell_radius. Returns (N,3).
    """
    shell_fracs = [float(s) for s in shell_fracs if float(s) > 0.0]
    pts_all = []
    if include_center:
        pts_all.append(center.reshape(1, 3).copy())

    for frac in shell_fracs:
        rad = frac * float(shell_radius)
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


def _radius_key(val: float, ndigits: int = 6) -> float:
    return round(float(val), ndigits)


def _shell_radius_from_row(
    row: Dict[str, float],
    prefix: str,
    D: int,
    center: np.ndarray,
) -> float:
    p = _entity_state_from_row(row, prefix, D)[:D]
    return float(np.linalg.norm(p - center))


def _format_shell_radius_counts(
    rows: List[Dict[str, float]],
    prefix: str,
    D: int,
    center: np.ndarray,
) -> str:
    counts: Dict[float, int] = {}
    for row in rows:
        key = _radius_key(_shell_radius_from_row(row, prefix, D, center))
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{r:g} x{counts[r]}" for r in sorted(counts))


def _format_valid_shell_pair_counts(
    cfg: Dict[str, Any],
    def_rows: List[Dict[str, float]],
    att_rows: List[Dict[str, float]],
    valid_pair_indices: List[Tuple[int, int]],
) -> str:
    D = int(cfg.get("D", 3))
    center, _arena_r = _get_center_and_radius(cfg, D)
    def_radii = [_radius_key(_shell_radius_from_row(row, "def", D, center)) for row in def_rows]
    att_radii = [_radius_key(_shell_radius_from_row(row, "att1", D, center)) for row in att_rows]

    pair_counts: Dict[Tuple[float, float], int] = {}
    for di, ai in valid_pair_indices:
        key = (def_radii[di], att_radii[ai])
        pair_counts[key] = pair_counts.get(key, 0) + 1

    if not pair_counts:
        return "none"
    return ", ".join(
        f"{r_def:g}->{r_att:g}: {count}"
        for (r_def, r_att), count in sorted(pair_counts.items())
    )



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


def _default_radial_advantage_margin(cfg: Dict[str, Any]) -> float:
    oi_r = float((cfg.get("oi", {}) or {}).get("r", 0.0))
    percent_advantage_defender = float(cfg.get("percent_advantage_defender", 0.75))
    radial_margin = float(percent_advantage_defender * np.pi * 2.0 * oi_r)
    if radial_margin < 0.0:
        raise ValueError(f"percent_advantage_defender must be >= 0, got {percent_advantage_defender}")
    return radial_margin


def _radial_advantage_scale(cfg: Dict[str, Any]) -> float:
    raw_scale = cfg.get("advantage_scale", 1.0)
    if raw_scale is None:
        return 1.0
    scale = float(raw_scale)
    if not np.isfinite(scale):
        raise ValueError(f"advantage_scale must be finite, got {raw_scale!r}")
    return scale


def _radial_advantage_margin(cfg: Dict[str, Any]) -> float:
    return float(_default_radial_advantage_margin(cfg) * _radial_advantage_scale(cfg))


def _advantage_constraint_satisfied(r_def: float, r_atts: List[float], radial_margin: float) -> bool:
    if radial_margin >= 0.0:
        return all(r_def <= (r_att - radial_margin) for r_att in r_atts)

    attacker_margin = -float(radial_margin)
    return all(r_att <= (r_def - attacker_margin) for r_att in r_atts)


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

            if require_training_advantage and not _advantage_constraint_satisfied(r_def, r_atts, radial_margin):
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


def _sample_x0_random_shell(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    num_attackers: int,
    vel_scale_override: Optional[float] = None,
    min_sep_override: Optional[float] = None,
) -> np.ndarray:
    """
    Mirror core/env.py random_shell so evaluation can sample the same
    shell-bounded initial-condition family used during training.
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

    r_def_min, r_def_max = resolve_start_radius_bounds(
        cfg,
        R,
        who="def",
        default_min_frac=0.0,
        default_max_frac=1.0,
    )
    r_att_min, r_att_max = resolve_start_radius_bounds(
        cfg,
        R,
        who="att",
        default_min_frac=0.0,
        default_max_frac=1.0,
    )

    p_def = _sample_in_shell(rng, center, r_def_min, r_def_max)
    v_def = rng.uniform(-v_max, v_max, size=D)

    xs = [np.concatenate([p_def, v_def], dtype=float)[:nx]]
    p_atts: List[np.ndarray] = []
    for _ in range(num_attackers):
        for _att_try in range(1000):
            p_att = _sample_in_shell(rng, center, r_att_min, r_att_max)
            if np.linalg.norm(p_att - p_def) < min_sep:
                continue
            if any(np.linalg.norm(p_att - p_prev) < min_sep for p_prev in p_atts):
                continue
            break
        else:
            raise RuntimeError(
                "random_shell: could not sample a feasible attacker initial condition after many attempts. "
                "Try relaxing r_att_min/r_att_max or reducing train_min_sep."
            )
        p_atts.append(p_att)
        v_att = rng.uniform(-v_max, v_max, size=D)
        xs.append(np.concatenate([p_att, v_att], dtype=float)[:nx])

    return np.asarray(xs, dtype=float)


def _sample_x0_random_shell_advantage(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
    num_attackers: int,
    vel_scale_override: Optional[float] = None,
    min_sep_override: Optional[float] = None,
) -> np.ndarray:
    """
    Mirror core/env.py random_shell_advantage so evaluation can sample the
    same training-style radial geometry used during training, with an
    optional eval-time signed advantage scale override.
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

    radial_margin = _radial_advantage_margin(cfg)

    r_def_min, r_def_max = resolve_start_radius_bounds(
        cfg,
        R,
        who="def",
        default_min_frac=0.0,
        default_max_frac=1.0,
    )
    r_att_min, r_att_max = resolve_start_radius_bounds(
        cfg,
        R,
        who="att",
        default_min_frac=0.0,
        default_max_frac=1.0,
    )
    margin_abs = abs(radial_margin)

    if radial_margin >= 0.0:
        r_att_min = max(r_att_min, margin_abs)
    else:
        r_att_max = min(r_att_max, r_def_max - margin_abs)

    if r_def_min > r_def_max:
        raise ValueError(f"Invalid defender shell: [{r_def_min}, {r_def_max}]")
    if r_att_min > r_att_max:
        raise ValueError(f"Invalid attacker shell: [{r_att_min}, {r_att_max}]")
    if num_attackers > 0:
        if radial_margin >= 0.0 and r_def_min > (r_att_max - margin_abs):
            raise ValueError(
                "Infeasible radial shells: defender cannot be at least "
                f"{margin_abs:.3f} m closer to center than attacker. "
                f"Got r_def_min={r_def_min:.3f}, r_att_max={r_att_max:.3f}."
            )
        if radial_margin < 0.0 and r_att_min > (r_def_max - margin_abs):
            raise ValueError(
                "Infeasible radial shells: attacker cannot be at least "
                f"{margin_abs:.3f} m closer to center than defender. "
                f"Got r_att_min={r_att_min:.3f}, r_def_max={r_def_max:.3f}."
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

        found_def = False
        if radial_margin >= 0.0:
            r_att_nearest = min(r_atts)
            r_def_max_eff = min(r_def_max, r_att_nearest - margin_abs)
            if r_def_max_eff < r_def_min:
                continue

            for _def_try in range(1000):
                cand = _sample_in_shell(rng, center, r_def_min, r_def_max_eff)
                if any(np.linalg.norm(cand - p_att) < min_sep for p_att in p_atts):
                    continue
                p_def = cand
                found_def = True
                break
        else:
            r_att_farthest = max(r_atts)
            r_def_min_eff = max(r_def_min, r_att_farthest + margin_abs)
            if r_def_min_eff > r_def_max:
                continue

            for _def_try in range(1000):
                cand = _sample_in_shell(rng, center, r_def_min_eff, r_def_max)
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
            "or adjusting advantage_scale / percent_advantage_defender."
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
    if train_ic_mode == "random_shell":
        vel_scale_override = float(vel_scale) if cli_present and "--vel_scale" in cli_present else None
        min_sep_override = float(min_sep) if cli_present and "--min_sep" in cli_present else None
        return _sample_x0_random_shell(
            cfg,
            rng,
            num_attackers=num_attackers,
            vel_scale_override=vel_scale_override,
            min_sep_override=min_sep_override,
        )
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


def _outcome_pretty_name(label: str) -> str:
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
    return pretty.get(str(label), str(label))


def _outcome_color(label: str) -> str:
    palette = {
        "defender_capture": "#2ca02c",
        "defender_success": "#1f77b4",
        "attacker_hit_oi": "#d62728",
        "defender_crashed_wall": "#9467bd",
        "attacker_crashed_wall": "#ff7f0e",
        "defender_hit_oi": "#8c564b",
        "collision_but_not_success": "#e377c2",
        "timeout_no_capture": "#bcbd22",
        "capture_required_not_met": "#17becf",
        "rollout_error": "#7f7f7f",
        "unclassified_failure": "#111111",
    }
    return palette.get(str(label), "#111111")


def _save_outcome_histogram(
    out_dir: Path,
    trial_rows: List[Dict[str, Any]],
    alpha: float = 0.05,
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
    stats = {
        k: _binary_summary_from_values(
            np.asarray([1.0 if _classify_trial_outcome(row) == k else 0.0 for row in trial_rows], dtype=float),
            alpha=float(alpha),
        )
        for k in labels
    }
    left_errs = [ _binary_ci_errorbars(stats[k])[0] for k in present ]
    right_errs = [ _binary_ci_errorbars(stats[k])[1] for k in present ]
    fig_h = max(4.0, 0.55 * len(present) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    bars = ax.barh(
        range(len(present)),
        values,
        xerr=np.asarray([left_errs, right_errs], dtype=float),
        color="steelblue",
        edgecolor="black",
        ecolor="black",
        capsize=4,
    )
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels([pretty[k] for k in present])
    xmax = max((v + r) for v, r in zip(values, right_errs)) if values else 1.0
    ax.set_xlim(0.0, max(1.0, xmax * 1.15))
    ci_pct = int(round((1.0 - float(alpha)) * 100.0))
    ax.set_xlabel(f"Proportion of trials (-), error bars = Wilson {ci_pct}% CI")
    ax.set_title("Trial Outcome Breakdown With Confidence Intervals")
    ax.grid(True, axis="x", alpha=0.3)

    for idx, (bar, key) in enumerate(zip(bars, present)):
        v = values[idx]
        lo = float(stats[key]["ci_wilson"]["lo"])
        hi = float(stats[key]["ci_wilson"]["hi"])
        ax.text(
            v + right_errs[idx] + 0.01 * max(1.0, xmax),
            bar.get_y() + bar.get_height() / 2.0,
            f"{counts[key]}/{n} ({v:.1%}, {ci_pct}% CI [{lo:.1%}, {hi:.1%}])",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_dir / "outcome_hist.png", dpi=180)
    plt.close(fig)

    return {
        "counts": counts,
        "proportions": {k: counts[k] / float(n) for k in labels},
        "proportion_stats": stats,
    }


def _downsample_xyz_path(xyz: np.ndarray, max_points: int = 256) -> np.ndarray:
    arr = np.asarray(xyz, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    arr = arr[:, :3]
    if arr.shape[0] <= max_points:
        return np.asarray(arr, dtype=np.float32)
    idx = np.linspace(0, arr.shape[0] - 1, num=max_points, dtype=int)
    idx = np.unique(idx)
    return np.asarray(arr[idx], dtype=np.float32)


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

    dt = float(cfg.get("dt", float("nan")))
    u_real_norms = out.get("u_real_norm_all", None)
    delta_v_def_mean = delta_v_att_mean = float("nan")
    delta_v_def_total = delta_v_att_total = float("nan")
    delta_v_def_max = delta_v_att_max = float("nan")
    if np.isfinite(dt) and dt > 0.0 and u_real_norms is not None and len(u_real_norms) >= 2:
        dv_def = np.asarray(u_real_norms[0], dtype=float) * dt
        dv_att = np.asarray(u_real_norms[1], dtype=float) * dt
        if dv_def.size:
            delta_v_def_mean = float(np.mean(dv_def))
            delta_v_def_total = float(np.sum(dv_def))
            delta_v_def_max = float(np.max(dv_def))
        if dv_att.size:
            delta_v_att_mean = float(np.mean(dv_att))
            delta_v_att_total = float(np.sum(dv_att))
            delta_v_att_max = float(np.max(dv_att))

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
        "delta_v_def_mean": delta_v_def_mean,
        "delta_v_att_mean": delta_v_att_mean,
        "delta_v_def_total": delta_v_def_total,
        "delta_v_att_total": delta_v_att_total,
        "delta_v_def_max": delta_v_def_max,
        "delta_v_att_max": delta_v_att_max,
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
    for key in [
        "est12_xyz",
        "est21_xyz",
        "meas12_azel",
        "meas21_azel",
        "meas12_innov_sq",
        "meas21_innov_sq",
        "trP12_pos",
        "trP21_pos",
    ]:
        if key in out:
            single[key] = out.get(key)
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


def _extract_kf_trial_metrics(out: Dict[str, Any], D: int) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    def _pos_err_metrics(est_key: str, true_key: str, prefix: str) -> None:
        est = np.asarray(out.get(est_key, []), dtype=float)
        true = np.asarray(out.get(true_key, []), dtype=float)
        if est.ndim != 2 or true.ndim != 2 or est.size == 0 or true.size == 0:
            return
        n = min(len(est), len(true))
        if n <= 0:
            return
        err = np.linalg.norm(est[:n, :3] - true[:n, :3], axis=1)
        if err.size == 0:
            return
        metrics[f"{prefix}_pos_err_mean"] = float(np.mean(err))
        metrics[f"{prefix}_pos_err_rms"] = float(np.sqrt(np.mean(err ** 2)))
        metrics[f"{prefix}_pos_err_max"] = float(np.max(err))
        metrics[f"{prefix}_pos_err_final"] = float(err[-1])

    def _measurement_metrics(meas_key: str, innov_key: str, trp_key: str, prefix: str) -> None:
        meas = out.get(meas_key, None)
        if isinstance(meas, list):
            n_total = max(0, len(meas) - 1)
            n_obs = int(sum(m is not None for m in meas))
            metrics[f"{prefix}_meas_count"] = n_obs
            metrics[f"{prefix}_meas_fraction"] = float(n_obs / max(1, n_total)) if n_total > 0 else float("nan")

        innov = np.asarray(out.get(innov_key, []), dtype=float).reshape(-1)
        innov = innov[np.isfinite(innov)]
        if innov.size:
            metrics[f"{prefix}_meas_innov_sq_mean"] = float(np.mean(innov))
            metrics[f"{prefix}_meas_innov_sq_max"] = float(np.max(innov))

        trp = np.asarray(out.get(trp_key, []), dtype=float).reshape(-1)
        trp = trp[np.isfinite(trp)]
        if trp.size:
            metrics[f"{prefix}_trPpos_mean"] = float(np.mean(trp))
            metrics[f"{prefix}_trPpos_final"] = float(trp[-1])

    _pos_err_metrics("est12_xyz", "exec2_xyz", "kf_def_att")
    _pos_err_metrics("est21_xyz", "exec1_xyz", "kf_att_def")
    _measurement_metrics("meas12_azel", "meas12_innov_sq", "trP12_pos", "kf_def_att")
    _measurement_metrics("meas21_azel", "meas21_innov_sq", "trP21_pos", "kf_att_def")
    metrics["kf_enabled_rollout"] = int(bool(metrics))
    return metrics


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
        "trial_delta_v_def_mean": _mean_metric(per_att_metrics, "delta_v_def_mean"),
        "trial_delta_v_att_mean": _mean_metric(per_att_metrics, "delta_v_att_mean"),
        "trial_delta_v_def_total": _mean_metric(per_att_metrics, "delta_v_def_total"),
        "trial_delta_v_att_total": _mean_metric(per_att_metrics, "delta_v_att_total"),
        "trial_delta_v_def_max": _mean_metric(per_att_metrics, "delta_v_def_max"),
        "trial_delta_v_att_max": _mean_metric(per_att_metrics, "delta_v_att_max"),
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
    alpha: float = 0.05,
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

    def _coords(arr: np.ndarray, plane: str) -> Tuple[np.ndarray, np.ndarray]:
        if plane == "xy":
            return arr[:, 0], arr[:, 1]
        if plane == "xz":
            return arr[:, 0], arr[:, 2]
        raise ValueError(f"unknown plane={plane}")

    # -------------------------
    # 1) Plain coverage plots
    # -------------------------
    # XY
    fig, ax_cov = plt.subplots(figsize=(7, 6))
    ax_cov.scatter(
        starts_def[:, 0], starts_def[:, 1],
        marker="o", s=marker_size,
        edgecolors="k", linewidths=edge_lw,
        label="def_start"
    )
    for j, sa in enumerate(starts_atts):
        if sa.size == 0:
            continue
        ax_cov.scatter(
            sa[:, 0], sa[:, 1],
            marker="o", s=marker_size,
            edgecolors="k", linewidths=edge_lw,
            label=f"att{j+1}_start"
        )
    _decorate(ax_cov, "xy", "Evaluated Starting Positions (XY)", "x", "y")
    ax_cov.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "starts_xy.png", dpi=180)
    plt.close(fig)

    # XZ
    fig, ax_cov = plt.subplots(figsize=(7, 6))
    ax_cov.scatter(
        starts_def[:, 0], starts_def[:, 2],
        marker="o", s=marker_size,
        edgecolors="k", linewidths=edge_lw,
        label="def_start"
    )
    for j, sa in enumerate(starts_atts):
        if sa.size == 0:
            continue
        ax_cov.scatter(
            sa[:, 0], sa[:, 2],
            marker="o", s=marker_size,
            edgecolors="k", linewidths=edge_lw,
            label=f"att{j+1}_start"
        )
    _decorate(ax_cov, "xz", "Evaluated Starting Positions (XZ)", "x", "z")
    ax_cov.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "starts_xz.png", dpi=180)
    plt.close(fig)

    # -------------------------
    # 2) Success-rate overlays
    # -------------------------
    def _key(p, nd=6):
        return (round(float(p[0]), nd), round(float(p[1]), nd), round(float(p[2]), nd))



    def_passes: Dict[Tuple[float, float, float], List[int]] = {}
    att_passes: Dict[Tuple[float, float, float], List[int]] = {}

    for r in trial_rows:
        p = int(r.get("pass_trial", 0))
        kd = _key([r["def_x"], r["def_y"], r["def_z"]])

        def_passes.setdefault(kd, []).append(p)

        n_att_row = max(1, int(r.get("num_attackers", 1)))
        for j in range(n_att_row):
            ax = r.get(f"att{j+1}_x", float("nan"))
            ay = r.get(f"att{j+1}_y", float("nan"))
            az = r.get(f"att{j+1}_z", float("nan"))
            if any(np.isnan(float(v)) for v in (ax, ay, az)):
                continue
            ka = _key([ax, ay, az])
            att_passes.setdefault(ka, []).append(p)

    def_pts = np.array(list(def_passes.keys()), dtype=float)
    def_rate_stats = [
        _binary_summary_from_values(np.asarray(def_passes[k], dtype=float), alpha=float(alpha))
        for k in def_passes
    ]
    def_rates = np.array([s["mean"] for s in def_rate_stats], dtype=float)

    att_pts = np.array(list(att_passes.keys()), dtype=float)
    att_rate_stats = [
        _binary_summary_from_values(np.asarray(att_passes[k], dtype=float), alpha=float(alpha))
        for k in att_passes
    ]
    att_rates = np.array([s["mean"] for s in att_rate_stats], dtype=float)


    # Defender success XY
    fig, ax_map = plt.subplots(figsize=(7, 6))
    sc = ax_map.scatter(
        def_pts[:, 0], def_pts[:, 1],
        c=def_rates, cmap=cmap,
        marker="o", s=marker_size,
        edgecolors="k", linewidths=edge_lw
    )
    _decorate(ax_map, "xy", "Defender start: pass rate over attacker starts (XY)", "x", "y")
    cbar = fig.colorbar(sc, ax=ax_map)
    cbar.set_label("pass rate")
    fig.tight_layout()
    fig.savefig(out_dir / "def_success_xy.png", dpi=180)
    plt.close(fig)

    # Defender success XZ
    fig, ax_map = plt.subplots(figsize=(7, 6))
    sc = ax_map.scatter(
        def_pts[:, 0], def_pts[:, 2],
        c=def_rates, cmap=cmap,
        marker="o", s=marker_size,
        edgecolors="k", linewidths=edge_lw
    )
    _decorate(ax_map, "xz", "Defender start: pass rate over attacker starts (XZ)", "x", "z")
    cbar = fig.colorbar(sc, ax=ax_map)
    cbar.set_label("pass rate")
    fig.tight_layout()
    fig.savefig(out_dir / "def_success_xz.png", dpi=180)
    plt.close(fig)

    # Attacker success XY
    fig, ax_map = plt.subplots(figsize=(7, 6))
    sc = ax_map.scatter(
        att_pts[:, 0], att_pts[:, 1],
        c=att_rates, cmap=cmap,
        marker="o", s=marker_size,
        edgecolors="k", linewidths=edge_lw
    )
    _decorate(ax_map, "xy", "Attacker start: pass rate over defender starts (XY)", "x", "y")
    cbar = fig.colorbar(sc, ax=ax_map)
    cbar.set_label("pass rate")
    fig.tight_layout()
    fig.savefig(out_dir / "att_success_xy.png", dpi=180)
    plt.close(fig)

    # Attacker success XZ
    fig, ax_map = plt.subplots(figsize=(7, 6))
    sc = ax_map.scatter(
        att_pts[:, 0], att_pts[:, 2],
        c=att_rates, cmap=cmap,
        marker="o", s=marker_size,
        edgecolors="k", linewidths=edge_lw
    )
    _decorate(ax_map, "xz", "Attacker start: pass rate over defender starts (XZ)", "x", "z")
    cbar = fig.colorbar(sc, ax=ax_map)
    cbar.set_label("pass rate")
    fig.tight_layout()
    fig.savefig(out_dir / "att_success_xz.png", dpi=180)
    plt.close(fig)


def _save_success_vs_delta_v_plot(
    out_dir: Path,
    trial_rows: List[Dict[str, Any]],
) -> Optional[str]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    outcome_order = [
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

    grouped: Dict[str, List[Tuple[float, float, float]]] = {}
    for row in trial_rows:
        try:
            dv_def = float(row.get("trial_delta_v_def_total", float("nan")))
            dv_att = float(row.get("trial_delta_v_att_total", float("nan")))
            pass_trial = float(row.get("pass_trial", float("nan")))
        except Exception:
            continue
        if not (np.isfinite(dv_def) and np.isfinite(dv_att) and np.isfinite(pass_trial)):
            continue
        label = str(row.get("outcome_label", _classify_trial_outcome(row)))
        grouped.setdefault(label, []).append((dv_def, dv_att, pass_trial))

    if not grouped:
        return None

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_success, ax_pareto = axes

    present_labels: List[str] = []
    for label in outcome_order:
        pts = grouped.get(label, [])
        if not pts:
            continue
        present_labels.append(label)
        arr = np.asarray(pts, dtype=float)
        color = _outcome_color(label)
        y_jitter = arr[:, 2] + rng.uniform(-0.06, 0.06, size=arr.shape[0])
        ax_success.scatter(
            arr[:, 0],
            y_jitter,
            s=12,
            alpha=0.45,
            c=color,
            edgecolors="none",
        )
        ax_pareto.scatter(
            arr[:, 0],
            arr[:, 1],
            s=12,
            alpha=0.45,
            c=color,
            edgecolors="none",
        )

    ax_success.set_title("Success vs Defender Total Delta-v")
    ax_success.set_xlabel("Defender total delta-v")
    ax_success.set_ylabel("Trial outcome")
    ax_success.set_yticks([0.0, 1.0])
    ax_success.set_yticklabels(["Fail", "Pass"])
    ax_success.set_ylim(-0.2, 1.2)
    ax_success.grid(True, alpha=0.3)

    ax_pareto.set_title("Delta-v Pareto View")
    ax_pareto.set_xlabel("Defender total delta-v")
    ax_pareto.set_ylabel("Attacker total delta-v")
    ax_pareto.grid(True, alpha=0.3)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=6,
            markerfacecolor=_outcome_color(label),
            markeredgecolor="none",
            label=_outcome_pretty_name(label),
        )
        for label in present_labels
    ]
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(3, len(legend_handles)),
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    path = out_dir / "success_vs_delta_v.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path.name)


def _save_time_to_event_histograms(
    out_dir: Path,
    trial_rows: List[Dict[str, Any]],
    dt: float,
) -> Optional[str]:
    import matplotlib.pyplot as plt

    if not np.isfinite(dt) or dt <= 0.0:
        return None

    capture_times: List[float] = []
    attacker_hit_times: List[float] = []
    timeout_times: List[float] = []

    for row in trial_rows:
        label = str(row.get("outcome_label", _classify_trial_outcome(row)))

        t_collide = row.get("trial_t_collide", float("nan"))
        try:
            t_collide_f = float(t_collide)
        except Exception:
            t_collide_f = float("nan")
        if label == "defender_capture" and np.isfinite(t_collide_f) and t_collide_f >= 0.0:
            capture_times.append(t_collide_f * float(dt))

        t_att_hit = row.get("trial_t_att_hit", float("nan"))
        try:
            t_att_hit_f = float(t_att_hit)
        except Exception:
            t_att_hit_f = float("nan")
        if int(row.get("trial_attacker_hit_any", 0)) == 1 and np.isfinite(t_att_hit_f) and t_att_hit_f >= 0.0:
            attacker_hit_times.append(t_att_hit_f * float(dt))

        if label == "timeout_no_capture":
            try:
                timeout_steps = float(row.get("trial_rollout_num_steps", float("nan")))
            except Exception:
                timeout_steps = float("nan")
            if np.isfinite(timeout_steps) and timeout_steps >= 0.0:
                timeout_times.append(timeout_steps * float(dt))

    datasets = [
        ("Time to capture", np.asarray(capture_times, dtype=float), "#2ca02c"),
        ("Time to attacker hit", np.asarray(attacker_hit_times, dtype=float), "#d62728"),
        ("Time to timeout", np.asarray(timeout_times, dtype=float), "#bcbd22"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (title, vals, color) in zip(axes, datasets):
        ax.set_title(title)
        ax.set_xlabel("Time from rollout start (s)")
        ax.set_ylabel("Trials")
        ax.grid(True, alpha=0.3)
        if vals.size == 0:
            ax.text(0.5, 0.5, "No events", ha="center", va="center", transform=ax.transAxes)
            continue

        bins = min(40, max(10, int(np.sqrt(vals.size))))
        ax.hist(vals, bins=bins, color=color, edgecolor="black", alpha=0.8)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=0))
        ax.axvline(mean, color="black", linestyle="--", linewidth=1.2)
        ax.text(
            0.98,
            0.95,
            f"n={vals.size}\nmean={mean:.2f}s\nstd={std:.2f}s",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    fig.tight_layout()
    path = out_dir / "time_to_event_histograms.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path.name)


def _save_trajectory_overlay_plots(
    out_dir: Path,
    cfg: Dict[str, Any],
    trajectory_records: List[Tuple[str, np.ndarray, np.ndarray]],
) -> List[str]:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.colors import to_rgba

    if not trajectory_records:
        return []

    D = int(cfg.get("D", 3))
    if D != 3:
        return []

    center, arena_r = _get_center_and_radius(cfg, D)
    oi = cfg.get("oi", {}) or {}
    oi_enabled = bool(oi.get("enabled", False))
    oi_r = float(oi.get("r", 0.0)) if oi_enabled else 0.0
    success_labels = {"defender_capture", "defender_success"}
    timeout_labels = {"timeout_no_capture"}

    def _segments(plane: str, role_idx: int, group: str) -> List[np.ndarray]:
        segs: List[np.ndarray] = []
        for outcome_label, def_xyz, att_xyz in trajectory_records:
            if group == "success":
                keep = outcome_label in success_labels
            elif group == "timeout":
                keep = outcome_label in timeout_labels
            elif group == "failure":
                keep = outcome_label not in success_labels and outcome_label not in timeout_labels
            else:
                raise ValueError(f"unknown trajectory group={group!r}")
            if not keep:
                continue
            xyz = def_xyz if role_idx == 0 else att_xyz
            if xyz.ndim != 2 or xyz.shape[0] < 2:
                continue
            if plane == "xy":
                segs.append(np.asarray(xyz[:, [0, 1]], dtype=float))
            else:
                segs.append(np.asarray(xyz[:, [0, 2]], dtype=float))
        return segs

    def _draw_circle(ax, plane: str, r: float, label: Optional[str] = None, lw: float = 1.2) -> None:
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        if plane == "xy":
            circ = plt.Circle((cx, cy), r, fill=False, linewidth=lw)
            ax.add_patch(circ)
            if label:
                ax.text(cx + r, cy, label, fontsize=8, va="center")
        else:
            circ = plt.Circle((cx, cz), r, fill=False, linewidth=lw)
            ax.add_patch(circ)
            if label:
                ax.text(cx + r, cz, label, fontsize=8, va="center")

    def _decorate(ax, plane: str, title: str, ylabel: str) -> None:
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        _draw_circle(ax, plane, float(arena_r), label="arena", lw=1.2)
        if oi_enabled and oi_r > 0.0:
            _draw_circle(ax, plane, float(oi_r), label="OI", lw=1.2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(float(center[0] - 1.05 * arena_r), float(center[0] + 1.05 * arena_r))
        if plane == "xy":
            ax.set_ylim(float(center[1] - 1.05 * arena_r), float(center[1] + 1.05 * arena_r))
        else:
            ax.set_ylim(float(center[2] - 1.05 * arena_r), float(center[2] + 1.05 * arena_r))

    def _save_plane(plane: str, filename: str, ylabel: str) -> str:
        success_def = _segments(plane, 0, "success")
        success_att = _segments(plane, 1, "success")
        timeout_def = _segments(plane, 0, "timeout")
        timeout_att = _segments(plane, 1, "timeout")
        fail_def = _segments(plane, 0, "failure")
        fail_att = _segments(plane, 1, "failure")

        fig, ax = plt.subplots(figsize=(8, 7))
        _decorate(
            ax,
            plane,
            f"Trajectory Overlay ({plane.upper()}): green success underlay, grey timeout mid-layer, red failure overlay",
            ylabel,
        )

        collections = [
            (success_def, to_rgba("#2ca02c", alpha=0.05), "solid", 1),
            (success_att, to_rgba("#2ca02c", alpha=0.05), "dashed", 1),
            (timeout_def, to_rgba("#7f7f7f", alpha=0.07), "solid", 2),
            (timeout_att, to_rgba("#7f7f7f", alpha=0.07), "dashed", 2),
            (fail_def, to_rgba("#d62728", alpha=0.08), "solid", 3),
            (fail_att, to_rgba("#d62728", alpha=0.08), "dashed", 3),
        ]
        for segs, color, linestyle, zorder in collections:
            if not segs:
                continue
            lc = LineCollection(
                segs,
                colors=[color],
                linewidths=0.5,
                linestyles=linestyle,
                zorder=zorder,
            )
            ax.add_collection(lc)

        n_success = sum(1 for label, _, _ in trajectory_records if label in success_labels)
        n_timeout = sum(1 for label, _, _ in trajectory_records if label in timeout_labels)
        n_fail = max(0, len(trajectory_records) - n_success - n_timeout)
        legend_handles = [
            Line2D([0], [0], color="#2ca02c", linestyle="solid", linewidth=1.0, label=f"Successful defender (n={n_success})"),
            Line2D([0], [0], color="#2ca02c", linestyle="dashed", linewidth=1.0, label="Successful attacker"),
            Line2D([0], [0], color="#7f7f7f", linestyle="solid", linewidth=1.0, label=f"Timeout defender (n={n_timeout})"),
            Line2D([0], [0], color="#7f7f7f", linestyle="dashed", linewidth=1.0, label="Timeout attacker"),
            Line2D([0], [0], color="#d62728", linestyle="solid", linewidth=1.0, label=f"Unsuccessful defender (n={n_fail})"),
            Line2D([0], [0], color="#d62728", linestyle="dashed", linewidth=1.0, label="Unsuccessful attacker"),
        ]
        ax.legend(handles=legend_handles, loc="upper right")
        fig.tight_layout()
        path = out_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return str(path.name)

    return [
        _save_plane("xy", "trajectory_overlay_xy.png", "y (m)"),
        _save_plane("xz", "trajectory_overlay_xz.png", "z (m)"),
    ]


def _save_kf_eval_plots(
    out_dir: Path,
    trial_rows: List[Dict[str, Any]],
    estimator_label: str,
) -> List[str]:
    import matplotlib.pyplot as plt

    def _vals(name: str) -> np.ndarray:
        vals = []
        for row in trial_rows:
            try:
                v = float(row.get(name, float("nan")))
            except Exception:
                continue
            if np.isfinite(v):
                vals.append(v)
        return np.asarray(vals, dtype=float)

    def _save_summary_barplot(filename: str, title: str, items: List[Tuple[str, str]]) -> Optional[str]:
        labels: List[str] = []
        means: List[float] = []
        stds: List[float] = []
        ns: List[int] = []
        for key, label in items:
            vals = _vals(key)
            if vals.size == 0:
                continue
            labels.append(label)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=0)))
            ns.append(int(vals.size))
        if not labels:
            return None

        fig_h = max(4.0, 0.6 * len(labels) + 1.5)
        fig, ax = plt.subplots(figsize=(9, fig_h))
        bars = ax.barh(
            range(len(labels)),
            means,
            xerr=stds,
            color="steelblue",
            edgecolor="black",
            ecolor="black",
            capsize=4,
        )
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title(title)
        ax.set_xlabel("Mean across trials, error bars = +/-1 SD")
        ax.grid(True, axis="x", alpha=0.3)
        xmax = max((m + s) for m, s in zip(means, stds)) if means else 1.0
        ax.set_xlim(0.0, max(1.0, xmax * 1.15))
        for idx, bar in enumerate(bars):
            ax.text(
                means[idx] + stds[idx] + 0.01 * max(1.0, xmax),
                bar.get_y() + bar.get_height() / 2.0,
                f"{means[idx]:.3g} +/- {stds[idx]:.3g} (n={ns[idx]})",
                va="center",
                fontsize=9,
            )
        fig.tight_layout()
        path = out_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return str(path)

    written: List[str] = []
    err_plot = _save_summary_barplot(
        "ukf_errors.png",
        f"{estimator_label} Evaluation Error Metrics",
        [
            ("kf_def_att_pos_err_rms", "Def->att RMS pos err (m)"),
            ("kf_att_def_pos_err_rms", "Att->def RMS pos err (m)"),
            ("kf_def_att_pos_err_final", "Def->att final pos err (m)"),
            ("kf_att_def_pos_err_final", "Att->def final pos err (m)"),
            ("kf_def_att_pos_err_max", "Def->att max pos err (m)"),
            ("kf_att_def_pos_err_max", "Att->def max pos err (m)"),
        ],
    )
    if err_plot is not None:
        written.append(err_plot)

    stats_plot = _save_summary_barplot(
        "ukf_stats.png",
        f"{estimator_label} Evaluation Estimator Stats",
        [
            ("kf_def_att_meas_innov_sq_mean", "Def->att meas innov sq mean"),
            ("kf_att_def_meas_innov_sq_mean", "Att->def meas innov sq mean"),
            ("kf_def_att_trPpos_mean", "Def->att tr(P_pos) mean"),
            ("kf_att_def_trPpos_mean", "Att->def tr(P_pos) mean"),
            ("kf_def_att_meas_fraction", "Def->att measurement fraction"),
            ("kf_att_def_meas_fraction", "Att->def measurement fraction"),
        ],
    )
    if stats_plot is not None:
        written.append(stats_plot)

    return written


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
    ap.add_argument(
        "--advantage_scale",
        type=float,
        default=None,
        help="Scale the config-derived default radial advantage used for random_shell_advantage "
             "sampling and auto-shell advantage filtering. 1.0 keeps the saved config behavior, "
             "0.0 removes the enforced advantage, and negative values flip the same magnitude "
             "of advantage to the attacker (for example -1.0).",
    )

    # common overrides
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="When using CUDA policy-vs-policy eval, number of rollouts to evaluate in parallel. "
             "If omitted, a conservative default batch size is chosen automatically.",
    )
    ap.add_argument("--def_ckpt_path", default=None)
    ap.add_argument("--att_ckpt_path", default=None)
    ap.add_argument(
        "--attacker_mode",
        default=None,
        choices=["rl", "rule", "train"],
        help="Override cfg['attacker_mode'] for 1v1 policy rollouts. 'train' is treated as 'rl'.",
    )
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
    ap.add_argument(
        "--deterministic",
        nargs="?",
        const=True,
        default=None,
        type=_parse_bool,
        help="Override rl_eval_deterministic. Pass --deterministic, --deterministic true, or --deterministic false.",
    )
    ap.add_argument(
        "--use_kf",
        nargs="?",
        const=True,
        default=None,
        type=_parse_bool,
        help="Override use_kf. Pass --use_kf, --use_kf true, or --use_kf false.",
    )
    ap.add_argument("--estimator_kind", choices=["ukf", "ekf"], default=None,
                    help="Force the enabled estimator kind. Setting this also enables the estimator.")
    ap.add_argument(
        "--kf_action_access",
        choices=["ground_truth", "measured", "none"],
        default=None,
        help="Override ukf.action_access for rollout estimation.",
    )
    ap.add_argument(
        "--kf_action_meas_std",
        default=None,
        type=_parse_scalar_or_vector,
        help="Override ukf.action_meas_std with a scalar or comma-separated x,y,z values.",
    )
    ap.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Override the rollout speed cap used by the velocity controller via cfg['vmax'] "
             "and safety_filter['vmax']. When this speed cap is active, whether from the CLI "
             "or the loaded eval config/run_manifest, Monte Carlo initial velocity "
             "sampling/jitter is projected to satisfy it.",
    )
    ap.add_argument(
        "--velocity_controller_enabled",
        nargs="?",
        const=True,
        default=None,
        type=_parse_bool,
        help="Override safety_filter.enabled. Pass true/false to match notebook single-rollout behavior.",
    )
    ap.add_argument(
        "--velocity_controller_speed",
        type=float,
        default=None,
        help="Legacy alias for --vmax.",
    )
    ap.add_argument("--umax", type=float, default=None, help="Override cfg['umax'] for rollout.")
    ap.add_argument("--arena_radius", type=float, default=None, help="Override spherical arena radius in meters.")
    ap.add_argument("--x0_pos_jitter", type=float, default=None)
    ap.add_argument(
        "--velocity_dispersion_std",
        "--x0_vel_jitter",
        dest="x0_vel_jitter",
        type=float,
        default=None,
        help="Override x0_jitter['vel'] as the per-axis Gaussian standard deviation used for "
             "initial velocity dispersion. --velocity_dispersion_std is the preferred name; "
             "--x0_vel_jitter remains supported.",
    )

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
    ap.add_argument("--save_rollout_error_cases", action="store_true",
                    help="Write rollout_error_cases.csv containing failed trials with their starting states.")

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
    ap.add_argument("--index", "--max_pairs", dest="max_pairs", type=int, default=None,
                    help="If set in cartesian mode, cap total evaluated pairs (sampled). "
                         "--index is the preferred name; --max_pairs remains supported.")

    # NEW: auto shell grid (no CSV)
    ap.add_argument("--auto_shell_grid", action="store_true",
                    help="Generate discrete defender/attacker start grids on spherical shells (no CSV).")
    ap.add_argument("--shell_fracs", type=str, default="0.2,0.4,0.6,0.8",
                    help="Comma-separated shell radii as fractions of each agent's shell planning radius (0,1].")
    ap.add_argument("--def_shell_radius", type=float, default=None,
                    help="Defender shell planning radius. If omitted, uses arena radius. Values in (0,1] are treated as fractions of arena radius; larger values are meters.")
    ap.add_argument("--att_shell_radius", type=float, default=None,
                    help="Attacker shell planning radius. If omitted, uses arena radius. Values in (0,1] are treated as fractions of arena radius; larger values are meters.")
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

    # Apply overrides
    if args.device is not None:
        cfg0["device"] = args.device

    _apply_ckpt_overrides(cfg0, args.def_ckpt_path, args.att_ckpt_path)
    if args.attacker_mode is not None:
        cfg0["attacker_mode"] = _normalize_attacker_mode(args.attacker_mode)

    cfg0["attacker_mode"] = _normalize_attacker_mode(cfg0.get("attacker_mode", "rl"))

    if args.deterministic is not None:
        cfg0["rl_eval_deterministic"] = bool(args.deterministic)
    if args.use_kf is not None:
        cfg0["use_kf"] = bool(args.use_kf)
    if args.estimator_kind is not None:
        if args.use_kf is False:
            raise RuntimeError("--estimator_kind cannot be combined with --use_kf false.")
        cfg0["use_kf"] = True
        cfg0["estimator_kind"] = str(args.estimator_kind)
    if args.kf_action_access is not None:
        cfg0.setdefault("ukf", {})
        cfg0["ukf"]["action_access"] = str(args.kf_action_access)
    if args.kf_action_meas_std is not None:
        cfg0.setdefault("ukf", {})
        cfg0["ukf"]["action_meas_std"] = args.kf_action_meas_std
    if args.vmax is not None and args.velocity_controller_speed is not None:
        if not np.isclose(float(args.vmax), float(args.velocity_controller_speed)):
            raise RuntimeError(
                "--vmax and --velocity_controller_speed were both provided with different values. "
                "Use just --vmax, or pass matching values."
            )
    vmax_override = args.vmax if args.vmax is not None else args.velocity_controller_speed
    if vmax_override is not None:
        vmax_override = float(vmax_override)
        if not np.isfinite(vmax_override) or vmax_override <= 0.0:
            raise RuntimeError(f"--vmax must be finite and > 0, got {vmax_override!r}.")
        cfg0["vmax"] = vmax_override
        cfg0.setdefault("safety_filter", {})
        cfg0["safety_filter"]["vmax"] = vmax_override
    if args.velocity_controller_enabled is not None:
        cfg0.setdefault("safety_filter", {})
        cfg0["safety_filter"]["enabled"] = bool(args.velocity_controller_enabled)
    if args.umax is not None:
        cfg0["umax"] = float(args.umax)
    if args.arena_radius is not None:
        cfg0.setdefault("arena", {})
        cfg0["arena"]["type"] = "sphere"
        cfg0["arena"]["r"] = float(args.arena_radius)

    if args.x0_pos_jitter is not None or args.x0_vel_jitter is not None:
        cfg0.setdefault("x0_jitter", {})
        if args.x0_pos_jitter is not None:
            cfg0["x0_jitter"]["pos"] = float(args.x0_pos_jitter)
        if args.x0_vel_jitter is not None:
            cfg0["x0_jitter"]["vel"] = float(args.x0_vel_jitter)

    if args.steps is not None:
        cfg0["T"] = int(args.steps)
    if args.advantage_scale is not None:
        cfg0["advantage_scale"] = float(args.advantage_scale)

    _validate_eval_inputs(args, cfg0)

    D = int(cfg0.get("D", 3))
    num_attackers = max(1, int(cfg0.get("num_attackers", 1)))
    if num_attackers != 1:
        raise RuntimeError(
            "evaluate_policy.py now supports only num_attackers=1 after the multi-attacker path removal."
        )
    dyn_name = str(cfg0.get("dynamics", "hcw"))
    use_kf = bool(cfg0.get("use_kf", False))
    estimator_kind = str(cfg0.get("estimator_kind", "ukf"))
    log(f"[eval] cfg summary: D={D} num_attackers={num_attackers} dynamics={dyn_name}")
    log(f"[eval] ckpts: def={cfg0.get('def_ckpt_path', None)} att={cfg0.get('att_ckpt_path', None)} "
        f"device={cfg0.get('device', None)} deterministic={cfg0.get('rl_eval_deterministic', None)} "
        f"use_kf={use_kf} estimator_kind={estimator_kind}")
    log(f"[eval] matchup: policy_role={args.policy_role} opponent_source={args.opponent_source}")
    log(
        "[eval] rollout knobs: "
        f"attacker_mode={cfg0.get('attacker_mode', 'rl')} "
        f"advantage_scale={_radial_advantage_scale(cfg0):.3f} "
        f"radial_advantage_effective_m={_radial_advantage_margin(cfg0):.3f} "
        f"kf_action_access={(cfg0.get('ukf', {}) or {}).get('action_access')} "
        f"kf_action_meas_std={(cfg0.get('ukf', {}) or {}).get('action_meas_std')} "
        f"velocity_ctrl={(cfg0.get('safety_filter', {}) or {}).get('enabled')} "
        f"vmax={_resolve_rollout_vmax(cfg0)} "
        f"umax={cfg0.get('umax')} arena_r={(cfg0.get('arena', {}) or {}).get('r')}"
    )

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
    def_shell_plan_radius: Optional[float] = None
    att_shell_plan_radius: Optional[float] = None

    n_total: int

    # 3) AUTO SHELL GRID (no CSV)
    if args.auto_shell_grid:
        if D != 3:
            raise RuntimeError("--auto_shell_grid currently supports only D=3 (sphere).")

        center, arena_r = _get_center_and_radius(cfg0, D)
        def_shell_plan_radius = _resolve_shell_plan_radius(
            args.def_shell_radius,
            arena_r,
            "--def_shell_radius",
        )
        att_shell_plan_radius = _resolve_shell_plan_radius(
            args.att_shell_radius,
            arena_r,
            "--att_shell_radius",
        )

        shell_fracs = [float(s.strip()) for s in args.shell_fracs.split(",") if s.strip() != ""]
        shell_fracs = [s for s in shell_fracs if 0.0 < s <= 1.0]
        if not shell_fracs:
            raise RuntimeError("No valid --shell_fracs provided (must be in (0,1]).")

        rng_grid = np.random.default_rng(int(args.seed) + 12345)

        def_pos = _generate_shelled_positions(center, def_shell_plan_radius, shell_fracs, int(args.points_per_shell),
                                              rng=rng_grid, include_center=bool(args.include_center))
        att_pos = _generate_shelled_positions(center, att_shell_plan_radius, shell_fracs, int(args.points_per_shell),
                                              rng=rng_grid, include_center=bool(args.include_center))

        def_rows = _rows_from_positions("def", def_pos)
        att_rows = _rows_from_positions("att1", att_pos)
        def_shell_summary = _format_shell_radius_counts(def_rows, "def", D, center)
        att_shell_summary = _format_shell_radius_counts(att_rows, "att1", D, center)
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
        valid_shell_pair_summary = _format_valid_shell_pair_counts(
            cfg0,
            def_rows,
            att_rows,
            valid_pair_indices,
        )
        log(f"[eval] auto_shell defender radii/counts: {def_shell_summary}")
        log(f"[eval] auto_shell attacker radii/counts: {att_shell_summary}")
        log(
            "[eval] auto_shell planning radii: "
            f"arena={arena_r:g} defender={def_shell_plan_radius:g} attacker={att_shell_plan_radius:g}"
        )
        log(
            "[eval] auto_shell valid shell-pair counts: "
            f"{valid_shell_pair_summary} | total_valid_pairs={total_pairs_valid}"
        )

        # Now choose how to pair them
        if args.grid_mode == "paired":
            n_total = min(int(args.num_trials), total_pairs_valid)
            paired_rows = _paired_rows_from_pair_indices(def_rows, att_rows, valid_pair_indices, n_total, D, num_attackers)
            log(f"[eval] auto_shell_grid paired: shells={shell_fracs} points_per_shell={args.points_per_shell} "
                f"include_center={args.include_center} def_shell_radius={def_shell_plan_radius:g} "
                f"att_shell_radius={att_shell_plan_radius:g} "
                f"valid_pairs={total_pairs_valid}/{total_pairs_all} -> paired_rows={len(paired_rows)}")
        else:
            if args.max_pairs is None:
                n_total = total_pairs_valid
                log(f"[eval] auto_shell_grid cartesian: evaluating ALL valid pairs = {total_pairs_valid} / {total_pairs_all}")
            else:
                n_total = min(int(args.max_pairs), total_pairs_valid)
                log(f"[eval] auto_shell_grid cartesian: valid pairs={total_pairs_valid} / {total_pairs_all}, evaluating={n_total} (sampled)")
            log(f"[eval] auto_shell_grid cartesian grids: def_rows={len(def_rows)} att_rows={len(att_rows)} "
                f"def_shell_radius={def_shell_plan_radius:g} att_shell_radius={att_shell_plan_radius:g}")

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
    starts_atts: List[List[np.ndarray]] = [[]]

    trial_rows: List[Dict[str, Any]] = []
    trajectory_records: List[Tuple[str, np.ndarray, np.ndarray]] = []
    passes = 0

    use_batched_cuda_eval = _supports_batched_cuda_eval(cfg0, args.opponent_source)
    if use_batched_cuda_eval:
        eval_batch_size = int(args.eval_batch_size) if args.eval_batch_size is not None else min(128, max(1, n_total))
        if eval_batch_size <= 0:
            raise RuntimeError(f"--eval_batch_size must be >= 1, got {eval_batch_size}.")
        log(
            f"[eval] batched CUDA eval enabled: batch_size={eval_batch_size} "
            f"(device={cfg0.get('device')}, opponent_source={args.opponent_source})"
        )
        log(
            "[eval] CUDA reproducibility guards enabled: TF32 off, deterministic torch kernels "
            "requested, and per-episode torch RNG streams seeded."
        )
    else:
        eval_batch_size = 1
        if _is_cuda_device(cfg0.get("device", None)) and str(args.opponent_source).strip().lower() == "policy":
            log(
                "[eval] CUDA requested, but batched eval is not active for this configuration; "
                "falling back to scalar rollouts."
            )

    log("[eval] beginning trials...")
    t_loop0 = time.time()
    first_out_logged = False
    pending_items: List[Dict[str, Any]] = []

    def _maybe_log_first_rollout(i_trial: int, out: Dict[str, Any]) -> None:
        nonlocal first_out_logged
        if first_out_logged or not args.print_first_out_keys:
            return
        first_out_logged = True
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
            return

        p1 = np.asarray(out.get("exec1_xyz", []), dtype=float)
        p2 = np.asarray(out.get("exec2_xyz", []), dtype=float)
        log(f"[eval] first rollout sanity: len(exec1_xyz)={len(p1)} len(exec2_xyz)={len(p2)}")
        if p1.size and p2.size:
            d0 = float(np.linalg.norm(p1[0, :D] - p2[0, :D]))
            dT = float(np.linalg.norm(p1[-1, :D] - p2[-1, :D]))
            log(f"[eval] first rollout sanity: rel_dist start={d0:.3f} end={dT:.3f}")

    def _record_trial_result(item: Dict[str, Any], out: Optional[Dict[str, Any]], error: Optional[Exception]) -> None:
        nonlocal passes

        i_trial = int(item["trial"])
        x0 = np.asarray(item["x0"], dtype=float)
        cfg_run = item["cfg_run"]
        cart_di = item["def_idx"]
        cart_ai = item["att_idx"]

        trial_errors = 0
        trial_metrics: Dict[str, Any]
        trial_def_xyz_down = None
        trial_att_xyz_down = None

        try:
            if error is not None:
                raise error
            assert out is not None
            _maybe_log_first_rollout(i_trial, out)

            rollout_timing = out.get("rollout_timing_sec", {}) or {}
            trial_metrics = _compute_trial_metrics(cfg_run, out)
            trial_metrics.update(_extract_kf_trial_metrics(out, D))
            rollout_steps = _rollout_num_steps(out)
            trial_metrics["rollout_num_steps"] = int(rollout_steps)
            p_def_xyz, p_att_xyz = _extract_positions(out, D)
            trial_def_xyz_down = _downsample_xyz_path(p_def_xyz)
            trial_att_xyz_down = _downsample_xyz_path(p_att_xyz)
            for timing_key in ("setup", "simulation", "total"):
                timing_val = float(rollout_timing.get(timing_key, float("nan")))
                trial_metrics[f"rollout_{timing_key}_sec"] = timing_val
                if rollout_steps > 0 and np.isfinite(timing_val):
                    trial_metrics[f"rollout_{timing_key}_sec_per_step"] = timing_val / float(rollout_steps)
                else:
                    trial_metrics[f"rollout_{timing_key}_sec_per_step"] = float("nan")

        except Exception as exc:
            trial_errors = 1
            trial_metrics = {"pass": 0, "error": str(exc)}
            if args.print_errors:
                log(f"[eval][trial {i_trial}/{n_total-1}] ERROR: {exc}")
                if args.trace_errors:
                    log(traceback.format_exc())

        pass_trial = int(trial_metrics.get("pass", 0))
        passes += pass_trial

        row: Dict[str, Any] = {
            "trial": i_trial,
            "seed": int(item["seed"]),
            "grid_mode": args.grid_mode if not args.auto_shell_grid else f"{args.grid_mode}+auto_shell",
            "pass_trial": pass_trial,
            "num_attackers": 1,
            "num_att_errors": trial_errors,
            "def_x": float(x0[0, 0]),
            "def_y": float(x0[0, 1]) if D >= 2 else 0.0,
            "def_z": float(x0[0, 2]) if D == 3 else 0.0,
            "def_vx": float(x0[0, D + 0]) if x0.shape[1] > D else 0.0,
            "def_vy": float(x0[0, D + 1]) if (x0.shape[1] > D + 1 and D >= 2) else 0.0,
            "def_vz": float(x0[0, D + 2]) if (x0.shape[1] > D + 2 and D == 3) else 0.0,
            "att1_x": float(x0[1, 0]) if x0.shape[0] > 1 else float("nan"),
            "att1_y": float(x0[1, 1]) if (x0.shape[0] > 1 and D >= 2) else float("nan"),
            "att1_z": float(x0[1, 2]) if (x0.shape[0] > 1 and D == 3) else float("nan"),
            "att1_vx": float(x0[1, D + 0]) if (x0.shape[0] > 1 and x0.shape[1] > D) else float("nan"),
            "att1_vy": float(x0[1, D + 1]) if (x0.shape[0] > 1 and x0.shape[1] > D + 1 and D >= 2) else float("nan"),
            "att1_vz": float(x0[1, D + 2]) if (x0.shape[0] > 1 and x0.shape[1] > D + 2 and D == 3) else float("nan"),
        }

        if (args.grid_mode == "cartesian") and (def_rows is not None) and (att_rows is not None):
            row["def_idx"] = cart_di
            row["att_idx"] = cart_ai

        row.update(_aggregate_trial_metrics([trial_metrics]))
        for k_, v_ in trial_metrics.items():
            row[f"att1_{k_}"] = v_
        row["outcome_label"] = _classify_trial_outcome(row)

        trial_rows.append(row)
        if trial_def_xyz_down is not None and trial_att_xyz_down is not None:
            trajectory_records.append((str(row["outcome_label"]), trial_def_xyz_down, trial_att_xyz_down))

        if (i_trial == 0) or ((i_trial + 1) % max(1, int(args.log_every)) == 0) or (i_trial == n_total - 1):
            sr_sofar = passes / float(i_trial + 1)
            dt_trial = time.time() - float(item["t_trial0"])
            last_min_rel = row.get("trial_min_rel_dist", row.get("att1_min_rel_dist", None))
            last_hit = row.get("trial_attacker_hit_any", row.get("att1_attacker_hit", None))
            last_def_term = row.get("trial_def_term_any", row.get("att1_def_term", None))
            last_att_term = row.get("trial_att_term_any", row.get("att1_att_term", None))
            last_rollout_sim = row.get("trial_rollout_simulation_sec", row.get("att1_rollout_simulation_sec", None))

            extra = ""
            if args.grid_mode == "cartesian":
                extra = f" | def_idx={row.get('def_idx')} att_idx={row.get('att_idx')}"

            log(
                f"[eval] progress {i_trial+1}/{n_total} | pass_so_far={passes} ({sr_sofar:.3f}) | "
                f"last_pass={pass_trial} | trial_time={dt_trial:.3f}s | "
                f"rollout_sim={last_rollout_sim} | "
                f"min_rel={last_min_rel} hit={last_hit} def_term={last_def_term} att_term={last_att_term} "
                f"errors_this_trial={trial_errors}{extra}"
            )

    def _run_scalar_rollout(cfg_run: Dict[str, Any], steps_run: int) -> Dict[str, Any]:
        if args.opponent_source == "policy":
            return run_rhc_with_rl_and_collect_frames_3d(cfg_run, steps=steps_run)
        return run_rhc_with_policy_vs_baseline_collect_frames_3d(
            cfg_run,
            policy_role=args.policy_role,
            opponent_baseline=args.opponent_source,
            steps=steps_run,
        )

    def _flush_pending_items() -> None:
        nonlocal pending_items
        if not pending_items:
            return

        if use_batched_cuda_eval:
            try:
                batch_episode_seeds = [int(item["ep_seed"]) for item in pending_items]
                set_seed(_combine_reproducibility_seed(batch_episode_seeds))
                batch_outs = run_batched_rhc_with_rl_and_collect_frames_3d(
                    [item["cfg_run"] for item in pending_items],
                    steps=int(pending_items[0]["steps_run"]),
                    episode_seeds=batch_episode_seeds,
                )
                if len(batch_outs) != len(pending_items):
                    raise RuntimeError(
                        f"Batched rollout returned {len(batch_outs)} outputs for {len(pending_items)} trials."
                    )
                for item, out in zip(pending_items, batch_outs):
                    _record_trial_result(item, out, None)
                pending_items = []
                return
            except Exception as batch_exc:
                log(f"[eval] batched CUDA rollout failed; retrying pending trials one-by-one. reason={batch_exc}")

        for item in pending_items:
            try:
                set_seed(int(item["ep_seed"]))
                out = _run_scalar_rollout(item["cfg_run"], int(item["steps_run"]))
                _record_trial_result(item, out, None)
            except Exception as exc:
                _record_trial_result(item, None, exc)
        pending_items = []

    for i in range(n_total):
        t_trial0 = time.time()
        seed = int(args.seed + i)

        set_seed(seed)
        rng = np.random.default_rng(seed)

        cfg_trial, _x0_unused, ep_seed = build_episode_cfg_and_x0(
            cfg0,
            episode_idx=i,
            trials_row=None,
        )

        if args.dynamics is not None:
            cfg_trial["dynamics"] = str(args.dynamics)
        if args.dt is not None:
            cfg_trial["dt"] = float(args.dt)

        set_seed(int(ep_seed))

        if "T" in cfg_trial and cfg_trial["T"] is not None:
            cfg_trial["T"] = int(cfg_trial["T"])

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
                    pair_idx = int(rng.integers(0, total_pairs))
                di = pair_idx // len(att_rows)
                ai = pair_idx % len(att_rows)

            cart_di, cart_ai = int(di), int(ai)
            rd = def_rows[di]
            ra = att_rows[ai]
            x0 = _build_cartesian_x0(rd, ra, D, num_attackers)

        elif args.sample_ic:
            x0 = _sample_x0(
                cfg_trial,
                rng,
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
        active_vel_jitter = float((cfg_trial.get("x0_jitter", {}) or {}).get("vel", 0.0))
        active_rollout_vmax = _resolve_rollout_vmax(cfg_trial)
        if active_rollout_vmax is not None and (
            args.sample_ic or args.auto_shell_grid or active_vel_jitter > 0.0
        ):
            x0 = _project_x0_velocities_to_vmax(x0, D, float(active_rollout_vmax))

        starts_def.append(x0[0, :D].copy())
        for j in range(num_attackers):
            if 1 + j < x0.shape[0]:
                starts_atts[j].append(x0[1 + j, :D].copy())

        cfg_run = copy.deepcopy(cfg_trial)
        cfg_run["x0"] = np.asarray(x0, dtype=float).tolist()
        cfg_run["seed"] = int(ep_seed)
        _apply_ckpt_overrides(cfg_run, args.def_ckpt_path, args.att_ckpt_path)

        steps_run = int(args.steps) if args.steps is not None else int(cfg_run.get("T", cfg0.get("T", 0)) or 0)
        if steps_run <= 0:
            raise RuntimeError(f"Invalid steps_run={steps_run}. Provide --steps or ensure cfg['T'] is set.")
        cfg_run["T"] = steps_run

        if pending_items and int(pending_items[0]["steps_run"]) != steps_run:
            _flush_pending_items()

        pending_items.append(
            {
                "trial": i,
                "seed": seed,
                "x0": np.asarray(x0, dtype=float),
                "cfg_run": cfg_run,
                "steps_run": steps_run,
                "def_idx": cart_di,
                "att_idx": cart_ai,
                "ep_seed": int(ep_seed),
                "t_trial0": t_trial0,
            }
        )

        if (not use_batched_cuda_eval) or (len(pending_items) >= eval_batch_size):
            _flush_pending_items()

    _flush_pending_items()

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



    outcome_breakdown = _save_outcome_histogram(out_dir, trial_rows, alpha=float(args.alpha))

    def _metric_summary(name: str) -> Dict[str, float]:
        vals = _col(name)
        return _numeric_summary_from_values(vals)


    def _rate_value_and_stats(name: str) -> Tuple[float, Dict[str, Any]]:
        stats = _binary_summary_from_values(_col(name), alpha=float(args.alpha))
        return float(stats["mean"]), stats


    def _event_time_summary(name: str) -> Dict[str, float]:
        return _event_time_summary_from_values(_col(name))


    def _attacker_metric_block(prefix: str) -> Dict[str, Any]:
        attacker_hit_rate, attacker_hit_rate_stats = _rate_value_and_stats(f"{prefix}_attacker_hit")
        collision_rate, collision_rate_stats = _rate_value_and_stats(f"{prefix}_collided")
        att_term_rate, att_term_rate_stats = _rate_value_and_stats(f"{prefix}_att_term")
        def_term_rate, def_term_rate_stats = _rate_value_and_stats(f"{prefix}_def_term")
        oi_viol_def_rate, oi_viol_def_rate_stats = _rate_value_and_stats(f"{prefix}_oi_viol_def")
        oi_viol_att_rate, oi_viol_att_rate_stats = _rate_value_and_stats(f"{prefix}_oi_viol_att")
        return {
            "min_rel_dist": _metric_summary(f"{prefix}_min_rel_dist"),
            "min_att_center": _metric_summary(f"{prefix}_min_att_center"),
            "attacker_hit_rate": attacker_hit_rate,
            "attacker_hit_rate_stats": attacker_hit_rate_stats,
            "collision_rate": collision_rate,
            "collision_rate_stats": collision_rate_stats,
            "att_term_rate": att_term_rate,
            "att_term_rate_stats": att_term_rate_stats,
            "def_term_rate": def_term_rate,
            "def_term_rate_stats": def_term_rate_stats,
            "oi_viol_def_rate": oi_viol_def_rate,
            "oi_viol_def_rate_stats": oi_viol_def_rate_stats,
            "oi_viol_att_rate": oi_viol_att_rate,
            "oi_viol_att_rate_stats": oi_viol_att_rate_stats,
            "udef_mean": _metric_summary(f"{prefix}_udef_mean"),
            "uatt_mean": _metric_summary(f"{prefix}_uatt_mean"),
            "delta_v_def_mean": _metric_summary(f"{prefix}_delta_v_def_mean"),
            "delta_v_att_mean": _metric_summary(f"{prefix}_delta_v_att_mean"),
            "delta_v_def_total": _metric_summary(f"{prefix}_delta_v_def_total"),
            "delta_v_att_total": _metric_summary(f"{prefix}_delta_v_att_total"),
            "delta_v_def_max": _metric_summary(f"{prefix}_delta_v_def_max"),
            "delta_v_att_max": _metric_summary(f"{prefix}_delta_v_att_max"),
            "rollout_num_steps": _metric_summary(f"{prefix}_rollout_num_steps"),
            "rollout_setup_sec": _metric_summary(f"{prefix}_rollout_setup_sec"),
            "rollout_setup_sec_per_step": _metric_summary(f"{prefix}_rollout_setup_sec_per_step"),
            "rollout_simulation_sec": _metric_summary(f"{prefix}_rollout_simulation_sec"),
            "rollout_simulation_sec_per_step": _metric_summary(f"{prefix}_rollout_simulation_sec_per_step"),
            "rollout_total_sec": _metric_summary(f"{prefix}_rollout_total_sec"),
            "rollout_total_sec_per_step": _metric_summary(f"{prefix}_rollout_total_sec_per_step"),
            "t_collide_given_collision": _event_time_summary(f"{prefix}_t_collide"),
            "t_att_hit_given_hit": _event_time_summary(f"{prefix}_t_att_hit"),
            "t_att_term_given_att_term": _event_time_summary(f"{prefix}_t_att_term"),
            "t_def_term_given_def_term": _event_time_summary(f"{prefix}_t_def_term"),
            "t_oi_viol_def_given_oi_viol_def": _event_time_summary(f"{prefix}_t_oi_viol_def"),
            "t_oi_viol_att_given_oi_viol_att": _event_time_summary(f"{prefix}_t_oi_viol_att"),
            "kf_enabled_rollout_rate": _rate_value_and_stats(f"{prefix}_kf_enabled_rollout")[0],
            "kf_enabled_rollout_rate_stats": _rate_value_and_stats(f"{prefix}_kf_enabled_rollout")[1],
            "kf_def_att_pos_err_mean": _metric_summary(f"{prefix}_kf_def_att_pos_err_mean"),
            "kf_def_att_pos_err_rms": _metric_summary(f"{prefix}_kf_def_att_pos_err_rms"),
            "kf_def_att_pos_err_max": _metric_summary(f"{prefix}_kf_def_att_pos_err_max"),
            "kf_def_att_pos_err_final": _metric_summary(f"{prefix}_kf_def_att_pos_err_final"),
            "kf_att_def_pos_err_mean": _metric_summary(f"{prefix}_kf_att_def_pos_err_mean"),
            "kf_att_def_pos_err_rms": _metric_summary(f"{prefix}_kf_att_def_pos_err_rms"),
            "kf_att_def_pos_err_max": _metric_summary(f"{prefix}_kf_att_def_pos_err_max"),
            "kf_att_def_pos_err_final": _metric_summary(f"{prefix}_kf_att_def_pos_err_final"),
            "kf_def_att_meas_count": _metric_summary(f"{prefix}_kf_def_att_meas_count"),
            "kf_def_att_meas_fraction": _metric_summary(f"{prefix}_kf_def_att_meas_fraction"),
            "kf_def_att_meas_innov_sq_mean": _metric_summary(f"{prefix}_kf_def_att_meas_innov_sq_mean"),
            "kf_def_att_trPpos_mean": _metric_summary(f"{prefix}_kf_def_att_trPpos_mean"),
            "kf_att_def_meas_count": _metric_summary(f"{prefix}_kf_att_def_meas_count"),
            "kf_att_def_meas_fraction": _metric_summary(f"{prefix}_kf_att_def_meas_fraction"),
            "kf_att_def_meas_innov_sq_mean": _metric_summary(f"{prefix}_kf_att_def_meas_innov_sq_mean"),
            "kf_att_def_trPpos_mean": _metric_summary(f"{prefix}_kf_att_def_trPpos_mean"),
        }


    success_rate_summary = _binary_summary_from_values(np.asarray([float(r.get("pass_trial", 0)) for r in trial_rows], dtype=float), alpha=float(args.alpha))
    trial_attacker_hit_vals = _col("trial_attacker_hit_any")
    attacker_primary_summary = _binary_summary_from_values(trial_attacker_hit_vals, alpha=float(args.alpha))
    trial_attacker_hit_rate, trial_attacker_hit_rate_stats = _rate_value_and_stats("trial_attacker_hit_any")
    trial_collision_any_rate, trial_collision_any_rate_stats = _rate_value_and_stats("trial_collided_any")
    trial_att_term_any_rate, trial_att_term_any_rate_stats = _rate_value_and_stats("trial_att_term_any")
    trial_def_term_any_rate, trial_def_term_any_rate_stats = _rate_value_and_stats("trial_def_term_any")
    trial_oi_viol_def_any_rate, trial_oi_viol_def_any_rate_stats = _rate_value_and_stats("trial_oi_viol_def_any")
    trial_oi_viol_att_any_rate, trial_oi_viol_att_any_rate_stats = _rate_value_and_stats("trial_oi_viol_att_any")
    trial_rollout_error_any_rate, trial_rollout_error_any_rate_stats = _rate_value_and_stats("trial_rollout_error_any")
    trial_kf_enabled_rate, trial_kf_enabled_rate_stats = _rate_value_and_stats("att1_kf_enabled_rollout")
    primary_metric_name = "defender_capture_pass_rate" if args.policy_role == "def" else "attacker_hit_rate"
    primary_metric_value = float(sr) if args.policy_role == "def" else (
        float(np.mean(trial_attacker_hit_vals)) if trial_attacker_hit_vals.size else float("nan")
    )
    rollout_cfg_summary = {
        "num_attackers": 1,
        "attacker_mode": str(cfg0.get("attacker_mode", "rl")),
        "use_kf": bool(cfg0.get("use_kf", False)),
        "estimator_kind": str(cfg0.get("estimator_kind", "ukf")),
        "kf_action_access": (cfg0.get("ukf", {}) or {}).get("action_access"),
        "kf_action_meas_std": (cfg0.get("ukf", {}) or {}).get("action_meas_std"),
        "velocity_controller_enabled": bool((cfg0.get("safety_filter", {}) or {}).get("enabled", False)),
        "velocity_controller_speed": _resolve_rollout_vmax(cfg0),
        "vmax": _resolve_rollout_vmax(cfg0),
        "umax": cfg0.get("umax"),
        "advantage_scale": _radial_advantage_scale(cfg0),
        "radial_advantage_effective_m": _radial_advantage_margin(cfg0),
        "percent_advantage_defender": cfg0.get("percent_advantage_defender"),
        "arena_radius": (cfg0.get("arena", {}) or {}).get("r"),
        "def_shell_radius": float(def_shell_plan_radius) if def_shell_plan_radius is not None else None,
        "att_shell_radius": float(att_shell_plan_radius) if att_shell_plan_radius is not None else None,
        "dynamics": cfg0.get("dynamics"),
        "dt": cfg0.get("dt"),
        "steps": cfg0.get("T"),
        "def_ckpt_path": cfg0.get("def_ckpt_path"),
        "att_ckpt_path": cfg0.get("att_ckpt_path"),
    }

    results = {
        "num_trials": n,
        "passes": k,
        "success_rate": float(sr),
        "success_rate_stderr": float(success_rate_summary["stderr"]),
        "success_rate_summary": success_rate_summary,
        "defender_pass_rate": float(sr),
        "defender_pass_rate_summary": success_rate_summary,
        "policy_role": args.policy_role,
        "opponent_source": args.opponent_source,
        "policy_primary_metric": {
            "name": primary_metric_name,
            "value": primary_metric_value,
            "summary": success_rate_summary if args.policy_role == "def" else attacker_primary_summary,
        },
        "success_rate_ci_wilson": {"alpha": float(args.alpha), "lo": float(lo), "hi": float(hi)},
        "config_module": args.config_module,
        "base_cfg_source": base_cfg_source,
        "run_manifest": str(manifest_path) if manifest_path is not None else None,
        "rollout_cfg": rollout_cfg_summary,
        "grid_mode": args.grid_mode,
        "auto_shell_grid": bool(args.auto_shell_grid),
        "shell_fracs": args.shell_fracs if args.auto_shell_grid else None,
        "def_shell_radius": float(def_shell_plan_radius) if def_shell_plan_radius is not None else None,
        "att_shell_radius": float(att_shell_plan_radius) if att_shell_plan_radius is not None else None,
        "points_per_shell": int(args.points_per_shell) if args.auto_shell_grid else None,
        "include_center": bool(args.include_center) if args.auto_shell_grid else None,
        "save_rollout_error_cases": bool(args.save_rollout_error_cases),
        "timing_sec": {
            "total": float(time.time() - t_global0),
            "trial_loop": float(time.time() - t_loop0),
        },
        "outcome_breakdown": outcome_breakdown,
        "rollout_error_cases_count": int(sum(1 for r in trial_rows if r.get("outcome_label") == "rollout_error")),
        "notes": [
            "Rollouts use the RL runner for policy-vs-policy and the mixed matchup runner for policy-vs-baseline evaluation.",
            "This evaluation harness is now 1v1-only and mirrors the single-rollout notebook workflow.",
            "When zero_sum_reward.mode is enabled and collision_radius_m > 0, PASS means the defender captures the attacker before any attacker hit or invalid termination/keepout event.",
            "Otherwise PASS uses the legacy verifier: attacker does not hit target and no terminations/keepout violations, with optional capture gating via cfg['verify_require_capture'].",
            "grid_mode=paired uses --trials_in rows or --auto_shell_grid pairing.",
            "grid_mode=cartesian uses product of --def_trials_in x --att_trials_in or --auto_shell_grid grids.",
            "If --index/--max_pairs is set in cartesian mode, pairs are sampled (not guaranteed unique).",
            "Delta-v metrics are computed from realized control norms (u_real) times dt, so they reflect executed per-step delta-v and trajectory-total delta-v.",
            "Trajectory overlay plots render all recorded trials with per-trajectory downsampling for plotting efficiency.",
            "When policy_role=att, success_rate remains the defender-centric pass metric; use policy_primary_metric or metrics_trial.attacker_hit_any_rate for attacker-side comparisons.",
        ],
        "metrics_trial": {
            "min_rel_dist": _metric_summary("trial_min_rel_dist"),
            "min_att_center": _metric_summary("trial_min_att_center"),
            "attacker_hit_any_rate": trial_attacker_hit_rate,
            "attacker_hit_any_rate_stats": trial_attacker_hit_rate_stats,
            "collision_any_rate": trial_collision_any_rate,
            "collision_any_rate_stats": trial_collision_any_rate_stats,
            "att_term_any_rate": trial_att_term_any_rate,
            "att_term_any_rate_stats": trial_att_term_any_rate_stats,
            "def_term_any_rate": trial_def_term_any_rate,
            "def_term_any_rate_stats": trial_def_term_any_rate_stats,
            "oi_viol_def_any_rate": trial_oi_viol_def_any_rate,
            "oi_viol_def_any_rate_stats": trial_oi_viol_def_any_rate_stats,
            "oi_viol_att_any_rate": trial_oi_viol_att_any_rate,
            "oi_viol_att_any_rate_stats": trial_oi_viol_att_any_rate_stats,
            "rollout_error_any_rate": trial_rollout_error_any_rate,
            "rollout_error_any_rate_stats": trial_rollout_error_any_rate_stats,
            "kf_enabled_rollout_rate": trial_kf_enabled_rate,
            "kf_enabled_rollout_rate_stats": trial_kf_enabled_rate_stats,
            "udef_mean": _metric_summary("trial_udef_mean"),
            "uatt_mean": _metric_summary("trial_uatt_mean"),
            "delta_v_def_mean": _metric_summary("trial_delta_v_def_mean"),
            "delta_v_att_mean": _metric_summary("trial_delta_v_att_mean"),
            "delta_v_def_total": _metric_summary("trial_delta_v_def_total"),
            "delta_v_att_total": _metric_summary("trial_delta_v_att_total"),
            "delta_v_def_max": _metric_summary("trial_delta_v_def_max"),
            "delta_v_att_max": _metric_summary("trial_delta_v_att_max"),
            "rollout_num_steps": _metric_summary("trial_rollout_num_steps"),
            "rollout_setup_sec": _metric_summary("trial_rollout_setup_sec"),
            "rollout_setup_sec_per_step": _metric_summary("trial_rollout_setup_sec_per_step"),
            "rollout_simulation_sec": _metric_summary("trial_rollout_simulation_sec"),
            "rollout_simulation_sec_per_step": _metric_summary("trial_rollout_simulation_sec_per_step"),
            "rollout_total_sec": _metric_summary("trial_rollout_total_sec"),
            "rollout_total_sec_per_step": _metric_summary("trial_rollout_total_sec_per_step"),
            "t_collide_given_collision": _event_time_summary("trial_t_collide"),
            "t_att_hit_given_hit": _event_time_summary("trial_t_att_hit"),
            "t_att_term_given_att_term": _event_time_summary("trial_t_att_term"),
            "t_def_term_given_def_term": _event_time_summary("trial_t_def_term"),
            "t_oi_viol_def_given_oi_viol_def": _event_time_summary("trial_t_oi_viol_def"),
            "t_oi_viol_att_given_oi_viol_att": _event_time_summary("trial_t_oi_viol_att"),
            "kf_def_att_pos_err_mean": _metric_summary("att1_kf_def_att_pos_err_mean"),
            "kf_def_att_pos_err_rms": _metric_summary("att1_kf_def_att_pos_err_rms"),
            "kf_def_att_pos_err_max": _metric_summary("att1_kf_def_att_pos_err_max"),
            "kf_def_att_pos_err_final": _metric_summary("att1_kf_def_att_pos_err_final"),
            "kf_att_def_pos_err_mean": _metric_summary("att1_kf_att_def_pos_err_mean"),
            "kf_att_def_pos_err_rms": _metric_summary("att1_kf_att_def_pos_err_rms"),
            "kf_att_def_pos_err_max": _metric_summary("att1_kf_att_def_pos_err_max"),
            "kf_att_def_pos_err_final": _metric_summary("att1_kf_att_def_pos_err_final"),
            "kf_def_att_meas_count": _metric_summary("att1_kf_def_att_meas_count"),
            "kf_def_att_meas_fraction": _metric_summary("att1_kf_def_att_meas_fraction"),
            "kf_def_att_meas_innov_sq_mean": _metric_summary("att1_kf_def_att_meas_innov_sq_mean"),
            "kf_def_att_trPpos_mean": _metric_summary("att1_kf_def_att_trPpos_mean"),
            "kf_att_def_meas_count": _metric_summary("att1_kf_att_def_meas_count"),
            "kf_att_def_meas_fraction": _metric_summary("att1_kf_att_def_meas_fraction"),
            "kf_att_def_meas_innov_sq_mean": _metric_summary("att1_kf_att_def_meas_innov_sq_mean"),
            "kf_att_def_trPpos_mean": _metric_summary("att1_kf_att_def_trPpos_mean"),
        },
    }
    results["metrics_att1"] = _attacker_metric_block("att1")


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

    rollout_error_rows = [r for r in trial_rows if r.get("outcome_label") == "rollout_error"]
    rollout_error_csv_path = out_dir / "rollout_error_cases.csv"
    if args.save_rollout_error_cases:
        with rollout_error_csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rollout_error_rows:
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
        alpha=float(args.alpha),
    )

    extra_plot_files: List[str] = []
    success_vs_delta_v_path = _save_success_vs_delta_v_plot(out_dir, trial_rows)
    if success_vs_delta_v_path is not None:
        extra_plot_files.append(success_vs_delta_v_path)
    event_hist_path = _save_time_to_event_histograms(out_dir, trial_rows, float(cfg0.get("dt", float("nan"))))
    if event_hist_path is not None:
        extra_plot_files.append(event_hist_path)
    extra_plot_files.extend(_save_trajectory_overlay_plots(out_dir, cfg0, trajectory_records))

    kf_plot_paths: List[str] = []
    if extra_plot_files:
        results["extra_plot_files"] = extra_plot_files
    if use_kf:
        kf_plot_paths = _save_kf_eval_plots(out_dir, trial_rows, estimator_kind.upper())
        if kf_plot_paths:
            results["kf_plot_files"] = [str(Path(p).name) for p in kf_plot_paths]
    if extra_plot_files or kf_plot_paths:
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    log(f"[eval] DONE | trials={n} passes={k} success_rate={sr:.3f} CI={lo:.3f}..{hi:.3f}")
    log(f"[eval] wrote: {out_dir/'results.json'}")
    log(f"[eval] wrote: {out_dir/'trials.csv'}")
    log(f"[eval] wrote: {out_dir/'timeout_no_capture_cases.csv'} ({len(timeout_rows)} rows)")
    if args.save_rollout_error_cases:
        log(f"[eval] wrote: {out_dir/'rollout_error_cases.csv'} ({len(rollout_error_rows)} rows)")
    log(f"[eval] wrote: {out_dir/'clean_non_hit_non_term_cases.csv'} ({len(clean_resolution_rows)} rows)")
    log(f"[eval] wrote: {out_dir/'starts_xy.png'}")
    log(f"[eval] wrote: {out_dir/'starts_xz.png'}")
    for name in extra_plot_files:
        log(f"[eval] wrote: {out_dir/name}")
    for p in kf_plot_paths:
        log(f"[eval] wrote: {p}")
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

#Policy
"""
  
python evaluate_policy.py \
--def_ckpt_path Training_Policy_0.75/def0_teacher.pt \
--att_ckpt_path Training_Policy_0.75/att1_teacher.pt \
--out_dir Training_Policy_0.75/MC_eval/def0_vs_att1 \
--auto_shell_grid \
--grid_mode cartesian \
--shell_fracs 0.2,0.4,0.6,0.8 \
--points_per_shell 40 \
--x0_vel_jitter 0.5
"""

# KF setting on:

"""
python evaluate_policy.py \
  --run_dir Training_Policy_0.75_EKF_BC_GT \
  --def_ckpt_path Training_Policy_0.75_EKF_BC_GT/def0_teacher.pt \
  --att_ckpt_path Training_Policy_0.75_EKF_BC_GT/att1_teacher.pt \
  --out_dir Training_Policy_0.75_EKF_BC_GT/MC_eval/def0_vs_att1 \
  --auto_shell_grid \
  --grid_mode cartesian \
  --shell_fracs 0.2,0.4,0.6,0.8 \
  --points_per_shell 40 \
  --x0_vel_jitter 0.5
"""

#Rule
"""
python evaluate_policy.py \
  --run_dir Training_Policy_0.75 \
  --def_ckpt_path Training_Policy_0.75/def0_teacher.pt \
  --policy_role def \
  --opponent_source rule \
  --out_dir Training_Policy_0.75/MC_eval/def0_vs_rule_attacker \
  --auto_shell_grid \
  --grid_mode cartesian \
  --shell_fracs 0.2,0.4,0.6,0.8 \
  --points_per_shell 40 \
  --x0_vel_jitter 0.5

"""

"""
python evaluate_policy.py \
  --run_dir Training_Policy_0.75_EKF_BC_GT \
  --def_ckpt_path Training_Policy_0.75_EKF_BC_GT/def0_teacher.pt \
  --policy_role def \
  --opponent_source rule \
  --out_dir Training_Policy_0.75_EKF_BC_GT/MC_eval/def0_vs_rule_attacker \
  --auto_shell_grid \
  --grid_mode cartesian \
  --shell_fracs 0.2,0.4,0.6,0.8 \
  --points_per_shell 40 \
  --x0_vel_jitter 0.5
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
  --index 5000
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


# Final tests:

"""
python evaluate_policy.py \
  --run_dir Training_Policy_Redo_OG_Actuation \
  --def_ckpt_path Training_Policy_Redo_OG_Actuation/def1_teacher.pt \
  --att_ckpt_path Training_Policy_Redo_OG_Actuation/att1_teacher.pt \
  --out_dir Training_Policy_Redo_OG_Actuation/MC_eval/def1_vs_att1 \
  --auto_shell_grid \
  --grid_mode cartesian \
  --shell_fracs 0.2,0.4,0.6,0.8 \
  --arena_radius 25.0 \

  --points_per_shell 40 \
  --x0_vel_jitter 0.5
"""


#20 meter

"""
python evaluate_policy.py \
  --run_dir Training_Policy_Redo_Mid_Actuation \
  --def_ckpt_path Training_Policy_Redo_Mid_Actuation/def1_teacher.pt \
  --att_ckpt_path Training_Policy_Redo_Mid_Actuation/att1_teacher.pt \
  --out_dir Training_Policy_Redo_Mid_Actuation/MC_eval_20m/def1_vs_att1 \
  --auto_shell_grid \
  --grid_mode cartesian \
  --shell_fracs 0.2,0.4,0.6,0.8 \
  --arena_radius 20.0 \
  --points_per_shell 40 \
  --x0_vel_jitter 0.5 \
  --umax 0.5 \
  --velocity_controller_enabled true \
  --velocity_controller_speed 1.0
"""

# 100 meter

"""
python evaluate_policy.py \
  --run_dir Training_Policy_Redo_Mid_Actuation \
  --def_ckpt_path Training_Policy_Redo_Mid_Actuation/def1_teacher.pt \
  --att_ckpt_path Training_Policy_Redo_Mid_Actuation/att1_teacher.pt \
  --out_dir Training_Policy_Redo_Mid_Actuation/MC_eval_100m/def1_vs_att1 \
  --auto_shell_grid \
  --grid_mode cartesian \
  --shell_fracs 0.2,0.4,0.6,0.8 \
  --arena_radius 100.0 \
  --points_per_shell 40 \
  --umax 0.5 \
  --velocity_controller_enabled true \
  --vmax 1.0
  --advantage_scale 1.0
  --save_rollout_error_cases

"""



#OG actuation

"""
python evaluate_policy.py \
  --run_dir Training_Policy_Redo_OG_Actuation \
  --def_ckpt_path Training_Policy_Redo_OG_Actuation/def1_teacher.pt \
  --att_ckpt_path Training_Policy_Redo_OG_Actuation/att1_teacher.pt \
  --out_dir Training_Policy_Redo_OG_Actuation/MC_eval_100m/def1_vs_att1 \
  --auto_shell_grid \
  --grid_mode cartesian \
  --shell_fracs 0.2,0.4,0.6,0.8 \
  --arena_radius 100.0 \
  --points_per_shell 40 \
  --velocity_controller_enabled true \
  --vmax 1.0
  --advantage_scale 1.0
  --save_rollout_error_cases

"""

"""
python evaluate_policy.py \
  --run_dir Training_Policy_Redo_Redox_Actuation \
  --def_ckpt_path Training_Policy_Redo_Redox_Actuation/def1_teacher.pt \
  --att_ckpt_path Training_Policy_Redo_Redox_Actuation/att1_teacher.pt \
  --out_dir Training_Policy_Redo_Redox_Actuation/MC_eval_100m/def1_vs_att1 \
  --auto_shell_grid \
  --grid_mode cartesian \
  --shell_fracs 0.2,0.4,0.6,0.8 \
  --arena_radius 100.0 \
  --points_per_shell 40 \
  --velocity_controller_enabled true \
  --vmax 1.0
  --advantage_scale 1.0
  --save_rollout_error_cases

"""
