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

NEW:
  - --grid_mode paired|cartesian
    * paired    : use --trials_in rows as (def,att) pairs (your current behavior)
    * cartesian : use --def_trials_in x --att_trials_in Cartesian product (optionally capped by --max_pairs)
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


# ------------------------- scenario generation -------------------------

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

def _save_start_plots(out_dir: Path, D: int,
                      starts_def: np.ndarray,
                      starts_atts: List[np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(starts_def[:, 0], starts_def[:, 1], marker="o", label="def_start")
    for j, sa in enumerate(starts_atts):
        if sa.size == 0:
            continue
        plt.scatter(sa[:, 0], sa[:, 1], marker="x", label=f"att{j+1}_start")
    plt.xlabel("x"); plt.ylabel("y")
    plt.title("Evaluated Starting Positions (XY)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "starts_xy.png", dpi=180)
    plt.close()

    plt.figure()
    z_def = starts_def[:, 2] if D == 3 else np.zeros(starts_def.shape[0])
    plt.scatter(starts_def[:, 0], z_def, marker="o", label="def_start")
    for j, sa in enumerate(starts_atts):
        if sa.size == 0:
            continue
        z_att = sa[:, 2] if D == 3 else np.zeros(sa.shape[0])
        plt.scatter(sa[:, 0], z_att, marker="x", label=f"att{j+1}_start")
    plt.xlabel("x"); plt.ylabel("z")
    plt.title("Evaluated Starting Positions (XZ)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "starts_xz.png", dpi=180)
    plt.close()


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

    ap.add_argument("--sample_ic", action="store_true",
                    help="If set, ignore cfg['x0'] and sample starts in arena.")
    ap.add_argument("--pos_scale", type=float, default=0.95, help="Sample within pos_scale * arena_r.")
    ap.add_argument("--vel_scale", type=float, default=0.0, help="Stddev of sampled initial velocities.")
    ap.add_argument("--min_sep", type=float, default=0.0, help="Min defender-attacker separation when sampling.")

    ap.add_argument("--multi_att_mode", default="worst",
                    choices=["worst", "avg"],
                    help="If num_attackers>1, evaluate defender vs each attacker separately then aggregate.")

    ap.add_argument("--device", default=None)
    ap.add_argument("--def_ckpt_path", default=None)
    ap.add_argument("--att_ckpt_path", default=None)
    ap.add_argument("--deterministic", action="store_true", help="Force rl_eval_deterministic=True")
    ap.add_argument("--use_ukf", action="store_true", help="Force use_ukf=True")
    ap.add_argument("--x0_pos_jitter", type=float, default=None)
    ap.add_argument("--x0_vel_jitter", type=float, default=None)

    ap.add_argument("--alpha", type=float, default=0.05, help="CI alpha (0.05 => 95% CI).")

    # ---- progress / debugging flags ----
    ap.add_argument("--log_every", type=int, default=10,
                    help="Print progress every N trials (default: 10).")
    ap.add_argument("--print_first_out_keys", action="store_true",
                    help="Print keys and a couple quick sanity checks from the first rollout.")
    ap.add_argument("--print_errors", action="store_true",
                    help="Print exception messages when a trial fails.")
    ap.add_argument("--trace_errors", action="store_true",
                    help="Also print full traceback for failed trials (implies --print_errors).")

    # ---- paired CSV (existing) ----
    ap.add_argument("--trials_in", default=None,
                    help="CSV of paired start states (def+att). Overrides cfg['x0'] and --sample_ic (paired mode).")

    # ---- NEW: grid mode ----
    ap.add_argument("--grid_mode", default="paired", choices=["paired", "cartesian"],
                    help="paired: use --trials_in as (def,att) rows. "
                         "cartesian: use --def_trials_in x --att_trials_in product.")
    ap.add_argument("--def_trials_in", default=None,
                    help="CSV of defender start states (def_x,def_y,def_z[,def_vx,def_vy,def_vz]).")
    ap.add_argument("--att_trials_in", default=None,
                    help="CSV of attacker start states (att1_x,att1_y,att1_z[,att1_vx,att1_vy,att1_vz]).")
    ap.add_argument("--max_pairs", type=int, default=None,
                    help="If set in cartesian mode, cap total evaluated pairs (sampled).")

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
    num_attackers = int(cfg0.get("num_attackers", 1))
    num_attackers = max(1, num_attackers)

    dyn_name = str(cfg0.get("dynamics", "hcw"))
    log(f"[eval] cfg summary: D={D} num_attackers={num_attackers} dynamics={dyn_name}")
    log(f"[eval] ckpts: def={cfg0.get('def_ckpt_path', None)} att={cfg0.get('att_ckpt_path', None)} "
        f"device={cfg0.get('device', None)} deterministic={cfg0.get('rl_eval_deterministic', None)} use_ukf={cfg0.get('use_ukf', False)}")

    # ---------------------------
    # Load trial lists for paired/cartesian
    # ---------------------------
    paired_rows: Optional[List[Dict[str, float]]] = None
    def_rows: Optional[List[Dict[str, float]]] = None
    att_rows: Optional[List[Dict[str, float]]] = None

    if args.grid_mode == "paired":
        if args.trials_in is not None:
            paired_rows = _load_trials_csv(args.trials_in, D)
            log(f"[eval] loaded paired trials_in: {args.trials_in} ({len(paired_rows)} rows)")
            if len(paired_rows) < int(args.num_trials):
                raise RuntimeError(
                    f"--trials_in has {len(paired_rows)} rows but --num_trials={args.num_trials}. "
                    "Either reduce --num_trials or provide more rows."
                )
        log(f"[eval] paired mode: num_trials={args.num_trials}")

        n_total = int(args.num_trials)

    elif args.grid_mode == "cartesian":
        if args.def_trials_in is None or args.att_trials_in is None:
            raise RuntimeError("cartesian mode requires --def_trials_in and --att_trials_in")

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

    else:
        raise RuntimeError(f"Unknown grid_mode={args.grid_mode}")

    # If neither paired csv nor cartesian csv is used, user may sample_ic or cfg['x0']
    if args.grid_mode == "paired" and paired_rows is None and (not args.sample_ic):
        log("[eval] using cfg['x0'] (no sampling).")
    if args.sample_ic:
        log(f"[eval] sampling ICs: pos_scale={args.pos_scale} vel_scale={args.vel_scale} min_sep={args.min_sep}")

    # Build dyn
    log("[eval] building dynamics via build_dyn(cfg)...")
    t_dyn0 = time.time()
    mod.build_dyn(cfg0)
    log(f"[eval] build_dyn done in {time.time() - t_dyn0:.3f}s. dyn keys={list((cfg0.get('dyn', {}) or {}).keys())}")

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
        np.random.seed(seed)  # runner uses np.random.randn for measurement noise
        rng = np.random.default_rng(seed)

        # Let your dispersion module mutate cfg per episode (kept)
        cfg_trial, x0_unused, ep_seed = build_episode_cfg_and_x0(
            cfg0,
            episode_idx=i,
            trials_row=None,  # we handle our own grids below
        )
        np.random.seed(ep_seed)

        # ---------------------------
        # Build x0
        # ---------------------------
        cart_di = cart_ai = None

        if args.grid_mode == "paired" and paired_rows is not None:
            r = paired_rows[i]
            p_def = np.array([r["def_x"], r["def_y"], r["def_z"]], dtype=float)
            v_def = np.array([r.get("def_vx", 0.0), r.get("def_vy", 0.0), r.get("def_vz", 0.0)], dtype=float)

            p_att = np.array([r["att1_x"], r["att1_y"], r["att1_z"]], dtype=float)
            v_att = np.array([r.get("att1_vx", 0.0), r.get("att1_vy", 0.0), r.get("att1_vz", 0.0)], dtype=float)

            x0 = np.stack([np.concatenate([p_def, v_def]),
                           np.concatenate([p_att, v_att])], axis=0)

        elif args.grid_mode == "cartesian":
            assert def_rows is not None and att_rows is not None
            total_pairs = len(def_rows) * len(att_rows)

            if args.max_pairs is None:
                pair_idx = i
            else:
                # cheap sampling; not guaranteeing uniqueness (fine for MC)
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

        # For multi-attackers, you evaluate def vs each attacker separately
        for j in range(num_attackers):
            if 1 + j >= x0.shape[0]:
                continue

            cfg_run = copy.deepcopy(cfg_trial)
            cfg_run["x0"] = np.asarray([x0[0], x0[1 + j]], dtype=float).tolist()

            # force CLI ckpt paths into the rollout cfg
            _apply_ckpt_overrides(cfg_run, args.def_ckpt_path, args.att_ckpt_path)

            try:
                out = run_rhc_with_rl_and_collect_frames_3d_diff(cfg_run, steps=args.steps)

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

        # Write one row per trial (stores metrics for attacker #1 only by default, consistent with your current file)
        row: Dict[str, Any] = {
            "trial": i,
            "seed": seed,
            "grid_mode": args.grid_mode,
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

        if args.grid_mode == "cartesian":
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
            "grid_mode=paired uses --trials_in rows; grid_mode=cartesian uses product of --def_trials_in x --att_trials_in.",
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
    _save_start_plots(out_dir, D, starts_def_arr, starts_att_arrs)

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