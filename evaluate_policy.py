#!/usr/bin/env python3
"""
evaluate_policy.py

Statistical verification harness for your Diff-Nash RL rollout runner:

  - game_runner_diff.run_rhc_with_rl_and_collect_frames_3d_diff(cfg, steps=...)

This version is explicitly aligned with your *current* game_runner_diff.py
(the one that:
  - uses RLPolicyDiff
  - supports HCW (LTI), elliptic LTV, and two-body nonlinear dynamics
  - optionally uses HCW-only UKF via cfg["use_ukf"]
  - logs commanded thrust via out["u_cmd_norm_all"], out["u_cmd_all"]
  - returns exec1_xyz/exec2_xyz etc. with attitude stubs)

Outputs:
  - results.json : aggregate stats + Wilson CI on pass rate
  - trials.csv   : per-trial metrics + seed + initial conditions
  - starts_xy.png, starts_xz.png : start position coverage (def + attacker(s))
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
from typing import Any, Dict, List, Tuple

import numpy as np

# IMPORTANT: this must match your CURRENT runner file
from game_runner_diff import run_rhc_with_rl_and_collect_frames_3d_diff
from dispersion import build_episode_cfg_and_x0

from pathlib import Path

def _apply_ckpt_overrides(cfg, def_path=None, att_path=None):
    if def_path is not None:
        dp = str(Path(def_path))
        # keys various loaders might use
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


def _load_trials_csv(path: str, D: int) -> List[Dict[str, float]]:
    """
    Expect columns:
      def_x,def_y,def_z,(optional def_vx,def_vy,def_vz)
      att1_x,att1_y,att1_z,(optional att1_vx,att1_vy,att1_vz)
    Returns list of dict rows with floats.
    """
    rows: List[Dict[str, float]] = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        required = ["def_x", "def_y", "def_z", "att1_x", "att1_y", "att1_z"]
        for k in required:
            if k not in r.fieldnames:
                raise RuntimeError(f"trials_in CSV missing required column: '{k}'")

        for row in r:
            out: Dict[str, float] = {}
            for k, v in row.items():
                if v is None or v == "":
                    continue
                try:
                    out[k] = float(v)
                except Exception:
                    # ignore non-numeric columns like 'trial' if present
                    pass
            rows.append(out)

    if D != 3:
        raise RuntimeError(f"_load_trials_csv currently expects D=3, got D={D}")
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
    ap.add_argument("--num_trials", type=int, default=200, help="Number of trials.")
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
    
    ap.add_argument("--trials_in", default=None,
                help="CSV of start states (def/att pos/vel). Overrides cfg['x0'] and --sample_ic.")


    args = ap.parse_args()
    if args.trace_errors:
        args.print_errors = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_global0 = time.time()
    log(f"[eval] starting; out_dir={out_dir.resolve()}")
    log(f"[eval] config_module={args.config_module} num_trials={args.num_trials} seed_base={args.seed}")
    if args.steps is not None:
        log(f"[eval] steps override: {args.steps}")

    # Import config module
    log("[eval] importing config module...")
    mod = importlib.import_module(args.config_module)
    if not hasattr(mod, "config_for_eval") or not hasattr(mod, "build_dyn"):
        raise RuntimeError(f"Module '{args.config_module}' must define config_for_eval(...) and build_dyn(cfg).")
    log("[eval] config module imported OK.")

    cfg0: Dict[str, Any] = mod.config_for_eval()
    log("[eval] base cfg created from config_for_eval().")

    trials_in_rows = None
    if args.trials_in is not None:
        trials_in_rows = _load_trials_csv(args.trials_in, int(cfg0.get("D", 3)))
        log(f"[eval] loaded trials_in: {args.trials_in} ({len(trials_in_rows)} rows)")
        if len(trials_in_rows) < args.num_trials:
            raise RuntimeError(
                f"--trials_in has {len(trials_in_rows)} rows but --num_trials={args.num_trials}. "
                "Either reduce --num_trials or generate a bigger CSV."
            )


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
    if num_attackers < 1:
        num_attackers = 1

    dyn_name = str(cfg0.get("dynamics", "hcw"))
    log(f"[eval] cfg summary: D={D} num_attackers={num_attackers} dynamics={dyn_name}")
    log(f"[eval] ckpts: def={cfg0.get('def_ckpt_path', None)} att={cfg0.get('att_ckpt_path', None)} "
        f"device={cfg0.get('device', None)} deterministic={cfg0.get('rl_eval_deterministic', None)} use_ukf={cfg0.get('use_ukf', False)}")
    if args.sample_ic:
        log(f"[eval] sampling ICs: pos_scale={args.pos_scale} vel_scale={args.vel_scale} min_sep={args.min_sep}")
    else:
        log("[eval] using cfg['x0'] (no sampling).")

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
    n_total = int(args.num_trials)

    log("[eval] beginning trials...")
    t_loop0 = time.time()

    for i in range(n_total):
        t_trial0 = time.time()
        seed = int(args.seed + i)
        np.random.seed(seed)  # runner uses np.random.randn for measurement noise
        rng = np.random.default_rng(seed)

        cfg_trial, x0, ep_seed = build_episode_cfg_and_x0(
            cfg0,
            episode_idx=i,
            trials_row=(trials_in_rows[i] if trials_in_rows is not None else None)
        )

        np.random.seed(ep_seed)  # consistent with cfg_trial dispersion seed


        # Build x0
        if trials_in_rows is not None:
            r = trials_in_rows[i]

            p_def = np.array([r["def_x"], r["def_y"], r["def_z"]], dtype=float)
            v_def = np.array([r.get("def_vx", 0.0), r.get("def_vy", 0.0), r.get("def_vz", 0.0)], dtype=float)

            p_att = np.array([r["att1_x"], r["att1_y"], r["att1_z"]], dtype=float)
            v_att = np.array([r.get("att1_vx", 0.0), r.get("att1_vy", 0.0), r.get("att1_vz", 0.0)], dtype=float)

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

        starts_def.append(x0[0, :D].copy())
        for j in range(num_attackers):
            if 1 + j < x0.shape[0]:
                starts_atts[j].append(x0[1 + j, :D].copy())

        per_att_metrics = []
        per_att_errors = 0

        for j in range(num_attackers):
            if 1 + j >= x0.shape[0]:
                continue

            # inside the inner loop, right before calling run_rhc...
            cfg_run = copy.deepcopy(cfg_trial)
            cfg_run["x0"] = np.asarray([x0[0], x0[1 + j]], dtype=float).tolist()

            # force the CLI ckpt paths onto the actual cfg used in the rollout
            _apply_ckpt_overrides(cfg_run, args.def_ckpt_path, args.att_ckpt_path)

            try:
                out = run_rhc_with_rl_and_collect_frames_3d_diff(cfg_run, steps=args.steps)

                if (i == 0 and j == 0 and args.print_first_out_keys):
                    log("[eval] first rollout returned keys:")
                    log("  " + ", ".join(sorted(list(out.keys()))))
                    # quick sanity checks
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

        row: Dict[str, Any] = {
            "trial": i,
            "seed": seed,
            "pass_trial": pass_trial,
            "num_attackers": num_attackers,
            "num_att_errors": per_att_errors,
            "def_x": float(x0[0, 0]),
            "def_y": float(x0[0, 1]) if D >= 2 else 0.0,
            "def_z": float(x0[0, 2]) if D == 3 else 0.0,
            "att1_x": float(x0[1, 0]),
            "att1_y": float(x0[1, 1]) if D >= 2 else 0.0,
            "att1_z": float(x0[1, 2]) if D == 3 else 0.0,
        }

        m0 = per_att_metrics[0]
        for k_, v_ in m0.items():
            row[f"att1_{k_}"] = v_

        trial_rows.append(row)

        # ---- progress print ----
        if (i == 0) or ((i + 1) % max(1, int(args.log_every)) == 0) or (i == n_total - 1):
            sr_sofar = passes / float(i + 1)
            dt_trial = time.time() - t_trial0

            # pick a few “am I alive?” indicators
            last_min_rel = row.get("att1_min_rel_dist", None)
            last_hit = row.get("att1_attacker_hit", None)
            last_def_term = row.get("att1_def_term", None)
            last_att_term = row.get("att1_att_term", None)

            log(f"[eval] progress {i+1}/{n_total} | pass_so_far={passes} ({sr_sofar:.3f}) | "
                f"last_pass={pass_trial} | trial_time={dt_trial:.3f}s | "
                f"min_rel={last_min_rel} hit={last_hit} def_term={last_def_term} att_term={last_att_term} "
                f"errors_this_trial={per_att_errors}")

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
        "multi_att_mode": args.multi_att_mode,
        "timing_sec": {
            "total": float(time.time() - t_global0),
            "trial_loop": float(time.time() - t_loop0),
        },
        "notes": [
            "Rollouts use game_runner_diff.run_rhc_with_rl_and_collect_frames_3d_diff (Diff-Nash RL-only runner).",
            "If num_attackers>1 we run separate episodes def vs att_j and aggregate by --multi_att_mode.",
            "PASS criterion defaults to 'attacker does not hit target and no terminations/keepout violations'.",
            "Set cfg['verify_require_capture']=True to require collision/capture as well.",
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
