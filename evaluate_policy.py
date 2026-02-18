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

Notes:
  - The runner is 1 defender vs 1 attacker per rollout.
    If cfg["num_attackers"] > 1, this harness evaluates defender vs attacker_j
    in separate episodes and aggregates using --multi_att_mode.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

# IMPORTANT: this must match your CURRENT runner file
from game_runner_diff import run_rhc_with_rl_and_collect_frames_3d_diff


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
    """
    game_runner_diff returns:
      out["exec1_xyz"] : list of (x,y,z) tuples
      out["exec2_xyz"] : list of (x,y,z) tuples
    We'll slice to D dims for computations.
    """
    p1 = np.asarray(out["exec1_xyz"], dtype=float)
    p2 = np.asarray(out["exec2_xyz"], dtype=float)
    p1 = p1[:, :max(3, D)]  # keep 3 if present, safe for plotting/compat
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

    # Collision / capture
    collision_r = float(cfg.get("collision_radius_m", 0.0))
    cap_idx = np.where(rel <= collision_r)[0] if collision_r > 0 else np.array([], dtype=int)
    collided = bool(cap_idx.size > 0)
    t_collide = int(cap_idx[0]) if collided else -1

    # Attacker target-hit (fraction-of-R semantics)
    att_hit_val = float(cfg.get("att_target_hit_radius", 0.0))
    att_hit_r = _radius_from_cfg(att_hit_val, arena_r)
    hit_idx = np.where(att_center <= att_hit_r)[0] if att_hit_r > 0 else np.array([], dtype=int)
    attacker_hit = bool(hit_idx.size > 0)
    t_att_hit = int(hit_idx[0]) if attacker_hit else -1

    # Arena termination / oob
    term_margin = float(cfg.get("arena_terminate_margin", 0.0))
    att_oob = bool(np.any(att_center > arena_r))
    def_oob = bool(np.any(def_center > arena_r))
    att_term = bool(np.any(att_center > arena_r + term_margin))
    def_term = bool(np.any(def_center > arena_r + term_margin))

    # Keep-out object (oi) violations
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
        # Convention ambiguity: be conservative and check both flags if present.
        if 0 in avoid_by:
            oi_viol_def = bool(np.any(def_center <= oi_safe_r))
        if 1 in avoid_by:
            oi_viol_att = bool(np.any(att_center <= oi_safe_r))

    # Thrust norms (post-clip) logged by your runner
    u_norms = out.get("u_cmd_norm_all", None)
    udef_mean = uatt_mean = udef_max = uatt_max = float("nan")
    if u_norms is not None and len(u_norms) >= 2:
        udef = np.asarray(u_norms[0], dtype=float)
        uatt = np.asarray(u_norms[1], dtype=float)
        if udef.size:
            udef_mean, udef_max = float(np.mean(udef)), float(np.max(udef))
        if uatt.size:
            uatt_mean, uatt_max = float(np.mean(uatt)), float(np.max(uatt))

    # Default verification objective for defender policy readiness:
    # PASS if attacker does NOT hit target AND no terminations AND no keepout violation
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

    # XY
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

    # XZ
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_module", default="config_rl",
                    help="Python module containing config_for_eval and build_dyn (default: config_rl).")
    ap.add_argument("--out_dir", default="eval_out", help="Output directory.")
    ap.add_argument("--num_trials", type=int, default=200, help="Number of trials.")
    ap.add_argument("--seed", type=int, default=0, help="Base seed.")
    ap.add_argument("--steps", type=int, default=None,
                    help="Override rollout steps (also sets cfg['T'] for eval and dyn sizing).")

    # Sampling controls (if you want randomized ICs)
    ap.add_argument("--sample_ic", action="store_true",
                    help="If set, ignore cfg['x0'] and sample starts in arena.")
    ap.add_argument("--pos_scale", type=float, default=0.95, help="Sample within pos_scale * arena_r.")
    ap.add_argument("--vel_scale", type=float, default=0.0, help="Stddev of sampled initial velocities.")
    ap.add_argument("--min_sep", type=float, default=0.0, help="Min defender-attacker separation when sampling.")

    # Multi-attacker handling (plotting always supports it)
    ap.add_argument("--multi_att_mode", default="worst",
                    choices=["worst", "avg"],
                    help="If num_attackers>1, evaluate defender vs each attacker separately "
                         "then aggregate per-trial as worst-case or average.")

    # Overrides
    ap.add_argument("--device", default=None)
    ap.add_argument("--def_ckpt_path", default=None)
    ap.add_argument("--att_ckpt_path", default=None)
    ap.add_argument("--deterministic", action="store_true", help="Force rl_eval_deterministic=True")
    ap.add_argument("--use_ukf", action="store_true", help="Force use_ukf=True")
    ap.add_argument("--x0_pos_jitter", type=float, default=None)
    ap.add_argument("--x0_vel_jitter", type=float, default=None)

    ap.add_argument("--alpha", type=float, default=0.05, help="CI alpha (0.05 => 95% CI).")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import config module
    mod = importlib.import_module(args.config_module)
    if not hasattr(mod, "config_for_eval") or not hasattr(mod, "build_dyn"):
        raise RuntimeError(f"Module '{args.config_module}' must define config_for_eval(...) and build_dyn(cfg).")

    # Base cfg from your canonical builder
    cfg0: Dict[str, Any] = mod.config_for_eval()

    # Apply simple overrides (used by RLPolicyDiff)
    if args.device is not None:
        cfg0["device"] = args.device
    if args.def_ckpt_path is not None:
        cfg0["def_ckpt_path"] = args.def_ckpt_path
    if args.att_ckpt_path is not None:
        cfg0["att_ckpt_path"] = args.att_ckpt_path
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

    # Steps override: set cfg["T"] BEFORE build_dyn so Ad_seq/Bd_seq sizing matches
    if args.steps is not None:
        cfg0["T"] = int(args.steps)

    # Determine attackers and dimension
    D = int(cfg0.get("D", 3))
    num_attackers = int(cfg0.get("num_attackers", 1))
    if num_attackers < 1:
        num_attackers = 1

    # Build dyn once for this cfg (may populate cfg0["dyn"] with Ad/Bd or sequences/caches)
    # Your runner can also build locally if dyn is missing, but prebuilding is preferred.
    mod.build_dyn(cfg0)

    # Start position logs for plots
    starts_def: List[np.ndarray] = []
    starts_atts: List[List[np.ndarray]] = [[] for _ in range(num_attackers)]

    # Trial rows
    trial_rows: List[Dict[str, Any]] = []
    passes = 0

    for i in range(int(args.num_trials)):
        seed = int(args.seed + i)
        np.random.seed(seed)  # runner uses np.random.randn for measurement noise
        rng = np.random.default_rng(seed)

        cfg_trial = copy.deepcopy(cfg0)

        # Build x0 (either sampled or from cfg)
        if args.sample_ic:
            x0 = _sample_x0(
                cfg_trial, rng,
                num_attackers=num_attackers,
                pos_scale=float(args.pos_scale),
                vel_scale=float(args.vel_scale),
                min_sep=float(args.min_sep),
            )
        else:
            x0 = np.asarray(cfg_trial["x0"], dtype=float)

        # Optional jitter
        x0 = _apply_x0_jitter(cfg_trial, x0, rng)

        # Record starts (all attackers for plotting)
        starts_def.append(x0[0, :D].copy())
        for j in range(num_attackers):
            if 1 + j < x0.shape[0]:
                starts_atts[j].append(x0[1 + j, :D].copy())

        # Evaluate:
        # runner is 1v1, so if multiple attackers we run separate episodes:
        per_att_metrics = []
        for j in range(num_attackers):
            if 1 + j >= x0.shape[0]:
                continue
            cfg_run = copy.deepcopy(cfg_trial)
            cfg_run["x0"] = np.asarray([x0[0], x0[1 + j]], dtype=float).tolist()

            try:
                out = run_rhc_with_rl_and_collect_frames_3d_diff(cfg_run, steps=args.steps)
                m = _compute_trial_metrics(cfg_run, out)
            except Exception as e:
                m = {"pass": 0, "error": str(e)}
            per_att_metrics.append(m)

        if not per_att_metrics:
            per_att_metrics = [{"pass": 0, "error": "no attackers present"}]

        # Aggregate across attackers for the trial
        if args.multi_att_mode == "worst":
            pass_trial = int(all(m.get("pass", 0) == 1 for m in per_att_metrics))
        else:
            pass_trial = int(round(np.mean([m.get("pass", 0) for m in per_att_metrics])))

        passes += pass_trial

        # Flatten a “primary attacker” row for CSV convenience (att1 metrics),
        # plus aggregate pass_trial and num_attackers.
        row: Dict[str, Any] = {
            "trial": i,
            "seed": seed,
            "pass_trial": pass_trial,
            "num_attackers": num_attackers,
            "def_x": float(x0[0, 0]),
            "def_y": float(x0[0, 1]) if D >= 2 else 0.0,
            "def_z": float(x0[0, 2]) if D == 3 else 0.0,
        }
        # attacker 1 start
        row.update({
            "att1_x": float(x0[1, 0]),
            "att1_y": float(x0[1, 1]) if D >= 2 else 0.0,
            "att1_z": float(x0[1, 2]) if D == 3 else 0.0,
        })

        # Add attacker-1 metrics if present
        m0 = per_att_metrics[0]
        for k, v in m0.items():
            row[f"att1_{k}"] = v

        trial_rows.append(row)

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

    # CSV
    csv_path = out_dir / "trials.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = sorted({k for r in trial_rows for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in trial_rows:
            w.writerow(r)

    # Start plots
    starts_def_arr = np.asarray(starts_def, dtype=float)
    starts_att_arrs = [
        np.asarray(sa, dtype=float) if len(sa) else np.zeros((0, D))
        for sa in starts_atts
    ]
    _save_start_plots(out_dir, D, starts_def_arr, starts_att_arrs)

    print(f"[eval] trials={n} passes={k} success_rate={sr:.3f} CI={lo:.3f}..{hi:.3f}")
    print(f"[eval] wrote: {out_dir/'results.json'}")
    print(f"[eval] wrote: {out_dir/'trials.csv'}")
    print(f"[eval] wrote: {out_dir/'starts_xy.png'}")
    print(f"[eval] wrote: {out_dir/'starts_xz.png'}")


if __name__ == "__main__":
    main()
