#!/usr/bin/env python3
"""
make_shelled_trials_csv.py

Generate a trials.csv containing evenly spaced start positions on concentric shells
inside the arena specified by config_module.config_for_eval().

Outputs rows with:
  trial, seed,
  def_x, def_y, def_z, def_vx, def_vy, def_vz,
  att1_x, att1_y, att1_z, att1_vx, att1_vy, att1_vz

Notes:
- Uses Fibonacci-sphere directions for uniform-ish points per shell.
- Defender/attacker pairing can be antipodal or index-offset to enforce separation.

Example:
  python make_shelled_trials_csv.py \
    --config_module config_rl \
    --out shelled_trials.csv \
    --pos_scale 0.95 \
    --num_shells 12 \
    --pts_per_shell 256 \
    --pairing offset \
    --random_rotate_dirs
"""

from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def get_center_and_radius(cfg: Dict[str, Any], D: int) -> Tuple[np.ndarray, float]:
    """Handle get center and radius for this workflow."""
    ar = cfg.get("arena", {}) or {}
    cx, cy = float(ar.get("cx", 0.0)), float(ar.get("cy", 0.0))
    cz = float(ar.get("cz", 0.0)) if D == 3 else 0.0
    r = float(ar.get("r", 20.0))
    center = np.array([cx, cy, cz], dtype=float)[:D]
    return center, r


def fibonacci_sphere(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Return ~uniform directions on the unit sphere, shape (n, 3).
    Deterministic given n (unless you ask for random rotation later).
    """
    if n <= 0:
        return np.zeros((0, 3), dtype=float)

    # Golden angle
    ga = np.pi * (3.0 - np.sqrt(5.0))

    i = np.arange(n, dtype=float)
    y = 1.0 - (2.0 * i + 1.0) / n
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    phi = ga * i
    x = np.cos(phi) * r
    z = np.sin(phi) * r
    dirs = np.stack([x, y, z], axis=1)

    # Optional: apply a random rotation so the “seams” don’t always align shell-to-shell
    if rng is not None:
        # random rotation via QR decomposition
        A = rng.normal(size=(3, 3))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        dirs = dirs @ Q.T

    # normalize (should already be unit-ish)
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    return dirs


def main():
    """Parse command-line arguments and run this script."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_module", default="config_rl",
                    help="Module with config_for_eval() (default: config_rl).")
    ap.add_argument("--out", default="shelled_trials.csv",
                    help="Output csv path (default: shelled_trials.csv).")

    ap.add_argument("--pos_scale", type=float, default=0.95,
                    help="Max radius as fraction of arena R (default 0.95).")
    ap.add_argument("--num_shells", type=int, default=12,
                    help="Number of shells (default 12).")
    ap.add_argument("--pts_per_shell", type=int, default=256,
                    help="Points per shell (default 256).")
    ap.add_argument("--include_center", action="store_true",
                    help="Include a center point (r=0) as an extra trial.")

    ap.add_argument("--min_sep", type=float, default=0.0,
                    help="Minimum defender-attacker separation in meters (default 0).")
    ap.add_argument("--pairing", choices=["antipodal", "offset"], default="offset",
                    help="How to pair attacker points relative to defender points.")
    ap.add_argument("--offset_k", type=int, default=None,
                    help="Index offset for 'offset' pairing; default = N//2.")

    ap.add_argument("--vel_scale", type=float, default=0.0,
                    help="Stddev of initial velocities per component (default 0 => zeros).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for optional shell random rotation / velocities (default 0).")
    ap.add_argument("--random_rotate_dirs", action="store_true",
                    help="Randomly rotate Fibonacci directions per shell (nice to avoid alignment).")

    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))

    mod = importlib.import_module(args.config_module)
    if not hasattr(mod, "config_for_eval"):
        raise RuntimeError(f"{args.config_module} must define config_for_eval().")

    cfg0: Dict[str, Any] = mod.config_for_eval()
    D = int(cfg0.get("D", 3))
    if D != 3:
        raise RuntimeError(f"This generator is intended for D=3, but config has D={D}.")

    center, R = get_center_and_radius(cfg0, D)
    r_max = float(args.pos_scale) * float(R)

    num_shells = int(args.num_shells)
    pts_per_shell = int(args.pts_per_shell)
    vel_scale = float(args.vel_scale)

    # Start at r_max / num_shells instead of zero; the center case is special enough
    # that it stays behind --include_center.
    shells = np.linspace(r_max / num_shells, r_max, num_shells)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "trial", "seed",
        "def_x", "def_y", "def_z", "def_vx", "def_vy", "def_vz",
        "att1_x", "att1_y", "att1_z", "att1_vx", "att1_vy", "att1_vz",
    ]

    rows = []
    trial_idx = 0

    def add_row(p_def, v_def, p_att, v_att):
        """Handle add row for this workflow."""
        nonlocal trial_idx
        # Keep row construction in one place so the CSV schema, trial ids, and
        # per-trial seed offsets cannot drift between the center row and shell rows.
        rows.append({
            "trial": trial_idx,
            "seed": int(args.seed) + trial_idx,
            "def_x": float(p_def[0]), "def_y": float(p_def[1]), "def_z": float(p_def[2]),
            "def_vx": float(v_def[0]), "def_vy": float(v_def[1]), "def_vz": float(v_def[2]),
            "att1_x": float(p_att[0]), "att1_y": float(p_att[1]), "att1_z": float(p_att[2]),
            "att1_vx": float(v_att[0]), "att1_vy": float(v_att[1]), "att1_vz": float(v_att[2]),
        })
        trial_idx += 1

    # Optional center trial
    if args.include_center:
        p_def = center.copy()
        p_att = center.copy()
        v_def = rng.normal(size=3) * vel_scale if vel_scale > 0 else np.zeros(3)
        v_att = rng.normal(size=3) * vel_scale if vel_scale > 0 else np.zeros(3)
        add_row(p_def, v_def, p_att, v_att)

    for r in shells:
        shell_rng = rng if args.random_rotate_dirs else None
        dirs = fibonacci_sphere(pts_per_shell, rng=shell_rng)

        # Defender positions on this shell
        P_def = center[None, :] + r * dirs

        # Pairing controls how hard the same-radius geometry is: antipodal maximizes
        # separation, while offset preserves shell coverage without identical directions.
        if args.pairing == "antipodal":
            P_att = center[None, :] + r * (-dirs)
        else:
            k = args.offset_k
            if k is None:
                k = pts_per_shell // 2
            P_att = np.roll(P_def, shift=int(k), axis=0)

        # Enforce min separation by nudging attacker pairing if needed
        if args.min_sep > 0:
            min_sep = float(args.min_sep)
            for i in range(pts_per_shell):
                if np.linalg.norm(P_att[i] - P_def[i]) < min_sep:
                    # Try nearby index offsets first before falling back to antipodal.
                    ok = False
                    for extra in range(1, min(pts_per_shell, 50)):
                        cand = np.roll(P_def, shift=int(k + extra), axis=0)[i]
                        if np.linalg.norm(cand - P_def[i]) >= min_sep:
                            P_att[i] = cand
                            ok = True
                            break
                    if not ok:
                        # fallback: antipodal
                        P_att[i] = center + r * (-dirs[i])

        # Velocities
        if vel_scale > 0:
            V_def = rng.normal(size=(pts_per_shell, 3)) * vel_scale
            V_att = rng.normal(size=(pts_per_shell, 3)) * vel_scale
        else:
            V_def = np.zeros((pts_per_shell, 3))
            V_att = np.zeros((pts_per_shell, 3))

        for i in range(pts_per_shell):
            add_row(P_def[i], V_def[i], P_att[i], V_att[i])

    # Write after generation succeeds, which avoids leaving behind a partial CSV if
    # a pairing or config issue raises midway through.
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[make_shelled_trials] wrote {len(rows)} trials to: {out_path.resolve()}")
    print(f"[make_shelled_trials] arena center={center.tolist()} R={R} r_max={r_max} shells={num_shells} pts/shell={pts_per_shell}")


if __name__ == "__main__":
    main()
