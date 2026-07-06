#!/usr/bin/env python3
"""Precompute/load the disk-backed full-orbit Elliptic LTV dynamics cache."""

from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path
from typing import Any, Dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_module", default="config_rl")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--dt", type=float, default=None)
    args = parser.parse_args()

    mod = importlib.import_module(args.config_module)
    if not hasattr(mod, "config_for_eval") or not hasattr(mod, "build_dyn"):
        raise RuntimeError(f"{args.config_module!r} must define config_for_eval and build_dyn.")

    cfg: Dict[str, Any] = mod.config_for_eval()
    cfg["dynamics"] = "elliptic_ltv"
    cfg["T"] = int(args.steps)
    if args.dt is not None:
        cfg["dt"] = float(args.dt)
    cfg.setdefault("elliptic_ltv", {})
    cfg["elliptic_ltv"]["randomize_nu0"] = True
    cfg["elliptic_ltv"]["use_full_orbit_cache"] = True
    cfg["elliptic_ltv"]["disk_cache_enabled"] = True

    t0 = time.time()
    mod.build_dyn(cfg)
    dyn = cfg.get("dyn", {}) or {}
    elapsed = time.time() - t0
    cache_path = dyn.get("full_orbit_cache_path", None)
    loaded = bool(dyn.get("full_orbit_cache_loaded", False))
    period_steps = dyn.get("full_orbit_period_steps", None)
    period_sec = dyn.get("full_orbit_period_sec", None)

    print(
        "[prewarm_elliptic_ltv_cache] "
        f"status={'loaded' if loaded else 'built'} "
        f"elapsed_sec={elapsed:.3f} "
        f"period_steps={period_steps} "
        f"period_sec={period_sec} "
        f"path={Path(cache_path) if cache_path else '<memory-only>'}"
    )


if __name__ == "__main__":
    main()
