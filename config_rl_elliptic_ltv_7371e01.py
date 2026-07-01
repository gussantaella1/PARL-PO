"""Evaluation config module for the corrected nominal Elliptic LTV orbit.

This keeps the regular training/evaluation config behavior, but overrides the
chief orbit before dynamics are built. It is intended for Monte Carlo reruns
that should not overwrite the original Elliptic LTV results.
"""

from __future__ import annotations

from typing import Any, Dict

import config_rl as _base


CHIEF_ORBIT_7371E01 = {
    "mu": 3.986004418e14,
    "a": 6_371_000.0 + 1_000_000.0,
    "e": 0.1,
    "i": 0.0,
    "raan": 0.0,
    "argp": 0.0,
    "nu0": 0.0,
}


def _apply_orbit(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg["chief_orbit"] = dict(CHIEF_ORBIT_7371E01)
    return cfg


def config_for_train(**overrides) -> Dict[str, Any]:
    cfg = _base.config_for_train(**overrides)
    return _apply_orbit(cfg)


def config_for_eval(**overrides) -> Dict[str, Any]:
    cfg = _base.config_for_eval(**overrides)
    return _apply_orbit(cfg)


def build_dyn(cfg: Dict[str, Any]):
    _apply_orbit(cfg)
    return _base.build_dyn(cfg)
