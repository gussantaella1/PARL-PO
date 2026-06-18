"""
dispersions.py

Editable Monte Carlo dispersion settings for evaluate_policy.py.

The evaluation harness first builds a nominal trial from the selected workflow
(auto shell grid, CSV rows, random shell sampling, or cfg["x0"]). It then applies
the dispersions below to that nominal trial.

Initial-state specs use:
  dist: "uniform" for bounded old-style x0_jitter draws, or "gaussian" for
        bell-curve draws.
  mean: additive mean offset for state dispersions.
  half_width: uniform draw half-width, used as mean + U(-half_width, +half_width).
  std: Gaussian standard deviation when dist is "gaussian".

KF/config parameter specs are Gaussian and use:
  mean: absolute mean value. Use None to keep the current config value.
  std:  Gaussian standard deviation.

Scalars apply to every axis. Length-3 lists apply to x/y/z or vx/vy/vz.
Config-derived values can be written as {"from_config": "path.to.key", "scale": 1.0}.

The non-KF defaults below match the old shell-grid Monte Carlo setup without
double-applying jitter: position and velocity dispersion use the old bounded
x0_jitter style, then evaluate_policy.py clears cfg["x0_jitter"] before rollout
so the environment does not jitter the already-dispersed x0 again.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


DISPERSIONS: Dict[str, Any] = {
    # Master switch for all evaluate_policy.py dispersion draws.
    "enabled": True,

    "seed": {
        # None means evaluate_policy.py uses its --seed as the episode seed base.
        # Set an integer here to decouple dispersion draws from --seed.
        "episode_seed_base": None,
    },

    # Additive uncertainty around the nominal initial condition.
    # For auto_shell_grid this means "relative to the generated shell point."
    "initial_state": {
        "enabled": True,
        "position": {
            "enabled": True,
            # Additive position error in meters, applied after the nominal shell
            # or CSV point is selected. This mirrors the old x0_jitter path:
            # x += U(-0.5, +0.5) per axis.
            "default": {"dist": "uniform", "mean": 0.0, "half_width": 0.5},
            # Role-specific entries are additional deltas on top of default.
            # Keep these at zero unless defender/attacker starts should differ.
            "def": {"dist": "uniform", "mean": [0.0, 0.0, 0.0], "half_width": [0.0, 0.0, 0.0]},
            "att": {"dist": "uniform", "mean": [0.0, 0.0, 0.0], "half_width": [0.0, 0.0, 0.0]},
        },
        "velocity": {
            "enabled": True,
            # Additive velocity draw in m/s after the nominal shell/CSV velocity
            # is selected. This mirrors the old eval-time x0_vel_jitter 0.5:
            # v += U(-0.5, +0.5) per axis. evaluate_policy.py then projects the
            # final velocity vector back to ||v|| <= vmax before rollout.
            "default": {"dist": "uniform", "mean": 0.0, "half_width": 0.5},
            # Role-specific entries are additional deltas on top of default.
            "def": {"dist": "uniform", "mean": [0.0, 0.0, 0.0], "half_width": [0.0, 0.0, 0.0]},
            "att": {"dist": "uniform", "mean": [0.0, 0.0, 0.0], "half_width": [0.0, 0.0, 0.0]},
        },
    },

    # Optional Gaussian draws for scalar/vector config values. These are applied
    # to the per-trial cfg copy only.
    "parameters": {
        "enabled": False,
        # Top-level config parameters can be randomized per trial. Means are
        # absolute values; mean=None keeps the loaded run config value.
        # "umax": {"mean": None, "std": 0.0, "min": 0.0},
        # "dt": {"mean": None, "std": 0.0, "min": 1e-9},
    },

    # KF/EKF/UKF traits. Applied only when cfg["use_kf"] is true.
    # Means are absolute values; mean=None keeps the current cfg["ukf"][key].
    "kf": {
        "enabled": True,
        # Prevents KF settings from touching non-KF evaluations.
        "only_when_use_kf": True,
        "parameters": {
            # Sensor angular noise standard deviations, radians.
            "sigma_az": {"mean": np.deg2rad(0.5), "std": 0.0, "min": 0.0},
            "sigma_el": {"mean": np.deg2rad(0.5), "std": 0.0, "min": 0.0},
            # Initial covariance diagonal scales for target position/velocity.
            "pos_std0": {"mean": 4.0, "std": 0.0, "min": 0.0},
            "vel_std0": {"mean": 0.01, "std": 0.0, "min": 0.0},
            # Initial estimator mean perturbation scales.
            "init_pos_std": {"mean": 4.0, "std": 0.0, "min": 0.0},
            "init_vel_std": {"mean": 0.01, "std": 0.0, "min": 0.0},
            "init_mean_pos_std": {"mean": 0.0, "std": 0.0, "min": 0.0},
            "init_mean_vel_std": {"mean": 0.0, "std": 0.0, "min": 0.0},
            # Control measurement noise used when ukf.action_access == "measured".
            # Mean is the loaded run's prior-action noise; std is 10% of that
            # value so folders with different umax/action noise scale naturally.
            "action_meas_std": {
                "mean": {"from_config": "ukf.action_meas_std"},
                "std": {"from_config": "ukf.action_meas_std", "scale": 0.10},
                "min": 0.0,
            },
            # Acceleration-state EKF/UKF traits for 9-state estimator variants.
            "accel_std0": {"mean": None, "std": 0.0, "min": 0.0},
            "init_mean_accel_std": {"mean": None, "std": 0.0, "min": 0.0},
            # Center-relative estimator variants, if enabled by the run config.
            "center_sigma_az": {"mean": None, "std": 0.0, "min": 0.0},
            "center_sigma_el": {"mean": None, "std": 0.0, "min": 0.0},
            "center_pos_std0": {"mean": None, "std": 0.0, "min": 0.0},
            "center_vel_std0": {"mean": None, "std": 0.0, "min": 0.0},
            "center_init_pos_std": {"mean": None, "std": 0.0, "min": 0.0},
            "center_init_vel_std": {"mean": None, "std": 0.0, "min": 0.0},
            "center_init_mean_pos_std": {"mean": None, "std": 0.0, "min": 0.0},
            "center_init_mean_vel_std": {"mean": None, "std": 0.0, "min": 0.0},
        },
    },
}


def config_for_eval() -> Dict[str, Any]:
    """Return a copy of the dispersion config for evaluate_policy.py."""
    import copy

    return copy.deepcopy(DISPERSIONS)
