from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np

from config_rl import build_dyn, config_for_train
from core.distill import distill_from_teacher


# =========================================================
# Edit These Knobs
# =========================================================

DISTILL_RUN: Dict[str, Any] = {
    # Required paths
    "teacher_ckpt": "Training_Policy/def1_teacher.pt",
    "out_path": "Distillation_Tester/def1_ukf_student_manual_intervention.pt",
    "metrics_out": "Distillation_Tester/distill_metrics_def1_student_manual_intervention.npz",

    # Basic role/mode selection
    "train_role": "def",          # "def" or "att"
    "attacker_mode": "rule",      # "rule" or "rl"
    "distill_method": "intervention",   # "modern", "teacher_forced", "intervention", or "original"

    # Optional frozen opponent checkpoints for attacker_mode="rl"
    "def_ckpt_path": None,
    "att_ckpt_path": None,

    # Optional top-level overrides applied on top of config_for_train(...)
    # Any key from config_rl can go here. Nested dicts are allowed.
    "cfg_overrides": {
        "device": "cpu",
        "distill": True,
        "use_ukf": True,

        # Modern distill knobs
        "distill_warm_start_from_teacher": True,
        "distill_teacher_label": "mean",
        "distill_batch_size": 2048,
        "distill_epochs": 4,
        "distill_buffer_capacity": None,
        "distill_w_nll": 1.0,
        # "distill_w_kl": 0.0,
        "distill_w_kl": 0.05,
        # "distill_w_mse": 0.0,
        "distill_w_mse": 0.25,
        "distill_w_v": 0.0,
        "distill_train_logstd": False,
        "distill_logstd_min": -5.0,
        "distill_logstd_max": 1.0,
        "distill_intervention_action_l2_thresh": None,
        "distill_intervention_oi_margin_m": 1.0,
        "distill_intervention_collision_margin_m": 1.0,
        "distill_log_every": 10,

        # Shared rollout / DAgger knobs
        "episodes_per_iter": 8,
        "max_steps": 300,
        "iters": 100,
        "dagger_beta_start": 1.0,
        "dagger_beta_end": 0.0,
        "dagger_decay_iters": 50,

        # UKF / belief knobs
        "reward_from_belief": True,
        "use_meas_reward": False,
        "meas_innov_coef": 0.0,
        "meas_cov_coef": 0.0,
    },
}


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def build_distill_cfg(run_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = config_for_train(
        attacker_mode=run_cfg["attacker_mode"],
        train_role=run_cfg["train_role"],
    )
    cfg["distill"] = True
    cfg["distill_method"] = run_cfg["distill_method"]
    cfg["use_ukf"] = True

    if run_cfg.get("def_ckpt_path") is not None:
        cfg["def_ckpt_path"] = run_cfg["def_ckpt_path"]
    if run_cfg.get("att_ckpt_path") is not None:
        cfg["att_ckpt_path"] = run_cfg["att_ckpt_path"]

    _deep_update(cfg, dict(run_cfg.get("cfg_overrides", {})))
    build_dyn(cfg)
    return cfg


def main() -> None:
    run_cfg = dict(DISTILL_RUN)
    cfg = build_distill_cfg(run_cfg)

    teacher_ckpt = os.path.abspath(run_cfg["teacher_ckpt"])
    out_path = os.path.abspath(run_cfg["out_path"])
    metrics_out = os.path.abspath(run_cfg["metrics_out"])

    print("[distill_policy] starting")
    print(
        json.dumps(
            {
                "teacher_ckpt": teacher_ckpt,
                "out_path": out_path,
                "metrics_out": metrics_out,
                "train_role": cfg["train_role"],
                "attacker_mode": cfg["attacker_mode"],
                "distill_method": cfg["distill_method"],
                "device": cfg["device"],
            },
            indent=2,
        )
    )

    _student, metrics = distill_from_teacher(
        cfg,
        teacher_ckpt,
        out_path=out_path,
    )

    os.makedirs(os.path.dirname(metrics_out) or ".", exist_ok=True)
    np.savez(metrics_out, **metrics)
    print(f"[distill_policy] saved metrics -> {metrics_out}")


if __name__ == "__main__":
    main()
