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
    "out_path": "Distillation_Tester/def1_ukf_student_manual_modern.pt",
    "metrics_out": "Distillation_Tester/distill_metrics_def1_student_manual_modern.npz",

    # Basic role/mode selection
    "train_role": "def",          # "def" or "att"
    "attacker_mode": "rule",      # "rule" or "rl"
    # "distill_method": "paper_recurrent",   # "modern" or "paper_recurrent"

    "distill_method": "paper_recurrent",
    "distill_collection_mode": "dagger",


    # Optional frozen opponent checkpoints for attacker_mode="rl"
    "def_ckpt_path": None,
    "att_ckpt_path": None,

    # Optional top-level overrides applied on top of config_for_train(...)
    # Any key from config_rl can go here. Nested dicts are allowed.
    "cfg_overrides": {
        "device": "cuda",
        "distill_paper_student_lstm_hidden": 256,
        "distill_paper_tbptt_chunk_len": 40,
        "distill_paper_train_episodes_per_iter": None,   # or 128 if runtime matters
        "distill_paper_lambda_intent": 0.25,
        "distill_paper_action_loss": "mse",
        "distill_paper_intent_loss": "mse",
        "distill_paper_grad_clip_norm": None,
        "distill_paper_log_every": 10,
        "distill_paper_allow_sim_intent_fallback": True,
        "episodes_per_iter": 8,
        "max_steps": 300,
        "iters": 100,
        "dagger_beta_start": 1.0,
        "dagger_beta_end": 0.0,
        "dagger_decay_iters": 50,
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
    if run_cfg.get("distill_collection_mode") is not None:
        cfg["distill_collection_mode"] = run_cfg["distill_collection_mode"]
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
                "distill_collection_mode": cfg.get("distill_collection_mode"),
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
