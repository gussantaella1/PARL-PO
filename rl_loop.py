"""
rl_loop.py
===================================
Single-file training & evaluation where **only the defender is learned (PPO)**
and the **attacker is a deterministic rule-based controller** that (a) drives to the
center and (b) repels away from the defender under HCW dynamics.

Key points
----------
- Single source of truth for config via: from config_rl import config_for_train, config_for_eval, build_dyn
- Clean attacker swap: set cfg["attacker_mode"] to "rule" (default) or "rl"
- Differentiable one-step ridge prior (DiffLS-style) blended into actor mean
- Minimal, single-process VecEnv for reproducible, fixed-length rollouts
"""

import os

import numpy as np

from config_rl import config_for_train
from core.plotting import (
    load_npz_metrics,
    plot_compare_phases,
    plot_ic_support_from_cfg,
    plot_ic_used_from_npz,
    plot_training_metrics,
)
from core.logger_utils import RunLogger
from core.train_eval import train_with_distill



#Distillation helpers. 
# Distillation based on https://arxiv.org/pdf/2308.16185


    # =============================================================
# Main: stepped training (Def₀ -> Att₁ -> Def₁) + defender distillation
# =============================================================
if __name__ == "__main__":

    do_phase_0 = True
    do_phase_1 = True
    do_phase_2 = True
    do_phase_3 = True
    do_phase_4 = True

    def0_teacher_ckpt = "Training_Policy/def0_teacher.pt"
    att1_teacher_ckpt = "Training_Policy/att1_teacher.pt"
    def1_teacher_ckpt = "Training_Policy/def1_teacher.pt"
    att2_teacher_ckpt = "Training_Policy/att2_teacher.pt"
 

    OUT_DIR = "Training_Policy"
    os.makedirs(OUT_DIR, exist_ok=True)

    PLOTS_ROOT = os.path.join(OUT_DIR, "Plots")
    PLOTS_DEF0 = os.path.join(PLOTS_ROOT, "def0")
    PLOTS_ATT1 = os.path.join(PLOTS_ROOT, "att1")
    PLOTS_DEF1 = os.path.join(PLOTS_ROOT, "def1")
    PLOTS_ATT2 = os.path.join(PLOTS_ROOT, "att2")
    PLOTS_DEF2 = os.path.join(PLOTS_ROOT, "def2")
    PLOTS_COMP = os.path.join(PLOTS_ROOT, "comparisons")
    for d in [PLOTS_DEF0, PLOTS_ATT1, PLOTS_DEF1, PLOTS_ATT2, PLOTS_DEF2, PLOTS_COMP]:
        os.makedirs(d, exist_ok=True)

    runlog = RunLogger(OUT_DIR, filename="run_manifest.json")

    cfg_distillation = config_for_train(
        attacker_mode="rl",   # attacker is RL now
        train_role="att",     # PPO will only update attacker
    )
    DISTILL = cfg_distillation["distill"]


    cfg_training_log = config_for_train(
        attacker_mode="train",
        train_role="rl",
    )

    runlog.set_config("training config used", cfg_training_log)

    # save the exact distillation config you started with

    # runlog.set_config("cfg_distillation", cfg_distillation)


    # =========================================================
    # PHASE 0: Defender₀ vs rule-based attacker (teacher + distill)
    # =========================================================
    if do_phase_0 is True:
        print("\n===== PHASE 0: Train DEFENDER_0 vs RULE attacker =====")

        phase0_extra = {
            "gamma": 0.991,
            "hit_buffer_def": 0.25,
        }
        with runlog.stage(
            "PHASE0_train_def0",
            phase_name="def0",
            attacker_mode="rule",
            extra_train_cfg=phase0_extra,
        ) as st:
            def0_teacher_ckpt, def0_student_ckpt, def0_meta = train_with_distill(
                out_dir = OUT_DIR,
                phase_name="def0",
                attacker_mode="rule",
                train_role="def",
                extra_train_cfg=phase0_extra,
            )
            st["outputs"] = def0_meta

        # def0_teacher_ckpt = "Training_Policy/def0_teacher.pt"

        m_def0_teacher = load_npz_metrics(os.path.join(OUT_DIR, "train_metrics_def0_teacher.npz"))
        plot_training_metrics(
            m_def0_teacher,
            title="def0_teacher",
            smooth="ema",
            smooth_param=0.2,
            show=False,
            out_dir=PLOTS_DEF0,
            save_prefix="def0_teacher",
        )

        plot_ic_support_from_cfg(
            cfg_training_log,
            n_scenes=30000,
            seed=123,
            title="def0 feasible IC support",
            out_path=os.path.join(PLOTS_DEF0, "def0_ic_support.png"),
            show=False,
        )

        plot_ic_used_from_npz(
            os.path.join(OUT_DIR, "ic_samples_def0_teacher.npz"),
            cfg_training_log,
            title="def0 ICs actually used during training",
            out_path=os.path.join(PLOTS_DEF0, "def0_ic_used.png"),
            show=False,
        )


    # =========================================================
    # PHASE 1: Attacker₁ vs fixed Defender₀ (teacher only, for now)
    # =========================================================
    if do_phase_1 is True:

        print("\n===== PHASE 1: Train ATTACKER_1 vs frozen DEFENDER_0 =====")
        phase1_extra = {
            "def_ckpt_path": def0_teacher_ckpt,
            "freeze_defender": True,
            "train_ic_mode": "random_shell",
            "r_att_min": 0.0,
            "gamma": 0.991,
            # "gamma": 0.994, #Aggressive, but can be confused

            # NEW:
            # "opp_mix": {
            #     "modes": ["none", "def0", "weak"],
            #     "probs": [0.05, 0.95, 0.00],     # must sum to 1
            #     "resample": "episode",           # "episode" or "never"
            #     "weak_scale": 0.25,              # 0.0 -> basically none, 1.0 -> full def0
            #     "weak_noise_std": 0.00,          # optional additive Gaussian in action space
            # },
            # Checkpoint-based defender mixture:
            "opp_mix": {
                "resample": "episode",
                "policies": [
                    {
                        "name": "def0_full",
                        "path": "Training_Policy/def0_teacher.pt",
                        "prob": 0.5,
                        "action_scale": 1.0,
                        "noise_std": 0.0,
                        "idle_prob": 0.0,
                    },
                    {
                        "name": "def0_weak",
                        "path": "Training_Policy/def0_teacher.pt",
                        "prob": 0.3,
                        "action_scale": 0.25,
                        "noise_std": 0.05,
                        "idle_prob": 0.0,
                    },
                    {
                        "name": "def0_very_weak",
                        "path": "Training_Policy/def0_teacher.pt",
                        "prob": 0.2,
                        "action_scale": 0.0,
                        "noise_std": 0.0,
                        "idle_prob": 0.0,
                    },
                ],
            },

        }

        with runlog.stage(
            "PHASE1_train_att1",
            phase_name="att1",
            attacker_mode="rl",
            extra_train_cfg=phase1_extra,
        ) as st:
            att1_teacher_ckpt, att1_student_ckpt, att1_meta = train_with_distill(
                out_dir = OUT_DIR,
                phase_name="att1",
                attacker_mode="rl",
                train_role="att",
                extra_train_cfg=phase1_extra,
            )
            st["outputs"] = att1_meta

        m_att1_teacher = load_npz_metrics(os.path.join(OUT_DIR, "train_metrics_att1_teacher.npz"))
        plot_training_metrics(
            m_att1_teacher,
            title="att1_teacher",
            smooth="ema",
            smooth_param=0.2,
            show=False,
            out_dir=PLOTS_ATT1,
            save_prefix="att1_teacher",
        )

        plot_ic_support_from_cfg(
            cfg_training_log,
            n_scenes=30000,
            seed=123,
            title="att1 feasible IC support",
            out_path=os.path.join(PLOTS_ATT1, "att1_ic_support.png"),
            show=False,
        )

        plot_ic_used_from_npz(
            os.path.join(OUT_DIR, "ic_samples_att1_teacher.npz"),
            cfg_training_log,
            title="att1 ICs actually used during training",
            out_path=os.path.join(PLOTS_ATT1, "att1_ic_used.png"),
            show=False,
        )


    # =========================================================
    # PHASE 2: Defender₁ vs frozen Attacker₁ (teacher + distill)
    # =========================================================
    if do_phase_2 is True:

        print("\n===== PHASE 2: Train DEFENDER_1 vs frozen ATTACKER_1 =====")
        phase2_extra = {"att_ckpt_path": att1_teacher_ckpt,
                        "def_ckpt_path": def0_teacher_ckpt,
                        "freeze_attacker": True,
                        # "gamma": 0.993,
                        # "gamma": 0.990, # Performed poorly against .994 attacker, good against .990 
                        # "gamma": 0.994, # Performed poorly against .994 attacker
                        "gamma": 0.991,
                        # Checkpoint-based attacker mixture:
                        "opp_mix": {
                            "resample": "episode",
                            "policies": [
                                {
                                    "name": "att1_full",
                                    "path": "Training_Policy/att1_teacher.pt",
                                    "prob": 0.5,
                                    "action_scale": 1.0,
                                    "noise_std": 0.0,
                                    "idle_prob": 0.0,
                                },
                                {
                                    "name": "att1_weak",
                                    "path": "Training_Policy/att1_teacher.pt",
                                    "prob": 0.3,
                                    "action_scale": 0.25,
                                    "noise_std": 0.05,
                                    "idle_prob": 0.0,
                                },
                                {
                                    "name": "att1_very_weak",
                                    "path": "Training_Policy/att1_teacher.pt",
                                    "prob": 0.2,
                                    "action_scale": 0.0,
                                    "noise_std": 0.0,
                                    "idle_prob": 0.0,
                                },
                            ],
                        },

                        }

        with runlog.stage(
            "PHASE2_train_def1",
            phase_name="def1",
            attacker_mode="rl",
            extra_train_cfg=phase2_extra,
        ) as st:
            def1_teacher_ckpt, def1_student_ckpt, def1_meta = train_with_distill(
                out_dir = OUT_DIR,
                phase_name="def1",
                attacker_mode="rl",
                train_role="def",
                extra_train_cfg=phase2_extra,
            )
            st["outputs"] = def1_meta

        # ---- def1 ----
        m_def1_teacher = load_npz_metrics(os.path.join(OUT_DIR, "train_metrics_def1_teacher.npz"))
        plot_training_metrics(
            m_def1_teacher,
            title="def1_teacher",
            smooth="ema",
            smooth_param=0.2,
            show=False,
            out_dir=PLOTS_DEF1,
            save_prefix="def1_teacher",
        )

        plot_ic_support_from_cfg(
            cfg_training_log,
            n_scenes=30000,
            seed=123,
            title="def1 feasible IC support",
            out_path=os.path.join(PLOTS_DEF1, "def1_ic_support.png"),
            show=False,
        )

        plot_ic_used_from_npz(
            os.path.join(OUT_DIR, "ic_samples_def1_teacher.npz"),
            cfg_training_log,
            title="def1 ICs actually used during training",
            out_path=os.path.join(PLOTS_DEF1, "def1_ic_used.png"),
            show=False,
        )

    # =========================================================
    # PHASE 3: Attacker_2 vs frozen Defender_1 (teacher + distill)
    # =========================================================
    if do_phase_3 is True:

        print("\n===== PHASE 3: Train ATTACKER_2 vs frozen DEFENDER_1 =====")
        phase3_extra = {
            "def_ckpt_path": def1_teacher_ckpt,
            "freeze_defender": True,
            "train_ic_mode": "random_shell",
            "r_att_min": 0.0,
            "gamma": 0.991,

            # Checkpoint-based attacker mixture:
            "opp_mix": {
                "resample": "episode",
                "policies": [
                    {
                        "name": "def0_full",
                        "path": "Training_Policy/def0_teacher.pt",
                        "prob": 0.15,
                        "action_scale": 1.0,
                        "noise_std": 0.0,
                        "idle_prob": 0.0,
                    },
                    {
                        "name": "def0_weak",
                        "path": "Training_Policy/def0_teacher.pt",
                        "prob": 0.15,
                        "action_scale": 0.25,
                        "noise_std": 0.05,
                        "idle_prob": 0.0,
                    },
                    {
                        "name": "def0_very_weak",
                        "path": "Training_Policy/def0_teacher.pt",
                        "prob": 0.1,
                        "action_scale": 0.0,
                        "noise_std": 0.0,
                        "idle_prob": 0.0,
                    },

                    {
                        "name": "def1_full",
                        "path": "Training_Policy/def1_teacher.pt",
                        "prob": 0.3,
                        "action_scale": 1.0,
                        "noise_std": 0.0,
                        "idle_prob": 0.0,
                    },
                    {
                        "name": "def1_weak",
                        "path": "Training_Policy/def1_teacher.pt",
                        "prob": 0.2,
                        "action_scale": 0.25,
                        "noise_std": 0.05,
                        "idle_prob": 0.0,
                    },
                    {
                        "name": "def1_very_weak",
                        "path": "Training_Policy/def1_teacher.pt",
                        "prob": 0.1,
                        "action_scale": 0.0,
                        "noise_std": 0.0,
                        "idle_prob": 0.0,
                    },
                ],
            },

        }

        with runlog.stage(
            "PHASE3_train_att2",
            phase_name="att2",
            attacker_mode="rl",
            extra_train_cfg=phase3_extra,
        ) as st:
            att2_teacher_ckpt, att2_student_ckpt, att2_meta = train_with_distill(
                out_dir=OUT_DIR,
                phase_name="att2",
                attacker_mode="rl",
                train_role="att",
                extra_train_cfg=phase3_extra,
            )
            st["outputs"] = att2_meta

        m_att2_teacher = load_npz_metrics(os.path.join(OUT_DIR, "train_metrics_att2_teacher.npz"))
        plot_training_metrics(
            m_att2_teacher,
            title="att2_teacher",
            smooth="ema",
            smooth_param=0.2,
            show=False,
            out_dir=PLOTS_ATT2,
            save_prefix="att2_teacher",
        )

        plot_ic_support_from_cfg(
            cfg_training_log,
            n_scenes=30000,
            seed=123,
            title="att2 feasible IC support",
            out_path=os.path.join(PLOTS_ATT2, "att2_ic_support.png"),
            show=False,
        )

        plot_ic_used_from_npz(
            os.path.join(OUT_DIR, "ic_samples_att2_teacher.npz"),
            cfg_training_log,
            title="att2 ICs actually used during training",
            out_path=os.path.join(PLOTS_ATT2, "att2_ic_used.png"),
            show=False,
        )

    # =========================================================
    # PHASE 4: Defender_2 vs frozen Attacker_2 (teacher + distill)
    # =========================================================
    if do_phase_4 is True:

        print("\n===== PHASE 4: Train DEFENDER_2 vs frozen ATTACKER_2 =====")
        phase4_extra = {
            "att_ckpt_path": att2_teacher_ckpt,
            "def_ckpt_path": def1_teacher_ckpt,
            "freeze_attacker": True,
            "gamma": 0.991,

                # Checkpoint-based attacker mixture:
            "opp_mix": {
                "resample": "episode",
                "policies": [
                    {
                        "name": "att1_full",
                        "path": "Training_Policy/att1_teacher.pt",
                        "prob": 0.15,
                        "action_scale": 1.0,
                        "noise_std": 0.0,
                        "idle_prob": 0.0,
                    },
                    {
                        "name": "att1_weak",
                        "path": "Training_Policy/att1_teacher.pt",
                        "prob": 0.15,
                        "action_scale": 0.25,
                        "noise_std": 0.05,
                        "idle_prob": 0.0,
                    },
                    {
                        "name": "att1_very_weak",
                        "path": "Training_Policy/att1_teacher.pt",
                        "prob": 0.1,
                        "action_scale": 0.0,
                        "noise_std": 0.0,
                        "idle_prob": 0.0,
                    },

                    {
                        "name": "att2_full",
                        "path": "Training_Policy/att2_teacher.pt",
                        "prob": 0.3,
                        "action_scale": 1.0,
                        "noise_std": 0.0,
                        "idle_prob": 0.0,
                    },
                    {
                        "name": "att2_weak",
                        "path": "Training_Policy/att2_teacher.pt",
                        "prob": 0.2,
                        "action_scale": 0.25,
                        "noise_std": 0.05,
                        "idle_prob": 0.0,
                    },
                    {
                        "name": "att2_very_weak",
                        "path": "Training_Policy/att2_teacher.pt",
                        "prob": 0.1,
                        "action_scale": 0.0,
                        "noise_std": 0.0,
                        "idle_prob": 0.0,
                    },
                ],
            },


            
        }

        with runlog.stage(
            "PHASE4_train_def2",
            phase_name="def2",
            attacker_mode="rl",
            extra_train_cfg=phase4_extra,
        ) as st:
            def2_teacher_ckpt, def2_student_ckpt, def2_meta = train_with_distill(
                out_dir=OUT_DIR,
                phase_name="def2",
                attacker_mode="rl",
                train_role="def",
                extra_train_cfg=phase4_extra,
            )
            st["outputs"] = def2_meta

        m_def2_teacher = load_npz_metrics(os.path.join(OUT_DIR, "train_metrics_def2_teacher.npz"))
        plot_training_metrics(
            m_def2_teacher,
            title="def2_teacher",
            smooth="ema",
            smooth_param=0.2,
            show=False,
            out_dir=PLOTS_DEF2,
            save_prefix="def2_teacher",
        )

        plot_ic_support_from_cfg(
            cfg_training_log,
            n_scenes=30000,
            seed=123,
            title="def2 feasible IC support",
            out_path=os.path.join(PLOTS_DEF2, "def2_ic_support.png"),
            show=False,
        )

        plot_ic_used_from_npz(
            os.path.join(OUT_DIR, "ic_samples_def2_teacher.npz"),
            cfg_training_log,
            title="def2 ICs actually used during training",
            out_path=os.path.join(PLOTS_DEF2, "def2_ic_used.png"),
            show=False,
        )

    # =========================================================
    # Multi phase Plotting 
    # =========================================================
    defender_phase_metrics = []
    if do_phase_0:
        defender_phase_metrics.append(
            ("def0_teacher", os.path.join(OUT_DIR, "train_metrics_def0_teacher.npz"), "R_def_mean")
        )
    if do_phase_2:
        defender_phase_metrics.append(
            ("def1_teacher", os.path.join(OUT_DIR, "train_metrics_def1_teacher.npz"), "R_def_mean")
        )
    if do_phase_4:
        defender_phase_metrics.append(
            ("def2_teacher", os.path.join(OUT_DIR, "train_metrics_def2_teacher.npz"), "R_def_mean")
        )

    if len(defender_phase_metrics) >= 2:
        plot_compare_phases(
            defender_phase_metrics,
            metric=None,
            ylabel="Mean defender return",
            title="Defender return across phases",
            smooth="ema",
            smooth_param=0.2,
            show=False,
            out_dir=PLOTS_COMP,
            filename="compare__R_def_mean__def0_vs_def1_vs_def2.png",
        )

    attacker_phase_metrics = []
    if do_phase_1:
        attacker_phase_metrics.append(
            ("att1_teacher", os.path.join(OUT_DIR, "train_metrics_att1_teacher.npz"), "R_att_mean")
        )
    if do_phase_3:
        attacker_phase_metrics.append(
            ("att2_teacher", os.path.join(OUT_DIR, "train_metrics_att2_teacher.npz"), "R_att_mean")
        )

    if len(attacker_phase_metrics) >= 2:
        plot_compare_phases(
            attacker_phase_metrics,
            metric=None,
            ylabel="Mean attacker return",
            title="Attacker return across phases",
            smooth="ema",
            smooth_param=0.2,
            show=False,
            out_dir=PLOTS_COMP,
            filename="compare__R_att_mean__att1_vs_att2.png",
        )

    all_phase_metrics = []
    if do_phase_0:
        all_phase_metrics.append(("def0_teacher", os.path.join(OUT_DIR, "train_metrics_def0_teacher.npz"), "R_def_mean"))
    if do_phase_1:
        all_phase_metrics.append(("att1_teacher", os.path.join(OUT_DIR, "train_metrics_att1_teacher.npz"), "R_att_mean"))
    if do_phase_2:
        all_phase_metrics.append(("def1_teacher", os.path.join(OUT_DIR, "train_metrics_def1_teacher.npz"), "R_def_mean"))
    if do_phase_3:
        all_phase_metrics.append(("att2_teacher", os.path.join(OUT_DIR, "train_metrics_att2_teacher.npz"), "R_att_mean"))
    if do_phase_4:
        all_phase_metrics.append(("def2_teacher", os.path.join(OUT_DIR, "train_metrics_def2_teacher.npz"), "R_def_mean"))

    if len(all_phase_metrics) >= 2:
        plot_compare_phases(
            all_phase_metrics,
            metric=None,
            ylabel="Mean return",
            title="Return across phases",
            smooth="ema",
            smooth_param=0.2,
            show=False,
            out_dir=PLOTS_COMP,
            filename="compare__R_mean__def0_vs_att1_vs_def1_vs_att2_vs_def2.png",
        )

    if not do_phase_0:
        def0_meta = None
    
    if not do_phase_1:
        att1_meta = None

    if not do_phase_2:
        def1_meta = None
    
    if not do_phase_3:
        att2_meta = None

    if not do_phase_4:
        def2_meta = None

        

    # record a final summary block too
    runlog.set_config("final_outputs", {
        "def0": def0_meta,
        "att1": att1_meta,
        "def1": def1_meta,
        "att2": att2_meta,
        "def2": def2_meta,
        "plots_root": PLOTS_ROOT,
    })

    print("Saved plots under:", PLOTS_ROOT)
    print(" -", PLOTS_DEF0)
    print(" -", PLOTS_ATT1)
    print(" -", PLOTS_DEF1)
    print(" -", PLOTS_ATT2)
    print(" -", PLOTS_DEF2)
    print(" -", PLOTS_COMP)

    print("\n===== ALL PHASES COMPLETE =====")
    if do_phase_0:
        print(f"Defender_0 teacher:  {def0_teacher_ckpt if def0_teacher_ckpt else '(skipped)'}")
        print(f"Defender_0 student:  {def0_student_ckpt if def0_student_ckpt else '(skipped)'}")
    else:
        print("Defender_0 training skipped")


    if do_phase_1:
        print(f"Attacker_1 teacher:  {att1_teacher_ckpt if att1_teacher_ckpt else '(skipped)'}")
        print(f"Attacker_1 student:  {att1_student_ckpt if att1_student_ckpt else '(skipped)'}")
    else:
        print("Attacker_1 training skipped")
    
    if do_phase_2:
        print(f"Defender_1 teacher:  {def1_teacher_ckpt if def1_teacher_ckpt else '(skipped)'}")
        print(f"Defender_1 student:  {def1_student_ckpt if def1_student_ckpt else '(skipped)'}")
    else:
        print("Defender_1 training skipped")

    if do_phase_3:
        print(f"Attacker_2 teacher:  {att2_teacher_ckpt if att2_teacher_ckpt else '(skipped)'}")
        print(f"Attacker_2 student:  {att2_student_ckpt if att2_student_ckpt else '(skipped)'}")
    else:
        print("Attacker_2 training skipped")

    if do_phase_4:
        print(f"Defender_2 teacher:  {def2_teacher_ckpt if def2_teacher_ckpt else '(skipped)'}")
        print(f"Defender_2 student:  {def2_student_ckpt if def2_student_ckpt else '(skipped)'}")
    else:
        print("Defender_2 training skipped")
