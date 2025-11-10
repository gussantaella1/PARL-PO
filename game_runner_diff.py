# game_runner_diff.py
# RL-only rollout for Diff-Nash policy; HCW dynamics; no attitude/FOV (identity stubs)

from __future__ import annotations
from typing import Dict, Any
import numpy as np

from dyn_models import hcw_mean_motion, hcw_discrete_mats, as_numpy_const
from rl_infer_diff import RLPolicyDiff


def run_rhc_with_rl_and_collect_frames_3d_diff(cfg: Dict[str, Any],
                                               steps: int | None = None,
                                               turn_len: int | None = None):
    """
    RL-only rollout (no attitude/FOV). Matches Diff-Nash training obs:
        obs = [p1-center, p2-center, (p2-p1), v1, v2]
    Returns keys compatible with your animator; attitude fields are identity stubs.
    """

    # -------------------- basics & dims --------------------
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx = 2 * D         # [p, v] per agent
    nu = D             # LVLH accel per agent
    T, dt = int(cfg["T"]), float(cfg["dt"])

    # -------------------- dynamics (Ad,Bd) --------------------
    dyn_name = (cfg.get("dynamics") or "hcw").lower()
    if dyn_name != "hcw":
        raise ValueError("This runner expects 'hcw' dynamics.")
    n = hcw_mean_motion(cfg.get("hcw", {}))
    Ad_mx, Bd_mx = hcw_discrete_mats(n, dt)
    Ad_np, Bd_np = as_numpy_const(Ad_mx), as_numpy_const(Bd_mx)

    def step_plant_single(x, u):
        x = np.asarray(x, dtype=np.float32)
        u = np.asarray(u, dtype=np.float32)
        return (Ad_np @ x + Bd_np @ u).astype(np.float32)

    # -------------------- sim length --------------------
    if steps is None:
        steps = int(cfg.get("T_eval", cfg.get("steps", 60)))
    if turn_len is None:
        turn_len = 1  # unused here; kept for API symmetry

    # -------------------- arena & center --------------------
    ar = cfg.setdefault("arena", {})
    ar.setdefault("type", "sphere")
    ar.setdefault("cx", 0.0); ar.setdefault("cy", 0.0); ar.setdefault("cz", 0.0)
    ar.setdefault("r", 30.0)
    center = np.array([ar["cx"], ar["cy"], (ar.get("cz", 0.0) if D == 3 else 0.0)], dtype=np.float32)[:D]

    # -------------------- initial states --------------------
    x1 = np.asarray(cfg["x0"][0], dtype=np.float32)[:2*D].copy()
    x2 = np.asarray(cfg["x0"][1], dtype=np.float32)[:2*D].copy()

    # -------------------- obs normalization (optional) --------------------
    use_obsnorm = bool(cfg.get("obsnorm", False))
    obs_mean = None; obs_std = None
    if use_obsnorm and "obs_stats" in cfg:
        import os
        mp = cfg["obs_stats"].get("mean_path", None)
        sp = cfg["obs_stats"].get("std_path", None)
        if mp and os.path.exists(mp): obs_mean = np.load(mp).astype(np.float32)
        if sp and os.path.exists(sp): obs_std  = np.load(sp).astype(np.float32)

    def build_train_obs(x1_vec: np.ndarray, x2_vec: np.ndarray) -> np.ndarray:
        p1 = x1_vec[:D]; v1 = x1_vec[D:2*D]
        p2 = x2_vec[:D]; v2 = x2_vec[D:2*D]
        obs = np.concatenate([p1 - center, p2 - center, (p2 - p1), v1, v2]).astype(np.float32)
        if use_obsnorm and (obs_mean is not None) and (obs_std is not None):
            obs = (obs - obs_mean) / (obs_std + 1e-8)
        return obs

    # -------------------- policy wrapper --------------------
    deterministic = bool(cfg.get("rl_eval_deterministic", True))
    umax = float(cfg.get("umax", 5e-4))
    debug_actions = bool(cfg.get("debug_actions", False))

    pol = RLPolicyDiff(cfg, device=cfg.get("device", "cpu"))
    # Optional safety check:
    din_def, din_att = pol.verify_ckpt_compat()
    if din_def != 5*D or din_att != 5*D:
        print(f"[warn] ckpt expects obs_dim (def,att)=({din_def},{din_att}) "
              f"but runner will feed {5*D}. Make sure checkpoints are Diff-Nash.")

    def _act(which: str, x1_vec: np.ndarray, x2_vec: np.ndarray, deterministic: bool):
        obs = build_train_obs(x1_vec, x2_vec)
        return (pol.act_def_obs(obs, deterministic=deterministic)
                if which == "def" else pol.act_att_obs(obs, deterministic=deterministic))

    # -------------------- tiny helpers for animator compatibility --------------------
    def _p3_row(x_row):
        return (float(x_row[0]), float(x_row[1]), float(x_row[2])) if D == 3 else \
               (float(x_row[0]), float(x_row[1]), 0.0)

    def _p3_vec(x_vec):
        return (float(x_vec[0]), float(x_vec[1]), float(x_vec[2])) if D == 3 else \
               (float(x_vec[0]), float(x_vec[1]), 0.0)

    def _identity_R():
        return np.eye(3, dtype=float)

    # -------------------- logs for animator (attitude = stubs) --------------------
    plan_hist1, plan_hist2 = [], []
    plan_att1,  plan_att2  = [], []
    exec_xyz1,  exec_xyz2  = [], []
    exec_att1,  exec_att2  = [], []
    phi_hist1,  phi_hist2  = [], []
    fov_axis_hist, fov_seen_mask = [], []

    # t=0 logs
    exec_xyz1.append(_p3_vec(x1)); exec_xyz2.append(_p3_vec(x2))
    exec_att1.append({"R": _identity_R(), "phi": 0.0}); phi_hist1.append(0.0)
    exec_att2.append({"R": _identity_R(), "phi": 0.0}); phi_hist2.append(0.0)
    fov_axis_hist.append(None); fov_seen_mask.append(False)

    # ==================== ROLLOUT ====================
    for k in range(steps):
        # 1) actions from policies
        u1 = np.clip(np.asarray(_act("def", x1, x2, deterministic), dtype=np.float32), -umax, +umax)
        u2 = np.clip(np.asarray(_act("att", x1, x2, deterministic), dtype=np.float32), -umax, +umax)
        if debug_actions and k < 3:
            obs_dbg = build_train_obs(x1, x2)
            print(f"[k={k:02d}] ||a_def||={np.linalg.norm(u1):.3g}, ||a_att||={np.linalg.norm(u2):.3g}, "
                  f"first obs entries={obs_dbg[:6]}")

        # 2) embed into full control vectors
        u1_full = np.zeros(nu, dtype=np.float32); u2_full = np.zeros(nu, dtype=np.float32)
        u1_full[:D] = u1;                            u2_full[:D] = u2

        # 3) plant step
        x1 = step_plant_single(x1, u1_full)
        x2 = step_plant_single(x2, u2_full)

        # 4) "plan" horizons (flat, just repeat current pos)
        plan1 = [_p3_row(x1)] * T
        plan2 = [_p3_row(x2)] * T
        plan_hist1.append(plan1); plan_hist2.append(plan2)

        # 5) attitude stubs (keep shapes)
        I = _identity_R()
        att_stub = [{"R": I, "phi": 0.0} for _ in range(T)]
        plan_att1.append(att_stub); plan_att2.append(att_stub)

        exec_att1.append({"R": I, "phi": 0.0})
        exec_att2.append({"R": I, "phi": 0.0})
        phi_hist1.append(0.0); phi_hist2.append(0.0)

        exec_xyz1.append(_p3_vec(x1)); exec_xyz2.append(_p3_vec(x2))
        fov_axis_hist.append(None); fov_seen_mask.append(False)

    # -------------------- pack results --------------------
    return {
        "plan_hist1": plan_hist1, "plan_hist2": plan_hist2,
        "plan_att1":  plan_att1,  "plan_att2":  plan_att2,      # identity-R horizons
        "exec1_xyz":  exec_xyz1,  "exec2_xyz":  exec_xyz2,
        "exec_att1":  exec_att1,  "exec_att2":  exec_att2,      # identity-R per step
        "phi_hist1":  phi_hist1,  "phi_hist2":  phi_hist2,      # zeros
        "fov_axis_hist": fov_axis_hist, "fov_seen_mask": fov_seen_mask,
    }
