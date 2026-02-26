# game_runner_diff.py
# RL-only rollout for Diff-Nash policy; supports HCW (LTI), elliptic LTV, and two-body nonlinear.
# No attitude/FOV (identity stubs)

from __future__ import annotations
from typing import Dict, Any
import numpy as np

from rl_infer import RLPolicyDiff
from ukf_estimator import KF_CV


def run_rhc_with_rl_and_collect_frames_3d_diff(
    cfg: Dict[str, Any],
    steps: int | None = None,
    turn_len: int | None = None
):
    # -------------------- basics & dims --------------------
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx = 2 * D         # [p, v] per agent in D dims
    nu = D             # accel command in D dims
    T, dt = int(cfg["T"]), float(cfg["dt"])

    dyn_name = (cfg.get("dynamics") or "hcw").lower()

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
    center = np.array(
        [ar["cx"], ar["cy"], (ar.get("cz", 0.0) if D == 3 else 0.0)],
        dtype=np.float32
    )[:D]

    # -------------------- initial states --------------------
    x1 = np.asarray(cfg["x0"][0], dtype=np.float32)[:nx].copy()
    x2 = np.asarray(cfg["x0"][1], dtype=np.float32)[:nx].copy()

    # -------------------- obs normalization (optional) --------------------
    use_obsnorm = bool(cfg.get("obsnorm", False))
    obs_mean = None; obs_std = None
    if use_obsnorm and "obs_stats" in cfg:
        import os
        mp = cfg["obs_stats"].get("mean_path", None)
        sp = cfg["obs_stats"].get("std_path", None)
        if mp and os.path.exists(mp):
            obs_mean = np.load(mp).astype(np.float32)
        if sp and os.path.exists(sp):
            obs_std  = np.load(sp).astype(np.float32)

    def build_train_obs(x1_vec: np.ndarray, x2_vec: np.ndarray) -> np.ndarray:
        p1 = x1_vec[:D]; v1 = x1_vec[D:2*D]
        p2 = x2_vec[:D]; v2 = x2_vec[D:2*D]
        obs = np.concatenate(
            [p1 - center, p2 - center, (p2 - p1), v1, v2],
            dtype=np.float32
        )
        if use_obsnorm and (obs_mean is not None) and (obs_std is not None):
            obs = (obs - obs_mean) / (obs_std + 1e-8)
        return obs

    # -------------------- policy wrapper --------------------
    deterministic = bool(cfg.get("rl_eval_deterministic", True))
    umax = float(cfg.get("umax", 5e-4))
    debug_actions = bool(cfg.get("debug_actions", False))

    pol = RLPolicyDiff(cfg, device=cfg.get("device", "cpu"))
    din_def, din_att = pol.verify_ckpt_compat()
    if din_def != 5*D or din_att != 5*D:
        print(
            f"[warn] ckpt expects obs_dim (def,att)=({din_def},{din_att}) "
            f"but runner will feed {5*D}. Make sure checkpoints are Diff-Nash."
        )

    # -------------------- tiny helpers for animator compatibility --------------------
    def _p3_row(x_row):
        return (float(x_row[0]), float(x_row[1]), float(x_row[2])) if D == 3 else \
               (float(x_row[0]), float(x_row[1]), 0.0)

    def _p3_vec(x_vec):
        return (float(x_vec[0]), float(x_vec[1]), float(x_vec[2])) if D == 3 else \
               (float(x_vec[0]), float(x_vec[1]), 0.0)

    def _identity_R():
        return np.eye(3, dtype=float)

    def _u3(uD: np.ndarray):
        """Pad D-accel to 3 for two-body RTN."""
        uD = np.asarray(uD, float).reshape(-1)
        if D == 3:
            return uD[:3]
        return np.array([uD[0], uD[1], 0.0], dtype=float)

    def _x6_from_xD(xD: np.ndarray) -> np.ndarray:
        """Embed D-state into 6-state [R,T,N,Rd,Td,Nd] for dynamics that operate in 3D."""
        xD = np.asarray(xD, dtype=np.float32).reshape(-1)
        if D == 3:
            return xD.astype(np.float32)
        # D==2: [x,y,vx,vy] -> [x,y,0,vx,vy,0]
        return np.array([xD[0], xD[1], 0.0, xD[2], xD[3], 0.0], dtype=np.float32)

    def _xD_from_x6(x6: np.ndarray) -> np.ndarray:
        """Project 6-state back to D-state."""
        x6 = np.asarray(x6, dtype=np.float32).reshape(-1)
        if D == 3:
            return x6.astype(np.float32)
        return np.array([x6[0], x6[1], x6[3], x6[4]], dtype=np.float32)

    def _act(which: str,
             x1_vec: np.ndarray,
             x2_true: np.ndarray,
             x2_est_vec: np.ndarray,
             deterministic: bool):
        if which == "def":
            x2_for_obs = x2_est_vec if (use_ukf and ukf12 is not None) else x2_true
        else:
            x2_for_obs = x2_true

        obs = build_train_obs(x1_vec, x2_for_obs)
        return (pol.act_def_obs(obs, deterministic=deterministic)
                if which == "def" else pol.act_att_obs(obs, deterministic=deterministic))

    # -------------------- dynamics selection (LTI/LTV/nonlinear) --------------------
    # Prefer already-built cfg["dyn"] if present, otherwise build locally.
    dyn = cfg.get("dyn", {}) if isinstance(cfg.get("dyn", {}), dict) else {}

    dyn_type = dyn.get("type", None)   # "lti" | "ltv" | "nonlinear"
    Ad = dyn.get("Ad", None)
    Bd = dyn.get("Bd", None)
    Ad_seq = dyn.get("Ad_seq", None)
    Bd_seq = dyn.get("Bd_seq", None)
    chief_cache = dyn.get("chief_cache", None)

    # Local build if needed
    if dyn_name == "hcw":
        if Ad is None or Bd is None:
            from dyn_models import hcw_mean_motion, hcw_discrete_mats, as_numpy_const
            n = hcw_mean_motion(cfg.get("hcw", {}))
            Ad_mx, Bd_mx = hcw_discrete_mats(n, dt)   # typically 6x6, 6x3
            Ad = as_numpy_const(Ad_mx).astype(np.float32)
            Bd = as_numpy_const(Bd_mx).astype(np.float32)
        dyn_type = "lti"

    elif dyn_name in ("elliptic_ltv", "elliptical_ltv", "th", "tschauner_hempel"):
        if Ad_seq is None or Bd_seq is None or chief_cache is None:
            from dyn_models import chief_orbit_cache_rtn, linearize_two_body_rtn_discrete
            orb = cfg.get("chief_orbit", {})
            chief_cache = chief_orbit_cache_rtn(orb, dt=dt, N=steps)  # only need up to steps
            Ad_seq, Bd_seq = linearize_two_body_rtn_discrete(chief_cache, dt=dt, eps=1e-5)
            Ad_seq = Ad_seq.astype(np.float32)
            Bd_seq = Bd_seq.astype(np.float32)
        dyn_type = "ltv"

    elif dyn_name in ("two_body", "twobody", "two-body"):
        if chief_cache is None:
            from dyn_models import chief_orbit_cache_rtn
            orb = cfg.get("chief_orbit", {})
            chief_cache = chief_orbit_cache_rtn(orb, dt=dt, N=steps)
        dyn_type = "nonlinear"

    else:
        raise ValueError(f"Unknown dynamics '{cfg.get('dynamics')}'")

    # Unified plant step: works even if D==2 by embedding/projection
    def step_plant_single(xD: np.ndarray, uD: np.ndarray, k: int) -> np.ndarray:
        x6 = _x6_from_xD(xD)
        u3 = _u3(uD)

        if dyn_type == "lti":
            # HCW: Ad,Bd are (6x6, 6x3)
            x6n = (Ad @ x6 + Bd @ u3).astype(np.float32)

        elif dyn_type == "ltv":
            # Elliptic LTV: Ad_seq,Bd_seq are (N,6,6) and (N,6,3)
            Ak = Ad_seq[k]
            Bk = Bd_seq[k]
            x6n = (Ak @ x6 + Bk @ u3).astype(np.float32)

        elif dyn_type == "nonlinear":
            from dyn_models import two_body_step_rtn
            x6n = two_body_step_rtn(x6, u3, k, chief_cache).astype(np.float32)

        else:
            raise RuntimeError(f"dyn_type='{dyn_type}' not recognized")

        return _xD_from_x6(x6n)

    # -------------------- logs for animator (attitude = stubs) --------------------
    plan_hist1, plan_hist2 = [], []
    plan_att1,  plan_att2  = [], []
    exec_xyz1,  exec_xyz2  = [], []
    exec_att1,  exec_att2  = [], []
    phi_hist1,  phi_hist2  = [], []
    fov_axis_hist, fov_seen_mask = [], []

    # command logs
    u_cmd_1, u_cmd_2 = [], []
    u_cmd_norm_1, u_cmd_norm_2 = [], []

    # -------------------- optional UKFs (HCW-only) --------------------
    use_ukf = bool(cfg.get("use_ukf", False))
    if use_ukf and dyn_name != "hcw":
        print("[warn] UKF in this runner is HCW-only; disabling use_ukf for non-HCW dynamics.")
        use_ukf = False

    est_hist = None
    x2_est = x2.copy()  # default: truth

    if use_ukf:
        ukf_cfg = cfg.get("ukf", {})

        sigma_az = float(ukf_cfg.get("sigma_az", np.deg2rad(0.5)))
        sigma_el = float(ukf_cfg.get("sigma_el", np.deg2rad(0.5)))
        pos_std0 = float(ukf_cfg.get("init_pos_std", 0.2 * float(ar["r"])))
        vel_std0 = float(ukf_cfg.get("init_vel_std", 0.01))
        Q_scale  = float(ukf_cfg.get("Q_scale", 1e-5))

        who = (ukf_cfg.get("who") or "both").lower()
        meas_every = int(ukf_cfg.get("every", 1))

        # KF expects 3D state; pad if D==2
        P0 = np.diag([pos_std0**2]*3 + [vel_std0**2]*3)
        Q  = Q_scale * np.diag([1, 1, 1, 1, 1, 1])
        Rm = np.diag([sigma_az**2, sigma_el**2])
        hcw_params = cfg.get("hcw", {})

        # Filter state is [pos(3), vel(3)] in RTN
        x2_6 = _x6_from_xD(x2)
        x1_6 = _x6_from_xD(x1)

        ukf12 = KF_CV(
            np.r_[x2_6[:3], x2_6[3:6]], P0, Q, Rm, dt,
            kind="ukf", dyn="hcw", hcw=hcw_params
        ) if who in ("both", "1->2") else None

        ukf21 = KF_CV(
            np.r_[x1_6[:3], x1_6[3:6]], P0, Q, Rm, dt,
            kind="ukf", dyn="hcw", hcw=hcw_params
        ) if who in ("both", "2->1") else None

        est_hist = {}
        if ukf12 is not None:
            est_hist["est12_xyz"] = [ukf12.x[:3].copy()]
            est_hist["meas12_azel"] = [None]
            # update x2_est (project back to D)
            x2_est = _xD_from_x6(np.r_[ukf12.x[:3], ukf12.x[3:6]]).astype(np.float32)

        if ukf21 is not None:
            est_hist["est21_xyz"] = [ukf21.x[:3].copy()]
            est_hist["meas21_azel"] = [None]

        meas_std_az, meas_std_el = sigma_az, sigma_el
    else:
        ukf12 = ukf21 = None
        meas_every = 1
        meas_std_az = meas_std_el = None

    # t=0 logs
    exec_xyz1.append(_p3_vec(x1)); exec_xyz2.append(_p3_vec(x2))
    exec_att1.append({"R": _identity_R(), "phi": 0.0}); phi_hist1.append(0.0)
    exec_att2.append({"R": _identity_R(), "phi": 0.0}); phi_hist2.append(0.0)
    fov_axis_hist.append(None); fov_seen_mask.append(False)

    # ==================== ROLLOUT ====================
    for k in range(steps):
        # 1) actions from policies
        u1 = np.clip(
            np.asarray(_act("def", x1, x2, x2_est, deterministic), dtype=np.float32),
            -umax, +umax
        )
        u2 = np.clip(
            np.asarray(_act("att", x1, x2, x2_est, deterministic), dtype=np.float32),
            -umax, +umax
        )

        # log commanded thrust (post-clip)
        u1_3 = _u3(u1)
        u2_3 = _u3(u2)
        u_cmd_1.append(u1_3.copy())
        u_cmd_2.append(u2_3.copy())
        u_cmd_norm_1.append(float(np.linalg.norm(u1_3)))
        u_cmd_norm_2.append(float(np.linalg.norm(u2_3)))

        if debug_actions and k < 3:
            x2_for_dbg = x2_est if (use_ukf and ukf12 is not None) else x2
            obs_dbg = build_train_obs(x1, x2_for_dbg)
            print(f"[k={k:02d}] ||a_def||={np.linalg.norm(u1):.3g}, "
                  f"||a_att||={np.linalg.norm(u2):.3g}, "
                  f"first obs entries={obs_dbg[:6]}")

        # 2) plant step (supports LTI/LTV/nonlinear)
        x1 = step_plant_single(x1, u1, k)
        x2 = step_plant_single(x2, u2, k)

        # 3) UKF estimation (HCW-only)
        if use_ukf and est_hist is not None:
            idx  = len(exec_xyz1)
            take = (idx % meas_every) == 0

            def _to3_pos(xD):
                p = np.zeros(3, dtype=float)
                p[:D] = np.asarray(xD[:D], float)
                return p

            def _to3_u(uD):
                return _u3(uD)

            if ukf12 is not None:
                ukf12.predict(dt, u=_to3_u(u2), u_cov=None)
                if take:
                    p_obs = _to3_pos(x1)
                    p_tgt = _to3_pos(x2)
                    d = p_tgt - p_obs
                    d /= (np.linalg.norm(d) + 1e-12)
                    b_b = d
                    az = np.arctan2(b_b[1], b_b[0]) + np.random.randn()*meas_std_az
                    el = np.arctan2(
                        b_b[2],
                        np.sqrt(max(b_b[0]**2 + b_b[1]**2, 1e-18))
                    ) + np.random.randn()*meas_std_el
                    z = np.array([az, el], float)
                    ukf12.update(z, p_obs=p_obs, R_wb=_identity_R())
                    est_hist["meas12_azel"].append((p_obs.copy(), z.copy()))
                else:
                    est_hist["meas12_azel"].append(None)
                est_hist["est12_xyz"].append(ukf12.x[:3].copy())

            if ukf21 is not None:
                ukf21.predict(dt, u=_to3_u(u1), u_cov=None)
                if take:
                    p_obs = _to3_pos(x2)
                    p_tgt = _to3_pos(x1)
                    d = p_tgt - p_obs
                    d /= (np.linalg.norm(d) + 1e-12)
                    b_b = d
                    az = np.arctan2(b_b[1], b_b[0]) + np.random.randn()*meas_std_az
                    el = np.arctan2(
                        b_b[2],
                        np.sqrt(max(b_b[0]**2 + b_b[1]**2, 1e-18))
                    ) + np.random.randn()*meas_std_el
                    z = np.array([az, el], float)
                    ukf21.update(z, p_obs=p_obs, R_wb=_identity_R())
                    est_hist["meas21_azel"].append((p_obs.copy(), z.copy()))
                else:
                    est_hist["meas21_azel"].append(None)
                est_hist["est21_xyz"].append(ukf21.x[:3].copy())

            if ukf12 is not None:
                # update x2_est from ukf12 state (project back to D)
                x2_est = _xD_from_x6(np.r_[ukf12.x[:3], ukf12.x[3:6]]).astype(np.float32)

        # 4) "plan" horizons (flat, just repeat current pos)
        plan1 = [_p3_row(x1)] * T
        plan2 = [_p3_row(x2)] * T
        plan_hist1.append(plan1); plan_hist2.append(plan2)

        # 5) attitude stubs
        I = _identity_R()
        att_stub = [{"R": I, "phi": 0.0} for _ in range(T)]
        plan_att1.append(att_stub); plan_att2.append(att_stub)

        exec_att1.append({"R": I, "phi": 0.0})
        exec_att2.append({"R": I, "phi": 0.0})
        phi_hist1.append(0.0); phi_hist2.append(0.0)

        exec_xyz1.append(_p3_vec(x1)); exec_xyz2.append(_p3_vec(x2))
        fov_axis_hist.append(None); fov_seen_mask.append(False)

    # -------------------- pack results --------------------
    out = {
        "plan_hist1": plan_hist1, "plan_hist2": plan_hist2,
        "plan_att1":  plan_att1,  "plan_att2":  plan_att2,
        "exec1_xyz":  exec_xyz1,  "exec2_xyz":  exec_xyz2,
        "exec_att1":  exec_att1,  "exec_att2":  exec_att2,
        "phi_hist1":  phi_hist1,  "phi_hist2":  phi_hist2,
        "fov_axis_hist": fov_axis_hist, "fov_seen_mask": fov_seen_mask,

        # commanded thrust logs
        "u_cmd_all": [
            np.asarray(u_cmd_1, dtype=float),
            np.asarray(u_cmd_2, dtype=float),
        ],
        "u_cmd_norm_all": [
            np.asarray(u_cmd_norm_1, dtype=float),
            np.asarray(u_cmd_norm_2, dtype=float),
        ],
    }
    if est_hist is not None:
        out.update(est_hist)
    return out
