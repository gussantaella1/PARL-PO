# game_runner.py
# RL-only rollout for Diff-Nash policy; supports HCW (LTI), elliptic LTV, and two-body nonlinear.
# No attitude/FOV (identity stubs)

from __future__ import annotations
from typing import Dict, Any, List
import time
import numpy as np

from paper_baseline_runner import _build_step_plant_single, _identity_R, _p3
from rl_infer import RLPolicyDiff
from rl_infer_1v2 import RLPolicy_Multi
from ukf_estimator import KF_CV, _azel_from_body_vec, _body_bearing_from_world


def _u3_from_action(u: np.ndarray, D: int) -> np.ndarray:
    u = np.asarray(u, float).reshape(-1)
    if D == 3:
        return u[:3]
    return np.array([u[0], u[1], 0.0], dtype=float)


def _normalize_kf_action_access(mode: Any) -> str:
    key = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
    if key == "groundtruth":
        key = "ground_truth"
    if key == "inferred":
        key = "measured"
    valid = {"ground_truth", "measured", "none"}
    if key not in valid:
        raise ValueError(f"Unsupported estimator action_access='{mode}'. Expected one of {sorted(valid)}.")
    return key


def _normalize_kf_control_noise_std(noise_std: Any, default_std: float) -> np.ndarray:
    arr = np.asarray(noise_std if noise_std is not None else default_std, dtype=float)
    if arr.ndim == 0:
        std = np.full(3, float(arr), dtype=float)
    else:
        arr = arr.reshape(-1)
        if arr.size != 3:
            raise ValueError(
                f"Expected estimator action_meas_std to be a scalar or length-3 sequence, got shape {arr.shape}."
            )
        std = arr.astype(float)
    return np.maximum(std, 0.0)


def _normalize_estimator_kind(cfg: Dict[str, Any]) -> str:
    kind = cfg.get("estimator_kind", "ukf")
    key = str(kind).strip().lower()
    if key not in {"ukf", "ekf"}:
        raise ValueError(f"Unsupported estimator_kind='{kind}'. Expected 'ukf' or 'ekf'.")
    return key


def _kf_predict_control(
    action_access: str,
    ground_truth_u: np.ndarray | None,
    control_meas_std: np.ndarray | None = None,
    control_limit: float | None = None,
):
    if ground_truth_u is None:
        return None, None
    u_true = np.asarray(ground_truth_u, dtype=float).reshape(3)
    if action_access == "ground_truth":
        return u_true, None
    if action_access == "measured":
        std = np.asarray(control_meas_std if control_meas_std is not None else np.zeros(3), dtype=float).reshape(3)
        noise = np.random.normal(loc=0.0, scale=std, size=3)
        u_meas = u_true + noise
        if control_limit is not None and np.isfinite(control_limit):
            u_meas = np.clip(u_meas, -control_limit, control_limit)
        return u_meas, np.diag(std**2)
    return None, None


def _pack_multi_agent_rollout(
    *,
    plan_hist_all,
    plan_att_all,
    exec_xyz_all,
    exec_att_all,
    phi_hist_all,
    fov_axis_hist,
    fov_seen_mask,
    u_cmd_all,
    u_cmd_norm_all,
    u_real_all,
    u_real_norm_all,
    done_info,
    rollout_timing_sec,
    fuel_frac_all=None,
    thrust_all=None,
    mdot_all=None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "num_attackers": max(0, len(exec_xyz_all) - 1),
        "plan_hist_all": plan_hist_all,
        "plan_att_all": plan_att_all,
        "exec_xyz_all": exec_xyz_all,
        "exec_att_all": exec_att_all,
        "phi_hist_all": phi_hist_all,
        "fov_axis_hist": fov_axis_hist,
        "fov_seen_mask": fov_seen_mask,
        "u_cmd_all": [np.asarray(u, dtype=float) for u in u_cmd_all],
        "u_cmd_norm_all": [np.asarray(u, dtype=float) for u in u_cmd_norm_all],
        "u_real_all": [np.asarray(u, dtype=float) for u in u_real_all],
        "u_real_norm_all": [np.asarray(u, dtype=float) for u in u_real_norm_all],
        "done_info": done_info,
        "rollout_timing_sec": rollout_timing_sec,
    }

    for idx in range(len(exec_xyz_all)):
        key = idx + 1
        out[f"plan_hist{key}"] = plan_hist_all[idx]
        out[f"plan_att{key}"] = plan_att_all[idx]
        out[f"exec{key}_xyz"] = exec_xyz_all[idx]
        out[f"exec_att{key}"] = exec_att_all[idx]
        out[f"phi_hist{key}"] = phi_hist_all[idx]

    if fuel_frac_all is not None:
        out["fuel_frac_all"] = [np.asarray(v, dtype=float) for v in fuel_frac_all]
    if thrust_all is not None:
        out["thrust_all"] = [np.asarray(v, dtype=float) for v in thrust_all]
    if mdot_all is not None:
        out["mdot_all"] = [np.asarray(v, dtype=float) for v in mdot_all]

    return out


def _run_rhc_with_rl_and_collect_frames_3d_multi(
    cfg: Dict[str, Any],
    steps: int | None = None,
    turn_len: int | None = None,
):
    t_fn0 = time.perf_counter()

    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx = 2 * D
    T = int(cfg["T"])
    dt = float(cfg["dt"])
    Na = int(cfg.get("num_attackers", 1))

    if Na <= 1:
        raise ValueError("_run_rhc_with_rl_and_collect_frames_3d_multi expects num_attackers > 1.")

    if steps is None:
        steps = int(cfg.get("T_eval", cfg.get("T", cfg.get("steps", 60))))
    if turn_len is None:
        turn_len = 1

    ar = cfg.setdefault("arena", {})
    ar.setdefault("type", "sphere")
    ar.setdefault("cx", 0.0)
    ar.setdefault("cy", 0.0)
    ar.setdefault("cz", 0.0)
    ar.setdefault("r", 30.0)
    if ar["type"] != "sphere":
        raise ValueError("Only spherical arena is supported in this rollout helper.")

    center = np.array(
        [ar["cx"], ar["cy"], (ar.get("cz", 0.0) if D == 3 else 0.0)],
        dtype=np.float32,
    )[:D]
    arena_r = float(ar["r"])

    x0 = np.asarray(cfg["x0"], dtype=np.float32)
    if x0.shape[0] < 1 + Na:
        raise ValueError(f"cfg['x0'] must contain defender plus {Na} attackers for num_attackers={Na}.")

    xD = x0[0, :nx].copy()
    xA_list = [x0[i + 1, :nx].copy() for i in range(Na)]

    deterministic = bool(cfg.get("rl_eval_deterministic", True))
    umax = float(cfg.get("umax", 5e-4))
    debug_actions = bool(cfg.get("debug_actions", False))
    stop_on_done = bool(cfg.get("stop_on_done", True))

    use_fuel = bool(cfg.get("fuel", {}).get("enable"))
    g0 = 9.80665

    if use_fuel:
        fuel_cfg = cfg["fuel"]
        fdef = fuel_cfg["def"]
        fatt = fuel_cfg["att"]

        m0_def = float(fdef["m0"])
        mdry_def = float(fdef["m_dry"])
        Tmax_def = float(fdef["Tmax"])
        Isp_def = float(fdef["Isp"])

        m0_att = float(fatt["m0"])
        mdry_att = float(fatt["m_dry"])
        Tmax_att = float(fatt["Tmax"])
        Isp_att = float(fatt["Isp"])

        m_def = m0_def
        m_att = [m0_att for _ in range(Na)]
    else:
        m0_def = mdry_def = Tmax_def = Isp_def = None
        m0_att = mdry_att = Tmax_att = Isp_att = None
        m_def = None
        m_att = [None for _ in range(Na)]

    def apply_propulsion(
        a_cmd: np.ndarray,
        m: float,
        m_dry: float,
        Tmax: float,
        Isp: float,
    ):
        a_cmd = np.asarray(a_cmd, dtype=float)

        if m <= m_dry + 1e-9:
            return np.zeros_like(a_cmd, dtype=np.float32), float(m_dry), True, 0.0, 0.0

        F_req = m * a_cmd
        F_req_norm = np.linalg.norm(F_req)

        if F_req_norm > Tmax:
            F = F_req * (Tmax / (F_req_norm + 1e-9))
        else:
            F = F_req

        a_real = F / max(m, 1e-9)
        thrust_norm = float(np.linalg.norm(F))
        mdot = thrust_norm / (Isp * g0)
        m_next = max(m_dry, m - mdot * dt)
        fuel_depleted = bool(m_next <= m_dry + 1e-9)
        return a_real.astype(np.float32), float(m_next), fuel_depleted, thrust_norm, float(mdot)

    use_obsnorm = bool(cfg.get("obsnorm", False))
    obs_mean = None
    obs_std = None
    obs_expected = (2 + 3 * Na) * D + (2 if use_fuel else 0)

    if use_obsnorm and "obs_stats" in cfg:
        import os

        mp = cfg["obs_stats"].get("mean_path", None)
        sp = cfg["obs_stats"].get("std_path", None)

        if mp and os.path.exists(mp):
            obs_mean = np.load(mp).astype(np.float32)
        if sp and os.path.exists(sp):
            obs_std = np.load(sp).astype(np.float32)

        if obs_mean is not None and obs_mean.shape[0] != obs_expected:
            raise ValueError(f"obs_mean has dim {obs_mean.shape[0]}, expected {obs_expected}.")
        if obs_std is not None and obs_std.shape[0] != obs_expected:
            raise ValueError(f"obs_std has dim {obs_std.shape[0]}, expected {obs_expected}.")

    def build_train_obs(
        xD_vec: np.ndarray,
        xA_vecs: List[np.ndarray],
        m_def_cur: float | None,
        m_att_cur: List[float | None],
    ) -> np.ndarray:
        pD = xD_vec[:D]
        vD = xD_vec[D:2 * D]
        parts: List[np.ndarray] = [pD - center]

        for xA_vec in xA_vecs:
            parts.append(xA_vec[:D] - center)
        for xA_vec in xA_vecs:
            parts.append(xA_vec[:D] - pD)

        parts.append(vD)
        for xA_vec in xA_vecs:
            parts.append(xA_vec[D:2 * D])

        if use_fuel:
            fuel_frac_def = (m_def_cur - mdry_def) / (m0_def - mdry_def + 1e-9)
            fuel_frac_att = (m_att_cur[0] - mdry_att) / (m0_att - mdry_att + 1e-9)
            parts.append(np.array([np.clip(fuel_frac_def, 0.0, 1.0)], dtype=np.float32))
            parts.append(np.array([np.clip(fuel_frac_att, 0.0, 1.0)], dtype=np.float32))

        obs = np.concatenate(parts).astype(np.float32)
        if use_obsnorm and (obs_mean is not None) and (obs_std is not None):
            obs = (obs - obs_mean) / (obs_std + 1e-8)
        return obs

    step_plant_single, center_from_dyn, _ = _build_step_plant_single(cfg, steps=steps, D=D)
    center = np.asarray(center_from_dyn, dtype=np.float32)

    dyn_name = str(cfg.get("dynamics", "hcw")).lower()
    use_kf = bool(cfg.get("use_kf", False))
    estimator_kind = _normalize_estimator_kind(cfg)
    if use_kf and dyn_name != "hcw":
        print(f"[warn] {estimator_kind.upper()} in this runner is HCW-only; disabling estimator for non-HCW dynamics.")
        use_kf = False

    def _x6_from_xD(xD: np.ndarray) -> np.ndarray:
        xD = np.asarray(xD, dtype=np.float32).reshape(-1)
        if D == 3:
            return xD.astype(np.float32)
        return np.array([xD[0], xD[1], 0.0, xD[2], xD[3], 0.0], dtype=np.float32)

    def _xD_from_x6(x6: np.ndarray) -> np.ndarray:
        x6 = np.asarray(x6, dtype=np.float32).reshape(-1)
        if D == 3:
            return x6.astype(np.float32)
        return np.array([x6[0], x6[1], x6[3], x6[4]], dtype=np.float32)

    ukf_cfg = dict(cfg.get("ukf", {}))
    if not use_kf:
        ukf_cfg = {}
    ukf_list = []
    xA_est_list = [xA.copy() for xA in xA_list]
    if use_kf:
        sigma_az = float(ukf_cfg.get("sigma_az", np.deg2rad(0.5)))
        sigma_el = float(ukf_cfg.get("sigma_el", np.deg2rad(0.5)))
        meas_every = max(1, int(ukf_cfg.get("every", 1)))
        ukf_action_access = _normalize_kf_action_access(ukf_cfg.get("action_access", "ground_truth"))
        ukf_control_meas_std = _normalize_kf_control_noise_std(
            ukf_cfg.get("action_meas_std", 0.1 * max(abs(umax), 1e-3)),
            default_std=0.1 * max(abs(umax), 1e-3),
        )
        ukf_state_dim = 6
        pos_std0 = float(ukf_cfg.get("pos_std0", 0.2 * arena_r))
        vel_std0 = float(ukf_cfg.get("vel_std0", 0.01))
        init_mean_pos_std = float(ukf_cfg.get("init_mean_pos_std", 0.0))
        init_mean_vel_std = float(ukf_cfg.get("init_mean_vel_std", 0.0))
        init_mean_accel_std = float(ukf_cfg.get("init_mean_accel_std", 0.0))
        Q_scale = float(ukf_cfg.get("Q_scale", 1e-5))
        accel_std0 = float(ukf_cfg.get("accel_std0", max(abs(umax), 1e-3)))
        accel_q_scale = float(ukf_cfg.get("accel_Q_scale", Q_scale))

        if ukf_state_dim == 9:
            P0 = np.diag([pos_std0**2] * 3 + [vel_std0**2] * 3 + [accel_std0**2] * 3)
            Q = np.diag([Q_scale] * 6 + [accel_q_scale] * 3)
        else:
            P0 = np.diag([pos_std0**2] * 3 + [vel_std0**2] * 3)
            Q = Q_scale * np.diag([1, 1, 1, 1, 1, 1])
        Rm = np.diag([sigma_az**2, sigma_el**2])
        hcw_params = cfg.get("hcw", {})

        for j in range(Na):
            xA6 = _x6_from_xD(xA_list[j])
            xA6_mean = xA6.copy()
            xA6_mean[:3] += np.random.normal(0.0, init_mean_pos_std, size=3)
            xA6_mean[3:6] += np.random.normal(0.0, init_mean_vel_std, size=3)
            xA0 = xA6_mean
            ukf_j = KF_CV(
                xA0,
                P0,
                Q,
                Rm,
                dt,
                kind=estimator_kind,
                dyn="hcw",
                hcw=hcw_params,
            )
            ukf_list.append(ukf_j)
            xA_est_list[j] = _xD_from_x6(np.r_[ukf_j.x[:3], ukf_j.x[3:6]]).astype(np.float32)
    else:
        ukf_action_access = "ground_truth"

    pol = RLPolicy_Multi(cfg, device=cfg.get("device", "cpu"))
    din_def, din_att = pol.verify_ckpt_compat()
    if din_def != obs_expected or din_att != obs_expected:
        raise RuntimeError(
            f"Policy obs dims mismatch: defender={din_def}, attacker={din_att}, "
            f"expected={obs_expected} (D={D}, Na={Na}, use_fuel={use_fuel})."
        )

    oi = cfg.get("oi", {}) or {}
    oi_radius = float(oi.get("r", 0.0))
    oi_radius_norm = oi_radius / arena_r if arena_r > 0 else 0.0
    hit_buffer_def = 0.0
    hit_buffer_att = 0.0
    collision_radius_m = float(cfg.get("collision_radius_m"))
    arena_margin = float(cfg.get("arena_terminate_margin"))

    def check_done(
        xD_vec: np.ndarray,
        xA_vecs: List[np.ndarray],
        fuel_depleted_def: bool,
        fuel_depleted_att: List[bool],
    ):
        pD = xD_vec[:D]
        pA_list = [xA[:D] for xA in xA_vecs]

        rhoD = np.linalg.norm(pD - center) / max(arena_r, 1e-9)
        rhoA = [np.linalg.norm(pA - center) / max(arena_r, 1e-9) for pA in pA_list]

        oob_def = rhoD >= arena_margin
        oob_att = [rho >= arena_margin for rho in rhoA]

        def_hit_target = False
        att_hit_target = [False for _ in range(Na)]
        if oi_radius_norm > 0.0:
            def_hit_target = rhoD <= (1.0 + hit_buffer_def) * oi_radius_norm
            att_hit_target = [rho <= (1.0 + hit_buffer_att) * oi_radius_norm for rho in rhoA]

        collision = [False for _ in range(Na)]
        if collision_radius_m > 0.0:
            collision = [np.linalg.norm(pA - pD) <= collision_radius_m for pA in pA_list]

        done = (
            oob_def
            or any(oob_att)
            or def_hit_target
            or any(att_hit_target)
            or any(collision)
            or (use_fuel and (fuel_depleted_def or any(fuel_depleted_att)))
        )

        reason = None
        attacker_idx = -1
        if done:
            if any(collision):
                attacker_idx = int(np.argmax(np.asarray(collision, dtype=int)))
                reason = "collision"
            elif any(att_hit_target):
                attacker_idx = int(np.argmax(np.asarray(att_hit_target, dtype=int)))
                reason = "attacker_hit_target"
            elif def_hit_target:
                reason = "defender_hit_target"
            elif oob_def:
                reason = "defender_oob"
            elif any(oob_att):
                attacker_idx = int(np.argmax(np.asarray(oob_att, dtype=int)))
                reason = "attacker_oob"
            elif use_fuel and fuel_depleted_def:
                reason = "defender_fuel_depleted"
            elif use_fuel and any(fuel_depleted_att):
                attacker_idx = int(np.argmax(np.asarray(fuel_depleted_att, dtype=int)))
                reason = "attacker_fuel_depleted"

        return done, {
            "oob_def": bool(oob_def),
            "oob_att": [bool(v) for v in oob_att],
            "oob_att_any": bool(any(oob_att)),
            "def_hit_target": bool(def_hit_target),
            "att_hit_target": [bool(v) for v in att_hit_target],
            "att_hit_target_any": bool(any(att_hit_target)),
            "collision": [bool(v) for v in collision],
            "collision_any": bool(any(collision)),
            "fuel_depleted_def": bool(fuel_depleted_def),
            "fuel_depleted_att": [bool(v) for v in fuel_depleted_att],
            "fuel_depleted_att_any": bool(any(fuel_depleted_att)),
            "done_reason": reason,
            "done_attacker_idx": int(attacker_idx),
        }

    plan_hist_all = [[] for _ in range(1 + Na)]
    plan_att_all = [[] for _ in range(1 + Na)]
    exec_xyz_all = [[] for _ in range(1 + Na)]
    exec_att_all = [[] for _ in range(1 + Na)]
    phi_hist_all = [[] for _ in range(1 + Na)]
    fov_axis_hist, fov_seen_mask = [], []

    u_cmd_all = [[] for _ in range(1 + Na)]
    u_cmd_norm_all = [[] for _ in range(1 + Na)]
    u_real_all = [[] for _ in range(1 + Na)]
    u_real_norm_all = [[] for _ in range(1 + Na)]

    fuel_frac_all = [[] for _ in range(1 + Na)]
    thrust_all = [[] for _ in range(1 + Na)]
    mdot_all = [[] for _ in range(1 + Na)]
    done_info = None

    I = _identity_R()
    all_states = [xD] + xA_list
    for idx, xs in enumerate(all_states):
        exec_xyz_all[idx].append(_p3(xs, D))
        exec_att_all[idx].append({"R": I, "phi": 0.0})
        phi_hist_all[idx].append(0.0)
    fov_axis_hist.append(None)
    fov_seen_mask.append(False)

    if use_fuel:
        fuel_frac_all[0].append(float(np.clip((m_def - mdry_def) / (m0_def - mdry_def + 1e-9), 0.0, 1.0)))
        for j in range(Na):
            fuel_frac_all[1 + j].append(float(np.clip((m_att[j] - mdry_att) / (m0_att - mdry_att + 1e-9), 0.0, 1.0)))

    t_roll0 = time.perf_counter()
    for k in range(steps):
        obs_xA = xA_est_list if (use_kf and ukf_list) else xA_list
        obs = build_train_obs(xD, obs_xA, m_def, m_att)
        uD_cmd = np.clip(
            np.asarray(pol.act_def_obs(obs, deterministic=deterministic), dtype=np.float32),
            -umax,
            +umax,
        )
        uA_cmd = [
            np.clip(
                np.asarray(pol.act_att_obs(obs, deterministic=deterministic, attacker_idx=j), dtype=np.float32),
                -umax,
                +umax,
            )
            for j in range(Na)
        ]

        cmd_actions = [uD_cmd] + uA_cmd
        for idx, u_cmd in enumerate(cmd_actions):
            u3 = _u3_from_action(u_cmd, D)
            u_cmd_all[idx].append(u3.copy())
            u_cmd_norm_all[idx].append(float(np.linalg.norm(u3)))

        if debug_actions and k < 3:
            att_norms = [float(np.linalg.norm(u)) for u in uA_cmd]
            print(
                f"[k={k:02d}] ||a_def_cmd||={np.linalg.norm(uD_cmd):.3g}, "
                f"att_cmd_norms={att_norms}, obs[:6]={obs[:6]}"
            )

        if use_fuel:
            uD_real, m_def, fuel_depleted_def, thrust_def, mdot_def = apply_propulsion(
                uD_cmd, m_def, mdry_def, Tmax_def, Isp_def
            )
            uA_real = []
            fuel_depleted_att = []
            thrust_att = []
            mdot_att = []
            for j in range(Na):
                u_real_j, m_att[j], fuel_j, thrust_j, mdot_j = apply_propulsion(
                    uA_cmd[j], m_att[j], mdry_att, Tmax_att, Isp_att
                )
                uA_real.append(u_real_j)
                fuel_depleted_att.append(bool(fuel_j))
                thrust_att.append(float(thrust_j))
                mdot_att.append(float(mdot_j))
        else:
            uD_real = uD_cmd
            uA_real = list(uA_cmd)
            fuel_depleted_def = False
            fuel_depleted_att = [False for _ in range(Na)]
            thrust_def = 0.0
            mdot_def = 0.0
            thrust_att = [0.0 for _ in range(Na)]
            mdot_att = [0.0 for _ in range(Na)]

        real_actions = [uD_real] + uA_real
        for idx, u_real in enumerate(real_actions):
            u3 = _u3_from_action(u_real, D)
            u_real_all[idx].append(u3.copy())
            u_real_norm_all[idx].append(float(np.linalg.norm(u3)))

        thrust_all[0].append(float(thrust_def))
        mdot_all[0].append(float(mdot_def))
        for j in range(Na):
            thrust_all[1 + j].append(float(thrust_att[j]))
            mdot_all[1 + j].append(float(mdot_att[j]))

        xD = step_plant_single(xD, uD_real, k)
        for j in range(Na):
            xA_list[j] = step_plant_single(xA_list[j], uA_real[j], k)

        if use_kf and ukf_list:
            p_obs = xD[:D]
            R_wb = np.eye(3)
            for j in range(Na):
                ukf_j = ukf_list[j]
                u_pred, u_cov = _kf_predict_control(
                    ukf_action_access,
                    _u3_from_action(uA_real[j], D),
                    control_meas_std=ukf_control_meas_std,
                    control_limit=umax,
                )
                ukf_j.predict(dt=dt, u=u_pred, u_cov=u_cov)
                if ((k + 1) % meas_every) == 0:
                    pA_true = xA_list[j][:D]
                    v_b = _body_bearing_from_world(p_obs, R_wb, pA_true)
                    az_true, el_true = _azel_from_body_vec(v_b)
                    z_true = np.array([az_true, el_true], float)
                    z_noise = np.random.multivariate_normal(mean=np.zeros(2), cov=ukf_j.R)
                    ukf_j.update(z_true + z_noise, p_obs, R_wb)
                xA_est_list[j] = _xD_from_x6(np.r_[ukf_j.x[:3], ukf_j.x[3:6]]).astype(np.float32)

        if use_fuel:
            fuel_frac_all[0].append(float(np.clip((m_def - mdry_def) / (m0_def - mdry_def + 1e-9), 0.0, 1.0)))
            for j in range(Na):
                fuel_frac_all[1 + j].append(float(np.clip((m_att[j] - mdry_att) / (m0_att - mdry_att + 1e-9), 0.0, 1.0)))

        cur_states = [xD] + xA_list
        for idx, xs in enumerate(cur_states):
            plan_hist_all[idx].append([_p3(xs, D)] * T)
            plan_att_all[idx].append([{"R": I, "phi": 0.0} for _ in range(T)])
            exec_xyz_all[idx].append(_p3(xs, D))
            exec_att_all[idx].append({"R": I, "phi": 0.0})
            phi_hist_all[idx].append(0.0)

        fov_axis_hist.append(None)
        fov_seen_mask.append(False)

        done, done_info = check_done(xD, xA_list, fuel_depleted_def, fuel_depleted_att)
        if done and stop_on_done:
            break

    t_fn1 = time.perf_counter()
    timing = {
        "setup": float(t_roll0 - t_fn0),
        "simulation": float(t_fn1 - t_roll0),
        "total": float(t_fn1 - t_fn0),
    }

    return _pack_multi_agent_rollout(
        plan_hist_all=plan_hist_all,
        plan_att_all=plan_att_all,
        exec_xyz_all=exec_xyz_all,
        exec_att_all=exec_att_all,
        phi_hist_all=phi_hist_all,
        fov_axis_hist=fov_axis_hist,
        fov_seen_mask=fov_seen_mask,
        u_cmd_all=u_cmd_all,
        u_cmd_norm_all=u_cmd_norm_all,
        u_real_all=u_real_all,
        u_real_norm_all=u_real_norm_all,
        done_info=done_info,
        rollout_timing_sec=timing,
        fuel_frac_all=(fuel_frac_all if use_fuel else None),
        thrust_all=(thrust_all if use_fuel else None),
        mdot_all=(mdot_all if use_fuel else None),
    )


def run_rhc_with_rl_and_collect_frames_3d(
    cfg: Dict[str, Any],
    steps: int | None = None,
    turn_len: int | None = None,
):
    t_fn0 = time.perf_counter()
    # -------------------- basics & dims --------------------
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx = 2 * D
    T = int(cfg["T"])
    dt = float(cfg["dt"])

    num_attackers = int(cfg.get("num_attackers", 1))
    if num_attackers > 1:
        return _run_rhc_with_rl_and_collect_frames_3d_multi(cfg, steps=steps, turn_len=turn_len)

    dyn_name = str(cfg.get("dynamics", "hcw")).lower()

    # -------------------- sim length --------------------
    if steps is None:
        steps = int(cfg.get("T_eval", cfg.get("T", cfg.get("steps", 60))))
    if turn_len is None:
        turn_len = 1  # kept for API symmetry

    # -------------------- arena & center --------------------
    ar = cfg.setdefault("arena", {})
    ar.setdefault("type", "sphere")
    ar.setdefault("cx", 0.0)
    ar.setdefault("cy", 0.0)
    ar.setdefault("cz", 0.0)
    ar.setdefault("r", 30.0)

    if ar["type"] != "sphere":
        raise ValueError("Only spherical arena is supported in this rollout helper.")

    center = np.array(
        [ar["cx"], ar["cy"], (ar.get("cz", 0.0) if D == 3 else 0.0)],
        dtype=np.float32,
    )[:D]
    arena_r = float(ar["r"])

    # -------------------- initial states --------------------
    x0 = np.asarray(cfg["x0"], dtype=np.float32)
    if x0.shape[0] < 2:
        raise ValueError("cfg['x0'] must contain at least defender and attacker rows.")

    x1 = x0[0, :nx].copy()
    x2 = x0[1, :nx].copy()

    # -------------------- rollout options --------------------
    deterministic = bool(cfg.get("rl_eval_deterministic", True))
    umax = float(cfg.get("umax", 5e-4))
    debug_actions = bool(cfg.get("debug_actions", False))
    stop_on_done = bool(cfg.get("stop_on_done", True))

    # -------------------- fuel setup --------------------
    use_fuel = bool(cfg.get("fuel", {}).get("enable"))
    g0 = 9.80665

    if use_fuel:
        fuel_cfg = cfg["fuel"]
        fdef = fuel_cfg["def"]
        fatt = fuel_cfg["att"]

        m0_def = float(fdef["m0"])
        mdry_def = float(fdef["m_dry"])
        Tmax_def = float(fdef["Tmax"])
        Isp_def = float(fdef["Isp"])

        m0_att = float(fatt["m0"])
        mdry_att = float(fatt["m_dry"])
        Tmax_att = float(fatt["Tmax"])
        Isp_att = float(fatt["Isp"])

        m_def = m0_def
        m_att = m0_att
    else:
        m0_def = mdry_def = Tmax_def = Isp_def = None
        m0_att = mdry_att = Tmax_att = Isp_att = None
        m_def = None
        m_att = None

    def apply_propulsion(
        a_cmd: np.ndarray,
        m: float,
        m_dry: float,
        Tmax: float,
        Isp: float,
    ):
        a_cmd = np.asarray(a_cmd, dtype=float)

        if m <= m_dry + 1e-9:
            return np.zeros_like(a_cmd, dtype=np.float32), float(m_dry), True, 0.0, 0.0

        F_req = m * a_cmd
        F_req_norm = np.linalg.norm(F_req)

        if F_req_norm > Tmax:
            F = F_req * (Tmax / (F_req_norm + 1e-9))
        else:
            F = F_req

        a_real = F / max(m, 1e-9)

        thrust_norm = float(np.linalg.norm(F))
        mdot = thrust_norm / (Isp * g0)
        m_next = max(m_dry, m - mdot * dt)
        fuel_depleted = bool(m_next <= m_dry + 1e-9)

        return a_real.astype(np.float32), float(m_next), fuel_depleted, thrust_norm, float(mdot)

    # -------------------- obs normalization (optional) --------------------
    use_obsnorm = bool(cfg.get("obsnorm", False))
    obs_mean = None
    obs_std = None

    obs_expected = 5 * D + (2 if use_fuel else 0)

    if use_obsnorm and "obs_stats" in cfg:
        import os

        mp = cfg["obs_stats"].get("mean_path", None)
        sp = cfg["obs_stats"].get("std_path", None)

        if mp and os.path.exists(mp):
            obs_mean = np.load(mp).astype(np.float32)
        if sp and os.path.exists(sp):
            obs_std = np.load(sp).astype(np.float32)

        if obs_mean is not None and obs_mean.shape[0] != obs_expected:
            raise ValueError(
                f"obs_mean has dim {obs_mean.shape[0]}, expected {obs_expected}."
            )
        if obs_std is not None and obs_std.shape[0] != obs_expected:
            raise ValueError(
                f"obs_std has dim {obs_std.shape[0]}, expected {obs_expected}."
            )

    def build_train_obs(
        x1_vec: np.ndarray,
        x2_vec: np.ndarray,
        m_def_cur: float | None,
        m_att_cur: float | None,
    ) -> np.ndarray:
        p1 = x1_vec[:D]
        v1 = x1_vec[D:2 * D]
        p2 = x2_vec[:D]
        v2 = x2_vec[D:2 * D]

        parts = [
            p1 - center,
            p2 - center,
            (p2 - p1),
            v1,
            v2,
        ]

        if use_fuel:
            fuel_frac_def = (m_def_cur - mdry_def) / (m0_def - mdry_def + 1e-9)
            fuel_frac_att = (m_att_cur - mdry_att) / (m0_att - mdry_att + 1e-9)

            parts.append(np.array([np.clip(fuel_frac_def, 0.0, 1.0)], dtype=np.float32))
            parts.append(np.array([np.clip(fuel_frac_att, 0.0, 1.0)], dtype=np.float32))

        obs = np.concatenate(parts).astype(np.float32)

        if use_obsnorm and (obs_mean is not None) and (obs_std is not None):
            obs = (obs - obs_mean) / (obs_std + 1e-8)

        return obs

    def build_student_sigma_feat():
        if use_kf and (ukf12 is not None):
            P = np.asarray(ukf12.P, dtype=np.float32)
            P_rel = P[: 2 * D, : 2 * D]
            iu = np.triu_indices(2 * D)
            return P_rel[iu].astype(np.float32)
        return None

    # -------------------- policy wrapper --------------------
    pol = RLPolicyDiff(cfg, device=cfg.get("device", "cpu"))
    din_def, din_att = pol.verify_ckpt_compat()

    if din_def != obs_expected or din_att != obs_expected:
        raise RuntimeError(
            f"Policy obs dims mismatch: defender={din_def}, attacker={din_att}, "
            f"expected={obs_expected} (D={D}, use_fuel={use_fuel})."
        )

    # -------------------- tiny helpers for animator compatibility --------------------
    def _p3_row(x_row):
        return (
            (float(x_row[0]), float(x_row[1]), float(x_row[2]))
            if D == 3
            else (float(x_row[0]), float(x_row[1]), 0.0)
        )

    def _p3_vec(x_vec):
        return (
            (float(x_vec[0]), float(x_vec[1]), float(x_vec[2]))
            if D == 3
            else (float(x_vec[0]), float(x_vec[1]), 0.0)
        )

    def _identity_R():
        return np.eye(3, dtype=float)

    def _u3(uD: np.ndarray):
        uD = np.asarray(uD, float).reshape(-1)
        if D == 3:
            return uD[:3]
        return np.array([uD[0], uD[1], 0.0], dtype=float)

    def _x6_from_xD(xD: np.ndarray) -> np.ndarray:
        xD = np.asarray(xD, dtype=np.float32).reshape(-1)
        if D == 3:
            return xD.astype(np.float32)
        return np.array([xD[0], xD[1], 0.0, xD[2], xD[3], 0.0], dtype=np.float32)

    def _xD_from_x6(x6: np.ndarray) -> np.ndarray:
        x6 = np.asarray(x6, dtype=np.float32).reshape(-1)
        if D == 3:
            return x6.astype(np.float32)
        return np.array([x6[0], x6[1], x6[3], x6[4]], dtype=np.float32)

    # -------------------- optional UKFs (HCW-only) --------------------
    use_kf = bool(cfg.get("use_kf", False))
    estimator_kind = _normalize_estimator_kind(cfg)
    if use_kf and dyn_name != "hcw":
        print(f"[warn] {estimator_kind.upper()} in this runner is HCW-only; disabling estimator for non-HCW dynamics.")
        use_kf = False

    est_hist = None
    x2_est = x2.copy()
    x1_est = x1.copy()

    if use_kf:
        ukf_cfg = dict(cfg.get("ukf", {}))

        sigma_az = float(ukf_cfg.get("sigma_az", np.deg2rad(0.5)))
        sigma_el = float(ukf_cfg.get("sigma_el", np.deg2rad(0.5)))
        ukf_action_access = _normalize_kf_action_access(ukf_cfg.get("action_access", "ground_truth"))
        ukf_control_meas_std = _normalize_kf_control_noise_std(
            ukf_cfg.get("action_meas_std", 0.1 * max(abs(umax), 1e-3)),
            default_std=0.1 * max(abs(umax), 1e-3),
        )
        ukf_state_dim = 6
        pos_std0 = float(ukf_cfg.get("pos_std0", 0.2 * arena_r))
        vel_std0 = float(ukf_cfg.get("vel_std0", 0.01))
        init_mean_pos_std = float(ukf_cfg.get("init_mean_pos_std", 0.0))
        init_mean_vel_std = float(ukf_cfg.get("init_mean_vel_std", 0.0))
        init_mean_accel_std = float(ukf_cfg.get("init_mean_accel_std", 0.0))
        Q_scale = float(ukf_cfg.get("Q_scale", 1e-5))
        accel_std0 = float(ukf_cfg.get("accel_std0", max(abs(umax), 1e-3)))
        accel_q_scale = float(ukf_cfg.get("accel_Q_scale", Q_scale))

        who = str(ukf_cfg.get("who", "both")).lower()
        meas_every = max(1, int(ukf_cfg.get("every", 1)))

        if ukf_state_dim == 9:
            P0 = np.diag([pos_std0**2] * 3 + [vel_std0**2] * 3 + [accel_std0**2] * 3)
            Q = np.diag([Q_scale] * 6 + [accel_q_scale] * 3)
        else:
            P0 = np.diag([pos_std0**2] * 3 + [vel_std0**2] * 3)
            Q = Q_scale * np.diag([1, 1, 1, 1, 1, 1])
        Rm = np.diag([sigma_az**2, sigma_el**2])
        hcw_params = cfg.get("hcw", {})

        x2_6 = _x6_from_xD(x2)
        x1_6 = _x6_from_xD(x1)

        x2_6_mean = x2_6.copy()
        x1_6_mean = x1_6.copy()
        x2_6_mean[:3] += np.random.normal(0.0, init_mean_pos_std, size=3)
        x2_6_mean[3:6] += np.random.normal(0.0, init_mean_vel_std, size=3)
        x1_6_mean[:3] += np.random.normal(0.0, init_mean_pos_std, size=3)
        x1_6_mean[3:6] += np.random.normal(0.0, init_mean_vel_std, size=3)
        x2_ukf0 = x2_6_mean
        x1_ukf0 = x1_6_mean

        ukf12 = (
            KF_CV(x2_ukf0, P0, Q, Rm, dt, kind=estimator_kind, dyn="hcw", hcw=hcw_params)
            if who in ("both", "1->2")
            else None
        )
        ukf21 = (
            KF_CV(x1_ukf0, P0, Q, Rm, dt, kind=estimator_kind, dyn="hcw", hcw=hcw_params)
            if who in ("both", "2->1")
            else None
        )

        est_hist = {}
        if ukf12 is not None:
            est_hist["est12_xyz"] = [ukf12.x[:3].copy()]
            est_hist["meas12_azel"] = [None]
            x2_est = _xD_from_x6(np.r_[ukf12.x[:3], ukf12.x[3:6]]).astype(np.float32)

        if ukf21 is not None:
            est_hist["est21_xyz"] = [ukf21.x[:3].copy()]
            est_hist["meas21_azel"] = [None]
            x1_est = _xD_from_x6(np.r_[ukf21.x[:3], ukf21.x[3:6]]).astype(np.float32)

        meas_std_az, meas_std_el = sigma_az, sigma_el
    else:
        ukf12 = ukf21 = None
        ukf_action_access = "ground_truth"
        meas_every = 1
        meas_std_az = meas_std_el = None

    # -------------------- dynamics selection (LTI/LTV/nonlinear) --------------------
    dyn = cfg.get("dyn", {}) if isinstance(cfg.get("dyn", {}), dict) else {}

    dyn_type = dyn.get("type", None)
    Ad = dyn.get("Ad", None)
    Bd = dyn.get("Bd", None)
    Ad_seq = dyn.get("Ad_seq", None)
    Bd_seq = dyn.get("Bd_seq", None)
    chief_cache = dyn.get("chief_cache", None)

    if dyn_name == "hcw":
        if Ad is None or Bd is None:
            from dyn_models import hcw_mean_motion, hcw_discrete_mats, as_numpy_const

            n = hcw_mean_motion(cfg.get("hcw", {}))
            Ad_mx, Bd_mx = hcw_discrete_mats(n, dt)
            Ad = as_numpy_const(Ad_mx).astype(np.float32)
            Bd = as_numpy_const(Bd_mx).astype(np.float32)
        dyn_type = "lti"

    elif dyn_name in ("elliptic_ltv", "elliptical_ltv", "th", "tschauner_hempel"):
        if Ad_seq is None or Bd_seq is None or chief_cache is None:
            from dyn_models import chief_orbit_cache_rtn, linearize_two_body_rtn_discrete

            orb = cfg.get("chief_orbit", {})
            chief_cache = chief_orbit_cache_rtn(orb, dt=dt, N=steps)
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

    def step_plant_single(xD: np.ndarray, uD: np.ndarray, k: int) -> np.ndarray:
        x6 = _x6_from_xD(xD)
        u3 = _u3(uD)

        if dyn_type == "lti":
            x6n = (Ad @ x6 + Bd @ u3).astype(np.float32)
        elif dyn_type == "ltv":
            Ak = Ad_seq[k]
            Bk = Bd_seq[k]
            x6n = (Ak @ x6 + Bk @ u3).astype(np.float32)
        elif dyn_type == "nonlinear":
            from dyn_models import two_body_step_rtn

            x6n = two_body_step_rtn(x6, u3, k, chief_cache).astype(np.float32)
        else:
            raise RuntimeError(f"dyn_type='{dyn_type}' not recognized")

        return _xD_from_x6(x6n)

    # -------------------- termination helper --------------------
    oi = cfg.get("oi", {}) or {}
    oi_radius = float(oi.get("r", 0.0))
    oi_radius_norm = oi_radius / arena_r if arena_r > 0 else 0.0
    # hit_buffer_def = float(cfg.get("hit_buffer_def"))
    # hit_buffer_att = float(cfg.get("hit_buffer_att"))
    hit_buffer_def = 0.0
    hit_buffer_att = 0.0
    collision_radius_m = float(cfg.get("collision_radius_m"))
    arena_margin = float(cfg.get("arena_terminate_margin"))

    def check_done(
        x1_vec: np.ndarray,
        x2_vec: np.ndarray,
        fuel_depleted_def: bool,
        fuel_depleted_att: bool,
    ):
        p1 = x1_vec[:D]
        p2 = x2_vec[:D]

        rho1 = np.linalg.norm(p1 - center) / max(arena_r, 1e-9)
        rho2 = np.linalg.norm(p2 - center) / max(arena_r, 1e-9)

        oob_def = rho1 >= arena_margin
        oob_att = rho2 >= arena_margin

        att_hit_target = False
        def_hit_target = False
        if oi_radius_norm > 0.0:
            def_hit_target = rho1 <= (1.0 + hit_buffer_def) * oi_radius_norm
            att_hit_target = rho2 <= (1.0 + hit_buffer_att) * oi_radius_norm

        collision = False
        if collision_radius_m > 0.0:
            collision = np.linalg.norm(p2 - p1) <= collision_radius_m

        done = (
            oob_def
            or oob_att
            or def_hit_target
            or att_hit_target
            or collision
            or (use_fuel and (fuel_depleted_def or fuel_depleted_att))
        )

        reason = None
        if done:
            if collision:
                reason = "collision"
            elif att_hit_target:
                reason = "attacker_hit_target"
            elif def_hit_target:
                reason = "defender_hit_target"
            elif oob_def:
                reason = "defender_oob"
            elif oob_att:
                reason = "attacker_oob"
            elif use_fuel and fuel_depleted_def:
                reason = "defender_fuel_depleted"
            elif use_fuel and fuel_depleted_att:
                reason = "attacker_fuel_depleted"

        return done, {
            "oob_def": bool(oob_def),
            "oob_att": bool(oob_att),
            "def_hit_target": bool(def_hit_target),
            "att_hit_target": bool(att_hit_target),
            "collision": bool(collision),
            "fuel_depleted_def": bool(fuel_depleted_def),
            "fuel_depleted_att": bool(fuel_depleted_att),
            "done_reason": reason,
        }

    # -------------------- logs for animator --------------------
    plan_hist1, plan_hist2 = [], []
    plan_att1, plan_att2 = [], []
    exec_xyz1, exec_xyz2 = [], []
    exec_att1, exec_att2 = [], []
    phi_hist1, phi_hist2 = [], []
    fov_axis_hist, fov_seen_mask = [], []

    u_cmd_1, u_cmd_2 = [], []
    u_cmd_norm_1, u_cmd_norm_2 = [], []

    u_real_1, u_real_2 = [], []
    u_real_norm_1, u_real_norm_2 = [], []

    fuel_frac_def_hist, fuel_frac_att_hist = [], []
    thrust_def_hist, thrust_att_hist = [], []
    mdot_def_hist, mdot_att_hist = [], []
    done_info = None

    # t=0 logs
    exec_xyz1.append(_p3_vec(x1))
    exec_xyz2.append(_p3_vec(x2))
    exec_att1.append({"R": _identity_R(), "phi": 0.0})
    exec_att2.append({"R": _identity_R(), "phi": 0.0})
    phi_hist1.append(0.0)
    phi_hist2.append(0.0)
    fov_axis_hist.append(None)
    fov_seen_mask.append(False)

    if use_fuel:
        fuel_frac_def_hist.append(float(np.clip((m_def - mdry_def) / (m0_def - mdry_def + 1e-9), 0.0, 1.0)))
        fuel_frac_att_hist.append(float(np.clip((m_att - mdry_att) / (m0_att - mdry_att + 1e-9), 0.0, 1.0)))

    # ==================== ROLLOUT ====================
    t_roll0 = time.perf_counter()
    for k in range(steps):
        # 1) build observations and query policies ONCE
        x2_for_def_obs = x2_est if (use_kf and ukf12 is not None) else x2
        x1_for_att_obs = x1_est if (use_kf and ukf21 is not None) else x1

        obs_def = build_train_obs(x1, x2_for_def_obs, m_def, m_att)
        obs_att = build_train_obs(x1_for_att_obs, x2, m_def, m_att)
        sigma_feat = build_student_sigma_feat()

        u1_cmd = np.clip(
            np.asarray(
                pol.act_def_obs(obs_def, deterministic=deterministic, sigma_feat=sigma_feat),
                dtype=np.float32,
            ),
            -umax,
            +umax,
        )
        u2_cmd = np.clip(
            np.asarray(
                pol.act_att_obs(obs_att, deterministic=deterministic, sigma_feat=sigma_feat),
                dtype=np.float32,
            ),
            -umax,
            +umax,
        )

        # 2) log commanded actions
        u1_cmd_3 = _u3(u1_cmd)
        u2_cmd_3 = _u3(u2_cmd)
        u_cmd_1.append(u1_cmd_3.copy())
        u_cmd_2.append(u2_cmd_3.copy())
        u_cmd_norm_1.append(float(np.linalg.norm(u1_cmd_3)))
        u_cmd_norm_2.append(float(np.linalg.norm(u2_cmd_3)))

        if debug_actions and k < 3:
            print(
                f"[k={k:02d}] ||a_def_cmd||={np.linalg.norm(u1_cmd):.3g}, "
                f"||a_att_cmd||={np.linalg.norm(u2_cmd):.3g}, "
                f"obs_def[:6]={obs_def[:6]}, obs_att[:6]={obs_att[:6]}"
            )

        # 3) propulsion / realized acceleration
        if use_fuel:
            u1_real, m_def, fuel_depleted_def, thrust_def, mdot_def = apply_propulsion(
                u1_cmd, m_def, mdry_def, Tmax_def, Isp_def
            )
            u2_real, m_att, fuel_depleted_att, thrust_att, mdot_att = apply_propulsion(
                u2_cmd, m_att, mdry_att, Tmax_att, Isp_att
            )
        else:
            u1_real = u1_cmd
            u2_real = u2_cmd
            fuel_depleted_def = False
            fuel_depleted_att = False
            thrust_def = thrust_att = 0.0
            mdot_def = mdot_att = 0.0

        u1_real_3 = _u3(u1_real)
        u2_real_3 = _u3(u2_real)
        u_real_1.append(u1_real_3.copy())
        u_real_2.append(u2_real_3.copy())
        u_real_norm_1.append(float(np.linalg.norm(u1_real_3)))
        u_real_norm_2.append(float(np.linalg.norm(u2_real_3)))

        thrust_def_hist.append(float(thrust_def))
        thrust_att_hist.append(float(thrust_att))
        mdot_def_hist.append(float(mdot_def))
        mdot_att_hist.append(float(mdot_att))

        # 4) plant step
        x1 = step_plant_single(x1, u1_real, k)
        x2 = step_plant_single(x2, u2_real, k)

        # 5) UKF estimation (HCW-only)
        if use_kf and est_hist is not None:
            idx = len(exec_xyz1)
            take = (idx % meas_every) == 0

            def _to3_pos(xD):
                p = np.zeros(3, dtype=float)
                p[:D] = np.asarray(xD[:D], float)
                return p

            if ukf12 is not None:
                u12_pred, u12_cov = _kf_predict_control(
                    ukf_action_access,
                    _u3(u2_real),
                    control_meas_std=ukf_control_meas_std,
                    control_limit=umax,
                )
                ukf12.predict(dt, u=u12_pred, u_cov=u12_cov)
                if take:
                    p_obs = _to3_pos(x1)
                    p_tgt = _to3_pos(x2)
                    d = p_tgt - p_obs
                    d /= (np.linalg.norm(d) + 1e-12)
                    az = np.arctan2(d[1], d[0]) + np.random.randn() * meas_std_az
                    el = np.arctan2(d[2], np.sqrt(max(d[0] ** 2 + d[1] ** 2, 1e-18))) + np.random.randn() * meas_std_el
                    z = np.array([az, el], float)
                    ukf12.update(z, p_obs=p_obs, R_wb=_identity_R())
                    est_hist["meas12_azel"].append((p_obs.copy(), z.copy()))
                else:
                    est_hist["meas12_azel"].append(None)
                est_hist["est12_xyz"].append(ukf12.x[:3].copy())

            if ukf21 is not None:
                u21_pred, u21_cov = _kf_predict_control(
                    ukf_action_access,
                    _u3(u1_real),
                    control_meas_std=ukf_control_meas_std,
                    control_limit=umax,
                )
                ukf21.predict(dt, u=u21_pred, u_cov=u21_cov)
                if take:
                    p_obs = _to3_pos(x2)
                    p_tgt = _to3_pos(x1)
                    d = p_tgt - p_obs
                    d /= (np.linalg.norm(d) + 1e-12)
                    az = np.arctan2(d[1], d[0]) + np.random.randn() * meas_std_az
                    el = np.arctan2(d[2], np.sqrt(max(d[0] ** 2 + d[1] ** 2, 1e-18))) + np.random.randn() * meas_std_el
                    z = np.array([az, el], float)
                    ukf21.update(z, p_obs=p_obs, R_wb=_identity_R())
                    est_hist["meas21_azel"].append((p_obs.copy(), z.copy()))
                else:
                    est_hist["meas21_azel"].append(None)
                est_hist["est21_xyz"].append(ukf21.x[:3].copy())

            if ukf12 is not None:
                x2_est = _xD_from_x6(np.r_[ukf12.x[:3], ukf12.x[3:6]]).astype(np.float32)
            if ukf21 is not None:
                x1_est = _xD_from_x6(np.r_[ukf21.x[:3], ukf21.x[3:6]]).astype(np.float32)

        # 6) fuel traces
        if use_fuel:
            fuel_frac_def_hist.append(
                float(np.clip((m_def - mdry_def) / (m0_def - mdry_def + 1e-9), 0.0, 1.0))
            )
            fuel_frac_att_hist.append(
                float(np.clip((m_att - mdry_att) / (m0_att - mdry_att + 1e-9), 0.0, 1.0))
            )

        # 7) logs for animation
        plan1 = [_p3_row(x1)] * T
        plan2 = [_p3_row(x2)] * T
        plan_hist1.append(plan1)
        plan_hist2.append(plan2)

        I = _identity_R()
        att_stub = [{"R": I, "phi": 0.0} for _ in range(T)]
        plan_att1.append(att_stub)
        plan_att2.append(att_stub)

        exec_att1.append({"R": I, "phi": 0.0})
        exec_att2.append({"R": I, "phi": 0.0})
        phi_hist1.append(0.0)
        phi_hist2.append(0.0)

        exec_xyz1.append(_p3_vec(x1))
        exec_xyz2.append(_p3_vec(x2))
        fov_axis_hist.append(None)
        fov_seen_mask.append(False)

        # 8) optional termination
        done, done_info = check_done(x1, x2, fuel_depleted_def, fuel_depleted_att)
        if done and stop_on_done:
            break

    t_fn1 = time.perf_counter()
    timing = {
        "setup": float(t_roll0 - t_fn0),
        "simulation": float(t_fn1 - t_roll0),
        "total": float(t_fn1 - t_fn0),
    }

    # -------------------- pack results --------------------
    out = {
        "plan_hist1": plan_hist1,
        "plan_hist2": plan_hist2,
        "plan_att1": plan_att1,
        "plan_att2": plan_att2,
        "exec1_xyz": exec_xyz1,
        "exec2_xyz": exec_xyz2,
        "exec_att1": exec_att1,
        "exec_att2": exec_att2,
        "phi_hist1": phi_hist1,
        "phi_hist2": phi_hist2,
        "fov_axis_hist": fov_axis_hist,
        "fov_seen_mask": fov_seen_mask,

        "u_cmd_all": [
            np.asarray(u_cmd_1, dtype=float),
            np.asarray(u_cmd_2, dtype=float),
        ],
        "u_cmd_norm_all": [
            np.asarray(u_cmd_norm_1, dtype=float),
            np.asarray(u_cmd_norm_2, dtype=float),
        ],

        "u_real_all": [
            np.asarray(u_real_1, dtype=float),
            np.asarray(u_real_2, dtype=float),
        ],
        "u_real_norm_all": [
            np.asarray(u_real_norm_1, dtype=float),
            np.asarray(u_real_norm_2, dtype=float),
        ],

        "done_info": done_info,
        "rollout_timing_sec": timing,
    }

    if use_fuel:
        out.update(
            {
                "fuel_frac_all": [
                    np.asarray(fuel_frac_def_hist, dtype=float),
                    np.asarray(fuel_frac_att_hist, dtype=float),
                ],
                "thrust_all": [
                    np.asarray(thrust_def_hist, dtype=float),
                    np.asarray(thrust_att_hist, dtype=float),
                ],
                "mdot_all": [
                    np.asarray(mdot_def_hist, dtype=float),
                    np.asarray(mdot_att_hist, dtype=float),
                ],
            }
        )

    if est_hist is not None:
        out.update(est_hist)

    return out
