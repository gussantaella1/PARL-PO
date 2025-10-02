# game_3dutils.py
# Dedicated 3D utilities and entrypoints. Depends on game_sharedutils.
from __future__ import annotations
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D


import importlib, game_3dutils, game_costs, neos_path_game, ukf_estimator, ekf_estimator
importlib.reload(game_3dutils)
importlib.reload(game_costs)
importlib.reload(neos_path_game)

from ukf_estimator import AgentUKF 
from ekf_estimator import AgentEKF 



importlib.reload(ukf_estimator)
importlib.reload(ekf_estimator)

__all__ = [
     "run_rhc_and_collect_frames_3d",
    "animate_rollout_3d", "interactive_rollout_3d"
]

try:
    from neos_path_game import (
        build_mcp_two_player_one_shot,
        solve_with_local_path,
        extract_trajectories,
    )


    HAS_PATH = True
except Exception:
    HAS_PATH = False



# ---- PATH/MCP: path-inequality builders (h(k) >= 0) -------------------------
def build_h_builders(cfg, nx, D):
    """
    Return a list of callables h(m,k) >= 0 built from whatever keys
    actually exist in cfg (arena, sep_min/min_sep, vmax/speed_max).
    """
    ar = (cfg.get("arena") or {})
    funcs = []

    def _x(m, agent, k, j):
        return m.x1[k, j] if agent == 1 else m.x2[k, j]

    # -------- arena --------
    # Box if explicit bounds present
    if {"xmin","xmax","ymin","ymax"} <= set(ar.keys()):
        xmin, xmax = float(ar["xmin"]), float(ar["xmax"])
        ymin, ymax = float(ar["ymin"]), float(ar["ymax"])
        have_z     = (D == 3) and ("zmin" in ar and "zmax" in ar)
        zmin, zmax = (float(ar["zmin"]), float(ar["zmax"])) if have_z else (None, None)

        for agent in (1, 2):
            funcs.append(lambda m,k,_a=agent,_b=xmin: _x(m,_a,k,0) - _b)  # x >= xmin
            funcs.append(lambda m,k,_a=agent,_b=xmax: _b - _x(m,_a,k,0))  # xmax - x
            if D >= 2:
                funcs.append(lambda m,k,_a=agent,_b=ymin: _x(m,_a,k,1) - _b)
                funcs.append(lambda m,k,_a=agent,_b=ymax: _b - _x(m,_a,k,1))
            if have_z:
                funcs.append(lambda m,k,_a=agent,_b=zmin: _x(m,_a,k,2) - _b)
                funcs.append(lambda m,k,_a=agent,_b=zmax: _b - _x(m,_a,k,2))

    # Sphere if given (your CONFIG uses this)
    elif {"cx","cy","cz","r"} <= set(ar.keys()) or ar.get("type") == "sphere":
        cx = float(ar.get("cx", 0.0))
        cy = float(ar.get("cy", 0.0))
        cz = float(ar.get("cz", 0.0))
        R2 = float(ar.get("r", 1.0))**2

        def _sphere_h(agent):
            def h(m,k,_a=agent,_cx=cx,_cy=cy,_cz=cz,_R2=R2):
                px = _x(m,_a,k,0) - _cx
                py = _x(m,_a,k,1) - _cy if D >= 2 else 0.0
                pz = _x(m,_a,k,2) - _cz if D == 3 else 0.0
                return _R2 - (px*px + py*py + pz*pz)  # inside sphere
            return h
        funcs += [_sphere_h(1), _sphere_h(2)]

    # -------- min separation --------
    sep = cfg["sep_min"] if "sep_min" in cfg else (cfg["min_sep"] if "min_sep" in cfg else None)
    if sep is not None:
        d2 = float(sep)**2
        def h_sep(m,k,_d2=d2):
            s = 0
            for j in range(D):
                s += (_x(m,1,k,j) - _x(m,2,k,j))**2
            return s - _d2  # >= 0
        funcs.append(h_sep)

    # -------- speed cap --------
    vmax = cfg["vmax"] if "vmax" in cfg else (cfg["speed_max"] if "speed_max" in cfg else None)
    if vmax is not None:
        vmax2 = float(vmax)**2
        vel_idx = list(range(D, 2*D))
        def _speed_h(agent):
            def h(m,k,_a=agent,_v=vel_idx,_v2=vmax2):
                vsq = 0
                for j in _v:
                    vsq += (m.x1[k,j] if _a == 1 else m.x2[k,j])**2
                return _v2 - vsq  # >= 0
            return h
        funcs += [_speed_h(1), _speed_h(2)]

    return funcs


def _filter_kwargs(fn, kwargs):
    """Keep only kwargs that `fn` accepts."""
    import inspect
    sig = inspect.signature(fn)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


class KF_CV:
    """
    Common interface over AgentUKF / AgentEKF.

    - ctor: KF_CV(x0, P0, Q, R, dt, kind='auto'|'ukf'|'ekf', **ukf_sigma_params)
    - predict(dt=None, u=None, **kwargs)   # extra kwargs ignored if EKF
    - update(z, p_obs, R_wb, **kwargs)     # same signature on both; forwards
    """
    def __init__(self, x0, P0, Q, R, dt, kind='auto', **kwargs):
        kind = (kind or 'auto').lower()
        impl = None

        if kind == 'ekf' and AgentEKF is not None:
            impl = AgentEKF(x0, P0, Q, R, dt)
        elif kind in ('ukf', 'auto') and AgentUKF is not None:
            # pass through optional sigma-point params if provided
            sp = {k: kwargs[k] for k in ('alpha', 'beta', 'kappa') if k in kwargs}
            impl = AgentUKF(x0, P0, Q, R, dt, **sp)
        elif AgentUKF is not None:  # fallback preference: UKF if available
            impl = AgentUKF(x0, P0, Q, R, dt)
        elif AgentEKF is not None:
            impl = AgentEKF(x0, P0, Q, R, dt)
        else:
            raise RuntimeError("Neither AgentUKF nor AgentEKF is importable.")

        self._impl = impl  # keep the concrete filter

    # expose state/cov/etc. transparently
    def __getattr__(self, name):
        return getattr(self._impl, name)

    # accept superset of kwargs; drop those the impl doesn't support
    def predict(self, dt=None, u=None, **kwargs):
        f = getattr(self._impl, 'predict')
        return f(dt=dt, u=u, **_filter_kwargs(f, kwargs))

    def update(self, z, p_obs, R_wb, **kwargs):
        f = getattr(self._impl, 'update')
        return f(z, p_obs=p_obs, R_wb=R_wb, **_filter_kwargs(f, kwargs))



# -------------------- RHC with execution & FOV (3D/2D-aware) --------------------
def run_rhc_and_collect_frames_3d(cfg: dict, cost_builder, steps: int | None = None,
                                  turn_len: int | None = None):
    """
    RHC rollout with optional attitude-in-state (roll φ). Boresight at t=0 is v/‖v‖.
    PATH-only: uses a persistent MCP model with warm-starts each turn.
    """
    N = 2
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx_tr, nu_tr = dims_from_D(D)
    T, dt  = int(cfg["T"]), float(cfg["dt"])

    # === estimation config (optional) ===
    est_cfg = dict(cfg.get('est', {}))
    do_est  = bool(est_cfg.get('enabled', False))

    # --- translational Ad,Bd (MX) ---
    dyn = (cfg.get("dynamics") or "double").lower()
    if dyn == "hcw":
        n = hcw_mean_motion(cfg.get("hcw", {}))
        Ad_tr, Bd_tr = hcw_discrete_mats(n, dt)        # MX
    else:
        Ad_tr, Bd_tr = step_double_integrator_D(D=D, dt=dt)  # MX

    # --- optional attitude augmentation (roll in state) ---
    att_cfg = cfg.get("att", {})
    use_att = bool(att_cfg)
    if use_att:
        Ad_mx, Bd_mx, idx = augment_AB_for_att(Ad_tr, Bd_tr, dt, att_cfg)
        nx, nu = idx["nx"], idx["nu"]
        i_phi  = idx["i_phi"]
    else:
        Ad_mx, Bd_mx = Ad_tr, Bd_tr
        nx, nu = nx_tr, nu_tr
        i_phi  = None

    # --- rollout length / turn length ---
    sim_time = cfg.get("sim_time", cfg.get("max_time", cfg.get("duration", None)))
    if steps is None:
        steps = max(1, int(np.ceil(float(sim_time)/dt))) if sim_time is not None else int(cfg.get("steps", 60))
    if turn_len is None:
        turn_len = int(cfg.get("turn_len", 3)) if "turn_seconds" not in cfg else \
                   max(1, int(round(float(cfg["turn_seconds"]) / float(dt))))

    # --- constraints (fixed Ad,Bd) ---
    gtil_fun = build_g_tilde_linear(nx, nu, T, N, Ad_mx, Bd_mx)
    x_lb, x_ub, u_lb, u_ub = make_bounds(cfg)
    if use_att:
        x_lb, x_ub, u_lb, u_ub = augment_bounds_with_att(x_lb, x_ub, u_lb, u_ub, att_cfg)
    htil_fun = build_h_tilde(nx, nu, T, N, x_lb, x_ub, u_lb, u_ub, cfg)

    # --- initial states (pad with φ if needed) ---
    x1 = pad_x0_with_att(cfg["x0"][0], att_cfg, D)[:nx] if use_att else np.asarray(cfg["x0"][0], float)[:nx].copy()
    x2 = pad_x0_with_att(cfg["x0"][1], att_cfg, D)[:nx] if use_att else np.asarray(cfg["x0"][1], float)[:nx].copy()
    theta_curr = np.r_[x1, x2]

    # --- numeric stepper (MX→np) ---
    Ad_np, Bd_np = as_numpy_const(Ad_mx), as_numpy_const(Bd_mx)
    def step_plant(x, u):
        return Ad_np @ np.asarray(x, float) + Bd_np @ np.asarray(u, float)

    # --- logs ---
    plan_hist1, plan_hist2 = [], []
    plan_att1,  plan_att2  = [], []
    exec_xyz1,  exec_xyz2  = [], []
    exec_att1,  exec_att2  = [], []
    phi_hist1,  phi_hist2  = [], []
    fov_axis_hist, fov_seen_mask = [], []

    # --- FOV config ---
    fov_cfg     = cfg.get("fov", {"enabled": False})
    fov_enabled = bool(fov_cfg.get("enabled", False))
    fov_agent   = int(fov_cfg.get("agent", 2))

    # -------------------- helpers --------------------
    def _p3_row(x_row):
        return (float(x_row[0]), float(x_row[1]), float(x_row[2])) if D==3 else (float(x_row[0]), float(x_row[1]), 0.0)
    def _p3_vec(x_vec):
        return (float(x_vec[0]), float(x_vec[1]), float(x_vec[2])) if D==3 else (float(x_vec[0]), float(x_vec[1]), 0.0)

    def attitude_from_state(x, prev_R=None, prev_axisD=None, att_cfg=None, use_att=True, i_phi=None):
        if att_cfg is None: att_cfg = {}
        align    = att_cfg.get("align", "x")
        world_up = np.asarray(att_cfg.get("up", [0, 0, 1]), float)
        vmin     = float(att_cfg.get("min_speed_for_axis", 1e-3))

        DD = 3 if len(x) >= 6 else 2
        v  = np.asarray(x[DD:2*DD], float)
        n  = np.linalg.norm(v)
        if n > vmin:
            axisD = v / n
        elif prev_axisD is not None:
            axisD = np.asarray(prev_axisD, float)
        else:
            axisD = np.array([1, 0, 0]) if align == "x" else np.array([0, 0, 1])

        axis3 = axisD if DD == 3 else np.array([axisD[0], axisD[1], 0.0], float)
        R = frame_from_axis_continuous(axis3, R_prev=prev_R, align=align, world_up=world_up)

        phi = 0.0
        if use_att and i_phi is not None and i_phi < len(x):
            phi = float(x[i_phi])
            R   = apply_roll_about_axis(R, phi, align=align)

        return R, axisD, phi

    def plan_attitudes_from_X(X, prev_axisD, prev_R):
        att_list = []
        ax_prev  = prev_axisD
        R_prev   = prev_R
        for t in range(T):
            R, ax_prev, phi_t = attitude_from_state(X[t], prev_R=R_prev, prev_axisD=ax_prev)
            att_list.append({"R": R, "phi": phi_t})
            R_prev = R
        return att_list, ax_prev, R_prev

    # --- t=0 attitude seed ---
    align    = att_cfg.get("align", "x")
    world_up = att_cfg.get("up", [0,0,1])
    vmin0    = float(att_cfg.get("min_speed_for_axis", 1e-3))
    def _axis_from_vel_t0(x):
        v = np.asarray(x[D:2*D], float); n = float(np.linalg.norm(v))
        aD = (v/n) if n > vmin0 else (np.array([1,0,0], float) if align=="x" else np.array([0,0,1], float))[:D]
        return aD, (aD if D==3 else np.array([aD[0], aD[1], 0.0], float))
    prev_axis1D, axis1_0 = _axis_from_vel_t0(x1)
    prev_axis2D, axis2_0 = _axis_from_vel_t0(x2)
    phi1_0 = float(x1[i_phi]) if (use_att and i_phi is not None) else 0.0
    phi2_0 = float(x2[i_phi]) if (use_att and i_phi is not None) else 0.0
    R1_0   = apply_roll_about_axis(world_to_body_R(axis1_0, 3, align=align, up=world_up), phi1_0, align=align)
    R2_0   = apply_roll_about_axis(world_to_body_R(axis2_0, 3, align=align, up=world_up), phi2_0, align=align)
    prev_R1, prev_R2 = R1_0, R2_0

    # === optional UKFs ===
    if do_est:
        s_az, s_el = np.deg2rad(est_cfg.get('meas_std_deg', (0.3, 0.3)))
        P0 = np.diag(est_cfg.get('P0_diag', [25,25,25, 1,1,1]))
        Q  = np.diag(est_cfg.get('Q_diag',  [1e-4,1e-4,1e-4, 1e-3,1e-3,1e-3]))
        Rm = np.diag([s_az**2, s_el**2])
        who   = (est_cfg.get('who') or 'both').lower()
        every = int(est_cfg.get('every', 1))

        est_hist = {'est12_xyz': [], 'est21_xyz': [], 'meas12_azel': [], 'meas21_azel': []}
        ukf12 = KF_CV(np.r_[x2[:3], np.zeros(3)], P0, Q, Rm, dt) if who in ('both','1->2') else None
        ukf21 = KF_CV(np.r_[x1[:3], np.zeros(3)], P0, Q, Rm, dt) if who in ('both','2->1') else None

        if ukf12 is not None:
            est_hist['est12_xyz'].append(ukf12.x[:3].copy()); est_hist['meas12_azel'].append(None)
        if ukf21 is not None:
            est_hist['est21_xyz'].append(ukf21.x[:3].copy()); est_hist['meas21_azel'].append(None)

    # log true t=0
    exec_xyz1.append(_p3_vec(x1)); exec_xyz2.append(_p3_vec(x2))
    exec_att1.append({"R": R1_0, "phi": phi1_0}); phi_hist1.append(phi1_0)
    exec_att2.append({"R": R2_0, "phi": phi2_0}); phi_hist2.append(phi2_0)

    if fov_enabled:
        R_def0 = R2_0 if fov_agent == 2 else R1_0
        x_def  = x2    if fov_agent == 2 else x1
        x_tgt  = x1    if fov_agent == 2 else x2
        a_def3 = R_def0[0] if align == 'x' else R_def0[2]
        fov_axis_hist.append(a_def3)
        cam_cfg = cfg.get("camera", None)
        if fov_cfg.get("type","cone") == "pinhole" and (cam_cfg is not None):
            _, _, _, ok0 = project_point_pinhole(X_w=x_tgt[:3], x_def=x_def, cam_cfg=cam_cfg, R_wb=R_def0)
            fov_seen_mask.append(bool(ok0))
        else:
            seen0, _ = in_fov(x_tgt[:D], x_def[:D], a_def3, fov_cfg, D)
            fov_seen_mask.append(bool(seen0))
    else:
        fov_axis_hist.append(None); fov_seen_mask.append(False)

    # -------------------- PATH persistent MCP --------------------
    # Wide var boxes; actual limits go in h(x) >= 0
    x_var_box = (-1e6, 1e6)
    u_var_box = (-1e3, 1e3)
    h_list    = build_h_builders(cfg, nx, D)

    m_path = build_mcp_two_player_one_shot(
        Ad=as_numpy_const(Ad_mx), Bd=as_numpy_const(Bd_mx),
        T=T, nx=nx, nu=nu, D=D,
        x0_1=x1, x0_2=x2,
        x_var_box=x_var_box, u_var_box=u_var_box,
        h_builders=h_list,
        cost_kind=cfg.get("setting","chase_escape_tail"),
        cost_cfg=cfg,
    )

    path_ctx = {
        "m": m_path,
        "A": as_numpy_const(Ad_mx),
        "B": as_numpy_const(Bd_mx),
        "X1_guess": None, "U1_guess": None,
        "X2_guess": None, "U2_guess": None,
    }

    def _shift_controls_one_step(U_prev, target_len, fill=0.0):
        U_prev = np.asarray(U_prev, float)
        old_len, nu_ = U_prev.shape if U_prev.ndim == 2 else (0, nu)
        U_new = np.full((target_len, nu), float(fill))
        if old_len >= 2:
            take = min(target_len-1, old_len-1)
            if take > 0:
                U_new[:take, :] = U_prev[1:1+take, :]
        return U_new

    def _forward_sim_x(A, B, x0, U):
        A = np.asarray(A, float); B = np.asarray(B, float)
        x0 = np.asarray(x0, float)
        Tm1, _ = U.shape
        nx_ = A.shape[0]
        X = np.zeros((Tm1+1, nx_), float)
        X[0,:] = x0
        for k in range(Tm1):
            X[k+1,:] = A @ X[k,:] + B @ U[k,:]
        return X

    def _seed_model_from_guess(m, X1, U1, X2, U2):
        for k in list(m.Kx):
            ki = int(k)
            for i in list(m.S):
                ii = int(i)
                m.x1[k, i].value = float(X1[ki, ii])
                m.x2[k, i].value = float(X2[ki, ii])
        for k in list(m.Ku):
            kk = int(k)
            for j in list(m.U):
                jj = int(j)
                m.u1[k, j].value = float(U1[kk, jj])
                m.u2[k, j].value = float(U2[kk, jj])

    def replan_path(theta_vec, prev1D, prev2D, prevR1, prevR2):
        x0_1 = theta_vec[:nx]
        x0_2 = theta_vec[nx:2*nx]
        m = path_ctx["m"]; A = path_ctx["A"]; B = path_ctx["B"]

        # refresh IC params + seed x(0)
        for i in range(nx):
            m.x01[i] = float(x0_1[i]); m.x02[i] = float(x0_2[i])
            m.x1[0, i].value = float(x0_1[i]); m.x2[0, i].value = float(x0_2[i])

        # warm-start controls (|Ku| = T)
        Ku_len = len(list(m.Ku))
        if path_ctx["U1_guess"] is None:
            U1g = np.zeros((Ku_len, nu))
            U2g = np.zeros((Ku_len, nu))
        else:
            U1g = _shift_controls_one_step(path_ctx["U1_guess"], target_len=Ku_len)
            U2g = _shift_controls_one_step(path_ctx["U2_guess"], target_len=Ku_len)

        # forward sim to get X guesses (|Kx| = T+1)
        X1g = _forward_sim_x(A, B, x0_1, U1g)
        X2g = _forward_sim_x(A, B, x0_2, U2g)

        _seed_model_from_guess(m, X1g, U1g, X2g, U2g)

        # solve
        solve_with_local_path(
            m,
            path_exe="/Users/gussantaella/Documents/UTAustin/Research/Code/Research_Repo/path_5/ampl/pathampl",
            tee=True,
        )

        # extract + cache warm start for next turn
        X1, U1, X2, U2 = extract_trajectories(m)
        path_ctx["X1_guess"], path_ctx["U1_guess"] = X1, U1
        path_ctx["X2_guess"], path_ctx["U2_guess"] = X2, U2

        # pack outputs for viz/exec
        plan1 = [_p3_row(X1[t, :]) for t in range(T)]
        plan2 = [_p3_row(X2[t, :]) for t in range(T)]
        att1, prev1_out, prevR1_out = plan_attitudes_from_X(X1, prev1D, prevR1)
        att2, prev2_out, prevR2_out = plan_attitudes_from_X(X2, prev2D, prevR2)
        z_dummy = np.zeros(1)  # for API consistency
        return z_dummy, plan1, plan2, U1, U2, att1, att2, prev1_out, prev2_out, prevR1_out, prevR2_out

    # --- first plan ---
    z_last = np.zeros(1)  # not used, kept for signature symmetry
    z_last, plan1, plan2, U1, U2, att1, att2, prev_axis1D, prev_axis2D, prev_R1, prev_R2 = \
        replan_path(theta_curr, prev_axis1D, prev_axis2D, prev_R1, prev_R2)
    step_in_turn = 0

    # -------------------- rollout --------------------
    for k in range(steps):
        # replan each turn
        if k % turn_len == 0 and k > 0:
            z_last, plan1, plan2, U1, U2, att1, att2, prev_axis1D, prev_axis2D, prev_R1, prev_R2 = \
                replan_path(theta_curr, prev_axis1D, prev_axis2D, prev_R1, prev_R2)
            step_in_turn = 0

        # log current plan (for this step)
        plan_hist1.append(plan1); plan_hist2.append(plan2)
        plan_att1.append(att1);   plan_att2.append(att2)

        # controls for this step
        u1 = U1[min(step_in_turn, len(U1)-1)]
        u2 = U2[min(step_in_turn, len(U2)-1)]
        step_in_turn += 1

        # 1) plant step
        x1 = step_plant(x1, u1)
        x2 = step_plant(x2, u2)
        theta_curr = np.r_[x1, x2]

        # 2) attitude from updated states
        R1, axis1D, phi1_now = attitude_from_state(x1, prev_R=prev_R1, prev_axisD=prev_axis1D,
                                                   att_cfg=att_cfg, use_att=use_att, i_phi=i_phi)
        R2, axis2D, phi2_now = attitude_from_state(x2, prev_R=prev_R2, prev_axisD=prev_axis2D,
                                                   att_cfg=att_cfg, use_att=use_att, i_phi=i_phi)
        prev_axis1D, prev_axis2D = axis1D, axis2D
        prev_R1, prev_R2 = R1, R2

        phi_hist1.append(phi1_now); phi_hist2.append(phi2_now)

        # 3) log executed pose + attitude
        exec_att1.append({"R": R1, "phi": phi1_now})
        exec_att2.append({"R": R2, "phi": phi2_now})
        exec_xyz1.append(_p3_vec(x1)); exec_xyz2.append(_p3_vec(x2))

        # 3b) UKF estimation (optional)
        if do_est:
            idx  = len(exec_xyz1) - 1
            take = (idx % every) == 0

            def _u_tr(u):
                u = np.asarray(u, float).ravel()
                return u[:3] if u.size >= 3 else np.zeros(3)

            if ukf12 is not None:
                u2_tr = _u_tr(u2)
                ukf12.predict(dt, u=u2_tr, u_cov=None)
                if take:
                    p_obs = np.asarray(exec_xyz1[-1], float)
                    p_tgt = np.asarray(exec_xyz2[-1], float)
                    d = p_tgt - p_obs; d /= (np.linalg.norm(d) + 1e-12)
                    b_b = R1 @ d
                    az = np.arctan2(b_b[1], b_b[0]) + np.random.randn()*s_az
                    el = np.arctan2(b_b[2], np.sqrt(max(b_b[0]**2 + b_b[1]**2, 1e-18))) + np.random.randn()*s_el
                    ukf12.update(np.array([az, el]), p_obs=p_obs, R_wb=R1)
                    est_hist['meas12_azel'].append((p_obs.copy(), np.array([az, el])))
                else:
                    est_hist['meas12_azel'].append(None)
                est_hist['est12_xyz'].append(ukf12.x[:3].copy())

            if ukf21 is not None:
                u1_tr = _u_tr(u1)
                ukf21.predict(dt, u=u1_tr)
                if take:
                    p_obs = np.asarray(exec_xyz2[-1], float)
                    p_tgt = np.asarray(exec_xyz1[-1], float)
                    d = p_tgt - p_obs; d /= (np.linalg.norm(d) + 1e-12)
                    b_b = R2 @ d
                    az = np.arctan2(b_b[1], b_b[0]) + np.random.randn()*s_az
                    el = np.arctan2(b_b[2], np.sqrt(max(b_b[0]**2 + b_b[1]**2, 1e-18))) + np.random.randn()*s_el
                    ukf21.update(np.array([az, el]), p_obs=p_obs, R_wb=R2)
                    est_hist['meas21_azel'].append((p_obs.copy(), np.array([az, el])))
                else:
                    est_hist['meas21_azel'].append(None)
                est_hist['est21_xyz'].append(ukf21.x[:3].copy())

        # 4) FOV (selected agent)
        if fov_enabled:
            R_def = R2 if fov_agent == 2 else R1
            x_def = x2 if fov_agent == 2 else x1
            x_tgt = x1 if fov_agent == 2 else x2
            a_def3 = R_def[0] if align == 'x' else R_def[2]
            fov_axis_hist.append(a_def3)

            cam_cfg = cfg.get("camera", None)
            if fov_cfg.get("type","cone") == "pinhole" and (cam_cfg is not None):
                _, _, _, ok = project_point_pinhole(X_w=x_tgt[:3], x_def=x_def, cam_cfg=cam_cfg, R_wb=R_def)
                fov_seen_mask.append(bool(ok))
            else:
                seen, _ = in_fov(x_tgt[:D], x_def[:D], a_def3, fov_cfg, D)
                fov_seen_mask.append(bool(seen))
        else:
            fov_axis_hist.append(None); fov_seen_mask.append(False)

    ret = {
        'plan_hist1': plan_hist1, 'plan_hist2': plan_hist2,
        'plan_att1': plan_att1,   'plan_att2': plan_att2,
        'exec1_xyz': exec_xyz1,   'exec2_xyz': exec_xyz2,
        'exec_att1': exec_att1,   'exec_att2': exec_att2,
        'phi_hist1': phi_hist1,   'phi_hist2': phi_hist2,
        'fov_axis_hist': fov_axis_hist, 'fov_seen_mask': fov_seen_mask,
    }
    if do_est:
        ret.update(est_hist)
    return ret



# -------------------- FOV artists & cones --------------------
def make_body_axes_artists_3d(ax, colors=('tab:red','tab:green','tab:blue'), lw=2, alpha=0.9):
    bx, = ax.plot([], [], [], '-', lw=lw, alpha=alpha, color=colors[0])  # x_b
    by, = ax.plot([], [], [], '-', lw=lw, alpha=alpha, color=colors[1])  # y_b
    bz, = ax.plot([], [], [], '-', lw=lw, alpha=alpha, color=colors[2])  # z_b
    return dict(bx=bx, by=by, bz=bz)

def update_body_axes_artists_3d(lines, p, R_wb, L=(0.4,0.4,0.6)):
    p = np.asarray(p, float); x_b, y_b, z_b = R_wb[0], R_wb[1], R_wb[2]
    ends = [p + L[0]*x_b, p + L[1]*y_b, p + L[2]*z_b]
    for (ln, q) in zip((lines['bx'], lines['by'], lines['bz']), ends):
        ln.set_data([p[0], q[0]], [p[1], q[1]])
        ln.set_3d_properties([p[2], q[2]])

def draw_fov_cone_3d(ax, x_def, axis, fov_cfg, n=24, color='C1', alpha=0.12, align="x"):
    """
    Draw a circular cone aligned with `axis` and apex at x_def[:3].
    fov_cfg: {"range": float, "hfov_deg": float}  # interpreted as a circular half-angle
    """
    p0   = np.asarray(x_def[:3], float)
    axis = _unit(np.asarray(axis, float))
    R    = world_to_body_R(axis, 3, align=align)  # rows: x_b,y_b,z_b (world→body)

    rng  = float(fov_cfg["range"])
    assert rng > 0.0
    half = 0.5*np.deg2rad(float(fov_cfg["hfov_deg"]))
    radius = rng * np.tan(half)

    n = max(int(n), 3)
    ts = np.linspace(0, 2*np.pi, n, endpoint=False)

    if align == "x":
        # circle in y–z plane at x = rng (x-forward)
        circ_b = np.vstack([np.full_like(ts, rng),
                            radius*np.cos(ts),
                            radius*np.sin(ts)])
    else:
        # circle in x–y plane at z = rng (z-forward)
        circ_b = np.vstack([radius*np.cos(ts),
                            radius*np.sin(ts),
                            np.full_like(ts, rng)])

    # body→world
    circ_w = (R.T @ circ_b).T + p0[None, :]

    # side faces (triangles fan)
    verts = [[p0, circ_w[i], circ_w[(i+1) % len(circ_w)]] for i in range(len(circ_w))]
    coll  = Poly3DCollection(verts, facecolors=color, alpha=alpha, edgecolors='none')
    ax.add_collection3d(coll)

    rim_line, = ax.plot(circ_w[:,0], circ_w[:,1], circ_w[:,2], color=color, alpha=0.4, lw=1)
    return coll, rim_line

def _image_corners_px(W, H):
    # (u, v) pixel corners in order: TL, TR, BR, BL
    return np.array([[0,   0],
                     [W-1, 0],
                     [W-1, H-1],
                     [0,   H-1]], dtype=float)

def _corner_rays_camera(corners_px, fx, fy, cx, cy, align="z"):
    """
    Return 4 direction vectors (not normalized) in camera frame that
    pass through image corners. For align='z', rays are [ (u-cx)/fx, (v-cy)/fy, 1 ].
    For align='x', rays are [ 1, (u-cx)/fx, (v-cy)/fy ].
    """
    u = corners_px[:, 0]
    v = corners_px[:, 1]
    if align == "z":
        return np.stack([(u - cx)/fx, (v - cy)/fy, np.ones_like(u)], axis=1)
    else:  # align == "x"
        return np.stack([np.ones_like(u), (u - cx)/fx, (v - cy)/fy], axis=1)

def _scale_rays_to_plane(rays, depth, align="z"):
    """
    Scale each ray so that it intersects the plane at 'depth':
    If align='z': set Z = depth.
    If align='x': set X = depth.
    """
    rays = np.asarray(rays, float)
    out = rays.copy()
    if align == "z":
        s = depth / rays[:, 2]  # Z component
        out *= s[:, None]
    else:  # align == "x"
        s = depth / rays[:, 0]  # X component
        out *= s[:, None]
    return out

def draw_camera_frustum_3d(ax, x_def, axis=None, cam_cfg=None,
                           color='C2', alpha=0.10,
                           draw_edges=True, draw_rays=True,
                           lw=1.0, rim_alpha=0.55, ray_alpha=0.35,
                           R_wb=None):
    """
    Draw a pinhole camera frustum using intrinsics and near/far planes.

    Args
    ----
    ax      : Matplotlib 3D axes
    x_def   : state vector; camera position at x_def[:3]
    axis    : boresight direction in WORLD coords
    cam_cfg : dict with keys {W,H,fx,fy,cx,cy,near,far,align}
              align is 'x' (x-forward) or 'z' (z-forward)

    Returns
    -------
    coll  : Poly3DCollection for filled frustum (near/far + 4 side quads)
    edges : list of Line3D objects for rims/rays (for later removal)
    """
    # --- pose + rotation conventions ---
    p_cam = np.asarray(x_def[:3], float)
    align = cam_cfg.get("align", "x")  # must match your boresight convention

    # world→camera; use transpose for camera→world
    if R_wb is None:
        axis = _unit(np.asarray(axis, float))
        R_wc = world_to_body_R(axis, 3, align=align)
    else:
        R_wc = np.asarray(R_wb, float)

    def cam_to_world(Pc):
        return (R_wc.T @ Pc.T).T + p_cam[None, :]

    # --- intrinsics / depth bounds ---
    W, H = float(cam_cfg["W"]), float(cam_cfg["H"])
    fx, fy = float(cam_cfg["fx"]), float(cam_cfg["fy"])
    cx, cy = float(cam_cfg["cx"]), float(cam_cfg["cy"])
    near, far = float(cam_cfg["near"]), float(cam_cfg["far"])
    assert near > 0.0 and far > near, "Require 0 < near < far"
    assert fx != 0.0 and fy != 0.0, "fx/fy must be nonzero"

    # --- corner rays (TL, TR, BR, BL) in camera frame ---
    corners_px = _image_corners_px(W, H)
    rays_c     = _corner_rays_camera(corners_px, fx, fy, cx, cy, align=align)

    # scale rays to near/far planes (depth along align axis)
    near_c = _scale_rays_to_plane(rays_c, near, align=align)  # (4,3)
    far_c  = _scale_rays_to_plane(rays_c,  far,  align=align)  # (4,3)

    # transform to world
    near_w = cam_to_world(near_c)
    far_w  = cam_to_world(far_c)

    # --- build faces: 4 side quads + caps ---
    # consistent winding using TL(0)->TR(1)->BR(2)->BL(3)
    quads = []
    for i, j in [(0,1), (1,2), (2,3), (3,0)]:
        quads.append([near_w[i], near_w[j], far_w[j], far_w[i]])   # sides
    quads.append([near_w[0], near_w[1], near_w[2], near_w[3]])     # near cap
    quads.append([far_w[0],  far_w[1],  far_w[2],  far_w[3]])      # far cap

    coll = Poly3DCollection(quads, facecolors=color, alpha=alpha, edgecolors='none')
    ax.add_collection3d(coll)

    # --- optional edges/rays for clarity (return handles so we can remove later) ---
    edges = []
    if draw_edges:
        (ln_far,)  = ax.plot(*far_w[[0,1,2,3,0]].T,  color=color, alpha=rim_alpha, lw=lw)
        (ln_near,) = ax.plot(*near_w[[0,1,2,3,0]].T, color=color, alpha=rim_alpha*0.9, lw=max(0.8*lw, 0.6))
        edges.extend([ln_far, ln_near])

    if draw_rays:
        for k in range(4):
            (ln,) = ax.plot(*np.vstack([p_cam, far_w[k]]).T, color=color, alpha=ray_alpha, lw=lw)
            edges.append(ln)

    return coll, edges


def project_point_pinhole(X_w, x_def, cam_cfg, axis=None, R_wb=None):
    """
    Project a world point into pixel coordinates.
    Returns (u, v, depth, visible_bool).
    depth = Z_cam if align='z', or X_cam if align='x'.
    """
    p_cam = np.asarray(x_def[:3], float)
    align = cam_cfg.get("align", "z")
    R_wc  = np.asarray(R_wb, float) if R_wb is not None else \
            world_to_body_R(_unit(np.asarray(axis,float)), 3, align=align)
    X_c = R_wc @ (np.asarray(X_w, float) - p_cam)

    # world → camera
    if align == "z":
        depth = X_c[2]
        if depth <= 0:
            return None, None, depth, False
        u = cam_cfg["fx"] * (X_c[0] / depth) + cam_cfg["cx"]
        v = cam_cfg["fy"] * (X_c[1] / depth) + cam_cfg["cy"]
    else:  # align == 'x'
        depth = X_c[0]
        if depth <= 0:
            return None, None, depth, False
        u = cam_cfg["fx"] * (X_c[1] / depth) + cam_cfg["cx"]
        v = cam_cfg["fy"] * (X_c[2] / depth) + cam_cfg["cy"]

    W,H   = cam_cfg["W"], cam_cfg["H"]
    near, far = cam_cfg["near"], cam_cfg["far"]

    in_front   = (depth >= near) and (depth <= far)
    in_pixels  = (0 <= u < W) and (0 <= v < H)
    return float(u), float(v), float(depth), (in_front and in_pixels)

def points_in_fov_mask(Xw_list, x_def, axis, cam_cfg):
    """Vector helper: returns a boolean mask for many points."""
    mask = []
    for X in Xw_list:
        _, _, _, ok = project_point_pinhole(X, x_def, axis, cam_cfg)
        mask.append(ok)
    return np.array(mask, dtype=bool)


# ---------- tiny helper ----------
def _legend_clean(handles):
    """Return (handles, labels) with any private/empty labels removed."""
    pairs = [(h, getattr(h, "get_label", lambda: "")()) for h in handles if h is not None]
    pairs = [(h, lab) for (h, lab) in pairs if lab and not str(lab).startswith('_')]
    return [h for (h, _) in pairs], [lab for (_, lab) in pairs]

# ---- axis label helper (unit label + optional tick scaling) ----
def _label_axes_3d(ax, scale: float | None = None, unit: str = "m", label_only: bool = True):
    """
    If scale is None or 1.0: show plain labels like 'x [m]'.
    If scale is e.g. 1e2 and label_only=True: labels show 'x (10^2 m)'.
    If scale is e.g. 1e2 and label_only=False: tick numbers are divided by 1e2
    and labels show 'x [m]'.
    """
    import numpy as np
    from matplotlib.ticker import FuncFormatter

    def _pow10_str(s):
        k = int(round(np.log10(s)))
        return r"$10^{%d}$" % k

    if scale is None or abs(scale - 1.0) < 1e-12:
        ax.set_xlabel(r"$x$ [%s]" % unit)
        ax.set_ylabel(r"$y$ [%s]" % unit)
        ax.set_zlabel(r"$z$ [%s]" % unit)
        return

    if label_only:
        s = _pow10_str(scale)
        ax.set_xlabel(rf"$x$ ({s} {unit})")
        ax.set_ylabel(rf"$y$ ({s} {unit})")
        ax.set_zlabel(rf"$z$ ({s} {unit})")
    else:
        fmt = FuncFormatter(lambda val, pos: f"{val/scale:g}")
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.zaxis.set_major_formatter(fmt)
        ax.set_xlabel(r"$x$ [%s]" % unit)
        ax.set_ylabel(r"$y$ [%s]" % unit)
        ax.set_zlabel(r"$z$ [%s]" % unit)


# -------------------- animation (triads + legend fixes) --------------------
def animate_rollout_3d(frames_dict, save_path="traj_3D.gif", fps=20, cfg=None,
                       show_fov=True, show_axes=True):
    import shutil
    from matplotlib import animation
    from matplotlib.animation import FFMpegWriter, PillowWriter

    if cfg is None:
        raise ValueError("cfg must be provided")

    fov_cfg  = cfg.get("fov", {})
    att_cfg  = cfg.get("att", {})
    viz_cfg  = cfg.get("viz", {})
    agent_id = int(fov_cfg.get("agent", 2))

    triad_colors = tuple(viz_cfg.get('triad_colors', ('tab:red','tab:green','tab:blue')))
    triad_labels = tuple(viz_cfg.get('triad_labels', ('x_b (boresight)', 'y_b', 'z_b')))
    triad_leg_loc = viz_cfg.get('triad_leg_loc', 'lower left')
    triad_leg_ncol = int(viz_cfg.get('triad_leg_ncol', 3))
    triad_leg_title = viz_cfg.get('triad_leg_title', 'Body axes')
    L_tri = tuple(cfg.get("viz", {}).get("triad_len", (0.35, 0.35, 0.55)))

    plan_hist1 = frames_dict.get('plan_hist1', [])
    plan_hist2 = frames_dict.get('plan_hist2', [])
    exec1      = frames_dict.get('exec1_xyz', [])
    exec2      = frames_dict.get('exec2_xyz', [])
    axis_hist  = frames_dict.get('fov_axis_hist', [])
    seen_mask  = frames_dict.get('fov_seen_mask', [])

    # estimation overlays
    est12 = frames_dict.get('est12_xyz', [])
    est21 = frames_dict.get('est21_xyz', [])
    has_est12 = bool(est12) and any(e is not None for e in est12)
    has_est21 = bool(est21) and any(e is not None for e in est21)
    est_enabled = bool(cfg.get('est', {}).get('enabled', False))

    only_est  = bool(cfg.get('viz', {}).get('only_est', False))
    show_meas = bool(cfg.get('viz', {}).get('show_meas', False))

    n_frames = min(len(exec1), len(exec2))
    if n_frames < 2:
        if n_frames == 1:
            exec1 = exec1 + [exec1[-1]]
            exec2 = exec2 + [exec2[-1]]
            plan_hist1 = plan_hist1 + [plan_hist1[-1] if plan_hist1 else []]
            plan_hist2 = plan_hist2 + [plan_hist2[-1] if plan_hist2 else []]
            n_frames = 2
        else:
            raise ValueError("No frames to animate.")

    fig = plt.figure(figsize=(7,6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.grid(True)

    # NEW: axis labels / tick scaling (viz.axis_scale, viz.axis_unit, viz.axis_label_only)
    scale = float(viz_cfg.get("axis_scale", 1.0))        # e.g., 1e2 for “10^2 m”
    unit  = str(viz_cfg.get("axis_unit", "m"))
    label_only = bool(viz_cfg.get("axis_label_only", True))
    _label_axes_3d(ax, scale=scale, unit=unit, label_only=label_only)

    # arena limits
    ar = cfg.get("arena", {})
    if ar.get("type") == "box":
        ax.set_xlim(ar["xmin"], ar["xmax"])
        ax.set_ylim(ar["ymin"], ar["ymax"])
        ax.set_zlim(ar.get("zmin", -0.5), ar.get("zmax", 0.5))
    elif ar.get("type") == "sphere":
        cx, cy, cz, R = ar["cx"], ar["cy"], ar["cz"], ar["r"]
        ax.set_xlim(cx-R, cx+R); ax.set_ylim(cy-R, cy+R); ax.set_zlim(cz-R, cz+R)

    # ---- artists ----
    plan1_ln = plan2_ln = exe1_ln = exe2_ln = dot1 = dot2 = None
    me12_ln = me21_ln = None
    handles_for_legend = []

    if not only_est:
        plan1_ln, = ax.plot([], [], [], '--', lw=1, alpha=0.6, label='Plan P1')
        plan2_ln, = ax.plot([], [], [], '--', lw=1, alpha=0.6, label='Plan P2')
        exe1_ln,  = ax.plot([], [], [], '-',  lw=2, label='Exec P1')
        exe2_ln,  = ax.plot([], [], [], '-',  lw=2, label='Exec P2')
        dot1,     = ax.plot([], [], [], 'o',  ms=6, label='_nolegend_')
        dot2,     = ax.plot([], [], [], 'o',  ms=6, label='_nolegend_')
        handles_for_legend += [plan1_ln, plan2_ln, exe1_ln, exe2_ln]

    est12_ln, = ax.plot([], [], [], ':', lw=1.8, marker='x', ms=6, mew=1.0,
                        label=('P2 est by P1' if (est_enabled and has_est12) else '_nolegend_'))
    est21_ln, = ax.plot([], [], [], ':', lw=1.8, marker='x', ms=6, mew=1.0,
                        label=('P1 est by P2' if (est_enabled and has_est21) else '_nolegend_'))
    if est_enabled and has_est12: handles_for_legend.append(est12_ln)
    if est_enabled and has_est21: handles_for_legend.append(est21_ln)

    if (not only_est) and show_meas:
        me12_ln,  = ax.plot([], [], [], '-',  lw=1.0, alpha=0.45, label='_nolegend_')
        me21_ln,  = ax.plot([], [], [], '-',  lw=1.0, alpha=0.45, label='_nolegend_')

    handles, labels = _legend_clean(handles_for_legend)
    leg_main = ax.legend(handles, labels, loc='upper left') if handles else None

    leg_axes = None
    if show_axes:
        leg_axes = add_triad_legend(ax,
                                    colors=triad_colors,
                                    labels=triad_labels,
                                    loc=triad_leg_loc,
                                    ncol=triad_leg_ncol,
                                    title=triad_leg_title,
                                    keep_legend=leg_main)

    att1_lines = make_body_axes_artists_3d(ax, colors=triad_colors) if show_axes else None
    att2_lines = make_body_axes_artists_3d(ax, colors=triad_colors) if show_axes else None
    for L in (att1_lines, att2_lines):
        if L:
            L['bx'].set_label('_nolegend_'); L['by'].set_label('_nolegend_'); L['bz'].set_label('_nolegend_')

    fov_art = {'coll': None, 'rim': None, 'edges': []}

    def _pos3(p_like):
        p = np.asarray(p_like, float).ravel()
        return p[:3] if p.size >= 3 else np.array([p[0], p[1], 0.0], float)

    def _def_pos(f):
        return exec2[f] if agent_id == 2 else exec1[f]

    def _clear_fov():
        for k in ('coll','rim'):
            art = fov_art.get(k)
            if art is not None:
                try: art.remove()
                except Exception: pass
                fov_art[k] = None
        for ln in fov_art.get('edges', []):
            try: ln.remove()
            except Exception: pass
        fov_art['edges'] = []

    def _R_exec(agent, idx):
        key = 'exec_att1' if agent == 1 else 'exec_att2'
        L = frames_dict.get(key, [])
        if L and idx < len(L) and 'R' in L[idx]:
            return np.asarray(L[idx]['R'], float)
        seq = exec1 if agent == 1 else exec2
        if idx > 0:
            dv = np.asarray(_pos3(seq[idx])) - np.asarray(_pos3(seq[idx-1]))
            axis = dv/(np.linalg.norm(dv)+1e-12) if np.linalg.norm(dv) > 1e-9 else np.array([1,0,0], float)
        else:
            axis = np.array([1,0,0], float)
        R = world_to_body_R(axis, 3, align=att_cfg.get('align','x'), up=att_cfg.get('up',[0,0,1]))
        phi_key = 'phi_hist1' if agent == 1 else 'phi_hist2'
        phis = frames_dict.get(phi_key, [])
        if phis and idx < len(phis):
            R = apply_roll_about_axis(R, float(phis[idx]), align=att_cfg.get('align','x'))
        return R

    def _set_fov(p, R_wb, idx):
        if not (show_fov and fov_cfg.get("enabled", False)):
            _clear_fov(); return
        col = fov_cfg.get("color","C1")
        if seen_mask and idx < len(seen_mask) and bool(seen_mask[idx]):
            col = 'tab:green'
        x_def = np.r_[p, [0,0,0]]
        if fov_cfg.get("type","cone") == "pinhole" and cfg.get("camera") is not None:
            coll, edges = draw_camera_frustum_3d(
                ax, x_def=x_def, cam_cfg=cfg["camera"],
                color=col, alpha=fov_cfg.get("alpha",0.15),
                R_wb=R_wb
            )
            coll.set_label('_nolegend_')
            for ln in edges: ln.set_label('_nolegend_')
            fov_art['coll'], fov_art['rim'], fov_art['edges'] = coll, None, edges
        else:
            coll, rim = draw_fov_cone_3d(
                ax, x_def, fov_cfg,
                color=col, alpha=fov_cfg.get("alpha",0.15),
                align=att_cfg.get('align','x'),
                R_wb=R_wb
            )
            coll.set_label('_nolegend_')
            if rim is not None: rim.set_label('_nolegend_')
            fov_art['coll'], fov_art['rim'] = coll, rim

    def init():
        for ln in (plan1_ln, plan2_ln, exe1_ln, exe2_ln, dot1, dot2,
                   est12_ln, est21_ln, me12_ln, me21_ln):
            if ln is not None:
                ln.set_data([], []); ln.set_3d_properties([])
        for L in (att1_lines, att2_lines):
            if L:
                for ln in (L['bx'], L['by'], L['bz']):
                    ln.set_data([], []); ln.set_3d_properties([])
        _clear_fov()
        return []

    def update(f):
        def _set3d(ln, xs, ys, zs):
            if ln is not None:
                ln.set_data(xs, ys); ln.set_3d_properties(zs)

        # estimates
        if est_enabled and has_est12:
            xs, ys, zs = zip(*est12[:f+1]); _set3d(est12_ln, xs, ys, zs)
        else:
            _set3d(est12_ln, [], [], [])
        if est_enabled and has_est21:
            xs, ys, zs = zip(*est21[:f+1]); _set3d(est21_ln, xs, ys, zs)
        else:
            _set3d(est21_ln, [], [], [])

        # plan/exec
        if not only_est:
            if plan_hist1 and f < len(plan_hist1) and plan_hist1[f]:
                xs, ys, zs = zip(*plan_hist1[f]); _set3d(plan1_ln, xs, ys, zs)
            else:
                _set3d(plan1_ln, [], [], [])
            if plan_hist2 and f < len(plan_hist2) and plan_hist2[f]:
                xs, ys, zs = zip(*plan_hist2[f]); _set3d(plan2_ln, xs, ys, zs)
            else:
                _set3d(plan2_ln, [], [], [])
            xs1 = [p[0] for p in exec1[:f+1]]; ys1 = [p[1] for p in exec1[:f+1]]; zs1 = [p[2] for p in exec1[:f+1]]
            xs2 = [p[0] for p in exec2[:f+1]]; ys2 = [p[1] for p in exec2[:f+1]]; zs2 = [p[2] for p in exec2[:f+1]]
            _set3d(exe1_ln, xs1, ys1, zs1); _set3d(exe2_ln, xs2, ys2, zs2)
            if dot1 is not None:
                x1,y1,z1 = exec1[min(f, len(exec1)-1)]
                dot1.set_data([x1],[y1]); dot1.set_3d_properties([z1])
            if dot2 is not None:
                x2,y2,z2 = exec2[min(f, len(exec2)-1)]
                dot2.set_data([x2],[y2]); dot2.set_3d_properties([z2])
        else:
            for ln in (plan1_ln, plan2_ln, exe1_ln, exe2_ln):
                _set3d(ln, [], [], [])
            if dot1 is not None: dot1.set_data([],[]); dot1.set_3d_properties([])
            if dot2 is not None: dot2.set_data([],[]); dot2.set_3d_properties([])

        # triads
        if show_axes:
            x1,y1,z1 = exec1[min(f, len(exec1)-1)]
            x2,y2,z2 = exec2[min(f, len(exec2)-1)]
            if att1_lines: update_body_axes_artists_3d(att1_lines, np.array([x1,y1,z1]), _R_exec(1, f), L=L_tri)
            if att2_lines: update_body_axes_artists_3d(att2_lines, np.array([x2,y2,z2]), _R_exec(2, f), L=L_tri)

        # FOV
        p_def = _pos3(_def_pos(f)); R_def = _R_exec(agent_id, f)
        _clear_fov(); _set_fov(p_def, R_def, f)
        return []

    out_path = save_path
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=n_frames, interval=int(1000//fps),
                                   blit=False, repeat=False)
    try:
        if out_path.lower().endswith(".mp4") and shutil.which("ffmpeg"):
            writer = FFMpegWriter(fps=fps, codec="libx264",
                                  bitrate=1800, extra_args=["-pix_fmt", "yuv420p"])
            anim.save(out_path, writer=writer, dpi=150)
        else:
            if not out_path.lower().endswith(".gif"):
                out_path = out_path.rsplit(".", 1)[0] + ".gif"
            writer = PillowWriter(fps=fps)
            anim.save(out_path, writer=writer)
    finally:
        plt.close(fig)
    print(f"Saved 3D animation to {out_path}")


# -------------------- interactive (triads + estimates + meas + legend hygiene) --------------------
def interactive_rollout_3d(
    frames_dict,
    cfg,
    title="Interactive 3D rollout",
    show_fov=True,
    show_axes=True,
    perf_skip_fov_every: int = 1,
):
    import ipywidgets as W
    from IPython.display import display

    def _legend_clean(handles):
        H = []
        for h in handles:
            if h is None:
                continue
            lab = getattr(h, "get_label", lambda: "")()
            if lab and not str(lab).startswith("_"):
                H.append(h)
        return H, [h.get_label() for h in H]

    def _as3_hist(seq):
        if not seq:
            return []
        if seq and seq[0] and len(seq[0][0]) == 2:
            return [[(x, y, 0.0) for (x, y) in fr] for fr in seq]
        return seq

    def _as3_exec(seq):
        if not seq:
            return []
        if seq and len(seq[0]) == 2:
            return [(x, y, 0.0) for (x, y) in seq]
        return seq

    fov_cfg  = cfg.get("fov", {})
    att_cfg  = cfg.get("att", {})
    viz_cfg  = cfg.get("viz", {})
    agent_id = int(fov_cfg.get("agent", 2))

    triad_colors   = tuple(viz_cfg.get("triad_colors", ("tab:red", "tab:green", "tab:blue")))
    triad_labels   = tuple(viz_cfg.get("triad_labels", ("x_b (boresight)", "y_b", "z_b")))
    triad_leg_loc  = viz_cfg.get("triad_leg_loc", "lower left")
    triad_leg_ncol = int(viz_cfg.get("triad_leg_ncol", 3))
    triad_leg_title = viz_cfg.get("triad_leg_title", "Body axes")
    L_tri = tuple(viz_cfg.get("triad_len", (0.35, 0.35, 0.55)))

    plan_hist1 = frames_dict.get("plan_hist1_3d", frames_dict.get("plan_hist1", []))
    plan_hist2 = frames_dict.get("plan_hist2_3d", frames_dict.get("plan_hist2", []))
    exec1      = frames_dict.get("exec1_xyz", frames_dict.get("exec1_xy", []))
    exec2      = frames_dict.get("exec2_xyz", frames_dict.get("exec2_xy", []))
    axis_hist  = frames_dict.get("fov_axis_hist", [])
    seen_mask  = frames_dict.get("fov_seen_mask", [])

    est12 = frames_dict.get("est12_xyz", [])
    est21 = frames_dict.get("est21_xyz", [])
    me12  = frames_dict.get("meas12_azel", [])
    me21  = frames_dict.get("meas21_azel", [])

    do_est_default    = bool(cfg.get("est", {}).get("enabled", False)) or bool(est12 or est21)
    show_meas_default = bool(viz_cfg.get("show_meas", False)) and do_est_default

    plan_hist1 = _as3_hist(plan_hist1)
    plan_hist2 = _as3_hist(plan_hist2)
    exec1      = _as3_exec(exec1)
    exec2      = _as3_exec(exec2)
    n_frames   = max(2, min(len(exec1), len(exec2)))

    L_meas = float(viz_cfg.get("meas_len", cfg.get("camera", {}).get("far", 15.0)))

    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.grid(True)
    ax.set_box_aspect((1, 1, 1))

    # NEW: axis labels / tick scaling (viz.axis_scale, viz.axis_unit, viz.axis_label_only)
    scale = float(viz_cfg.get("axis_scale", 1.0))
    unit  = str(viz_cfg.get("axis_unit", "m"))
    label_only = bool(viz_cfg.get("axis_label_only", True))
    _label_axes_3d(ax, scale=scale, unit=unit, label_only=label_only)

    # bounds
    ar = cfg.get("arena", {})
    if ar.get("type") == "box" or ({"xmin", "xmax", "ymin", "ymax"} <= set(ar.keys())):
        xmin, xmax = ar.get("xmin", -3), ar.get("xmax", 3)
        ymin, ymax = ar.get("ymin", -3), ar.get("ymax", 3)
        if "zmin" in ar and "zmax" in ar:
            zmin, zmax = ar["zmin"], ar["zmax"]
        else:
            zs = [p[2] for p in (exec1 + exec2)] or [0.0]
            zmin, zmax = min(zs) - 0.5, max(zs) + 0.5
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)
    elif ar.get("type") == "sphere" or ({"cx", "cy", "cz", "r"} <= set(ar.keys())):
        cx, cy, cz = ar.get("cx", 0.0), ar.get("cy", 0.0), ar.get("cz", 0.0)
        Rb = ar.get("r", 3.0)
        ax.set_xlim(cx - Rb, cx + Rb); ax.set_ylim(cy - Rb, cy + Rb); ax.set_zlim(cz - Rb, cz + Rb)
    else:
        pts = np.array(exec1 + exec2, float)
        if pts.size > 0:
            mn = pts.min(0); mx = pts.max(0)
            span = np.maximum(mx - mn, 1e-3); pad = 0.1 * span
            lo = mn - pad; hi = mx + pad
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])

    # artists
    plan1_ln, = ax.plot([], [], [], "--", lw=1, alpha=0.6, label="Plan P1", color="blue")
    plan2_ln, = ax.plot([], [], [], "--", lw=1, alpha=0.6, label="Plan P2", color="orange")
    exe1_ln,  = ax.plot([], [], [], "-",  lw=2, label="Exec P1", color="green")
    exe2_ln,  = ax.plot([], [], [], "-",  lw=2, label="Exec P2", color="red")

    dot1, = ax.plot([], [], [], "o", ms=6, color="cyan",  mec="k", mew=0.6, label="_nolegend_")
    dot2, = ax.plot([], [], [], "o", ms=6, color="orange", mec="k", mew=0.6, label="_nolegend_")

    est12_ln, = ax.plot([], [], [], ":", lw=1.8, marker="x", ms=6, mew=1.0, label="P2 est by P1")
    est21_ln, = ax.plot([], [], [], ":", lw=1.8, marker="x", ms=6, mew=1.0, label="P1 est by P2")

    def _blank_line3d(ln):
        # NaNs are the safest "no geometry" sentinel for 3D lines
        ln.set_data([np.nan], [np.nan])
        ln.set_3d_properties([np.nan])

    me12_ln, = ax.plot([], [], [], "-", lw=0.8, alpha=0.25, zorder=-10, label="_nolegend_")
    me21_ln, = ax.plot([], [], [], "-", lw=0.8, alpha=0.25, zorder=-10, label="_nolegend_")

    # start fully blank & hidden unless the checkbox turns them on
    _blank_line3d(me12_ln); me12_ln.set_visible(False)
    _blank_line3d(me21_ln); me21_ln.set_visible(False)


    att1_lines = make_body_axes_artists_3d(ax, colors=triad_colors) if show_axes else None
    att2_lines = make_body_axes_artists_3d(ax, colors=triad_colors) if show_axes else None

    fov_art = {"coll": None, "rim": None, "edges": []}

    def _clear_fov():
        for k in ("coll", "rim"):
            art = fov_art.get(k)
            if art is not None:
                try:
                    art.remove()
                except Exception:
                    pass
                fov_art[k] = None
        for ln in fov_art.get("edges", []):
            try:
                ln.remove()
            except Exception:
                pass
        fov_art["edges"] = []

    def _pos3(p_like):
        p = np.asarray(p_like, float).ravel()
        return p[:3] if p.size >= 3 else np.array([p[0], p[1], 0.0], float)

    def _R_exec(agent, idx):
        key = "exec_att1" if agent == 1 else "exec_att2"
        L = frames_dict.get(key, [])
        if L and idx < len(L) and "R" in L[idx]:
            return np.asarray(L[idx]["R"], float)
        seq = exec1 if agent == 1 else exec2
        if idx > 0:
            dv = np.asarray(_pos3(seq[idx])) - np.asarray(_pos3(seq[idx - 1]))
            n = np.linalg.norm(dv)
            axis = (dv / (n + 1e-12)) if n > 1e-9 else np.array([1, 0, 0], float)
        else:
            axis = np.array([1, 0, 0], float)
        R = world_to_body_R(axis, 3, align=att_cfg.get("align", "x"), up=att_cfg.get("up", [0, 0, 1]))
        phi_key = "phi_hist1" if agent == 1 else "phi_hist2"
        phis = frames_dict.get(phi_key, [])
        if phis and idx < len(phis):
            R = apply_roll_about_axis(R, float(phis[idx]), align=att_cfg.get("align", "x"))
        return R

    def _set3d(ln, xs, ys, zs):
        if ln is not None:
            ln.set_data(xs, ys); ln.set_3d_properties(zs)

    def _meas_ray(p_obs, R_wb, az, el, L):
        c = np.cos(el)
        v_b = np.array([c * np.cos(az), c * np.sin(az), np.sin(el)])
        v_w = R_wb.T @ v_b
        pF = p_obs + L * v_w
        return p_obs, pF

    def _draw_fov(idx, p, R_wb):
        _clear_fov()
        if not (t_fov.value and fov_cfg.get("enabled", False)):
            return
        col = fov_cfg.get("color", "C1")
        if seen_mask and idx < len(seen_mask) and bool(seen_mask[idx]):
            col = "tab:green"
        x_def = np.r_[p, [0, 0, 0]]
        if fov_cfg.get("type", "cone") == "pinhole" and cfg.get("camera") is not None:
            coll, edges = draw_camera_frustum_3d(
                ax, x_def=x_def, cam_cfg=cfg["camera"], color=col,
                alpha=fov_cfg.get("alpha", 0.15), R_wb=R_wb
            )
            coll.set_label("_nolegend_")
            for ln in edges:
                ln.set_label("_nolegend_")
            fov_art["coll"], fov_art["rim"], fov_art["edges"] = coll, None, edges
        else:
            coll, rim = draw_fov_cone_3d(
                ax, x_def, fov_cfg, color=col, alpha=fov_cfg.get("alpha", 0.15),
                align=att_cfg.get("align", "x"), R_wb=R_wb
            )
            coll.set_label("_nolegend_")
            if rim is not None:
                rim.set_label("_nolegend_")
            fov_art["coll"], fov_art["rim"] = coll, rim

    s_frame = W.IntSlider(min=0, max=n_frames - 1, step=1, value=0, description="frame")
    s_azim  = W.IntSlider(min=-180, max=180, step=1, value=45, description="azim")
    s_elev  = W.IntSlider(min=-10,  max=90,  step=1, value=25, description="elev")

    t_plan  = W.Checkbox(value=True,              description="show plan")
    t_axes  = W.Checkbox(value=show_axes,         description="show axes")
    t_fov   = W.Checkbox(value=show_fov,          description="show FOV")
    t_est   = W.Checkbox(value=do_est_default,    description="show estimates",
                         disabled=not do_est_default)
    t_meas  = W.Checkbox(value=show_meas_default, description="show meas rays",
                         disabled=not do_est_default)

    play = W.Play(min=0, max=n_frames - 1, step=1, interval=50, value=0)
    W.jslink((play, "value"), (s_frame, "value"))

    leg_main = None
    leg_axes = None

    def build_main_legend(include_est=False):
        nonlocal leg_main
        if leg_main is not None:
            try:
                leg_main.remove()
            except Exception:
                pass
            leg_main = None
        handles = [plan1_ln, plan2_ln, exe1_ln, exe2_ln]
        if include_est and do_est_default:
            handles += [est12_ln, est21_ln]
        H, Lbls = _legend_clean(handles)
        leg_main = ax.legend(H, Lbls, loc="upper left") if H else None
        if leg_axes is not None and leg_main is not None:
            ax.add_artist(leg_axes)

    def ensure_triad_legend():
        nonlocal leg_axes
        if leg_axes is None:
            proxies = [Line2D([0], [0], lw=2, color=c, label=lab)
                       for c, lab in zip(triad_colors, triad_labels)]
            leg_axes = ax.legend(
                proxies, triad_labels, loc=triad_leg_loc,
                ncol=triad_leg_ncol, title=triad_leg_title
            )
            for p in proxies:
                p.set_label("_nolegend_")
            if leg_main is not None:
                ax.add_artist(leg_main)

    build_main_legend(include_est=t_est.value)
    if show_axes:
        ensure_triad_legend()

    def redraw(f):
        if t_plan.value and plan_hist1:
            ph1 = plan_hist1[min(f, len(plan_hist1) - 1)]
            if ph1: xs, ys, zs = zip(*ph1); _set3d(plan1_ln, xs, ys, zs)
            else:   _set3d(plan1_ln, [], [], [])
            ph2 = plan_hist2[min(f, len(plan_hist2) - 1)] if plan_hist2 else []
            if ph2: xs, ys, zs = zip(*ph2); _set3d(plan2_ln, xs, ys, zs)
            else:   _set3d(plan2_ln, [], [], [])
        else:
            _set3d(plan1_ln, [], [], [])
            _set3d(plan2_ln, [], [], [])

        xs1 = [p[0] for p in exec1[: f + 1]]; ys1 = [p[1] for p in exec1[: f + 1]]; zs1 = [p[2] for p in exec1[: f + 1]]
        xs2 = [p[0] for p in exec2[: f + 1]]; ys2 = [p[1] for p in exec2[: f + 1]]; zs2 = [p[2] for p in exec2[: f + 1]]
        _set3d(exe1_ln, xs1, ys1, zs1)
        _set3d(exe2_ln, xs2, ys2, zs2)

        p1 = np.asarray(exec1[min(f, len(exec1) - 1)])
        p2 = np.asarray(exec2[min(f, len(exec2) - 1)])
        dot1.set_data([p1[0]], [p1[1]]); dot1.set_3d_properties([p1[2]])
        dot2.set_data([p2[0]], [p2[1]]); dot2.set_3d_properties([p2[2]])

        if t_axes.value:
            if att1_lines: update_body_axes_artists_3d(att1_lines, p1, _R_exec(1, f), L=L_tri)
            if att2_lines: update_body_axes_artists_3d(att2_lines, p2, _R_exec(2, f), L=L_tri)
        else:
            for L in (att1_lines, att2_lines):
                if L:
                    for ln in (L["bx"], L["by"], L["bz"]):
                        _set3d(ln, [], [], [])

        if t_est.value and est12:
            xs, ys, zs = zip(*est12[: f + 1]); _set3d(est12_ln, xs, ys, zs)
        else:
            _set3d(est12_ln, [], [], [])
        if t_est.value and est21:
            xs, ys, zs = zip(*est21[: f + 1]); _set3d(est21_ln, xs, ys, zs)
        else:
            _set3d(est21_ln, [], [], [])

        if perf_skip_fov_every < 1:
            kskip = 1
        else:
            kskip = perf_skip_fov_every
        if f % kskip == 0:
            p_def = np.asarray(exec2[f] if agent_id == 2 else exec1[f])
            R_def = _R_exec(agent_id, f)
            _draw_fov(f, p_def, R_def)

        if t_meas.value and do_est_default:
            if me12 and f < len(me12) and me12[f] is not None:
                p_obs, z = me12[f]
                R1 = _R_exec(1, f)
                p0, pF = _meas_ray(np.asarray(p_obs, float), R1, float(z[0]), float(z[1]), L_meas)
                me12_ln.set_data([p0[0], pF[0]], [p0[1], pF[1]])
                me12_ln.set_3d_properties([p0[2], pF[2]])
            else:
                me12_ln.set_data([], []); me12_ln.set_3d_properties([])

            if me21 and f < len(me21) and me21[f] is not None:
                p_obs, z = me21[f]
                R2 = _R_exec(2, f)
                p0, pF = _meas_ray(np.asarray(p_obs, float), R2, float(z[0]), float(z[1]), L_meas)
                me21_ln.set_data([p0[0], pF[0]], [p0[1], pF[1]])
                me21_ln.set_3d_properties([p0[2], pF[2]])
            else:
                me21_ln.set_data([], []); me21_ln.set_3d_properties([])
        else:
            me12_ln.set_data([], []); me12_ln.set_3d_properties([])
            me21_ln.set_data([], []); me21_ln.set_3d_properties([])

        ax.view_init(elev=s_elev.value, azim=s_azim.value)
        fig.canvas.draw_idle()

    s_frame.observe(lambda ch: redraw(ch["new"]), names="value")
    s_azim.observe(lambda ch: redraw(s_frame.value), names="value")
    s_elev.observe(lambda ch: redraw(s_frame.value), names="value")

    def _rebuild_and_redraw(_=None):
        build_main_legend(include_est=t_est.value)
        redraw(s_frame.value)

    t_plan.observe(_rebuild_and_redraw, names="value")
    t_est.observe(_rebuild_and_redraw,  names="value")
    t_axes.observe(lambda ch: redraw(s_frame.value), names="value")
    t_fov.observe(lambda ch: redraw(s_frame.value), names="value")
    t_meas.observe(lambda ch: redraw(s_frame.value), names="value")

    redraw(0)
    ui = W.VBox([
        W.HBox([play, s_frame]),
        W.HBox([s_azim, s_elev, t_plan, t_axes, t_fov, t_est, t_meas]),
    ])
    display(ui)


# -------------------- Triad legend (unchanged API; silences proxies) --------------------
def add_triad_legend(ax, colors=('tab:red','tab:green','tab:blue'),
                     labels=('x_b (boresight)', 'y_b', 'z_b'),
                     loc='lower left', ncol=1, title='Body axes',
                     keep_legend=None):
    proxies = [Line2D([0], [0], lw=2, color=c, label=lab) for c, lab in zip(colors, labels)]
    leg_axes = ax.legend(proxies, labels, loc=loc, ncol=ncol, title=title)
    if keep_legend is not None:
        ax.add_artist(keep_legend)
    # keep proxies out of future legend calls
    for p in proxies:
        p.set_label('_nolegend_')
    return leg_axes



# -------------------- dims & dynamics (generic)--------------------
def dims_from_D(D: int):
    assert D in (2,3), "Only D=2 or D=3 supported."
    return 2*D, D

def step_double_integrator_D(x, u, dt, D=3):
    """
    One forward-Euler step for a D-dim double integrator.
    x = [p(1..D), v(1..D)], u = [a(1..D)]
    """
    x = np.asarray(x, float).copy()
    u = np.asarray(u, float)
    p = x[:D]; v = x[D:]
    p_next = p + dt * v
    v_next = v + dt * u
    return np.r_[p_next, v_next]

# -------------------- dims & dynamics (orbital HCW)--------------------

# --- in game_sharedutils.py ---
def as_numpy_const(M):
    """
    Convert a constant CasADi matrix (MX/SX/DM) or array-like to a numpy float array.
    Works across CasADi variants: DM, MX->Function->DM, dict-return Functions.
    """
    import numpy as np
    import casadi as ca

    # Fast path: already numpy
    if isinstance(M, np.ndarray):
        return M.astype(float)

    # If it already behaves like DM (has .full()), use it
    try:
        return np.array(M.full(), dtype=float)
    except Exception:
        pass

    # Evaluate constant MX/SX via a zero-input Function.
    f = ca.Function('const_mat', [], [ca.MX(M)])
    out = f()  # can be DM, or sometimes a dict like {'o0': DM(...)} depending on build

    # Normalize to a DM-like object
    if isinstance(out, dict):
        # take first (and only) value
        out = next(iter(out.values()))
    elif isinstance(out, (list, tuple)):
        out = out[0]

    # Convert to numpy
    try:
        return np.array(out.full(), dtype=float)
    except Exception:
        # Fallback: last resort
        return np.array(out, dtype=float)



def hcw_mean_motion(hcw_cfg: dict):
    """
    Compute mean motion n [rad/s].
    Provide either {'n': ...} or {'mu': ..., 'r0': ...}, where
    mu ≈ 3.986004418e14 m^3/s^2 (Earth), r0 is circular ref. radius [m].
    """
    if "n" in hcw_cfg:
        return float(hcw_cfg["n"])
    mu = float(hcw_cfg.get("mu", 3.986004418e14))
    r0 = float(hcw_cfg["r0"])   # must exist if n not given
    return float(np.sqrt(mu / (r0**3)))

def _discretize_linear(Ac, Bc, dt, series_terms=18):
    """
    Discretize xdot = Ac x + Bc u with ZOH over dt.
    Returns (Ad, Bd) as casadi.MX. No CasADi expm plugin required.
    """
    import numpy as np
    import casadi as ca

    nx = int(Ac.shape[0])
    nu = int(Bc.shape[1])

    # Robustly get numpy copies even if Ac/Bc are MX
    Ac_np = as_numpy_const(Ac)
    Bc_np = as_numpy_const(Bc)

    # Build augmented matrix and exponentiate
    M = np.block([[Ac_np,              Bc_np],
                  [np.zeros((nu, nx)), np.zeros((nu, nu))]])

    try:
        from scipy.linalg import expm
        E = expm(M * dt)
    except Exception:
        # Small-dt fallback: truncated series
        E = np.eye(nx + nu)
        term = np.eye(nx + nu)
        Mdt = M * dt
        for k in range(1, series_terms + 1):
            term = term @ (Mdt / k)
            E += term

    Ad_np = E[:nx, :nx]
    Bd_np = E[:nx, nx:nx+nu]
    return ca.MX(Ad_np), ca.MX(Bd_np)


def hcw_discrete_mats(n: float, dt: float):
    """
    HCW dynamics in LVLH with state x=[dx,dy,dz,vx,vy,vz], input u=[ax,ay,az].
    Continuous-time:
      ẍ - 3 n^2 x - 2 n ẏ = ax
      ÿ + 2 n ẋ           = ay
      z̈ + n^2 z           = az
    Returns (Ad, Bd) as CasADi MX for one step Δt.
    """
    n = float(n)
    Ac = ca.MX.zeros(6,6)
    # kinematics
    Ac[0,3] = 1.0
    Ac[1,4] = 1.0
    Ac[2,5] = 1.0
    # dynamics
    Ac[3,0] = 3*n*n
    Ac[3,4] = 2*n
    Ac[4,3] = -2*n
    Ac[5,2] = -n*n

    Bc = ca.MX.zeros(6,3)
    Bc[3,0] = 1.0
    Bc[4,1] = 1.0
    Bc[5,2] = 1.0

    return _discretize_linear(Ac, Bc, dt)

def build_g_tilde_linear(nx, nu, T, N, Ad: ca.MX, Bd: ca.MX):
    """
    Linear shared equality constraints with fixed (Ad,Bd):
      x_t - (Ad x_{t-1} + Bd u_{t-1}) = 0, and x_0 - θᶦ = 0 for each player.
    Drop-in replacement for build_g_tilde(...).
    """
    nprim = T*nx + (T-1)*nu

    def g_tilde(tau, theta):
        g_list = []
        ofs = 0
        for p in range(N):
            tau_p = tau[ofs:ofs+nprim]; ofs += nprim
            xs, us = unpack_trajectory(tau_p, nx, nu, T)
            th_p = theta[p*nx:(p+1)*nx]
            g_list.append(xs[0] - th_p)  # IC
            for t in range(1, T):
                g_list.append(xs[t] - (Ad @ xs[t-1] + Bd @ us[t-1]))
        return ca.vcat(g_list)

    return g_tilde


# ---------- Attitude helpers (shared) ----------

def apply_roll_about_axis(R_wb, phi: float, align: str = "x"):
    """
    Rotate the body frame by roll φ around the boresight axis.
    Returns rows [x_b'; y_b'; z_b'].
    """
    x_b, y_b, z_b = R_wb[0], R_wb[1], R_wb[2]
    c, s = np.cos(phi), np.sin(phi)

    if align == "x":           # roll about x_b
        y_p =  c*y_b + s*z_b
        z_p = -s*y_b + c*z_b
        return np.vstack([x_b, y_p, z_p])

    # legacy: roll about z_b
    x_p =  c*x_b + s*y_b
    y_p = -s*x_b + c*y_b
    return np.vstack([x_p, y_p, z_b])


# --- Attitude (roll) augmentation -------------------------------------------


def augment_AB_for_att(Ad_tr, Bd_tr, dt, att_cfg):
    """
    Augment translational Ad,Bd with attitude states [φ, θ, ψ].
    Each evolves with a simple integrator: angle_{k+1} = angle_k + dt * rate.
    """
    Ad_tr = ca.MX(Ad_tr)
    Bd_tr = ca.MX(Bd_tr)
    nx_tr = Ad_tr.size1()
    nu_tr = Bd_tr.size2()

    n_att = 3   # roll, pitch, yaw
    n_ctrl = 3  # their rates

    # --- Augmented A ---
    Ad_aug = ca.MX.zeros(nx_tr + n_att, nx_tr + n_att)
    Ad_aug[:nx_tr, :nx_tr] = Ad_tr
    Ad_aug[nx_tr:, nx_tr:] = ca.MX_eye(n_att)  # φ,θ,ψ → themselves

    # --- Augmented B ---
    Bd_aug = ca.MX.zeros(nx_tr + n_att, nu_tr + n_ctrl)
    Bd_aug[:nx_tr, :nu_tr] = Bd_tr              # translational part unchanged
    Bd_aug[nx_tr:, nu_tr:] = dt * ca.MX_eye(n_att)  # angle += dt * rate

    idx = {
        "nx": nx_tr + n_att,
        "nu": nu_tr + n_ctrl,
        "i_phi":   nx_tr + 0,
        "i_theta": nx_tr + 1,
        "i_psi":   nx_tr + 2,
        "i_u_phi": nu_tr + 0,
        "i_u_theta": nu_tr + 1,
        "i_u_psi": nu_tr + 2,
    }
    return Ad_aug, Bd_aug, idx

def augment_bounds_with_att(x_lb, x_ub, u_lb, u_ub, att_cfg):
    """
    Expand bounds to include φ, θ, ψ states and their rates.
    """
    x_lb = np.asarray(x_lb, float).ravel()
    x_ub = np.asarray(x_ub, float).ravel()
    u_lb = np.asarray(u_lb, float).ravel()
    u_ub = np.asarray(u_ub, float).ravel()

    # Angle limits (can tweak via att_cfg)
    phi_lim   = att_cfg.get("phi_lim",   (-np.pi, np.pi))
    theta_lim = att_cfg.get("theta_lim", (-np.pi/2, np.pi/2))
    psi_lim   = att_cfg.get("psi_lim",   (-np.pi, np.pi))

    # Rate limits
    phi_dot_lim   = att_cfg.get("phi_dot_lim",   (-1.0, 1.0))
    theta_dot_lim = att_cfg.get("theta_dot_lim", (-1.0, 1.0))
    psi_dot_lim   = att_cfg.get("psi_dot_lim",   (-1.0, 1.0))

    # Expand state bounds
    x_lb = np.concatenate([x_lb, [phi_lim[0],   theta_lim[0],   psi_lim[0]]])
    x_ub = np.concatenate([x_ub, [phi_lim[1],   theta_lim[1],   psi_lim[1]]])

    # Expand control bounds
    u_lb = np.concatenate([u_lb, [phi_dot_lim[0],   theta_dot_lim[0],   psi_dot_lim[0]]])
    u_ub = np.concatenate([u_ub, [phi_dot_lim[1],   theta_dot_lim[1],   psi_dot_lim[1]]])

    return x_lb, x_ub, u_lb, u_ub


def pad_x0_with_att(x0_row, att_cfg, D):
    """
    Extend initial state with φ, θ, ψ.
    Defaults = 0.0 unless att_cfg specifies otherwise.
    """
    x0_row = np.asarray(x0_row, float).ravel()
    phi0   = att_cfg.get("phi0", 0.0)
    theta0 = att_cfg.get("theta0", 0.0)
    psi0   = att_cfg.get("psi0", 0.0)

    # Append angles to whatever translational state we have
    return np.concatenate([x0_row, [phi0, theta0, psi0]])


# -------------------- bounds & geometry --------------------
def make_bounds(cfg: dict):
    """
    Build box bounds for states and controls in D dims.
    x = [p(1..D), v(1..D)], u = [a(1..D)]
    If arena is a box/cuboid, use tight pos bounds; otherwise keep pos wide and let h~ handle geometry.
    """
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx, nu = dims_from_D(D)
    vmax, umax = float(cfg["vmax"]), float(cfg["umax"])
    ar = cfg["arena"]
    ar_type = ar.get("type", "box")

    # Control bounds
    u_lb = -umax * np.ones(nu, float)
    u_ub =  umax * np.ones(nu, float)

    BIG = 1e6
    if ar_type == "box":
        if D == 2:
            keys = ("xmin","xmax","ymin","ymax")
            assert all(k in ar for k in keys), "Missing 2D box bounds"
            p_lb = np.array([ar["xmin"], ar["ymin"]], float)
            p_ub = np.array([ar["xmax"], ar["ymax"]], float)
        else:
            keys = ("xmin","xmax","ymin","ymax","zmin","zmax")
            assert all(k in ar for k in keys), "Missing 3D box bounds"
            p_lb = np.array([ar["xmin"], ar["ymin"], ar["zmin"]], float)
            p_ub = np.array([ar["xmax"], ar["ymax"], ar["zmax"]], float)
    else:
        p_lb = -BIG * np.ones(D, float)
        p_ub =  BIG * np.ones(D, float)

    v_lb = -float(cfg["vmax"]) * np.ones(D, float)
    v_ub =  float(cfg["vmax"]) * np.ones(D, float)
    x_lb = np.r_[p_lb, v_lb]
    x_ub = np.r_[p_ub, v_ub]
    return x_lb, x_ub, u_lb, u_ub


# -------------------- shared constraints builders --------------------

def build_h_tilde(nx: int, nu: int, T: int, N: int, x_lb, x_ub, u_lb, u_ub, cfg: dict):
    """
    h̃(τ,θ) ≥ 0: bounds + arena + keep-out + pairwise separation.
    D-aware; supports 2D (box/circle/polygon + circles) and 3D (box/sphere/polyhedron + spheres).
    """
    D = int(cfg.get("D", 3))
    nprim = T*nx + (T-1)*nu
    dmin2 = float(cfg["sep_min"])**2
    arena  = cfg["arena"]
    artype = arena.get("type", "box")

    # Precompute polygon/polyhedron keep-in if needed
    polyA = polyb = None
    if artype == "polygon":
        assert D == 2, "polygon keep-in is 2D; use polyhedron for 3D."
        polyA_np, polyb_np = polygon_halfspaces(arena["vertices"])
        polyA, polyb = ca.MX(polyA_np), ca.MX(polyb_np)
    elif artype == "polyhedron":
        polyA = ca.MX(np.asarray(arena["A"], float))
        polyb = ca.MX(np.asarray(arena["b"], float))

    # obstacle key by D
    sphere_key = "circles" if D == 2 else "spheres"

    def h_fun(tau, theta):
        h_list, ofs = [], 0
        taus = []

        for _ in range(N):
            tau_p = tau[ofs:ofs+nprim]; ofs += nprim
            taus.append(tau_p)
            xs, us = unpack_trajectory(tau_p, nx, nu, T)

            # box bounds
            for t in range(T):
                if x_lb is not None: h_list.append(xs[t] - x_lb)
                if x_ub is not None: h_list.append(x_ub - xs[t])
            for t in range(T-1):
                if u_lb is not None: h_list.append(us[t] - u_lb)
                if u_ub is not None: h_list.append(u_ub - us[t])

            # arena keep-in
            if artype in ("circle","sphere"):
                c = ca.MX(np.array([arena.get(k) for k in (["cx","cy"] if D==2 else ["cx","cy","cz"])], float))
                r2 = float(arena["r"])**2
                for t in range(T):
                    p = xs[t][0:D]
                    h_list.append(r2 - ca.sumsqr(p - c))
            elif artype == "polygon":  # 2D
                for t in range(T):
                    p = xs[t][0:2]
                    h_list.append(polyb - polyA @ p)  # vector
            elif artype == "polyhedron":  # 3D
                for t in range(T):
                    p = xs[t][0:3]
                    h_list.append(polyb - polyA @ p)  # vector

            # keep-out spheres/circles
            for sph in cfg.get(sphere_key, []):
                o = ca.MX(np.array([sph.get(k) for k in (["cx","cy"] if D==2 else ["cx","cy","cz"])], float))
                r2 = float(sph["r"])**2
                for t in range(T):
                    p = xs[t][0:D]
                    h_list.append(ca.sumsqr(p - o) - r2)

        # pairwise separation for N=2
        if N == 2:
            xs1, _ = unpack_trajectory(taus[0], nx, nu, T)
            xs2, _ = unpack_trajectory(taus[1], nx, nu, T)
            for t in range(T):
                p1 = xs1[t][0:D]; p2 = xs2[t][0:D]
                h_list.append(ca.sumsqr(p1 - p2) - dmin2)

        return ca.vcat(h_list)

    return h_fun

# -------------------- frames & FOV --------------------
def _unit(v, eps: float = 1e-12):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / (n + eps)

def _skew(w):
    wx, wy, wz = w
    return np.array([[0, -wz,  wy],
                     [wz,  0, -wx],
                     [-wy, wx,  0]], float)

def minimal_rotation(a, b, eps: float = 1e-9):
    """3D: rotation matrix R s.t. R@a ≈ b with minimal angle (no spin)."""
    a = _unit(a); b = _unit(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v); c = float(np.dot(a, b))
    if s < eps:
        if c > 0: return np.eye(3)  # a ~ b
        # a ~ -b: 180° around any axis ⟂ a
        axis = _unit(np.cross(a, np.array([1,0,0]) if abs(a[0]) < 0.9 else np.array([0,1,0])))
        K = _skew(axis)
        return np.eye(3) + 2*(K @ K)
    axis = v / s
    K = _skew(axis)
    angle = np.arctan2(s, c)
    return np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)


def frame_from_axis_continuous(axis,
                               R_prev=None,
                               align: str = "x",
                               world_up=np.array([0.0, 0.0, 1.0])):
    """
    Build/propagate a continuous body frame with minimal spin about the boresight.
    Returns rows [x_b; y_b; z_b] expressed in WORLD.
    - align='x': x_b is boresight (x-forward).
    - align='z': z_b is boresight (legacy).
    Uses minimal_rotation(prev_boresight, new_boresight) to avoid flips.
    """
    a = _unit(np.asarray(axis, float))
    up = _unit(np.asarray(world_up, float))

    if R_prev is not None:
        if align == "x":
            # rotate previous frame so x_prev -> a with minimum angle
            x_prev = _unit(R_prev[0])
            Rdel = minimal_rotation(x_prev, a)   # Rdel @ x_prev ≈ a
            R = Rdel @ R_prev

            # re-orthonormalize while preserving x_b ~ a
            x_b = a
            y_tmp = R[1] - x_b*np.dot(R[1], x_b)
            if np.linalg.norm(y_tmp) < 1e-8:
                # singular: pick a safe reference not parallel to x_b
                ref = up if abs(np.dot(up, x_b)) < 0.98 else np.array([0,1,0], float)
                y_tmp = np.cross(ref, x_b)
            y_b = _unit(y_tmp)
            z_b = _unit(np.cross(x_b, y_b))
            y_b = _unit(np.cross(z_b, x_b))  # final Gram-Schmidt pass
            return np.vstack([x_b, y_b, z_b])

        else:  # align == "z"
            z_prev = _unit(R_prev[2])
            Rdel = minimal_rotation(z_prev, a)  # Rdel @ z_prev ≈ a
            R = Rdel @ R_prev

            z_b = a
            x_tmp = R[0] - z_b*np.dot(R[0], z_b)
            if np.linalg.norm(x_tmp) < 1e-8:
                ref = up if abs(np.dot(up, z_b)) < 0.98 else np.array([1,0,0], float)
                x_tmp = np.cross(ref, z_b)
            x_b = _unit(x_tmp)
            y_b = _unit(np.cross(z_b, x_b))
            x_b = _unit(np.cross(y_b, z_b))
            return np.vstack([x_b, y_b, z_b])

    # First frame (no R_prev): construct from world_up safely
    if align == "x":
        if abs(np.dot(a, up)) > 0.98:
            up = np.array([0,1,0], float) if abs(a[1]) < 0.9 else np.array([1,0,0], float)
        y_b = _unit(np.cross(up, a))
        z_b = _unit(np.cross(a, y_b))
        return np.vstack([a, y_b, z_b])
    else:
        if abs(np.dot(a, up)) > 0.98:
            up = np.array([1,0,0], float)
        x_b = _unit(np.cross(up, a))
        y_b = _unit(np.cross(a, x_b))
        return np.vstack([x_b, y_b, a])


def att_lin3d_AB(dt: float, att_cfg: dict):
    """
    Linear small-angle attitude model about a reference:
      x_att = [δθ(3); ω(3)],  u_att = τ(3)
      δθ^+ = δθ + dt * (ω - ω_ref)   # set ω_ref=0 to keep strictly linear
      ω^+  = ω  + dt * J^{-1} τ
    We keep ω_ref=0 to stay strictly linear (Ax+Bu); if you want ω_ref≠0, treat it as a known affine term.
    """
    import numpy as np
    J = np.asarray(att_cfg.get("J", [12.0, 10.0, 8.0]), float)
    if J.ndim == 2:   # if full matrix was given, use only the diagonal for B
        Jx, Jy, Jz = np.diag(J)
    else:
        Jx, Jy, Jz = J

    A = np.block([
        [np.eye(3), dt*np.eye(3)],   # δθ^+ = δθ + dt ω
        [np.zeros((3,3)), np.eye(3)] # ω^+  = ω
    ])
    Bin = np.diag([dt/Jx, dt/Jy, dt/Jz])
    B = np.vstack([np.zeros((3,3)), Bin])   # τ only affects ω-row
    return A.astype(float), B.astype(float)

def world_to_body_R(axis, D: int = 3, align: str = "x", up=(0.0, 0.0, 1.0)):
    """
    Build body frame rows [x_b; y_b; z_b].
    align='x' -> x_b aligned to axis (your boresight).
    align='z' -> z_b aligned to axis (legacy).
    """
    axis = _unit(axis)
    up   = _unit(np.asarray(up, float))

    if D == 2:
        # 2D: keep old behavior (no roll in plane)
        x_b = axis
        y_b = np.array([-axis[1], axis[0]])
        return np.vstack([x_b, y_b])

    # 3D:
    if align == "x":
        x_b = axis
        # pick a usable up vector (not parallel to axis)
        if abs(np.dot(x_b, up)) > 0.98:
            up = np.array([0.0, 1.0, 0.0]) if abs(x_b[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        y_b = _unit(np.cross(up, x_b))       # pitch axis
        z_b = _unit(np.cross(x_b, y_b))      # yaw axis
        return np.vstack([x_b, y_b, z_b])

    # align == "z" (legacy)
    z_b = axis
    ref = np.array([1.0, 0.0, 0.0]) if abs(z_b[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_b = _unit(np.cross(ref, z_b))
    y_b = _unit(np.cross(z_b, x_b))
    return np.vstack([x_b, y_b, z_b])


def in_fov(p_t, x_def, axis, fov_cfg, D: int):
    """Check if target p_t is within defender x_def FOV cone/wedge."""
    p_def = np.asarray(x_def[:D], float)
    rel   = np.asarray(p_t[:D], float) - p_def
    dist  = float(np.linalg.norm(rel))
    if dist > float(fov_cfg["range"]):
        return False, dist
    half = 0.5*np.deg2rad(float(fov_cfg["hfov_deg"]))
    cosang = float(np.dot(_unit(rel), _unit(axis)))
    return (cosang >= np.cos(half)), dist

