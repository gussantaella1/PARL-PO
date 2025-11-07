# game_3dutils.py
# Dedicated 3D utilities and entrypoints. Depends on game_sharedutils.
from __future__ import annotations
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from dyn_models import (
    dims_from_D,
    hcw_mean_motion, hcw_discrete_mats, as_numpy_const,
    augment_AB_for_att, augment_bounds_with_att, pad_x0_with_att,
    world_to_body_R, frame_from_axis_continuous, apply_roll_about_axis,
    make_bounds,
)



import importlib, game_3dutils, game_costs, neos_path_game,rl_infer
importlib.reload(game_3dutils)
importlib.reload(game_costs)
importlib.reload(neos_path_game)
importlib.reload(rl_infer)

# Keep only module-level imports and always reference via the module
import ukf_estimator, ekf_estimator
import importlib
importlib.reload(ukf_estimator)
importlib.reload(ekf_estimator)

# (Optional) local aliases to current classes after reload
AgentUKF = ukf_estimator.AgentUKF
AgentEKF = ekf_estimator.AgentEKF




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
    # Sphere if given (your CONFIG uses this)
    if {"cx","cy","cz","r"} <= set(ar.keys()) or ar.get("type") == "sphere":
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

    # -------- object-of-interest (keep-out for selected agents) --------
    # cfg example:
    # cfg["oi"] = {
    #   "enabled": True,
    #   "cx": 0.0, "cy": 0.0, "cz": 0.0,   # cz optional if D==2
    #   "r":  2.0,                         # keep-out radius
    #   "avoid_by": [1]                    # which agents must avoid (defaults to [1])
    # }
    oi = cfg.get("oi", {})
    if bool(oi.get("enabled", False)):
        oc = [float(oi.get(k, 0.0)) for k in (["cx","cy"] if D == 2 else ["cx","cy","cz"])]
        r2 = float(oi.get("r", 1.0))**2
        avoid_by = list(oi.get("avoid_by", [1]))
        for agent in (1, 2):
            if agent not in avoid_by:
                continue
            def _h_oi(m,k,_a=agent,_oc=tuple(oc),_r2=r2):
                s = 0.0
                for j in range(D):
                    s += (_x(m,_a,k,j) - _oc[j])**2
                return s - _r2   # ≥ 0 => outside the object
            funcs.append(_h_oi)

    return funcs


def _filter_kwargs(fn, kwargs):
    """Keep only kwargs that `fn` accepts."""
    import inspect
    sig = inspect.signature(fn)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


class KF_CV:
    """
    Common interface over AgentUKF / AgentEKF.

    - ctor: KF_CV(x0, P0, Q, R, dt, kind='auto'|'ukf'|'ekf', **kwargs)
      UKF-recognized kwargs: alpha, beta, kappa, dyn, hcw
    - predict(dt=None, u=None, **kwargs)
    - update(z, p_obs, R_wb, **kwargs)
    """
    def __init__(self, x0, P0, Q, R, dt, kind='auto', **kwargs):
        kind = (kind or 'auto').lower()

        has_ukf = (AgentUKF is not None)
        has_ekf = (AgentEKF is not None)

        if kind == 'ekf' and has_ekf:
            self._impl = AgentEKF(x0, P0, Q, R, dt)
        elif kind in ('ukf', 'auto') and has_ukf:
            # forward UKF-specific ctor kwargs, incl. dynamics selection
            ukf_ctor_keys = ('alpha', 'beta', 'kappa', 'dyn', 'hcw')
            ukf_ctor = {k: kwargs[k] for k in ukf_ctor_keys if k in kwargs}
            self._impl = AgentUKF(x0, P0, Q, R, dt, **ukf_ctor)
        elif has_ukf:
            self._impl = AgentUKF(x0, P0, Q, R, dt)
        elif has_ekf:
            self._impl = AgentEKF(x0, P0, Q, R, dt)
        else:
            raise RuntimeError("Neither AgentUKF nor AgentEKF is importable.")

    def __getattr__(self, name):
        return getattr(self._impl, name)

    def predict(self, dt=None, u=None, **kwargs):
        f = getattr(self._impl, 'predict')
        return f(dt=dt, u=u, **_filter_kwargs(f, kwargs))

    def update(self, z, p_obs, R_wb, **kwargs):
        f = getattr(self._impl, 'update')
        return f(z, p_obs=p_obs, R_wb=R_wb, **_filter_kwargs(f, kwargs))



# -------------------- RHC with execution & FOV (3D/2D-aware) --------------------
def run_rhc_and_collect_frames_3d(cfg: dict, steps: int | None = None,
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
    dyn = (cfg.get("dynamics") or "hcw").lower()
    if dyn == "hcw":
        n = hcw_mean_motion(cfg.get("hcw", {}))
        Ad_tr, Bd_tr = hcw_discrete_mats(n, dt)        # MX
    # else:
    #     Ad_tr, Bd_tr = step_double_integrator_D(D=D, dt=dt)  # MX

    # --- optional attitude augmentation (roll in state) ---
    att_cfg = cfg.get("att", {})
    use_att = bool(att_cfg.get('enabled', True))

    print(use_att)

    # raise("Debug")
    if use_att:
        Ad_mx, Bd_mx, idx = augment_AB_for_att(Ad_tr, Bd_tr, dt, att_cfg)
        nx, nu = idx["nx"], idx["nu"]
        i_phi  = idx["i_phi"]
    else:
        Ad_mx, Bd_mx = Ad_tr, Bd_tr
        nx, nu = nx_tr, nu_tr
        i_phi  = None

    # print("Ad_mx shape:", Ad_mx.size1(), "+",Ad_mx.size2())
    # print("Bd_mx shape:", Bd_mx.size1(), "+",Bd_mx.size2())

    # raise("Debug")

    # --- rollout length / turn length ---
    sim_time = cfg.get("sim_time", cfg.get("max_time", cfg.get("duration", None)))
    if steps is None:
        steps = max(1, int(np.ceil(float(sim_time)/dt))) if sim_time is not None else int(cfg.get("steps", 60))
    if turn_len is None:
        turn_len = int(cfg.get("turn_len", 3)) if "turn_seconds" not in cfg else \
                   max(1, int(round(float(cfg["turn_seconds"]) / float(dt))))

    # --- constraints (fixed Ad,Bd) ---
    x_lb, x_ub, u_lb, u_ub = make_bounds(cfg)
    if use_att:
        x_lb, x_ub, u_lb, u_ub = augment_bounds_with_att(x_lb, x_ub, u_lb, u_ub, att_cfg)

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

    # === optional UKFs (HCW-only) ===
    if do_est:
        s_az, s_el = np.deg2rad(est_cfg.get('meas_std_deg', (0.3, 0.3)))
        P0 = np.diag(est_cfg.get('P0_diag', [25,25,25, 1,1,1]))
        Q  = np.diag(est_cfg.get('Q_diag',  [1e-4,1e-4,1e-4, 1e-3,1e-3,1e-3]))
        Rm = np.diag([s_az**2, s_el**2])
        who   = (est_cfg.get('who') or 'both').lower()
        every = int(est_cfg.get('every', 1))

        # Force HCW in the estimator
        est_hist = {'est12_xyz': [], 'est21_xyz': [], 'meas12_azel': [], 'meas21_azel': []}
        hcw_cfg = cfg.get('hcw', {})           # pass same params used by the plant

        ukf12 = KF_CV(np.r_[x2[:3], np.zeros(3)], P0, Q, Rm, dt,
                    kind='ukf', dyn='hcw', hcw=hcw_cfg) if who in ('both','1->2') else None
        ukf21 = KF_CV(np.r_[x1[:3], np.zeros(3)], P0, Q, Rm, dt,
                    kind='ukf', dyn='hcw', hcw=hcw_cfg) if who in ('both','2->1') else None

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


def run_rhc_with_rl_and_collect_frames_3d(cfg: Dict[str, Any],
                                          steps: int | None = None,
                                          turn_len: int | None = None):
    """
    RL-only rollout that feeds the PPO policies with the SAME observation used in training:
        obs = [p1-center, p2-center, (p2-p1), v1, v2]
    and (optionally) the SAME observation normalization.
    """
    # -------------------- basics & dims --------------------
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx_tr, nu_tr = dims_from_D(D)
    T, dt = int(cfg["T"]), float(cfg["dt"])

    # -------------------- dynamics (Ad,Bd) --------------------
    dyn_name = (cfg.get("dynamics") or "hcw").lower()
    if dyn_name != "hcw":
        raise ValueError("This runner currently expects 'hcw' dynamics.")
    n = hcw_mean_motion(cfg.get("hcw", {}))
    Ad_tr, Bd_tr = hcw_discrete_mats(n, dt)

    # -------------------- optional attitude aug --------------------
    att_cfg = dict(cfg.get("att", {}))
    use_att = bool(att_cfg.get("enabled", False))
    if use_att:
        Ad_mx, Bd_mx, idx = augment_AB_for_att(Ad_tr, Bd_tr, dt, att_cfg)
        nx, nu = idx["nx"], idx["nu"]
        i_phi  = idx["i_phi"]
    else:
        Ad_mx, Bd_mx = Ad_tr, Bd_tr
        nx, nu = nx_tr, nu_tr
        i_phi  = None

    # -------------------- sim length --------------------
    if steps is None:
        steps = int(cfg.get("T_eval", cfg.get("steps", 60)))
    if turn_len is None:
        turn_len = 1  # N/A for RL; kept for API symmetry

    # -------------------- initial states --------------------
    x1 = pad_x0_with_att(cfg["x0"][0], att_cfg, D)[:nx] if use_att else np.asarray(cfg["x0"][0], float)[:nx].copy()
    x2 = pad_x0_with_att(cfg["x0"][1], att_cfg, D)[:nx] if use_att else np.asarray(cfg["x0"][1], float)[:nx].copy()

    # -------------------- numeric stepper --------------------
    Ad_np, Bd_np = as_numpy_const(Ad_mx), as_numpy_const(Bd_mx)
    def step_plant_single(x, u):
        x = np.asarray(x, float); u = np.asarray(u, float)
        return Ad_np @ x + Bd_np @ u

    # -------------------- arena & center --------------------
    ar = cfg.setdefault("arena", {})
    ar.setdefault("type", "sphere")
    ar.setdefault("cx", 0.0)
    ar.setdefault("cy", 0.0)
    ar.setdefault("cz", 0.0)
    ar.setdefault("r", 30.0)
    center = np.array([ar["cx"], ar["cy"], (ar.get("cz", 0.0) if D == 3 else 0.0)], float)[:D]

    # -------------------- obs normalization (if used in training) --------------------
    use_obsnorm = bool(cfg.get("obsnorm", False))
    obs_mean = None; obs_std = None
    if use_obsnorm and "obs_stats" in cfg:
        import os
        mp = cfg["obs_stats"].get("mean_path", None)
        sp = cfg["obs_stats"].get("std_path", None)
        if mp and os.path.exists(mp): obs_mean = np.load(mp)
        if sp and os.path.exists(sp): obs_std  = np.load(sp)

    def build_train_obs(x1_vec: np.ndarray, x2_vec: np.ndarray) -> np.ndarray:
        """Match Env._obs() from training exactly."""
        p1 = x1_vec[:D]; v1 = x1_vec[D:2*D]
        p2 = x2_vec[:D]; v2 = x2_vec[D:2*D]
        obs = np.concatenate([
            p1 - center,
            p2 - center,
            (p2 - p1),
            v1, v2
        ]).astype(np.float32)
        if use_obsnorm and (obs_mean is not None) and (obs_std is not None):
            obs = (obs - obs_mean.astype(np.float32)) / (obs_std.astype(np.float32) + 1e-8)
        return obs

    # -------------------- policy wrapper --------------------
    deterministic = bool(cfg.get("rl_eval_deterministic", True))
    pol = rl_infer.RLPolicy(cfg, device=cfg.get("device", "cpu"))
    umax = float(cfg.get("umax", 5e-4))
    debug_actions = bool(cfg.get("debug_actions", False))

    # choose best-available API on RLPolicy
    has_obs_api = hasattr(pol, "act_def_obs") and hasattr(pol, "act_att_obs")

    def _act_with_train_obs(which: str, x1_vec: np.ndarray, x2_vec: np.ndarray, deterministic: bool):
        obs = build_train_obs(x1_vec, x2_vec)
        if has_obs_api:
            if which == "def":
                return pol.act_def_obs(obs, deterministic=deterministic)
            else:
                return pol.act_att_obs(obs, deterministic=deterministic)
        else:
            # Fallback: if your RLPolicy ONLY accepts raw [p1,v1,p2,v2], we emulate Env._obs inside RLPolicy.
            # Pack raw, but warn once so you can update RLPolicy.
            if not getattr(_act_with_train_obs, "_warned", False):
                print("[info] RLPolicy has no *_obs API; falling back to raw pack_state. "
                      "Ensure RLPolicy reproduces training obs internally.")
                _act_with_train_obs._warned = True
            p1 = x1_vec[:D]; v1 = x1_vec[D:2*D]
            p2 = x2_vec[:D]; v2 = x2_vec[D:2*D]
            raw_state = rl_infer.RLPolicy.pack_state(p1, v1, p2, v2)
            if which == "def":
                return pol.act_def(raw_state, deterministic=deterministic)
            else:
                return pol.act_att(raw_state, deterministic=deterministic)

    # -------------------- viz helpers --------------------
    def _p3_row(x_row):
        return (float(x_row[0]), float(x_row[1]), float(x_row[2])) if D == 3 else \
               (float(x_row[0]), float(x_row[1]), 0.0)

    def _p3_vec(x_vec):
        return (float(x_vec[0]), float(x_vec[1]), float(x_vec[2])) if D == 3 else \
               (float(x_vec[0]), float(x_vec[1]), 0.0)

    # ---- attitude helpers (unchanged) ----
    def attitude_from_state(x, prev_R=None, prev_axisD=None):
        align    = att_cfg.get("align", "x")
        world_up = np.asarray(att_cfg.get("up", [0, 0, 1]), float)
        vmin     = float(att_cfg.get("min_speed_for_axis", 1e-3))
        DD = 3 if len(x) >= 6 else 2
        v  = np.asarray(x[DD:2*DD], float)
        n  = np.linalg.norm(v)
        if n > vmin:
            axisD = v / max(n, 1e-9)
        elif prev_axisD is not None:
            axisD = np.asarray(prev_axisD, float)
        else:
            axisD = np.array([1, 0, 0]) if align == "x" else np.array([0, 0, 1])
        axis3 = axisD if DD == 3 else np.array([axisD[0], axisD[1], 0.0], float)
        R = frame_from_axis_continuous(axis3, R_prev=prev_R, align=align, world_up=world_up)
        phi = 0.0
        if use_att and (i_phi is not None) and (i_phi < len(x)):
            phi = float(x[i_phi]); R = apply_roll_about_axis(R, phi, align=align)
        return R, axisD, phi

    def plan_attitudes_from_X(X, prev_axisD, prev_R):
        att_list, ax_prev, R_prev = [], prev_axisD, prev_R
        for _ in range(T):
            R, ax_prev, phi_t = attitude_from_state(X, prev_R=R_prev, prev_axisD=ax_prev)
            att_list.append({"R": R, "phi": phi_t}); R_prev = R
        return att_list, ax_prev, R_prev

    # -------------------- initial attitude seeds --------------------
    def _axis_from_vel_t0(x):
        v = np.asarray(x[D:2*D], float); n = float(np.linalg.norm(v))
        if n > float(att_cfg.get("min_speed_for_axis", 1e-3)):
            aD = v / n
        else:
            aD = np.array([1, 0, 0], float) if att_cfg.get("align","x") == "x" else np.array([0, 0, 1], float)
        if D == 2: return aD[:2], np.array([aD[0], aD[1], 0.0], float)
        return aD, aD

    prev_axis1D, axis1_0 = _axis_from_vel_t0(x1)
    prev_axis2D, axis2_0 = _axis_from_vel_t0(x2)
    phi1_0 = float(x1[i_phi]) if (use_att and i_phi is not None) else 0.0
    phi2_0 = float(x2[i_phi]) if (use_att and i_phi is not None) else 0.0
    R1_0 = apply_roll_about_axis(world_to_body_R(axis1_0, 3, align=att_cfg.get("align","x"), up=att_cfg.get("up",[0,0,1])), phi1_0, align=att_cfg.get("align","x"))
    R2_0 = apply_roll_about_axis(world_to_body_R(axis2_0, 3, align=att_cfg.get("align","x"), up=att_cfg.get("up",[0,0,1])), phi2_0, align=att_cfg.get("align","x"))
    prev_R1, prev_R2 = R1_0, R2_0

    # -------------------- logs for animator --------------------
    plan_hist1, plan_hist2 = [], []
    plan_att1,  plan_att2  = [], []
    exec_xyz1,  exec_xyz2  = [], []
    exec_att1,  exec_att2  = [], []
    phi_hist1,  phi_hist2  = [], []
    fov_axis_hist, fov_seen_mask = [], []

    # t=0 logs
    exec_xyz1.append(_p3_vec(x1)); exec_xyz2.append(_p3_vec(x2))
    exec_att1.append({"R": R1_0, "phi": phi1_0}); phi_hist1.append(phi1_0)
    exec_att2.append({"R": R2_0, "phi": phi2_0}); phi_hist2.append(phi2_0)
    fov_axis_hist.append(None); fov_seen_mask.append(False)  # keep shape even if FOV disabled

    # ==================== ROLLOUT ====================
    for k in range(steps):
        # 1) get actions from policies using TRAINER-STYLE OBS
        u1_tr = _act_with_train_obs("def", x1, x2, deterministic)
        u2_tr = _act_with_train_obs("att", x1, x2, deterministic)

        # 2) clip and embed into full control vectors
        u1_tr = np.clip(np.asarray(u1_tr, float), -umax, +umax)
        u2_tr = np.clip(np.asarray(u2_tr, float), -umax, +umax)
        if debug_actions and k < 3:
            obs_dbg = build_train_obs(x1, x2)
            print(f"[k={k:02d}] ||a_def||={np.linalg.norm(u1_tr):.3g}, ||a_att||={np.linalg.norm(u2_tr):.3g}, "
                  f"first obs entries={obs_dbg[:6]}")
        u1_full = np.zeros(nu); u2_full = np.zeros(nu)
        u1_full[:D] = u1_tr; u2_full[:D] = u2_tr

        # 3) plant step
        x1 = step_plant_single(x1, u1_full)
        x2 = step_plant_single(x2, u2_full)

        # 4) synth fixed-length "plan" horizons for animator
        plan1 = [_p3_row(x1)] * T
        plan2 = [_p3_row(x2)] * T
        plan_hist1.append(plan1); plan_hist2.append(plan2)

        # 5) attitude updates & logs
        R1, axis1D, phi1_now = attitude_from_state(x1, prev_R=prev_R1, prev_axisD=prev_axis1D)
        R2, axis2D, phi2_now = attitude_from_state(x2, prev_R=prev_R2, prev_axisD=prev_axis2D)
        prev_axis1D, prev_axis2D = axis1D, axis2D
        prev_R1, prev_R2 = R1, R2
        phi_hist1.append(phi1_now); phi_hist2.append(phi2_now)
        att1_stub, _, _ = plan_attitudes_from_X(x1, prev_axis1D, prev_R1)
        att2_stub, _, _ = plan_attitudes_from_X(x2, prev_axis2D, prev_R2)
        plan_att1.append(att1_stub); plan_att2.append(att2_stub)

        exec_att1.append({"R": R1, "phi": phi1_now})
        exec_att2.append({"R": R2, "phi": phi2_now})
        exec_xyz1.append(_p3_vec(x1)); exec_xyz2.append(_p3_vec(x2))

        # (FOV block omitted for brevity; keep yours if needed)

    # -------------------- pack results --------------------
    return {
        "plan_hist1": plan_hist1, "plan_hist2": plan_hist2,
        "plan_att1":  plan_att1,  "plan_att2":  plan_att2,
        "exec1_xyz":  exec_xyz1,  "exec2_xyz":  exec_xyz2,
        "exec_att1":  exec_att1,  "exec_att2":  exec_att2,
        "phi_hist1":  phi_hist1,  "phi_hist2":  phi_hist2,
        "fov_axis_hist": fov_axis_hist, "fov_seen_mask": fov_seen_mask,
    }



# -------------------- FOV artists & cones --------------------



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


# -------------------- frames & FOV --------------------

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

