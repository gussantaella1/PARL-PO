# game_3dutils.py
# Dedicated 3D utilities and entrypoints. Depends on game_sharedutils.
from __future__ import annotations
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D




import importlib, game_sharedutils, game_3dutils, game_costs, neos_path_game, ukf_estimator, ekf_estimator
importlib.reload(game_sharedutils)
importlib.reload(game_3dutils)
importlib.reload(game_costs)
importlib.reload(neos_path_game)

from ukf_estimator import AgentUKF 
from ekf_estimator import AgentEKF 



importlib.reload(ukf_estimator)
importlib.reload(ekf_estimator)

from game_sharedutils import (
    dims_from_D, step_double_integrator_D, fb,
    pack_trajectory, unpack_trajectory, unpack_tau_flat, split_players_from_z,
    make_bounds, polygon_halfspaces, build_g_tilde, build_h_tilde, build_g_tilde_linear,
    world_to_body_R, fov_axis_from_vel, in_fov, _unit,
    as_numpy_const, hcw_mean_motion, hcw_discrete_mats,
    step_phi,                     # <-- import this, not step_roll
    frame_from_axis, 
    apply_roll_about_axis,
    augment_AB_for_att, augment_bounds_with_att, pad_x0_with_att, frame_from_axis_continuous

)




__all__ = [
    "solve_game_once_3d", "run_rhc_and_collect_frames_3d",
    "plot_planned_trajectories", "plot_planned_trajectories_3d",
    "animate_rollout_3d", "interactive_rollout_3d",
    "make_body_axes_artists_3d", "update_body_axes_artists_3d",
    "draw_fov_cone_3d"
]

__all__ += ["draw_camera_frustum_3d", "project_point_pinhole", "points_in_fov_mask"]

try:
    from neos_path_game import (
        build_mcp_two_player_one_shot,
        solve_with_local_path,
        extract_trajectories,
    )


    HAS_PATH = True
except Exception:
    HAS_PATH = False


def _pack_tau_numpy(X, U):
    # X: (T, nx), U: (T-1, nu) as numpy arrays
    return np.concatenate([X.reshape(-1), U.reshape(-1)])



# ---- PATH/MCP: path-inequality builders (h(k) >= 0) -------------------------
# game_3dutils.py  (replace your build_h_builders with this)
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



# -------------------- 3D solver entrypoint --------------------

def _solve_once_with_path(cfg, Ad_mx, Bd_mx, T, nx, nu, D, x0_1, x0_2,
                          x_lb=None, x_ub=None, u_lb=None, u_ub=None):
    if not HAS_PATH:
        raise RuntimeError("PATH/NEOS not available. Did you create neos_path_game.py and install pyomo?")

    # --- helpers ---
    def _finite(arr, big=1e6):
        a = np.asarray(arr, float)
        mask = ~np.isfinite(a)
        if mask.any():
            a[mask] = np.sign(a[mask]) * big
        return a

    def _ensure_bounds(x_lb, x_ub, u_lb, u_ub):
        # Controls MUST be finite for a well-posed VI; default if missing.
        if (u_lb is None) or (u_ub is None):
            umax = float(cfg.get("u_max", 0.25))   # reasonable accel bound for dt≈0.1–0.2
            u_lb = -umax * np.ones(nu)
            u_ub =  umax * np.ones(nu)
        else:
            u_lb = _finite(u_lb[:nu]); u_ub = _finite(u_ub[:nu])

        # State bounds: if missing, derive a safe box from the arena or fall back to big finite
        if (x_lb is None) or (x_ub is None):
            ar = cfg.get("arena", {})
            if ({"xmin","xmax","ymin","ymax"} <= set(ar.keys())):
                lo = [ar["xmin"], ar["ymin"]] + [-np.inf]*(nx-2)
                hi = [ar["xmax"], ar["ymax"]] + [ np.inf]*(nx-2)
                if D == 3:
                    lo = [ar.get("xmin",-3), ar.get("ymin",-3), ar.get("zmin",-3)] + [-np.inf]*(nx-3)
                    hi = [ar.get("xmax", 3), ar.get("ymax", 3), ar.get("zmax", 3)] + [ np.inf]*(nx-3)
                x_lb = np.array(lo, float); x_ub = np.array(hi, float)
            else:
                x_lb = -1e3*np.ones(nx); x_ub = 1e3*np.ones(nx)
        else:
            x_lb = _finite(x_lb[:nx]); x_ub = _finite(x_ub[:nx])
        return (x_lb, x_ub), (u_lb, u_ub)

    def _pick_cost_kind(cfg):
        setting = str(cfg.get("setting", "lqr")).lower()
        if setting in {"chase_escape_tail", "tail", "chase-escape", "ce"}:
            return "chase_escape_tail"
        return "lqr"

    # --- numeric data ---
    Ad_np = as_numpy_const(Ad_mx); Bd_np = as_numpy_const(Bd_mx)
    (x_bounds, u_bounds) = _ensure_bounds(x_lb, x_ub, u_lb, u_ub)

    # --- path-inequalities (arena+sep+speed) ---
    h_list = build_h_builders(cfg, nx, D)

    # --- build MCP ---
    cost_kind = _pick_cost_kind(cfg)
    m = build_mcp_two_player(
        Ad_np, Bd_np, T, nx, nu,
        x0_1, x0_2,
        Q=np.eye(nx), R1=np.eye(nu), R2=np.eye(nu),   # only used for LQ fallback
        D=D,
        x_bounds=x_bounds, u_bounds=u_bounds,
        h_builders=h_list,
        cost_kind=cost_kind,
        cost_cfg=cfg,
    )

    # --- warm start: roll forward zero-controls to seed x; zeros for u, λ ---
    #   This massively stabilizes PATH on the first solve.
    def _roll(A, B, x0):
        X = np.zeros((T+1, nx)); U = np.zeros((T, nu))
        X[0] = x0
        for k in range(T):
            X[k+1] = A @ X[k] + B @ U[k]
        return X, U

    X1_0, U1_0 = _roll(Ad_np, Bd_np, np.asarray(x0_1, float))
    X2_0, U2_0 = _roll(Ad_np, Bd_np, np.asarray(x0_2, float))

    for k in range(T+1):
        for i in range(nx):
            m.x1[k, i].value = float(np.clip(X1_0[k, i], x_bounds[0][i], x_bounds[1][i]))
            m.x2[k, i].value = float(np.clip(X2_0[k, i], x_bounds[0][i], x_bounds[1][i]))
        if k < T:
            for j in range(nu):
                m.u1[k, j].value = 0.0
                m.u2[k, j].value = 0.0
    for k in range(T+1):
        for i in range(nx):
            m.lam1[k, i].value = 0.0
            m.lam2[k, i].value = 0.0

    # --- solve with PATH ---
    path_exe = (cfg.get("solvers", {}) or {}).get("pathampl")
    print(f"[PATH] cost_kind={cost_kind}, |H|={len(h_list)}, bounds: "
          f"u∈[{u_bounds[0].min():.3g},{u_bounds[1].max():.3g}] "
          f"x-box finite? {np.isfinite(x_bounds[0]).all() and np.isfinite(x_bounds[1]).all()}")
    res = solve_with_local_path(
            m,
            path_exe="/Users/gussantaella/Documents/UTAustin/Research/Code/Research_Repo/path_5/ampl/pathampl",
            tee=True,
        )


    # --- extract and sanity-check ---
    X1, U1, X2, U2 = extract_trajectories(m)

    # quick post-check on path-ineqs
    def _min_h(hfun):
        v = +np.inf
        for k in range(T+1):
            try:
                v = min(v, float(value(hfun(m, k))))
            except Exception:
                pass
        return v

    if h_list:
        mins = [_min_h(hf) for hf in h_list]
        print("[PATH] min h over horizon:", ", ".join(f"{v:.2e}" for v in mins))

    return X1, U1, X2, U2



def solve_game_once_3d(cfg: dict, cost_builder, ipopt_opts: dict | None = None,
                       solver_kind: str = "ipopt"):
    """
    Solve once (open-loop) with optional roll augmentation inside the state.

    solver_kind:
      - "ipopt": original KKT-residual least-squares (default)
      - "path" : solve an MCP via PATH/NEOS (LQ assumptions in the helper)
    """
    N = 2
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx_tr, nu_tr = dims_from_D(D)
    T, dt  = int(cfg["T"]), float(cfg["dt"])

    # --- translational Ad,Bd (MX) ---
    dyn = (cfg.get("dynamics") or "double").lower()
    if dyn == "hcw":
        n = hcw_mean_motion(cfg.get("hcw", {}))
        Ad_tr, Bd_tr = hcw_discrete_mats(n, dt)     # MX
    else:
        Ad_tr, Bd_tr = step_double_integrator_D(D=D, dt=dt)  # MX (already discrete)

    # --- optional attitude augmentation ---
    att_cfg = cfg.get("att", {})
    use_att = bool(att_cfg)
    if use_att:
        Ad_mx, Bd_mx, idx = augment_AB_for_att(Ad_tr, Bd_tr, dt, att_cfg)
        nx, nu = idx["nx"], idx["nu"]
    else:
        Ad_mx, Bd_mx = Ad_tr, Bd_tr
        nx, nu = nx_tr, nu_tr

    # --- bounds (augment if attitude is on) ---
    x_lb, x_ub, u_lb, u_ub = make_bounds(cfg)
    if use_att:
        x_lb, x_ub, u_lb, u_ub = augment_bounds_with_att(x_lb, x_ub, u_lb, u_ub, att_cfg)

    # --- constraints builders (fixed Ad,Bd) ---
    gtil_fun = build_g_tilde_linear(nx, nu, T, N, Ad_mx, Bd_mx)
    htil_fun = build_h_tilde(nx, nu, T, N, x_lb, x_ub, u_lb, u_ub, cfg)

    # --- theta (initial param; pad with φ if needed) ---
    x0_rows = np.asarray(cfg["x0"], float)
    assert x0_rows.shape[0] == N, f"cfg['x0'] must have {N} rows (one per agent)"
    theta_parts = []
    for i in range(N):
        x0_i = pad_x0_with_att(x0_rows[i], att_cfg, D)[:nx] if use_att else x0_rows[i][:nx]
        theta_parts.append(x0_i)
    theta0 = np.hstack(theta_parts)

    # --- PATH branch (optional) ---
    if (solver_kind or "").lower() == "path":
        # solve via MCP (PATH/NEOS) and pack into z so plotting stays intact
        x0_1, x0_2 = theta0[:nx], theta0[nx:2*nx]
        X1, U1, X2, U2 = _solve_once_with_path(
            cfg, Ad_mx, Bd_mx, T, nx, nu, D, x0_1, x0_2,
            x_lb=x_lb, x_ub=x_ub, u_lb=u_lb, u_ub=u_ub
        )

        nprim = T*nx + (T-1)*nu
        tau1 = _pack_tau_numpy(X1, U1)
        tau2 = _pack_tau_numpy(X2, U2)

        # sizes for lam, mu (we provide zeros as placeholders)
        t_sym  = ca.MX.sym('t', N*nprim); th_sym = ca.MX.sym('th', nx*N)
        n_g    = int(gtil_fun(t_sym, th_sym).shape[0])
        n_h    = int(htil_fun(t_sym, th_sym).shape[0])

        zstar = np.r_[tau1, tau2, np.zeros(n_g), np.zeros(n_h)]
        print("[3D/PATH] Solved MCP with PATH/NEOS; packing plan for plotting.")
        return dict(
            z=zstar, residual=np.nan,
            meta=dict(T=T, nx=nx, nu=nu, N=N, D=D, nprim=nprim,
                      n_g=n_g, n_h=n_h)
        )

    # -------------------- IPOPT branch (your original) --------------------
    nprim = T*nx + (T-1)*nu
    taus  = [ca.MX.sym(f"tau{i+1}", nprim) for i in range(N)]
    tau   = ca.vcat(taus)
    theta = ca.MX.sym('theta', nx*N)

    fs   = cost_builder(nx, nu, T, N, cfg)
    gtil = gtil_fun(tau, theta)
    htil = htil_fun(tau, theta)

    lam_t = ca.MX.sym('lam_t', gtil.shape[0])
    mu_t  = ca.MX.sym('mu_t',  htil.shape[0])

    grads = []
    for i in range(N):
        Li = fs[i](tau, theta) - lam_t.T @ gtil - mu_t.T @ htil
        grads.append(ca.gradient(Li, taus[i]))
    FB  = ca.vcat([fb(htil[i], mu_t[i]) for i in range(htil.shape[0])])
    G   = ca.vcat(grads + [gtil, FB])
    z   = ca.vcat(taus + [lam_t, mu_t])
    obj = 0.5 * ca.dot(G, G)

    opts = {
        "ipopt.tol": 1e-4,
        "ipopt.acceptable_tol": 1e-3,
        "ipopt.max_iter": 100,
        "ipopt.print_level": 0,
        "print_time": 0
    }
    if ipopt_opts: opts.update(ipopt_opts)
    solver = ca.nlpsol('solver', 'ipopt', {'x': z, 'p': theta, 'f': obj}, opts)

    # μ ≥ 0
    lbz = -np.inf*np.ones(z.shape[0]); ubz =  np.inf*np.ones(z.shape[0])
    lbz[N*nprim + gtil.shape[0]:] = 0.0

    # solve
    sol = solver(x0=np.zeros(z.shape[0]), lbx=lbz, ubx=ubz, p=theta0)
    zstar = np.array(sol['x']).squeeze()
    residual = float(sol['f'])
    print(f"[3D] Objective residual (0.5||G||^2): {residual:.3e}")
    print(f"[3D] z dim: {z.shape[0]} (taus={N*nprim}, lam={gtil.shape[0]}, mu={htil.shape[0]})")

    return dict(
        z=zstar, residual=residual,
        meta=dict(T=T, nx=nx, nu=nu, N=N, D=D, nprim=nprim,
                  n_g=gtil.shape[0], n_h=htil.shape[0])
    )


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
                                  turn_len: int | None = None, ipopt_opts: dict | None = None,
                                  solver_kind: str = "ipopt"):
    """
    RHC rollout with optional attitude-in-state (roll φ). Boresight at t=0 is v/‖v‖.

    solver_kind:
      - "ipopt": original KKT-residual replans
      - "path" : one big MCP via PATH, persistent model + warm starts
    """
    N = 2
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx_tr, nu_tr = dims_from_D(D)
    T, dt  = int(cfg["T"]), float(cfg["dt"])

    # === ADD: estimation config & UKF import ===
    est_cfg = dict(cfg.get('est', {}))
    do_est = bool(est_cfg.get('enabled', False))


    # --- translational Ad,Bd (MX) ---
    dyn = (cfg.get("dynamics") or "double").lower()
    if dyn == "hcw":
        n = hcw_mean_motion(cfg.get("hcw", {}))
        Ad_tr, Bd_tr = hcw_discrete_mats(n, dt)         # MX
    else:
        Ad_tr, Bd_tr = step_double_integrator_D(D=D, dt=dt)  # MX

    # --- optional attitude augmentation (roll in state) ---
    att_cfg = cfg.get("att", {})
    use_att = bool(att_cfg)
    if use_att:
        Ad_mx, Bd_mx, idx = augment_AB_for_att(Ad_tr, Bd_tr, dt, att_cfg)
        nx, nu = idx["nx"], idx["nu"]
        i_phi  = idx["i_phi"]      # index of φ in x
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

    # --- constraints (with fixed Ad,Bd) ---
    gtil_fun = build_g_tilde_linear(nx, nu, T, N, Ad_mx, Bd_mx)
    x_lb, x_ub, u_lb, u_ub = make_bounds(cfg)
    if use_att:
        x_lb, x_ub, u_lb, u_ub = augment_bounds_with_att(x_lb, x_ub, u_lb, u_ub, att_cfg)
    htil_fun = build_h_tilde(nx, nu, T, N, x_lb, x_ub, u_lb, u_ub, cfg)

    # --- symbols & IPOPT solver (only used in IPOPT branch) ---
    nprim     = T*nx + (T-1)*nu
    taus_syms = [ca.MX.sym(f"tau{i+1}", nprim) for i in range(N)]
    tau_sym   = ca.vcat(taus_syms)
    theta_sym = ca.MX.sym('theta', nx*N)

    fs    = cost_builder(nx, nu, T, N, cfg)
    gtil  = gtil_fun(tau_sym, theta_sym)
    htil  = htil_fun(tau_sym, theta_sym)
    lam_t = ca.MX.sym('lam_t', gtil.shape[0])
    mu_t  = ca.MX.sym('mu_t',  htil.shape[0])

    grads = []
    for i in range(N):
        Li = fs[i](tau_sym, theta_sym) - lam_t.T @ gtil - mu_t.T @ htil
        grads.append(ca.gradient(Li, taus_syms[i]))
    FB   = ca.vcat([fb(htil[i], mu_t[i]) for i in range(htil.shape[0])])
    G    = ca.vcat(grads + [gtil, FB])
    z_sym = ca.vcat(taus_syms + [lam_t, mu_t])
    obj  = 0.5 * ca.dot(G, G) + 1e-10 * ca.dot(z_sym, z_sym)

    opts = {'ipopt.print_level': 0, 'print_time': 0}
    if ipopt_opts: opts.update(ipopt_opts)
    solver = ca.nlpsol('solver', 'ipopt', {'x': z_sym, 'p': theta_sym, 'f': obj}, opts)

    # bounds on decision vector for IPOPT (μ ≥ 0)
    BIG = 1e8
    lbz = -BIG*np.ones(z_sym.shape[0]); ubz = BIG*np.ones(z_sym.shape[0])
    lbz[N*nprim + gtil.shape[0]:] = 0.0

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

    # --- attitude_from_state: keep prev_R and return R for chaining ---
    def attitude_from_state(x, prev_R=None, prev_axisD=None, att_cfg=None, use_att=True, i_phi=None):
        if att_cfg is None:
            att_cfg = {}
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

    # --- plan side: carry R_prev through the horizon ---
    def plan_attitudes_from_X(X, prev_axisD, prev_R):
        att_list = []
        ax_prev  = prev_axisD
        R_prev   = prev_R
        for t in range(T):
            R, ax_prev, phi_t = attitude_from_state(X[t], prev_R=R_prev, prev_axisD=ax_prev)
            att_list.append({"R": R, "phi": phi_t})
            R_prev = R
        return att_list, ax_prev, R_prev

    # --- t=0 attitude seed from initial velocity (so first frame draws correctly) ---
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

    # === ADD: build UKFs and histories ===
    if do_est:
        s_az, s_el = np.deg2rad(est_cfg.get('meas_std_deg', (0.3, 0.3)))
        P0 = np.diag(est_cfg.get('P0_diag', [25,25,25, 1,1,1]))
        Q  = np.diag(est_cfg.get('Q_diag',  [1e-4,1e-4,1e-4, 1e-3,1e-3,1e-3]))
        Rm = np.diag([s_az**2, s_el**2])
        who = (est_cfg.get('who') or 'both').lower()
        every = int(est_cfg.get('every', 1))

        est_hist = {'est12_xyz': [], 'est21_xyz': [], 'meas12_azel': [], 'meas21_azel': []}

        ukf12 = KF_CV(np.r_[x2[:3], np.zeros(3)], P0, Q, Rm, dt) if who in ('both','1->2') else None
        ukf21 = KF_CV(np.r_[x1[:3], np.zeros(3)], P0, Q, Rm, dt) if who in ('both','2->1') else None

        # log initial guesses
        if ukf12 is not None:
            est_hist['est12_xyz'].append(ukf12.x[:3].copy())
            est_hist['meas12_azel'].append(None)
        if ukf21 is not None:
            est_hist['est21_xyz'].append(ukf21.x[:3].copy())
            est_hist['meas21_azel'].append(None)


    # log true t=0 snapshot
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
            _, _, _, ok0 = project_point_pinhole(
                X_w=x_tgt[:3], x_def=x_def, cam_cfg=cam_cfg, R_wb=R_def0
            )
            fov_seen_mask.append(bool(ok0))
        else:
            seen0, _ = in_fov(x_tgt[:D], x_def[:D], a_def3, fov_cfg, D)
            fov_seen_mask.append(bool(seen0))
    else:
        fov_axis_hist.append(None); fov_seen_mask.append(False)

    # -------------------- PATH (one persistent MCP) --------------------
    use_path = (solver_kind or "").lower() == "path"
    path_ctx = None
    if use_path:
        # Wide var boxes; real limits go into h(x) >= 0
        x_var_box = (-1e6, 1e6)
        u_var_box = (-1e3, 1e3)
        h_list    = build_h_builders(cfg, nx, D)

        # Build once, keep across replans
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

    # --- small warm-start utils (PATH) ---
    def _shift_controls_one_step(U_prev, target_len, fill=0.0):
        """
        Shift controls forward by one step for warm-starting PATH.
        Returns an array with shape (target_len, nu).
        - copies U_prev[1:] into the front
        - fills the tail with `fill` (default 0.0)
        """
        U_prev = np.asarray(U_prev, float)
        if U_prev.ndim != 2:
            raise ValueError(f"_shift_controls_one_step expects 2D array, got {U_prev.shape}")
        old_len, nu = U_prev.shape
        U_new = np.full((target_len, nu), float(fill))
        if old_len >= 2:
            take = min(target_len-1, old_len-1)
            if take > 0:
                U_new[:take, :] = U_prev[1:1+take, :]
        # else: nothing to copy; stays filled with `fill`
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
        # X* shape: (T+1, nx); U* shape: (T, nu)
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


    # --- replanners ---
    def replan_ipopt(theta_vec, z_init, prev1D, prev2D, prevR1, prevR2):
        sol  = solver(x0=z_init, lbx=lbz, ubx=ubz, p=theta_vec)
        z_new = np.array(sol['x']).squeeze()
        taus  = split_players_from_z(z_new, N, T, nx, nu)
        X1, U1 = unpack_tau_flat(taus[0], nx, nu, T)
        X2, U2 = unpack_tau_flat(taus[1], nx, nu, T)
        plan1  = [_p3_row(X1[t,:]) for t in range(T)]
        plan2  = [_p3_row(X2[t,:]) for t in range(T)]
        att1, prev1_out, prevR1_out = plan_attitudes_from_X(X1, prev1D, prevR1)
        att2, prev2_out, prevR2_out = plan_attitudes_from_X(X2, prev2D, prevR2)
        return z_new, plan1, plan2, U1, U2, att1, att2, prev1_out, prev2_out, prevR1_out, prevR2_out

    def replan_path(theta_vec, prev1D, prev2D, prevR1, prevR2):
        x0_1 = theta_vec[:nx]
        x0_2 = theta_vec[nx:2*nx]

        m = path_ctx["m"]
        A = path_ctx["A"]; B = path_ctx["B"]

        # --- refresh IC params (mutable) + seed x(0) to match
        for i in range(nx):
            m.x01[i] = float(x0_1[i])
            m.x02[i] = float(x0_2[i])
            m.x1[0, i].value = float(x0_1[i])
            m.x2[0, i].value = float(x0_2[i])

        # --- warm-start controls (length = |Ku| = T)
        Ku_len = len(list(m.Ku))
        if path_ctx["U1_guess"] is None:
            U1g = np.zeros((Ku_len, nu))
            U2g = np.zeros((Ku_len, nu))
        else:
            # make sure the shifter returns (Ku_len, nu)
            U1g = _shift_controls_one_step(path_ctx["U1_guess"], target_len=Ku_len)
            U2g = _shift_controls_one_step(path_ctx["U2_guess"], target_len=Ku_len)

        # --- forward-sim to get X guesses (length = |Kx| = T+1)
        X1g = _forward_sim_x(A, B, x0_1, U1g)  # expect shape (Ku_len+1, nx)
        X2g = _forward_sim_x(A, B, x0_2, U2g)

        # --- seed model values (uses Kx/Ku, not K)
        _seed_model_from_guess(m, X1g, U1g, X2g, U2g)

        # --- solve
        solve_with_local_path(
            m,
            path_exe="/Users/gussantaella/Documents/UTAustin/Research/Code/Research_Repo/path_5/ampl/pathampl",
            tee=True,
        )

        # --- extract + cache warm start for next turn
        X1, U1, X2, U2 = extract_trajectories(m)
        path_ctx["X1_guess"], path_ctx["U1_guess"] = X1, U1
        path_ctx["X2_guess"], path_ctx["U2_guess"] = X2, U2

        # --- pack outputs for viz/exec (use first T states for planning visuals)
        plan1 = [_p3_row(X1[t, :]) for t in range(T)]
        plan2 = [_p3_row(X2[t, :]) for t in range(T)]
        att1, prev1_out, prevR1_out = plan_attitudes_from_X(X1, prev1D, prevR1)
        att2, prev2_out, prevR2_out = plan_attitudes_from_X(X2, prev2D, prevR2)
        z_new = np.zeros(1)  # dummy (PATH branch doesn't use z)
        return z_new, plan1, plan2, U1, U2, att1, att2, prev1_out, prev2_out, prevR1_out, prevR2_out


    # --- first plan (uses current θ and previous boresights) ---
    z_last = np.zeros(z_sym.shape[0]); z_last[N*nprim + gtil.shape[0]:] = 1e-3
    if use_path:
        z_last, plan1, plan2, U1, U2, att1, att2, prev_axis1D, prev_axis2D, prev_R1, prev_R2 = \
            replan_path(theta_curr, prev_axis1D, prev_axis2D, prev_R1, prev_R2)
    else:
        z_last, plan1, plan2, U1, U2, att1, att2, prev_axis1D, prev_axis2D, prev_R1, prev_R2 = \
            replan_ipopt(theta_curr, z_last, prev_axis1D, prev_axis2D, prev_R1, prev_R2)
    step_in_turn = 0

    # -------------------- rollout --------------------
    for k in range(steps):
        # replan each turn
        if k % turn_len == 0 and k > 0:
            if use_path:
                z_last, plan1, plan2, U1, U2, att1, att2, prev_axis1D, prev_axis2D, prev_R1, prev_R2 = \
                    replan_path(theta_curr, prev_axis1D, prev_axis2D, prev_R1, prev_R2)
            else:
                z_last, plan1, plan2, U1, U2, att1, att2, prev_axis1D, prev_axis2D, prev_R1, prev_R2 = \
                    replan_ipopt(theta_curr, z_last, prev_axis1D, prev_axis2D, prev_R1, prev_R2)
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
        R1, axis1D, phi1_now = attitude_from_state(
            x1, prev_R=prev_R1, prev_axisD=prev_axis1D, att_cfg=att_cfg, use_att=use_att, i_phi=i_phi
        )
        R2, axis2D, phi2_now = attitude_from_state(
            x2, prev_R=prev_R2, prev_axisD=prev_axis2D, att_cfg=att_cfg, use_att=use_att, i_phi=i_phi
        )

        prev_axis1D, prev_axis2D = axis1D, axis2D
        prev_R1, prev_R2 = R1, R2

        phi_hist1.append(phi1_now)
        phi_hist2.append(phi2_now)

        # 3) log executed pose + attitude
        exec_att1.append({"R": R1, "phi": phi1_now})
        exec_att2.append({"R": R2, "phi": phi2_now})
        exec_xyz1.append(_p3_vec(x1)); exec_xyz2.append(_p3_vec(x2))

        # 3b) UKF estimation at this executed frame (after R1/R2 exist and exec_* logged)
        if do_est:
            idx  = len(exec_xyz1) - 1
            take = (idx % every) == 0

            def _u_tr(u):
                u = np.asarray(u, float).ravel()
                return u[:3] if u.size >= 3 else np.zeros(3)  # translational accel only


            # agent 1 estimates agent 2

            if ukf12 is not None:
                u2_tr = _u_tr(u2)
                ukf12.predict(dt, u=u2_tr, u_cov=None)
                if take:
                    p_obs = np.asarray(exec_xyz1[-1], float)  # P1 observer (world)
                    p_tgt = np.asarray(exec_xyz2[-1], float)  # P2 target   (world)
                    d = p_tgt - p_obs
                    d /= (np.linalg.norm(d) + 1e-12)
                    b_b = R1 @ d  # R1: world->body (already computed this frame)
                    az = np.arctan2(b_b[1], b_b[0]) + np.random.randn()*s_az
                    el = np.arctan2(b_b[2], np.sqrt(max(b_b[0]**2 + b_b[1]**2, 1e-18))) + np.random.randn()*s_el
                    p_obs1_world = np.asarray(exec_xyz1[-1], float)  # agent 1 position (world)
                    ukf12.update(np.array([az, el]), p_obs=p_obs1_world, R_wb=R1)
                    est_hist['meas12_azel'].append((p_obs.copy(), np.array([az, el])))
                else:
                    est_hist['meas12_azel'].append(None)
                est_hist['est12_xyz'].append(ukf12.x[:3].copy())


            # agent 2 estimates agent 1

            if ukf21 is not None:
                u1_tr = _u_tr(u1)
                ukf21.predict(dt, u=u1_tr)
                if take:
                    p_obs = np.asarray(exec_xyz2[-1], float)  # P2 observer
                    p_tgt = np.asarray(exec_xyz1[-1], float)  # P1 target
                    d = p_tgt - p_obs
                    d /= (np.linalg.norm(d) + 1e-12)
                    b_b = R2 @ d
                    az = np.arctan2(b_b[1], b_b[0]) + np.random.randn()*s_az
                    el = np.arctan2(b_b[2], np.sqrt(max(b_b[0]**2 + b_b[1]**2, 1e-18))) + np.random.randn()*s_el
                    p_obs2_world = np.asarray(exec_xyz2[-1], float)  # agent 2 position (world)
                    ukf21.update(np.array([az, el]), p_obs=p_obs2_world, R_wb=R2)
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
                _, _, _, ok = project_point_pinhole(
                    X_w=x_tgt[:3], x_def=x_def, cam_cfg=cam_cfg, R_wb=R_def
                )
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


# -------------------- plotting --------------------
def plot_planned_trajectories(sol, cfg: dict):
    """2D quick plot; used as fallback when D=2 from 3D meta."""
    z = sol['z']; meta = sol['meta']
    T, nx, nu, N = meta['T'], meta['nx'], meta['nu'], meta['N']
    taus = split_players_from_z(z, N, T, nx, nu)
    X1, U1 = unpack_tau_flat(taus[0], nx, nu, T)
    X2, U2 = unpack_tau_flat(taus[1], nx, nu, T)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal'); ax.grid(True, ls=':')
    ar = cfg.get("arena", None)
    if ar is not None and "xmin" in ar:
        ax.set_xlim(ar["xmin"]-0.5, ar["xmax"]+0.5)
        ax.set_ylim(ar["ymin"]-0.5, ar["ymax"]+0.5)
        ax.add_patch(plt.Rectangle((ar["xmin"], ar["ymin"]),
                                   ar["xmax"]-ar["xmin"], ar["ymax"]-ar["ymin"],
                                   fill=False, lw=1.5, ls='-', alpha=0.6))
    for circ in cfg.get("circles", []):
        ax.add_patch(plt.Circle((circ["cx"], circ["cy"]), circ["r"],
                                 fill=False, lw=1.2, ls='--', alpha=0.6))
    ax.plot(X1[:,0], X1[:,1], '-o', label='P1')
    ax.plot(X2[:,0], X2[:,1], '-o', label='P2')
    ax.scatter([X1[0,0], X2[0,0]], [X1[0,1], X2[0,1]], s=80, marker='*', zorder=5)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(loc='upper right'); ax.set_title('Planned trajectories (horizon)')
    plt.show()

def plot_planned_trajectories_3d(sol: dict, cfg: dict):
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    z = sol['z']; meta = sol['meta']
    T, nx, nu, N = meta['T'], meta['nx'], meta['nu'], meta['N']
    taus = split_players_from_z(z, N, T, nx, nu)
    X1, U1 = unpack_tau_flat(taus[0], nx, nu, T)
    X2, U2 = unpack_tau_flat(taus[1], nx, nu, T)
    if D == 2:
        return plot_planned_trajectories(sol, cfg)
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(True)
    ar = cfg.get("arena", {})
    if ar.get("type","box") == "box":
        ax.set_xlim(ar["xmin"], ar["xmax"])
        ax.set_ylim(ar["ymin"], ar["ymax"])
        ax.set_zlim(ar["zmin"], ar["zmax"])
    elif ar.get("type") == "sphere":
        R = ar["r"]; cx,cy,cz = ar["cx"], ar["cy"], ar["cz"]
        ax.set_xlim(cx-R, cx+R); ax.set_ylim(cy-R, cy+R); ax.set_zlim(cz-R, cz+R)
    elif ar.get("type") == "polyhedron" and "V" in ar:
        V = np.asarray(ar["V"], float)
        ax.set_xlim(V[:,0].min(), V[:,0].max())
        ax.set_ylim(V[:,1].min(), V[:,1].max())
        ax.set_zlim(V[:,2].min(), V[:,2].max())
    for sph in cfg.get("spheres", []):
        u = np.linspace(0, 2*np.pi, 24); v = np.linspace(0, np.pi, 12)
        r = sph["r"]; cx,cy,cz = sph["cx"], sph["cy"], sph["cz"]
        xs = cx + r*np.outer(np.cos(u), np.sin(v))
        ys = cy + r*np.outer(np.sin(u), np.sin(v))
        zs = cz + r*np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, linewidth=0.4, alpha=0.3)
    ax.plot(X1[:,0], X1[:,1], X1[:,2], '-o', label='P1')
    ax.plot(X2[:,0], X2[:,1], X2[:,2], '-o', label='P2')
    ax.scatter([X1[0,0], X2[0,0]], [X1[0,1], X2[0,1]], [X1[0,2], X2[0,2]], s=50, marker='*', zorder=5)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.legend(loc='upper left'); ax.set_title('Planned trajectories (3D)')
    plt.show()

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
