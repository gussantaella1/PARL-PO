# game_3dutils.py
# Dedicated 3D utilities and entrypoints. Depends on game_sharedutils.
from __future__ import annotations
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D


import importlib, game_sharedutils, game_3dutils, game_costs, game_attquat
importlib.reload(game_sharedutils)
importlib.reload(game_3dutils)
importlib.reload(game_costs)
importlib.reload(game_attquat)



from game_sharedutils import (
    dims_from_D, step_double_integrator_D, fb,
    pack_trajectory, unpack_trajectory, unpack_tau_flat, split_players_from_z,
    make_bounds, build_h_tilde, build_g_tilde_linear,
    world_to_body_R, fov_axis_from_vel, in_fov, _unit,
    as_numpy_const, hcw_mean_motion, hcw_discrete_mats,
    apply_roll_about_axis, fb_eps
)


from game_attquat import (
    AttState, q_to_R, R_to_q, step_attitude_quat,
    tau_point_boresight, saturate_norm
)



__all__ = [
    "solve_game_once_3d", "run_rhc_and_collect_frames_3d",
    "plot_planned_trajectories", "plot_planned_trajectories_3d",
    "animate_rollout_3d", "interactive_rollout_3d",
    "make_body_axes_artists_3d", "update_body_axes_artists_3d",
    "draw_fov_cone_3d"
]

__all__ += ["draw_camera_frustum_3d", "project_point_pinhole", "points_in_fov_mask"]


# -------------------- 3D solver entrypoint --------------------

def _unit(v):
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else (v / n)

def _axis_from_vel(xtr, D, align="x", min_speed=1e-3, prev_axis=None):
    v = np.asarray(xtr[D:2*D], float)
    n = float(np.linalg.norm(v))
    if n < min_speed:
        if prev_axis is not None:
            return prev_axis
        return np.array([1,0,0], float) if align=="x" else np.array([0,0,1], float)
    a = v/n
    if D == 2: a = np.array([a[0], a[1], 0.0], float)
    if prev_axis is not None and np.dot(a, prev_axis) < 0.0:
        a = -a  # hysteresis: keep co-directional to avoid 180 flips
    return a

def _world_to_body_continuous(a, world_up, align="x", R_prev=None, a_prev=None, alpha=1.0):
    a = _unit(a)
    # 1) Try transporting the previous frame
    R_tr = _transport_R(R_prev, a_prev, a) if (R_prev is not None and a_prev is not None) else None
    # 2) If no prev frame, build a seed once (project-up is fine just for seeding)
    if R_tr is None:
        y0 = world_up - a*np.dot(world_up, a)
        if np.linalg.norm(y0) < 1e-8:
            # pick a stable basis axis least aligned with a
            e = np.eye(3)[np.argmin(np.abs(a))]
            y0 = e - a*np.dot(e, a)
        R_nom = _R_from_axis_tangent(a, _unit(y0), align)
        return R_nom
    # 3) Optional smoothing toward transported frame
    return _slerp_R(R_prev, R_tr, alpha=alpha)

def attitude_from_state(x, prev_axisD, prev_R):
    R, axisD, extra = transported_R_from_state(
        x=x, D=D, att_cfg=att_cfg,
        prev_axisD=prev_axisD, prev_R=prev_R
    )
    # For legacy callers that expect 'phi' (roll models)
    phi = float(extra.get("phi", 0.0))
    return R, axisD, phi



def _closest_tangent(a, up, y_prev=None):
    a = a/ (np.linalg.norm(a)+1e-12)
    up = up/ (np.linalg.norm(up)+1e-12)
    y = up - a*np.dot(up, a)     # project up onto plane ⟂ a
    if np.linalg.norm(y) < 1e-9:
        # pick any orthogonal
        tmp = np.array([0,1,0]) if abs(a[2])>0.9 else np.array([0,0,1])
        y = tmp - a*np.dot(tmp, a)
    y = y / (np.linalg.norm(y)+1e-12)
    if y_prev is not None and np.dot(y, y_prev) < 0.0:
        y = -y  # flip tangent to stay close to previous
    return y

def _R_from_axis_tangent(a, y, align="x"):
    a = a/ (np.linalg.norm(a)+1e-12)
    y = y/ (np.linalg.norm(y)+1e-12)
    z = np.cross(a, y); z /= (np.linalg.norm(z)+1e-12)
    # rows [x_b;y_b;z_b] in world (like your existing convention)
    if align == "x":
        return np.vstack([a, y, z])
    else:  # align boresight with +z row
        x = np.cross(y, a); x /= (np.linalg.norm(x)+1e-12)
        return np.vstack([x, y, a])
    
def _log_SO3(R):
    # rotation vector (axis*angle). stable for small angles
    tr = np.clip((np.trace(R)-1)/2.0, -1.0, 1.0)
    th = np.arccos(tr)
    if th < 1e-8:
        return np.zeros(3)
    w = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])/(2*np.sin(th))
    return w*th

def _exp_SO3(w):
    th = float(np.linalg.norm(w))
    if th < 1e-8:
        return np.eye(3)
    k = w/th
    K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]], float)
    return np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)

def _slerp_R(R_prev, R_nom, alpha=1.0):
    if R_prev is None or alpha >= 0.9999:
        return R_nom
    Rdel = R_prev.T @ R_nom
    w = _log_SO3(Rdel)
    return R_prev @ _exp_SO3(alpha*w)

# ---- roll-only helpers ----
def _att_roll1d_blocks_linear_mx(dt, Jx):
    import casadi as ca
    I1 = ca.MX(1,1); I1[0,0] = 1.0
    O1 = ca.MX(1,1); O1[0,0] = 0.0
    A  = ca.blockcat([[I1, dt*I1],
                      [O1,    I1]])
    Bin = 1.0/float(Jx)
    B  = ca.blockcat([[O1],
                      [dt*Bin*I1]])
    return A, B  # A:2x2, B:2x1

def _phi_from_statevec(x_like, nx_tr):
    x_like = np.asarray(x_like, float).reshape(-1)
    return float(x_like[nx_tr]) if x_like.size >= nx_tr+1 else 0.0

def _phidot_from_statevec(x_like, nx_tr):
    x_like = np.asarray(x_like, float).reshape(-1)
    return float(x_like[nx_tr+1]) if x_like.size >= nx_tr+2 else 0.0

# -------------------- 3D solver entrypoint (NONLINEAR quat-ready) --------------------
def solve_game_once_3d(cfg: dict, cost_builder, ipopt_opts: dict | None = None):
    """
    Open-loop solve.
      - att.mode == "lin3d" / "lin2d_roll" → small-angle δθ, δω, τ (6 states, 3 torques)
      - att.mode == "roll1d"               → roll φ, φdot, τx (2 states, 1 torque)
      - else                               → translation only
    Inequality bounds (h~) are translation-only by default.
    """
    import numpy as np
    import casadi as ca

    # ---------- sizes / timing ----------
    N = 2
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx_tr, nu_tr = dims_from_D(D)
    T, dt = int(cfg["T"]), float(cfg["dt"])

    # ---------- translation dynamics (MX) ----------
    dyn = (cfg.get("dynamics") or "double").lower()
    if dyn == "hcw":
        n = hcw_mean_motion(cfg.get("hcw", {}))
        Ad_tr, Bd_tr = hcw_discrete_mats(n, dt)
    else:
        Ad_tr, Bd_tr = step_double_integrator_D(D=D, dt=dt)

    # ---------- attitude mode ----------
    att_cfg  = cfg.get("att", {})
    mode     = (att_cfg.get("mode") or "lin3d").lower()

    if mode in ("lin3d", "lin2d_roll"):
        J = att_cfg.get("J", [12.0, 10.0, 8.0])
        A_att, B_att = _att_blocks_linear_mx(mode, dt, J)
        nx,  nu  = nx_tr + 6, nu_tr + 3
        A_mx = _blkdiag_mx(Ad_tr, A_att)
        B_mx = _blkdiag_mx(Bd_tr, B_att)
        att_kind = "lin6"
    elif mode == "roll1d":
        Jx = float(att_cfg.get("Jx", 10.0))
        A_roll, B_roll = _att_roll1d_blocks_linear_mx(dt, Jx)  # 2x2, 2x1
        nx,  nu  = nx_tr + 2, nu_tr + 1
        A_mx = _blkdiag_mx(Ad_tr, A_roll)
        B_mx = _blkdiag_mx(Bd_tr, B_roll)
        att_kind = "roll1d"
    else:
        A_mx, B_mx = Ad_tr, Bd_tr
        nx,  nu    = nx_tr, nu_tr
        att_kind   = "none"

    # ---------- equality constraints g~ (linear builder) ----------
    gtil_fun = build_g_tilde_linear(nx, nu, T, N, A_mx, B_mx)

    # ---------- inequality bounds (translation-only when attitude on) ----------
    x_lb_tr, x_ub_tr, u_lb_tr, u_ub_tr = make_bounds(cfg)
    if use_lin:
        x_lb, x_ub, u_lb, u_ub = _pad_trans_bounds_only_finite(
            x_lb_tr, x_ub_tr, u_lb_tr, u_ub_tr, nx, nu, nx_tr, nu_tr, BIG=1e6
        )
    else:
        # even here, avoid ±inf
        def _clip_inf(v, BIG=1e6): 
            vv = np.asarray(v, float).copy()
            vv[~np.isfinite(vv)] = 0.0
            return np.clip(vv, -BIG, BIG)
        x_lb, x_ub, u_lb, u_ub = _clip_inf(x_lb_tr), _clip_inf(x_ub_tr), _clip_inf(u_lb_tr), _clip_inf(u_ub_tr)

    htil_fun = build_h_tilde(nx, nu, T, N, x_lb, x_ub, u_lb, u_ub, cfg)
    gtil_fun = build_g_tilde_linear(nx, nu, T, N, A_mx, B_mx)


    # ---------- solver (KKT residual) ----------
    nprim     = T*nx + (T-1)*nu
    taus_syms = [ca.MX.sym(f"tau{i+1}", nprim) for i in range(N)]
    tau_sym   = ca.vcat(taus_syms)
    theta_sym = ca.MX.sym('theta', nx*N)

    fs    = cost_builder(nx, nu, T, N, cfg)
    gtil  = gtil_fun(tau_sym, theta_sym)
    htil  = htil_fun(tau_sym, theta_sym)

    lam_t = ca.MX.sym('lam_t', gtil.shape[0])
    mu_t  = ca.MX.sym('mu_t',  htil.shape[0])

    # Gradients per player
    grads = []
    for i in range(N):
        Li = fs[i](tau_sym, theta_sym) - lam_t.T @ gtil  # L_i excludes inequalities
        grads.append(ca.gradient(Li, taus_syms[i]))

    # Smoothed FB map (no NaNs if h, mu are finite)
    FB = ca.vcat([_fb_eps(htil[i], mu_t[i], eps=1e-8) for i in range(htil.shape[0])])

    # Full residual vector: [∂L1/∂τ1; ∂L2/∂τ2; g_tilde; FB]
    G    = ca.vcat(grads + [gtil, FB])
    z_sym = ca.vcat(taus_syms + [lam_t, mu_t])

    # Tiny Tikhonov and multiplier regularization
    obj   = 0.5 * ca.dot(G, G) + 1e-12 * ca.dot(z_sym, z_sym) + 1e-10 * ca.dot(mu_t, mu_t)

    opts = {'ipopt.print_level': 0, 'print_time': 0, 'expand': True}
    if ipopt_opts: opts.update(ipopt_opts)
    solver = ca.nlpsol('solver', 'ipopt', {'x': z_sym, 'p': theta_sym, 'f': obj}, opts)

    # ---------- theta seed (x0 per agent) ----------
    x0_rows = np.asarray(cfg["x0"], float)
    assert x0_rows.shape[0] == N, f"cfg['x0'] must have {N} rows"
    theta_parts = []
    for i in range(N):
        xtr = x0_rows[i][:nx_tr]
        if att_kind == "lin6":
            x0_i = np.hstack([xtr, np.zeros(3), np.zeros(3)])  # δθ, δω
        elif att_kind == "roll1d":
            x0_i = np.hstack([xtr, 0.0, 0.0])                  # φ, φdot
        else:
            x0_i = xtr
        theta_parts.append(x0_i)
    theta0 = np.hstack(theta_parts)

    # ---------- μ ≥ 0 ----------
    lbz = -np.inf*np.ones(z_sym.shape[0]); ubz = np.inf*np.ones(z_sym.shape[0])
    lbz[N*nprim + gtil.shape[0]:] = 0.0

    # ---------- solve ----------
    sol = solver(x0=np.zeros(z_sym.shape[0]), lbx=lbz, ubx=ubz, p=theta0)
    zstar = np.array(sol['x']).squeeze()
    residual = float(sol['f'])
    print(f"[solve] residual 0.5||G||^2 = {residual:.3e}")
    print(f"[solve] dims: nx={nx}, nu={nu}, T={T}, z={int(z_sym.shape[0])}")

    return dict(
        z=zstar, residual=residual,
        meta=dict(T=T, nx=nx, nu=nu, N=N, D=D, nx_tr=nx_tr, nu_tr=nu_tr,
                  nprim=nprim, n_g=int(gtil.shape[0]), n_h=int(htil.shape[0]),
                  att_mode=mode, att_kind=att_kind)
    )


def _att_blocks_linear_mx(mode, dt, J):
    """
    Discrete linear 6-state small-angle attitude:
      δθ_{k+1} = δθ_k + dt δω_k
      δω_{k+1} = δω_k + dt J^{-1} τ_k
    """
    # J can be list/np → DM diag; or already MX/DM 3x3
    if isinstance(J, (list, tuple, np.ndarray)):
        Jm = ca.diag(ca.DM([float(J[0]), float(J[1]), float(J[2])]))
    else:
        Jm = J
    I3 = ca.MX.eye(3); O3 = ca.MX.zeros(3,3)
    A  = ca.blockcat([[I3, dt*I3],
                      [O3,    I3]])
    Bin = ca.mtimes(ca.inv(Jm), I3)
    B  = ca.blockcat([[O3],
                      [dt*Bin]])
    return A, B



def _skew_np(e):
    ex, ey, ez = float(e[0]), float(e[1]), float(e[2])
    return np.array([[   0, -ez,  ey],
                     [  ez,   0, -ex],
                     [ -ey,  ex,   0]], float)

def _eps3_from_statevec(x_like, nx_tr):
    """Return δθ (3,) if present, else zeros(3,)."""
    x_like = np.asarray(x_like, float).reshape(-1)
    return (x_like[nx_tr:nx_tr+3] if x_like.size >= nx_tr+3 else np.zeros(3))


def _apply_small_angle(R0, eps):
    eps = np.asarray(eps, float).reshape(-1)
    if eps.size < 3:
        eps = np.zeros(3)
    return R0 @ (np.eye(3) + _skew_np(eps))


def _blkdiag_mx(A, B):
    """Block-diagonal for MX."""
    ar, ac = A.shape
    br, bc = B.shape
    Z_ab = ca.MX.zeros(ar, bc)
    Z_ba = ca.MX.zeros(br, ac)
    return ca.blockcat([[A, Z_ab],
                        [Z_ba, B]])

def _pad_trans_bounds_only(x_lb_tr, x_ub_tr, u_lb_tr, u_ub_tr, nx, nu, nx_tr, nu_tr):
    """Embed translation bounds; leave attitude unbounded."""
    x_lb = np.full(nx, -np.inf); x_ub = np.full(nx,  np.inf)
    u_lb = np.full(nu, -np.inf); u_ub = np.full(nu,  np.inf)
    x_lb[:nx_tr] = np.asarray(x_lb_tr, float)
    x_ub[:nx_tr] = np.asarray(x_ub_tr, float)
    u_lb[:nu_tr] = np.asarray(u_lb_tr, float)
    u_ub[:nu_tr] = np.asarray(u_ub_tr, float)
    return x_lb, x_ub, u_lb, u_ub

def _pad_trans_bounds_only_finite(x_lb_tr, x_ub_tr, u_lb_tr, u_ub_tr,
                                  nx, nu, nx_tr, nu_tr, BIG=1e6):
    x_lb = -BIG*np.ones(nx); x_ub = BIG*np.ones(nx)
    u_lb = -BIG*np.ones(nu); u_ub = BIG*np.ones(nu)
    x_lb[:nx_tr] = np.asarray(x_lb_tr, float)
    x_ub[:nx_tr] = np.asarray(x_ub_tr, float)
    u_lb[:nu_tr] = np.asarray(u_lb_tr, float)
    u_ub[:nu_tr] = np.asarray(u_ub_tr, float)
    return x_lb, x_ub, u_lb, u_ub



# -------------------- RHC with execution & FOV (3D/2D-aware) --------------------
# -------------------- RHC with execution & FOV (quat-aware) --------------------
def run_rhc_and_collect_frames_3d(cfg: dict, cost_builder, steps: int | None = None,
                                  turn_len: int | None = None, ipopt_opts: dict | None = None):
    """
    Receding-horizon rollout.

    Attitude modes:
      - "lin3d" / "lin2d_roll": small-angle δθ(3), δω(3) with τ(3)  → nx += 6, nu += 3
      - "roll1d":               roll φ, φdot with τx                 → nx += 2, nu += 1
      - anything else:          translation only

    g~ is linear (Ax+Bu over translation + attitude block).
    h~ is translation-only by default (attitude states/inputs left unbounded).
    Rendering: nominal frame from velocity (continuous), then:
      - lin6:   apply small-angle δθ via first-order exp
      - roll1d: apply roll φ about boresight
    """
    import numpy as np
    import casadi as ca

    # ---------- sizes / timing ----------
    N = 2
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx_tr, nu_tr = dims_from_D(D)
    T, dt  = int(cfg["T"]), float(cfg["dt"])

    # ---------- translation dynamics ----------
    dyn = (cfg.get("dynamics") or "double").lower()
    if dyn == "hcw":
        n = hcw_mean_motion(cfg.get("hcw", {}))
        Ad_tr, Bd_tr = hcw_discrete_mats(n, dt)
    else:
        Ad_tr, Bd_tr = step_double_integrator_D(D=D, dt=dt)

    # ---------- attitude blocks ----------
    att_cfg = cfg.get("att", {})
    mode    = (att_cfg.get("mode") or "lin3d").lower()

    if mode in ("lin3d", "lin2d_roll"):
        J_diag  = att_cfg.get("J", [12.0, 10.0, 8.0])
        A_att, B_att = _att_blocks_linear_mx(mode, dt, J_diag)
        nx, nu = nx_tr + 6, nu_tr + 3
        A_mx = _blkdiag_mx(Ad_tr, A_att)
        B_mx = _blkdiag_mx(Bd_tr, B_att)
        att_kind = "lin6"
    elif mode == "roll1d":
        Jx = float(att_cfg.get("Jx", 10.0))
        A_roll, B_roll = _att_roll1d_blocks_linear_mx(dt, Jx)  # 2x2, 2x1
        nx, nu = nx_tr + 2, nu_tr + 1
        A_mx = _blkdiag_mx(Ad_tr, A_roll)
        B_mx = _blkdiag_mx(Bd_tr, B_roll)
        att_kind = "roll1d"
    else:
        A_mx, B_mx = Ad_tr, Bd_tr
        nx, nu = nx_tr, nu_tr
        att_kind = "none"

    # ---------- rollout schedule ----------
    sim_time = cfg.get("sim_time", cfg.get("max_time", cfg.get("duration", None)))
    if steps is None:
        steps = max(1, int(np.ceil(float(sim_time)/dt))) if sim_time is not None else int(cfg.get("steps", 60))
    if turn_len is None:
        turn_len = int(cfg.get("turn_len", 3)) if "turn_seconds" not in cfg else \
                   max(1, int(round(float(cfg["turn_seconds"]) / float(dt))))

    # ---------- inequality bounds (translation-only when attitude present) ----------
    x_lb_tr, x_ub_tr, u_lb_tr, u_ub_tr = make_bounds(cfg)
    if att_kind in ("lin6", "roll1d"):
        x_lb, x_ub, u_lb, u_ub = _pad_trans_bounds_only_finite(
            x_lb_tr, x_ub_tr, u_lb_tr, u_ub_tr, nx, nu, nx_tr, nu_tr
        )
    else:
        x_lb, x_ub, u_lb, u_ub = x_lb_tr, x_ub_tr, u_lb_tr, u_ub_tr

    htil_fun = build_h_tilde(nx, nu, T, N, x_lb, x_ub, u_lb, u_ub, cfg)
    gtil_fun = build_g_tilde_linear(nx, nu, T, N, A_mx, B_mx)

    # ---------- solver (KKT residual) ----------
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
    FB   = ca.vcat([fb_eps(htil[i], mu_t[i]) for i in range(htil.shape[0])])
    G    = ca.vcat(grads + [gtil, FB])
    z_sym = ca.vcat(taus_syms + [lam_t, mu_t])
    obj   = 0.5 * ca.dot(G, G) + 1e-10 * ca.dot(z_sym, z_sym)  # tiny Tikhonov

    opts = {'ipopt.print_level': 0, 'print_time': 0, 'expand': True}
    if ipopt_opts: opts.update(ipopt_opts)
    solver = ca.nlpsol('solver', 'ipopt', {'x': z_sym, 'p': theta_sym, 'f': obj}, opts)

    # ---------- μ ≥ 0 ----------
    BIG = 1e8
    lbz = -BIG*np.ones(z_sym.shape[0]); ubz = BIG*np.ones(z_sym.shape[0])
    lbz[N*nprim + gtil.shape[0]:] = 0.0

    # ---------- initial states ----------
    x0_rows = np.asarray(cfg["x0"], float)
    x1_tr = x0_rows[0][:nx_tr].copy()
    x2_tr = x0_rows[1][:nx_tr].copy()
    if att_kind == "lin6":
        x1 = np.hstack([x1_tr, np.zeros(3), np.zeros(3)])  # δθ, δω
        x2 = np.hstack([x2_tr, np.zeros(3), np.zeros(3)])
    elif att_kind == "roll1d":
        x1 = np.hstack([x1_tr, 0.0, 0.0])                  # φ, φdot
        x2 = np.hstack([x2_tr, 0.0, 0.0])
    else:
        x1, x2 = x1_tr, x2_tr
    theta_curr = np.r_[x1, x2]

    # ---------- numeric stepper ----------
    Ad_np = as_numpy_const(A_mx); Bd_np = as_numpy_const(B_mx)
    def _step_plant(x, u): return Ad_np @ np.asarray(x, float) + Bd_np @ np.asarray(u, float)

    # ---------- logs ----------
    plan_hist1, plan_hist2 = [], []
    plan_att1,  plan_att2  = [], []
    exec_xyz1,  exec_xyz2  = [], []
    exec_att1,  exec_att2  = [], []
    phi_hist1,  phi_hist2  = [], []
    fov_axis_hist, fov_seen_mask = [], []

    # ---------- rendering / continuity ----------
    align    = att_cfg.get("align","x")
    world_up = np.asarray(att_cfg.get("up",[0,0,1]), float)
    vmin     = float(att_cfg.get("min_speed_for_axis", 1e-3))
    alpha    = float(att_cfg.get("boresight_lpf_alpha", 1.0))

    R0_prev = [None, None]  # carry nominal frames per agent

    def _R0_from_xtr_cont(xtr, agent_idx):
        v = np.asarray(xtr[D:2*D], float)
        if float(np.linalg.norm(v)) < vmin and (R0_prev[agent_idx] is not None):
            return R0_prev[agent_idx]
        prev_b = None
        if R0_prev[agent_idx] is not None:
            prev_b = R0_prev[agent_idx][0] if align == "x" else R0_prev[agent_idx][2]
        a = _axis_from_vel(xtr, D, align=align, min_speed=vmin, prev_axis=prev_b)
        R0 = _world_to_body_continuous(a, world_up, align, R_prev=R0_prev[agent_idx], alpha=alpha)
        R0_prev[agent_idx] = R0
        return R0

    def _render_R_from_state(x, agent_idx):
        xtr = x[:nx_tr]
        R0  = _R0_from_xtr_cont(xtr, agent_idx)  # boresight from velocity
        if att_kind == "lin6":
            eps = _eps3_from_statevec(x, nx_tr)
            R   = _apply_small_angle(R0, eps); phi = float(eps[0])
            return R, phi
        elif att_kind == "roll1d":
            phi = _phi_from_statevec(x, nx_tr)
            R   = apply_roll_about_axis(R0, phi, align=align)
            return R, float(phi)
        else:
            return R0, 0.0

    def _p3_from_state(x):
        return (float(x[0]), float(x[1]), float(x[2] if D==3 else 0.0))

    # ---------- t=0 snapshot ----------
    R1_0, phi1_0 = _render_R_from_state(x1, 0)
    R2_0, phi2_0 = _render_R_from_state(x2, 1)
    exec_xyz1.append(_p3_from_state(x1)); exec_xyz2.append(_p3_from_state(x2))
    exec_att1.append({"R": R1_0, "phi": phi1_0}); phi_hist1.append(phi1_0)
    exec_att2.append({"R": R2_0, "phi": phi2_0}); phi_hist2.append(phi2_0)

    # ---------- FOV at t=0 ----------
    fov_cfg     = cfg.get("fov", {"enabled": False})
    fov_enabled = bool(fov_cfg.get("enabled", False))
    fov_agent   = int(fov_cfg.get("agent", 2))
    if fov_enabled:
        R_def0 = R2_0 if fov_agent == 2 else R1_0
        x_def  = x2    if fov_agent == 2 else x1
        x_tgt  = x1    if fov_agent == 2 else x2
        a_def3 = R_def0[0] if align == 'x' else R_def0[2]
        fov_axis_hist.append(a_def3)
        cam_cfg = cfg.get("camera", None)
        if fov_cfg.get("type","cone") == "pinhole" and (cam_cfg is not None):
            Xw = np.array([x_tgt[0], x_tgt[1], (x_tgt[2] if D==3 else 0.0)], float)
            _, _, _, ok0 = project_point_pinhole(X_w=Xw, x_def=x_def, cam_cfg=cam_cfg, R_wb=R_def0)
            fov_seen_mask.append(bool(ok0))
        else:
            seen0, _ = in_fov((x_tgt[:D]), (x_def[:D]), a_def3, fov_cfg, D)
            fov_seen_mask.append(bool(seen0))
    else:
        fov_axis_hist.append(None); fov_seen_mask.append(False)

    # ---------- planning helpers ----------
    z_last = np.zeros(z_sym.shape[0]); z_last[N*nprim + gtil.shape[0]:] = 1e-3  # μ init > 0

    def _plan_attitudes_from_X(X, agent_idx):
        out = []
        R0p = R0_prev[agent_idx]
        cols = X.shape[1]
        for t in range(T):
            xtr_t = X[t, :nx_tr]
            v = np.asarray(xtr_t[D:2*D], float)

            if float(np.linalg.norm(v)) < vmin and (R0p is not None):
                R0_t = R0p
            else:
                prev_b = None
                if R0p is not None:
                    prev_b = R0p[0] if align == "x" else R0p[2]
                a_t  = _axis_from_vel(xtr_t, D, align=align, min_speed=vmin, prev_axis=prev_b)
                R0_t = _world_to_body_continuous(a_t, world_up, align, R_prev=R0p, alpha=alpha)

            if att_kind == "lin6":
                eps_t = (X[t, nx_tr:nx_tr+3] if cols >= nx_tr+3 else np.zeros(3))
                R_t   = _apply_small_angle(R0_t, eps_t); phi_t = float(eps_t[0])
            elif att_kind == "roll1d":
                phi_t = (float(X[t, nx_tr]) if cols >= nx_tr+1 else 0.0)
                R_t   = apply_roll_about_axis(R0_t, phi_t, align=align)
            else:
                R_t, phi_t = R0_t, 0.0

            out.append({"R": R_t, "phi": phi_t})
            R0p = R0_t
        return out, R0p

    def replan(theta_vec, z_init):
        sol  = solver(x0=z_init, lbx=lbz, ubx=ubz, p=theta_vec)
        z_new = np.array(sol['x']).squeeze()
        taus  = split_players_from_z(z_new, N, T, nx, nu)
        X1, U1 = unpack_tau_flat(taus[0], nx, nu, T)
        X2, U2 = unpack_tau_flat(taus[1], nx, nu, T)
        plan1  = [(float(X1[t,0]), float(X1[t,1]), float(X1[t,2] if D==3 else 0.0)) for t in range(T)]
        plan2  = [(float(X2[t,0]), float(X2[t,1]), float(X2[t,2] if D==3 else 0.0)) for t in range(T)]
        att1_plan, R0_prev[0] = _plan_attitudes_from_X(X1, 0)
        att2_plan, R0_prev[1] = _plan_attitudes_from_X(X2, 1)
        return z_new, plan1, plan2, U1, U2, att1_plan, att2_plan

    # ---------- initial plan ----------
    z_last, plan1, plan2, U1, U2, att1_plan, att2_plan = replan(theta_curr, z_last)
    step_in_turn = 0

    # ---------- rollout ----------
    for k in range(steps):
        if k % turn_len == 0 and k > 0:
            z_last, plan1, plan2, U1, U2, att1_plan, att2_plan = replan(theta_curr, z_last)
            step_in_turn = 0

        plan_hist1.append(plan1); plan_hist2.append(plan2)
        plan_att1.append(att1_plan); plan_att2.append(att2_plan)

        u1 = U1[min(step_in_turn, len(U1)-1)]
        u2 = U2[min(step_in_turn, len(U2)-1)]
        step_in_turn += 1

        x1 = _step_plant(x1, u1)
        x2 = _step_plant(x2, u2)
        theta_curr = np.r_[x1, x2]

        R1, ph1 = _render_R_from_state(x1, 0)
        R2, ph2 = _render_R_from_state(x2, 1)
        exec_att1.append({"R": R1, "phi": ph1}); phi_hist1.append(ph1)
        exec_att2.append({"R": R2, "phi": ph2}); phi_hist2.append(ph2)
        exec_xyz1.append(_p3_from_state(x1)); exec_xyz2.append(_p3_from_state(x2))

        if fov_enabled:
            R_def = (exec_att2[-1]["R"] if fov_agent == 2 else exec_att1[-1]["R"])
            x_def = (x2 if fov_agent == 2 else x1)
            x_tgt = (x1 if fov_agent == 2 else x2)
            a_def3 = (R_def[0] if align == 'x' else R_def[2])
            fov_axis_hist.append(a_def3)
            cam_cfg = cfg.get("camera", None)
            if fov_cfg.get("type","cone") == "pinhole" and (cam_cfg is not None):
                Xw = np.array([x_tgt[0], x_tgt[1], (x_tgt[2] if D==3 else 0.0)], float)
                _, _, _, ok = project_point_pinhole(X_w=Xw, x_def=x_def, cam_cfg=cam_cfg, R_wb=R_def)
                fov_seen_mask.append(bool(ok))
            else:
                seen, _ = in_fov((x_tgt[:D]), (x_def[:D]), a_def3, fov_cfg, D)
                fov_seen_mask.append(bool(seen))
        else:
            fov_axis_hist.append(None); fov_seen_mask.append(False)

    return {
        'plan_hist1': plan_hist1,
        'plan_hist2': plan_hist2,
        'plan_att1' : plan_att1,
        'plan_att2' : plan_att2,
        'exec1_xyz' : exec_xyz1,
        'exec2_xyz' : exec_xyz2,
        'exec_att1' : exec_att1,
        'exec_att2' : exec_att2,
        'phi_hist1' : phi_hist1,
        'phi_hist2' : phi_hist2,
        'fov_axis_hist': fov_axis_hist,
        'fov_seen_mask': fov_seen_mask,
    }


# -------------------- plotting --------------------

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


def _transport_R(prev_R, a_prev, a_new):
    """Minimal rotation that maps a_prev → a_new, then carry y/z with it."""
    if prev_R is None:
        return None  # caller will seed with a standard frame
    a_prev = _unit(a_prev); a_new = _unit(a_new)
    c = float(np.clip(np.dot(a_prev, a_new), -1.0, 1.0))
    v = np.cross(a_prev, a_new); s = float(np.linalg.norm(v))
    if s < 1e-8:
        # almost identical or 180°: choose rotation about previous y to break tie
        if c > 0.0:  # tiny motion
            Rdel = np.eye(3)
        else:       # ~180°, rotate π about prev y (any axis ⟂ a_prev is valid)
            y_axis = prev_R[1] - a_prev*np.dot(prev_R[1], a_prev)
            y_axis = _unit(y_axis) if np.linalg.norm(y_axis) > 1e-9 else _unit(np.array([1,0,0]) - a_prev*a_prev[0])
            K = _skew_np(y_axis)
            Rdel = np.eye(3) + 2*K@K   # Rodrigues with θ=π
    else:
        k = v / s
        K = _skew_np(k)
        Rdel = np.eye(3) + K*s + K@K*((1-c))  # Rodrigues
    return Rdel @ prev_R


def project_point_pinhole(X_w, x_def, cam_cfg, axis=None, R_wb=None):
    """
    Project a world point into pixel coordinates.
    Returns (u, v, depth, visible_bool).
    depth = Z_cam if align='z', or X_cam if align='x'.
    """
    p_cam = np.asarray(x_def[:3], float)
    align = cam_cfg.get("align", "x")
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


# -------------------- animation & interactive --------------------
# -------------------- animation (triads + triad legend) --------------------
def animate_rollout_3d(frames_dict, save_path="traj_3D.gif", fps=20, cfg=None,
                       show_fov=True, show_axes=True):
    import shutil
    from matplotlib import animation
    from matplotlib.animation import FFMpegWriter, PillowWriter

    if cfg is None:
        raise ValueError("cfg must be provided")
    
    R_prev_anim = [None, None]


    fov_cfg  = cfg.get("fov", {})
    att_cfg  = cfg.get("att", {})
    viz_cfg  = cfg.get("viz", {})
    agent_id = int(fov_cfg.get("agent", 2))

    triad_colors = tuple(viz_cfg.get('triad_colors',
                                     ('tab:red','tab:green','tab:blue')))
    triad_labels = tuple(viz_cfg.get('triad_labels',
                                     ('x_b (boresight)', 'y_b', 'z_b')))
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
        
    # Seed previous-R cache from stored exec attitudes if present
    ea1 = frames_dict.get('exec_att1', [])
    ea2 = frames_dict.get('exec_att2', [])
    R_prev_anim[0] = np.asarray(ea1[0]['R'], float) if ea1 and 'R' in ea1[0] else None
    R_prev_anim[1] = np.asarray(ea2[0]['R'], float) if ea2 and 'R' in ea2[0] else None


    fig = plt.figure(figsize=(7,6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.grid(True)

    # Arena limits
    ar = cfg.get("arena", {})
    if ar.get("type") == "box":
        ax.set_xlim(ar["xmin"], ar["xmax"])
        ax.set_ylim(ar["ymin"], ar["ymax"])
        ax.set_zlim(ar.get("zmin", -0.5), ar.get("zmax", 0.5))
    elif ar.get("type") == "sphere":
        cx, cy, cz, R = ar["cx"], ar["cy"], ar["cz"], ar["r"]
        ax.set_xlim(cx-R, cx+R); ax.set_ylim(cy-R, cy+R); ax.set_zlim(cz-R, cz+R)

    # Paths/points + main legend
    plan1_ln, = ax.plot([], [], [], '--', lw=1, alpha=0.6, label='Plan P1')
    plan2_ln, = ax.plot([], [], [], '--', lw=1, alpha=0.6, label='Plan P2')
    exe1_ln,  = ax.plot([], [], [], '-',  lw=2, label='Exec P1')
    exe2_ln,  = ax.plot([], [], [], '-',  lw=2, label='Exec P2')
    dot1,     = ax.plot([], [], [], 'o',  ms=6)
    dot2,     = ax.plot([], [], [], 'o',  ms=6)
    leg_main  = ax.legend(loc='upper left')

    # Triad legend (x/y/z color mapping)
    leg_axes = None
    if show_axes:
        leg_axes = add_triad_legend(ax,
                                    colors=triad_colors,
                                    labels=triad_labels,
                                    loc=triad_leg_loc,
                                    ncol=triad_leg_ncol,
                                    title=triad_leg_title,
                                    keep_legend=leg_main)

    # Triads for BOTH agents (same colors for x/y/z)
    att1_lines = make_body_axes_artists_3d(ax, colors=triad_colors) if show_axes else None
    att2_lines = make_body_axes_artists_3d(ax, colors=triad_colors) if show_axes else None

    fov_art = {'coll': None, 'rim': None}

    # ---------- helpers ----------
    def _pos3(p_like):
        p = np.asarray(p_like, float).ravel()
        return p[:3] if p.size >= 3 else np.array([p[0], p[1], 0.0], float)

    def _axis3(a_like):
        a = np.asarray(a_like, float).ravel()
        if a.size < 3: a = np.array([a[0], a[1], 0.0], float)
        return _unit(a)

    def _def_pos(f):
        return exec2[f] if agent_id == 2 else exec1[f]

    def _def_axis(f):
        if axis_hist and f < len(axis_hist):
            return _axis3(axis_hist[f])
        if f > 0:
            p  = np.array(_pos3(_def_pos(f)))
            pp = np.array(_pos3(_def_pos(f-1)))
            dv = p - pp; n = np.linalg.norm(dv)
            return _axis3(dv/(n+1e-12)) if n > 1e-9 else np.array([1,0,0])
        return np.array([1,0,0])
    
    fov_art = {'coll': None, 'rim': None, 'edges': []}



    def _clear_fov():
        for k in ('coll','rim'):
            art = fov_art.get(k)
            if art is not None:
                try: art.remove()
                except Exception: pass
                fov_art[k] = None
        # remove edge lines
        for ln in fov_art.get('edges', []):
            try: ln.remove()
            except Exception: pass
        fov_art['edges'] = []


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
            fov_art['coll'], fov_art['rim'], fov_art['edges'] = coll, None, edges
        else:
            # derive cone axis from the executed rotation (consistent with viz)
            align = att_cfg.get('align','x')
            axis  = R_wb[0] if align == 'x' else R_wb[2]
            coll, rim = draw_fov_cone_3d(
                ax, x_def, axis, fov_cfg,
                color=col, alpha=fov_cfg.get("alpha",0.15),
                align=align
            )
            fov_art['coll'], fov_art['rim'] = coll, rim




    def _R_exec(agent, idx):
        key = 'exec_att1' if agent == 1 else 'exec_att2'
        L = frames_dict.get(key, [])
        # If we have any stored attitudes, clamp to last and return it.
        if L:
            j = min(idx, len(L) - 1)
            R = L[j].get('R', None)
            if R is not None:
                return np.asarray(R, float)
        # As a final fallback (should be rare), reuse previous animation R
        if R_prev_anim[agent-1] is not None:
            return R_prev_anim[agent-1]
        # Absolute last resort: identity aligned with +align axis
        align = cfg.get('att', {}).get('align', 'x')
        return np.vstack([
            np.array([1,0,0]) if align=='x' else np.array([1,0,0]),
            np.array([0,1,0]),
            np.array([0,0,1]) if align=='x' else np.array([0,0,1]),
        ])




    # ---------- animation callbacks ----------
    def init():
        for ln in (plan1_ln, plan2_ln, exe1_ln, exe2_ln, dot1, dot2):
            ln.set_data([], []); ln.set_3d_properties([])
        for L in (att1_lines, att2_lines):
            if L:
                for ln in (L['bx'], L['by'], L['bz']):
                    ln.set_data([], []); ln.set_3d_properties([])
        _clear_fov()
        artists = [plan1_ln, plan2_ln, exe1_ln, exe2_ln, dot1, dot2]
        if leg_main: artists.append(leg_main)
        if leg_axes: artists.append(leg_axes)
        return tuple(artists)

    def update(f):
        # planned
        if plan_hist1 and f < len(plan_hist1) and plan_hist1[f]:
            xs, ys, zs = zip(*plan_hist1[f]); plan1_ln.set_data(xs, ys); plan1_ln.set_3d_properties(zs)
        if plan_hist2 and f < len(plan_hist2) and plan_hist2[f]:
            xs, ys, zs = zip(*plan_hist2[f]); plan2_ln.set_data(xs, ys); plan2_ln.set_3d_properties(zs)

        # executed
        xs1 = [p[0] for p in exec1[:f+1]]; ys1 = [p[1] for p in exec1[:f+1]]; zs1 = [p[2] for p in exec1[:f+1]]
        xs2 = [p[0] for p in exec2[:f+1]]; ys2 = [p[1] for p in exec2[:f+1]]; zs2 = [p[2] for p in exec2[:f+1]]
        exe1_ln.set_data(xs1, ys1); exe1_ln.set_3d_properties(zs1)
        exe2_ln.set_data(xs2, ys2); exe2_ln.set_3d_properties(zs2)

        # points
        x1,y1,z1 = exec1[min(f, len(exec1)-1)]
        x2,y2,z2 = exec2[min(f, len(exec2)-1)]
        dot1.set_data([x1], [y1]); dot1.set_3d_properties([z1])
        dot2.set_data([x2], [y2]); dot2.set_3d_properties([z2])

        # triads for BOTH agents
        if show_axes:
            if att1_lines:
                R1 = _R_exec(1, f)
                update_body_axes_artists_3d(att1_lines, np.array([x1,y1,z1]), R1, L=L_tri)
            if att2_lines:
                R2 = _R_exec(2, f)
                update_body_axes_artists_3d(att2_lines, np.array([x2,y2,z2]), R2, L=L_tri)

        # FOV (selected agent)
        p_def = _pos3(_def_pos(f))
        R_def = _R_exec(agent_id, f)  # 1 or 2
        _clear_fov(); _set_fov(p_def, R_def, f)

        return plan1_ln, plan2_ln, exe1_ln, exe2_ln, dot1, dot2

    # Render
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

# -------------------- interactive (triads + triad legend) --------------------
def interactive_rollout_3d(frames_dict, cfg, title="Interactive 3D rollout",
                           show_fov=True, show_axes=True):
    import ipywidgets as W
    from IPython.display import display

    R_prev_ui = [None, None]


    fov_cfg  = cfg.get("fov", {})
    att_cfg  = cfg.get("att", {})
    viz_cfg  = cfg.get("viz", {})
    agent_id = int(fov_cfg.get("agent", 2))

    triad_colors = tuple(viz_cfg.get('triad_colors',
                                     ('tab:red','tab:green','tab:blue')))
    triad_labels = tuple(viz_cfg.get('triad_labels',
                                     ('x_b (boresight)', 'y_b', 'z_b')))
    triad_leg_loc = viz_cfg.get('triad_leg_loc', 'lower left')
    triad_leg_ncol = int(viz_cfg.get('triad_leg_ncol', 3))
    triad_leg_title = viz_cfg.get('triad_leg_title', 'Body axes')
    L_tri = tuple(cfg.get("viz", {}).get("triad_len", (0.35, 0.35, 0.55)))


    plan_hist1 = frames_dict.get('plan_hist1_3d', frames_dict.get('plan_hist1', []))
    plan_hist2 = frames_dict.get('plan_hist2_3d', frames_dict.get('plan_hist2', []))
    exec1      = frames_dict.get('exec1_xyz',    frames_dict.get('exec1_xy', []))
    exec2      = frames_dict.get('exec2_xyz',    frames_dict.get('exec2_xy', []))
    axis_hist  = frames_dict.get('fov_axis_hist', [])
    seen_mask  = frames_dict.get('fov_seen_mask', [])

    def _as3_hist(seq):
        if not seq: return []
        if seq and seq[0] and len(seq[0][0]) == 2:
            return [[(x,y,0.0) for (x,y) in fr] for fr in seq]
        return seq
    def _as3_exec(seq):
        if not seq: return []
        if seq and len(seq[0]) == 2:
            return [(x,y,0.0) for (x,y) in seq]
        return seq

    plan_hist1 = _as3_hist(plan_hist1); plan_hist2 = _as3_hist(plan_hist2)
    exec1      = _as3_exec(exec1);      exec2      = _as3_exec(exec2)
    n_frames   = max(2, min(len(exec1), len(exec2)))

    fig = plt.figure(figsize=(7,6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.grid(True)
    ax.set_box_aspect((1,1,1))

    # Bounds
    ar = cfg.get("arena", {})
    if ar.get("type") == "box" or ({"xmin","xmax","ymin","ymax"} <= set(ar.keys())):
        xmin, xmax = ar.get("xmin",-3), ar.get("xmax",3)
        ymin, ymax = ar.get("ymin",-3), ar.get("ymax",3)
        if "zmin" in ar and "zmax" in ar:
            zmin, zmax = ar["zmin"], ar["zmax"]
        else:
            zs = [p[2] for p in (exec1+exec2)] or [0.0]
            zmin, zmax = min(zs)-0.5, max(zs)+0.5
        ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax); ax.set_zlim(zmin,zmax)
    elif ar.get("type") == "sphere" or ({"cx","cy","cz","r"} <= set(ar.keys())):
        cx, cy, cz = ar.get("cx",0.0), ar.get("cy",0.0), ar.get("cz",0.0)
        R = ar.get("r", 3.0)
        ax.set_xlim(cx-R, cx+R); ax.set_ylim(cy-R, cy+R); ax.set_zlim(cz-R, cz+R)
    else:
        pts = np.array(exec1+exec2, float)
        if pts.size > 0:
            mn = pts.min(0); mx = pts.max(0)
            span = np.maximum(mx - mn, 1e-3); pad = 0.1*span
            lo = mn - pad; hi = mx + pad
            ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1]); ax.set_zlim(lo[2],hi[2])

    # Lines + main legend
    plan1_ln, = ax.plot([], [], [], '--', lw=1, alpha=0.6, label='Plan P1',color = 'blue')
    plan2_ln, = ax.plot([], [], [], '--', lw=1, alpha=0.6, label='Plan P2',color = 'orange')
    exe1_ln,  = ax.plot([], [], [], '-',  lw=2, label='Exec P1',color='green')
    exe2_ln,  = ax.plot([], [], [], '-',  lw=2, label='Exec P2',color = 'red')
    dot1,     = ax.plot([], [], [], 'o',  ms=6,color='cyan', mec='k', mew=0.6)
    dot2,     = ax.plot([], [], [], 'o',  ms=6,color='orange', mec='k', mew=0.6)
    leg_main  = ax.legend(loc='upper left')

    # Triad legend
    if show_axes:
        add_triad_legend(ax,
                         colors=triad_colors,
                         labels=triad_labels,
                         loc=triad_leg_loc,
                         ncol=triad_leg_ncol,
                         title=triad_leg_title,
                         keep_legend=leg_main)

    # Triads (same colors for both agents)
    att1_lines = make_body_axes_artists_3d(ax, colors=triad_colors) if show_axes else None
    att2_lines = make_body_axes_artists_3d(ax, colors=triad_colors) if show_axes else None

    fov_art = {'coll': None, 'rim': None, 'edges': []}

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


    # helpers
    def _pos3(p_like):
        p = np.asarray(p_like, float).ravel()
        return p[:3] if p.size >= 3 else np.array([p[0], p[1], 0.0], float)
    def _axis3(a_like):
        a = np.asarray(a_like, float).ravel()
        if a.size < 3: a = np.array([a[0], a[1], 0.0], float)
        return a/(np.linalg.norm(a)+1e-12)
    def _def_pos(f):
        return exec2[f] if agent_id == 2 else exec1[f]
    def _def_axis(f):
        if axis_hist and f < len(axis_hist):
            return _axis3(axis_hist[f])
        if f > 0:
            p  = np.array(_pos3(_def_pos(f)))
            pp = np.array(_pos3(_def_pos(f-1)))
            dv = p - pp
            return _axis3(dv) if np.linalg.norm(dv) > 1e-9 else np.array([1,0,0])
        return np.array([1,0,0])

    def _R_exec(agent, idx):
        key = 'exec_att1' if agent == 1 else 'exec_att2'
        L = frames_dict.get(key, [])
        if L and idx < len(L) and 'R' in L[idx]:
            return np.asarray(L[idx]['R'], float)
        pos_seq = exec1 if agent == 1 else exec2
        if idx > 0:
            p  = np.array(_pos3(pos_seq[idx])); pp = np.array(_pos3(pos_seq[idx-1]))
            dv = p - pp
            axis = dv/(np.linalg.norm(dv)+1e-12) if np.linalg.norm(dv) > 1e-9 else np.array([1,0,0], float)
        else:
            axis = np.array([1,0,0], float)
        R = _world_to_body_continuous(
                axis,
                att_cfg.get('up',[0,0,1]),
                att_cfg.get('align','x'),
                R_prev=R_prev_ui[agent-1],
                alpha=float(att_cfg.get('boresight_lpf_alpha', 1.0))
            )
        R_prev_ui[agent-1] = R
        return R


    def _draw_fov(f, p, R_wb):
        _clear_fov()
        if not (t_fov.value and fov_cfg.get("enabled", False)):
            return
        col = fov_cfg.get("color","C1")
        if seen_mask and f < len(seen_mask) and bool(seen_mask[f]):
            col = 'tab:green'
        x_def = np.r_[p, [0,0,0]]
        if fov_cfg.get("type","cone") == "pinhole" and cfg.get("camera") is not None:
            coll, edges = draw_camera_frustum_3d(
                ax, x_def=x_def, cam_cfg=cfg["camera"],
                color=col, alpha=fov_cfg.get("alpha",0.15),
                R_wb=R_wb
            )
            fov_art['coll'], fov_art['rim'], fov_art['edges'] = coll, None, edges
        else:
            coll, rim = draw_fov_cone_3d(
                ax, x_def, fov_cfg,
                color=col, alpha=fov_cfg.get("alpha",0.15),
                align=att_cfg.get('align','x'),
                R_wb=R_wb
            )
            fov_art['coll'], fov_art['rim'] = coll, rim



    # Widgets
    s_frame = W.IntSlider(min=0, max=n_frames-1, step=1, value=0, description='frame')
    s_azim  = W.IntSlider(min=-180, max=180, step=1, value=45, description='azim')
    s_elev  = W.IntSlider(min=-10,  max=90,  step=1, value=25, description='elev')
    t_plan  = W.Checkbox(value=True,      description='show plan')
    t_axes  = W.Checkbox(value=show_axes, description='show axes')
    t_fov   = W.Checkbox(value=show_fov,  description='show FOV')
    play    = W.Play(min=0, max=n_frames-1, step=1, interval=50, value=0)
    W.jslink((play, 'value'), (s_frame, 'value'))

    def redraw(f):
        # planned
        if t_plan.value and plan_hist1:
            ph1 = plan_hist1[min(f, len(plan_hist1)-1)]
            if ph1:
                xs, ys, zs = zip(*ph1); plan1_ln.set_data(xs, ys); plan1_ln.set_3d_properties(zs)
            else:
                plan1_ln.set_data([], []); plan1_ln.set_3d_properties([])
            ph2 = plan_hist2[min(f, len(plan_hist2)-1)] if plan_hist2 else []
            if ph2:
                xs, ys, zs = zip(*ph2); plan2_ln.set_data(xs, ys); plan2_ln.set_3d_properties(zs)
            else:
                plan2_ln.set_data([], []); plan2_ln.set_3d_properties([])
        else:
            plan1_ln.set_data([], []); plan1_ln.set_3d_properties([])
            plan2_ln.set_data([], []); plan2_ln.set_3d_properties([])

        # executed
        xs1 = [p[0] for p in exec1[:f+1]]; ys1 = [p[1] for p in exec1[:f+1]]; zs1 = [p[2] for p in exec1[:f+1]]
        xs2 = [p[0] for p in exec2[:f+1]]; ys2 = [p[1] for p in exec2[:f+1]]; zs2 = [p[2] for p in exec2[:f+1]]
        exe1_ln.set_data(xs1, ys1); exe1_ln.set_3d_properties(zs1)
        exe2_ln.set_data(xs2, ys2); exe2_ln.set_3d_properties(zs2)

        # points
        p1 = np.array(_pos3(exec1[min(f, len(exec1)-1)]))
        p2 = np.array(_pos3(exec2[min(f, len(exec2)-1)]))
        dot1.set_data([p1[0]], [p1[1]]); dot1.set_3d_properties([p1[2]])
        dot2.set_data([p2[0]], [p2[1]]); dot2.set_3d_properties([p2[2]])

        # triads (both agents)
        if t_axes.value:
            if att1_lines: update_body_axes_artists_3d(att1_lines, p1, _R_exec(1, f), L=L_tri)
            if att2_lines: update_body_axes_artists_3d(att2_lines, p2, _R_exec(2, f), L=L_tri)
        else:
            for L in (att1_lines, att2_lines):
                if L:
                    for ln in (L['bx'], L['by'], L['bz']):
                        ln.set_data([], []); ln.set_3d_properties([])

        # FOV
        p_def = np.array(_pos3(exec2[f] if agent_id == 2 else exec1[f]))
        R_def = _R_exec(agent_id, f)
        _draw_fov(f, p_def, R_def)



        ax.view_init(elev=s_elev.value, azim=s_azim.value)
        fig.canvas.draw_idle()

    # wire up
    s_frame.observe(lambda ch: redraw(ch['new']), names='value')
    s_azim.observe(lambda ch: redraw(s_frame.value), names='value')
    s_elev.observe(lambda ch: redraw(s_frame.value), names='value')
    t_plan.observe(lambda ch: redraw(s_frame.value), names='value')
    t_axes.observe(lambda ch: redraw(s_frame.value), names='value')
    t_fov.observe(lambda ch: redraw(s_frame.value), names='value')

    redraw(0)
    ui = W.VBox([W.HBox([play, s_frame]), W.HBox([s_azim, s_elev, t_plan, t_axes, t_fov])])
    display(ui)  # intentionally not displaying 'fig' to avoid a static duplicate


def add_triad_legend(ax, colors=('tab:red','tab:green','tab:blue'),
                     labels=('x_b (boresight)', 'y_b', 'z_b'),
                     loc='lower left', ncol=1, title='Body axes',
                     keep_legend=None):
    proxies = [Line2D([0], [0], lw=2, color=c) for c in colors]
    # Create the axes-legend for the triad
    leg_axes = ax.legend(proxies, labels, loc=loc, ncol=ncol, title=title)
    # Re-add the previous legend (plans/exec) on top
    if keep_legend is not None:
        ax.add_artist(keep_legend)
    return leg_axes