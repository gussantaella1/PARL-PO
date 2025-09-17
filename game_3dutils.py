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
    dims_from_D, step_double_integrator_D, unpack_tau_flat, split_players_from_z,
    make_bounds, build_h_tilde, build_g_tilde_linear,
    world_to_body_R, in_fov, _unit,
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

def _unit(v, eps=1e-12):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v if n < eps else v/n

def _seed_from_boresight(a, world_up=(0,0,1), align='x'):
    # One-shot TRIAD seed when no previous R exists.
    a = _unit(a); u = _unit(world_up)
    if align == 'x':
        y = u - a*np.dot(u, a)
        if np.linalg.norm(y) < 1e-8:  # pick something ⟂ a
            e = np.eye(3)[np.argmin(np.abs(a))]
            y = e - a*np.dot(e, a)
        y = _unit(y)
        z = _unit(np.cross(a, y))
        return np.vstack([a, y, z])
    else:  # align == 'z'
        x = u - a*np.dot(u, a)
        if np.linalg.norm(x) < 1e-8:
            e = np.eye(3)[np.argmin(np.abs(a))]
            x = e - a*np.dot(e, a)
        x = _unit(x)
        y = _unit(np.cross(a, x))
        return np.vstack([x, y, a])
    
def evolve_dcm(R_prev, a_des, *, align='x', world_up=(0,0,1),
               k_roll=0.0, yaw_lock=False, eps=1e-9, yaw_gain=1.0):
    """
    Rows-as-axes convention:
      R[0] = x_b in world, R[1] = y_b in world, R[2] = z_b in world.
    Step 1: minimal transport (RIGHT-multiply by Q.T).
    Step 2: yaw-only horizon lock using the TRANSPORTED tangent as the sign reference.
    """
    I  = np.eye(3)
    a  = _unit(a_des)
    up = _unit(world_up)

    # ---- Step 1: transport previous frame to align boresight (rows convention) ----
    if R_prev is None:
        R_tr = _seed_from_boresight(a, up, align)
    else:
        b_prev = (R_prev[0] if align == 'x' else R_prev[2])  # previous boresight (world)
        v = np.cross(b_prev, a); s = float(np.linalg.norm(v))
        c = float(np.clip(np.dot(b_prev, a), -1.0, 1.0))
        if s < 1e-8:
            if c > 0.0:
                R_tr = R_prev.copy()
            else:
                # 180°: rotate π about previous tangent to break tie
                axis = _unit(R_prev[1] if align == 'x' else R_prev[0])
                K = _skew_np(axis)
                Q = I + 2.0*(K @ K)         # Rodrigues for θ=π
                R_tr = R_prev @ Q.T         # <-- rows-as-axes: right-multiply by Q^T
        else:
            k  = v / s
            K  = _skew_np(k)
            th = np.arctan2(s, c)
            Q  = I + np.sin(th)*K + (1.0 - np.cos(th))*(K @ K)
            R_tr = R_prev @ Q.T            # <-- rows-as-axes

    # ---- Step 2: yaw lock or tiny roll trim (about boresight) ----
    if yaw_lock:
        # transported tangent as sign reference
        y_ref = (R_tr[1] if align == 'x' else R_tr[0])

        t = up - a*np.dot(up, a)           # project 'up' onto plane ⟂ a
        n = float(np.linalg.norm(t))
        if n > eps:
            t = t / n
            # pick sign closest to transported tangent to avoid π jumps
            if np.dot(t, y_ref) < 0.0:
                t = -t
            if align == 'x':
                y = t
                z = _unit(np.cross(a, y))
                R = np.vstack([a, y, z])
            else:  # align == 'z'
                y = t
                x = _unit(np.cross(y, a))
                R = np.vstack([x, y, a])
        else:
            # a ~ ±up: no reliable horizon; keep transported frame
            R = R_tr

        # optional partial lock (damping)
        if 0.0 < yaw_gain < 1.0:
            # slerp between transported and locked frames (world left, rows-as-axes via SO(3) log/exp on R^T)
            R = _slerp_R(R_tr.T, R.T, yaw_gain).T
    else:
        R = R_tr
        if k_roll != 0.0:
            y_cur = (R_tr[1] if align == 'x' else R_tr[0])
            t = up - a*np.dot(up, a)
            n = float(np.linalg.norm(t))
            if n > eps:
                t = t / n
                num  = np.dot(np.cross(y_cur, t), a)
                den  = float(np.clip(np.dot(y_cur, t), -1.0, 1.0))
                dphi = k_roll * np.arctan2(num, den)
                if abs(dphi) > 1e-12:
                    Ka = _skew_np(a)
                    Q  = I + np.sin(dphi)*Ka + (1.0 - np.cos(dphi))*(Ka @ Ka)
                    R  = R @ Q.T            # <-- rows-as-axes
            # else: keep R as-is

    # ---- Step 3: safety re-orthonormalization ----
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R




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

    if att_kind in ("roll1d","lin6"):        # <-- FIX the undefined 'use_lin' bug
        x_lb, x_ub, u_lb, u_ub = _pad_trans_bounds_only_finite(
            x_lb_tr, x_ub_tr, u_lb_tr, u_ub_tr, nx, nu, nx_tr, nu_tr, BIG=1e6
        )
    else:
        x_lb, x_ub, u_lb, u_ub = x_lb_tr, x_ub_tr, u_lb_tr, u_ub_tr

    htil_fun = build_h_tilde(nx, nu, T, N, x_lb, x_ub, u_lb, u_ub, cfg)

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
    att0_list = _read_att_x0_unified(att_cfg, att_kind, N)

    x0_rows = np.asarray(cfg["x0"], float)
    assert x0_rows.shape[0] == N, f"cfg['x0'] must have {N} rows"
    theta_parts = []
    for i in range(N):
        xtr = x0_rows[i][:nx_tr]
        x0_i = np.hstack([xtr, att0_list[i]]) if att_kind != "none" else xtr
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
    Receding-horizon rollout using evolve_dcm() for attitude baseline.
    Attitude modes:
      - "lin3d"/"lin2d_roll": small-angle δθ(3), δω(3) + τ(3)  → nx += 6, nu += 3
      - "roll1d":            roll φ, φdot + τx                 → nx += 2, nu += 1
      - anything else:       translation only
    Inequality bounds are translation-only when attitude is present.
    Rendering baseline attitude comes from velocity → evolve_dcm(yaw_lock=True),
    then we apply δθ (lin6) or φ (roll1d) on top.
    """
    import numpy as np
    import casadi as ca

    # ---------- sizes / timing ----------
    N = 2
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx_tr, nu_tr = dims_from_D(D)
    T, dt = int(cfg["T"]), float(cfg["dt"])

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

    # ---------- inequality bounds ----------
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
    obj   = 0.5 * ca.dot(G, G) + 1e-10 * ca.dot(z_sym, z_sym)

    opts = {'ipopt.print_level': 0, 'print_time': 0, 'expand': True}
    if ipopt_opts: opts.update(ipopt_opts)
    solver = ca.nlpsol('solver', 'ipopt', {'x': z_sym, 'p': theta_sym, 'f': obj}, opts)

    # ---------- μ ≥ 0 ----------
    BIG = 1e8
    lbz = -BIG*np.ones(z_sym.shape[0]); ubz = BIG*np.ones(z_sym.shape[0])
    lbz[N*nprim + gtil.shape[0]:] = 0.0

    # ---------- initial states ----------
    att0_list = _read_att_x0_unified(att_cfg, att_kind, N)
    x0_rows = np.asarray(cfg["x0"], float)
    x1_tr = x0_rows[0][:nx_tr].copy()
    x2_tr = x0_rows[1][:nx_tr].copy()
    x1 = np.hstack([x1_tr, att0_list[0]]) if att_kind != "none" else x1_tr
    x2 = np.hstack([x2_tr, att0_list[1]]) if att_kind != "none" else x2_tr
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

    # ---------- attitude baseline via evolve_dcm ----------
    align     = att_cfg.get("align","x")
    world_up  = np.asarray(att_cfg.get("up",[0,0,1]), float)
    vmin      = float(att_cfg.get("min_speed_for_axis", 1e-3))
    yaw_lock  = bool(att_cfg.get("horizon_lock", True))
    roll_gain = float(att_cfg.get("roll_gain", 0.0))

    # carry previous baseline DCM per agent
    R_prev = [None, None]

    def _default_boresight():
        return np.array([1,0,0]) if align == 'x' else np.array([0,0,1])

    axis_cache = [None, None]

    def _boresight_from_xtr(xtr, agent_idx):
        v = np.asarray(xtr[D:2*D], float); spd = float(np.linalg.norm(v))
        if spd >= vmin:
            a = v/spd; a = np.array([a[0], a[1], 0.0]) if D==2 else a
            axis_cache[agent_idx] = a
            return a
        return axis_cache[agent_idx] if axis_cache[agent_idx] is not None else _default_boresight()


    def _R_nom_from_xtr(xtr, agent_idx):
        a = _boresight_from_xtr(xtr, agent_idx)
        R_new = evolve_dcm(R_prev[agent_idx], a,
                           align=align, world_up=world_up,
                           k_roll=roll_gain, yaw_lock=yaw_lock)
        R_prev[agent_idx] = R_new
        return R_new

    def _render_R_from_state(x, agent_idx):
        xtr = x[:nx_tr]
        R_nom = _R_nom_from_xtr(xtr, agent_idx)
        if att_kind == "lin6":
            eps = _eps3_from_statevec(x, nx_tr)
            R   = _apply_small_angle(R_nom, eps); phi = float(eps[0])
            return R, phi
        elif att_kind == "roll1d":
            phi = _phi_from_statevec(x, nx_tr)
            R   = apply_roll_about_axis(R_nom, phi, align=align)
            return R, float(phi)
        else:
            return R_nom, 0.0

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
        """
        Build a nominal orientation sequence for the planned states X WITHOUT
        mutating the execution baseline R_prev[agent_idx].

        Returns
        -------
        out   : list of dicts per t, each {"R": 3x3 DCM, "phi": float}
        R_end : final nominal DCM after stepping through the plan (not written back)
        """
        Tloc = int(X.shape[0])
        out  = []

        # Local read-only seed; do NOT assign to R_prev here.
        Rloc = R_prev[agent_idx]

        # Keep a local last-used boresight for low-speed frames inside this plan pass
        last_axis = None

        for t in range(Tloc):
            xtr_t = X[t, :nx_tr]

            # ----- choose boresight w/ graceful low-speed handling -----
            v   = np.asarray(xtr_t[D:2*D], float)
            spd = float(np.linalg.norm(v))
            if spd >= vmin:
                a_t = v / spd
                if D == 2:  # lift to 3D if working in 2D
                    a_t = np.array([a_t[0], a_t[1], 0.0], float)
                last_axis = a_t
            else:
                if last_axis is not None:
                    a_t = last_axis
                elif Rloc is not None:
                    a_t = (Rloc[0] if align == 'x' else Rloc[2])
                else:
                    a_t = np.array([1,0,0], float) if align == 'x' else np.array([0,0,1], float)

            # ----- evolve nominal DCM (minimal-rotation + yaw lock/roll trim) -----
            R_nom = evolve_dcm(
                Rloc, a_t,
                align=align, world_up=world_up,
                k_roll=roll_gain, yaw_lock=yaw_lock
            )
            Rloc = R_nom  # advance local nominal only

            # ----- apply attitude state on top of nominal -----
            if att_kind == "lin6":
                eps_t = _eps3_from_statevec(X[t], nx_tr)  # δθ about body axes (small-angle)
                R_t   = _apply_small_angle(R_nom, eps_t)
                phi_t = float(eps_t[0])  # keep for plotting/debug (matches prior use)
            elif att_kind == "roll1d":
                phi_t = float(X[t, nx_tr]) if X.shape[1] >= nx_tr+1 else 0.0
                R_t   = apply_roll_about_axis(R_nom, phi_t, align=align)
            else:
                R_t, phi_t = R_nom, 0.0

            out.append({"R": R_t, "phi": phi_t})

        # IMPORTANT: do not assign back to R_prev[agent_idx] here.
        return out, Rloc


    def replan(theta_vec, z_init):
        sol  = solver(x0=z_init, lbx=lbz, ubx=ubz, p=theta_vec)
        z_new = np.array(sol['x']).squeeze()
        taus  = split_players_from_z(z_new, N, T, nx, nu)
        X1, U1 = unpack_tau_flat(taus[0], nx, nu, T)
        X2, U2 = unpack_tau_flat(taus[1], nx, nu, T)

        plan1 = [(float(X1[t,0]), float(X1[t,1]), float(X1[t,2] if D==3 else 0.0)) for t in range(T)]
        plan2 = [(float(X2[t,0]), float(X2[t,1]), float(X2[t,2] if D==3 else 0.0)) for t in range(T)]

        att1_plan, _ = _plan_attitudes_from_X(X1, 0)
        att2_plan, _ = _plan_attitudes_from_X(X2, 1)
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

def _proj_horizon_tangent(a, up, t_ref=None, eps=1e-9):
    a = _unit(a); up = _unit(up)
    t = up - a*np.dot(up, a)             # project 'up' onto plane ⟂ a
    n = np.linalg.norm(t)
    if n < eps:
        # boresight ~ up: keep previous tangent if we have one; else pick any ⟂ a
        if t_ref is not None:
            return _unit(t_ref)
        e = np.eye(3)[np.argmin(np.abs(a))]
        t = e - a*np.dot(e, a)
        n = np.linalg.norm(t)
        if n < eps:
            t = np.array([0,1,0])  # final fallback
    t = t / n
    # sign continuity: make t close to t_ref
    if t_ref is not None and np.dot(t, t_ref) < 0.0:
        t = -t
    return t


# -------------------- animation (triads + triad legend) --------------------
def animate_rollout_3d(frames_dict, save_path="traj_3D.gif", fps=20, cfg=None,
                       show_fov=True, show_axes=True):
    import shutil
    import numpy as np
    import matplotlib.pyplot as plt
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

    # attitude cache per agent (for evolve_dcm continuity)
    _anim_cache = [None, None]

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

    # Triad legend
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

    # helpers
    def _pos3(p_like):
        p = np.asarray(p_like, float).ravel()
        return p[:3] if p.size >= 3 else np.array([p[0], p[1], 0.0], float)

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

    align    = att_cfg.get('align','x')
    up       = np.asarray(att_cfg.get('up',[0,0,1]), float)
    yaw_lock = bool(att_cfg.get('horizon_lock', True))
    roll_gain = float(att_cfg.get('roll_gain', 0.0))
    vmin     = float(att_cfg.get('min_speed_for_axis', 1e-3))

    def _default_boresight():
        return np.array([1,0,0]) if align == 'x' else np.array([0,0,1])

    def _R_exec(agent, idx):
        # use stored exec attitudes if present
        key = 'exec_att1' if agent == 1 else 'exec_att2'
        L = frames_dict.get(key, [])
        if L and idx < len(L) and 'R' in L[idx] and L[idx]['R'] is not None:
            _anim_cache[agent-1] = np.asarray(L[idx]['R'], float)
            return _anim_cache[agent-1]

        # otherwise derive boresight and evolve
        pos_seq = exec1 if agent == 1 else exec2
        if axis_hist and idx < len(axis_hist) and axis_hist[idx] is not None:
            a_des = np.asarray(axis_hist[idx], float)
            if a_des.shape[0] == 2: a_des = np.array([a_des[0], a_des[1], 0.0], float)
            n = np.linalg.norm(a_des); a_des = a_des if n < 1e-12 else a_des/n
        else:
            if idx > 0:
                p  = np.asarray(pos_seq[idx], float)
                pp = np.asarray(pos_seq[idx-1], float)
                dv = p - pp; n = np.linalg.norm(dv)
                if n < vmin:
                    if _anim_cache[agent-1] is not None:
                        a_des = (_anim_cache[agent-1][0] if align=='x' else _anim_cache[agent-1][2])
                    else:
                        a_des = _default_boresight()
                else:
                    a_des = dv/n
            else:
                a_des = _default_boresight()

        R_prev = _anim_cache[agent-1]
        R_new  = evolve_dcm(R_prev, a_des, align=align, world_up=up,
                            k_roll=roll_gain, yaw_lock=yaw_lock)
        _anim_cache[agent-1] = R_new
        return R_new

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
            axis = R_wb[0] if align == 'x' else R_wb[2]
            coll, rim = draw_fov_cone_3d(
                ax, x_def, axis, fov_cfg,
                color=col, alpha=fov_cfg.get("alpha",0.15),
                align=align
            )
            fov_art['coll'], fov_art['rim'] = coll, rim

    # ---------- animation callbacks ----------
    def init():
        for ln in (plan1_ln, plan2_ln, exe1_ln, exe2_ln, dot1, dot2):
            ln.set_data([], []); ln.set_3d_properties([])
        if leg_main: ax.add_artist(leg_main)
        return plan1_ln, plan2_ln, exe1_ln, exe2_ln, dot1, dot2

    def update(f):
        # planned
        if plan_hist1 and f < len(plan_hist1) and plan_hist1[f]:
            xs, ys, zs = zip(*plan_hist1[f]); plan1_ln.set_data(xs, ys); plan1_ln.set_3d_properties(zs)
        if plan_hist2 and f < len(plan_hist2) and plan_hist2[f]:
            xs, ys, zs = zip(*plan_hist2[f]); plan2_ln.set_data(xs, ys); plan2_ln.set_3d_properties(zs)

        # executed path to frame f
        xs1 = [p[0] for p in exec1[:f+1]]; ys1 = [p[1] for p in exec1[:f+1]]; zs1 = [p[2] for p in exec1[:f+1]]
        xs2 = [p[0] for p in exec2[:f+1]]; ys2 = [p[1] for p in exec2[:f+1]]; zs2 = [p[2] for p in exec2[:f+1]]
        exe1_ln.set_data(xs1, ys1); exe1_ln.set_3d_properties(zs1)
        exe2_ln.set_data(xs2, ys2); exe2_ln.set_3d_properties(zs2)

        # points
        x1,y1,z1 = exec1[min(f, len(exec1)-1)]
        x2,y2,z2 = exec2[min(f, len(exec2)-1)]
        dot1.set_data([x1], [y1]); dot1.set_3d_properties([z1])
        dot2.set_data([x2], [y2]); dot2.set_3d_properties([z2])

        # triads
        if show_axes:
            R1 = _R_exec(1, f); R2 = _R_exec(2, f)
            if att1_lines: update_body_axes_artists_3d(att1_lines, np.array([x1,y1,z1]), R1, L=L_tri)
            if att2_lines: update_body_axes_artists_3d(att2_lines, np.array([x2,y2,z2]), R2, L=L_tri)
        # FOV (selected agent)
        p_def = np.asarray(_pos3(exec2[f] if agent_id == 2 else exec1[f]))
        R_def = _R_exec(agent_id, f)
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
    import numpy as np
    import ipywidgets as W
    import matplotlib.pyplot as plt
    from IPython.display import display

    # local evolve cache
    R_prev_ui = [None, None]

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
    ax.set_title(title); ax.grid(True); ax.set_box_aspect((1,1,1))

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
    plan1_ln, = ax.plot([], [], [], '--', lw=1, alpha=0.6, label='Plan P1', color='blue')
    plan2_ln, = ax.plot([], [], [], '--', lw=1, alpha=0.6, label='Plan P2', color='orange')
    exe1_ln,  = ax.plot([], [], [], '-',  lw=2, label='Exec P1', color='green')
    exe2_ln,  = ax.plot([], [], [], '-',  lw=2, label='Exec P2', color='red')
    dot1,     = ax.plot([], [], [], 'o',  ms=6, color='cyan',   mec='k', mew=0.6)
    dot2,     = ax.plot([], [], [], 'o',  ms=6, color='orange', mec='k', mew=0.6)
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

    align     = att_cfg.get('align','x')
    up        = np.asarray(att_cfg.get('up',[0,0,1]), float)
    yaw_lock  = bool(att_cfg.get('horizon_lock', True))
    roll_gain = float(att_cfg.get('roll_gain', 0.0))
    vmin      = float(att_cfg.get('min_speed_for_axis', 1e-3))

    def _default_boresight():
        return np.array([1,0,0]) if align == 'x' else np.array([0,0,1])

    def _R_exec(agent, idx):
        # Prefer stored exec attitudes
        key = 'exec_att1' if agent == 1 else 'exec_att2'
        L = frames_dict.get(key, [])
        if L and idx < len(L) and 'R' in L[idx] and L[idx]['R'] is not None:
            R = np.asarray(L[idx]['R'], float)
            R_prev_ui[agent-1] = R
            return R

        # Otherwise derive from path / axis_hist and evolve
        if axis_hist and idx < len(axis_hist) and axis_hist[idx] is not None:
            a_des = np.asarray(axis_hist[idx], float)
            if a_des.shape[0] == 2: a_des = np.array([a_des[0], a_des[1], 0.0], float)
            n = np.linalg.norm(a_des); a_des = a_des if n < 1e-12 else a_des/n
        else:
            pos_seq = exec1 if agent == 1 else exec2
            if idx > 0:
                p  = np.asarray(pos_seq[idx], float)
                pp = np.asarray(pos_seq[idx-1], float)
                dv = p - pp; n = np.linalg.norm(dv)
                if n < vmin:
                    if R_prev_ui[agent-1] is not None:
                        a_des = (R_prev_ui[agent-1][0] if align=='x' else R_prev_ui[agent-1][2])
                    else:
                        a_des = _default_boresight()
                else:
                    a_des = dv/n
            else:
                a_des = _default_boresight()

        R_prev = R_prev_ui[agent-1]
        R_new  = evolve_dcm(R_prev, a_des, align=align, world_up=up,
                            k_roll=roll_gain, yaw_lock=yaw_lock)
        R_prev_ui[agent-1] = R_new
        return R_new

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
                ax, x_def, (R_wb[0] if align=='x' else R_wb[2]),
                fov_cfg, color=col, alpha=fov_cfg.get("alpha",0.15), align=align
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

def _read_att_x0_unified(att_cfg, att_kind, N):
    """
    Unified attitude initializer:
      - Accepts att.init.rpy0 and att.init.rpy_rate0 (either (3,) or (N,3))
      - Optional att.init.degrees=True to interpret inputs as degrees
      - Returns per-agent blocks that match the chosen attitude state layout.

    att_kind:
      "lin6"   -> returns [δθ0(3), δω0(3)] per agent  (6 vals)
      "roll1d" -> returns [φ0, φdot0] per agent       (2 vals)
      "none"   -> returns [] per agent
    """
    init = att_cfg.get("init", {})
    rpy0 = np.asarray(init.get("rpy0", [0.0, 0.0, 0.0]), dtype=float)
    rpy_rate0 = np.asarray(init.get("rpy_rate0", [0.0, 0.0, 0.0]), dtype=float)
    use_deg = bool(init.get("degrees", False))

    # broadcast shapes
    if rpy0.ndim == 1:       rpy0 = np.tile(rpy0[None, :], (N, 1))
    if rpy_rate0.ndim == 1:  rpy_rate0 = np.tile(rpy_rate0[None, :], (N, 1))

    assert rpy0.shape == (N, 3),      f"att.init.rpy0 must be (N,3) or (3,), got {rpy0.shape}"
    assert rpy_rate0.shape == (N, 3), f"att.init.rpy_rate0 must be (N,3) or (3,), got {rpy_rate0.shape}"

    if use_deg:
        rpy0 = np.deg2rad(rpy0)
        rpy_rate0 = np.deg2rad(rpy_rate0)

    if att_kind == "lin6":
        # small-angle offsets around nominal frame
        return [np.hstack([rpy0[i], rpy_rate0[i]]) for i in range(N)]
    elif att_kind == "roll1d":
        return [np.array([rpy0[i, 0], rpy_rate0[i, 0]], dtype=float) for i in range(N)]
    else:
        return [np.array([], dtype=float) for _ in range(N)]