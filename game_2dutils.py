# game_2dutils.py
# Dedicated 2D utilities and entrypoints. Depends on game_sharedutils.
from __future__ import annotations
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

from game_sharedutils import (
    dims_from_D, step_double_integrator_D, fb,
    pack_trajectory, unpack_trajectory, unpack_tau_flat, split_players_from_z,
    make_bounds, polygon_halfspaces, build_g_tilde, build_h_tilde,
    world_to_body_R, fov_axis_from_vel, in_fov, _unit
)
__all__ = [
    "solve_game_once_2d", "run_rhc_and_collect_frames_2d",
    "plot_planned_trajectories_2d", "draw_arena_2d", "draw_fov_2d"
]

# -------------------- Drawing helpers (2D) --------------------
def draw_arena_2d(ax, cfg):
    ar = cfg.get("arena", {"type":"box"})
    t = ar.get("type", "box")
    if t == "box":
        for k in ("xmin","xmax","ymin","ymax"):
            assert k in ar, f"Missing arena bound: {k}"
        ax.set_xlim(ar["xmin"]-0.5, ar["xmax"]+0.5)
        ax.set_ylim(ar["ymin"]-0.5, ar["ymax"]+0.5)
        ax.add_patch(plt.Rectangle((ar["xmin"], ar["ymin"]),
                                   ar["xmax"]-ar["xmin"], ar["ymax"]-ar["ymin"],
                                   fill=False, lw=1.5, ls='-', alpha=0.7))
    elif t == "circle":
        R = ar["r"]; cx, cy = ar["cx"], ar["cy"]
        ax.set_xlim(cx-R-0.5, cx+R+0.5); ax.set_ylim(cy-R-0.5, cy+R+0.5)
        ax.add_patch(plt.Circle((cx, cy), R, fill=False, lw=1.5, ls='-', alpha=0.7))
    elif t == "polygon":
        V = np.asarray(ar["vertices"], dtype=float)
        ax.set_xlim(V[:,0].min()-0.5, V[:,0].max()+0.5)
        ax.set_ylim(V[:,1].min()-0.5, V[:,1].max()+0.5)
        ax.add_patch(plt.Polygon(V, closed=True, fill=False, lw=1.5, ls='-', alpha=0.7))
    else:
        raise ValueError("Unknown arena type")
    # circular keep-out
    for circ in cfg.get("circles", []):
        ax.add_patch(plt.Circle((circ["cx"], circ["cy"]), circ["r"],
                                fill=False, lw=1.2, ls='--', alpha=0.6))

def draw_fov_2d(ax, x_def, axis, fov_cfg, color='C1', alpha=0.12):
    origin = np.asarray(x_def[:2], float)
    rng = float(fov_cfg["range"])
    half = 0.5*float(fov_cfg["hfov_deg"])
    ang_deg = np.rad2deg(np.arctan2(axis[1], axis[0]))
    wedge = Wedge(origin, rng, ang_deg - half, ang_deg + half, color=color, alpha=alpha)
    ax.add_patch(wedge)
    return wedge

# -------------------- 2D solver entrypoint --------------------
def solve_game_once_2d(cfg: dict, cost_builder, ipopt_opts: dict | None = None):
    N = 2
    D = 2
    nx, nu = dims_from_D(D)
    T, dt = int(cfg["T"]), float(cfg["dt"])
    nprim  = T*nx + (T-1)*nu

    taus = [ca.MX.sym(f"tau{i+1}", nprim) for i in range(N)]
    tau  = ca.vcat(taus)
    theta = ca.MX.sym('theta', nx*N)

    fs = cost_builder(nx, nu, T, N, cfg)
    gtil_fun  = build_g_tilde(nx, nu, T, N, D=D, dt=dt)
    x_lb, x_ub, u_lb, u_ub = make_bounds({**cfg, "D": 2})
    htil_fun  = build_h_tilde(nx, nu, T, N, x_lb, x_ub, u_lb, u_ub, {**cfg, "D": 2})

    gtil = gtil_fun(tau, theta)
    htil = htil_fun(tau, theta)

    lam_t = ca.MX.sym('lam_t', gtil.shape[0])
    mu_t  = ca.MX.sym('mu_t',  htil.shape[0])

    grads = []
    for i in range(N):
        Li = fs[i](tau, theta) - lam_t.T @ gtil - mu_t.T @ htil
        grads.append(ca.gradient(Li, taus[i]))
    FB = ca.vcat([fb(htil[i], mu_t[i]) for i in range(htil.shape[0])])
    G  = ca.vcat(grads + [gtil, FB])
    z  = ca.vcat(taus + [lam_t, mu_t])
    obj = 0.5 * ca.dot(G, G)

    opts = {'ipopt.print_level': 0, 'print_time': 0}
    if ipopt_opts: opts.update(ipopt_opts)
    solver = ca.nlpsol('solver', 'ipopt', {'x': z, 'p': theta, 'f': obj}, opts)

    x0_all = np.asarray(cfg["x0"], float)
    assert x0_all.shape[0] == N and x0_all.shape[1] >= nx, "x0 must be (N,4) for 2D"
    theta0 = np.hstack([x0_all[i, :nx] for i in range(N)])

    lbz = -np.inf*np.ones(z.shape[0]); ubz =  np.inf*np.ones(z.shape[0])
    lbz[N*nprim + gtil.shape[0]:] = 0.0

    sol = solver(x0=np.zeros(z.shape[0]), lbx=lbz, ubx=ubz, p=theta0)
    zstar = np.array(sol['x']).squeeze()
    residual = float(sol['f'])
    print(f"[2D] Objective residual (0.5||G||^2): {residual:.3e}")
    print(f"[2D] z dim: {z.shape[0]} (taus={N*nprim}, lam={gtil.shape[0]}, mu={htil.shape[0]})")

    return dict(
        z=zstar, residual=residual,
        meta=dict(T=T, nx=nx, nu=nu, N=N, D=2, nprim=nprim,
                  n_g=gtil.shape[0], n_h=htil.shape[0])
    )

# -------------------- 2D RHC with execution & FOV --------------------
def run_rhc_and_collect_frames_2d(cfg: dict, cost_builder, steps: int | None = None,
                                  turn_len: int | None = None, ipopt_opts: dict | None = None):
    N = 2
    D = 2
    nx, nu = dims_from_D(D)
    T, dt  = int(cfg["T"]), float(cfg["dt"])
    nprim  = T*nx + (T-1)*nu

    sim_time = cfg.get("sim_time", cfg.get("max_time", cfg.get("duration", None)))
    if steps is None:
        steps = max(1, int(np.ceil(float(sim_time)/dt))) if sim_time is not None else int(cfg.get("steps", 60))
    if turn_len is None:
        turn_len = int(cfg.get("turn_len", 3)) if "turn_seconds" not in cfg else max(1, int(round(float(cfg["turn_seconds"])/dt)))

    taus_syms = [ca.MX.sym(f"tau{i+1}", nprim) for i in range(N)]
    tau_sym   = ca.vcat(taus_syms)
    theta_sym = ca.MX.sym('theta', nx*N)

    fs        = cost_builder(nx, nu, T, N, cfg)
    gtil_fun  = build_g_tilde(nx, nu, T, N, D=D, dt=dt)
    x_lb, x_ub, u_lb, u_ub = make_bounds({**cfg, "D": 2})
    htil_fun  = build_h_tilde(nx, nu, T, N, x_lb, x_ub, u_lb, u_ub, {**cfg, "D": 2})

    gtil = gtil_fun(tau_sym, theta_sym)
    htil = htil_fun(tau_sym, theta_sym)
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

    BIG = 1e8
    lbz = -BIG*np.ones(z_sym.shape[0]); ubz = BIG*np.ones(z_sym.shape[0])
    lbz[N*nprim + gtil.shape[0]:] = 0.0

    x1 = np.asarray(cfg["x0"][0], float)[:nx].copy()
    x2 = np.asarray(cfg["x0"][1], float)[:nx].copy()
    theta_curr = np.r_[x1, x2]

    z_last = np.zeros(z_sym.shape[0])
    z_last[N*nprim + gtil.shape[0]:] = 1e-3

    plan_hist1, plan_hist2 = [], []
    exec_xy1, exec_xy2 = [], []

    fov_cfg = cfg.get("fov", {"enabled": False})
    fov_enabled = bool(fov_cfg.get("enabled", False))
    fov_agent = int(fov_cfg.get("agent", 2))
    fov_axis_hist, fov_seen_mask = [], []
    prev_axis = None

    def replan(theta_vec, z_init):
        sol = solver(x0=z_init, lbx=lbz, ubx=ubz, p=theta_vec)
        z_new = np.array(sol['x']).squeeze()
        taus = split_players_from_z(z_new, N, T, nx, nu)
        X1, U1 = unpack_tau_flat(taus[0], nx, nu, T)
        X2, U2 = unpack_tau_flat(taus[1], nx, nu, T)
        plan1 = [(float(X1[t,0]), float(X1[t,1])) for t in range(T)]
        plan2 = [(float(X2[t,0]), float(X2[t,1])) for t in range(T)]
        return z_new, plan1, plan2, U1, U2

    z_last, plan1, plan2, U1, U2 = replan(theta_curr, z_last)
    step_in_turn = 0

    for k in range(steps):
        if k % turn_len == 0 and k > 0:
            z_last, plan1, plan2, U1, U2 = replan(theta_curr, z_last)
            step_in_turn = 0

        plan_hist1.append(plan1); plan_hist2.append(plan2)

        u1 = U1[min(step_in_turn, len(U1)-1)]
        u2 = U2[min(step_in_turn, len(U2)-1)]
        step_in_turn += 1

        x1 = step_double_integrator_D(x1, u1, dt, D=2)
        x2 = step_double_integrator_D(x2, u2, dt, D=2)
        theta_curr = np.r_[x1, x2]

        exec_xy1.append((float(x1[0]), float(x1[1])))
        exec_xy2.append((float(x2[0]), float(x2[1])))

        if fov_enabled:
            x_def = x2 if fov_agent == 2 else x1
            x_tgt = x1 if fov_agent == 2 else x2
            axis = fov_axis_from_vel(x_def, D=2, prev_axis=prev_axis,
                                     min_speed=fov_cfg.get("min_speed_for_axis",1e-3))
            prev_axis = axis
            fov_axis_hist.append(axis)
            seen, _ = in_fov(np.asarray(x_tgt[:2]), np.asarray(x_def[:2]), axis, fov_cfg, D=2)
            fov_seen_mask.append(bool(seen))
        else:
            fov_axis_hist.append(None)
            fov_seen_mask.append(False)

    return {
        'plan_hist1': plan_hist1,
        'plan_hist2': plan_hist2,
        'exec1_xy'  : exec_xy1,
        'exec2_xy'  : exec_xy2,
        'fov_axis_hist': fov_axis_hist,
        'fov_seen_mask': fov_seen_mask,
    }

# -------------------- 2D plot --------------------
def plot_planned_trajectories_2d(sol: dict, cfg: dict):
    z = sol['z']; meta = sol['meta']
    T, nx, nu, N = meta['T'], meta['nx'], meta['nu'], meta['N']
    taus = split_players_from_z(z, N, T, nx, nu)
    X1, U1 = unpack_tau_flat(taus[0], nx, nu, T)
    X2, U2 = unpack_tau_flat(taus[1], nx, nu, T)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal'); ax.grid(True, ls=':')
    draw_arena_2d(ax, {**cfg, "D": 2})

    for circ in cfg.get("circles", []):
        ax.add_patch(plt.Circle((circ["cx"], circ["cy"]), circ["r"],
                                fill=False, lw=1.2, ls='--', alpha=0.6))
    ax.plot(X1[:,0], X1[:,1], '-o', label='P1')
    ax.plot(X2[:,0], X2[:,1], '-o', label='P2')
    ax.scatter([X1[0,0], X2[0,0]], [X1[0,1], X2[0,1]], s=80, marker='*', zorder=5)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(loc='upper right')
    ax.set_title('Planned trajectories (2D)')
    plt.show()