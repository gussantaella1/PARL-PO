# game_sharedutils.py
# Shared, dimension-aware utilities for the double-integrator multi-player game.
# Only generic, reusable pieces live here (no plotting UI beyond minimal needs).
from __future__ import annotations
import numpy as np
import casadi as ca

# ---- exports ----
__all__ = [
    # dims & dynamics
    "dims_from_D", "double_integrator", "step_double_integrator_D",
    # complementarity
    "fb",
    # trajectory pack/unpack
    "pack_trajectory", "unpack_trajectory", "unpack_tau_flat", "split_players_from_z",
    # bounds & geometry
    "make_bounds", "regular_polygon", "polygon_halfspaces",
    # constraints
    "build_g_tilde", "build_h_tilde",
    # FOV / frames
    "_unit", "world_to_body_R", "minimal_rotation", "frame_from_axis_continuous",
    "fov_axis_from_vel", "in_fov",
]

__all__ += ["step_roll", "frame_from_axis", "apply_roll_about_axis",
            "as_numpy_const", "hcw_mean_motion", "hcw_discrete_mats", "build_g_tilde_linear"]

# --- add to game_sharedutils.py top-level exports ---
__all__ += ["augment_AB_for_att", "augment_bounds_with_att", "pad_x0_with_att"]

__all__ += ["augment_bounds_with_quat", "build_g_tilde_tr_plus_quat"]


# add to exports
__all__ += ["build_g_tilde_tr_plus_quat_locked_boresight"]

__all__ += [
    "augment_bounds_with_quat",
    "build_g_tilde_tr_plus_quat",
    "q_to_R", "R_to_q","transported_R_from_state"
]



# -------------------- dims & dynamics (generic)--------------------
def dims_from_D(D: int):
    assert D in (2,3), "Only D=2 or D=3 supported."
    return 2*D, D

def double_integrator(D: int = 3, dt: float = 1.0):
    """
    Discrete-time linear double integrator in D dims:
      x = [p(1..D), v(1..D)], u = [a(1..D)]
      p⁺ = p + dt*v, v⁺ = v + dt*a
    Returns A (2D×2D), B (2D×D) as casadi.MX.
    """
    nx, nu = 2*D, D
    A = np.eye(nx, dtype=float)
    for d in range(D):
        A[d, D + d] = dt
    B = np.zeros((nx, nu), dtype=float)
    for d in range(D):
        B[D + d, d] = dt
    return ca.MX(A), ca.MX(B)

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

# --------- attitude 

# ---------- Attitude helpers (shared) ----------

# --- in game_sharedutils.py ---

def step_roll(psi, dt, att_cfg):
    """
    Decoupled roll kinematics.
    Modes:
      - hold (default): psi stays constant
      - rate: psi += dt * psi_rate (or psi_rate_1/_2 per-agent elsewhere)
      - track: first-order track to psi_target with time constant tau
    """
    mode = (att_cfg.get("mode", "hold") or "hold").lower()
    if mode == "rate":
        rate = float(att_cfg.get("psi_rate", 0.0))
        return psi + dt*rate
    elif mode == "track":
        tau  = float(att_cfg.get("tau", 1.0))
        pdes = float(att_cfg.get("psi_target", psi))
        return psi + dt*(pdes - psi)/max(tau, 1e-6)
    else:
        return psi
    
def step_phi(phi: float, dt: float, att_cfg: dict):
    mode = (att_cfg.get("mode", "hold") or "hold").lower()
    if mode == "hold":
        return phi
    if mode == "rate":
        return phi + float(att_cfg.get("phi_rate", 0.0)) * dt
    if mode == "track":
        tau = max(1e-3, float(att_cfg.get("tau", 1.0)))
        phi_target = float(att_cfg.get("phi_target", 0.0))
        a = dt / tau
        return (1 - a) * phi + a * phi_target
    return phi


def frame_from_axis(axis, align="x", world_up=(0,0,1), stabilize_up=False, prev_R=None):
    """
    Build body frame rows [x_b; y_b; z_b].
    - align="x": x_b is along `axis` (boresight).
    - stabilize_up=False: do NOT force yaw to be 'up' (reduces flipping).
    - stabilize_up=True: tries to keep z_b close to world_up (can flip near poles).
    - prev_R: optional previous frame; used only to survive singularities.
    """
    a = _unit(np.asarray(axis, float))

    if align != "x":
        # keep your existing 'z'-forward behavior if you use it elsewhere
        return world_to_body_R(a, 3, align=align)

    x_b = a
    if stabilize_up:
        up = _unit(np.asarray(world_up, float))
        y_tmp = np.cross(up, x_b)       # “upright” y
        n = np.linalg.norm(y_tmp)
        if n < 1e-8:
            # boresight ~ up; reuse previous y to avoid a jump, or pick a safe ref
            if prev_R is not None:
                y_b = _unit(prev_R[1])
            else:
                ref = np.array([0,1,0]) if abs(x_b[1]) < 0.9 else np.array([0,0,1])
                y_b = _unit(np.cross(ref, x_b))
        else:
            y_b = y_tmp / n
        z_b = _unit(np.cross(x_b, y_b))
    else:
        # “free” mode: choose a fixed reference that avoids degeneracy; no up preference
        ref = np.array([0,0,1]) if abs(x_b[2]) < 0.9 else np.array([0,1,0])
        y_b = _unit(np.cross(ref, x_b))
        z_b = _unit(np.cross(x_b, y_b))

    return np.vstack([x_b, y_b, z_b])


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
def att_roll_dims(att_cfg: dict):
    """
    Return (nxa, nua, model) for attitude subsystem.
    model 'roll1' : x_att = [phi],    u_att = [omega]      (phi_{k+1} = phi_k + dt*omega_k)
    model 'roll2' : x_att = [phi, w], u_att = [alpha]      (second-order integrator)
    """
    model = (att_cfg.get("model") or "roll1").lower()
    if model == "roll1":
        return 1, 1, "roll1"
    elif model == "roll2":
        return 2, 1, "roll2"
    else:
        raise ValueError(f"Unknown att.model={model}")

def att_roll_AB(dt: float, att_cfg: dict):
    """Return (A_att, B_att) for the chosen roll model as numpy arrays."""
    nxa, nua, model = att_roll_dims(att_cfg)
    if model == "roll1":
        A = np.array([[1.0]], float)
        B = np.array([[dt ]], float)     # phi^+ = phi + dt*omega
    else:  # roll2
        A = np.array([[1.0, dt],
                      [0.0, 1.0]], float)
        B = np.array([[0.0],
                      [dt ]], float)     # [phi,w]^+ = A[phi,w] + B*alpha
    return A, B

def blkdiag2D(A, B):
    """Block-diagonal for square A,B."""
    return np.block([[A, np.zeros((A.shape[0], B.shape[1]))],
                     [np.zeros((B.shape[0], A.shape[1])), B]])

def augment_AB_for_att(Ad_tr_mx, Bd_tr_mx, dt: float, att_cfg: dict):
    Ad_tr = as_numpy_const(Ad_tr_mx)
    Bd_tr = as_numpy_const(Bd_tr_mx)
    nx_tr = Ad_tr.shape[0]
    nu_tr = Bd_tr.shape[1]

    model = (att_cfg.get("model") or "roll1").lower()

    if model in ("roll1", "roll2", "roll2_tau"):
        A_att, B_att = att_roll_AB(dt, att_cfg)   # your existing roll variants
        nxa, nua = A_att.shape[0], B_att.shape[1]
    elif model == "lin3d":
        A_att, B_att = att_lin3d_AB(dt, att_cfg)  # NEW
        nxa, nua = A_att.shape[0], B_att.shape[1]
    else:
        raise ValueError(f"Unknown att.model={model}")

    # augmented A, B
    Ad_aug = np.block([
        [Ad_tr,                         np.zeros((nx_tr, nxa))],
        [np.zeros((nxa, Ad_tr.shape[1])), A_att               ],
    ])
    Bd_aug = np.block([
        [Bd_tr,                         np.zeros((nx_tr, nua))],
        [np.zeros((nxa, nu_tr)),        B_att                 ],
    ])

    idx = dict(
        nx_tr=nx_tr, nu_tr=nu_tr,
        nx=nx_tr + nxa, nu=nu_tr + nua,
        model=model,
        i_phi=None,
        i_w=None,
        i_dtheta=(nx_tr, nx_tr+3) if model=="lin3d" else None,
        i_w3=(nx_tr+3, nx_tr+6)   if model=="lin3d" else None,
        i_u_att=(nu_tr, nu_tr+nua)
    )
    return ca.MX(Ad_aug), ca.MX(Bd_aug), idx



def augment_bounds_with_att(x_lb, x_ub, u_lb, u_ub, att_cfg: dict):
    """
    Append attitude bounds to the existing translational bounds.
    Expected keys (examples):
      roll1 : att.phi_min/max, att.omega_max
      roll2 : att.phi_min/max, att.w_max, att.alpha_max
    If a bound is omitted, we use a safe default.
    """
    model = (att_cfg.get("model") or "roll1").lower()
    if model == "roll1":
        phi_min = float(att_cfg.get("phi_min", -np.pi))
        phi_max = float(att_cfg.get("phi_max",  np.pi))
        omg_max = float(att_cfg.get("omega_max", 0.0))  # rad/s (input)
        x_lb_att = np.array([phi_min], float)
        x_ub_att = np.array([phi_max], float)
        u_lb_att = np.array([-omg_max], float)
        u_ub_att = np.array([ omg_max], float)

    elif model == "roll2":
        phi_min = float(att_cfg.get("phi_min", -np.pi))
        phi_max = float(att_cfg.get("phi_max",  np.pi))
        w_max   = float(att_cfg.get("w_max",   0.6))     # rad/s (state)
        a_max   = float(att_cfg.get("alpha_max", 0.0))   # rad/s^2 (input)
        x_lb_att = np.array([phi_min, -w_max], float)
        x_ub_att = np.array([phi_max,  w_max], float)
        u_lb_att = np.array([-a_max], float)
        u_ub_att = np.array([ a_max], float)

    elif model == "roll2_tau":
        phi_min = float(att_cfg.get("phi_min", -np.pi))
        phi_max = float(att_cfg.get("phi_max",  np.pi))
        w_max   = float(att_cfg.get("w_max",   0.6))     # rad/s (state)
        tau_max = float(att_cfg.get("tau_max", 0.05))    # N·m (input)
        x_lb_att = np.array([phi_min, -w_max], float)
        x_ub_att = np.array([phi_max,  w_max], float)
        u_lb_att = np.array([-tau_max], float)
        u_ub_att = np.array([ tau_max], float)

    elif model == "lin3d":
        dth_max = np.asarray(att_cfg.get("dtheta_bounds", [0.25, 0.25, 0.25]), float)  # rad
        w_max   = np.asarray(att_cfg.get("w_bounds",      [0.6,  0.6,  0.6 ]),  float) # rad/s
        tau_max = np.asarray(att_cfg.get("tau_bounds",    [0.05, 0.05, 0.05]), float) # N·m
        x_lb_att = np.r_[-dth_max, -w_max]
        x_ub_att = np.r_[ dth_max,  w_max]
        u_lb_att = -tau_max
        u_ub_att =  tau_max

    else:
        raise ValueError(f"Unknown att.model={model}")

    return (np.r_[x_lb, x_lb_att],
            np.r_[x_ub, x_ub_att],
            np.r_[u_lb, u_lb_att],
            np.r_[u_ub, u_ub_att])

def pad_x0_with_att(x0_row: np.ndarray, att_cfg: dict, D: int):
    """
    Ensure x0 includes attitude entries based on att.model:
      - roll1      : x_att = [phi]
      - roll2      : x_att = [phi, w]
      - roll2_tau  : x_att = [phi, w]
      - lin3d      : x_att = [dtheta(3), w(3)]
    """
    import numpy as np
    nx_tr, _ = dims_from_D(D)
    model = (att_cfg.get("model") or "roll1").lower()
    x0 = np.asarray(x0_row, float).copy()

    if model == "roll1":
        phi0 = float(att_cfg.get("phi0", 0.0))
        x0_att = np.array([phi0], dtype=float)
        return np.r_[x0[:nx_tr], x0_att]

    if model in ("roll2", "roll2_tau"):
        phi0 = float(att_cfg.get("phi0", 0.0))
        w0   = float(att_cfg.get("w0",   0.0))
        x0_att = np.array([phi0, w0], dtype=float)
        return np.r_[x0[:nx_tr], x0_att]

    if model == "lin3d":
        dtheta0 = np.asarray(att_cfg.get("dtheta0", [0.0, 0.0, 0.0]), dtype=float).reshape(3,)
        w0      = np.asarray(att_cfg.get("w0",      [0.0, 0.0, 0.0]), dtype=float).reshape(3,)
        return np.r_[x0[:nx_tr], dtheta0, w0]

    raise ValueError(f"Unknown att.model={model}")



# -------------------- complementarity --------------------
def fb(a, b, eps: float = 1e-6):
    """Smooth Fischer–Burmeister complementarity φ(a,b)≈0 enforcing a≥0 ⟂ b≥0."""
    return ca.sqrt(a*a + b*b + 2*eps) - a - b

# -------------------- pack/unpack helpers --------------------
def pack_trajectory(xs, us):
    """Pack T states then T-1 controls into a single vector τᵢ."""
    return ca.vcat([ca.vcat(xs), ca.vcat(us)])

def unpack_trajectory(tau, nx: int, nu: int, T: int):
    """Inverse of pack: return lists xs (len T) and us (len T-1)."""
    nX = nx*T
    X = tau[:nX]; U = tau[nX:]
    xs = [X[i*nx:(i+1)*nx] for i in range(T)]
    us = [U[i*nu:(i+1)*nu] for i in range(T-1)]
    return xs, us

def unpack_tau_flat(tau_flat, nx: int, nu: int, T: int):
    """Flattened τᵢ -> numeric arrays X(T×nx), U((T-1)×nu)."""
    nX = nx*T
    X = np.array(tau_flat[:nX]).reshape(T, nx)
    U = np.array(tau_flat[nX:]).reshape(T-1, nu)
    return X, U

def split_players_from_z(z, N: int, T: int, nx: int, nu: int):
    """Split stacked primal z -> [τ1, τ2, ..., τN]."""
    nprim = T*nx + (T-1)*nu
    taus, ofs = [], 0
    for _ in range(N):
        taus.append(z[ofs:ofs+nprim]); ofs += nprim
    return taus

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

def regular_polygon(center=(0.0, 0.0), radius=3.0, n=3, rotation_deg=0.0):
    """Generate CCW vertices of a regular n-gon in 2D."""
    cx, cy = center
    ang0 = np.deg2rad(rotation_deg)
    return [
        [cx + radius*np.cos(ang0 + 2*np.pi*k/n),
         cy + radius*np.sin(ang0 + 2*np.pi*k/n)]
        for k in range(n)
    ]

def polygon_halfspaces(vertices):
    """(2D) Convert polygon CCW vertices into A p ≤ b for keep-in."""
    V = np.asarray(vertices, float)
    n = len(V)
    area = 0.5 * np.sum(V[:,0]*np.roll(V[:,1], -1) - V[:,1]*np.roll(V[:,0], -1))
    ccw = area > 0
    A_rows, b_vals = [], []
    for i in range(n):
        p0 = V[i]; p1 = V[(i+1) % n]
        e = p1 - p0
        n_in = np.array([-e[1], e[0]]) if ccw else np.array([e[1], -e[0]])
        nrm = np.linalg.norm(n_in)
        if nrm > 0: n_in = n_in / nrm
        A_rows.append(n_in)
        b_vals.append(n_in @ p0)
    return np.asarray(A_rows, float), np.asarray(b_vals, float)

# -------------------- shared constraints builders --------------------
def build_g_tilde(nx: int, nu: int, T: int, N: int, D: int = 3, dt: float = 1.0):
    """
    g̃(τ,θ)=0: initial conditions + linear dynamics for each player.
    Returns closure g_fun(tau, theta).
    """
    A, B = double_integrator(D=D, dt=dt)
    nprim = T*nx + (T-1)*nu

    def g_fun(tau, theta):
        g_list = []; ofs = 0
        for p in range(N):
            tau_p = tau[ofs:ofs+nprim]; ofs += nprim
            xs, us = unpack_trajectory(tau_p, nx, nu, T)
            th_p = theta[p*nx:(p+1)*nx]
            g_list.append(xs[0] - th_p)  # IC
            for t in range(1, T):
                g_list.append(xs[t] - (A @ xs[t-1] + B @ us[t-1]))
        return ca.vcat(g_list)

    return g_fun

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

# in game_sharedutils.py
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


def fov_axis_from_vel(x_def, D: int, prev_axis=None, min_speed: float = 1e-3):
    """Choose look-axis from velocity (with hold when slow)."""
    v = np.asarray(x_def[D:2*D], float)
    if np.linalg.norm(v) >= float(min_speed):
        return _unit(v[:D])
    if prev_axis is not None and np.linalg.norm(prev_axis) > 0:
        return _unit(prev_axis)
    a = np.zeros(D); a[0] = 1.0
    return a

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

# --- NEW: MX helpers for quaternions and decoupled attitude constraints ---

def _R_of_q_mx(q):
    """3x3 DCM from unit quaternion q=[w,x,y,z] (MX, scalar-first)."""
    import casadi as ca
    w,x,y,z = q[0], q[1], q[2], q[3]
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    return ca.vertcat(
        ca.hcat([1-2*(yy+zz), 2*(xy - wz),   2*(xz + wy)]),
        ca.hcat([2*(xy + wz), 1-2*(xx+zz),   2*(yz - wx)]),
        ca.hcat([2*(xz - wy), 2*(yz + wx),   1-2*(xx+yy)])
    )

def _quat_mul_mx(q2, q1):
    """q_tot = q2 ⊗ q1 (MX)."""
    import casadi as ca
    w2,x2,y2,z2 = q2[0],q2[1],q2[2],q2[3]
    w1,x1,y1,z1 = q1[0],q1[1],q1[2],q1[3]
    return ca.vertcat(
        w2*w1 - x2*x1 - y2*y1 - z2*z1,
        w2*x1 + x2*w1 + y2*z1 - z2*y1,
        w2*y1 - x2*z1 + y2*w1 + z2*x1,
        w2*z1 + x2*y1 - y2*x1 + z2*w1
    )

def _quat_inc_from_w_mx(w_rel, dt):
    """Incremental quaternion δq from body rate (body wrt H) over dt (MX)."""
    import casadi as ca
    th = ca.norm_2(w_rel) * dt
    half = 0.5 * th
    c = ca.cos(half)
    s = ca.sin(half)
    denom = ca.fmax(1e-12, ca.norm_2(w_rel))
    u = w_rel / denom
    return ca.vertcat(c, s*u)

def build_g_tilde_tr_plus_quat(nx_tr, nu_tr, T, N, Ad_tr, Bd_tr, dt, J_np):
    """
    Mixed constraints for translation (linear) + attitude (nonlinear):
      - x_tr^+ = Ad_tr x_tr + Bd_tr u_tr
      - q^+    = normalize(q + 0.5*dt * Omega(w) q)
      - w^+    = w + dt * J^{-1}( tau - w×(J w) )
      - x_0    = θᵢ   (for each player i)
    Assumes per-player state x = [x_tr(nx_tr); q(4); w(3)],
                     input u = [u_tr(nu_tr); tau(3)].
    """
    Ad_tr = ca.MX(Ad_tr); Bd_tr = ca.MX(Bd_tr)
    Jinv  = ca.MX(np.linalg.inv(np.asarray(J_np, float)))

    nx = nx_tr + 7
    nu = nu_tr + 3
    nprim = T*nx + (T-1)*nu

    def g_fun(tau, theta):
        g_list = []
        ofs = 0
        for p in range(N):
            tau_p = tau[ofs:ofs+nprim]; ofs += nprim
            xs, us = unpack_trajectory(tau_p, nx, nu, T)  # CasADi slices

            # initial condition
            x0 = xs[0]
            th_p = theta[p*nx:(p+1)*nx]
            g_list.append(x0 - th_p)

            for t in range(1, T):
                x_prev = xs[t-1]; x_curr = xs[t]
                u_prev = us[t-1]

                # split
                xtr_prev = x_prev[0:nx_tr]
                xtr_curr = x_curr[0:nx_tr]
                utr_prev = u_prev[0:nu_tr]

                q_prev  = x_prev[nx_tr:nx_tr+4]
                w_prev  = x_prev[nx_tr+4:nx_tr+7]
                q_curr  = x_curr[nx_tr:nx_tr+4]
                w_curr  = x_curr[nx_tr+4:nx_tr+7]
                tau_prev = u_prev[nu_tr:nu_tr+3]

                # translation equality (linear)
                g_list.append(xtr_curr - (Ad_tr @ xtr_prev + Bd_tr @ utr_prev))

                # quaternion equality (Euler + renorm)
                q_pred = q_euler_step(q_prev, w_prev, dt)
                g_list.append(q_curr - q_pred)

                # rigid-body rotational dynamics (Euler's eq)
                Jw = ca.MX(J_np) @ w_prev
                wdot = Jinv @ (tau_prev - _skew_ca(w_prev) @ Jw)
                g_list.append(w_curr - (w_prev + dt*wdot))

                # (optional) hard unit-norm constraint (add if you prefer)
                # g_list.append(ca.dot(q_curr,q_curr) - 1.0)

        return ca.vcat(g_list)

    return g_fun


def build_g_tilde_tr_plus_quat_locked_boresight(
    nx_tr, nu_tr, T, N, Ad_tr, Bd_tr, dt, n, J_np, D=3, align="x",
    vmin=1e-3, gate_k=20.0
):
    """
    Quaternion attitude with boresight hard-coupled to v-hat:
      R_BH(q_t) e_align  = vhat(x_tr,t)    (gated near |v|≈0)
    State per agent: x = [x_tr(nx_tr), q(4), w(3)]
    Input per agent: u = [u_tr(nu_tr), tau(3)]
    Also enforces translation step, Euler rigid-body, quat kinematics, and ||q||=1.
    """
    import numpy as np
    import casadi as ca

    Ad_tr = ca.MX(Ad_tr); Bd_tr = ca.MX(Bd_tr)
    J = ca.MX(J_np if hasattr(J_np, "shape") else np.asarray(J_np, float))

    nx = nx_tr + 7
    nu = nu_tr + 3
    nprim = T*nx + (T-1)*nu

    ex_H   = ca.MX([1,0,0])
    wH_I_H = ca.MX([0,0,float(n)])
    e_alignB = ca.MX([1,0,0]) if align == "x" else ca.MX([0,0,1])

    def vhat_of_xtr(xtr):
        v = xtr[D:2*D]
        v3 = ca.vertcat(v[0], v[1], 0) if D == 2 else v
        nrm = ca.sqrt(ca.dot(v3, v3))
        return v3 / ca.fmax(nrm, 1e-6), nrm

    def g_fun(tau, theta):
        g_list = []
        ofs = 0
        for p in range(N):
            tau_p = tau[ofs:ofs+nprim]; ofs += nprim
            xs, us = unpack_trajectory(tau_p, nx, nu, T)
            th_p = theta[p*nx:(p+1)*nx]

            # t=0: IC, unit quaternion, alignment
            g_list.append(xs[0] - th_p)
            q0 = xs[0][nx_tr:nx_tr+4]
            g_list.append(ca.dot(q0, q0) - 1.0)
            xtr0 = xs[0][:nx_tr]
            vhat0, vnorm0 = vhat_of_xtr(xtr0)
            R0 = _R_of_q_mx(q0)
            xB0 = R0 @ e_alignB
            gate0 = ca.tanh(gate_k * (vnorm0 - vmin))**2
            g_list.append(gate0 * (xB0 - vhat0))

            for t in range(1, T):
                x_prev = xs[t-1]; x_curr = xs[t]; u_prev = us[t-1]

                # translation
                xtr_prev = x_prev[:nx_tr]; xtr_curr = x_curr[:nx_tr]
                utr_prev = u_prev[:nu_tr]
                g_list.append(xtr_curr - (Ad_tr @ xtr_prev + Bd_tr @ utr_prev))

                # attitude dynamics
                q_prev = x_prev[nx_tr:nx_tr+4]
                w_prev = x_prev[nx_tr+4:nx_tr+7]
                q_curr = x_curr[nx_tr:nx_tr+4]
                w_curr = x_curr[nx_tr+4:nx_tr+7]
                tau_prev = u_prev[nu_tr:nu_tr+3]

                R_BH = _R_of_q_mx(q_prev); R_HB = R_BH.T
                cB = R_HB @ ex_H
                tau_gg = 3.0*(n**2) * ca.cross(cB, J @ cB)

                rhs = tau_prev + tau_gg - ca.cross(w_prev, J @ w_prev)
                w_pred = w_prev + dt * ca.solve(J, rhs)
                g_list.append(w_curr - w_pred)

                w_rel = w_prev - (R_HB @ wH_I_H)
                dq = _quat_inc_from_w_mx(w_rel, dt)
                q_pred = _quat_mul_mx(dq, q_prev)
                g_list.append(q_curr - q_pred)
                g_list.append(ca.dot(q_curr, q_curr) - 1.0)

                # alignment at t
                xtr_t = x_curr[:nx_tr]
                vhat_t, vnorm_t = vhat_of_xtr(xtr_t)
                Rt = _R_of_q_mx(q_curr)
                xB_t = Rt @ e_alignB
                gate_t = ca.tanh(gate_k * (vnorm_t - vmin))**2
                g_list.append(gate_t * (xB_t - vhat_t))

        return ca.vcat(g_list)

    return g_fun


# ---------- Quaternion helpers ----------
def _skew_ca(w):
    wx, wy, wz = w[0], w[1], w[2]
    return ca.vertcat(
        ca.hcat([0,    -wz,  wy]),
        ca.hcat([wz,    0,  -wx]),
        ca.hcat([-wy,  wx,   0 ])
    )

def _Omega_of_w(w):
    wx, wy, wz = w[0], w[1], w[2]
    return ca.vertcat(
        ca.hcat([0,   -wx, -wy, -wz]),
        ca.hcat([wx,   0,   wz, -wy]),
        ca.hcat([wy,  -wz,  0,   wx]),
        ca.hcat([wz,   wy, -wx,  0 ])
    )

def q_normalize(q, eps=1e-12):
    return q / ca.sqrt(ca.dot(q, q) + eps)

def q_euler_step(q, w, dt):
    # q_{k+1} ≈ normalize(q_k + 0.5*dt*Omega(w_k)*q_k)
    return q_normalize(q + 0.5*dt * (_Omega_of_w(w) @ q))

def q_to_R(q):
    # q = [qw, qx, qy, qz] (scalar-first)
    import numpy as np
    if isinstance(q, ca.MX):
        qn = ca.DM(q) / ca.sqrt(ca.dot(q,q) + 1e-12)
        qw, qx, qy, qz = [qn[i] for i in range(4)]
        R = ca.vertcat(
            ca.hcat([1-2*(qy*qy+qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)]),
            ca.hcat([2*(qx*qy + qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz - qx*qw)]),
            ca.hcat([2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1-2*(qx*qx+qy*qy)])
        )
        return R
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    qw, qx, qy, qz = q
    R = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1-2*(qx*qx+qy*qy)]
    ], float)
    return R

def R_to_q(R):
    # scalar-first
    import numpy as np
    if isinstance(R, ca.MX):
        # convert via numeric path for simplicity (small overhead during seeding only)
        R = np.array(ca.DM(R)).astype(float)
    else:
        R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            qw = (R[2,1] - R[1,2]) / S
            qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S
            qz = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            qw = (R[0,2] - R[2,0]) / S
            qx = (R[0,1] + R[1,0]) / S
            qy = 0.25 * S
            qz = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            qw = (R[1,0] - R[0,1]) / S
            qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S
            qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], float)
    return q / np.linalg.norm(q)


def _bounds_3_from_cfg(att_cfg, prefix: str, default_max: float):
    """
    Returns (lb[3], ub[3]) from att_cfg for key family:
      - f"{prefix}_bounds": scalar, 2, 3, or 6 elements supported
      - f"{prefix}_max":    scalar or 3 vector (interpreted as symmetric ±)
    Examples:
      tau_bounds=0.05            -> [-0.05,0.05]*3
      tau_bounds=[-0.1,0.1]      -> [-0.1]*3, [0.1]*3
      tau_bounds=[0.02,0.03,0.04]-> [-.02,-.03,-.04], [+.02,+.03,+.04]
      tau_bounds=[-0.1,-0.2,-0.3, 0.1,0.2,0.3]
      tau_max=0.05               -> [-0.05,0.05]*3
      tau_max=[0.02,0.03,0.04]   -> [-.02,-.03,-.04], [+.02,+.03,+.04]
    """
    import numpy as np

    if f"{prefix}_bounds" in att_cfg:
        arr = np.asarray(att_cfg[f"{prefix}_bounds"], float).ravel()
        if arr.size == 1:
            lb = -float(arr[0]) * np.ones(3)
            ub =  float(arr[0]) * np.ones(3)
        elif arr.size == 2:
            lb = float(arr[0]) * np.ones(3)
            ub = float(arr[1]) * np.ones(3)
        elif arr.size == 3:
            lb = -arr.copy()
            ub =  arr.copy()
        elif arr.size == 6:
            lb = arr[:3].copy()
            ub = arr[3:6].copy()
        else:
            raise ValueError(f"{prefix}_bounds must have 1,2,3, or 6 elements")
        return lb, ub

    if f"{prefix}_max" in att_cfg:
        arr = np.asarray(att_cfg[f"{prefix}_max"], float).ravel()
        if arr.size == 1:
            lb = -float(arr[0]) * np.ones(3)
            ub =  float(arr[0]) * np.ones(3)
        elif arr.size == 3:
            lb = -arr.copy()
            ub =  arr.copy()
        else:
            raise ValueError(f"{prefix}_max must be scalar or length-3")
        return lb, ub

    # default symmetric
    return (-default_max*np.ones(3), default_max*np.ones(3))


def augment_bounds_with_quat(x_lb, x_ub, u_lb, u_ub, att_cfg: dict):
    """
    Append quaternion and body-rate bounds to state, and torque bounds to inputs.
    Translation bounds come from make_bounds(). We just extend them.

    State appends: q(4), w(3)
      - q is unit-norm, but box-bounded here to something safe, e.g., [-1.1,1.1].
      - w bounded by att_cfg (w_bounds or w_max), default ±1 rad/s.
    Input appends: τ(3)
      - τ bounded by att_cfg (tau_bounds or tau_max), default ±0.05 N·m.
    """
    import numpy as np

    x_lb = np.asarray(x_lb, float).ravel()
    x_ub = np.asarray(x_ub, float).ravel()
    u_lb = np.asarray(u_lb, float).ravel()
    u_ub = np.asarray(u_ub, float).ravel()

    # --- quaternion elementwise box (unit-norm is enforced via equality constraint elsewhere)
    q_min = float(att_cfg.get("q_min", -1.1))
    q_max = float(att_cfg.get("q_max",  1.1))
    q_lb  = np.full(4, q_min, float)
    q_ub  = np.full(4, q_max, float)

    # --- angular rate bounds
    w_lb, w_ub = _bounds_3_from_cfg(att_cfg, "w", default_max=1.0)  # rad/s

    # --- torque bounds
    tau_lb, tau_ub = _bounds_3_from_cfg(att_cfg, "tau", default_max=0.05)  # N·m

    x_lb_aug = np.r_[x_lb, q_lb, w_lb]
    x_ub_aug = np.r_[x_ub, q_ub, w_ub]
    u_lb_aug = np.r_[u_lb, tau_lb]
    u_ub_aug = np.r_[u_ub, tau_ub]

    # sanity
    assert x_lb_aug.shape == x_ub_aug.shape, f"x bounds shape mismatch: {x_lb_aug.shape} vs {x_ub_aug.shape}"
    assert u_lb_aug.shape == u_ub_aug.shape, f"u bounds shape mismatch: {u_lb_aug.shape} vs {u_ub_aug.shape}"

    return x_lb_aug, x_ub_aug, u_lb_aug, u_ub_aug


# --- block-diag for CasADi matrices (works for MX/DM mix) ---
def _blkdiag_mx(*mats):
    out = ca.DM([])
    for M in mats:
        out = ca.diagcat(out, M) if out.numel() else M
    return ca.MX(out)  # ensure MX

# --- small-angle attitude blocks (linear, discrete Euler) ---
def _att_blocks_linear_mx(mode: str, dt: float, J_like) -> tuple[ca.MX, ca.MX]:
    """
    mode: 'lin3d' (δθ=[δφ,δθ_y,δθ_z], δω=[..]; τ=[τx,τy,τz])
          'lin2d_roll' (same size; roll is separate 1D block + 2D yaw/pitch block)
    returns: (A_att 6x6, B_att 6x3) as MX
    """
    J = np.asarray(J_like, float)
    if J.ndim == 1:
        J = np.diag(J)
    Jinv = np.linalg.inv(J)

    I3 = np.eye(3)
    Z3 = np.zeros((3,3))

    if mode == "lin3d":
        A_att = np.block([[I3, dt*I3],
                          [Z3,     I3]])
        B_att = np.vstack([np.zeros((3,3)), dt*Jinv])
        return ca.MX(A_att), ca.MX(B_att)

    if mode == "lin2d_roll":
        # pitch/yaw -> use y,z axes; roll -> x axis
        # A_roll (2x2), B_roll (2x1)
        A_roll = np.array([[1.0, dt],
                           [0.0, 1.0]], float)
        B_roll = np.array([[0.0],
                           [dt/float(J[0,0])]], float)  # Jxx

        # A_yz (4x4), B_yz (4x2) for [θy,θz, ωy,ωz]
        I2 = np.eye(2); Z2 = np.zeros((2,2))
        A_yz = np.block([[I2, dt*I2],
                         [Z2,     I2]])
        Jyz_inv = np.linalg.inv(J[1:,1:])  # (Jyy,Jzz)
        B_yz = np.vstack([np.zeros((2,2)), dt*Jyz_inv])

        # Assemble in order [φ, θy, θz, ωx, ωy, ωz]
        A_att = np.block([
            [A_roll,                 np.zeros((2,4))],
            [np.zeros((4,2)),        A_yz          ],
        ])
        B_att = np.block([
            [B_roll,                 np.zeros((2,2))],
            [np.zeros((4,1)),        B_yz          ],
        ])
        return ca.MX(A_att), ca.MX(B_att)

    raise ValueError("att.mode must be 'lin3d' or 'lin2d_roll'")

# --- pad inequality bounds to translation-only (att free) ---
def _pad_trans_bounds_only(x_lb_tr, x_ub_tr, u_lb_tr, u_ub_tr, nx, nu, nx_tr, nu_tr):
    x_lb = np.full(nx, -np.inf); x_ub = np.full(nx,  np.inf)
    u_lb = np.full(nu, -np.inf); u_ub = np.full(nu,  np.inf)
    x_lb[:nx_tr] = x_lb_tr; x_ub[:nx_tr] = x_ub_tr
    u_lb[:nu_tr] = u_lb_tr; u_ub[:nu_tr] = u_ub_tr
    return x_lb, x_ub, u_lb, u_ub

# --- nominal boresight from velocity (works for D=2 or D=3) ---
def _axis_from_vel(x_tr: np.ndarray, D: int, align: str = "x", min_speed: float = 1e-3):
    v = np.asarray(x_tr[D:2*D], float)
    n = float(np.linalg.norm(v))
    if n <= min_speed:
        return np.array([1,0,0], float) if align == "x" else np.array([0,0,1], float)
    if D == 3:
        a = v / n
    else:
        a = np.array([v[0]/n, v[1]/n, 0.0], float)
    return a

# --- small-angle first-order correction: R ≈ R0 (I + [ε]x) ---
def _apply_small_angle(R0: np.ndarray, eps_xyz: np.ndarray) -> np.ndarray:
    ex, ey, ez = eps_xyz
    S = np.array([[ 0, -ez,  ey],
                  [ ez,  0, -ex],
                  [-ey,  ex,  0]], float)
    return R0 @ (np.eye(3) + S)

def transported_R_from_state(x, D: int, att_cfg: dict,
                             prev_axisD=None, prev_R=None):
    """
    Unified, continuous attitude constructor.
    Returns (R_wb, axisD_out, extra) where:
      - R_wb: 3x3 rows [x_b; y_b; z_b] in WORLD coords
      - axisD_out: the (D-dim) velocity-based boresight we used (for hysteresis)
      - extra: dict with model-specific extras (e.g., {'phi': ...})
    Supports:
      - att.model in {'roll1','roll2','roll2_tau'}   -> transport boresight + roll
      - att.model == 'lin3d'                          -> transport boresight + small-angle δθ
    """
    model    = (att_cfg.get("model") or "roll1").lower()
    align    = att_cfg.get("align", "x")
    world_up = np.asarray(att_cfg.get("up", [0,0,1]), float)
    vmin     = float(att_cfg.get("min_speed_for_axis",
                                 att_cfg.get("vmin_for_axis", 1e-3)))

    # figure out where attitude state starts
    nx_tr, _ = dims_from_D(D)
    axisD = fov_axis_from_vel(x, D, prev_axis=prev_axisD, min_speed=vmin)
    axis3 = axisD if D==3 else np.array([axisD[0], axisD[1], 0.0], float)

    # 1) always transport the boresight first (minimal spin, no flips)
    R_axis = frame_from_axis_continuous(axis3, R_prev=prev_R,
                                        align=align, world_up=world_up)

    if model in ("roll1", "roll2", "roll2_tau"):
        # roll is the first att state in both roll1 and roll2(_tau)
        phi = float(np.asarray(x, float)[nx_tr])
        R_wb = apply_roll_about_axis(R_axis, phi, align=align)
        return R_wb, axisD, {"phi": phi}

    if model == "lin3d":
        # x_att = [δθ(3), ω(3)]
        dtheta = np.asarray(x, float)[nx_tr:nx_tr+3]
        R_wb   = _apply_small_angle(R_axis, dtheta)  # small-angle about body axes
        return R_wb, axisD, {"dtheta": dtheta}

    # Fallback: just return the transported boresight
    return R_axis, axisD, {}
