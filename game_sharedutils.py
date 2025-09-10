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


def frame_from_axis(axis, align="z", world_up=(0,0,1)):
    """
    Build body frame rows [x_b; y_b; z_b] given desired axis alignment.
    align='z' -> z_b aligned to axis (current default)
    align='x' -> x_b aligned to axis
    """
    axis = _unit(axis)
    if align == "z":
        return world_to_body_R(axis, 3)
    # align x_b to axis: construct like world_to_body_R but swap roles
    ref = np.array([1,0,0], float) if abs(axis[0]) < 0.9 else np.array([0,1,0], float)
    z_b = _unit(np.cross(ref, axis))
    if np.linalg.norm(z_b) < 1e-9:   # fallback if parallel
        ref = np.array([0,0,1.0])
        z_b = _unit(np.cross(ref, axis))
    y_b = _unit(np.cross(z_b, axis))
    x_b = axis
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
    """
    Build augmented (Ad,Bd) with roll appended after translation state/control.
    Returns (Ad_aug_mx, Bd_aug_mx, idx) where idx contains useful indices.
    """
    # translation sizes from inputs
    Ad_tr = as_numpy_const(Ad_tr_mx)
    Bd_tr = as_numpy_const(Bd_tr_mx)
    nx_tr = Ad_tr.shape[0]
    nu_tr = Bd_tr.shape[1]

    # attitude blocks
    A_att, B_att = att_roll_AB(dt, att_cfg)
    nxa, nua, model = att_roll_dims(att_cfg)

    # augmented A: block-diagonal
    Ad_aug = blkdiag2D(Ad_tr, A_att)

    # augmented B: block "diagonal" (top-left Bd_tr, bottom-right B_att)
    Bd_aug = np.block([
        [Bd_tr,                        np.zeros((nx_tr, nua))],
        [np.zeros((nxa, nu_tr)),       B_att                  ],
    ])

    # indices in augmented vectors
    idx = dict(
        nx_tr=nx_tr, nu_tr=nu_tr,
        nx=nx_tr + nxa, nu=nu_tr + nua,
        i_phi=nx_tr,           # phi is first of attitude states
        i_w=nx_tr+1 if model=="roll2" else None,
        i_u_roll=nu_tr         # last block of controls is attitude
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
    nxa, nua, model = att_roll_dims(att_cfg)

    # --- state bounds ---
    phi_min = float(att_cfg.get("phi_min", -np.pi))
    phi_max = float(att_cfg.get("phi_max",  np.pi))

    if model == "roll1":
        x_lb_att = np.array([phi_min])
        x_ub_att = np.array([phi_max])
    else:
        w_max = float(att_cfg.get("w_max", 0.5))  # rad/s
        x_lb_att = np.array([phi_min, -w_max])
        x_ub_att = np.array([phi_max,  w_max])

    # --- input bounds ---
    if model == "roll1":
        omg_max = float(att_cfg.get("omega_max", 0.0))   # 0.0 keeps roll constant
        u_lb_att = np.array([-omg_max])
        u_ub_att = np.array([ omg_max])
    else:
        a_max = float(att_cfg.get("alpha_max", 0.0))     # 0.0 keeps roll const.
        u_lb_att = np.array([-a_max])
        u_ub_att = np.array([ a_max])

    return (np.r_[x_lb, x_lb_att], np.r_[x_ub, x_ub_att],
            np.r_[u_lb, u_lb_att], np.r_[u_ub, u_ub_att])

def pad_x0_with_att(x0_row: np.ndarray, att_cfg: dict, D: int):
    """
    Ensure x0 has (translation + attitude) length.
    If attitude entries are missing, append from att_cfg: phi0 (and w0).
    """
    nx_tr, _ = dims_from_D(D)
    nxa, _, model = att_roll_dims(att_cfg)
    x0 = np.asarray(x0_row, float).copy()
    need = nx_tr + nxa
    if x0.size >= need:
        return x0[:need]
    # append attitude initials
    phi0 = float(att_cfg.get("phi0", 0.0))
    if model == "roll1":
        x0_att = np.array([phi0])
    else:
        w0 = float(att_cfg.get("w0", 0.0))
        x0_att = np.array([phi0, w0])
    return np.r_[x0[:nx_tr], x0_att]


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

def frame_from_axis_continuous(a_cur, R_prev=None, world_up=np.array([0,0,1.0])):
    """
    Build/propagate a continuous body frame with z_b aligned to a_cur.
    If R_prev is given, apply minimal rotation; else build from world_up.
    Returns 3x3 with rows [x_b; y_b; z_b].
    """
    a_cur = _unit(a_cur)
    if R_prev is not None:
        z_prev = R_prev[2]
        Rdel = minimal_rotation(z_prev, a_cur)
        R = Rdel @ R_prev
        # re-orthonormalize
        z_b = _unit(R[2])
        x_b = _unit(R[0] - z_b*np.dot(R[0], z_b))
        y_b = _unit(np.cross(z_b, x_b))
        return np.vstack([x_b, y_b, z_b])
    # first frame
    if abs(np.dot(a_cur, world_up)) > 0.98:
        world_up = np.array([1,0,0], float)
    x_b = _unit(np.cross(world_up, a_cur))
    y_b = _unit(np.cross(a_cur, x_b))
    z_b = a_cur
    return np.vstack([x_b, y_b, z_b])

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