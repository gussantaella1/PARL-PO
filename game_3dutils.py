# game_3dutils.py
# 3D/2D utilities + RHC entrypoints using the new Julia-style ParametricGame stack.
from __future__ import annotations

import numpy as np
import casadi as ca

# Optional viz deps (safe if you don't call plotting helpers)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401

# Optional filters (wrapped behind a small façade)
try:
    from ukf_estimator import AgentUKF  # noqa: F401
except Exception:
    AgentUKF = None
try:
    from ekf_estimator import AgentEKF  # noqa: F401
except Exception:
    AgentEKF = None

# NEW: Julia-style game stack (PYOMO-based MCP)
from pyomo.environ import sqrt  # for py_norm2
from parametric_game import ParametricGame, ParametricGameSpec

__all__ = [
    "run_rhc_and_collect_frames_3d",
    "animate_rollout_3d",
    "interactive_rollout_3d",
    "world_to_body_R",
    "_unit",
    "apply_roll_about_axis",
]

# ---------------------------------------------------------------------------
# Small convenience: tolerant import reloader for live tuning (optional)
def _maybe_reload(mod):
    try:
        import importlib
        importlib.reload(mod)
    except Exception:
        pass
# ---------------------------------------------------------------------------

# ---- Pyomo algebra helpers ----
def py_sumsqr(vec): return sum(v*v for v in vec)
def py_dot(a,b):    return sum(ai*bi for ai,bi in zip(a,b))
def py_norm2(vec, eps=0.0): return sqrt(py_sumsqr(vec) + eps)

def py_unpack_tau(tau, nx, nu, T):
    X = tau[:T*nx]; U = tau[T*nx:]
    xs = [X[t*nx:(t+1)*nx] for t in range(T)]
    us = [U[t*nu:(t+1)*nu] for t in range(T-1)]
    return xs, us

# -------------------- dims & basic trajectory utils --------------------
def dims_from_D(D: int):
    assert D in (2, 3), "Only D=2 or D=3 supported."
    return 2 * D, D  # nx, nu  (double-integrator: x=[p(1..D), v(1..D)], u=[a(1..D)])

def pack_trajectory(X: np.ndarray, U: np.ndarray):
    return np.concatenate([X.reshape(-1), U.reshape(-1)])

def unpack_trajectory(tau: np.ndarray, nx: int, nu: int, T: int):
    X_flat = tau[: T * nx]
    U_flat = tau[T * nx :]
    X = X_flat.reshape(T, nx)
    U = U_flat.reshape(T - 1, nu)
    return X, U

def rollout_linear(Ad: np.ndarray, Bd: np.ndarray, x0: np.ndarray, U: np.ndarray):
    Tm1, nu = U.shape
    nx = Ad.shape[0]
    X = np.zeros((Tm1 + 1, nx), float)
    X[0] = x0
    for k in range(Tm1):
        X[k + 1] = Ad @ X[k] + Bd @ U[k]
    return X

def di_discrete_AB(dt: float, D: int = 2):
    nx, nu = dims_from_D(D)
    A = np.eye(nx)
    A[0:D, D : 2 * D] = dt * np.eye(D)
    B = np.zeros((nx, nu))
    B[D : 2 * D, :] = dt * np.eye(D)
    return A.astype(float), B.astype(float)

# -------------------- CasADi helpers & orbital dynamics --------------------
def as_numpy_const(M):
    """Robustly turn constant CasADi matrices (MX/SX/DM) or array-likes into numpy float arrays."""
    if isinstance(M, np.ndarray):
        return M.astype(float)
    try:
        return np.array(M.full(), dtype=float)
    except Exception:
        pass
    f = ca.Function("const_mat", [], [ca.MX(M)])
    out = f()
    if isinstance(out, dict):
        out = next(iter(out.values()))
    elif isinstance(out, (list, tuple)):
        out = out[0]
    try:
        return np.array(out.full(), dtype=float)
    except Exception:
        return np.array(out, dtype=float)

def hcw_mean_motion(hcw_cfg: dict):
    if "n" in hcw_cfg:
        return float(hcw_cfg["n"])
    mu = float(hcw_cfg.get("mu", 3.986004418e14))
    r0 = float(hcw_cfg["r0"])
    return float(np.sqrt(mu / (r0**3)))

def _discretize_linear(Ac, Bc, dt, series_terms=18):
    nx = int(Ac.shape[0]); nu = int(Bc.shape[1])
    Ac_np = as_numpy_const(Ac); Bc_np = as_numpy_const(Bc)
    M = np.block([[Ac_np, Bc_np], [np.zeros((nu, nx)), np.zeros((nu, nu))]])
    try:
        from scipy.linalg import expm
        E = expm(M * dt)
    except Exception:
        E = np.eye(nx + nu)
        term = np.eye(nx + nu)
        Mdt = M * dt
        for k in range(1, series_terms + 1):
            term = term @ (Mdt / k)
            E += term
    Ad_np = E[:nx, :nx]
    Bd_np = E[:nx, nx : nx + nu]
    return ca.MX(Ad_np), ca.MX(Bd_np)

def hcw_discrete_mats(n: float, dt: float, D: int = 3):
    n = float(n)
    if D == 2:
        Ac = ca.MX.zeros(4, 4)
        Ac[0, 2] = 1.0
        Ac[1, 3] = 1.0
        Ac[2, 0] = 3 * n * n
        Ac[2, 3] = 2 * n
        Ac[3, 2] = -2 * n
        Bc = ca.MX.zeros(4, 2)
        Bc[2, 0] = 1.0
        Bc[3, 1] = 1.0
        return _discretize_linear(Ac, Bc, dt)

    Ac = ca.MX.zeros(6, 6)
    Ac[0, 3] = 1.0
    Ac[1, 4] = 1.0
    Ac[2, 5] = 1.0
    Ac[3, 0] = 3 * n * n
    Ac[3, 4] = 2 * n
    Ac[4, 3] = -2 * n
    Ac[5, 2] = -n * n
    Bc = ca.MX.zeros(6, 3)
    Bc[3, 0] = 1.0
    Bc[4, 1] = 1.0
    Bc[5, 2] = 1.0
    return _discretize_linear(Ac, Bc, dt)

# -------------------- PYOMO-native MCP builders (objectives/constraints) --------------------


def make_pyomo_g_tilde(nx, nu, T, D, Ad_np, Bd_np):
    """Shared equalities: IC + linear dynamics (as Pyomo expressions)."""
    def g_tilde(tau_all, theta):
        nprim = T*nx + (T-1)*nu
        g = []
        for p in range(2):
            tau = tau_all[p*nprim:(p+1)*nprim]
            xs, us = py_unpack_tau(tau, nx, nu, T)
            th = theta[p*nx:(p+1)*nx]
            # IC
            for i in range(nx):
                g.append(xs[0][i] - th[i])
            # dynamics
            for t in range(1, T):
                Ax = [sum(Ad_np[i, j]*xs[t-1][j] for j in range(nx)) for i in range(nx)]
                Bu = [sum(Bd_np[i, j]*us[t-1][j] for j in range(nu)) for i in range(nx)]
                for i in range(nx):
                    g.append(xs[t][i] - (Ax[i] + Bu[i]))
        return g
    return g_tilde

def make_pyomo_h_tilde(nx, nu, T, D, cfg):
    """Shared inequalities: box bounds + arena + keep-out + pairwise sep (as Pyomo expressions)."""
    x_lb, x_ub, u_lb, u_ub = make_bounds(cfg)
    arena = cfg["arena"]; artype = arena.get("type", "box")
    sep2 = float(cfg.get("sep_min", cfg.get("min_sep", 0.0)))**2
    sphere_key = "circles" if D == 2 else "spheres"
    polyA = polyb = None
    if artype in ("polygon", "polyhedron"):
        polyA = np.asarray(arena["A"], float)
        polyb = np.asarray(arena["b"], float)

    def h_tilde(tau_all, theta):
        nprim = T*nx + (T-1)*nu
        H = []
        taus = []
        for p in range(2):
            tau = tau_all[p*nprim:(p+1)*nprim]
            xs, us = py_unpack_tau(tau, nx, nu, T)
            taus.append((xs, us))
            # bounds
            for t in range(T):
                if x_lb is not None:
                    for i in range(nx): H.append(xs[t][i] - x_lb[i])
                if x_ub is not None:
                    for i in range(nx): H.append(x_ub[i] - xs[t][i])
            for t in range(T-1):
                if u_lb is not None:
                    for j in range(nu): H.append(us[t][j] - u_lb[j])
                if u_ub is not None:
                    for j in range(nu): H.append(u_ub[j] - us[t][j])

            # arena keep-in
            if artype in ("circle", "sphere"):
                c = np.array([arena.get(k, 0.0) for k in (["cx","cy"] if D==2 else ["cx","cy","cz"])], float)
                r2 = float(arena["r"])**2
                for t in range(T):
                    p = xs[t][0:D]
                    H.append(r2 - sum((p[j]-c[j])*(p[j]-c[j]) for j in range(D)))
            elif artype in ("polygon", "polyhedron"):
                useD = 2 if D==2 else 3
                for t in range(T):
                    p = xs[t][0:useD]
                    for r in range(polyA.shape[0]):
                        H.append(polyb[r] - sum(polyA[r, j]*p[j] for j in range(useD)))

            # keep-out spheres/circles
            for s in cfg.get(sphere_key, []):
                o = np.array([s.get(k, 0.0) for k in (["cx","cy"] if D==2 else ["cx","cy","cz"])], float)
                r2 = float(s["r"])**2
                for t in range(T):
                    p = xs[t][0:D]
                    H.append(sum((p[j]-o[j])*(p[j]-o[j]) for j in range(D)) - r2)

        # pairwise separation
        (xs1, _), (xs2, _) = taus[0], taus[1]
        for t in range(T):
            p1 = xs1[t][0:D]; p2 = xs2[t][0:D]
            H.append(sum((p1[j]-p2[j])*(p1[j]-p2[j]) for j in range(D)) - sep2)

        return H
    return h_tilde

# -------------------- bounds & attitude augmentation --------------------
def make_bounds(cfg: dict):
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx, nu = dims_from_D(D)
    vmax = float(cfg.get("vmax", 1.0))
    umax = float(cfg.get("umax", 1.0))
    ar = cfg["arena"]
    ar_type = ar.get("type", "box")

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
        p_lb = -BIG*np.ones(D, float)
        p_ub =  BIG*np.ones(D, float)

    v_lb = -vmax*np.ones(D, float)
    v_ub =  vmax*np.ones(D, float)
    x_lb = np.r_[p_lb, v_lb]
    x_ub = np.r_[p_ub, v_ub]
    return x_lb, x_ub, u_lb, u_ub

def augment_AB_for_att(Ad_tr, Bd_tr, dt, att_cfg):
    Ad_tr = ca.MX(Ad_tr)
    Bd_tr = ca.MX(Bd_tr)
    nx_tr = Ad_tr.size1()
    nu_tr = Bd_tr.size2()
    n_att = 3
    n_ctrl = 3

    Ad_aug = ca.MX.zeros(nx_tr + n_att, nx_tr + n_att)
    Ad_aug[:nx_tr, :nx_tr] = Ad_tr
    Ad_aug[nx_tr:, nx_tr:] = ca.MX_eye(n_att)

    Bd_aug = ca.MX.zeros(nx_tr + n_att, nu_tr + n_ctrl)
    Bd_aug[:nx_tr, :nu_tr] = Bd_tr
    Bd_aug[nx_tr:, nu_tr:] = dt * ca.MX_eye(n_att)

    idx = {
        "nx": nx_tr + n_att,
        "nu": nu_tr + n_ctrl,
        "i_phi":   nx_tr + 0,
        "i_theta": nx_tr + 1,
        "i_psi":   nx_tr + 2,
        "i_u_phi":   nu_tr + 0,
        "i_u_theta": nu_tr + 1,
        "i_u_psi":   nu_tr + 2,
    }
    return Ad_aug, Bd_aug, idx

def augment_bounds_with_att(x_lb, x_ub, u_lb, u_ub, att_cfg):
    x_lb = np.asarray(x_lb, float).ravel()
    x_ub = np.asarray(x_ub, float).ravel()
    u_lb = np.asarray(u_lb, float).ravel()
    u_ub = np.asarray(u_ub, float).ravel()

    phi_lim   = att_cfg.get("phi_lim",   (-np.pi, np.pi))
    theta_lim = att_cfg.get("theta_lim", (-np.pi/2, np.pi/2))
    psi_lim   = att_cfg.get("psi_lim",   (-np.pi, np.pi))

    phi_dot_lim   = att_cfg.get("phi_dot_lim",   (-1.0, 1.0))
    theta_dot_lim = att_cfg.get("theta_dot_lim", (-1.0, 1.0))
    psi_dot_lim   = att_cfg.get("psi_dot_lim",   (-1.0, 1.0))

    x_lb = np.concatenate([x_lb, [phi_lim[0], theta_lim[0], psi_lim[0]]])
    x_ub = np.concatenate([x_ub, [phi_lim[1], theta_lim[1], psi_lim[1]]])

    u_lb = np.concatenate([u_lb, [phi_dot_lim[0], theta_dot_lim[0], psi_dot_lim[0]]])
    u_ub = np.concatenate([u_ub, [phi_dot_lim[1], theta_dot_lim[1], psi_dot_lim[1]]])
    return x_lb, x_ub, u_lb, u_ub  # <-- fixed (was returning u_ub twice)

def pad_x0_with_att(x0_row, att_cfg, D):
    x0_row = np.asarray(x0_row, float).ravel()
    phi0   = att_cfg.get("phi0", 0.0)
    theta0 = att_cfg.get("theta0", 0.0)
    psi0   = att_cfg.get("psi0", 0.0)
    return np.concatenate([x0_row, [phi0, theta0, psi0]])

# -------------------- attitude & FOV helpers --------------------
def _unit(v, eps: float = 1e-12):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / (n + eps)

def _skew(w):
    wx, wy, wz = w
    return np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]], float)

def minimal_rotation(a, b, eps: float = 1e-9):
    a = _unit(a); b = _unit(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v); c = float(np.dot(a, b))
    if s < eps:
        if c > 0:
            return np.eye(3)
        axis = _unit(np.cross(a, np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])))
        K = _skew(axis)
        return np.eye(3) + 2 * (K @ K)
    axis = v / s
    K = _skew(axis)
    angle = np.arctan2(s, c)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def world_to_body_R(axis, D: int = 3, align: str = "x", up=(0.0, 0.0, 1.0)):
    axis = _unit(axis)
    up = _unit(np.asarray(up, float))
    if D == 2:
        x_b = axis
        y_b = np.array([-axis[1], axis[0]])
        return np.vstack([x_b, y_b])
    if align == "x":
        x_b = axis
        if abs(np.dot(x_b, up)) > 0.98:
            up = np.array([0.0, 1.0, 0.0]) if abs(x_b[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        y_b = _unit(np.cross(up, x_b))
        z_b = _unit(np.cross(x_b, y_b))
        return np.vstack([x_b, y_b, z_b])
    z_b = axis
    ref = np.array([1.0, 0.0, 0.0]) if abs(z_b[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_b = _unit(np.cross(ref, z_b))
    y_b = _unit(np.cross(z_b, x_b))
    return np.vstack([x_b, y_b, z_b])

def frame_from_axis_continuous(axis, R_prev=None, align: str = "x", world_up=np.array([0.0, 0.0, 1.0])):
    a = _unit(np.asarray(axis, float))
    up = _unit(np.asarray(world_up, float))
    if R_prev is not None:
        if align == "x":
            x_prev = _unit(R_prev[0])
            Rdel = minimal_rotation(x_prev, a)
            R = Rdel @ R_prev
            x_b = a
            y_tmp = R[1] - x_b * np.dot(R[1], x_b)
            if np.linalg.norm(y_tmp) < 1e-8:
                ref = up if abs(np.dot(up, x_b)) < 0.98 else np.array([0, 1, 0], float)
                y_tmp = np.cross(ref, x_b)
            y_b = _unit(y_tmp)
            z_b = _unit(np.cross(x_b, y_b))
            y_b = _unit(np.cross(z_b, x_b))
            return np.vstack([x_b, y_b, z_b])
        z_prev = _unit(R_prev[2])
        Rdel = minimal_rotation(z_prev, a)
        R = Rdel @ R_prev
        z_b = a
        x_tmp = R[0] - z_b * np.dot(R[0], z_b)
        if np.linalg.norm(x_tmp) < 1e-8:
            ref = up if abs(np.dot(up, z_b)) < 0.98 else np.array([1, 0, 0], float)
            x_tmp = np.cross(ref, z_b)
        x_b = _unit(x_tmp)
        y_b = _unit(np.cross(z_b, x_b))
        x_b = _unit(np.cross(y_b, z_b))
        return np.vstack([x_b, y_b, z_b])
    if align == "x":
        if abs(np.dot(a, up)) > 0.98:
            up = np.array([0, 1, 0], float) if abs(a[1]) < 0.9 else np.array([1, 0, 0], float)
        y_b = _unit(np.cross(up, a))
        z_b = _unit(np.cross(a, y_b))
        return np.vstack([a, y_b, z_b])
    if abs(np.dot(a, up)) > 0.98:
        up = np.array([1, 0, 0], float)
    x_b = _unit(np.cross(up, a))
    y_b = _unit(np.cross(a, x_b))
    return np.vstack([x_b, y_b, a])

def apply_roll_about_axis(R_wb, phi: float, align: str = "x"):
    x_b, y_b, z_b = R_wb[0], R_wb[1], R_wb[2]
    c, s = np.cos(phi), np.sin(phi)
    if align == "x":
        y_p = c*y_b + s*z_b
        z_p = -s*y_b + c*z_b
        return np.vstack([x_b, y_p, z_p])
    x_p = c*x_b + s*y_b
    y_p = -s*x_b + c*y_b
    return np.vstack([x_p, y_p, z_b])

def in_fov(p_t, x_def, axis, fov_cfg, D: int):
    p_def = np.asarray(x_def[:D], float)
    rel = np.asarray(p_t[:D], float) - p_def
    dist = float(np.linalg.norm(rel))
    if dist > float(fov_cfg["range"]):
        return False, dist
    half = 0.5 * np.deg2rad(float(fov_cfg["hfov_deg"]))
    cosang = float(np.dot(_unit(rel), _unit(axis)))
    return (cosang >= np.cos(half)), dist

def project_point_pinhole(X_w, x_def, cam_cfg, axis=None, R_wb=None):
    p_cam = np.asarray(x_def[:3], float)
    align = cam_cfg.get("align", "z")
    R_wc = np.asarray(R_wb, float) if R_wb is not None else world_to_body_R(_unit(np.asarray(axis, float)), 3, align=align)
    X_c = R_wc @ (np.asarray(X_w, float) - p_cam)
    if align == "z":
        depth = X_c[2]
        if depth <= 0:
            return None, None, depth, False
        u = cam_cfg["fx"] * (X_c[0] / depth) + cam_cfg["cx"]
        v = cam_cfg["fy"] * (X_c[1] / depth) + cam_cfg["cy"]
    else:
        depth = X_c[0]
        if depth <= 0:
            return None, None, depth, False
        u = cam_cfg["fx"] * (X_c[1] / depth) + cam_cfg["cx"]
        v = cam_cfg["fy"] * (X_c[2] / depth) + cam_cfg["cy"]
    W, H = cam_cfg["W"], cam_cfg["H"]
    near, far = cam_cfg["near"], cam_cfg["far"]
    in_front = (depth >= near) and (depth <= far)
    in_pixels = (0 <= u < W) and (0 <= v < H)
    return float(u), float(v), float(depth), (in_front and in_pixels)

# -------------------- small UKF/EKF façade --------------------
class KF_CV:
    """
    Common interface over AgentUKF / AgentEKF.
    ctor: KF_CV(x0, P0, Q, R, dt, kind='auto'|'ukf'|'ekf', **sigma)
    """
    def __init__(self, x0, P0, Q, R, dt, kind='auto', **kwargs):
        kind = (kind or 'auto').lower()
        impl = None
        if kind == 'ekf' and AgentEKF is not None:
            impl = AgentEKF(x0, P0, Q, R, dt)
        elif kind in ('ukf', 'auto') and AgentUKF is not None:
            sp = {k: kwargs[k] for k in ('alpha', 'beta', 'kappa') if k in kwargs}
            impl = AgentUKF(x0, P0, Q, R, dt, **sp)
        elif AgentUKF is not None:
            impl = AgentUKF(x0, P0, Q, R, dt)
        elif AgentEKF is not None:
            impl = AgentEKF(x0, P0, Q, R, dt)
        else:
            raise RuntimeError("Neither AgentUKF nor AgentEKF is importable.")
        self._impl = impl

    def __getattr__(self, name):
        return getattr(self._impl, name)

    def predict(self, dt=None, u=None, **kwargs):
        f = getattr(self._impl, 'predict')
        import inspect
        sig = inspect.signature(f)
        kw = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return f(dt=dt, u=u, **kw)

    def update(self, z, p_obs, R_wb, **kwargs):
        f = getattr(self._impl, 'update')
        import inspect
        sig = inspect.signature(f)
        kw = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return f(z, p_obs=p_obs, R_wb=R_wb, **kw)

# ======================================================================
# ==================  RHC main entrypoint (NEW STACK)  ==================
# ======================================================================
def run_rhc_and_collect_frames_3d(cfg: dict, cost_builder=None,
                                  steps: int | None = None, turn_len: int | None = None):
    """
    Receding-horizon rollout using a Pyomo-MCP ParametricGame for the translational states only.
    Attitude/FOV are derived for viz; NOT included in the MCP.
    """
    # --- basic dims & horizon ---
    N = 2
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx_tr, nu_tr = dims_from_D(D)
    T, dt = int(cfg["T"]), float(cfg["dt"])

    # === dynamics (TRANSLATIONAL ONLY) ===
    dyn = (cfg.get("dynamics") or "double").lower()
    if dyn == "hcw":
        n = hcw_mean_motion(cfg.get("hcw", {}))
        Ad_mx, Bd_mx = hcw_discrete_mats(n, dt, D=D)
    else:
        Ad_np0, Bd_np0 = di_discrete_AB(dt, D=D)
        Ad_mx, Bd_mx = ca.MX(Ad_np0), ca.MX(Bd_np0)

    # NO attitude in MCP
    nx, nu = nx_tr, nu_tr

    # rollout & turn
    sim_time = cfg.get("sim_time", cfg.get("max_time", cfg.get("duration", None)))
    if steps is None:
        steps = max(1, int(np.ceil(float(sim_time)/dt))) if sim_time is not None else int(cfg.get("steps", 60))
    if turn_len is None:
        turn_len = int(cfg.get("turn_len", 3)) if "turn_seconds" not in cfg else \
                   max(1, int(round(float(cfg["turn_seconds"]) / float(dt))))

    # bounds (TRANSLATIONAL ONLY)
    x_lb, x_ub, u_lb, u_ub = make_bounds(cfg)

    # initial states (TRANSLATIONAL ONLY)
    x1 = np.asarray(cfg["x0"][0], float)[:nx].copy()
    x2 = np.asarray(cfg["x0"][1], float)[:nx].copy()

    # numeric stepper
    Ad_np, Bd_np = as_numpy_const(Ad_mx), as_numpy_const(Bd_mx)
    def step_plant(x, u):
        return Ad_np @ np.asarray(x, float) + Bd_np @ np.asarray(u, float)

    # logs
    plan_hist1, plan_hist2 = [], []
    plan_att1, plan_att2 = [], []
    exec_xyz1, exec_xyz2 = [], []
    exec_att1, exec_att2 = [], []
    phi_hist1, phi_hist2 = [], []
    fov_axis_hist, fov_seen_mask = [], []

    # FOV/attitude config (viz only)
    att_cfg   = cfg.get("att", {})
    align     = att_cfg.get("align", "x")
    world_up  = att_cfg.get("up", [0, 0, 1])
    vmin0     = float(att_cfg.get("min_speed_for_axis", 1e-3))
    fov_cfg   = cfg.get("fov", {"enabled": False})
    fov_on    = bool(fov_cfg.get("enabled", False))
    fov_agent = int(fov_cfg.get("agent", 2))

    def _p3_row(x_row):
        return (float(x_row[0]), float(x_row[1]), float(x_row[2])) if D==3 else (float(x_row[0]), float(x_row[1]), 0.0)

    def _axis_from_vel(x):
        v = np.asarray(x[D:2*D], float); n = float(np.linalg.norm(v))
        aD = (v/n) if n > vmin0 else (np.array([1,0,0]) if align=="x" else np.array([0,0,1]))[:D]
        return aD, (aD if D==3 else np.array([aD[0], aD[1], 0.0], float))

    def plan_attitudes_from_X(X, prev_axisD, prev_R):
        att_list = []; ax_prev = prev_axisD; R_prev = prev_R
        for t in range(T):
            v = np.asarray(X[t, D:2*D], float); n = np.linalg.norm(v)
            axisD = v/n if n > 1e-3 else ax_prev
            axis3 = axisD if D==3 else np.array([axisD[0], axisD[1], 0.0])
            R = frame_from_axis_continuous(axis3, R_prev=R_prev, align=align, world_up=np.asarray(world_up,float))
            phi = 0.0  # no attitude state in MCP
            att_list.append({"R": R, "phi": phi})
            ax_prev, R_prev = axisD, R
        return att_list, ax_prev, R_prev

    # t=0 frames
    prev_axis1D, axis1_0 = _axis_from_vel(x1)
    prev_axis2D, axis2_0 = _axis_from_vel(x2)
    R1_0 = world_to_body_R(axis1_0, 3, align=align, up=world_up)
    R2_0 = world_to_body_R(axis2_0, 3, align=align, up=world_up)
    prev_R1, prev_R2 = R1_0, R2_0
    exec_xyz1.append(_p3_row(x1)); exec_xyz2.append(_p3_row(x2))
    exec_att1.append({"R": R1_0, "phi": 0.0}); phi_hist1.append(0.0)
    exec_att2.append({"R": R2_0, "phi": 0.0}); phi_hist2.append(0.0)

    if fov_on:
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

    # === Build the ParametricGame (TRANSLATIONAL ONLY, PYOMO-NATIVE) ===
    nprim    = T*nx + (T-1)*nu
    Ad_np_s  = as_numpy_const(Ad_mx)
    Bd_np_s  = as_numpy_const(Bd_mx)

    f_list  = make_pyomo_objectives(nx=nx, nu=nu, T=T, D=D, cfg=cfg)
    g_tilde  = make_pyomo_g_tilde(nx, nu, T, D, Ad_np=Ad_np_s, Bd_np=Bd_np_s)
    h_tilde  = make_pyomo_h_tilde(nx, nu, T, D, cfg)

    # determine shared inequality count once using dummy numbers
    dummy_tau   = [0.0]*(2*nprim)
    dummy_theta = [0.0]*(2*nx)
    shared_ineq = len(h_tilde(dummy_tau, dummy_theta))

    spec = ParametricGameSpec(
        objectives=f_list,
        indiv_equalities=[lambda tau, theta, i=0: [], lambda tau, theta, i=1: []],
        indiv_inequalities=[lambda tau, theta, i=0: [], lambda tau, theta, i=1: []],
        shared_equality=g_tilde,
        shared_inequality=h_tilde,
        parameter_dim=2*nx,                 # θ = [x1_0; x2_0]
        primal_dims=[nprim, nprim],         # τ1, τ2
        equality_dims=[0, 0],
        inequality_dims=[0, 0],
        shared_equality_dim=2*nx*T,         # IC + dyn (both players)
        shared_inequality_dim=shared_ineq,
    )
    PG = ParametricGame(spec)

    def _pack_theta(x0_1, x0_2):
        return np.r_[np.asarray(x0_1, float)[:nx], np.asarray(x0_2, float)[:nx]]

    def _unpack_players(primals):
        XU = []
        for p in range(2):
            tau_p = np.asarray(primals[p]).reshape(-1)
            Xp, Up = unpack_trajectory(tau_p, nx, nu, T)
            XU.append((Xp, Up))
        return XU[0][0], XU[0][1], XU[1][0], XU[1][1]

    def replan(theta_vec, prev1D, prev2D, prevR1, prevR2):
        PG.set_theta(theta_vec)
        PG.solve_with_path(path_exe=cfg.get("path_exe","pathampl"), tee=cfg.get("path_tee", True))
        taus = PG.extract_tau()
        X1, U1, X2, U2 = _unpack_players(taus)
        plan1 = [_p3_row(X1[t, :]) for t in range(T)]
        plan2 = [_p3_row(X2[t, :]) for t in range(T)]
        att1, prev1_out, prevR1_out = plan_attitudes_from_X(X1, prev1D, prevR1)
        att2, prev2_out, prevR2_out = plan_attitudes_from_X(X2, prev2D, prevR2)
        return np.zeros(1), plan1, plan2, U1, U2, att1, att2, prev1_out, prev2_out, prevR1_out, prevR2_out

    # first plan
    z_last, plan1, plan2, U1, U2, att1, att2, prev_axis1D, prev_axis2D, prev_R1, prev_R2 = \
        replan(_pack_theta(x1, x2), prev_axis1D, prev_axis2D, prev_R1, prev_R2)
    step_in_turn = 0

    # rollout loop
    for k in range(steps):
        if k % turn_len == 0 and k > 0:
            z_last, plan1, plan2, U1, U2, att1, att2, prev_axis1D, prev_axis2D, prev_R1, prev_R2 = \
                replan(_pack_theta(x1, x2), prev_axis1D, prev_axis2D, prev_R1, prev_R2)
            step_in_turn = 0

        plan_hist1.append(plan1); plan_hist2.append(plan2)
        plan_att1.append(att1);   plan_att2.append(att2)

        u1 = U1[min(step_in_turn, len(U1)-1)]
        u2 = U2[min(step_in_turn, len(U2)-1)]
        step_in_turn += 1

        # plant step (translational)
        x1 = step_plant(x1, u1)
        x2 = step_plant(x2, u2)

        # derive attitude for viz only
        def att_from_x(x, prev_R, prev_axisD):
            v = np.asarray(x[D:2*D], float); n = np.linalg.norm(v)
            axisD = (v/n) if n > 1e-3 else prev_axisD
            axis3 = axisD if D==3 else np.array([axisD[0], axisD[1], 0.0], float)
            R = frame_from_axis_continuous(axis3, R_prev=prev_R, align=align, world_up=np.asarray(world_up,float))
            return R, axisD, 0.0

        R1, axis1D, phi1_now = att_from_x(x1, prev_R1, prev_axis1D)
        R2, axis2D, phi2_now = att_from_x(x2, prev_R2, prev_axis2D)
        prev_R1, prev_R2, prev_axis1D, prev_axis2D = R1, R2, axis1D, axis2D
        exec_att1.append({"R": R1, "phi": phi1_now}); phi_hist1.append(phi1_now)
        exec_att2.append({"R": R2, "phi": phi2_now}); phi_hist2.append(phi2_now)
        exec_xyz1.append(_p3_row(x1)); exec_xyz2.append(_p3_row(x2))

        # FOV
        if fov_on:
            R_def = R2 if fov_agent == 2 else R1
            x_def = x2 if fov_agent == 2 else x1
            x_tgt = x1 if fov_agent == 2 else x2
            a_def3 = R_def[0] if align == "x" else R_def[2]
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
    return ret

# -------------------- Viz wrappers (lazy import to avoid circulars) --------------------
def animate_rollout_3d(*args, **kwargs):
    from game_viz import animate_rollout_3d as _anim
    return _anim(*args, **kwargs)

def interactive_rollout_3d(*args, **kwargs):
    from game_viz import interactive_rollout_3d as _inter
    return _inter(*args, **kwargs)


def make_pyomo_objectives(nx, nu, T, D, cfg):
    """
    Return [f1, f2] where each f_i(tau_all, theta) -> Pyomo scalar expression.
    (No player index arg; ParametricGame calls with 2 args.)
    """
    eff1 = float(cfg.get("effort_w1", 0.01))
    eff2 = float(cfg.get("effort_w2", 0.01))
    w_far = float(cfg.get("w_far", 1.0))
    w_term = float(cfg.get("w_term", 1.0))
    d_des = float(cfg.get("follow_gap", 0.6))
    w_long = float(cfg.get("w_tail_long", 1.0))
    w_lat = float(cfg.get("w_tail_lat", 8.0))
    v_ref = float(cfg.get("tail_v_ref", 0.2))

    def tail_cost(pL, pF, vL):
        speed2 = sum(vL[j]*vL[j] for j in range(D)) + 1e-9
        r = [pF[j] - pL[j] for j in range(D)]
        spar = sum(r[j]*vL[j] for j in range(D)) / (speed2 + 1e-9)
        rlat = [r[j] - spar*vL[j] for j in range(D)]
        tail_quad = w_long*(spar + d_des)*(spar + d_des) + w_lat*sum(rlat[j]*rlat[j] for j in range(D))
        blend = speed2 / (speed2 + v_ref*v_ref)
        return blend*tail_quad + (1.0 - blend)*sum(r[j]*r[j] for j in range(D))

    def f1(tau_all, theta):
        # P1 tails P2
        nprim = T*nx + (T-1)*nu
        tau1 = tau_all[0:nprim]
        tau2 = tau_all[nprim:2*nprim]
        xs1, us1 = py_unpack_tau(tau1, nx, nu, T)
        xs2, _    = py_unpack_tau(tau2, nx, nu, T)
        J = 0.0
        for t in range(T-1):
            p2 = xs2[t][0:D]; v2 = xs2[t][D:2*D]; p1 = xs1[t][0:D]
            J += tail_cost(p2, p1, v2) + eff1*sum(u*u for u in us1[t])
        p2T = xs2[T-1][0:D]; v2T = xs2[T-1][D:2*D]; p1T = xs1[T-1][0:D]
        J += w_term*tail_cost(p2T, p1T, v2T)
        return J

    def f2(tau_all, theta):
        # P2 increases separation (minimize negative dist^2)
        nprim = T*nx + (T-1)*nu
        tau1 = tau_all[0:nprim]
        tau2 = tau_all[nprim:2*nprim]
        xs1, _  = py_unpack_tau(tau1, nx, nu, T)
        xs2, us2 = py_unpack_tau(tau2, nx, nu, T)
        J = 0.0
        for t in range(T-1):
            r = [xs2[t][j] - xs1[t][j] for j in range(D)]
            J += -w_far*py_sumsqr(r) + eff2*py_sumsqr(us2[t])
        rT = [xs2[T-1][j] - xs1[T-1][j] for j in range(D)]
        J += -w_term*py_sumsqr(rT)
        return J

    return [f1, f2]
