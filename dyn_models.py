# dyn_models.py
from __future__ import annotations
import numpy as np
import casadi as ca

# -------------------- small helpers --------------------
def _unit(v, eps: float = 1e-12):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    return v / (n + eps)

def _skew(w):
    wx, wy, wz = w
    return np.array([[0, -wz,  wy],
                     [wz,  0, -wx],
                     [-wy, wx,  0]], float)

def minimal_rotation(a, b, eps: float = 1e-9):
    a = _unit(a); b = _unit(b)
    v = np.cross(a, b); s = np.linalg.norm(v); c = float(np.dot(a, b))
    if s < eps:
        if c > 0: return np.eye(3)
        axis = _unit(np.cross(a, np.array([1,0,0]) if abs(a[0]) < 0.9 else np.array([0,1,0])))
        K = _skew(axis)
        return np.eye(3) + 2*(K @ K)
    axis = v / s; K = _skew(axis); ang = np.arctan2(s, c)
    return np.eye(3) + np.sin(ang)*K + (1 - np.cos(ang))*(K @ K)

def frame_from_axis_continuous(axis, R_prev=None, align: str = "x",
                               world_up=np.array([0.0, 0.0, 1.0])):
    a = _unit(np.asarray(axis, float))
    up = _unit(np.asarray(world_up, float))
    if R_prev is not None:
        if align == "x":
            x_prev = _unit(R_prev[0]); Rdel = minimal_rotation(x_prev, a); R = Rdel @ R_prev
            x_b = a
            y_tmp = R[1] - x_b*np.dot(R[1], x_b)
            if np.linalg.norm(y_tmp) < 1e-8:
                ref = up if abs(np.dot(up, x_b)) < 0.98 else np.array([0,1,0], float)
                y_tmp = np.cross(ref, x_b)
            y_b = _unit(y_tmp); z_b = _unit(np.cross(x_b, y_b)); y_b = _unit(np.cross(z_b, x_b))
            return np.vstack([x_b, y_b, z_b])
        else:
            z_prev = _unit(R_prev[2]); Rdel = minimal_rotation(z_prev, a); R = Rdel @ R_prev
            z_b = a
            x_tmp = R[0] - z_b*np.dot(R[0], z_b)
            if np.linalg.norm(x_tmp) < 1e-8:
                ref = up if abs(np.dot(up, z_b)) < 0.98 else np.array([1,0,0], float)
                x_tmp = np.cross(ref, z_b)
            x_b = _unit(x_tmp); y_b = _unit(np.cross(z_b, x_b)); x_b = _unit(np.cross(y_b, z_b))
            return np.vstack([x_b, y_b, z_b])
    if align == "x":
        if abs(np.dot(a, up)) > 0.98:
            up = np.array([0,1,0], float) if abs(a[1]) < 0.9 else np.array([1,0,0], float)
        y_b = _unit(np.cross(up, a)); z_b = _unit(np.cross(a, y_b))
        return np.vstack([a, y_b, z_b])
    else:
        if abs(np.dot(a, up)) > 0.98: up = np.array([1,0,0], float)
        x_b = _unit(np.cross(up, a)); y_b = _unit(np.cross(a, x_b))
        return np.vstack([x_b, y_b, a])

def world_to_body_R(axis, D: int = 3, align: str = "x", up=(0.0, 0.0, 1.0)):
    axis = _unit(axis); up = _unit(np.asarray(up, float))
    if D == 2:
        x_b = axis; y_b = np.array([-axis[1], axis[0]])
        return np.vstack([x_b, y_b])
    if align == "x":
        x_b = axis
        if abs(np.dot(x_b, up)) > 0.98:
            up = np.array([0.0, 1.0, 0.0]) if abs(x_b[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        y_b = _unit(np.cross(up, x_b)); z_b = _unit(np.cross(x_b, y_b))
        return np.vstack([x_b, y_b, z_b])
    z_b = axis
    ref = np.array([1.0, 0.0, 0.0]) if abs(z_b[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_b = _unit(np.cross(ref, z_b)); y_b = _unit(np.cross(z_b, x_b))
    return np.vstack([x_b, y_b, z_b])

def apply_roll_about_axis(R_wb, phi: float, align: str = "x"):
    x_b, y_b, z_b = R_wb[0], R_wb[1], R_wb[2]
    c, s = np.cos(phi), np.sin(phi)
    if align == "x":
        y_p =  c*y_b + s*z_b; z_p = -s*y_b + c*z_b
        return np.vstack([x_b, y_p, z_p])
    x_p =  c*x_b + s*y_b; y_p = -s*x_b + c*y_b
    return np.vstack([x_p, y_p, z_b])

def dims_from_D(D: int):
    assert D in (2,3)
    return 2*D, D

def as_numpy_const(M):
    if isinstance(M, np.ndarray): return M.astype(float)
    try:
        return np.array(M.full(), dtype=float)
    except Exception:
        pass
    f = ca.Function('const_mat', [], [ca.MX(M)])
    out = f()
    if isinstance(out, dict): out = next(iter(out.values()))
    elif isinstance(out, (list, tuple)): out = out[0]
    try:
        return np.array(out.full(), dtype=float)
    except Exception:
        return np.array(out, dtype=float)

# -------------------- HCW (and discretization) --------------------
def hcw_mean_motion(hcw_cfg: dict):
    if "n" in hcw_cfg: return float(hcw_cfg["n"])
    mu = float(hcw_cfg.get("mu", 3.986004418e14))
    r0 = float(hcw_cfg["r0"])
    return float(np.sqrt(mu / (r0**3)))

def _discretize_linear(Ac, Bc, dt, series_terms=18):
    nx = int(Ac.shape[0]); nu = int(Bc.shape[1])
    Ac_np = as_numpy_const(Ac); Bc_np = as_numpy_const(Bc)
    M = np.block([[Ac_np,              Bc_np],
                  [np.zeros((nu, nx)), np.zeros((nu, nu))]])
    try:
        from scipy.linalg import expm
        E = expm(M * dt)
    except Exception:
        E = np.eye(nx + nu); term = np.eye(nx + nu); Mdt = M * dt
        for k in range(1, series_terms + 1):
            term = term @ (Mdt / k); E += term
    Ad_np = E[:nx, :nx]; Bd_np = E[:nx, nx:nx+nu]
    return ca.MX(Ad_np), ca.MX(Bd_np)

def hcw_discrete_mats(n: float, dt: float):
    n = float(n)
    Ac = ca.MX.zeros(6,6)
    Ac[0,3] = 1.0; Ac[1,4] = 1.0; Ac[2,5] = 1.0
    Ac[3,0] = 3*n*n; Ac[3,4] = 2*n
    Ac[4,3] = -2*n;  Ac[5,2] = -n*n
    Bc = ca.MX.zeros(6,3); Bc[3,0] = 1.0; Bc[4,1] = 1.0; Bc[5,2] = 1.0
    return _discretize_linear(Ac, Bc, dt)

# -------------------- attitude augmentation & bounds --------------------
def augment_AB_for_att(Ad_tr, Bd_tr, dt, att_cfg):
    Ad_tr = ca.MX(Ad_tr); Bd_tr = ca.MX(Bd_tr)
    nx_tr = Ad_tr.size1(); nu_tr = Bd_tr.size2()
    n_att = 3; n_ctrl = 3
    Ad_aug = ca.MX.zeros(nx_tr + n_att, nx_tr + n_att)
    Ad_aug[:nx_tr, :nx_tr] = Ad_tr
    Ad_aug[nx_tr:, nx_tr:] = ca.MX_eye(n_att)
    Bd_aug = ca.MX.zeros(nx_tr + n_att, nu_tr + n_ctrl)
    Bd_aug[:nx_tr, :nu_tr] = Bd_tr
    Bd_aug[nx_tr:, nu_tr:] = dt * ca.MX_eye(n_att)
    idx = {"nx": nx_tr+n_att, "nu": nu_tr+n_ctrl,
           "i_phi": nx_tr+0, "i_theta": nx_tr+1, "i_psi": nx_tr+2,
           "i_u_phi": nu_tr+0, "i_u_theta": nu_tr+1, "i_u_psi": nu_tr+2}
    return Ad_aug, Bd_aug, idx

def augment_bounds_with_att(x_lb, x_ub, u_lb, u_ub, att_cfg):
    x_lb = np.asarray(x_lb, float).ravel(); x_ub = np.asarray(x_ub, float).ravel()
    u_lb = np.asarray(u_lb, float).ravel(); u_ub = np.asarray(u_ub, float).ravel()
    phi_lim   = att_cfg.get("phi_lim",   (-np.pi, np.pi))
    theta_lim = att_cfg.get("theta_lim", (-np.pi/2, np.pi/2))
    psi_lim   = att_cfg.get("psi_lim",   (-np.pi, np.pi))
    phi_dot_lim   = att_cfg.get("phi_dot_lim",   (-1.0, 1.0))
    theta_dot_lim = att_cfg.get("theta_dot_lim", (-1.0, 1.0))
    psi_dot_lim   = att_cfg.get("psi_dot_lim",   (-1.0, 1.0))
    x_lb = np.concatenate([x_lb, [phi_lim[0],   theta_lim[0],   psi_lim[0]]])
    x_ub = np.concatenate([x_ub, [phi_lim[1],   theta_lim[1],   psi_lim[1]]])
    u_lb = np.concatenate([u_lb, [phi_dot_lim[0], theta_dot_lim[0], psi_dot_lim[0]]])
    u_ub = np.concatenate([u_ub, [phi_dot_lim[1], theta_dot_lim[1], psi_dot_lim[1]]])
    return x_lb, x_ub, u_lb, u_ub

def pad_x0_with_att(x0_row, att_cfg, D):
    x0_row = np.asarray(x0_row, float).ravel()
    phi0   = att_cfg.get("phi0", 0.0)
    theta0 = att_cfg.get("theta0", 0.0)
    psi0   = att_cfg.get("psi0", 0.0)
    return np.concatenate([x0_row, [phi0, theta0, psi0]])

# -------------------- env/bounds --------------------
def make_bounds(cfg: dict):
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx, nu = dims_from_D(D)
    vmax, umax = float(cfg["vmax"]), float(cfg["umax"])
    ar = cfg["arena"]; ar_type = ar.get("type", "box")
    u_lb = -umax * np.ones(nu, float); u_ub = umax * np.ones(nu, float)
    BIG = 1e6
    if ar_type == "box":
        if D == 2:
            p_lb = np.array([ar["xmin"], ar["ymin"]], float)
            p_ub = np.array([ar["xmax"], ar["ymax"]], float)
        else:
            p_lb = np.array([ar["xmin"], ar["ymin"], ar["zmin"]], float)
            p_ub = np.array([ar["xmax"], ar["ymax"], ar["zmax"]], float)
    else:
        p_lb = -BIG * np.ones(D, float); p_ub = BIG * np.ones(D, float)
    v_lb = -vmax * np.ones(D, float); v_ub = vmax * np.ones(D, float)
    x_lb = np.r_[p_lb, v_lb]; x_ub = np.r_[p_ub, v_ub]
    return x_lb, x_ub, u_lb, u_ub
