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
    sf_cfg = dict(cfg.get("safety_filter", {}) or {})
    raw_vmax = sf_cfg.get("vmax", None)
    if raw_vmax is None:
        raw_vmax = cfg.get("vmax", None)
    vmax = float(raw_vmax) if raw_vmax is not None else 1e6
    umax = float(cfg["umax"])
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



# Additions for extra dynamics models:

# ----------------------------
# Kepler helpers
# ----------------------------
def _R1(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]], dtype=float)

def _R3(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]], dtype=float)

def _kepler_E_from_M(M, e, tol=1e-12, it=50):
    # Newton solve: E - e sinE = M
    E = M if e < 0.8 else np.pi
    for _ in range(it):
        f = E - e*np.sin(E) - M
        fp = 1 - e*np.cos(E)
        dE = -f / fp
        E = E + dE
        if abs(dE) < tol:
            break
    return E

def _nu_from_E(E, e):
    # true anomaly
    s = np.sqrt(1+e)*np.sin(E/2)
    c = np.sqrt(1-e)*np.cos(E/2)
    return 2*np.arctan2(s, c)

def _M_from_nu(nu, e):
    # convert nu -> E -> M
    E = 2*np.arctan2(np.sqrt(1-e)*np.sin(nu/2), np.sqrt(1+e)*np.cos(nu/2))
    return E - e*np.sin(E)

def rv_from_orbital_elements(mu, a, e, inc, raan, argp, nu):
    p = a*(1 - e**2)
    r = p/(1 + e*np.cos(nu))

    r_pf = np.array([r*np.cos(nu), r*np.sin(nu), 0.0], dtype=float)
    v_pf = np.sqrt(mu/p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0], dtype=float)

    Q = _R3(raan) @ _R1(inc) @ _R3(argp)  # perifocal -> inertial
    r_I = Q @ r_pf
    v_I = Q @ v_pf
    return r_I, v_I

def rtn_dcm_from_rv(r_I, v_I):
    rhat = r_I / np.linalg.norm(r_I)
    h = np.cross(r_I, v_I)
    nhat = h / np.linalg.norm(h)
    that = np.cross(nhat, rhat)
    # columns are RTN axes expressed in inertial
    C_RTN2I = np.column_stack((rhat, that, nhat))
    C_I2RTN = C_RTN2I.T
    return C_RTN2I, C_I2RTN

def omega_rtn_from_rv(r_I, v_I):
    # angular rate magnitude = |h| / r^2, along +N
    h = np.cross(r_I, v_I)
    r = np.linalg.norm(r_I)
    w = np.linalg.norm(h) / (r*r)
    return np.array([0.0, 0.0, w], dtype=float)  # in RTN coords

# ----------------------------
# Chief orbit cache
# ----------------------------
def chief_orbit_cache_rtn(chief_orbit: dict, dt: float, N: int):
    """
    Precompute chief r,v, DCMs, omega for k=0..N.
    chief_orbit keys: mu,a,e,i,raan,argp,nu0
    """
    mu   = float(chief_orbit["mu"])
    a    = float(chief_orbit["a"])
    e    = float(chief_orbit["e"])
    inc  = float(chief_orbit.get("i", 0.0))
    raan = float(chief_orbit.get("raan", 0.0))
    argp = float(chief_orbit.get("argp", 0.0))
    nu0  = float(chief_orbit.get("nu0", 0.0))

    n = np.sqrt(mu / a**3)
    M0 = _M_from_nu(nu0, e)

    times = np.arange(N+1, dtype=float) * dt

    rC = np.zeros((N+1, 3), dtype=float)
    vC = np.zeros((N+1, 3), dtype=float)
    C_RTN2I = np.zeros((N+1, 3, 3), dtype=float)
    C_I2RTN = np.zeros((N+1, 3, 3), dtype=float)
    w_rtn = np.zeros((N+1, 3), dtype=float)

    for k, t in enumerate(times):
        M = M0 + n*t
        E = _kepler_E_from_M(M, e)
        nu = _nu_from_E(E, e)
        r_I, v_I = rv_from_orbital_elements(mu, a, e, inc, raan, argp, nu)
        CR2I, CI2R = rtn_dcm_from_rv(r_I, v_I)

        rC[k] = r_I
        vC[k] = v_I
        C_RTN2I[k] = CR2I
        C_I2RTN[k] = CI2R
        w_rtn[k] = omega_rtn_from_rv(r_I, v_I)

    return {
        "mu": mu,
        "dt": float(dt),
        "N": int(N),
        "rC": rC,
        "vC": vC,
        "C_RTN2I": C_RTN2I,
        "C_I2RTN": C_I2RTN,
        "w_rtn": w_rtn,
    }

# ----------------------------
# Two-body RK4 step in inertial, state kept in RTN
# ----------------------------
def _two_body_acc(mu, r):
    rn = np.linalg.norm(r)
    return -mu * r / (rn**3)

def two_body_step_rtn(x6, u3, k, cache):
    """
    x6 = [rho_R, rho_T, rho_N, rhod_R, rhod_T, rhod_N] in chief RTN.
    u3 is accel command in RTN (m/s^2).
    """
    mu = cache["mu"]
    dt = cache["dt"]
    rC = cache["rC"][k]
    vC = cache["vC"][k]
    CR2I = cache["C_RTN2I"][k]
    CI2R = cache["C_I2RTN"][k]
    w = cache["w_rtn"][k]  # in RTN

    rho = x6[:3]
    rhod = x6[3:]

    # RTN -> inertial deputy
    rD0 = rC + CR2I @ rho
    vD0 = vC + CR2I @ (rhod + np.cross(w, rho))

    aU_I = CR2I @ u3

    def f(state):
        r = state[:3]
        v = state[3:]
        a = _two_body_acc(mu, r) + aU_I
        return np.hstack((v, a))

    s0 = np.hstack((rD0, vD0))
    k1 = f(s0)
    k2 = f(s0 + 0.5*dt*k1)
    k3 = f(s0 + 0.5*dt*k2)
    k4 = f(s0 + dt*k3)
    s1 = s0 + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    # Convert back inertial -> RTN at k+1 frame
    rC1 = cache["rC"][k+1]
    vC1 = cache["vC"][k+1]
    CI2R1 = cache["C_I2RTN"][k+1]
    w1 = cache["w_rtn"][k+1]

    rho1 = CI2R1 @ (s1[:3] - rC1)
    vrel_rtn = CI2R1 @ (s1[3:] - vC1)
    rhod1 = vrel_rtn - np.cross(w1, rho1)

    return np.hstack((rho1, rhod1))

# ----------------------------
# Linearize discrete map for LTV
# ----------------------------
def linearize_two_body_rtn_discrete(cache, dt: float, eps: float = 1e-5):
    """
    Returns Ad_seq, Bd_seq for k=0..N-1 by finite-diff linearizing the discrete step
    around x=0, u=0.
    """
    N = cache["N"]
    Ad_seq = np.zeros((N, 6, 6), dtype=float)
    Bd_seq = np.zeros((N, 6, 3), dtype=float)

    x0 = np.zeros(6, dtype=float)
    u0 = np.zeros(3, dtype=float)

    for k in range(N):
        fx0 = two_body_step_rtn(x0, u0, k, cache)

        # A
        for j in range(6):
            dx = np.zeros(6); dx[j] = eps
            f_p = two_body_step_rtn(x0 + dx, u0, k, cache)
            f_m = two_body_step_rtn(x0 - dx, u0, k, cache)
            Ad_seq[k, :, j] = (f_p - f_m) / (2*eps)

        # B
        for j in range(3):
            du = np.zeros(3); du[j] = eps
            f_p = two_body_step_rtn(x0, u0 + du, k, cache)
            f_m = two_body_step_rtn(x0, u0 - du, k, cache)
            Bd_seq[k, :, j] = (f_p - f_m) / (2*eps)

    return Ad_seq, Bd_seq
