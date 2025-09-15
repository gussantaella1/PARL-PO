# game_attquat.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# ============================================================
# constants & tiny helpers
# ============================================================

_EPS = 1e-12
_BIG = 1e12


def _norm_safe(v: np.ndarray, eps: float = _EPS) -> float:
    n = float(np.linalg.norm(v))
    return n if np.isfinite(n) and n > eps else eps


def _unit(v: np.ndarray, eps: float = _EPS) -> np.ndarray:
    n = _norm_safe(v, eps)
    return np.asarray(v, float) / n


# ============================================================
# quaternions (scalar-first: q = [w, x, y, z])
# ============================================================

def q_norm(q: np.ndarray) -> np.ndarray:
    """
    Normalize scalar-first quaternion robustly.
    Falls back to identity if input is near-zero or non-finite.
    """
    q = np.asarray(q, float).reshape(4,)
    s = float(np.dot(q, q))
    if not np.isfinite(s) or s < _EPS**2:
        return np.array([1.0, 0.0, 0.0, 0.0], float)
    return q / np.sqrt(s)


def q_mul(q2: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """
    Hamilton product: q = q2 * q1 (apply q1, then q2).
    All scalar-first.
    """
    w2, x2, y2, z2 = np.asarray(q2, float).reshape(4,)
    w1, x1, y1, z1 = np.asarray(q1, float).reshape(4,)
    return np.array([
        w2*w1 - x2*x1 - y2*y1 - z2*z1,
        w2*x1 + x2*w1 + y2*z1 - z2*y1,
        w2*y1 - x2*z1 + y2*w1 + z2*x1,
        w2*z1 + x2*y1 - y2*x1 + z2*w1
    ], float)


def q_from_axis_angle(u: np.ndarray, th: float) -> np.ndarray:
    """
    Quaternion from rotation of angle th about axis u (any length).
    """
    u = np.asarray(u, float).reshape(3,)
    n = _norm_safe(u)
    u = u / n
    half = 0.5 * float(th)
    c, s = np.cos(half), np.sin(half)
    return q_norm(np.array([c, *(u*s)], float))


def q_increment(q: np.ndarray, w_body_rel: np.ndarray, dt: float) -> np.ndarray:
    """
    Increment quaternion by integrating a body-relative angular rate
    over dt using an axis-angle exponential map (first-order).
    """
    q = q_norm(q)
    w_body_rel = np.asarray(w_body_rel, float).reshape(3,)
    dt = float(dt)
    th = float(np.linalg.norm(w_body_rel)) * dt
    if not np.isfinite(th) or th < 1e-14:
        return q
    dq = q_from_axis_angle(w_body_rel, th)
    return q_norm(q_mul(dq, q))


def q_to_R(q: np.ndarray) -> np.ndarray:
    """
    DCM R_BH from quaternion mapping Body→H (scalar-first).
    """
    w, x, y, z = q_norm(q)
    # rows are body axes expressed in H
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], float)


def R_to_q(R: np.ndarray) -> np.ndarray:
    """
    Robust scalar-first quaternion from a proper-orthogonal 3x3 matrix.
    """
    R = np.asarray(R, float).reshape(3, 3)
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    t = m00 + m11 + m22
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    else:
        if m00 > m11 and m00 > m22:
            s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s
    return q_norm(np.array([w, x, y, z], float))


# ============================================================
# rigid-body helpers
# ============================================================

def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, float).reshape(3,)
    return np.array([[0, -z,  y],
                     [z,  0, -x],
                     [-y, x,  0]], float)


def vee(S: np.ndarray) -> np.ndarray:
    """
    Inverse of skew for a 3x3 skew-symmetric matrix.
    """
    S = np.asarray(S, float).reshape(3, 3)
    return np.array([S[2, 1], S[0, 2], S[1, 0]], float)


@dataclass
class AttState:
    """
    Attitude state:
      q_BH   : scalar-first quaternion mapping Body→H
      w_BI_B : body angular rate wrt inertial, expressed in Body
    """
    q_BH: np.ndarray
    w_BI_B: np.ndarray


def gravity_gradient_tau(q_BH: np.ndarray, n: float, J: np.ndarray) -> np.ndarray:
    """
    Gravity-gradient torque for LVLH (H) frame.
    τ_gg = 3 n^2 (c × J c), with c = R_HB e_xH.
    """
    if abs(n) < _EPS:
        return np.zeros(3)
    R_BH = q_to_R(q_BH)
    R_HB = R_BH.T
    ex_H = np.array([1.0, 0.0, 0.0], float)
    c_B = R_HB @ ex_H
    return 3.0 * (n**2) * np.cross(c_B, J @ c_B)


def euler_rhs(w_BI_B: np.ndarray, tau_B: np.ndarray, J: np.ndarray) -> np.ndarray:
    """
    Euler equation: J wdot + w×(Jw) = τ  →  wdot = J^{-1}(τ − w×(Jw))
    """
    w = np.asarray(w_BI_B, float).reshape(3,)
    tau = np.asarray(tau_B, float).reshape(3,)
    J = np.asarray(J, float).reshape(3, 3)
    return np.linalg.solve(J, tau - np.cross(w, J @ w))


def step_attitude_quat(state: AttState,
                       tau_ctrl_B: np.ndarray,
                       dt: float,
                       n: float,
                       J: np.ndarray,
                       tau_ext_B: np.ndarray | None = None,
                       include_gravity_gradient: bool = True) -> AttState:
    """
    One integration step for quaternion attitude in LVLH:
      - Integrate Euler (body rates) with RK2
      - Kinematics use body-relative rate w_BH_B = w_BI_B - R_HB w_HI_H
      - Update quaternion with exponential-map increment & renormalize

    Args
    ----
    state : AttState(q_BH, w_BI_B)
    tau_ctrl_B : control torque in Body
    dt   : step [s]
    n    : mean motion [rad/s] (LVLH spin about +z_H); set 0 for inertial target
    J    : inertia (3x3)
    tau_ext_B : optional disturbance torque in Body
    include_gravity_gradient : add gravity-gradient torque if True

    Returns
    -------
    AttState
    """
    q = q_norm(state.q_BH)
    w = np.asarray(state.w_BI_B, float).reshape(3,)
    tau_tot = np.asarray(tau_ctrl_B, float).reshape(3,).copy()

    if include_gravity_gradient and abs(n) > _EPS:
        tau_tot += gravity_gradient_tau(q, n, J)
    if tau_ext_B is not None:
        tau_tot += np.asarray(tau_ext_B, float).reshape(3,)

    # RK2 on angular rate
    wdot0 = euler_rhs(w, tau_tot, J)
    w_mid = w + 0.5 * float(dt) * wdot0
    wdotm = euler_rhs(w_mid, tau_tot, J)
    w_next = w + float(dt) * wdotm

    # LVLH-relative kinematics
    R_BH = q_to_R(q)
    R_HB = R_BH.T
    w_HI_H = np.array([0.0, 0.0, float(n)], float)
    w_BH_B = 0.5 * (w + w_next) - (R_HB @ w_HI_H)

    q_next = q_increment(q, w_BH_B, dt)
    return AttState(q_BH=q_next, w_BI_B=w_next)


# ============================================================
# simple PD pointing control (Lee-style attitude error)
# ============================================================

def desired_R_from_axis(a_des_H: np.ndarray,
                        align: str = "x",
                        world_up: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """
    Build a desired DCM (rows are body axes in H) with:
      - if align == 'x': x_b aligned to a_des_H
      - if align == 'z': z_b aligned to a_des_H
    Uses world_up to fix the roll about the alignment axis.
    """
    a = _unit(np.asarray(a_des_H, float).reshape(3,))
    up = _unit(np.asarray(world_up, float).reshape(3,))
    if align == "x":
        # prevent near-collinearity with up
        if abs(np.dot(a, up)) > 0.98:
            up = np.array([0.0, 1.0, 0.0], float)
        y = _unit(np.cross(up, a))
        z = _unit(np.cross(a, y))
        return np.vstack([a, y, z])
    else:
        if abs(np.dot(a, up)) > 0.98:
            up = np.array([1.0, 0.0, 0.0], float)
        x = _unit(np.cross(up, a))
        y = _unit(np.cross(a, x))
        return np.vstack([x, y, a])


def tau_point_boresight(q_BH: np.ndarray,
                        w_BI_B: np.ndarray,
                        a_des_H: np.ndarray,
                        n: float,
                        kp: float = 0.8,
                        kd: float = 0.2,
                        align: str = "x",
                        world_up: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """
    PD-like torque that points the body alignment axis to a desired H-axis.

    Error metric:
      e_R = 0.5 * vee(R_des^T R - R^T R_des)  (Lee et al., 2010)
      w_rel = w_BI_B - R_HB w_HI_H (body rate relative to LVLH)

    Returns
    -------
    tau_B : np.ndarray, shape (3,)
    """
    R = q_to_R(q_BH)
    Rdes = desired_R_from_axis(a_des_H, align=align, world_up=world_up)
    S = Rdes.T @ R - R.T @ Rdes
    e_R = 0.5 * vee(S)

    R_HB = R.T
    w_rel = np.asarray(w_BI_B, float).reshape(3,) - (R_HB @ np.array([0.0, 0.0, float(n)], float))
    return -float(kp) * e_R - float(kd) * w_rel


def saturate_norm(v: np.ndarray, vmax: float) -> np.ndarray:
    """
    Limit the vector 2-norm to vmax (no-op if vmax<=0).
    """
    vmax = float(vmax)
    v = np.asarray(v, float).reshape(-1)
    if vmax <= 0:
        return v
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= vmax:
        return v
    return v * (vmax / (n + _EPS))
