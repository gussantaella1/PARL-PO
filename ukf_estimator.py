"""
ukf_estimator.py

Unscented Kalman filter implementation for relative navigation and bearing measurements.
"""

# ukf_estimator.py
from __future__ import annotations
import numpy as np

from game_3dutils import hcw_discrete_mats as _hcw_disc_mx
from game_3dutils import hcw_mean_motion as _hcw_mean_motion
from game_3dutils import as_numpy_const as _as_numpy_const

from dyn_models import hcw_mean_motion, hcw_discrete_mats, as_numpy_const

# -------------------------------
# Helpers
# -------------------------------

# Add control input into time update, if we have overconfidence issue add process bias, 

def _ukf_weights(n: int, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
    """
    Julier/Uhlmann UKF weights.
    Returns (lam, Wm0, Wmi, Wc0, Wci, c) where c = n + lam.
    """
    lam = alpha**2 * (n + kappa) - n
    c = n + lam
    Wm0 = lam / c
    Wmi = 1.0 / (2.0 * c)
    Wc0 = Wm0 + (1.0 - alpha**2 + beta)
    Wci = Wmi
    return lam, Wm0, Wmi, Wc0, Wci, c

def _safe_cholesky(A: np.ndarray, jitter: float = 1e-10, max_tries: int = 5) -> np.ndarray:
    """Cholesky with small diagonal jitter if needed. Returns lower-triangular."""
    k = 0
    while k < max_tries:
        try:
            return np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            A = A + (jitter * (10.0 ** k)) * np.eye(A.shape[0])
            k += 1
    return np.linalg.cholesky(A)  # last attempt may still raise

def _symmetrize(M: np.ndarray) -> np.ndarray:
    """Internal helper for symmetrize."""
    return 0.5 * (M + M.T)

def _psd_enforce(M: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    """Force symmetric PSD by eigenvalue flooring."""
    M = _symmetrize(M)
    eig, V = np.linalg.eigh(M)
    eig = np.maximum(eig, floor)
    return (V * eig) @ V.T

def _safe_inv_2x2(S: np.ndarray,
                  jitter: float = 1e-8,
                  max_tries: int = 5) -> np.ndarray:
    """
    Robust inverse for small 2x2 PSD-ish matrices.
    Tries to add diagonal jitter and invert via Cholesky; falls back to pinv.
    """
    S = np.asarray(S, float)
    I = np.eye(S.shape[0])

    for k in range(max_tries):
        try:
            # Enforce symmetry and positiveness first
            S_reg = _psd_enforce(_symmetrize(S + (jitter * (10.0**k)) * I),
                                 floor=jitter * (10.0**k))
            # Cholesky-based inverse: S_reg = L L^T  →  S_reg^{-1} = L^{-T} L^{-1}
            L = np.linalg.cholesky(S_reg)
            Linv = np.linalg.inv(L)
            return Linv.T @ Linv
        except np.linalg.LinAlgError:
            continue

    # Last resort: Moore–Penrose pseudo-inverse
    return np.linalg.pinv(S)


def _normalize_angle(a: float) -> float:
    """Normalize angle into the canonical representation used here."""
    return (a + np.pi) % (2*np.pi) - np.pi

def _sigma_points(x: np.ndarray, P: np.ndarray, c_scale: float):
    """Symmetric sigma points around x using Cholesky of c_scale*P."""
    n = x.size
    P = _psd_enforce(P, floor=1e-12)
    S = _safe_cholesky(c_scale * P + 1e-12*np.eye(n))  # lower
    Xi = np.zeros((2*n+1, n))
    Xi[0] = x
    for i in range(n):
        Xi[i+1]   = x + S[:, i]
        Xi[n+i+1] = x - S[:, i]
    return Xi

def _body_bearing_from_world(p_obs: np.ndarray, R_wb: np.ndarray, p_tgt: np.ndarray):
    """
    Unit vector from observer to target, expressed in OBSERVER BODY frame.
    R_wb maps world -> body (consistent with your animator).
    """
    d_w = np.asarray(p_tgt, float) - np.asarray(p_obs, float)
    n = np.linalg.norm(d_w)
    if n < 1e-12:
        d_w = np.array([1.0, 0.0, 0.0]); n = 1.0
    b_w = d_w / n
    return R_wb @ b_w

def _azel_from_body_vec(vb: np.ndarray):
    """Internal helper for azel from body vec."""
    x, y, z = vb
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.sqrt(max(x*x + y*y, 1e-18)))
    return az, el

def _body_vec_from_azel(az: float, el: float):
    """Internal helper for body vec from azel."""
    c = np.cos(el)
    return np.array([c*np.cos(az), c*np.sin(az), np.sin(el)], float)


# -------------------------------
# Agent UKF (bearing-only, CV)
# -------------------------------

class AgentUKF:
    """
    UKF with selectable linear dynamics:
      - dyn='cv'  : constant-velocity (legacy)
      - dyn='hcw' : HCW relative motion (needs dt and hcw params)
    State:
      - 6D: x=[px,py,pz,vx,vy,vz]
      - 9D: x=[px,py,pz,vx,vy,vz,ax,ay,az]
    External control input u=[ax,ay,az] remains optional.
    """
    def __init__(self, x0, P0, Q, R, dt,
                 alpha=1e-3, beta=2.0, kappa=0.0,
                 dyn='cv', hcw=None):
        """Store configuration and initialize the runtime state for this object."""
        self.x  = np.asarray(x0, float).reshape(-1)
        if self.x.size not in (6, 9):
            raise ValueError(f"AgentUKF expects 6D or 9D state, got {self.x.size}.")
        self.P  = _psd_enforce(np.asarray(P0, float))
        self.Q  = _psd_enforce(np.asarray(Q,  float))
        self.R  = _psd_enforce(np.asarray(R,  float))
        self.dt = float(dt)
        self.n  = int(self.x.size)

        lam, Wm0, Wmi, Wc0, Wci, c = _ukf_weights(self.n, alpha, beta, kappa)
        self._Wm0, self._Wmi, self._Wc0, self._Wci, self._c = Wm0, Wmi, Wc0, Wci, c

        self._dyn = (dyn or 'cv').lower()
        self._hcw_params = (hcw or {})
        self._Ad = None
        self._Bd = None
        if self._dyn == 'hcw':
            n = hcw_mean_motion(self._hcw_params)
            Ad_mx, Bd_mx = hcw_discrete_mats(n, self.dt)
            self._Ad = as_numpy_const(Ad_mx)
            self._Bd = as_numpy_const(Bd_mx)

    # ---- CV matrices (legacy) ----
    def F(self, dt=None):
        """Return the current discrete state-transition matrix."""
        if dt is None: dt = self.dt
        F = np.eye(6)
        F[0,3]=dt; F[1,4]=dt; F[2,5]=dt
        return F

    def B_accel(self, dt=None):
        """Return the acceleration-input matrix for the current dynamics model."""
        if dt is None: dt = self.dt
        B = np.zeros((6,3))
        B[0:3, 0:3] = 0.5 * (dt**2) * np.eye(3)
        B[3:6, 0:3] = dt * np.eye(3)
        return B

    def linear_dynamics_mats(self, dt=None):
        """Return the linear dynamics matrices used by the estimator or plant step."""
        if dt is None:
            dt = self.dt
        if self._dyn == 'hcw':
            if dt != self.dt or self._Ad is None:
                n = hcw_mean_motion(self._hcw_params)
                Ad_mx, Bd_mx = hcw_discrete_mats(n, dt)
                Ad6 = as_numpy_const(Ad_mx)
                Bd6 = as_numpy_const(Bd_mx)
            else:
                Ad6, Bd6 = self._Ad, self._Bd
        else:
            Ad6, Bd6 = self.F(dt), self.B_accel(dt)

        if self.n == 6:
            return Ad6, Bd6

        A = np.eye(9)
        A[:6, :6] = Ad6
        A[:6, 6:9] = Bd6
        B = np.zeros((9, 3))
        B[:6, :] = Bd6
        return A, B

    # ---- measurement model (bearing in observer BODY frame) ----
    def h(self, x, p_obs, R_wb):
        """Evaluate the measurement model at the provided state."""
        p_tgt = x[:3]
        v_b = _body_bearing_from_world(p_obs, R_wb, p_tgt)
        az, el = _azel_from_body_vec(v_b)
        return np.array([az, el], float)

    # ---- dynamics step used by sigma-point propagation ----
    def f(self, x, dt=None, u=None, u_frame='world', R_wb_tgt=None):
        """Evaluate the process model for one prediction step."""
        if dt is None: dt = self.dt
        u_world = None
        if u is not None:
            u_world = np.asarray(u, float).reshape(3)
            if u_frame == 'body':
                if R_wb_tgt is None:
                    raise ValueError("u in body frame but R_wb_tgt is None")
                u_world = R_wb_tgt.T @ u_world

        A, B = self.linear_dynamics_mats(dt)
        x_next = A @ x
        if u_world is not None:
            x_next = x_next + B @ u_world
        return x_next

    # ---- UKF time update ----
    def predict(self, dt=None, u=None, u_cov=None, u_frame='world', R_wb_tgt=None):
        """Propagate the estimator belief through the dynamics model."""
        if dt is None: dt = self.dt
        self.P = _psd_enforce(self.P)

        # refresh cached Ad/Bd if HCW and dt changed
        if self._dyn == 'hcw' and (self._Ad is None or dt != self.dt):
            n = hcw_mean_motion(self._hcw_params)
            Ad_mx, Bd_mx = hcw_discrete_mats(n, dt)
            self._Ad = as_numpy_const(Ad_mx)
            self._Bd = as_numpy_const(Bd_mx)

        Xi = _sigma_points(self.x, self.P, self._c)
        Xi_pred = np.array([ self.f(xi, dt=dt, u=u, u_frame=u_frame, R_wb_tgt=R_wb_tgt) for xi in Xi ])

        # mean
        x_pred = self._Wm0 * Xi_pred[0] + self._Wmi * Xi_pred[1:].sum(axis=0)

        # covariance from sigma spread + process noise
        P_pred = self.Q.copy()
        dx0 = Xi_pred[0] - x_pred
        P_pred += self._Wc0 * np.outer(dx0, dx0)
        for i in range(1, Xi_pred.shape[0]):
            dxi = Xi_pred[i] - x_pred
            P_pred += self._Wci * np.outer(dxi, dxi)

        # add input uncertainty if provided
        if u_cov is not None:
            U = _psd_enforce(np.asarray(u_cov, float))
            _, Bd = self.linear_dynamics_mats(dt)
            P_pred += Bd @ U @ Bd.T

        self.x = x_pred
        self.P = _psd_enforce(_symmetrize(P_pred), floor=1e-12)
        self.dt = dt  # track last-step dt

    # ---- UKF measurement update (bearing) ----
    def update(self, z, p_obs, R_wb):
        """Assimilate a measurement into the estimator belief."""
        self.P = _psd_enforce(self.P)
        Xi = _sigma_points(self.x, self.P, self._c)
        Zsig = np.array([ self.h(xi, p_obs, R_wb) for xi in Xi ])

        # predicted measurement mean (angle-normalized)
        z_pred = self._Wm0 * Zsig[0] + self._Wmi * Zsig[1:].sum(axis=0)
        z_pred[0] = _normalize_angle(z_pred[0])
        z_pred[1] = _normalize_angle(z_pred[1])

        # innovation & cross-cov
        S   = self.R.copy()
        Pxz = np.zeros((self.n, 2))

        dz0 = Zsig[0] - z_pred
        dz0[0] = _normalize_angle(dz0[0]); dz0[1] = _normalize_angle(dz0[1])
        dx0 = Xi[0] - self.x
        S   += self._Wc0 * np.outer(dz0, dz0)
        Pxz += self._Wc0 * np.outer(dx0, dz0)

        for i in range(1, Zsig.shape[0]):
            dzi = Zsig[i] - z_pred
            dzi[0] = _normalize_angle(dzi[0]); dzi[1] = _normalize_angle(dzi[1])
            dxi = Xi[i] - self.x
            S   += self._Wci * np.outer(dzi, dzi)
            Pxz += self._Wci * np.outer(dxi, dzi)

        S = _psd_enforce(_symmetrize(S), floor=1e-12)
        # K = Pxz @ np.linalg.inv(S + 1e-12*np.eye(2))

        # S is already symmetrized/PSD-enforced above
        Sinv = _safe_inv_2x2(S)
        K = Pxz @ Sinv


        y = np.asarray(z, float) - z_pred
        y[0] = _normalize_angle(y[0]); y[1] = _normalize_angle(y[1])

        self.x = self.x + K @ y
        self.P = _psd_enforce(_symmetrize(self.P - K @ S @ K.T), floor=1e-12)
        return self.x, self.P


# -------------------------------
# Backwards-compatible factory
# -------------------------------

def KF_CV(x0, P0, Q, R, dt,
          kind: str = "ukf",
          dyn: str = "cv",
          hcw: dict | None = None,
          **kwargs):
    """
    Backwards-compatible helper so older code can do:
        KF_CV(x0, P0, Q, R, dt, kind="ukf", dyn="hcw", hcw=hcw_params)
        KF_CV(x0, P0, Q, R, dt, kind="ekf", dyn="hcw", hcw=hcw_params)
    """
    kind = (kind or "ukf").lower()
    dyn = (dyn or "cv").lower()
    if kind == "ekf":
        from ekf_estimator import AgentEKF

        return AgentEKF(
            x0=x0,
            P0=P0,
            Q=Q,
            R=R,
            dt=dt,
            dyn=dyn,
            hcw=hcw,
            **kwargs,
        )

    return AgentUKF(
        x0=x0,
        P0=P0,
        Q=Q,
        R=R,
        dt=dt,
        dyn=dyn,
        hcw=hcw,
        **kwargs,
    )
