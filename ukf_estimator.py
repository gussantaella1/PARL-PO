# ukf_estimator.py
from __future__ import annotations
import numpy as np

# -------------------------------
# Helpers
# -------------------------------

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
    return 0.5 * (M + M.T)

def _psd_enforce(M: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    """Force symmetric PSD by eigenvalue flooring."""
    M = _symmetrize(M)
    eig, V = np.linalg.eigh(M)
    eig = np.maximum(eig, floor)
    return (V * eig) @ V.T

def _normalize_angle(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi

def _normalize_meas_vec(v: np.ndarray) -> np.ndarray:
    """Normalize az and (optionally) el components depending on length."""
    v = np.asarray(v, float).copy()
    if v.size >= 1:
        v[0] = _normalize_angle(v[0])
    if v.size >= 2:
        v[1] = _normalize_angle(v[1])
    return v

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
    R_wb maps world -> body.
    """
    d_w = np.asarray(p_tgt, float) - np.asarray(p_obs, float)
    n = np.linalg.norm(d_w)
    if n < 1e-12:
        d_w = np.array([1.0, 0.0, 0.0]); n = 1.0
    b_w = d_w / n
    return R_wb @ b_w

def _azel_from_body_vec(vb: np.ndarray):
    x, y, z = vb
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.sqrt(max(x*x + y*y, 1e-18)))
    return az, el


# -------------------------------
# Agent UKF (bearing-only, CV)
# -------------------------------

class AgentUKF:
    """
    Unscented Kalman Filter for a target with constant-velocity dynamics observed by an agent.
    State:   x = [px, py, pz, vx, vy, vz]^T   (world frame)
    Process: x_{k+1} = F(dt) x_k + B(dt)*u_k + w_k
             where u is interpreted as acceleration in WORLD by default.
    Meas:
      - If R is 2x2: z = [az, el]^T   (3D bearing in OBSERVER BODY frame)
      - If R is 1x1: z = [az]^T       (2D bearing-only)
    """
    def __init__(self, x0, P0, Q, R, dt, alpha=1e-3, beta=2.0, kappa=0.0):
        self.x  = np.asarray(x0, float).reshape(6)
        self.P  = _psd_enforce(np.asarray(P0, float))
        self.Q  = _psd_enforce(np.asarray(Q,  float))
        self.R  = _psd_enforce(np.asarray(R,  float))
        self.dt = float(dt)
        self.n  = 6
        # measurement dimension (1 for 2D bearing-only, 2 for az+el)
        self.m  = int(self.R.shape[0])

        lam, Wm0, Wmi, Wc0, Wci, c = _ukf_weights(self.n, alpha, beta, kappa)
        self._Wm0, self._Wmi = Wm0, Wmi
        self._Wc0, self._Wci = Wc0, Wci
        self._c = c  # used as the scaling on P for sigma points

    # ----- linear CV dynamics -----
    def F(self, dt=None):
        if dt is None: dt = self.dt
        F = np.eye(6)
        F[0,3] = dt; F[1,4] = dt; F[2,5] = dt
        return F

    def B_accel(self, dt=None):
        """Input matrix for acceleration input u (3-vector)."""
        if dt is None: dt = self.dt
        B = np.zeros((6,3))
        B[0:3, 0:3] = 0.5 * (dt**2) * np.eye(3)
        B[3:6, 0:3] = dt * np.eye(3)
        return B

    def f(self, x, dt=None, u=None, u_frame='world', R_wb_tgt=None):
        """
        Deterministic dynamics: x+ = F x + B u  (if u provided).
        If u_frame='body', convert u_body -> u_world using R_wb_tgt (world->body).
        """
        F = self.F(dt)
        x_next = F @ x
        if u is not None:
            u = np.asarray(u, float).reshape(3)
            if u_frame == 'body':
                if R_wb_tgt is None:
                    raise ValueError("u given in 'body' frame but R_wb_tgt is None")
                # body -> world: v_w = R_wb^T v_b
                u = R_wb_tgt.T @ u
            x_next = x_next + self.B_accel(dt) @ u
        return x_next

    # ----- measurement: bearing in observer body frame -----
    def h(self, x, p_obs, R_wb):
        p_tgt = x[:3]
        v_b = _body_bearing_from_world(p_obs, R_wb, p_tgt)
        az, el = _azel_from_body_vec(v_b)
        if self.m == 1:
            return np.array([az], float)
        return np.array([az, el], float)

    # ----- UKF steps -----
    def predict(self, dt=None, u=None, u_cov=None, u_frame='world', R_wb_tgt=None):
        """
        Unscented prediction. If u is provided, apply affine term B u to each sigma point.
        If u_cov (3x3) is provided, add B * u_cov * B^T to P to model input uncertainty.
        """
        if dt is None: dt = self.dt
        self.P = _psd_enforce(self.P)

        Xi = _sigma_points(self.x, self.P, self._c)
        # propagate sigma points with shared input u
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
            Bu = self.B_accel(dt)
            U = _psd_enforce(np.asarray(u_cov, float))
            P_pred += Bu @ U @ Bu.T

        self.x = x_pred
        self.P = _psd_enforce(_symmetrize(P_pred), floor=1e-12)

    def update(self, z, p_obs, R_wb):
        self.P = _psd_enforce(self.P)

        # Sigma points and their measurements
        Xi = _sigma_points(self.x, self.P, self._c)
        Zsig = np.zeros((Xi.shape[0], self.m))
        for i in range(Xi.shape[0]):
            Zsig[i] = _normalize_meas_vec(self.h(Xi[i], p_obs, R_wb))

        # Predicted measurement mean
        z_pred = self._Wm0 * Zsig[0] + self._Wmi * Zsig[1:].sum(axis=0)
        z_pred = _normalize_meas_vec(z_pred)

        # Innovation covariance S and cross-covariance Pxz
        S   = self.R.copy()
        Pxz = np.zeros((self.n, self.m))

        # i=0
        dz0 = _normalize_meas_vec(Zsig[0] - z_pred)
        dx0 = Xi[0] - self.x
        S   += self._Wc0 * np.outer(dz0, dz0)
        Pxz += self._Wc0 * np.outer(dx0, dz0)

        # remaining sigma points
        for i in range(1, Zsig.shape[0]):
            dz = _normalize_meas_vec(Zsig[i] - z_pred)
            dx = Xi[i] - self.x
            S   += self._Wci * np.outer(dz, dz)
            Pxz += self._Wci * np.outer(dx, dz)

        S = _psd_enforce(_symmetrize(S), floor=1e-12)
        K = Pxz @ np.linalg.inv(S + 1e-12*np.eye(self.m))

        y = _normalize_meas_vec(np.asarray(z, float).reshape(self.m) - z_pred)
        self.x = self.x + K @ y
        self.P = _psd_enforce(_symmetrize(self.P - K @ S @ K.T), floor=1e-12)
        return self.x, self.P
