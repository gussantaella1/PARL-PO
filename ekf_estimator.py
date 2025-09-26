# ekf_estimator.py
from __future__ import annotations
import numpy as np

# -------------------------------
# Helpers 
# -------------------------------

def _symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)

def _psd_enforce(M: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Force symmetric PSD by eigenvalue flooring."""
    M = _symmetrize(M)
    eig, V = np.linalg.eigh(M)
    eig = np.maximum(eig, floor)
    return (V * eig) @ V.T

def _normalize_angle(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi

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
    x, y, z = vb
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.sqrt(max(x*x + y*y, 1e-18)))
    return az, el


# -------------------------------
# Agent EKF (bearing-only, CV)
# -------------------------------

class AgentEKF:
    """
    Extended Kalman Filter for a target with constant-velocity dynamics observed by an agent.
    State:   x = [px, py, pz, vx, vy, vz]^T   (world frame)
    Process: x_{k+1} = F(dt) x_k + B(dt)*u_k + w_k
             where u is acceleration (world by default; or 'body' with R_wb_tgt).
    Meas:    z = [az, el]^T (bearing in OBSERVER BODY frame)

    API mirrors AgentUKF:
        ekf.predict(dt)
        ekf.predict(dt, u=u_world)
        ekf.predict(dt, u=u_body, u_frame='body', R_wb_tgt=R_wb_target)
        ekf.predict(dt, u=u_world, u_cov=Sigma_u)
        ekf.update(z, p_obs, R_wb)
    """
    def __init__(self, x0, P0, Q, R, dt):
        self.x  = np.asarray(x0, float).reshape(6)
        self.P  = _psd_enforce(np.asarray(P0, float))
        self.Q  = _psd_enforce(np.asarray(Q,  float))
        self.R  = _psd_enforce(np.asarray(R,  float))
        self.dt = float(dt)
        self.n  = 6

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
                u = np.asarray(R_wb_tgt, float).T @ u
            x_next = x_next + self.B_accel(dt) @ u
        return x_next

    # ----- measurement -----
    def h(self, x, p_obs, R_wb):
        p_tgt = x[:3]
        v_b = _body_bearing_from_world(p_obs, R_wb, p_tgt)
        az, el = _azel_from_body_vec(v_b)
        return np.array([az, el], float)

    def H_pos(self, x, p_obs, R_wb):
        """
        Jacobian H wrt state x = [p; v]. Only depends on position (first 3 cols).
        We derive d(az,el)/dx using chain rule:
          r = p - p_obs
          b_w = r / ||r||
          b_b = R_wb @ b_w
          z = [atan2(by,bx), atan2(bz, sqrt(bx^2+by^2))]
        Let J_bwr = d b_b / d r = R_wb @ ((I - b_w b_w^T) / ||r||)
        Then dz/dp = (dz/db_b) @ J_bwr
        """
        p = x[:3].astype(float)
        p_obs = np.asarray(p_obs, float)
        R_wb  = np.asarray(R_wb, float)

        r = p - p_obs
        n = np.linalg.norm(r)
        if n < 1e-12:
            # avoid singularity; linearize around a tiny offset along camera x-axis
            r = np.array([1.0, 0.0, 0.0])
            n = 1.0
        b_w = r / n
        b_b = R_wb @ b_w
        bx, by, bz = b_b

        # d b_b / d r
        I = np.eye(3)
        J_bwr = R_wb @ ((I - np.outer(b_w, b_w)) / n)  # (3x3)

        # dz / d b_b
        eps = 1e-12
        rho2 = bx*bx + by*by
        rho  = np.sqrt(max(rho2, eps))
        den_el = (rho2 + bz*bz)  # = 1 (since b_b is unit), but keep for clarity/robustness

        # ∂az/∂b = [-by/rho2, bx/rho2, 0]
        daz_db = np.array([ -by / max(rho2, eps),  bx / max(rho2, eps), 0.0 ])

        # ∂el/∂b using el = atan2(bz, rho)
        # d el = (rho * dbz - bz * drho) / (rho^2 + bz^2)
        # drho = (bx dbx + by dby)/rho  => ∂el/∂bx = (-bz * (bx/rho)) / den_el, similarly for by; ∂el/∂bz =  rho / den_el
        del_db = np.array([
            -bz * (bx / max(rho, eps)) / max(den_el, eps),
            -bz * (by / max(rho, eps)) / max(den_el, eps),
             rho / max(den_el, eps)
        ])

        DzDb = np.vstack([daz_db, del_db])  # (2x3)

        # chain: dz/dp = DzDb @ J_bwr
        H_p = DzDb @ J_bwr  # (2x3)

        # full H wrt x = [p; v]
        H = np.zeros((2, 6))
        H[:, :3] = H_p
        # H[:, 3:] = 0
        return H

    # ----- EKF steps -----
    def predict(self, dt=None, u=None, u_cov=None, u_frame='world', R_wb_tgt=None):
        """
        EKF prediction. If u is provided, apply affine term B u.
        If u_cov (3x3) is provided, add B * u_cov * B^T to P to model input uncertainty.
        """
        if dt is None:
            dt = self.dt

        # state
        self.x = self.f(self.x, dt=dt, u=u, u_frame=u_frame, R_wb_tgt=R_wb_tgt)

        # covariance
        Fd = self.F(dt)
        Pp = Fd @ self.P @ Fd.T + self.Q

        if u_cov is not None:
            Bu = self.B_accel(dt)
            U  = _psd_enforce(np.asarray(u_cov, float))
            Pp += Bu @ U @ Bu.T

        self.P = _psd_enforce(_symmetrize(Pp))

    def update(self, z, p_obs, R_wb):
        """
        Bearing (az, el) update in observer BODY frame.
        """
        z = np.asarray(z, float).reshape(2)
        # predicted measurement
        z_pred = self.h(self.x, p_obs, R_wb)
        # innovation (angle wrap)
        y = z - z_pred
        y[0] = _normalize_angle(y[0]); y[1] = _normalize_angle(y[1])

        # Jacobian
        H = self.H_pos(self.x, p_obs, R_wb)

        # innovation cov and Kalman gain
        S = H @ self.P @ H.T + self.R
        S = _psd_enforce(_symmetrize(S))
        K = self.P @ H.T @ np.linalg.inv(S + 1e-12*np.eye(2))

        # state/cov updates (Joseph form for numerical robustness)
        I = np.eye(self.n)
        self.x = self.x + K @ y
        IKH = (I - K @ H)
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
        self.P = _psd_enforce(_symmetrize(self.P))

        return self.x, self.P
