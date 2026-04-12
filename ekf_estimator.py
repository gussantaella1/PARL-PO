# ekf_estimator.py
from __future__ import annotations
import numpy as np

from dyn_models import hcw_mean_motion, hcw_discrete_mats, as_numpy_const

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
    Extended Kalman Filter with selectable linear dynamics observed by an agent.

    State:
      - 6D: x = [px, py, pz, vx, vy, vz]^T
      - 9D: x = [px, py, pz, vx, vy, vz, ax, ay, az]^T

    Process:
      - dyn='cv'  : constant-velocity with optional external accel input u
      - dyn='hcw' : HCW relative motion with optional external accel input u

    Meas:
      z = [az, el]^T (bearing in OBSERVER BODY frame)

    API mirrors AgentUKF:
        ekf.predict(dt)
        ekf.predict(dt, u=u_world)
        ekf.predict(dt, u=u_body, u_frame='body', R_wb_tgt=R_wb_target)
        ekf.predict(dt, u=u_world, u_cov=Sigma_u)
        ekf.update(z, p_obs, R_wb)
    """
    _GLOBAL_LINEARIZATION_CACHE = {}

    def __init__(
        self,
        x0,
        P0,
        Q,
        R,
        dt,
        dyn='cv',
        hcw=None,
        jacobian_mode='exact',
        linearization_group='default',
    ):
        self.x  = np.asarray(x0, float).reshape(-1)
        if self.x.size not in (6, 9):
            raise ValueError(f"AgentEKF expects 6D or 9D state, got {self.x.size}.")
        self.P  = _psd_enforce(np.asarray(P0, float))
        self.Q  = _psd_enforce(np.asarray(Q,  float))
        self.R  = _psd_enforce(np.asarray(R,  float))
        self.dt = float(dt)
        self.n  = int(self.x.size)

        self._dyn = (dyn or 'cv').lower()
        self._hcw_params = (hcw or {})
        self._jacobian_mode = self._normalize_jacobian_mode(jacobian_mode)
        self._linearization_group = str(linearization_group)
        self._Ad = None
        self._Bd = None
        if self._dyn == 'hcw':
            n = hcw_mean_motion(self._hcw_params)
            Ad_mx, Bd_mx = hcw_discrete_mats(n, self.dt)
            self._Ad = as_numpy_const(Ad_mx)
            self._Bd = as_numpy_const(Bd_mx)

    def _normalize_jacobian_mode(self, mode):
        key = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
        if key in {"frozen_first", "frozen_global"}:
            return "frozen"
        valid = {"exact", "frozen"}
        if key not in valid:
            raise ValueError(f"Unsupported EKF jacobian_mode='{mode}'. Expected one of {sorted(valid)}.")
        return key

    @classmethod
    def clear_global_linearization_cache(cls):
        cls._GLOBAL_LINEARIZATION_CACHE.clear()

    def _global_linearization_key(self):
        return (self._linearization_group, self.n, self._dyn, float(self.dt))

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

    def linear_dynamics_mats(self, dt=None):
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

    def f(self, x, dt=None, u=None, u_frame='world', R_wb_tgt=None):
        """
        Deterministic dynamics: x+ = A x + B u  (if u provided).
        If u_frame='body', convert u_body -> u_world using R_wb_tgt (world->body).
        """
        A, B = self.linear_dynamics_mats(dt)
        x_next = A @ x
        if u is not None:
            u = np.asarray(u, float).reshape(3)
            if u_frame == 'body':
                if R_wb_tgt is None:
                    raise ValueError("u given in 'body' frame but R_wb_tgt is None")
                # body -> world: v_w = R_wb^T v_b
                u = np.asarray(R_wb_tgt, float).T @ u
            x_next = x_next + B @ u
        return x_next

    # ----- measurement -----
    def h(self, x, p_obs, R_wb):
        p_tgt = x[:3]
        v_b = _body_bearing_from_world(p_obs, R_wb, p_tgt)
        az, el = _azel_from_body_vec(v_b)
        return np.array([az, el], float)

    def H_pos(self, x, p_obs, R_wb):
        """
        Jacobian H wrt state x. Only depends on position (first 3 cols).
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

        # full H wrt x; extra state dimensions have zero measurement sensitivity
        H = np.zeros((2, self.n))
        H[:, :3] = H_p
        return H

    def _compute_measurement_linearization(self, x_ref, p_obs, R_wb):
        H = self.H_pos(x_ref, p_obs, R_wb)
        offset = self.h(x_ref, p_obs, R_wb) - H @ x_ref
        return {
            "H": H,
            "offset": offset,
        }

    def _get_measurement_linearization(self, x, p_obs, R_wb):
        if self._jacobian_mode == "exact":
            return None

        key = self._global_linearization_key()
        if key not in self._GLOBAL_LINEARIZATION_CACHE:
            self._GLOBAL_LINEARIZATION_CACHE[key] = self._compute_measurement_linearization(x, p_obs, R_wb)
        return self._GLOBAL_LINEARIZATION_CACHE[key]

    def measurement_prediction(self, p_obs, R_wb):
        linearization = self._get_measurement_linearization(self.x, p_obs, R_wb)
        if linearization is None:
            return self.h(self.x, p_obs, R_wb)
        return linearization["H"] @ self.x + linearization["offset"]

    # ----- EKF steps -----
    def predict(self, dt=None, u=None, u_cov=None, u_frame='world', R_wb_tgt=None):
        """
        EKF prediction. If u is provided, apply affine term B u.
        If u_cov (3x3) is provided, add B * u_cov * B^T to P to model input uncertainty.
        """
        if dt is None:
            dt = self.dt

        if self._dyn == 'hcw' and (self._Ad is None or dt != self.dt):
            n = hcw_mean_motion(self._hcw_params)
            Ad_mx, Bd_mx = hcw_discrete_mats(n, dt)
            self._Ad = as_numpy_const(Ad_mx)
            self._Bd = as_numpy_const(Bd_mx)

        # state
        self.x = self.f(self.x, dt=dt, u=u, u_frame=u_frame, R_wb_tgt=R_wb_tgt)

        # covariance
        Ad, Bd = self.linear_dynamics_mats(dt)
        Pp = Ad @ self.P @ Ad.T + self.Q

        if u_cov is not None:
            U  = _psd_enforce(np.asarray(u_cov, float))
            Pp += Bd @ U @ Bd.T

        self.P = _psd_enforce(_symmetrize(Pp))
        self.dt = dt

    def update(self, z, p_obs, R_wb):
        """
        Bearing (az, el) update in observer BODY frame.
        """
        z = np.asarray(z, float).reshape(2)
        linearization = self._get_measurement_linearization(self.x, p_obs, R_wb)
        if linearization is None:
            z_pred = self.h(self.x, p_obs, R_wb)
            H = self.H_pos(self.x, p_obs, R_wb)
        else:
            H = linearization["H"]
            z_pred = H @ self.x + linearization["offset"]
        # innovation (angle wrap)
        y = z - z_pred
        y[0] = _normalize_angle(y[0]); y[1] = _normalize_angle(y[1])

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
