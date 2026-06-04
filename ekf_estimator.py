# ekf_estimator.py
from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is expected in this repo, but keep import-safe.
    torch = None

from dyn_models import as_numpy_const, hcw_discrete_mats, hcw_mean_motion


def _canonical_dyn_name(name: Any) -> str:
    key = str(name or "cv").strip().lower()
    if key in {"elliptic_ltv", "elliptical_ltv", "th", "tschauner_hempel"}:
        return "elliptic_ltv"
    return key


def _is_torch_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _symmetrize(M):
    if _is_torch_tensor(M):
        return 0.5 * (M + M.transpose(-1, -2))
    return 0.5 * (M + M.T)


def _psd_enforce(M, floor: float = 1e-12):
    """Force symmetric PSD by eigenvalue flooring."""
    M = _symmetrize(M)
    if _is_torch_tensor(M):
        eig, V = torch.linalg.eigh(M)
        eig = torch.clamp(eig, min=floor)
        return V @ torch.diag(eig) @ V.transpose(-1, -2)
    eig, V = np.linalg.eigh(M)
    eig = np.maximum(eig, floor)
    return (V * eig) @ V.T


def _normalize_angle(a):
    if _is_torch_tensor(a):
        return torch.remainder(a + np.pi, 2.0 * np.pi) - np.pi
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _body_bearing_from_world(p_obs, R_wb, p_tgt):
    """
    Unit vector from observer to target, expressed in OBSERVER BODY frame.
    R_wb maps world -> body (consistent with your animator).
    """
    if _is_torch_tensor(p_tgt) or _is_torch_tensor(p_obs) or _is_torch_tensor(R_wb):
        d_w = p_tgt - p_obs
        n = torch.linalg.norm(d_w)
        if float(n.item()) < 1e-12:
            d_w = torch.tensor([1.0, 0.0, 0.0], dtype=p_tgt.dtype, device=p_tgt.device)
            n = torch.tensor(1.0, dtype=p_tgt.dtype, device=p_tgt.device)
        b_w = d_w / n
        return R_wb @ b_w

    d_w = np.asarray(p_tgt, float) - np.asarray(p_obs, float)
    n = np.linalg.norm(d_w)
    if n < 1e-12:
        d_w = np.array([1.0, 0.0, 0.0], dtype=float)
        n = 1.0
    b_w = d_w / n
    return np.asarray(R_wb, float) @ b_w


def _azel_from_body_vec(vb):
    if _is_torch_tensor(vb):
        x, y, z = vb.unbind()
        rho = torch.sqrt(torch.clamp(x * x + y * y, min=1e-18))
        az = torch.atan2(y, x)
        el = torch.atan2(z, rho)
        return az, el

    x, y, z = vb
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.sqrt(max(x * x + y * y, 1e-18)))
    return az, el


class _BackendArrayView:
    """NumPy-friendly facade over the EKF's internal state/covariance tensors."""

    def __init__(self, owner: "AgentEKF", attr_name: str):
        self._owner = owner
        self._attr_name = attr_name

    def _value(self):
        return getattr(self._owner, self._attr_name)

    @property
    def shape(self):
        return tuple(self._value().shape)

    @property
    def size(self) -> int:
        return int(self._value().numel() if _is_torch_tensor(self._value()) else self._value().size)

    @property
    def ndim(self) -> int:
        return int(self._value().ndim)

    @property
    def dtype(self):
        return self._value().dtype

    def copy(self):
        return np.array(self, copy=True)

    def __array__(self, dtype=None):
        arr = self._owner._to_numpy(self._value())
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    def __getitem__(self, key):
        return self._owner._backend_value_to_python(self._value()[key])

    def __setitem__(self, key, value):
        self._owner._assign_backend_slice(self._attr_name, key, value)

    def __len__(self):
        return len(self._value())

    def __repr__(self):
        return repr(np.asarray(self))


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

    @staticmethod
    def _numel(value: Any) -> int:
        if _is_torch_tensor(value):
            return int(value.numel())
        return int(np.asarray(value).size)

    def __init__(
        self,
        x0,
        P0,
        Q,
        R,
        dt,
        dyn="cv",
        hcw=None,
        ltv=None,
        jacobian_mode="exact",
        linearization_group="default",
        use_torch_backend: bool = False,
        device: str | None = None,
    ):
        self._use_torch_backend = bool(use_torch_backend)
        self._device = self._resolve_device(device) if self._use_torch_backend else None
        self._torch_dtype = torch.float64 if torch is not None else None

        self._x = self._vector_to_backend(x0)
        x_numel = self._numel(self._x)
        if x_numel not in (6, 9):
            raise ValueError(
                f"AgentEKF expects 6D or 9D state, got {x_numel}."
            )
        self.n = x_numel

        self._P = _psd_enforce(self._matrix_to_backend(P0, shape=(self.n, self.n)))
        self._Q = _psd_enforce(self._matrix_to_backend(Q, shape=(self.n, self.n)))
        self._R = _psd_enforce(self._matrix_to_backend(R, shape=(2, 2)))
        self.dt = float(dt)

        self._dyn = _canonical_dyn_name(dyn)
        self._hcw_params = (hcw or {})
        self._ltv_params = (ltv or {})
        self._jacobian_mode = self._normalize_jacobian_mode(jacobian_mode)
        self._linearization_group = str(linearization_group)
        self._Ad = None
        self._Bd = None
        self._Ad_seq = None
        self._Bd_seq = None
        self._predict_step = 0
        if self._dyn == "hcw":
            n = hcw_mean_motion(self._hcw_params)
            Ad_mx, Bd_mx = hcw_discrete_mats(n, self.dt)
            self._Ad = self._matrix_to_backend(as_numpy_const(Ad_mx))
            self._Bd = self._matrix_to_backend(as_numpy_const(Bd_mx))
        elif self._dyn == "elliptic_ltv":
            Ad_seq = self._ltv_params.get("Ad_seq", None)
            Bd_seq = self._ltv_params.get("Bd_seq", None)
            if Ad_seq is None or Bd_seq is None:
                raise ValueError("AgentEKF dyn='elliptic_ltv' requires ltv['Ad_seq'] and ltv['Bd_seq'].")
            self._Ad_seq = self._to_backend(Ad_seq, copy=True)
            self._Bd_seq = self._to_backend(Bd_seq, copy=True)
            if self._Ad_seq.ndim != 3 or tuple(self._Ad_seq.shape[1:]) != (6, 6):
                raise ValueError(
                    f"Expected ltv['Ad_seq'] to have shape (N, 6, 6), got {tuple(self._Ad_seq.shape)}."
                )
            if self._Bd_seq.ndim != 3 or tuple(self._Bd_seq.shape[1:]) != (6, 3):
                raise ValueError(
                    f"Expected ltv['Bd_seq'] to have shape (N, 6, 3), got {tuple(self._Bd_seq.shape)}."
                )
            if int(self._Ad_seq.shape[0]) != int(self._Bd_seq.shape[0]):
                raise ValueError(
                    "ltv['Ad_seq'] and ltv['Bd_seq'] must have the same number of time steps."
                )

    def _resolve_device(self, device: str | None):
        if torch is None:
            raise RuntimeError("EKF torch backend requested, but PyTorch could not be imported.")
        resolved = torch.device(device or "cpu")
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"EKF torch backend requested on device='{resolved}', but CUDA is not available."
            )
        return resolved

    def _to_backend(self, value: Any, *, copy: bool = False):
        if self._use_torch_backend:
            if _is_torch_tensor(value):
                out = value.to(device=self._device, dtype=self._torch_dtype)
                return out.clone() if copy else out
            out = torch.as_tensor(value, dtype=self._torch_dtype, device=self._device)
            return out.clone() if copy else out
        return np.array(value, dtype=float, copy=copy) if copy else np.asarray(value, dtype=float)

    def _vector_to_backend(self, value: Any):
        out = self._to_backend(value, copy=True).reshape(-1)
        return out

    def _matrix_to_backend(self, value: Any, shape: tuple[int, int] | None = None):
        out = self._to_backend(value, copy=True)
        if out.ndim != 2:
            raise ValueError(f"Expected a matrix, got ndim={out.ndim}.")
        if shape is not None and tuple(out.shape) != tuple(shape):
            raise ValueError(f"Expected matrix shape {shape}, got {tuple(out.shape)}.")
        return out

    def _to_numpy(self, value: Any) -> np.ndarray:
        if _is_torch_tensor(value):
            return value.detach().cpu().numpy().copy()
        return np.asarray(value, dtype=float).copy()

    def _backend_value_to_python(self, value: Any):
        if _is_torch_tensor(value):
            if value.ndim == 0:
                return float(value.item())
            return value.detach().cpu().numpy().copy()
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return float(arr.item())
        return arr.copy()

    def _assign_backend_slice(self, attr_name: str, key: Any, value: Any):
        target = getattr(self, attr_name)
        if self._use_torch_backend:
            target[key] = self._to_backend(value)
        else:
            target[key] = np.asarray(value, dtype=float)

    def _zeros(self, shape: tuple[int, ...]):
        if self._use_torch_backend:
            return torch.zeros(shape, dtype=self._torch_dtype, device=self._device)
        return np.zeros(shape, dtype=float)

    def _eye(self, n: int):
        if self._use_torch_backend:
            return torch.eye(n, dtype=self._torch_dtype, device=self._device)
        return np.eye(n, dtype=float)

    @property
    def x(self):
        return _BackendArrayView(self, "_x")

    @x.setter
    def x(self, value):
        x_new = self._vector_to_backend(value)
        if self._numel(x_new) != self.n:
            raise ValueError(f"Expected x to have length {self.n}.")
        self._x = x_new

    @property
    def P(self):
        return _BackendArrayView(self, "_P")

    @P.setter
    def P(self, value):
        self._P = _psd_enforce(self._matrix_to_backend(value, shape=(self.n, self.n)))

    @property
    def Q(self):
        return _BackendArrayView(self, "_Q")

    @Q.setter
    def Q(self, value):
        self._Q = _psd_enforce(self._matrix_to_backend(value, shape=(self.n, self.n)))

    @property
    def R(self):
        return _BackendArrayView(self, "_R")

    @R.setter
    def R(self, value):
        self._R = _psd_enforce(self._matrix_to_backend(value, shape=(2, 2)))

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
        backend_key = "torch" if self._use_torch_backend else "numpy"
        device_key = str(self._device) if self._use_torch_backend else "cpu"
        return (self._linearization_group, self.n, self._dyn, float(self.dt), backend_key, device_key)

    def F(self, dt=None):
        if dt is None:
            dt = self.dt
        F = self._eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    def B_accel(self, dt=None):
        """Input matrix for acceleration input u (3-vector)."""
        if dt is None:
            dt = self.dt
        B = self._zeros((6, 3))
        B[0:3, 0:3] = 0.5 * (dt ** 2) * self._eye(3)
        B[3:6, 0:3] = dt * self._eye(3)
        return B

    def _ltv_step_mats(self, step_index: int | None = None):
        if self._Ad_seq is None or self._Bd_seq is None:
            raise RuntimeError("LTV dynamics selected, but Ad/Bd sequences were not initialized.")
        step = self._predict_step if step_index is None else int(step_index)
        max_idx = int(self._Ad_seq.shape[0] - 1)
        idx = min(max(step, 0), max_idx)
        return self._Ad_seq[idx], self._Bd_seq[idx]

    def linear_dynamics_mats(self, dt=None, step_index: int | None = None):
        if dt is None:
            dt = self.dt
        if self._dyn == "hcw":
            if dt != self.dt or self._Ad is None:
                n = hcw_mean_motion(self._hcw_params)
                Ad_mx, Bd_mx = hcw_discrete_mats(n, dt)
                Ad6 = self._matrix_to_backend(as_numpy_const(Ad_mx))
                Bd6 = self._matrix_to_backend(as_numpy_const(Bd_mx))
            else:
                Ad6, Bd6 = self._Ad, self._Bd
        elif self._dyn == "elliptic_ltv":
            Ad6, Bd6 = self._ltv_step_mats(step_index=step_index)
        else:
            Ad6, Bd6 = self.F(dt), self.B_accel(dt)

        if self.n == 6:
            return Ad6, Bd6

        A = self._eye(9)
        A[:6, :6] = Ad6
        A[:6, 6:9] = Bd6
        B = self._zeros((9, 3))
        B[:6, :] = Bd6
        return A, B

    def _f_backend(self, x, dt=None, u=None, u_frame="world", R_wb_tgt=None, step_index: int | None = None):
        """
        Deterministic dynamics: x+ = A x + B u  (if u provided).
        If u_frame='body', convert u_body -> u_world using R_wb_tgt (world->body).
        """
        A, B = self.linear_dynamics_mats(dt, step_index=step_index)
        x_state = self._vector_to_backend(x)
        x_next = A @ x_state
        if u is not None:
            u_vec = self._vector_to_backend(u)
            if self._numel(u_vec) != 3:
                raise ValueError("EKF control input u must be a 3-vector.")
            if u_frame == "body":
                if R_wb_tgt is None:
                    raise ValueError("u given in 'body' frame but R_wb_tgt is None")
                u_vec = self._matrix_to_backend(R_wb_tgt, shape=(3, 3)).transpose(-1, -2) @ u_vec
            x_next = x_next + B @ u_vec
        return x_next

    def f(self, x, dt=None, u=None, u_frame="world", R_wb_tgt=None, step_index: int | None = None):
        return self._to_numpy(
            self._f_backend(x, dt=dt, u=u, u_frame=u_frame, R_wb_tgt=R_wb_tgt, step_index=step_index)
        )

    def _h_backend(self, x, p_obs, R_wb):
        x_state = self._vector_to_backend(x)
        p_tgt = x_state[:3]
        p_obs_b = self._vector_to_backend(p_obs)
        R_wb_b = self._matrix_to_backend(R_wb, shape=(3, 3))
        v_b = _body_bearing_from_world(p_obs_b, R_wb_b, p_tgt)
        az, el = _azel_from_body_vec(v_b)
        if self._use_torch_backend:
            return torch.stack([az, el])
        return np.array([az, el], dtype=float)

    def h(self, x, p_obs, R_wb):
        return self._to_numpy(self._h_backend(x, p_obs, R_wb))

    def _H_pos_backend(self, x, p_obs, R_wb):
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
        x_state = self._vector_to_backend(x)
        p = x_state[:3]
        p_obs_b = self._vector_to_backend(p_obs)
        R_wb_b = self._matrix_to_backend(R_wb, shape=(3, 3))

        r = p - p_obs_b
        n = torch.linalg.norm(r) if self._use_torch_backend else np.linalg.norm(r)
        if float(n.item() if _is_torch_tensor(n) else n) < 1e-12:
            r = self._vector_to_backend([1.0, 0.0, 0.0])
            n = self._to_backend(1.0)
        b_w = r / n
        b_b = R_wb_b @ b_w
        if self._use_torch_backend:
            bx, by, bz = b_b.unbind()
            I = self._eye(3)
            J_bwr = R_wb_b @ ((I - torch.outer(b_w, b_w)) / n)
            eps = self._to_backend(1e-12)
            rho2 = bx * bx + by * by
            rho = torch.sqrt(torch.clamp(rho2, min=float(eps.item())))
            den_el = rho2 + bz * bz
            daz_db = torch.stack(
                [
                    -by / torch.clamp(rho2, min=float(eps.item())),
                    bx / torch.clamp(rho2, min=float(eps.item())),
                    self._to_backend(0.0),
                ]
            )
            del_db = torch.stack(
                [
                    -bz * (bx / torch.clamp(rho, min=float(eps.item())))
                    / torch.clamp(den_el, min=float(eps.item())),
                    -bz * (by / torch.clamp(rho, min=float(eps.item())))
                    / torch.clamp(den_el, min=float(eps.item())),
                    rho / torch.clamp(den_el, min=float(eps.item())),
                ]
            )
            DzDb = torch.stack([daz_db, del_db], dim=0)
        else:
            bx, by, bz = b_b
            I = self._eye(3)
            J_bwr = R_wb_b @ ((I - np.outer(b_w, b_w)) / n)
            eps = 1e-12
            rho2 = bx * bx + by * by
            rho = np.sqrt(max(rho2, eps))
            den_el = rho2 + bz * bz
            daz_db = np.array([-by / max(rho2, eps), bx / max(rho2, eps), 0.0], dtype=float)
            del_db = np.array(
                [
                    -bz * (bx / max(rho, eps)) / max(den_el, eps),
                    -bz * (by / max(rho, eps)) / max(den_el, eps),
                    rho / max(den_el, eps),
                ],
                dtype=float,
            )
            DzDb = np.vstack([daz_db, del_db])

        H_p = DzDb @ J_bwr
        H = self._zeros((2, self.n))
        H[:, :3] = H_p
        return H

    def H_pos(self, x, p_obs, R_wb):
        return self._to_numpy(self._H_pos_backend(x, p_obs, R_wb))

    def _compute_measurement_linearization(self, x_ref, p_obs, R_wb):
        H = self._H_pos_backend(x_ref, p_obs, R_wb)
        x_ref_b = self._vector_to_backend(x_ref)
        offset = self._h_backend(x_ref_b, p_obs, R_wb) - H @ x_ref_b
        return {"H": H, "offset": offset}

    def _get_measurement_linearization(self, x, p_obs, R_wb):
        if self._jacobian_mode == "exact":
            return None

        key = self._global_linearization_key()
        if key not in self._GLOBAL_LINEARIZATION_CACHE:
            self._GLOBAL_LINEARIZATION_CACHE[key] = self._compute_measurement_linearization(x, p_obs, R_wb)
        return self._GLOBAL_LINEARIZATION_CACHE[key]

    def measurement_prediction(self, p_obs, R_wb):
        linearization = self._get_measurement_linearization(self._x, p_obs, R_wb)
        if linearization is None:
            return self._to_numpy(self._h_backend(self._x, p_obs, R_wb))
        return self._to_numpy(linearization["H"] @ self._x + linearization["offset"])

    def predict(self, dt=None, u=None, u_cov=None, u_frame="world", R_wb_tgt=None):
        """
        EKF prediction. If u is provided, apply affine term B u.
        If u_cov (3x3) is provided, add B * u_cov * B^T to P to model input uncertainty.
        """
        if dt is None:
            dt = self.dt

        if self._dyn == "hcw" and (self._Ad is None or dt != self.dt):
            n = hcw_mean_motion(self._hcw_params)
            Ad_mx, Bd_mx = hcw_discrete_mats(n, dt)
            self._Ad = self._matrix_to_backend(as_numpy_const(Ad_mx))
            self._Bd = self._matrix_to_backend(as_numpy_const(Bd_mx))

        step_index = self._predict_step if self._dyn == "elliptic_ltv" else None
        self._x = self._f_backend(
            self._x,
            dt=dt,
            u=u,
            u_frame=u_frame,
            R_wb_tgt=R_wb_tgt,
            step_index=step_index,
        )

        Ad, Bd = self.linear_dynamics_mats(dt, step_index=step_index)
        Pp = Ad @ self._P @ Ad.transpose(-1, -2) + self._Q

        if u_cov is not None:
            U = _psd_enforce(self._matrix_to_backend(u_cov, shape=(3, 3)))
            Pp = Pp + Bd @ U @ Bd.transpose(-1, -2)

        self._P = _psd_enforce(_symmetrize(Pp))
        self.dt = float(dt)
        if self._dyn == "elliptic_ltv":
            self._predict_step += 1

    def update(self, z, p_obs, R_wb):
        """
        Bearing (az, el) update in observer BODY frame.
        """
        z_b = self._vector_to_backend(z)
        if self._numel(z_b) != 2:
            raise ValueError("EKF measurement z must be a 2-vector [az, el].")

        linearization = self._get_measurement_linearization(self._x, p_obs, R_wb)
        if linearization is None:
            z_pred = self._h_backend(self._x, p_obs, R_wb)
            H = self._H_pos_backend(self._x, p_obs, R_wb)
        else:
            H = linearization["H"]
            z_pred = H @ self._x + linearization["offset"]

        y = z_b - z_pred
        y[0] = _normalize_angle(y[0])
        y[1] = _normalize_angle(y[1])

        S = H @ self._P @ H.transpose(-1, -2) + self._R
        S = _psd_enforce(_symmetrize(S))
        if self._use_torch_backend:
            S_reg = S + 1e-12 * self._eye(2)
            K = self._P @ H.transpose(-1, -2) @ torch.linalg.inv(S_reg)
        else:
            K = self._P @ H.T @ np.linalg.inv(S + 1e-12 * self._eye(2))

        I = self._eye(self.n)
        self._x = self._x + K @ y
        IKH = I - K @ H
        self._P = IKH @ self._P @ IKH.transpose(-1, -2) + K @ self._R @ K.transpose(-1, -2)
        self._P = _psd_enforce(_symmetrize(self._P))

        return self._to_numpy(self._x), self._to_numpy(self._P)
