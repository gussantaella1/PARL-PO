# ukf_estimator.py
from __future__ import annotations
import numpy as np

# ---- This file expects the following helper functions to exist elsewhere ----
# def Recef2enu(r0G: np.ndarray) -> np.ndarray: ...
# def wahbaSolver(aVec: np.ndarray, vIMat: np.ndarray, vBMat: np.ndarray) -> np.ndarray: ...
# def euler2dcm(e: np.ndarray) -> np.ndarray: ...
# def h_meas(x: np.ndarray, meas_noise: np.ndarray, RBIBark: np.ndarray,
#            rXIMat: np.ndarray, mcVeck: np.ndarray, P) -> np.ndarray: ...
# def f_dynamics(x: np.ndarray, u: np.ndarray, proc_noise: np.ndarray,
#                delt: float, RBIHatk: np.ndarray, P) -> np.ndarray: ...
# -----------------------------------------------------------------------------

def _blkdiag(*mats: np.ndarray) -> np.ndarray:
    """Lightweight block-diagonal builder (avoids scipy dependency)."""
    rows = sum(m.shape[0] for m in mats)
    cols = sum(m.shape[1] for m in mats)
    out = np.zeros((rows, cols), dtype=float)
    r, c = 0, 0
    for m in mats:
        rr, cc = m.shape
        out[r:r+rr, c:c+cc] = m
        r += rr
        c += cc
    return out

def _safe_cholesky(A: np.ndarray, jitter: float = 1e-10, max_tries: int = 5) -> np.ndarray:
    """Cholesky with tiny diagonal jitter for numerical robustness (returns lower-triangular)."""
    k = 0
    while k < max_tries:
        try:
            return np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            A = A + (jitter * (10.0 ** k)) * np.eye(A.shape[0])
            k += 1
    # Final try—raise if still not PD
    return np.linalg.cholesky(A)

def _ukf_weights(n_aug: int, alpha: float, beta: float, kappa: float):
    """Compute UKF weights and scaling for a given augmented dimension."""
    lam = alpha**2 * (kappa + n_aug) - n_aug
    c = np.sqrt(n_aug + lam)
    Wm0 = lam / (n_aug + lam)
    Wc0 = Wm0 + (1.0 - alpha**2 + beta)
    Wmi = 1.0 / (2.0 * (n_aug + lam))
    Wci = Wmi
    return lam, c, Wm0, Wmi, Wc0, Wci

class StateEstimatorUKF:
    """
    Unscented Kalman Filter for a quadcopter state:
    x = [ rI(3), vI(3), e(3), ba(3), bg(3) ]  (nx=15)
    Process noise v has nv=12, grouped as [Qg, Qg2, Qa, Qa2] blocks (3x3 each) per your MATLAB.
    """
    def __init__(self, P, alphaUKF: float = 1e-3, betaUKF: float = 2.0, kappaUKF: float = 0.0):
        self.P = P
        self.nx = 15
        self.nv = 12
        self.alpha = alphaUKF
        self.beta = betaUKF
        self.kappa = kappaUKF

        # Persistent state
        self.xBark: np.ndarray | None = None   # (15,)
        self.PBark: np.ndarray | None = None   # (15,15)
        self.RBIBark: np.ndarray | None = None # (3,3)

        # Small numeric epsilon (same intent as MATLAB)
        self.eps = 1e-8

    def reset(self):
        self.xBark = None
        self.PBark = None
        self.RBIBark = None

    def step(self, S, M, P=None):
        """
        One complete UKF cycle: measurement update at tk, then propagate to tk+Δt.
        Inputs:
          S: object with fields
             - rXIMat (Nf x 3) feature coordinates in I
             - delt (float) measurement update interval
          M: object with fields
             - tk (float) time of measurements
             - rpGtilde (3,) primary GNSS ECEF rel. ref antenna
             - rbGtilde (3,) secondary GNSS ECEF rel. primary antenna (norm=b)
             - rxMat (Nf x 2) pixel measurements (NaN,NaN if not visible)
             - ftildeB (3,) accelerometer specific force (m/s^2)
             - omegaBtilde (3,) gyro rates (rad/s)
          P: (optional) parameter object with fields:
             - sensorParams: holds camera, GNSS, IMU, noise params (see MATLAB names)
        Returns:
          E: dict with keys {"statek": {...}, "Pk": (15,15)}
        """
        if P is None:
            P = self.P

        # --- Setup and constants ---
        RIG = Recef2enu(P.sensorParams.r0G)   # ECEF->I rotation
        raB = P.sensorParams.raB              # (3x2) antenna locations in body frame
        rbB = raB[:, 1] - raB[:, 0]
        rbBu = rbB / np.linalg.norm(rbB)
        rpB = raB[:, 0]
        e3 = np.array([0.0, 0.0, 1.0])

        nx = self.nx
        nv = self.nv
        alpha, beta, kappa = self.alpha, self.beta, self.kappa

        # --- Convert GNSS to I frame ---
        rpItilde = RIG @ M.rpGtilde          # primary antenna in I
        rbItilde = RIG @ M.rbGtilde          # baseline vector in I
        rbItildeu = rbItilde / np.linalg.norm(rbItilde)

        # --- Initialize (first call) ---
        if self.xBark is None:
            # Attitude init via Wahba using {rb, e3} directions
            vIMat = np.vstack([rbItildeu, e3])  # shape (2,3)
            vBMat = np.vstack([rbBu,      e3])  # shape (2,3)
            aVec  = np.ones(2)
            self.RBIBark = wahbaSolver(aVec, vIMat, vBMat)  # (3,3)

            rIBark = rpItilde - self.RBIBark.T @ rpB
            self.xBark = np.concatenate([rIBark, np.zeros(12)])

            # Steady-state bias covariances
            # Allow Qa2, Qg2 to be matrices; element-wise division by scalar is OK in numpy.
            QbaSS = P.sensorParams.Qa2 / (1.0 - P.sensorParams.alphaa**2)
            QbgSS = P.sensorParams.Qg2 / (1.0 - P.sensorParams.alphag**2)

            # Initial covariance (diag of blocks mimics MATLAB)
            RpL = P.sensorParams.RpL
            sigmab = P.sensorParams.sigmab

            self.PBark = _blkdiag(
                2.0 * np.diag(np.diag(RpL)),          # position
                0.001 * np.eye(3),                    # velocity
                2.0 * (sigmab**2) * np.eye(3),        # attitude error angles (e)
                QbaSS,                                 # accel bias
                QbgSS                                  # gyro bias
            )

        # ---------------- Measurement update at tk ----------------
        # Assemble measurement vector zk: [ rpItilde (3), rbItildeu (3),  {feature unit vectors in C (3 each)} ]
        zk_parts = [rpItilde, rbItildeu]
        Rc_list = []         # list of 3x3 covariances for each visible feature’s unit vector
        rxMat = M.rxMat      # (Nf x 2)
        Nf = rxMat.shape[0]
        mcVeck = np.zeros(Nf)  # 1 if feature i is used, else 0

        pixelSize = P.sensorParams.pixelSize
        f = P.sensorParams.f
        Rc_pix = P.sensorParams.Rc  # 2x2 pixel noise covariance (we use Rc(1,1) like MATLAB)

        for i in range(Nf):
            if not np.isnan(rxMat[i, 0]):
                mcVeck[i] = 1.0
                viCtilde = np.array([pixelSize * rxMat[i, 0],
                                     pixelSize * rxMat[i, 1],
                                     f], dtype=float)
                nrm = np.linalg.norm(viCtilde)
                viCtildeu = viCtilde / (nrm + 0.0)

                zk_parts.append(viCtildeu)

                # Unit-vector covariance on the sphere (projected) + epsilon*I
                sigmac = np.sqrt(Rc_pix[0, 0]) * pixelSize / nrm
                RcC = sigmac**2 * (np.eye(3) - np.outer(viCtildeu, viCtildeu)) + self.eps * np.eye(3)
                Rc_list.append(RcC)

        zk = np.concatenate(zk_parts)  # length nz

        # Measurement noise block Rk = blkdiag(RpI, RbI, RcC_1, ..., RcC_Nfk)
        RpI = P.sensorParams.RpL

        # Baseline direction covariance (in I) around predicted direction
        rbIu_pred = self.RBIBark.T @ rbBu
        RbI = (P.sensorParams.sigmab**2) * (np.eye(3) - np.outer(rbIu_pred, rbIu_pred)) + self.eps * np.eye(3)

        if len(Rc_list) > 0:
            Rk = _blkdiag(RpI, RbI, *_ensure_3x3_list(Rc_list))
        else:
            Rk = _blkdiag(RpI, RbI)

        nz = zk.shape[0]

        # UKF weights for measurement update (augmented dimension = nx + nz)
        _lam_u, c_u, Wm0_u, Wmi_u, Wc0_u, Wci_u = _ukf_weights(nx + nz, alpha, beta, kappa)

        # Augmented a priori state and cov
        xBarAugk = np.concatenate([self.xBark, np.zeros(nz)])
        PBarAugk = _blkdiag(self.PBark, Rk)
        SxBar = _safe_cholesky(PBarAugk)  # lower-triangular

        # Sigma points in augmented space
        n_aug_u = nx + nz
        sp0_u = xBarAugk
        spMat_u = np.zeros((n_aug_u, 2 * n_aug_u))
        # plus/minus
        for j in range(n_aug_u):
            col = c_u * SxBar[:, j]
            spMat_u[:, j]           = sp0_u +  col
            spMat_u[:, j + n_aug_u] = sp0_u -  col

        # Push sigma points through measurement function
        def eval_h(sp):
            x = sp[:nx]
            meas_noise = sp[nx:]
            return h_meas(x, meas_noise, self.RBIBark, S.rXIMat, mcVeck, P)

        zp0 = eval_h(sp0_u)
        zpMat = np.column_stack([eval_h(spMat_u[:, i]) for i in range(2 * n_aug_u)])

        # Recombine sigma points (measurement mean)
        zBark = Wm0_u * zp0 + Wmi_u * np.sum(zpMat, axis=1)

        # Covariances
        def outer(a, b): return np.outer(a, b)

        dz0 = zp0 - zBark
        Pzz = Wc0_u * outer(dz0, dz0)
        Pxz = Wc0_u * outer(sp0_u[:nx] - self.xBark, dz0)
        for i in range(2 * n_aug_u):
            dzi = zpMat[:, i] - zBark
            dxi = spMat_u[:nx, i] - self.xBark
            Pzz += Wci_u * outer(dzi, dzi)
            Pxz += Wci_u * outer(dxi, dzi)

        # Kalman update (solve instead of explicit inverse)
        # K = Pxz @ inv(Pzz)
        K = np.linalg.solve(Pzz.T, Pxz.T).T  # equivalent to Pxz @ inv(Pzz)
        innov = zk - zBark
        xHatk = self.xBark + K @ innov
        Pk = self.PBark - K @ Pzz @ K.T

        # Package output state at tk
        statek = {}
        statek["rI"] = xHatk[0:3]
        statek["vI"] = xHatk[3:6]
        ek = xHatk[6:9]
        statek["RBI"] = euler2dcm(ek) @ self.RBIBark
        statek["ba"] = xHatk[9:12]
        statek["bg"] = xHatk[12:15]
        statek["omegaB"] = M.omegaBtilde - statek["bg"]

        E = {"statek": statek, "Pk": Pk}

        # ---------------- Propagate to tk+Δt ----------------
        RBIHatk = statek["RBI"].copy()

        # Set error Euler angles to zero after attitude injection
        xHatk_prop = xHatk.copy()
        xHatk_prop[6:9] = 0.0

        # Process noise covariance Qk (12x12) = blkdiag(Qg, Qg2, Qa, Qa2)
        Qk = _blkdiag(P.sensorParams.Qg, P.sensorParams.Qg2,
                      P.sensorParams.Qa, P.sensorParams.Qa2)

        # Augment with process noise
        xHatAugk = np.concatenate([xHatk_prop, np.zeros(self.nv)])
        PAugk = _blkdiag(Pk, Qk)
        Sx = _safe_cholesky(PAugk)  # lower

        # UKF weights for propagation (augmented dimension = nx + nv)
        _lam_p, c_p, Wm0_p, Wmi_p, Wc0_p, Wci_p = _ukf_weights(nx + nv, alpha, beta, kappa)

        # Sigma points for propagation
        n_aug_p = nx + nv
        sp0_p = xHatAugk
        spMat_p = np.zeros((n_aug_p, 2 * n_aug_p))
        for j in range(n_aug_p):
            col = c_p * Sx[:, j]
            spMat_p[:, j]           = sp0_p +  col
            spMat_p[:, j + n_aug_p] = sp0_p -  col

        # Push through dynamics
        u_k = np.concatenate([M.omegaBtilde, M.ftildeB])  # (6,)
        def eval_f(sp):
            x = sp[:nx]
            proc_noise = sp[nx:]
            return f_dynamics(x, u_k, proc_noise, S.delt, RBIHatk, P)

        xp0 = eval_f(sp0_p)
        xpMat = np.column_stack([eval_f(spMat_p[:, i]) for i in range(2 * n_aug_p)])

        # Recombine dynamic sigma points
        xBarkp1 = Wm0_p * xp0 + Wmi_p * np.sum(xpMat, axis=1)

        dx0 = xp0 - xBarkp1
        PBarkp1 = Wc0_p * outer(dx0, dx0)
        for i in range(2 * n_aug_p):
            dxi = xpMat[:, i] - xBarkp1
            PBarkp1 += Wci_p * outer(dxi, dxi)

        # Inject attitude error and zero the error angles (same as MATLAB)
        ekp1 = xBarkp1[6:9]
        RBIBarkp1 = euler2dcm(ekp1) @ RBIHatk
        xBarkp1 = xBarkp1.copy()
        xBarkp1[6:9] = 0.0

        # Persist for next call
        self.RBIBark = RBIBarkp1
        self.xBark = xBarkp1
        self.PBark = PBarkp1

        return E

def _ensure_3x3_list(mats):
    """Helpful assertion to keep shapes tidy."""
    out = []
    for m in mats:
        m = np.asarray(m, dtype=float)
        assert m.shape == (3, 3), f"Expected 3x3, got {m.shape}"
        out.append(m)
    return out
