# nash_ipopt_solver.py
# -------------------------------------------------------------
# One-step approximate Nash equilibrium solver using IPOPT/CasADi.
#
# Intended to be called from DiffNashLayer as:
#
#   from nash_ipopt_solver import solve_nash_ipopt
#   u1_opt, u2_opt = solve_nash_ipopt(x1, x2, params)
#
# where:
#   - x1, x2 are np.ndarray of shape (2D,) -> [p, v]
#   - params is a dict containing dynamics, arena, and weight settings.
# -------------------------------------------------------------

from __future__ import annotations

from typing import Tuple, Dict, Any
import numpy as np


def solve_nash_ipopt(
    x1: np.ndarray,
    x2: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-step approximate Nash equilibrium via IPOPT/CasADi.

    Game (single-step, continuous actions):
    --------------------------------------
    Dynamics:
        x1⁺ = Ad x1 + Bd u1
        x2⁺ = Ad x2 + Bd u2
    where:
        - x1, x2 ∈ R^{2D} (pos, vel stacked)
        - u1, u2 ∈ R^D       (control accelerations)

    Roles:
        - Player 1 (defender):
            * Wants the attacker far from the center.
            * Wants to avoid getting too close to the center.
            * Regularizes its own control effort.

        - Player 2 (attacker):
            * Wants the attacker close to the center.
            * Wants to avoid colliding with the defender.
            * Regularizes its own control effort.

    We seek a local Nash equilibrium by solving the first-order conditions:
        ∂J1/∂u1 = 0, ∂J2/∂u2 = 0
    via minimizing the penalty:
        F(u1, u2) = 0.5 * (‖∂J1/∂u1‖² + ‖∂J2/∂u2‖²)
    under box constraints on u1, u2.

    Parameters (params dict)
    ------------------------
    Required keys:
        - "Ad": (2D x 2D) numpy array
        - "Bd": (2D x D)  numpy array
        - "center": (D,) numpy array (arena center in position coordinates)
        - "umax": scalar control bound (|u_i| ≤ umax componentwise)
        - "R": arena radius (used for spatial normalization)

    Optional weight keys (default values provided):
        Defender:
            - "w_def_far_center"     (default +1.0)
            - "w_def_keepout_center" (default 10.0)
            - "w_def_u"              (default 0.1)

        Attacker:
            - "w_att_center"         (default 1.0)
            - "w_att_avoid_def"      (default 10.0)
            - "w_att_u"              (default 0.1)

        Barrier epsilons:
            - "eps_center"           (default 1e-3)
            - "eps_rel"              (default 1e-3)

    Returns
    -------
    u1_opt : np.ndarray, shape (D,)
        Defender approximate Nash action for this step.
    u2_opt : np.ndarray, shape (D,)
        Attacker approximate Nash action for this step.

    Fallback behaviour
    ------------------
    If CasADi or IPOPT is not available or the solver fails, returns zeros
    (i.e., both players apply u = 0 for this step).
    """
    # Lazy import to avoid forcing CasADi on all environments
    try:
        import casadi as ca
    except ImportError:
        D = x1.shape[0] // 2
        return np.zeros(D, dtype=np.float32), np.zeros(D, dtype=np.float32)

    # ----------------- unpack dimensions & core params -----------------
    x1 = np.asarray(x1, dtype=float).ravel()
    x2 = np.asarray(x2, dtype=float).ravel()
    assert x1.shape == x2.shape, "x1 and x2 must have same shape"
    assert x1.ndim == 1, "x1, x2 must be 1D state vectors"

    D = x1.shape[0] // 2  # pos + vel per agent

    Ad = np.asarray(params["Ad"], dtype=float)
    Bd = np.asarray(params["Bd"], dtype=float)
    center = np.asarray(params["center"], dtype=float).ravel()
    umax = float(params["umax"])
    R = float(params.get("R", 1.0))
    if R <= 0:
        R = 1.0  # safety against degenerate radius

    # Defender weights
    w_def_far_center     = float(params.get("w_def_far_center",     +1.0))
    w_def_keepout_center = float(params.get("w_def_keepout_center", 10.0))
    w_def_u              = float(params.get("w_def_u",              0.1))

    # Attacker weights
    w_att_center         = float(params.get("w_att_center",         1.0))
    w_att_avoid_def      = float(params.get("w_att_avoid_def",     10.0))
    w_att_u              = float(params.get("w_att_u",              0.1))

    # Barrier epsilons
    eps_center = float(params.get("eps_center", 1e-3))
    eps_rel    = float(params.get("eps_rel",    1e-3))

    # Normalize positions by R so costs are less scale-sensitive
    center_norm = center / R

    # CasADi constants for state and dynamics
    Ad_dm = ca.DM(Ad)
    Bd_dm = ca.DM(Bd)
    x1_dm = ca.DM(x1)
    x2_dm = ca.DM(x2)

    # ----------------- decision variables u1, u2 -----------------
    u1 = ca.SX.sym("u1", D)
    u2 = ca.SX.sym("u2", D)

    # ----------------- one-step dynamics -----------------
    x1_next = Ad_dm @ x1_dm + Bd_dm @ u1  # [2D]
    x2_next = Ad_dm @ x2_dm + Bd_dm @ u2  # [2D]

    # Positions at next step, normalized by R
    p1_next = x1_next[:D] / R
    p2_next = x2_next[:D] / R

    # ----------------- geometry & distances -----------------
    # Attacker distance to center
    d_center_vec = p2_next - center_norm
    d_center_sq  = ca.sumsqr(d_center_vec)     # (‖p2_next - center‖^2) / R^2

    # Defender distance to center (keep-out)
    d_def_center_vec = p1_next - center_norm
    d_def_center_sq  = ca.sumsqr(d_def_center_vec)

    # Relative distance attacker–defender
    rel_vec = p2_next - p1_next
    rel_sq  = ca.sumsqr(rel_vec)

    # ----------------- player 1 cost J1 (defender) -----------------
    # Wants attacker far from center:
    #   J1_far_center = - w_def_far_center * d_center_sq
    # (negative sign: minimizing J1 pushes d_center_sq upwards)
    J1_far_center = -w_def_far_center * d_center_sq

    # Wants to avoid crashing into center:
    #   barrier ~ w_def_keepout_center / (dist(def-center)^2 + eps)
    J1_keepout_center = w_def_keepout_center / (d_def_center_sq + eps_center)

    # Control regularization:
    J1_u = w_def_u * ca.sumsqr(u1)

    J1 = J1_far_center + J1_keepout_center + J1_u

    # ----------------- player 2 cost J2 (attacker) -----------------
    # Wants attacker close to center:
    #   J2_center = w_att_center * d_center_sq
    # (minimizing J2 pushes d_center_sq downward → closer to center)
    J2_center = w_att_center * d_center_sq

    # Wants to avoid colliding with defender:
    #   barrier ~ w_att_avoid_def / (rel_sq + eps_rel)
    J2_avoid_def = w_att_avoid_def / (rel_sq + eps_rel)

    # Control regularization:
    J2_u = w_att_u * ca.sumsqr(u2)

    J2 = J2_center + J2_avoid_def + J2_u

    # ----------------- first-order conditions -----------------
    grad_J1_u1 = ca.gradient(J1, u1)  # ∂J1/∂u1
    grad_J2_u2 = ca.gradient(J2, u2)  # ∂J2/∂u2

    # Stack F(z) = [∂J1/∂u1; ∂J2/∂u2] and minimize 0.5 * ||F||^2
    F = ca.vertcat(grad_J1_u1, grad_J2_u2)
    z = ca.vertcat(u1, u2)  # decision vector of size 2D
    obj = 0.5 * ca.dot(F, F)

    nlp = {"x": z, "f": obj}

    # ----------------- box constraints on controls -----------------
    lbz = np.full(2 * D, -umax, dtype=float)
    ubz = np.full(2 * D,  umax, dtype=float)

    opts = {
        "ipopt.print_level": 0,
        "print_time": False,
        "ipopt.tol": 1e-4,
        "ipopt.max_iter": 50,
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    # Initial guess: zero controls (could warm-start from previous solution)
    z0 = np.zeros(2 * D, dtype=float)

    try:
        sol = solver(x0=z0, lbx=lbz, ubx=ubz)
        z_opt = np.array(sol["x"]).ravel()
        u1_opt = z_opt[:D].astype(np.float32)
        u2_opt = z_opt[D:].astype(np.float32)
    except Exception:
        # IPOPT failed or raised; return trivial fallback
        u1_opt = np.zeros(D, dtype=np.float32)
        u2_opt = np.zeros(D, dtype=np.float32)

    return u1_opt, u2_opt
