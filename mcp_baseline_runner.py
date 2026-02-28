# mcp_baseline_runner.py
"""
MCP baseline runner for your PPO pursuit-evasion game.

Each sim step k:
  - Build a single big Mixed Complementarity Problem (MCP) capturing the KKT
    conditions for each player's bound-constrained optimization.
  - Solve with PATH via Pyomo (pyomo.mpec + SolverFactory("path")).
  - Apply the resulting controls to the plant and roll out.

Supports:
  - 1v1: defender vs attacker
  - 1v2: defender vs attacker TEAM (attacker controls are concatenated as one player)

Cost modes:
  - "security": zero-sum saddle using a scalar security value g
      defender maximizes g  <=> minimizes J1 = -g
      attacker minimizes g  <=> minimizes J2 = +g (+ optional effort)
  - "default": general-sum Nash using per-agent PPO-ish rewards (smooth versions)
      J1 = -r_def_step
      J2 = -r_att_step

Config (optional):
  cfg["mcp"] = {
      "mode": "security" or "default",
      "solver": "path",
      "tee": False,
      "kappa": 40.0,                # softplus sharpness
      "g_clip": None or float,      # optional clipping of g BEFORE mapping to J
      "att_reward_style": "progress_close" or "symmetric",
      "def_use_keepout": True,
      "include_terminal_in_mcp": False,  # keep False (terminal logic is non-smooth)
      "softmin_tau": 30.0,          # only relevant for 1v2 threat smoothmin
      "threat_mode": "idx0" or "softmin",# for 1v2 security threat selection
  }

Notes:
  - Terminal penalties (collision/oob/hit) are NOT embedded in the MCP objective by default,
    because they are discontinuous and will destabilize PATH.
  - Rollout termination is still computed after stepping the plant (as in your env).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

# Pyomo imports guarded so the file can still import in environments without it
try:
    import pyomo.environ as pyo
    import pyomo.mpec as mpec
    from pyomo.core.expr.calculus.derivatives import differentiate
except Exception as e:  # pragma: no cover
    pyo = None
    mpec = None
    differentiate = None
    _PYO_IMPORT_ERR = e
else:
    _PYO_IMPORT_ERR = None


# ============================================================
# MCP params
# ============================================================

@dataclass
class MCPParams:
    mode: str = "security"               # "security" or "default"
    solver: str = "path"                 # "path" (recommended)
    tee: bool = False

    # smoothing
    kappa: float = 40.0                  # softplus sharpness
    softmin_tau: float = 30.0            # 1v2 threat smoothmin sharpness
    threat_mode: str = "idx0"            # "idx0" or "softmin" (1v2 only)

    # optional clipping of g before mapping to J (helps PPO; optional here)
    g_clip: Optional[float] = None

    # which attacker reward shape to mimic in default mode
    att_reward_style: str = "progress_close"  # "progress_close" or "symmetric"

    # keepout term
    def_use_keepout: bool = True

    # include terminal in solver objective (do NOT unless you smooth it hard)
    include_terminal_in_mcp: bool = False


def _mcp_params_from_cfg(cfg: Dict[str, Any]) -> MCPParams:
    p = MCPParams()
    d = cfg.get("mcp", {}) or {}
    for k, v in d.items():
        if hasattr(p, k):
            setattr(p, k, v)
    p.mode = (p.mode or "security").lower()
    p.solver = (p.solver or "path").lower()
    p.att_reward_style = (p.att_reward_style or "progress_close").lower()
    p.threat_mode = (p.threat_mode or "idx0").lower()
    return p


# ============================================================
# Small math helpers (smooth penalties)
# ============================================================

def _softplus_expr(x, kappa: float):
    # smooth max(0,x): (1/k) log(1 + exp(k x))
    # guard: for large kappa, exp can overflow; keep kappa moderate (20..80)
    return (1.0 / float(kappa)) * pyo.log(1.0 + pyo.exp(float(kappa) * x))


def _smooth_hinge_sq(x, kappa: float):
    sp = _softplus_expr(x, kappa)
    return sp * sp


def _softmin_expr(vals: List, tau: float):
    """
    smooth min(vals) ≈ -(1/tau) log(sum(exp(-tau * v_i)))
    Larger tau => sharper min.
    """
    tau = float(tau)
    return -(1.0 / tau) * pyo.log(sum(pyo.exp(-tau * v) for v in vals))


def _rho_expr(p_list: List, center: np.ndarray, R: float):
    # p_list are pyomo expressions for each coord
    return pyo.sqrt(sum((p_list[i] - float(center[i])) ** 2 for i in range(len(p_list))) + 1e-12) / (float(R) + 1e-12)


def _wall_penalty_expr(p_list: List, center: np.ndarray, R: float, soft_wall: float, wallK: float, kappa: float):
    rho = _rho_expr(p_list, center, R)
    return float(wallK) * _smooth_hinge_sq(rho - float(soft_wall), kappa)


def _keepout_penalty_expr(p_def_list: List, center: np.ndarray, oi_r_m: float, buf_m: float, coef: float, kappa: float):
    """
    Smooth keepout: penalize if ||p_def-center|| < oi_r + buf
      penalty = coef * softplus(r_keep - dist)^2
    """
    if (oi_r_m <= 0.0) or (coef <= 0.0):
        return 0.0
    r_keep = float(oi_r_m + buf_m)
    dist = pyo.sqrt(sum((p_def_list[i] - float(center[i])) ** 2 for i in range(len(p_def_list))) + 1e-12)
    return float(coef) * _smooth_hinge_sq(r_keep - dist, kappa)


def _dot_expr(v_list: List):
    return sum(v_list[i] * v_list[i] for i in range(len(v_list)))


# ============================================================
# Dynamics helpers: step matrices for HCW/LTV
# ============================================================

def _extract_step_mats(cfg: Dict[str, Any], k: int, D: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (Ad, Bd) for state dimension 2D and action dimension D.

    Supports:
      - cfg["dyn"]["Ad"], ["Bd"] (LTI)
      - cfg["dyn"]["Ad_seq"], ["Bd_seq"] (LTV) (uses step k)

    Handles the common case where matrices are 6x6,6x3 but D==2
    by selecting planar indices [0,1,3,4] and control cols [0,1].
    """
    dyn = cfg.get("dyn", {}) if isinstance(cfg.get("dyn", {}), dict) else {}
    Ad = dyn.get("Ad", None)
    Bd = dyn.get("Bd", None)
    Ad_seq = dyn.get("Ad_seq", None)
    Bd_seq = dyn.get("Bd_seq", None)

    if Ad_seq is not None and Bd_seq is not None:
        Ad = Ad_seq[k]
        Bd = Bd_seq[k]

    if Ad is None or Bd is None:
        raise ValueError("MCP baseline requires cfg['dyn']['Ad'/'Bd'] or Ad_seq/Bd_seq to be present.")

    Ad = np.asarray(Ad, float)
    Bd = np.asarray(Bd, float)

    # Most common: D==3 => Ad 6x6, Bd 6x3, OK.
    if D == 3:
        if Ad.shape != (6, 6) or Bd.shape != (6, 3):
            # If user built different but still consistent, accept 2D case etc
            if Ad.shape == (2 * D, 2 * D) and Bd.shape == (2 * D, D):
                return Ad, Bd
            raise ValueError(f"Unexpected Ad/Bd shapes for D=3: Ad{Ad.shape}, Bd{Bd.shape}")
        return Ad, Bd

    # D == 2: allow either 4x4/4x2 or 6x6/6x3
    if D == 2:
        if Ad.shape == (4, 4) and Bd.shape == (4, 2):
            return Ad, Bd
        if Ad.shape == (6, 6) and Bd.shape == (6, 3):
            idx = [0, 1, 3, 4]  # x,y,xdot,ydot
            Ad2 = Ad[np.ix_(idx, idx)]
            Bd2 = Bd[np.ix_(idx, [0, 1])]
            return Ad2, Bd2
        raise ValueError(f"Unexpected Ad/Bd shapes for D=2: Ad{Ad.shape}, Bd{Bd.shape}")

    raise ValueError(f"Unsupported D={D}. MCP baseline supports D=2 or D=3.")


def _step_lti(x: np.ndarray, u: np.ndarray, Ad: np.ndarray, Bd: np.ndarray) -> np.ndarray:
    return (Ad @ x + Bd @ u).astype(np.float32)


# ============================================================
# PPO-aligned cost builders (smooth)
# ============================================================

def _build_security_g_1v1(
    *,
    D: int,
    center: np.ndarray,
    R: float,
    x2_pos_prev: np.ndarray,           # numpy (D,)
    p1n: List, p2n: List,              # pyomo expr lists
    u1: List, u2: List,                # pyomo vars lists
    cfg: Dict[str, Any],
    mcpp: MCPParams,
) -> Any:
    # Use same knobs as your env step
    k_pos = float(cfg.get("step_pos_coef", 0.0))
    alpha = float(cfg.get("dense_coef", 0.0))
    lD    = float(cfg.get("effort_def", 0.0))
    lA    = float(cfg.get("effort_att", 0.0))  # optionally in attacker J

    wallK = float(cfg.get("wall_penalty", 0.0))
    soft_wall = float(cfg.get("soft_wall_start", 0.7))

    oi = cfg.get("oi", {}) or {}
    oi_r_m = float(oi.get("r", 0.0))
    def_keepout_buf_m = float(cfg.get("def_keepout_buffer_m", 0.0))
    def_center_avoid_coef = float(cfg.get("def_center_avoid_coef", 0.0))

    # d2 next (normalized squared)
    d2_next = (sum((p2n[i] - float(center[i])) ** 2 for i in range(D))) / (float(R) * float(R) + 1e-12)
    d2_prev = float(np.dot(x2_pos_prev - center, x2_pos_prev - center)) / (float(R) * float(R) + 1e-12)
    delta_d2 = d2_next - d2_prev

    # smooth wall + keepout
    wall1 = _wall_penalty_expr(p1n, center, R, soft_wall, wallK, mcpp.kappa)
    keepout = 0.0
    if bool(mcpp.def_use_keepout):
        keepout = _keepout_penalty_expr(p1n, center, oi_r_m, def_keepout_buf_m, def_center_avoid_coef, mcpp.kappa)

    u1n2 = _dot_expr(u1)
    # security g (defender wants large)
    g = (k_pos * d2_next) + (alpha * delta_d2) - (lD * u1n2) - wall1 - keepout

    # Optional clipping of g for stability (mostly PPO-motivated; available here too)
    if mcpp.g_clip is not None and float(mcpp.g_clip) > 0:
        gc = float(mcpp.g_clip)
        # hard clip is non-smooth; but PATH can still sometimes handle it poorly.
        # We approximate clip with a smooth saturation: gc * tanh(g/gc)
        g = gc * pyo.tanh(g / gc)

    # Attacker effort can be added in J2 (not inside g)
    return g


def _build_default_rewards_1v1(
    *,
    D: int,
    center: np.ndarray,
    R: float,
    x1_prev: np.ndarray,
    x2_prev: np.ndarray,
    p1n: List, p2n: List,
    u1: List, u2: List,
    cfg: Dict[str, Any],
    mcpp: MCPParams,
) -> Tuple[Any, Any]:
    """
    Returns (r_def_step, r_att_step) as smooth expressions.
    """
    # Shared knobs
    alpha = float(cfg.get("dense_coef", 0.0))
    k_pos = float(cfg.get("step_pos_coef", 0.0))
    k_rel = float(cfg.get("step_rel_coef", 0.0))
    lD    = float(cfg.get("effort_def", 0.0))
    lA    = float(cfg.get("effort_att", 0.0))
    wallK = float(cfg.get("wall_penalty", 0.0))
    soft_wall = float(cfg.get("soft_wall_start", 0.7))

    # OI keepout
    oi = cfg.get("oi", {}) or {}
    oi_r_m = float(oi.get("r", 0.0))
    def_keepout_buf_m = float(cfg.get("def_keepout_buffer_m", 0.0))
    def_center_avoid_coef = float(cfg.get("def_center_avoid_coef", 0.0))

    # Distances
    d2_next = (sum((p2n[i] - float(center[i])) ** 2 for i in range(D))) / (float(R) * float(R) + 1e-12)
    d2_prev = float(np.dot(x2_prev[:D] - center, x2_prev[:D] - center)) / (float(R) * float(R) + 1e-12)
    delta_d2 = d2_next - d2_prev

    rel2_next = (sum((p2n[i] - p1n[i]) ** 2 for i in range(D))) / (float(R) * float(R) + 1e-12)

    wall1 = _wall_penalty_expr(p1n, center, R, soft_wall, wallK, mcpp.kappa)
    wall2 = _wall_penalty_expr(p2n, center, R, soft_wall, wallK, mcpp.kappa)

    keepout = 0.0
    if bool(mcpp.def_use_keepout):
        keepout = _keepout_penalty_expr(p1n, center, oi_r_m, def_keepout_buf_m, def_center_avoid_coef, mcpp.kappa)

    u1n2 = _dot_expr(u1)
    u2n2 = _dot_expr(u2)

    # Defender reward (smooth version of your common form)
    r_def = (alpha * delta_d2) + (k_pos * d2_next) - (lD * u1n2) - wall1 - keepout
    if k_rel != 0.0:
        # In your code you sometimes used -k_rel * rel2; choose sign by cfg convention
        r_def = r_def - (k_rel * rel2_next)

    # Attacker reward variants
    if mcpp.att_reward_style == "symmetric":
        # r_att = -alpha*delta_d2 - k_pos*d2 + k_rel*rel2 - lA*||u2||^2 - wall2
        r_att = (-alpha * delta_d2) - (k_pos * d2_next) - (lA * u2n2) - wall2
        if k_rel != 0.0:
            r_att = r_att + (k_rel * rel2_next)

    else:
        # "progress_close": mimic your newer attacker shaping:
        # progress = d2_prev - d2_next (positive if moving inward)
        # close_pen = softplus(1 - dist/min_sep)^2
        att = cfg.get("att_reward", {}) or {}
        att_rule = cfg.get("att_rule", {}) or {}
        k_prog = float(att.get("k_prog", 2.0))
        k_close = float(att.get("k_close", 2.0))
        min_sep = float(att.get("min_sep", att_rule.get("min_sep", 3.0)))

        # progress (inward)
        progress = (d2_prev - d2_next)

        # smooth close penalty
        dist = pyo.sqrt(sum((p2n[i] - p1n[i]) ** 2 for i in range(D)) + 1e-12)
        x = dist / (float(min_sep) + 1e-12)
        close_pen = _smooth_hinge_sq(1.0 - x, mcpp.kappa)

        r_att = (k_prog * progress) - (k_close * close_pen) - (lA * u2n2) - wall2

    return r_def, r_att


# ============================================================
# MCP solver (1v1)
# ============================================================

def solve_one_step_mcp_1v1(
    cfg: Dict[str, Any],
    *,
    x1: np.ndarray,
    x2: np.ndarray,
    k: int,
    u1_prev: Optional[np.ndarray] = None,
    u2_prev: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if pyo is None or mpec is None or differentiate is None:
        raise RuntimeError(
            f"Pyomo MPEC is not available in this environment: {_PYO_IMPORT_ERR}\n"
            f"Install: pip install pyomo  (and ensure PATH solver is installed/licensed)."
        )

    mcpp = _mcp_params_from_cfg(cfg)
    D = int(cfg.get("D", 3))
    umax = float(cfg.get("umax", 5e-4))

    ar = cfg.get("arena", {}) or {}
    R = float(ar.get("r", 30.0))
    center = np.array([ar.get("cx", 0.0), ar.get("cy", 0.0), (ar.get("cz", 0.0) if D == 3 else 0.0)], dtype=float)[:D]

    x1 = np.asarray(x1, float).reshape(2 * D)
    x2 = np.asarray(x2, float).reshape(2 * D)

    Ad, Bd = _extract_step_mats(cfg, k=k, D=D)

    # Warm-start defaults
    if u1_prev is None:
        u1_prev = np.zeros((D,), dtype=float)
    if u2_prev is None:
        u2_prev = np.zeros((D,), dtype=float)
    u1_prev = np.asarray(u1_prev, float).reshape(D)
    u2_prev = np.asarray(u2_prev, float).reshape(D)

    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, D - 1)

    # controls (unbounded; bounds imposed via complementarity)
    m.u1 = pyo.Var(m.I, initialize=lambda mdl, i: float(u1_prev[i]))
    m.u2 = pyo.Var(m.I, initialize=lambda mdl, i: float(u2_prev[i]))

    # duals for bounds (nonneg)
    m.l1 = pyo.Var(m.I, within=pyo.NonNegativeReals, initialize=0.0)
    m.u1m = pyo.Var(m.I, within=pyo.NonNegativeReals, initialize=0.0)
    m.l2 = pyo.Var(m.I, within=pyo.NonNegativeReals, initialize=0.0)
    m.u2m = pyo.Var(m.I, within=pyo.NonNegativeReals, initialize=0.0)

    # Next state expressions (one-step)
    def x1n(j: int):
        return sum(float(Ad[j, t]) * float(x1[t]) for t in range(2 * D)) + sum(float(Bd[j, i]) * m.u1[i] for i in range(D))

    def x2n(j: int):
        return sum(float(Ad[j, t]) * float(x2[t]) for t in range(2 * D)) + sum(float(Bd[j, i]) * m.u2[i] for i in range(D))

    p1n = [x1n(i) for i in range(D)]
    p2n = [x2n(i) for i in range(D)]

    # Build objectives
    if mcpp.mode == "security":
        g = _build_security_g_1v1(
            D=D, center=center, R=R,
            x2_pos_prev=x2[:D].copy(),
            p1n=p1n, p2n=p2n,
            u1=[m.u1[i] for i in range(D)],
            u2=[m.u2[i] for i in range(D)],
            cfg=cfg, mcpp=mcpp,
        )
        # defender minimizes -g ; attacker minimizes +g (+ optional effort)
        lA = float(cfg.get("effort_att", 0.0))
        u2n2 = _dot_expr([m.u2[i] for i in range(D)])
        J1 = -g
        J2 = +g + (lA * u2n2)

    else:
        r_def, r_att = _build_default_rewards_1v1(
            D=D, center=center, R=R,
            x1_prev=x1, x2_prev=x2,
            p1n=p1n, p2n=p2n,
            u1=[m.u1[i] for i in range(D)],
            u2=[m.u2[i] for i in range(D)],
            cfg=cfg, mcpp=mcpp
        )
        J1 = -r_def
        J2 = -r_att
        g = None  # not used

    m.J1 = pyo.Expression(expr=J1)
    m.J2 = pyo.Expression(expr=J2)

    # Stationarity (bound-only KKT)
    def stat1_rule(mdl, i):
        return differentiate(mdl.J1, mdl.u1[i]) + mdl.u1m[i] - mdl.l1[i] == 0

    def stat2_rule(mdl, i):
        return differentiate(mdl.J2, mdl.u2[i]) + mdl.u2m[i] - mdl.l2[i] == 0

    m.stat1 = pyo.Constraint(m.I, rule=stat1_rule)
    m.stat2 = pyo.Constraint(m.I, rule=stat2_rule)

    # Complementarity for bounds
    lo = -umax
    hi = +umax

    m.c1L = mpec.Complementarity(
        m.I, rule=lambda mdl, i: mpec.complements(mdl.u1[i] - lo >= 0, mdl.l1[i] >= 0)
    )
    m.c1U = mpec.Complementarity(
        m.I, rule=lambda mdl, i: mpec.complements(hi - mdl.u1[i] >= 0, mdl.u1m[i] >= 0)
    )
    m.c2L = mpec.Complementarity(
        m.I, rule=lambda mdl, i: mpec.complements(mdl.u2[i] - lo >= 0, mdl.l2[i] >= 0)
    )
    m.c2U = mpec.Complementarity(
        m.I, rule=lambda mdl, i: mpec.complements(hi - mdl.u2[i] >= 0, mdl.u2m[i] >= 0)
    )

    # Transform to standard form for PATH
    try:
        pyo.TransformationFactory("mpec.standard_form").apply_to(m)
    except Exception as e:
        raise RuntimeError(
            "Failed to apply Pyomo MPEC transformation mpec.standard_form. "
            "Check your Pyomo install/version."
        ) from e

    # Solve
    solver = pyo.SolverFactory(mcpp.solver)
    if solver is None or not solver.available(exception_flag=False):
        raise RuntimeError(
            f"Solver '{mcpp.solver}' is not available to Pyomo. "
            "For PATH you typically need a PATH installation and license."
        )

    res = solver.solve(m, tee=bool(mcpp.tee))

    u1_star = np.array([pyo.value(m.u1[i]) for i in range(D)], dtype=np.float32)
    u2_star = np.array([pyo.value(m.u2[i]) for i in range(D)], dtype=np.float32)

    dbg = {
        "mode": mcpp.mode,
        "solver": mcpp.solver,
        "termination": str(getattr(res.solver, "termination_condition", "")),
        "J1": float(pyo.value(m.J1)),
        "J2": float(pyo.value(m.J2)),
    }
    if g is not None:
        dbg["g"] = float(pyo.value(g))

    return u1_star, u2_star, dbg


# ============================================================
# MCP solver (1v2 defender vs attacker TEAM)
# ============================================================

def solve_one_step_mcp_1v2_team(
    cfg: Dict[str, Any],
    *,
    x1: np.ndarray,
    x2a: np.ndarray,
    x2b: np.ndarray,
    k: int,
    u1_prev: Optional[np.ndarray] = None,
    u2_prev: Optional[np.ndarray] = None,   # shape (2, D) or flat (2D,)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Defender vs attacker TEAM:
      attacker decision variable is U2 = [u2a; u2b] in R^(2D)
      KKT stationarity is w.r.t. those 2D vars.

    Security mode only (recommended). Default general-sum for 3 players is possible
    but not included here (it’s a larger MCP with 3 KKT blocks).
    """
    if pyo is None or mpec is None or differentiate is None:
        raise RuntimeError(
            f"Pyomo MPEC is not available in this environment: {_PYO_IMPORT_ERR}\n"
            f"Install: pip install pyomo  (and ensure PATH solver is installed/licensed)."
        )

    mcpp = _mcp_params_from_cfg(cfg)
    D = int(cfg.get("D", 3))
    umax = float(cfg.get("umax", 5e-4))

    ar = cfg.get("arena", {}) or {}
    R = float(ar.get("r", 30.0))
    center = np.array([ar.get("cx", 0.0), ar.get("cy", 0.0), (ar.get("cz", 0.0) if D == 3 else 0.0)], dtype=float)[:D]

    x1 = np.asarray(x1, float).reshape(2 * D)
    x2a = np.asarray(x2a, float).reshape(2 * D)
    x2b = np.asarray(x2b, float).reshape(2 * D)

    Ad, Bd = _extract_step_mats(cfg, k=k, D=D)

    # Warm-start
    if u1_prev is None:
        u1_prev = np.zeros((D,), dtype=float)
    u1_prev = np.asarray(u1_prev, float).reshape(D)

    if u2_prev is None:
        u2_prev = np.zeros((2, D), dtype=float)
    u2_prev = np.asarray(u2_prev, float)
    if u2_prev.ndim == 1:
        u2_prev = u2_prev.reshape(2, D)
    else:
        u2_prev = u2_prev.reshape(2, D)

    # Indices for team control vector
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, D - 1)
    m.J = pyo.RangeSet(0, 2 * D - 1)

    m.u1 = pyo.Var(m.I, initialize=lambda mdl, i: float(u1_prev[i]))
    m.u2 = pyo.Var(m.J, initialize=lambda mdl, j: float(u2_prev[j // D, j % D]))

    m.l1 = pyo.Var(m.I, within=pyo.NonNegativeReals, initialize=0.0)
    m.u1m = pyo.Var(m.I, within=pyo.NonNegativeReals, initialize=0.0)

    m.l2 = pyo.Var(m.J, within=pyo.NonNegativeReals, initialize=0.0)
    m.u2m = pyo.Var(m.J, within=pyo.NonNegativeReals, initialize=0.0)

    # helper to slice attacker controls
    def u2a_i(i):  # i in [0..D-1]
        return m.u2[i]

    def u2b_i(i):
        return m.u2[D + i]

    def x1n(j: int):
        return sum(float(Ad[j, t]) * float(x1[t]) for t in range(2 * D)) + sum(float(Bd[j, i]) * m.u1[i] for i in range(D))

    def x2an(j: int):
        return sum(float(Ad[j, t]) * float(x2a[t]) for t in range(2 * D)) + sum(float(Bd[j, i]) * u2a_i(i) for i in range(D))

    def x2bn(j: int):
        return sum(float(Ad[j, t]) * float(x2b[t]) for t in range(2 * D)) + sum(float(Bd[j, i]) * u2b_i(i) for i in range(D))

    p1n = [x1n(i) for i in range(D)]
    p2an = [x2an(i) for i in range(D)]
    p2bn = [x2bn(i) for i in range(D)]

    if mcpp.mode != "security":
        raise ValueError("solve_one_step_mcp_1v2_team currently supports mode='security' only.")

    # Threat selection for g: idx0 (use attacker0) or smoothmin over attackers
    d2a_next = (sum((p2an[i] - float(center[i])) ** 2 for i in range(D))) / (float(R) * float(R) + 1e-12)
    d2b_next = (sum((p2bn[i] - float(center[i])) ** 2 for i in range(D))) / (float(R) * float(R) + 1e-12)

    if mcpp.threat_mode == "softmin":
        d2_threat_next = _softmin_expr([d2a_next, d2b_next], tau=mcpp.softmin_tau)
        # softmin returns something like min, good.
    else:
        d2_threat_next = d2a_next  # idx0

    # d2_prev threat uses current true positions
    d2a_prev = float(np.dot(x2a[:D] - center, x2a[:D] - center)) / (float(R) * float(R) + 1e-12)
    d2b_prev = float(np.dot(x2b[:D] - center, x2b[:D] - center)) / (float(R) * float(R) + 1e-12)
    if mcpp.threat_mode == "softmin":
        # smoothmin over constants is just numeric
        # use the same formula in numpy:
        tau = float(mcpp.softmin_tau)
        d2_threat_prev = -(1.0 / tau) * np.log(np.exp(-tau * d2a_prev) + np.exp(-tau * d2b_prev))
    else:
        d2_threat_prev = d2a_prev

    alpha = float(cfg.get("dense_coef", 0.0))
    k_pos = float(cfg.get("step_pos_coef", 0.0))
    lD    = float(cfg.get("effort_def", 0.0))
    lA    = float(cfg.get("effort_att", 0.0))
    wallK = float(cfg.get("wall_penalty", 0.0))
    soft_wall = float(cfg.get("soft_wall_start", 0.7))

    oi = cfg.get("oi", {}) or {}
    oi_r_m = float(oi.get("r", 0.0))
    def_keepout_buf_m = float(cfg.get("def_keepout_buffer_m", 0.0))
    def_center_avoid_coef = float(cfg.get("def_center_avoid_coef", 0.0))

    delta = d2_threat_next - float(d2_threat_prev)

    wall1 = _wall_penalty_expr(p1n, center, R, soft_wall, wallK, mcpp.kappa)
    keepout = 0.0
    if bool(mcpp.def_use_keepout):
        keepout = _keepout_penalty_expr(p1n, center, oi_r_m, def_keepout_buf_m, def_center_avoid_coef, mcpp.kappa)

    u1n2 = _dot_expr([m.u1[i] for i in range(D)])
    u2n2 = _dot_expr([m.u2[j] for j in range(2 * D)])

    g = (k_pos * d2_threat_next) + (alpha * delta) - (lD * u1n2) - wall1 - keepout
    if mcpp.g_clip is not None and float(mcpp.g_clip) > 0:
        gc = float(mcpp.g_clip)
        g = gc * pyo.tanh(g / gc)

    J1 = -g
    J2 = +g + (lA * u2n2)

    m.J1 = pyo.Expression(expr=J1)
    m.J2 = pyo.Expression(expr=J2)

    # Stationarity
    def stat1_rule(mdl, i):
        return differentiate(mdl.J1, mdl.u1[i]) + mdl.u1m[i] - mdl.l1[i] == 0

    def stat2_rule(mdl, j):
        return differentiate(mdl.J2, mdl.u2[j]) + mdl.u2m[j] - mdl.l2[j] == 0

    m.stat1 = pyo.Constraint(m.I, rule=stat1_rule)
    m.stat2 = pyo.Constraint(m.J, rule=stat2_rule)

    lo = -umax
    hi = +umax

    m.c1L = mpec.Complementarity(m.I, rule=lambda mdl, i: mpec.complements(mdl.u1[i] - lo >= 0, mdl.l1[i] >= 0))
    m.c1U = mpec.Complementarity(m.I, rule=lambda mdl, i: mpec.complements(hi - mdl.u1[i] >= 0, mdl.u1m[i] >= 0))

    m.c2L = mpec.Complementarity(m.J, rule=lambda mdl, j: mpec.complements(mdl.u2[j] - lo >= 0, mdl.l2[j] >= 0))
    m.c2U = mpec.Complementarity(m.J, rule=lambda mdl, j: mpec.complements(hi - mdl.u2[j] >= 0, mdl.u2m[j] >= 0))

    try:
        pyo.TransformationFactory("mpec.standard_form").apply_to(m)
    except Exception as e:
        raise RuntimeError("Failed to apply mpec.standard_form for 1v2 team MCP.") from e

    solver = pyo.SolverFactory(mcpp.solver)
    if solver is None or not solver.available(exception_flag=False):
        raise RuntimeError(f"Solver '{mcpp.solver}' not available to Pyomo (need PATH).")

    res = solver.solve(m, tee=bool(mcpp.tee))

    u1_star = np.array([pyo.value(m.u1[i]) for i in range(D)], dtype=np.float32)
    u2_star = np.array([pyo.value(m.u2[j]) for j in range(2 * D)], dtype=np.float32).reshape(2, D)
    u2a_star = u2_star[0].copy()
    u2b_star = u2_star[1].copy()

    dbg = {
        "mode": mcpp.mode,
        "solver": mcpp.solver,
        "termination": str(getattr(res.solver, "termination_condition", "")),
        "J1": float(pyo.value(m.J1)),
        "J2": float(pyo.value(m.J2)),
        "g": float(pyo.value(g)),
        "threat_mode": mcpp.threat_mode,
    }
    return u1_star, u2a_star, u2b_star, dbg


# ============================================================
# Rollout helpers (frames_dict styling like your other runners)
# ============================================================

def _p3(xD: np.ndarray, D: int):
    xD = np.asarray(xD, float).reshape(-1)
    if D == 3:
        return (float(xD[0]), float(xD[1]), float(xD[2]))
    return (float(xD[0]), float(xD[1]), 0.0)


def _identity_R():
    return np.eye(3, dtype=float)


def _pad3(u: np.ndarray, D: int):
    u = np.asarray(u, float).reshape(-1)
    return np.array([u[0], u[1], u[2] if D == 3 else 0.0], dtype=float)


# ============================================================
# Rollout runner: 1v1
# ============================================================

def run_rhc_with_mcp_game_1v1_collect_frames_3d(
    cfg: Dict[str, Any],
    steps: int | None = None,
    turn_len: int | None = None,
):
    """
    1 defender vs 1 attacker MCP baseline rollout.
    Returns dict compatible with your animator keys.
    """
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx = 2 * D
    T_horizon = int(cfg.get("T", 1))
    umax = float(cfg.get("umax", 5e-4))

    if steps is None:
        steps = int(cfg.get("T_eval", cfg.get("steps", 60)))
    if turn_len is None:
        turn_len = 1  # unused; kept for API symmetry

    # arena
    ar = cfg.get("arena", {}) or {}
    ar.setdefault("type", "sphere")
    ar.setdefault("cx", 0.0); ar.setdefault("cy", 0.0); ar.setdefault("cz", 0.0)
    ar.setdefault("r", 30.0)
    center = np.array([ar["cx"], ar["cy"], (ar.get("cz", 0.0) if D == 3 else 0.0)], dtype=np.float32)[:D]
    R = float(ar["r"])

    # init states
    x0 = np.asarray(cfg["x0"], dtype=np.float32)
    x1 = x0[0, :nx].copy()
    x2 = x0[1, :nx].copy()

    # logs
    plan_hist1, plan_hist2 = [], []
    plan_att1, plan_att2 = [], []
    exec_xyz1, exec_xyz2 = [], []
    exec_att1, exec_att2 = [], []
    phi_hist1, phi_hist2 = [], []
    fov_axis_hist, fov_seen_mask = [], []

    u_cmd_1, u_cmd_2 = [], []
    u_cmd_norm_1, u_cmd_norm_2 = [], []

    dbg_hist = []

    # t=0
    exec_xyz1.append(_p3(x1[:D], D))
    exec_xyz2.append(_p3(x2[:D], D))
    exec_att1.append({"R": _identity_R(), "phi": 0.0}); phi_hist1.append(0.0)
    exec_att2.append({"R": _identity_R(), "phi": 0.0}); phi_hist2.append(0.0)
    fov_axis_hist.append(None); fov_seen_mask.append(False)

    # warm start
    u1_prev = np.zeros((D,), dtype=np.float32)
    u2_prev = np.zeros((D,), dtype=np.float32)

    for k in range(steps):
        # Solve MCP for this step
        u1, u2, dbg = solve_one_step_mcp_1v1(
            cfg, x1=x1, x2=x2, k=k,
            u1_prev=u1_prev, u2_prev=u2_prev
        )
        u1 = np.clip(u1, -umax, +umax).astype(np.float32)
        u2 = np.clip(u2, -umax, +umax).astype(np.float32)
        dbg_hist.append(dbg)

        # log thrust
        u1_3 = _pad3(u1, D); u2_3 = _pad3(u2, D)
        u_cmd_1.append(u1_3.copy()); u_cmd_norm_1.append(float(np.linalg.norm(u1_3)))
        u_cmd_2.append(u2_3.copy()); u_cmd_norm_2.append(float(np.linalg.norm(u2_3)))

        # plant step (use matrices for this k)
        Ad, Bd = _extract_step_mats(cfg, k=k, D=D)
        x1 = _step_lti(x1, u1, Ad, Bd)
        x2 = _step_lti(x2, u2, Ad, Bd)

        # update warm start
        u1_prev = u1.copy()
        u2_prev = u2.copy()

        # flat plans
        plan1 = [_p3(x1[:D], D)] * T_horizon
        plan2 = [_p3(x2[:D], D)] * T_horizon
        plan_hist1.append(plan1); plan_hist2.append(plan2)

        I = _identity_R()
        att_stub = [{"R": I, "phi": 0.0} for _ in range(T_horizon)]
        plan_att1.append(att_stub); plan_att2.append(att_stub)

        exec_att1.append({"R": I, "phi": 0.0}); phi_hist1.append(0.0)
        exec_att2.append({"R": I, "phi": 0.0}); phi_hist2.append(0.0)

        exec_xyz1.append(_p3(x1[:D], D))
        exec_xyz2.append(_p3(x2[:D], D))
        fov_axis_hist.append(None); fov_seen_mask.append(False)

    out = {
        "plan_hist1": plan_hist1, "plan_hist2": plan_hist2,
        "plan_att1": plan_att1, "plan_att2": plan_att2,
        "exec1_xyz": exec_xyz1, "exec2_xyz": exec_xyz2,
        "exec_att1": exec_att1, "exec_att2": exec_att2,
        "phi_hist1": phi_hist1, "phi_hist2": phi_hist2,
        "fov_axis_hist": fov_axis_hist, "fov_seen_mask": fov_seen_mask,

        "u_cmd_all": [np.asarray(u_cmd_1, float), np.asarray(u_cmd_2, float)],
        "u_cmd_norm_all": [np.asarray(u_cmd_norm_1, float), np.asarray(u_cmd_norm_2, float)],

        "mcp_dbg_hist": dbg_hist,
        "mcp_params": _mcp_params_from_cfg(cfg).__dict__,
    }
    return out


# ============================================================
# Rollout runner: 1v2 TEAM security baseline
# ============================================================

def run_rhc_with_mcp_game_1v2_team_collect_frames_3d(
    cfg: Dict[str, Any],
    steps: int | None = None,
    turn_len: int | None = None,
):
    """
    1 defender vs 2 attackers, treated as a single attacker TEAM (2D decision variables).
    Solves a 2-player MCP (defender vs team).
    """
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx = 2 * D
    T_horizon = int(cfg.get("T", 1))
    umax = float(cfg.get("umax", 5e-4))

    if steps is None:
        steps = int(cfg.get("T_eval", cfg.get("steps", 60)))
    if turn_len is None:
        turn_len = 1

    # init
    x0 = np.asarray(cfg["x0"], dtype=np.float32)
    x1 = x0[0, :nx].copy()
    x2a = x0[1, :nx].copy()
    x2b = (x0[2, :nx].copy() if x0.shape[0] >= 3 else x0[1, :nx].copy())

    # logs
    plan_hist1, plan_hist2, plan_hist3 = [], [], []
    plan_att1, plan_att2, plan_att3 = [], [], []
    exec_xyz1, exec_xyz2, exec_xyz3 = [], [], []
    exec_att1, exec_att2, exec_att3 = [], [], []
    phi_hist1, phi_hist2, phi_hist3 = [], [], []
    fov_axis_hist, fov_seen_mask = [], []

    u_cmd_1, u_cmd_2, u_cmd_3 = [], [], []
    u_cmd_norm_1, u_cmd_norm_2, u_cmd_norm_3 = [], [], []

    dbg_hist = []

    # t=0
    exec_xyz1.append(_p3(x1[:D], D))
    exec_xyz2.append(_p3(x2a[:D], D))
    exec_xyz3.append(_p3(x2b[:D], D))
    I = _identity_R()
    exec_att1.append({"R": I, "phi": 0.0}); phi_hist1.append(0.0)
    exec_att2.append({"R": I, "phi": 0.0}); phi_hist2.append(0.0)
    exec_att3.append({"R": I, "phi": 0.0}); phi_hist3.append(0.0)
    fov_axis_hist.append(None); fov_seen_mask.append(False)

    # warm start
    u1_prev = np.zeros((D,), dtype=np.float32)
    u2_prev = np.zeros((2, D), dtype=np.float32)

    for k in range(steps):
        u1, u2a, u2b, dbg = solve_one_step_mcp_1v2_team(
            cfg, x1=x1, x2a=x2a, x2b=x2b, k=k,
            u1_prev=u1_prev, u2_prev=u2_prev
        )
        u1 = np.clip(u1, -umax, +umax).astype(np.float32)
        u2a = np.clip(u2a, -umax, +umax).astype(np.float32)
        u2b = np.clip(u2b, -umax, +umax).astype(np.float32)
        dbg_hist.append(dbg)

        # log thrust
        u1_3 = _pad3(u1, D); u2a_3 = _pad3(u2a, D); u2b_3 = _pad3(u2b, D)
        u_cmd_1.append(u1_3.copy()); u_cmd_norm_1.append(float(np.linalg.norm(u1_3)))
        u_cmd_2.append(u2a_3.copy()); u_cmd_norm_2.append(float(np.linalg.norm(u2a_3)))
        u_cmd_3.append(u2b_3.copy()); u_cmd_norm_3.append(float(np.linalg.norm(u2b_3)))

        # plant step
        Ad, Bd = _extract_step_mats(cfg, k=k, D=D)
        x1 = _step_lti(x1, u1, Ad, Bd)
        x2a = _step_lti(x2a, u2a, Ad, Bd)
        x2b = _step_lti(x2b, u2b, Ad, Bd)

        # update warm start
        u1_prev = u1.copy()
        u2_prev = np.stack([u2a.copy(), u2b.copy()], axis=0)

        # flat plans
        plan1 = [_p3(x1[:D], D)] * T_horizon
        plan2 = [_p3(x2a[:D], D)] * T_horizon
        plan3 = [_p3(x2b[:D], D)] * T_horizon
        plan_hist1.append(plan1); plan_hist2.append(plan2); plan_hist3.append(plan3)

        att_stub = [{"R": I, "phi": 0.0} for _ in range(T_horizon)]
        plan_att1.append(att_stub); plan_att2.append(att_stub); plan_att3.append(att_stub)

        exec_att1.append({"R": I, "phi": 0.0}); phi_hist1.append(0.0)
        exec_att2.append({"R": I, "phi": 0.0}); phi_hist2.append(0.0)
        exec_att3.append({"R": I, "phi": 0.0}); phi_hist3.append(0.0)

        exec_xyz1.append(_p3(x1[:D], D))
        exec_xyz2.append(_p3(x2a[:D], D))
        exec_xyz3.append(_p3(x2b[:D], D))
        fov_axis_hist.append(None); fov_seen_mask.append(False)

    out = {
        "plan_hist1": plan_hist1, "plan_hist2": plan_hist2,
        "plan_att1": plan_att1, "plan_att2": plan_att2,
        "exec1_xyz": exec_xyz1, "exec2_xyz": exec_xyz2,
        "exec_att1": exec_att1, "exec_att2": exec_att2,
        "phi_hist1": phi_hist1, "phi_hist2": phi_hist2,

        "plan_hist3": plan_hist3,
        "plan_att3": plan_att3,
        "exec3_xyz": exec_xyz3,
        "exec_att3": exec_att3,
        "phi_hist3": phi_hist3,

        "fov_axis_hist": fov_axis_hist, "fov_seen_mask": fov_seen_mask,

        "u_cmd_all": [
            np.asarray(u_cmd_1, float),
            np.asarray(u_cmd_2, float),
            np.asarray(u_cmd_3, float),
        ],
        "u_cmd_norm_all": [
            np.asarray(u_cmd_norm_1, float),
            np.asarray(u_cmd_norm_2, float),
            np.asarray(u_cmd_norm_3, float),
        ],

        "mcp_dbg_hist": dbg_hist,
        "mcp_params": _mcp_params_from_cfg(cfg).__dict__,
    }
    return out


__all__ = [
    "solve_one_step_mcp_1v1",
    "solve_one_step_mcp_1v2_team",
    "run_rhc_with_mcp_game_1v1_collect_frames_3d",
    "run_rhc_with_mcp_game_1v2_team_collect_frames_3d",
]