# mcp_baseline_runner.py
"""
MCP baseline runner for the current PPO zero-sum pursuit-evasion game.

Each sim step k:
  - Build a single big Mixed Complementarity Problem (MCP) capturing the KKT
    conditions for each player's bound-constrained optimization.
  - Solve with PATH via Pyomo (pyomo.mpec + SolverFactory("path")).
  - Apply the resulting controls to the plant and roll out.

Supports:
  - 1v1: defender vs attacker
  - 1v2: defender vs attacker TEAM (attacker controls are concatenated as one player)

Objective mode:
  - "zero_sum" / "security": solver-safe relaxation of the env's current one-step
    zero-sum reward
      g = k_pos * d2 - k_dock * dock_gap - lD * ||uD||^2 + lA * ||uA||^2
    where dock_gap is smoothed for PATH.

Config (optional):
  cfg["mcp"] = {
      "mode": "zero_sum" or "security",
      "solver": "path",
      "tee": False,
      "kappa": 40.0,                # softplus sharpness
      "g_clip": None or float,      # optional clipping of g BEFORE mapping to J
      "reg_u": 0.01,                # small quadratic regularizer for PATH
      "def_use_keepout": True,
      "include_terminal_in_mcp": False,  # keep False (terminal logic is non-smooth)
      "include_step_walls": False,  # optional smooth stabilizers, not in env g
      "include_keepout": False,     # optional smooth stabilizers, not in env g
      "softmin_tau": 30.0,          # only relevant for 1v2 threat smoothmin
      "threat_mode": "idx0" or "softmin",# for 1v2 threat selection
  }

Notes:
  - Terminal penalties (collision/oob/hit) are NOT embedded in the MCP objective by default,
    because they are discontinuous and destabilize PATH.
  - Rollout termination is computed exactly after each applied step using the same event order
    as core/env.py.
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
    mode: str = "zero_sum"               # "zero_sum" or "security"
    solver: str = "path"                 # "path" (recommended)
    tee: bool = False

    # smoothing
    kappa: float = 40.0                  # softplus sharpness
    softmin_tau: float = 30.0            # 1v2 threat smoothmin sharpness
    threat_mode: str = "idx0"            # "idx0" or "softmin" (1v2 only)

    # optional clipping of g before mapping to J (helps PPO; optional here)
    g_clip: Optional[float] = None

    # small stabilizing regularizer
    reg_u: float = 0.01

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
    p.mode = (p.mode or "zero_sum").lower()
    if p.mode == "default":
        p.mode = "zero_sum"
    p.solver = (p.solver or "path").lower()
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


def _ensure_step_mats(cfg: Dict[str, Any]) -> None:
    dyn = cfg.get("dyn", {}) if isinstance(cfg.get("dyn", {}), dict) else {}
    if dyn.get("Ad", None) is None or dyn.get("Bd", None) is None:
        from config_rl import build_dyn

        build_dyn(cfg)


def _terminal_info_numpy(
    *,
    p_def: np.ndarray,
    p_att_list: List[np.ndarray],
    center: np.ndarray,
    cfg: Dict[str, Any],
    primary_idx: int = 0,
) -> Dict[str, Any]:
    R = float(cfg.get("arena", {}).get("r", 30.0))
    eps = 1e-12
    margin = float(cfg.get("arena_terminate_margin", 1.0))
    oi_r = float((cfg.get("oi", {}) or {}).get("r", 0.0))
    oi_r_norm = oi_r / (R + eps) if R > 0.0 else 0.0
    hit_buffer_def = float(cfg.get("hit_buffer_def", 0.0))
    hit_buffer_att = float(cfg.get("hit_buffer_att", 0.0))
    collision_radius_m = float(cfg.get("collision_radius_m", 0.0))

    rho_def = float(np.linalg.norm(p_def - center) / (R + eps))
    rho_att = float(np.linalg.norm(p_att_list[primary_idx] - center) / (R + eps))

    thresh_def = (1.0 + hit_buffer_def) * oi_r_norm
    thresh_att = (1.0 + hit_buffer_att) * oi_r_norm

    att_hit_target = (oi_r_norm > 0.0) and (rho_att <= thresh_att)
    def_hit_target = (oi_r_norm > 0.0) and (rho_def <= thresh_def)
    hit_target = bool(att_hit_target or def_hit_target)

    collision = False
    if collision_radius_m > 0.0:
        for p_att in p_att_list:
            if float(np.linalg.norm(p_att - p_def)) <= collision_radius_m:
                collision = True
                break

    oob_def = bool(rho_def >= margin)
    oob_att = any(float(np.linalg.norm(p_att - center) / (R + eps)) >= margin for p_att in p_att_list)
    done = bool(collision or hit_target or oob_def or oob_att)

    g_term = 0.0
    if done:
        if collision:
            g_term += float(cfg.get("collision_penalty", 0.0))
        elif att_hit_target or def_hit_target:
            g_term -= float(cfg.get("target_hit_reward_penalty", 0.0))
        elif oob_def:
            g_term -= float(cfg.get("wall_penalty", 0.0))
        elif oob_att:
            g_term += float(cfg.get("wall_penalty", 0.0))

    return {
        "done": bool(done),
        "collision": bool(collision),
        "att_hit_target": bool(att_hit_target),
        "def_hit_target": bool(def_hit_target),
        "oob_def": bool(oob_def),
        "oob_att": bool(oob_att),
        "terminal_g": float(g_term),
        "primary_idx": int(primary_idx),
    }


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
    _ensure_step_mats(cfg)
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
# PPO-aligned cost builders (solver-safe smoothing around the env zero-sum g)
# ============================================================

def _build_zero_sum_g_1v1(
    *,
    D: int,
    center: np.ndarray,
    R: float,
    p1n: List, p2n: List,
    u1: List,
    u2: List,
    cfg: Dict[str, Any],
    mcpp: MCPParams,
) -> Any:
    mcp_cfg = cfg.get("mcp", {}) or {}
    k_pos = float(cfg.get("k_pos", cfg.get("step_pos_coef", 0.0)))
    k_dock = float(cfg.get("k_dock", 0.0))
    lD = float(cfg.get("effort_def", 0.0))
    lA = float(cfg.get("effort_att", 0.0))
    collision_radius_m = float(cfg.get("collision_radius_m", 0.0))
    umax = float(cfg.get("umax", 1.0))

    d2_next = sum((p2n[i] - float(center[i]))**2 for i in range(D)) / (float(R)*float(R) + 1e-12)
    rel_dist = pyo.sqrt(sum((p2n[i] - p1n[i]) ** 2 for i in range(D)) + 1e-12)
    dock_gap = _softplus_expr(rel_dist - collision_radius_m, mcpp.kappa) / (float(R) + 1e-12)

    umax2 = float(umax) * float(umax) + 1e-12
    u1n2 = _dot_expr(u1) / umax2
    u2n2 = _dot_expr(u2) / umax2

    g = (
        k_pos * d2_next
        - k_dock * dock_gap
        - lD * u1n2
        + lA * u2n2
    )

    # Optional smooth stabilizers. Off by default to mirror env g.
    if bool(mcp_cfg.get("include_step_walls", False)):
        wallK = float(cfg.get("wall_penalty", 0.0))
        soft_wall = float(cfg.get("soft_wall_start", 0.5))
        wall1 = _wall_penalty_expr(p1n, center, R, soft_wall, wallK, mcpp.kappa)
        g = g - wall1

    if bool(mcp_cfg.get("include_keepout", False)) and bool(mcpp.def_use_keepout):
        oi = cfg.get("oi", {}) or {}
        oi_r_m = float(oi.get("r", 0.0))
        buf_m = float(cfg.get("def_keepout_buffer_m", 0.0))
        coef = float(cfg.get("def_center_avoid_coef", 0.0))
        keepout = _keepout_penalty_expr(p1n, center, oi_r_m, buf_m, coef, mcpp.kappa)
        g = g - keepout

    # Optional smooth clipping (keeps PATH numerics nicer)
    if mcpp.g_clip is not None and float(mcpp.g_clip) > 0:
        gc = float(mcpp.g_clip)
        g = gc * pyo.tanh(g / gc)

    return g


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
    if mcpp.mode not in ("zero_sum", "security"):
        raise ValueError(f"Unsupported MCP mode={mcpp.mode!r}; use 'zero_sum' or 'security'.")
    if mcpp.include_terminal_in_mcp:
        raise NotImplementedError(
            "include_terminal_in_mcp is not implemented for the PATH-based zero-sum MCP. "
            "Terminal events are applied exactly after each rollout step instead."
        )

    g = _build_zero_sum_g_1v1(
        D=D, center=center, R=R,
        p1n=p1n, p2n=p2n,
        u1=[m.u1[i] for i in range(D)],
        u2=[m.u2[i] for i in range(D)],
        cfg=cfg, mcpp=mcpp,
    )

    u1n2 = _dot_expr([m.u1[i] for i in range(D)])   # = ||u1||^2
    u2n2 = _dot_expr([m.u2[i] for i in range(D)])   # = ||u2||^2

    umax2 = float(umax) * float(umax) + 1e-12
    reg = float(cfg.get("mcp", {}).get("reg_u", mcpp.reg_u))

    J1 = -g + reg * (u1n2 / umax2)
    J2 = +g + reg * (u2n2 / umax2)

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

    if mcpp.mode not in ("zero_sum", "security"):
        raise ValueError(
            f"Unsupported MCP mode={mcpp.mode!r}; use 'zero_sum' or 'security'."
        )
    if mcpp.include_terminal_in_mcp:
        raise NotImplementedError(
            "include_terminal_in_mcp is not implemented for the PATH-based zero-sum MCP. "
            "Terminal events are applied exactly after each rollout step instead."
        )

    # Threat selection for g: idx0 (primary attacker) or smoothmin over attackers.
    d2a_next = (sum((p2an[i] - float(center[i])) ** 2 for i in range(D))) / (float(R) * float(R) + 1e-12)
    d2b_next = (sum((p2bn[i] - float(center[i])) ** 2 for i in range(D))) / (float(R) * float(R) + 1e-12)
    rela_next = pyo.sqrt(sum((p2an[i] - p1n[i]) ** 2 for i in range(D)) + 1e-12)
    relb_next = pyo.sqrt(sum((p2bn[i] - p1n[i]) ** 2 for i in range(D)) + 1e-12)

    collision_radius_m = float(cfg.get("collision_radius_m", 0.0))
    dock_gap_a = _softplus_expr(rela_next - collision_radius_m, mcpp.kappa) / (float(R) + 1e-12)
    dock_gap_b = _softplus_expr(relb_next - collision_radius_m, mcpp.kappa) / (float(R) + 1e-12)

    if mcpp.threat_mode == "softmin":
        d2_threat_next = _softmin_expr([d2a_next, d2b_next], tau=mcpp.softmin_tau)
        dock_gap = _softmin_expr([dock_gap_a, dock_gap_b], tau=mcpp.softmin_tau)
    else:
        d2_threat_next = d2a_next  # idx0
        dock_gap = dock_gap_a

    k_pos = float(cfg.get("k_pos", cfg.get("step_pos_coef", 0.0)))
    k_dock = float(cfg.get("k_dock", 0.0))
    lD = float(cfg.get("effort_def", 0.0))
    lA = float(cfg.get("effort_att", 0.0))
    umax2 = float(umax) * float(umax) + 1e-12
    u1n2 = _dot_expr([m.u1[i] for i in range(D)]) / umax2
    u2n2 = (
        _dot_expr([u2a_i(i) for i in range(D)]) / umax2
        + _dot_expr([u2b_i(i) for i in range(D)]) / umax2
    ) / 2.0

    g = (
        k_pos * d2_threat_next
        - k_dock * dock_gap
        - lD * u1n2
        + lA * u2n2
    )

    # Optional smooth stabilizers. Off by default to mirror env g.
    mcp_cfg = cfg.get("mcp", {}) or {}
    if bool(mcp_cfg.get("include_step_walls", False)):
        wallK = float(cfg.get("wall_penalty", 0.0))
        soft_wall = float(cfg.get("soft_wall_start", 0.5))
        wall1 = _wall_penalty_expr(p1n, center, R, soft_wall, wallK, mcpp.kappa)
        g = g - wall1

    if bool(mcp_cfg.get("include_keepout", False)) and bool(mcpp.def_use_keepout):
        oi = cfg.get("oi", {}) or {}
        oi_r_m = float(oi.get("r", 0.0))
        buf_m = float(cfg.get("def_keepout_buffer_m", 0.0))
        coef = float(cfg.get("def_center_avoid_coef", 0.0))
        keepout = _keepout_penalty_expr(p1n, center, oi_r_m, buf_m, coef, mcpp.kappa)
        g = g - keepout

    if mcpp.g_clip is not None and float(mcpp.g_clip) > 0:
        gc = float(mcpp.g_clip)
        g = gc * pyo.tanh(g / gc)

    reg = float(cfg.get("mcp", {}).get("reg_u", mcpp.reg_u))
    J1 = -g + reg * u1n2
    J2 = +g + reg * u2n2

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
    _ensure_step_mats(cfg)
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
    terminal_hist = []
    stopped_early = False

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

        term_dbg = _terminal_info_numpy(
            p_def=np.asarray(x1[:D], float),
            p_att_list=[np.asarray(x2[:D], float)],
            center=center,
            cfg=cfg,
            primary_idx=0,
        )
        terminal_hist.append(term_dbg)

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

        if term_dbg["done"] and bool(cfg.get("stop_on_done", True)):
            stopped_early = True
            break

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
        "terminal_hist": terminal_hist,
        "stopped_early": bool(stopped_early),
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
    _ensure_step_mats(cfg)
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx = 2 * D
    T_horizon = int(cfg.get("T", 1))
    umax = float(cfg.get("umax", 5e-4))

    if steps is None:
        steps = int(cfg.get("T_eval", cfg.get("steps", 60)))
    if turn_len is None:
        turn_len = 1

    ar = cfg.get("arena", {}) or {}
    ar.setdefault("type", "sphere")
    ar.setdefault("cx", 0.0); ar.setdefault("cy", 0.0); ar.setdefault("cz", 0.0)
    ar.setdefault("r", 30.0)
    center = np.array(
        [ar["cx"], ar["cy"], (ar.get("cz", 0.0) if D == 3 else 0.0)],
        dtype=np.float32,
    )[:D]

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
    terminal_hist = []
    stopped_early = False

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

        term_dbg = _terminal_info_numpy(
            p_def=np.asarray(x1[:D], float),
            p_att_list=[np.asarray(x2a[:D], float), np.asarray(x2b[:D], float)],
            center=center,
            cfg=cfg,
            primary_idx=0,
        )
        terminal_hist.append(term_dbg)

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

        if term_dbg["done"] and bool(cfg.get("stop_on_done", True)):
            stopped_early = True
            break

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
        "terminal_hist": terminal_hist,
        "stopped_early": bool(stopped_early),
    }
    return out


__all__ = [
    "solve_one_step_mcp_1v1",
    "solve_one_step_mcp_1v2_team",
    "run_rhc_with_mcp_game_1v1_collect_frames_3d",
    "run_rhc_with_mcp_game_1v2_team_collect_frames_3d",
]
