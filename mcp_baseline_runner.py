# mcp_baseline_runner.py
"""
mcp_baseline_runner.py

Historical MCP baseline runner for balloon-capture / trajectory-game experiments.

This file is not part of the current PPO/KF training or evaluation workflow. It
is kept as an optional comparison experiment for the old PATH/ASL setup, so it is
documented but intentionally isolated from game_runner.py and evaluate_policy.py.

The 1v1 path now mirrors the older Julia setup much more closely:
  - open-loop dynamic game over a control sequence
  - shared dynamics constraints and shared path inequalities
  - PATH solved through ASL / `pathampl`
  - receding-horizon execution with plan reuse for `turn_len` steps

Important practical caveat:
  - the local PATH binary in this workspace is the demo build, so the fully explicit
    Julia-style MCP must sometimes shorten the requested horizon to fit the license cap
    in 3D. The runner does that automatically for the 1v1 path.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

# Pyomo imports guarded so the file can still import in environments without it
try:
    import pyomo.environ as pyo
    import pyomo.mpec as mpec
    from pyomo.core.expr.calculus.derivatives import differentiate, Modes
except Exception as e:  # pragma: no cover
    pyo = None
    mpec = None
    differentiate = None
    Modes = None
    _PYO_IMPORT_ERR = e
else:
    _PYO_IMPORT_ERR = None


# ============================================================
# MCP params
# ============================================================

@dataclass
class MCPParams:
    """Configuration values for the historical MCP/PATH baseline runner."""
    mode: str = "balloon_capture"        # balloon-style open-loop dynamic game
    solver: str = "path"                 # retained for config compatibility
    executable: Optional[str] = None     # explicit solver binary, e.g. ~/Research/path_5/ampl/pathampl
    tee: bool = False
    pred_horizon: int = 10               # open-loop horizon
    stage_discount: float = 1.0          # per-stage discount, matching the Julia reducer when 1.0
    turn_len: int = 2                    # plan reuse length in RHC execution

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

    # Balloon/target-capture parameters ported from the Julia examples.
    target_radius: Optional[float] = None
    defender_attacker_sep: Optional[float] = None
    attacker_attacker_sep: Optional[float] = None
    defender_proximity_penalty: float = 30.0
    attacker_proximity_penalty: float = 0.5
    defender_control_weight: float = 1e-4
    attacker_control_weight: float = 1e-2
    eps: float = 1e-1
    tolerance_from_target: float = 3.0
    early_warning_rate: float = 1e-3
    include_env_constraints: bool = True

    # Julia-style asymmetric actuator limits. If unset, defaults are chosen to
    # match the old balloon-capture examples instead of a shared global `umax`.
    defender_umax: Optional[float] = None
    attacker_umax: Optional[float] = None

    # PATH / ASL options
    path_proximal: float = 0.01
    path_start: int = 1
    path_crash: int = 1
    path_major_iteration_limit: int = 20000

    # Wide boxes on primal variables; real limits are enforced through shared inequalities.
    x_var_box: float = 1e6
    u_var_box: float = 1e3


def _mcp_params_from_cfg(cfg: Dict[str, Any]) -> MCPParams:
    """Internal helper for mcp params from cfg."""
    p = MCPParams()
    d = cfg.get("mcp", {}) or {}
    for k, v in d.items():
        if hasattr(p, k):
            setattr(p, k, v)
    p.mode = (p.mode or "balloon_capture").lower()
    if p.mode == "default":
        p.mode = "balloon_capture"
    p.solver = (p.solver or "path").lower()
    if p.executable:
        p.executable = os.path.expanduser(str(p.executable))
    p.threat_mode = (p.threat_mode or "idx0").lower()
    return p


# ============================================================
# Small math helpers (smooth penalties)
# ============================================================

def _softplus_expr(x, kappa: float):
    # Numerically stable softplus:
    #   softplus(z) = max(z, 0) + log(1 + exp(-|z|))
    # with z = kappa * x, and max(z, 0) = (z + |z|) / 2.
    # This keeps the exponential argument <= 0 and avoids overflow in Pyomo AD.
    """Internal helper for softplus expr."""
    kappa = float(kappa)
    z = kappa * x
    z_abs = abs(z)
    z_pos = 0.5 * (z + z_abs)
    return (z_pos + pyo.log(1.0 + pyo.exp(-z_abs))) / kappa


def _smooth_hinge_sq(x, kappa: float):
    """Internal helper for smooth hinge sq."""
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
    """Internal helper for rho expr."""
    return pyo.sqrt(sum((p_list[i] - float(center[i])) ** 2 for i in range(len(p_list))) + 1e-12) / (float(R) + 1e-12)


def _wall_penalty_expr(p_list: List, center: np.ndarray, R: float, soft_wall: float, wallK: float, kappa: float):
    """Internal helper for wall penalty expr."""
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
    """Internal helper for dot expr."""
    return sum(v_list[i] * v_list[i] for i in range(len(v_list)))


def _stationarity_eq(expr: Any, var: Any):
    """Internal helper for stationarity eq."""
    d_expr = differentiate(expr, wrt=var, mode=Modes.reverse_symbolic)
    if isinstance(d_expr, (int, float, np.floating)):
        if abs(float(d_expr)) <= 1e-12:
            return pyo.Constraint.Feasible
        return pyo.Constraint.Infeasible
    return d_expr == 0


def _stationarity_residual(expr: Any, var: Any):
    """Internal helper for stationarity residual."""
    d_expr = differentiate(expr, wrt=var, mode=Modes.reverse_symbolic)
    if isinstance(d_expr, (int, float, np.floating)):
        return 0.0 * var + float(d_expr)
    return d_expr


def _bounded_stationarity_comp(var: Any, lb: float, ub: float, expr: Any):
    """Internal helper for bounded stationarity comp."""
    return mpec.complements(
        pyo.inequality(lb, var, ub),
        _stationarity_residual(expr, var) == 0,
    )


def _normalize_control_guess(
    u_prev: Optional[np.ndarray],
    horizon_steps: int,
    shape_tail: Tuple[int, ...],
) -> np.ndarray:
    """Normalize control guess into the canonical representation used here."""
    base_shape = (horizon_steps,) + tuple(shape_tail)
    if u_prev is None:
        return np.zeros(base_shape, dtype=float)

    arr = np.asarray(u_prev, dtype=float)
    if arr.shape == shape_tail:
        return np.repeat(arr[None, ...], horizon_steps, axis=0)

    if arr.ndim == len(shape_tail) + 1:
        if arr.shape[1:] != shape_tail:
            arr = arr.reshape((-1,) + tuple(shape_tail))
        if arr.shape[0] >= horizon_steps:
            return arr[:horizon_steps].copy()
        pad = np.repeat(arr[-1:, ...], horizon_steps - arr.shape[0], axis=0)
        return np.concatenate([arr, pad], axis=0)

    arr = arr.reshape(shape_tail)
    return np.repeat(arr[None, ...], horizon_steps, axis=0)


def _shift_plan_guess(arr: Optional[np.ndarray], shift: int) -> Optional[np.ndarray]:
    """Internal helper for shift plan guess."""
    if arr is None:
        return None
    out = np.asarray(arr, dtype=float)
    if out.ndim == 0 or out.shape[0] == 0:
        return out
    shift = max(0, int(shift))
    if shift == 0:
        return out.copy()
    if shift >= out.shape[0]:
        return np.repeat(out[-1:, ...], out.shape[0], axis=0)
    tail = out[shift:, ...]
    pad = np.repeat(out[-1:, ...], shift, axis=0)
    return np.concatenate([tail, pad], axis=0)


def _rollout_linear_guess(
    x0: np.ndarray,
    u_guess: np.ndarray,
    Ad_seq: List[np.ndarray],
    Bd_seq: List[np.ndarray],
) -> np.ndarray:
    """Internal helper for rollout linear guess."""
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    xs = np.zeros((len(u_guess) + 1, x0.shape[0]), dtype=float)
    xs[0] = x0
    for t in range(len(u_guess)):
        xs[t + 1] = Ad_seq[t] @ xs[t] + Bd_seq[t] @ np.asarray(u_guess[t], dtype=float).reshape(-1)
    return xs


def _eval_var_grid(var: Any, idx0: range, idx1: range) -> np.ndarray:
    """Internal helper for eval var grid."""
    return np.array([[float(pyo.value(var[i, j])) for j in idx1] for i in idx0], dtype=float)


def _eval_var_vec_grid(var: Any, idx0: range) -> np.ndarray:
    """Internal helper for eval var vec grid."""
    return np.array([float(pyo.value(var[i])) for i in idx0], dtype=float)


def _eval_state_sequence_np(seq: List[List[Any]]) -> np.ndarray:
    """Internal helper for eval state sequence np."""
    out = np.zeros((len(seq), len(seq[0])), dtype=float)
    for t, row in enumerate(seq):
        for j, expr in enumerate(row):
            out[t, j] = float(pyo.value(expr))
    return out


def _solve_with_pathampl(model: Any, mcpp: MCPParams):
    """Internal helper for solve with pathampl."""
    exe = mcpp.executable or os.environ.get("PATHAMPL") or "pathampl"
    solver = pyo.SolverFactory("asl")
    solver.options["solver"] = exe
    solver.options["proximal"] = float(mcpp.path_proximal)
    solver.options["start"] = int(mcpp.path_start)
    solver.options["crash"] = int(mcpp.path_crash)
    solver.options["major_iteration_limit"] = int(mcpp.path_major_iteration_limit)
    return solver.solve(model, tee=bool(mcpp.tee), load_solutions=False)


def _is_path_success(res: Any) -> bool:
    """Internal helper for is path success."""
    if res is None:
        return False
    status = str(getattr(res.solver, "status", "")).lower()
    term = getattr(res.solver, "termination_condition", None)
    return (status == "ok") and term in (
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.locallyOptimal,
        pyo.TerminationCondition.feasible,
    )


def _defender_umax_from_cfg(cfg: Dict[str, Any], mcpp: MCPParams) -> float:
    """Internal helper for defender umax from cfg."""
    if mcpp.defender_umax is not None:
        return float(mcpp.defender_umax)
    mcp_cfg = cfg.get("mcp", {}) or {}
    if "defender_umax" in mcp_cfg:
        return float(mcp_cfg["defender_umax"])
    return 10.0


def _attacker_umax_from_cfg(cfg: Dict[str, Any], mcpp: MCPParams) -> float:
    """Internal helper for attacker umax from cfg."""
    if mcpp.attacker_umax is not None:
        return float(mcpp.attacker_umax)
    mcp_cfg = cfg.get("mcp", {}) or {}
    if "attacker_umax" in mcp_cfg:
        return float(mcp_cfg["attacker_umax"])
    return 1.0


def _solve_mcp_model(model: Any, mcpp: MCPParams):
    """Internal helper for solve mcp model."""
    solver_name = str(mcpp.solver or "path").lower()
    if solver_name == "path":
        solver = pyo.SolverFactory("asl")
        solver.options["solver"] = mcpp.executable or "path"
        if solver is None or not solver.available(exception_flag=False):
            raise RuntimeError(
                f"ASL/PATH solver is not available. Configured PATH executable: "
                f"{mcpp.executable or '(default PATH lookup)'}."
            )
        return solver.solve(model, tee=bool(mcpp.tee))

    solver_kwargs = {}
    if mcpp.executable:
        solver_kwargs["executable"] = mcpp.executable
    solver = pyo.SolverFactory(solver_name, **solver_kwargs)
    if solver is None or not solver.available(exception_flag=False):
        raise RuntimeError(
            f"Solver '{solver_name}' is not available to Pyomo. "
            f"Configured executable: {mcpp.executable or '(default PATH lookup)'}."
        )
    return solver.solve(model, tee=bool(mcpp.tee))


def _ensure_step_mats(cfg: Dict[str, Any]) -> None:
    """Internal helper for ensure step mats."""
    if str(cfg.get("dynamics", "")).lower() in ("double_integrator", "balloon_capture", "planar_double_integrator"):
        return
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
    """Internal helper for terminal info numpy."""
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
    dyn_name = str(cfg.get("dynamics", "")).lower()
    if dyn_name in ("double_integrator", "balloon_capture", "planar_double_integrator"):
        dt = float(cfg.get("dt", 1.0))
        I = np.eye(D, dtype=float)
        Z = np.zeros((D, D), dtype=float)
        Ad = np.block([[I, dt * I], [Z, I]])
        Bd = np.vstack([0.5 * (dt ** 2) * I, dt * I])
        return Ad, Bd

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
    """Internal helper for step lti."""
    return (Ad @ x + Bd @ u).astype(np.float32)


# ============================================================
# Balloon-capture cost builders
# ============================================================

def _norm_sqr_expr(v: List) -> Any:
    """Internal helper for norm sqr expr."""
    return sum(v_i * v_i for v_i in v)


def _target_center(cfg: Dict[str, Any], D: int) -> np.ndarray:
    """Internal helper for target center."""
    oi = cfg.get("oi", {}) or {}
    if bool(oi.get("enabled", True)):
        return np.array(
            [oi.get("cx", 0.0), oi.get("cy", 0.0), (oi.get("cz", 0.0) if D == 3 else 0.0)],
            dtype=float,
        )[:D]
    ar = cfg.get("arena", {}) or {}
    return np.array(
        [ar.get("cx", 0.0), ar.get("cy", 0.0), (ar.get("cz", 0.0) if D == 3 else 0.0)],
        dtype=float,
    )[:D]


def _target_radius_from_cfg(cfg: Dict[str, Any], mcpp: MCPParams) -> float:
    """Internal helper for target radius from cfg."""
    if mcpp.target_radius is not None:
        return float(mcpp.target_radius)
    return 0.1


def _def_att_sep_from_cfg(cfg: Dict[str, Any], mcpp: MCPParams) -> float:
    """Internal helper for def att sep from cfg."""
    if mcpp.defender_attacker_sep is not None:
        return float(mcpp.defender_attacker_sep)
    return 1.5


def _att_att_sep_from_cfg(cfg: Dict[str, Any], mcpp: MCPParams) -> float:
    """Internal helper for att att sep from cfg."""
    if mcpp.attacker_attacker_sep is not None:
        return float(mcpp.attacker_attacker_sep)
    return 1.5


def _balloon_stage_cost_1v1(
    *,
    p_def: List,
    p_att: List,
    u_def: List,
    u_att: List,
    center: np.ndarray,
    mcpp: MCPParams,
) -> Tuple[Any, Any]:
    """Internal helper for balloon stage cost 1v1."""
    x_att_dist = [p_att[i] - float(center[i]) for i in range(len(p_att))]
    x_att_dist_def = [p_att[i] - p_def[i] for i in range(len(p_att))]

    norm_x_att_dist = _norm_sqr_expr(x_att_dist)
    norm_x_att_dist_def = _norm_sqr_expr(x_att_dist_def)

    adjusted_norm_x_att_dist = norm_x_att_dist + float(mcpp.tolerance_from_target)
    early_warning_penalty = pyo.exp(-adjusted_norm_x_att_dist / float(mcpp.early_warning_rate))

    defender_cost = (
        -((float(mcpp.defender_proximity_penalty) / (float(mcpp.eps) + norm_x_att_dist_def)) + adjusted_norm_x_att_dist)
        - early_warning_penalty
        + float(mcpp.defender_control_weight) * _norm_sqr_expr(u_def)
    )

    attacker_cost = (
        norm_x_att_dist
        + (float(mcpp.attacker_proximity_penalty) / (float(mcpp.eps) + norm_x_att_dist_def))
        + float(mcpp.attacker_control_weight) * _norm_sqr_expr(u_att)
    )
    return defender_cost, attacker_cost


def _balloon_stage_cost_1v2_team(
    *,
    p_def: List,
    p_att_a: List,
    p_att_b: List,
    u_def: List,
    u_att_a: List,
    u_att_b: List,
    center: np.ndarray,
    mcpp: MCPParams,
) -> Tuple[Any, Any]:
    """Internal helper for balloon stage cost 1v2 team."""
    x2_dist = [p_att_a[i] - float(center[i]) for i in range(len(p_def))]
    x3_dist = [p_att_b[i] - float(center[i]) for i in range(len(p_def))]
    x2_dist_def = [p_att_a[i] - p_def[i] for i in range(len(p_def))]
    x3_dist_def = [p_att_b[i] - p_def[i] for i in range(len(p_def))]

    norm_x2_dist = _norm_sqr_expr(x2_dist)
    norm_x3_dist = _norm_sqr_expr(x3_dist)
    norm_x2_dist_def = _norm_sqr_expr(x2_dist_def)
    norm_x3_dist_def = _norm_sqr_expr(x3_dist_def)

    adjusted_norm_x2_dist = norm_x2_dist + float(mcpp.tolerance_from_target)
    adjusted_norm_x3_dist = norm_x3_dist + float(mcpp.tolerance_from_target)

    weight_x2 = 1.0 / (float(mcpp.eps) + adjusted_norm_x2_dist)
    weight_x3 = 1.0 / (float(mcpp.eps) + adjusted_norm_x3_dist)
    total_weight = weight_x2 + weight_x3 + 1e-12
    normalized_weight_x2 = weight_x2 / total_weight
    normalized_weight_x3 = weight_x3 / total_weight

    early_warning_penalty_x2 = pyo.exp(-adjusted_norm_x2_dist / float(mcpp.early_warning_rate))
    early_warning_penalty_x3 = pyo.exp(-adjusted_norm_x3_dist / float(mcpp.early_warning_rate))

    defender_cost = (
        -(
            normalized_weight_x2
            * ((float(mcpp.defender_proximity_penalty) / (float(mcpp.eps) + norm_x2_dist_def)) + adjusted_norm_x2_dist)
            + normalized_weight_x3
            * ((float(mcpp.defender_proximity_penalty) / (float(mcpp.eps) + norm_x3_dist_def)) + adjusted_norm_x3_dist)
        )
        - early_warning_penalty_x2
        - early_warning_penalty_x3
        + float(mcpp.defender_control_weight) * _norm_sqr_expr(u_def)
    )

    attacker_team_cost = (
        norm_x2_dist
        + (float(mcpp.attacker_proximity_penalty) / (float(mcpp.eps) + norm_x2_dist_def))
        + float(mcpp.attacker_control_weight) * _norm_sqr_expr(u_att_a)
        + norm_x3_dist
        + (float(mcpp.attacker_proximity_penalty) / (float(mcpp.eps) + norm_x3_dist_def))
        + float(mcpp.attacker_control_weight) * _norm_sqr_expr(u_att_b)
    )
    return defender_cost, attacker_team_cost

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
    """Build zero sum g 1v1 for the current workflow."""
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
    """Handle solve one step mcp 1v1 for this workflow."""
    if pyo is None or mpec is None or differentiate is None:
        raise RuntimeError(
            f"Pyomo MPEC is not available in this environment: {_PYO_IMPORT_ERR}\n"
            f"Install: pip install pyomo  (and ensure PATH solver is installed/licensed)."
        )

    mcpp = _mcp_params_from_cfg(cfg)
    D = int(cfg.get("D", 3))
    nx = 2 * D
    pred_horizon = max(1, int(mcpp.pred_horizon))
    stage_discount = float(mcpp.stage_discount)
    total_steps = int(cfg.get("T", pred_horizon))
    requested_horizon = min(pred_horizon, max(1, total_steps - k))
    center = _target_center(cfg, D)
    ar = cfg.get("arena", {}) or {}
    R = float(ar.get("r", 30.0))
    target_radius = _target_radius_from_cfg(cfg, mcpp)
    sep_def_att = _def_att_sep_from_cfg(cfg, mcpp)
    def_umax = _defender_umax_from_cfg(cfg, mcpp)
    att_umax = _attacker_umax_from_cfg(cfg, mcpp)

    x1 = np.asarray(x1, float).reshape(nx)
    x2 = np.asarray(x2, float).reshape(nx)
    last_res = None
    last_exc: Optional[Exception] = None

    def _attempt_1v1(horizon_steps: int, use_env_constraints: bool):
        """Internal helper for attempt 1v1."""
        u1_guess = _normalize_control_guess(u1_prev, horizon_steps, (D,))
        u2_guess = _normalize_control_guess(u2_prev, horizon_steps, (D,))

        Ad_seq: List[np.ndarray] = []
        Bd_seq: List[np.ndarray] = []
        for t in range(horizon_steps):
            Ad_h, Bd_h = _extract_step_mats(cfg, k=min(k + t, total_steps - 1), D=D)
            Ad_seq.append(np.asarray(Ad_h, dtype=float))
            Bd_seq.append(np.asarray(Bd_h, dtype=float))

        x1_guess = _rollout_linear_guess(x1, u1_guess, Ad_seq, Bd_seq)
        x2_guess = _rollout_linear_guess(x2, u2_guess, Ad_seq, Bd_seq)

        m = pyo.ConcreteModel()
        m.Kx = pyo.RangeSet(0, horizon_steps)
        m.Ku = pyo.RangeSet(0, horizon_steps - 1)
        m.S = pyo.RangeSet(0, nx - 1)
        m.I = pyo.RangeSet(0, D - 1)

        m.x1 = pyo.Var(m.Kx, m.S, bounds=(-float(mcpp.x_var_box), float(mcpp.x_var_box)))
        m.x2 = pyo.Var(m.Kx, m.S, bounds=(-float(mcpp.x_var_box), float(mcpp.x_var_box)))
        m.u1 = pyo.Var(m.Ku, m.I, bounds=(-float(mcpp.u_var_box), float(mcpp.u_var_box)))
        m.u2 = pyo.Var(m.Ku, m.I, bounds=(-float(mcpp.u_var_box), float(mcpp.u_var_box)))

        m.lam_ic1 = pyo.Var(m.S, domain=pyo.Reals)
        m.lam_ic2 = pyo.Var(m.S, domain=pyo.Reals)
        m.lam_dyn1 = pyo.Var(m.Ku, m.S, domain=pyo.Reals)
        m.lam_dyn2 = pyo.Var(m.Ku, m.S, domain=pyo.Reals)

        m.mu_sep = pyo.Var(m.Kx, domain=pyo.NonNegativeReals)
        if target_radius > 0.0:
            m.mu_keepout = pyo.Var(m.Kx, domain=pyo.NonNegativeReals)
        if use_env_constraints:
            m.mu_env1 = pyo.Var(m.Kx, domain=pyo.NonNegativeReals)
            m.mu_env2 = pyo.Var(m.Kx, domain=pyo.NonNegativeReals)
        m.mu_u1_lo = pyo.Var(m.Ku, m.I, domain=pyo.NonNegativeReals)
        m.mu_u1_hi = pyo.Var(m.Ku, m.I, domain=pyo.NonNegativeReals)
        m.mu_u2_lo = pyo.Var(m.Ku, m.I, domain=pyo.NonNegativeReals)
        m.mu_u2_hi = pyo.Var(m.Ku, m.I, domain=pyo.NonNegativeReals)

        def g_ic1_expr(_m, i):
            """Handle g ic1 expr for this workflow."""
            return _m.x1[0, i] - float(x1[i])

        def g_ic2_expr(_m, i):
            """Handle g ic2 expr for this workflow."""
            return _m.x2[0, i] - float(x2[i])

        def g_dyn1_expr(_m, t, i):
            """Handle g dyn1 expr for this workflow."""
            return _m.x1[t + 1, i] - sum(float(Ad_seq[t][i, j]) * _m.x1[t, j] for j in _m.S) - sum(
                float(Bd_seq[t][i, j]) * _m.u1[t, j] for j in _m.I
            )

        def g_dyn2_expr(_m, t, i):
            """Handle g dyn2 expr for this workflow."""
            return _m.x2[t + 1, i] - sum(float(Ad_seq[t][i, j]) * _m.x2[t, j] for j in _m.S) - sum(
                float(Bd_seq[t][i, j]) * _m.u2[t, j] for j in _m.I
            )

        m.ic1 = pyo.Constraint(m.S, rule=lambda _m, i: g_ic1_expr(_m, i) == 0)
        m.ic2 = pyo.Constraint(m.S, rule=lambda _m, i: g_ic2_expr(_m, i) == 0)
        m.dyn1 = pyo.Constraint(m.Ku, m.S, rule=lambda _m, t, i: g_dyn1_expr(_m, t, i) == 0)
        m.dyn2 = pyo.Constraint(m.Ku, m.S, rule=lambda _m, t, i: g_dyn2_expr(_m, t, i) == 0)

        m.h_sep = pyo.Expression(
            m.Kx,
            rule=lambda _m, t: _norm_sqr_expr([_m.x2[t, i] - _m.x1[t, i] for i in range(D)]) - sep_def_att,
        )
        if target_radius > 0.0:
            m.h_keepout = pyo.Expression(
                m.Kx,
                rule=lambda _m, t: _norm_sqr_expr([_m.x1[t, i] - float(center[i]) for i in range(D)]) - target_radius,
            )
        if use_env_constraints:
            m.h_env1 = pyo.Expression(
                m.Kx,
                rule=lambda _m, t: float(R * R) - _norm_sqr_expr([_m.x1[t, i] - float(center[i]) for i in range(D)]),
            )
            m.h_env2 = pyo.Expression(
                m.Kx,
                rule=lambda _m, t: float(R * R) - _norm_sqr_expr([_m.x2[t, i] - float(center[i]) for i in range(D)]),
            )
        m.h_u1_lo = pyo.Expression(m.Ku, m.I, rule=lambda _m, t, i: _m.u1[t, i] + def_umax)
        m.h_u1_hi = pyo.Expression(m.Ku, m.I, rule=lambda _m, t, i: def_umax - _m.u1[t, i])
        m.h_u2_lo = pyo.Expression(m.Ku, m.I, rule=lambda _m, t, i: _m.u2[t, i] + att_umax)
        m.h_u2_hi = pyo.Expression(m.Ku, m.I, rule=lambda _m, t, i: att_umax - _m.u2[t, i])

        m.comp_sep = mpec.Complementarity(m.Kx, rule=lambda _m, t: mpec.complements(_m.mu_sep[t] >= 0, _m.h_sep[t] >= 0))
        if target_radius > 0.0:
            m.comp_keepout = mpec.Complementarity(
                m.Kx, rule=lambda _m, t: mpec.complements(_m.mu_keepout[t] >= 0, _m.h_keepout[t] >= 0)
            )
        if use_env_constraints:
            m.comp_env1 = mpec.Complementarity(
                m.Kx, rule=lambda _m, t: mpec.complements(_m.mu_env1[t] >= 0, _m.h_env1[t] >= 0)
            )
            m.comp_env2 = mpec.Complementarity(
                m.Kx, rule=lambda _m, t: mpec.complements(_m.mu_env2[t] >= 0, _m.h_env2[t] >= 0)
            )
        m.comp_u1_lo = mpec.Complementarity(
            m.Ku, m.I, rule=lambda _m, t, i: mpec.complements(_m.mu_u1_lo[t, i] >= 0, _m.h_u1_lo[t, i] >= 0)
        )
        m.comp_u1_hi = mpec.Complementarity(
            m.Ku, m.I, rule=lambda _m, t, i: mpec.complements(_m.mu_u1_hi[t, i] >= 0, _m.h_u1_hi[t, i] >= 0)
        )
        m.comp_u2_lo = mpec.Complementarity(
            m.Ku, m.I, rule=lambda _m, t, i: mpec.complements(_m.mu_u2_lo[t, i] >= 0, _m.h_u2_lo[t, i] >= 0)
        )
        m.comp_u2_hi = mpec.Complementarity(
            m.Ku, m.I, rule=lambda _m, t, i: mpec.complements(_m.mu_u2_hi[t, i] >= 0, _m.h_u2_hi[t, i] >= 0)
        )

        stage_costs_1 = []
        stage_costs_2 = []
        for t in range(horizon_steps):
            c1_t, c2_t = _balloon_stage_cost_1v1(
                p_def=[m.x1[t, i] for i in range(D)],
                p_att=[m.x2[t, i] for i in range(D)],
                u_def=[m.u1[t, i] for i in range(D)],
                u_att=[m.u2[t, i] for i in range(D)],
                center=center,
                mcpp=mcpp,
            )
            w = stage_discount ** t
            stage_costs_1.append(w * c1_t)
            stage_costs_2.append(w * c2_t)
        m.J1 = pyo.Expression(expr=sum(stage_costs_1) / float(horizon_steps))
        m.J2 = pyo.Expression(expr=sum(stage_costs_2) / float(horizon_steps))

        def eq_grad_lam_1(_m, var, t_hint):
            """Handle eq grad lam 1 for this workflow."""
            out = 0.0
            if t_hint == 0:
                for j in _m.S:
                    out += _m.lam_ic1[j] * differentiate(g_ic1_expr(_m, j), wrt=var, mode=Modes.reverse_symbolic)
            if t_hint in _m.Ku:
                for j in _m.S:
                    out += _m.lam_dyn1[t_hint, j] * differentiate(g_dyn1_expr(_m, t_hint, j), wrt=var, mode=Modes.reverse_symbolic)
            if (t_hint - 1) in _m.Ku:
                for j in _m.S:
                    out += _m.lam_dyn1[t_hint - 1, j] * differentiate(
                        g_dyn1_expr(_m, t_hint - 1, j), wrt=var, mode=Modes.reverse_symbolic
                    )
            return out

        def eq_grad_lam_2(_m, var, t_hint):
            """Handle eq grad lam 2 for this workflow."""
            out = 0.0
            if t_hint == 0:
                for j in _m.S:
                    out += _m.lam_ic2[j] * differentiate(g_ic2_expr(_m, j), wrt=var, mode=Modes.reverse_symbolic)
            if t_hint in _m.Ku:
                for j in _m.S:
                    out += _m.lam_dyn2[t_hint, j] * differentiate(g_dyn2_expr(_m, t_hint, j), wrt=var, mode=Modes.reverse_symbolic)
            if (t_hint - 1) in _m.Ku:
                for j in _m.S:
                    out += _m.lam_dyn2[t_hint - 1, j] * differentiate(
                        g_dyn2_expr(_m, t_hint - 1, j), wrt=var, mode=Modes.reverse_symbolic
                    )
            return out

        def ineq_grad(_m, var, t_hint):
            """Handle ineq grad for this workflow."""
            out = _m.mu_sep[t_hint] * differentiate(_m.h_sep[t_hint], wrt=var, mode=Modes.reverse_symbolic)
            if target_radius > 0.0:
                out += _m.mu_keepout[t_hint] * differentiate(_m.h_keepout[t_hint], wrt=var, mode=Modes.reverse_symbolic)
            if use_env_constraints:
                out += _m.mu_env1[t_hint] * differentiate(_m.h_env1[t_hint], wrt=var, mode=Modes.reverse_symbolic)
                out += _m.mu_env2[t_hint] * differentiate(_m.h_env2[t_hint], wrt=var, mode=Modes.reverse_symbolic)
            if t_hint in _m.Ku:
                for j in _m.I:
                    out += _m.mu_u1_lo[t_hint, j] * differentiate(_m.h_u1_lo[t_hint, j], wrt=var, mode=Modes.reverse_symbolic)
                    out += _m.mu_u1_hi[t_hint, j] * differentiate(_m.h_u1_hi[t_hint, j], wrt=var, mode=Modes.reverse_symbolic)
                    out += _m.mu_u2_lo[t_hint, j] * differentiate(_m.h_u2_lo[t_hint, j], wrt=var, mode=Modes.reverse_symbolic)
                    out += _m.mu_u2_hi[t_hint, j] * differentiate(_m.h_u2_hi[t_hint, j], wrt=var, mode=Modes.reverse_symbolic)
            return out

        m.st_x1 = pyo.Constraint(
            m.Kx,
            m.S,
            rule=lambda _m, t, i: differentiate(_m.J1, wrt=_m.x1[t, i], mode=Modes.reverse_symbolic)
            - eq_grad_lam_1(_m, _m.x1[t, i], t)
            - ineq_grad(_m, _m.x1[t, i], t)
            == 0,
        )
        m.st_x2 = pyo.Constraint(
            m.Kx,
            m.S,
            rule=lambda _m, t, i: differentiate(_m.J2, wrt=_m.x2[t, i], mode=Modes.reverse_symbolic)
            - eq_grad_lam_2(_m, _m.x2[t, i], t)
            - ineq_grad(_m, _m.x2[t, i], t)
            == 0,
        )
        m.st_u1 = pyo.Constraint(
            m.Ku,
            m.I,
            rule=lambda _m, t, i: differentiate(_m.J1, wrt=_m.u1[t, i], mode=Modes.reverse_symbolic)
            - eq_grad_lam_1(_m, _m.u1[t, i], t)
            - ineq_grad(_m, _m.u1[t, i], t)
            == 0,
        )
        m.st_u2 = pyo.Constraint(
            m.Ku,
            m.I,
            rule=lambda _m, t, i: differentiate(_m.J2, wrt=_m.u2[t, i], mode=Modes.reverse_symbolic)
            - eq_grad_lam_2(_m, _m.u2[t, i], t)
            - ineq_grad(_m, _m.u2[t, i], t)
            == 0,
        )

        for t in range(horizon_steps + 1):
            for j in range(nx):
                m.x1[t, j].value = float(x1_guess[t, j])
                m.x2[t, j].value = float(x2_guess[t, j])
        for t in range(horizon_steps):
            for j in range(D):
                m.u1[t, j].value = float(u1_guess[t, j])
                m.u2[t, j].value = float(u2_guess[t, j])
        for var in m.component_data_objects(pyo.Var):
            if var.value is None:
                var.value = 0.0

        res = _solve_with_pathampl(m, mcpp)
        if not _is_path_success(res):
            return None, res

        m.solutions.load_from(res, default_variable_value=None)
        plan_x1 = _eval_var_grid(m.x1, range(horizon_steps + 1), range(nx))
        plan_x2 = _eval_var_grid(m.x2, range(horizon_steps + 1), range(nx))
        plan_u1 = _eval_var_grid(m.u1, range(horizon_steps), range(D))
        plan_u2 = _eval_var_grid(m.u2, range(horizon_steps), range(D))
        u1_star = plan_u1[0].astype(np.float32)
        u2_star = plan_u2[0].astype(np.float32)
        dbg = {
            "mode": mcpp.mode,
            "solver": "asl/pathampl",
            "pred_horizon": int(horizon_steps),
            "requested_horizon": int(requested_horizon),
            "used_horizon": int(horizon_steps),
            "stage_discount": float(stage_discount),
            "termination": str(getattr(res.solver, "termination_condition", "")),
            "status": str(getattr(res.solver, "status", "")),
            "J1": float(pyo.value(m.J1)),
            "J2": float(pyo.value(m.J2)),
            "u1_norm": float(np.linalg.norm(u1_star)),
            "u2_norm": float(np.linalg.norm(u2_star)),
            "target_radius": float(target_radius),
            "defender_attacker_sep": float(sep_def_att),
            "defender_umax": float(def_umax),
            "attacker_umax": float(att_umax),
            "env_constraints_active": bool(use_env_constraints),
            "env_constraints_relaxed": bool(bool(mcpp.include_env_constraints) and not use_env_constraints),
            "license_capped": bool(horizon_steps != requested_horizon),
            "plan_x1": plan_x1,
            "plan_x2": plan_x2,
            "plan_u1": plan_u1,
            "plan_u2": plan_u2,
        }
        return (u1_star, u2_star, dbg), res

    env_constraint_modes = [bool(mcpp.include_env_constraints)]
    if bool(mcpp.include_env_constraints):
        env_constraint_modes.append(False)

    for use_env_constraints in env_constraint_modes:
        for horizon_steps in range(requested_horizon, 0, -1):
            try:
                out, res = _attempt_1v1(horizon_steps, use_env_constraints)
            except Exception as exc:
                last_exc = exc
                continue
            last_res = res
            if out is not None:
                return out

    msg = "PATH failed for every attempted 1v1 horizon."
    if last_res is not None:
        msg += (
            f" Last status={getattr(last_res.solver, 'status', '')},"
            f" termination={getattr(last_res.solver, 'termination_condition', '')}."
        )
    if last_exc is not None:
        msg += f" Last exception={type(last_exc).__name__}: {last_exc}"
    raise RuntimeError(msg)


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
    """Handle solve one step mcp 1v2 team for this workflow."""
    if pyo is None or mpec is None or differentiate is None:
        raise RuntimeError(
            f"Pyomo MPEC is not available in this environment: {_PYO_IMPORT_ERR}\n"
            f"Install: pip install pyomo  (and ensure PATH solver is installed/licensed)."
        )

    mcpp = _mcp_params_from_cfg(cfg)
    D = int(cfg.get("D", 3))
    nx = 2 * D
    umax = float(cfg.get("umax", 5e-4))
    pred_horizon = max(1, int(mcpp.pred_horizon))
    stage_discount = float(mcpp.stage_discount)
    total_steps = int(cfg.get("T", pred_horizon))
    horizon_steps = min(pred_horizon, max(1, total_steps - k))
    center = _target_center(cfg, D)
    ar = cfg.get("arena", {}) or {}
    R = float(ar.get("r", 30.0))
    target_radius = _target_radius_from_cfg(cfg, mcpp)
    sep_def_att = _def_att_sep_from_cfg(cfg, mcpp)
    sep_att_att = _att_att_sep_from_cfg(cfg, mcpp)

    x1 = np.asarray(x1, float).reshape(nx)
    x2a = np.asarray(x2a, float).reshape(nx)
    x2b = np.asarray(x2b, float).reshape(nx)

    u1_guess = _normalize_control_guess(u1_prev, horizon_steps, (D,))
    u2_guess = _normalize_control_guess(u2_prev, horizon_steps, (2, D))

    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, horizon_steps)
    m.U = pyo.RangeSet(0, horizon_steps - 1)
    m.I = pyo.RangeSet(0, D - 1)

    m.u1 = pyo.Var(
        m.U,
        m.I,
        bounds=(-umax, umax),
        initialize=lambda mdl, t, i: float(u1_guess[t, i]),
    )
    m.u2a = pyo.Var(
        m.U,
        m.I,
        bounds=(-umax, umax),
        initialize=lambda mdl, t, i: float(u2_guess[t, 0, i]),
    )
    m.u2b = pyo.Var(
        m.U,
        m.I,
        bounds=(-umax, umax),
        initialize=lambda mdl, t, i: float(u2_guess[t, 1, i]),
    )

    m.mu_sep_12 = pyo.Var(m.T, within=pyo.NonNegativeReals, initialize=0.0)
    m.mu_sep_23 = pyo.Var(m.T, within=pyo.NonNegativeReals, initialize=0.0)
    m.mu_sep_31 = pyo.Var(m.T, within=pyo.NonNegativeReals, initialize=0.0)
    m.mu_keepout = pyo.Var(m.T, within=pyo.NonNegativeReals, initialize=0.0)
    if bool(mcpp.include_env_constraints):
        m.mu_env1 = pyo.Var(m.T, within=pyo.NonNegativeReals, initialize=0.0)
        m.mu_env2a = pyo.Var(m.T, within=pyo.NonNegativeReals, initialize=0.0)
        m.mu_env2b = pyo.Var(m.T, within=pyo.NonNegativeReals, initialize=0.0)

    x1_seq: List[List[Any]] = [[float(x1[j]) for j in range(nx)]]
    x2a_seq: List[List[Any]] = [[float(x2a[j]) for j in range(nx)]]
    x2b_seq: List[List[Any]] = [[float(x2b[j]) for j in range(nx)]]
    for t in range(horizon_steps):
        Ad_h, Bd_h = _extract_step_mats(cfg, k=min(k + t, total_steps - 1), D=D)
        x1_seq.append(
            [
                sum(float(Ad_h[j, s]) * x1_seq[t][s] for s in range(nx))
                + sum(float(Bd_h[j, i]) * m.u1[t, i] for i in range(D))
                for j in range(nx)
            ]
        )
        x2a_seq.append(
            [
                sum(float(Ad_h[j, s]) * x2a_seq[t][s] for s in range(nx))
                + sum(float(Bd_h[j, i]) * m.u2a[t, i] for i in range(D))
                for j in range(nx)
            ]
        )
        x2b_seq.append(
            [
                sum(float(Ad_h[j, s]) * x2b_seq[t][s] for s in range(nx))
                + sum(float(Bd_h[j, i]) * m.u2b[t, i] for i in range(D))
                for j in range(nx)
            ]
        )

    m.h_sep_12 = pyo.Expression(
        m.T,
        rule=lambda mdl, t: _norm_sqr_expr([x2a_seq[t][i] - x1_seq[t][i] for i in range(D)]) - sep_def_att,
    )
    m.h_sep_23 = pyo.Expression(
        m.T,
        rule=lambda mdl, t: _norm_sqr_expr([x2b_seq[t][i] - x2a_seq[t][i] for i in range(D)]) - sep_att_att,
    )
    m.h_sep_31 = pyo.Expression(
        m.T,
        rule=lambda mdl, t: _norm_sqr_expr([x1_seq[t][i] - x2b_seq[t][i] for i in range(D)]) - sep_def_att,
    )
    m.h_keepout = pyo.Expression(
        m.T,
        rule=lambda mdl, t: _norm_sqr_expr([x1_seq[t][i] - float(center[i]) for i in range(D)]) - target_radius,
    )
    if bool(mcpp.include_env_constraints):
        m.h_env1 = pyo.Expression(
            m.T,
            rule=lambda mdl, t: float(R * R)
            - _norm_sqr_expr([x1_seq[t][i] - float(center[i]) for i in range(D)]),
        )
        m.h_env2a = pyo.Expression(
            m.T,
            rule=lambda mdl, t: float(R * R)
            - _norm_sqr_expr([x2a_seq[t][i] - float(center[i]) for i in range(D)]),
        )
        m.h_env2b = pyo.Expression(
            m.T,
            rule=lambda mdl, t: float(R * R)
            - _norm_sqr_expr([x2b_seq[t][i] - float(center[i]) for i in range(D)]),
        )

    stage_costs_1 = []
    stage_costs_2 = []
    for t in range(horizon_steps):
        c1_t, c2_t = _balloon_stage_cost_1v2_team(
            p_def=[x1_seq[t][i] for i in range(D)],
            p_att_a=[x2a_seq[t][i] for i in range(D)],
            p_att_b=[x2b_seq[t][i] for i in range(D)],
            u_def=[m.u1[t, i] for i in range(D)],
            u_att_a=[m.u2a[t, i] for i in range(D)],
            u_att_b=[m.u2b[t, i] for i in range(D)],
            center=center,
            mcpp=mcpp,
        )
        w = stage_discount ** t
        stage_costs_1.append(w * c1_t)
        stage_costs_2.append(w * c2_t)

    m.J1 = pyo.Expression(expr=sum(stage_costs_1) / float(horizon_steps))
    m.J2 = pyo.Expression(expr=sum(stage_costs_2) / float(horizon_steps))

    ineq_terms = []
    ineq_mults = []
    for t in range(horizon_steps + 1):
        ineq_terms.extend([m.h_sep_12[t], m.h_sep_23[t], m.h_sep_31[t]])
        ineq_mults.extend([m.mu_sep_12[t], m.mu_sep_23[t], m.mu_sep_31[t]])
        if target_radius > 0.0:
            ineq_terms.append(m.h_keepout[t]); ineq_mults.append(m.mu_keepout[t])
        if bool(mcpp.include_env_constraints):
            ineq_terms.extend([m.h_env1[t], m.h_env2a[t], m.h_env2b[t]])
            ineq_mults.extend([m.mu_env1[t], m.mu_env2a[t], m.mu_env2b[t]])

    m.L1 = pyo.Expression(expr=m.J1 - sum(ineq_mults[i] * ineq_terms[i] for i in range(len(ineq_terms))))
    m.L2 = pyo.Expression(expr=m.J2 - sum(ineq_mults[i] * ineq_terms[i] for i in range(len(ineq_terms))))

    m.comp_u1 = mpec.Complementarity(
        m.U,
        m.I,
        rule=lambda mdl, t, i: mpec.complements(pyo.inequality(-umax, mdl.u1[t, i], umax), _stationarity_residual(mdl.L1.expr, mdl.u1[t, i])),
    )
    m.comp_u2a = mpec.Complementarity(
        m.U,
        m.I,
        rule=lambda mdl, t, i: mpec.complements(pyo.inequality(-umax, mdl.u2a[t, i], umax), _stationarity_residual(mdl.L2.expr, mdl.u2a[t, i])),
    )
    m.comp_u2b = mpec.Complementarity(
        m.U,
        m.I,
        rule=lambda mdl, t, i: mpec.complements(pyo.inequality(-umax, mdl.u2b[t, i], umax), _stationarity_residual(mdl.L2.expr, mdl.u2b[t, i])),
    )
    m.comp_sep_12 = mpec.Complementarity(
        m.T, rule=lambda mdl, t: mpec.complements(mdl.h_sep_12[t] >= 0, mdl.mu_sep_12[t] >= 0)
    )
    m.comp_sep_23 = mpec.Complementarity(
        m.T, rule=lambda mdl, t: mpec.complements(mdl.h_sep_23[t] >= 0, mdl.mu_sep_23[t] >= 0)
    )
    m.comp_sep_31 = mpec.Complementarity(
        m.T, rule=lambda mdl, t: mpec.complements(mdl.h_sep_31[t] >= 0, mdl.mu_sep_31[t] >= 0)
    )
    if target_radius > 0.0:
        m.comp_keepout = mpec.Complementarity(
            m.T, rule=lambda mdl, t: mpec.complements(mdl.h_keepout[t] >= 0, mdl.mu_keepout[t] >= 0)
        )
    if bool(mcpp.include_env_constraints):
        m.comp_env1 = mpec.Complementarity(
            m.T, rule=lambda mdl, t: mpec.complements(mdl.h_env1[t] >= 0, mdl.mu_env1[t] >= 0)
        )
        m.comp_env2a = mpec.Complementarity(
            m.T, rule=lambda mdl, t: mpec.complements(mdl.h_env2a[t] >= 0, mdl.mu_env2a[t] >= 0)
        )
        m.comp_env2b = mpec.Complementarity(
            m.T, rule=lambda mdl, t: mpec.complements(mdl.h_env2b[t] >= 0, mdl.mu_env2b[t] >= 0)
        )

    try:
        pyo.TransformationFactory("mpec.standard_form").apply_to(m)
    except Exception as e:
        raise RuntimeError("Failed to apply mpec.standard_form for 1v2 team MCP.") from e

    solver_kwargs = {}
    if mcpp.executable:
        solver_kwargs["executable"] = mcpp.executable
    solver = pyo.SolverFactory(mcpp.solver, **solver_kwargs)
    if solver is None or not solver.available(exception_flag=False):
        raise RuntimeError(
            f"Solver '{mcpp.solver}' not available to Pyomo. "
            f"Configured executable: {mcpp.executable or '(default PATH lookup)'}. "
            "Need a working PATH installation."
        )

    res = solver.solve(m, tee=bool(mcpp.tee))

    u1_star = np.array([pyo.value(m.u1[0, i]) for i in range(D)], dtype=np.float32)
    u2a_star = np.array([pyo.value(m.u2a[0, i]) for i in range(D)], dtype=np.float32)
    u2b_star = np.array([pyo.value(m.u2b[0, i]) for i in range(D)], dtype=np.float32)

    dbg = {
        "mode": mcpp.mode,
        "solver": mcpp.solver,
        "pred_horizon": int(horizon_steps),
        "stage_discount": float(stage_discount),
        "termination": str(getattr(res.solver, "termination_condition", "")),
        "J1": float(pyo.value(m.J1)),
        "J2": float(pyo.value(m.J2)),
        "u1_norm": float(np.linalg.norm(u1_star)),
        "u2a_norm": float(np.linalg.norm(u2a_star)),
        "u2b_norm": float(np.linalg.norm(u2b_star)),
        "target_radius": float(target_radius),
        "defender_attacker_sep": float(sep_def_att),
        "attacker_attacker_sep": float(sep_att_att),
        "plan_x1": _eval_state_sequence_np(x1_seq),
        "plan_x2a": _eval_state_sequence_np(x2a_seq),
        "plan_x2b": _eval_state_sequence_np(x2b_seq),
        "plan_u1": np.array([[pyo.value(m.u1[t, i]) for i in range(D)] for t in range(horizon_steps)], dtype=float),
        "plan_u2a": np.array([[pyo.value(m.u2a[t, i]) for i in range(D)] for t in range(horizon_steps)], dtype=float),
        "plan_u2b": np.array([[pyo.value(m.u2b[t, i]) for i in range(D)] for t in range(horizon_steps)], dtype=float),
    }
    return u1_star, u2a_star, u2b_star, dbg


# ============================================================
# Rollout helpers (frames_dict styling like your other runners)
# ============================================================

def _p3(xD: np.ndarray, D: int):
    """Internal helper for p3."""
    xD = np.asarray(xD, float).reshape(-1)
    if D == 3:
        return (float(xD[0]), float(xD[1]), float(xD[2]))
    return (float(xD[0]), float(xD[1]), 0.0)


def _identity_R():
    """Internal helper for identity  r."""
    return np.eye(3, dtype=float)


def _pad3(u: np.ndarray, D: int):
    """Internal helper for pad3."""
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
    mcpp = _mcp_params_from_cfg(cfg)
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx = 2 * D
    T_horizon = max(1, int(mcpp.pred_horizon))
    def_umax = _defender_umax_from_cfg(cfg, mcpp)
    att_umax = _attacker_umax_from_cfg(cfg, mcpp)

    if steps is None:
        steps = int(cfg.get("T_eval", cfg.get("T", cfg.get("steps", 60))))
    if turn_len is None:
        turn_len = int(mcpp.turn_len)
    turn_len = max(1, int(turn_len))

    # arena
    ar = cfg.get("arena", {}) or {}
    ar.setdefault("type", "sphere")
    ar.setdefault("cx", 0.0); ar.setdefault("cy", 0.0); ar.setdefault("cz", 0.0)
    ar.setdefault("r", 30.0)
    center = np.array([ar["cx"], ar["cy"], (ar.get("cz", 0.0) if D == 3 else 0.0)], dtype=np.float32)[:D]
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

    active_plan: Optional[Dict[str, Any]] = None
    active_plan_step = 0
    active_plan_len = 0

    def _plan_xyz(plan_x: np.ndarray, step_idx: int) -> List[Tuple[float, float, float]]:
        """Internal helper for plan xyz."""
        tail = np.asarray(plan_x[max(0, step_idx):, :D], dtype=float)
        if tail.shape[0] == 0:
            tail = np.asarray(plan_x[-1:, :D], dtype=float)
        pts = [_p3(row, D) for row in tail]
        if len(pts) < T_horizon:
            pts.extend([pts[-1]] * (T_horizon - len(pts)))
        return pts[:T_horizon]

    for k in range(steps):
        need_replan = (
            active_plan is None
            or active_plan_step >= active_plan_len
            or active_plan_step >= int(np.asarray(active_plan["plan_u1"]).shape[0])
        )
        if need_replan:
            warm_u1 = None if active_plan is None else _shift_plan_guess(active_plan["plan_u1"], active_plan_step)
            warm_u2 = None if active_plan is None else _shift_plan_guess(active_plan["plan_u2"], active_plan_step)
            _, _, active_plan = solve_one_step_mcp_1v1(cfg, x1=x1, x2=x2, k=k, u1_prev=warm_u1, u2_prev=warm_u2)
            active_plan_step = 0
            active_plan_len = max(1, min(turn_len, int(active_plan["used_horizon"])))

        u1 = np.asarray(active_plan["plan_u1"][active_plan_step], dtype=np.float32)
        u2 = np.asarray(active_plan["plan_u2"][active_plan_step], dtype=np.float32)
        u1 = np.clip(u1, -def_umax, +def_umax).astype(np.float32)
        u2 = np.clip(u2, -att_umax, +att_umax).astype(np.float32)

        dbg_step = dict(active_plan)
        dbg_step["plan_step_idx"] = int(active_plan_step)
        dbg_step["turn_len"] = int(active_plan_len)
        dbg_hist.append(dbg_step)

        # log thrust
        u1_3 = _pad3(u1, D); u2_3 = _pad3(u2, D)
        u_cmd_1.append(u1_3.copy()); u_cmd_norm_1.append(float(np.linalg.norm(u1_3)))
        u_cmd_2.append(u2_3.copy()); u_cmd_norm_2.append(float(np.linalg.norm(u2_3)))

        plan_hist1.append(_plan_xyz(np.asarray(active_plan["plan_x1"], dtype=float), active_plan_step))
        plan_hist2.append(_plan_xyz(np.asarray(active_plan["plan_x2"], dtype=float), active_plan_step))
        I = _identity_R()
        att_stub = [{"R": I, "phi": 0.0} for _ in range(T_horizon)]
        plan_att1.append(att_stub); plan_att2.append(att_stub)

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

        exec_att1.append({"R": I, "phi": 0.0}); phi_hist1.append(0.0)
        exec_att2.append({"R": I, "phi": 0.0}); phi_hist2.append(0.0)

        exec_xyz1.append(_p3(x1[:D], D))
        exec_xyz2.append(_p3(x2[:D], D))
        fov_axis_hist.append(None); fov_seen_mask.append(False)
        active_plan_step += 1

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
        steps = int(cfg.get("T_eval", cfg.get("T", cfg.get("steps", 60))))
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
