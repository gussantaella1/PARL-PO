# paper_baseline_runner.py
# Paper: Multiple to one orbital pursuit: https://ieeexplore.ieee.org/document/10700692


# ============================================================
# Baseline rollout runners for:
#   (1) "paper" objectives (as in the Chinese paper):
#         - paper_ne  : ARS best-response on (Je, Jpi)
#         - minmax    : zero-sum security (maximin) on a scalar g derived from the paper
#
#   (2) "ppo_zero_sum" objectives (match the current PPO env zero-sum g):
#         legacy_dock:
#           g = k_pos * d2 - k_dock * dock_gap - lD * ||uD||^2 + lA * ||uA||^2
#         progress_barrier:
#           g = k_pos * d2 - k_prog * safe_progress + k_close * unsafe_close
#               - lD * ||uD||^2 + lA * ||uA||^2
#         plus the same terminal events/order used in core/env.py
#
# Output dict keys are styled to match your RL rollout runners:
#   exec1_xyz/exec2_xyz/(exec3_xyz), plan_hist*, u_cmd_all, u_cmd_norm_all, etc.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


# ============================================================
# Paper parameter bundle
# ============================================================

@dataclass
class PaperParams:
    # Strategy-space discretization (Eq. 15 idea)
    Nup: int = 3                 # pursuer grid half-width
    Nue: int = 3                 # evader grid half-width
    dup: float = 1e-3            # pursuer accel grid unit
    due: float = 1e-3            # evader accel grid unit
    grid_mode: str = "cartesian" # "cartesian" = per-axis grid around u_prev

    # Prediction horizon (tau_k)
    tau: int = 5                 # number of sample steps to predict

    # Distance / guidance factors (Eqs. 17–20)
    beta1: float = 0.5
    beta2: float = 0.5
    alpha1: float = 2.5
    alpha2: float = 2.5
    alpha3: float = 2.5
    alpha4: float = 2.5
    alpha5: float = 2.5

    # Performance index weights (Eqs. 21–22), c1 + c2 + c3 = 1
    c1: float = 0.475
    c2: float = 0.475
    c3: float = 0.05

    # Pursuer safety constraint (Eq. 22): d_{pi,pj}(k+1) > dsafe
    dsafe: float = 1.0

    # ARS / best-response loop controls (Algorithm 1/2 spirit)
    max_ars_iter: int = 30
    ne_tol: float = 0.0          # 0.0 means exact match on discrete grid
    break_on_cycle: bool = True

    # ===== Added for your baseline needs =====
    # minmax "paper" security payoff choice:
    #   "je" (paper’s Je), "dmin_1", "dmin_tau"
    g_mode: str = "dmin_tau"

    # Optional effort tie-breakers in minmax mode
    w_uE: float = 0.0
    w_uP: float = 0.0

    # Minmax solver preference:
    #   "ars" (fast; works for 1v2),
    #   "bruteforce" (exact, but only supported for 1v1)
    minmax_solver: str = "ars"


def _paper_params_from_cfg(cfg: Dict[str, Any]) -> PaperParams:
    """
    Override defaults via cfg["paper"] dict.
    """
    p = PaperParams()
    d = cfg.get("paper", {}) or {}
    for k, v in d.items():
        if hasattr(p, k):
            setattr(p, k, v)

    # keep c1+c2+c3 sane
    s = float(p.c1 + p.c2 + p.c3)
    if abs(s - 1.0) > 1e-6 and s > 1e-12:
        p.c1 /= s
        p.c2 /= s
        p.c3 /= s

    p.grid_mode = (p.grid_mode or "cartesian").lower()
    p.g_mode = (p.g_mode or "dmin_tau").lower()
    p.minmax_solver = (p.minmax_solver or "ars").lower()
    return p


def _paper_control_limits_from_cfg(cfg: Dict[str, Any]) -> Tuple[float, float]:
    """
    Paper simulations use asymmetric acceleration limits:
      - evader   = 0.2 m/s^2
      - pursuers = 0.15 m/s^2

    We allow overrides through cfg["paper"] while keeping the current project
    behavior of falling back to the shared cfg["umax"] when no paper-specific
    limits are supplied.
    """
    paper_cfg = cfg.get("paper", {}) or {}
    umax_default = float(cfg.get("umax", 5e-4))
    umax_e = float(paper_cfg.get("umax_e", cfg.get("umax_e", umax_default)))
    umax_p = float(paper_cfg.get("umax_p", cfg.get("umax_p", umax_default)))
    return umax_e, umax_p


def _paper_parameter_check(
    cfg: Dict[str, Any],
    paper_params: PaperParams,
    *,
    turn_len: int,
) -> Dict[str, Any]:
    """
    Compare the active configuration against the paper's reported defaults.

    Verified from the attached paper tables / text:
      - Table I:  u_e,max = 0.2, u_p,max = 0.15
      - Table II: Nup = 3, Nue = 3, dup = 1e-3, due = 1e-3
      - Table III: alpha1..alpha5 = 2.5, c1 = c2 = 0.475, c3 = 0.05
      - Text: beta scenarios are (0.9, 0.1) or (0.5, 0.5), and c1+c2+c3 = 1
      - Text: control update period is 1 s
    """
    umax_e, umax_p = _paper_control_limits_from_cfg(cfg)
    dt = float(cfg.get("dt", 0.0))
    sample_period_s = float(turn_len) * dt

    checks: List[Dict[str, Any]] = []

    def add_check(name: str, actual: float | int, expected: float | int, tol: float = 1e-12):
        match = abs(float(actual) - float(expected)) <= tol
        checks.append(
            {
                "name": name,
                "actual": float(actual) if isinstance(actual, (int, float)) else actual,
                "expected": float(expected) if isinstance(expected, (int, float)) else expected,
                "match": bool(match),
            }
        )

    add_check("umax_e", umax_e, 0.2)
    add_check("umax_p", umax_p, 0.15)
    add_check("Nup", paper_params.Nup, 3, tol=0.0)
    add_check("Nue", paper_params.Nue, 3, tol=0.0)
    add_check("dup", paper_params.dup, 1e-3)
    add_check("due", paper_params.due, 1e-3)
    add_check("alpha1", paper_params.alpha1, 2.5)
    add_check("alpha2", paper_params.alpha2, 2.5)
    add_check("alpha3", paper_params.alpha3, 2.5)
    add_check("alpha4", paper_params.alpha4, 2.5)
    add_check("alpha5", paper_params.alpha5, 2.5)
    add_check("c1", paper_params.c1, 0.475)
    add_check("c2", paper_params.c2, 0.475)
    add_check("c3", paper_params.c3, 0.05)
    add_check("control_update_period_s", sample_period_s, 1.0)

    beta_pair = (round(float(paper_params.beta1), 6), round(float(paper_params.beta2), 6))
    beta_matches_scenario_1 = beta_pair == (0.9, 0.1)
    beta_matches_scenario_2 = beta_pair == (0.5, 0.5)

    notes = []
    if not (beta_matches_scenario_1 or beta_matches_scenario_2):
        notes.append(
            "beta1/beta2 do not match either reported paper scenario; "
            "the paper uses (0.9, 0.1) or (0.5, 0.5)."
        )
    if abs(float(paper_params.c1 + paper_params.c2 + paper_params.c3) - 1.0) > 1e-6:
        notes.append("c1 + c2 + c3 is not normalized to 1.")
    if abs(float(paper_params.beta1 + paper_params.beta2) - 1.0) > 1e-6:
        notes.append("beta1 + beta2 is not normalized to 1.")
    notes.append(
        "dsafe is part of the paper constraints, but its simulation value is not recoverable "
        "from the text layer of the attached PDF and is therefore reported as configurable."
    )

    mismatches = [c["name"] for c in checks if not c["match"]]
    return {
        "reference": {
            "umax_e": 0.2,
            "umax_p": 0.15,
            "Nup": 3,
            "Nue": 3,
            "dup": 1e-3,
            "due": 1e-3,
            "alpha1": 2.5,
            "alpha2": 2.5,
            "alpha3": 2.5,
            "alpha4": 2.5,
            "alpha5": 2.5,
            "c1": 0.475,
            "c2": 0.475,
            "c3": 0.05,
            "beta_scenarios": [(0.9, 0.1), (0.5, 0.5)],
            "control_update_period_s": 1.0,
        },
        "checks": checks,
        "beta_pair": beta_pair,
        "beta_matches_scenario_1": bool(beta_matches_scenario_1),
        "beta_matches_scenario_2": bool(beta_matches_scenario_2),
        "dsafe_configured": float(paper_params.dsafe),
        "matches_reference_defaults": len(mismatches) == 0 and (beta_matches_scenario_1 or beta_matches_scenario_2),
        "mismatches": mismatches,
        "notes": notes,
    }


def _pick_paper_game_mode(cfg: Dict[str, Any]) -> str:
    """
    Paper-only game mode:
      - "paper_ne" : ARS on (Je, Jpi)
      - "minmax"   : security strategy on scalar g derived from the paper
    """
    paper_cfg = cfg.get("paper", {}) or {}
    mode = (paper_cfg.get("game_mode", "paper_ne") or "paper_ne").lower()
    if mode not in ("paper_ne", "minmax"):
        mode = "paper_ne"
    return mode


# ============================================================
# Baseline objective mode switch
# ============================================================

def _baseline_objective_mode(cfg: Dict[str, Any]) -> str:
    """
    Outer objective switch:
      - "paper"          : replicate the paper’s pursuit-evasion objectives
      - "ppo_zero_sum"   : match the current PPO zero-sum env reward
    """
    d = cfg.get("paper_baseline", {}) or {}
    mode = (d.get("objective") or "paper").lower()
    if mode in ("ppo_oi_minmax", "zero_sum"):
        mode = "ppo_zero_sum"
    if mode not in ("paper", "ppo_zero_sum"):
        mode = "paper"
    return mode


# ============================================================
# PPO zero-sum objective bundle (for security / maximin)
# ============================================================

@dataclass
class PPOObjectiveParams:
    # Core zero-sum step reward used in core/env.py
    k_pos: float = 1.0
    k_dock: float = 0.0
    lD: float = 0.0
    lA: float = 0.0
    collision_radius_m: float = 0.0
    gamma: float = 1.0
    zero_sum_mode: str = "legacy_dock"
    progress_coef: float = 0.0
    safe_sep_m: float = 0.0
    unsafe_close_coef: float = 0.0
    unsafe_close_power: float = 2.0
    progress_gate_power: float = 1.0

    # Optional extras (off by default; not part of the default PPO g)
    include_step_walls: bool = False
    include_keepout: bool = False
    include_terminal: bool = True
    lookahead: int = 1
    objective_mode: str = "discounted_sum"

    # Step-shaping stabilizers / terminal params
    wallK: float = 10.0
    wall_penalty: float = 0.0
    collision_penalty: float = 0.0
    target_hit_reward_penalty: float = 0.0
    soft_wall: float = 0.7
    margin: float = 1.0

    oi_radius_m: float = 0.0
    def_keepout_buffer_m: float = 0.0
    def_center_avoid_coef: float = 0.0
    hit_buffer_def: float = 0.0
    hit_buffer_att: float = 0.0

    # Threat selection
    threat_mode: str = "idx0"   # "idx0" matches core/env.py


def _ppo_obj_params_from_cfg(cfg: Dict[str, Any]) -> PPOObjectiveParams:
    return PPOObjectiveParams()

def _populate_ppo_obj_from_cfg(cfg: Dict[str, Any], p: PPOObjectiveParams) -> PPOObjectiveParams:
    oi = cfg.get("oi", {}) or {}
    att_reward = cfg.get("att_reward", {}) or {}
    zero_sum = cfg.get("zero_sum_reward", {}) or {}
    p.oi_radius_m = float(oi.get("r", p.oi_radius_m))

    p.def_keepout_buffer_m = float(cfg.get("def_keepout_buffer_m", p.def_keepout_buffer_m))
    p.def_center_avoid_coef = float(cfg.get("def_center_avoid_coef", p.def_center_avoid_coef))

    p.wallK = float(cfg.get("wall_penalty", p.wallK))
    p.wall_penalty = float(cfg.get("wall_penalty", p.wall_penalty))
    p.collision_penalty = float(cfg.get("collision_penalty", p.collision_penalty))
    p.target_hit_reward_penalty = float(
        cfg.get("target_hit_reward_penalty", p.target_hit_reward_penalty)
    )
    p.soft_wall = float(cfg.get("soft_wall_start", p.soft_wall))
    p.margin = float(cfg.get("arena_terminate_margin", p.margin))

    # Match the current zero-sum env reward.
    p.k_pos = float(cfg.get("k_pos", cfg.get("step_pos_coef", p.k_pos)))
    p.k_dock = float(cfg.get("k_dock", p.k_dock))
    p.lD = float(cfg.get("effort_def", p.lD))
    p.lA = float(cfg.get("effort_att", p.lA))
    p.gamma = float(cfg.get("gamma", p.gamma))
    p.collision_radius_m = float(cfg.get("collision_radius_m", p.collision_radius_m))
    p.zero_sum_mode = str(zero_sum.get("mode", p.zero_sum_mode)).lower()
    p.progress_coef = float(zero_sum.get("progress_coef", att_reward.get("k_prog", p.progress_coef)))
    p.safe_sep_m = float(zero_sum.get("safe_sep_m", att_reward.get("min_sep", p.safe_sep_m)))
    p.unsafe_close_coef = float(
        zero_sum.get("unsafe_close_coef", att_reward.get("k_close", p.unsafe_close_coef))
    )
    p.unsafe_close_power = float(zero_sum.get("unsafe_close_power", p.unsafe_close_power))
    p.progress_gate_power = float(zero_sum.get("progress_gate_power", p.progress_gate_power))
    if p.zero_sum_mode not in ("legacy_dock", "progress_barrier"):
        raise ValueError(
            "zero_sum_reward.mode must be 'progress_barrier' or 'legacy_dock', "
            f"got {p.zero_sum_mode!r}"
        )
    p.hit_buffer_def = float(cfg.get("hit_buffer_def", p.hit_buffer_def))
    p.hit_buffer_att = float(cfg.get("hit_buffer_att", p.hit_buffer_att))

    pb = cfg.get("paper_baseline", {}) or {}
    obj = (pb.get("zero_sum", {}) or pb.get("ppo_obj", {}) or {})
    p.include_step_walls = bool(obj.get("include_step_walls", p.include_step_walls))
    p.include_keepout    = bool(obj.get("include_keepout", p.include_keepout))
    p.include_terminal   = bool(obj.get("include_terminal", p.include_terminal))
    p.lookahead          = int(obj.get("lookahead", p.lookahead))
    p.objective_mode     = (obj.get("objective_mode", p.objective_mode) or p.objective_mode).lower()
    p.threat_mode        = (obj.get("threat_mode", p.threat_mode) or p.threat_mode).lower()

    return p

# ============================================================
# Dynamics adapter (matches your RL runners)
# ============================================================

def _build_step_plant_single(cfg: Dict[str, Any], steps: int, D: int):
    """
    Returns (step_plant_single, center, dyn_type_str).
    Uses cfg["dynamics"] and cfg["dyn"] if present.
    Supports: "hcw", "elliptic_ltv", "two_body" (same pattern as your runner).
    """
    dt = float(cfg["dt"])
    dyn_name = (cfg.get("dynamics") or "hcw").lower()

    ar = cfg.setdefault("arena", {})
    ar.setdefault("type", "sphere")
    ar.setdefault("cx", 0.0)
    ar.setdefault("cy", 0.0)
    ar.setdefault("cz", 0.0)
    ar.setdefault("r", 30.0)

    center = np.array(
        [ar["cx"], ar["cy"], (ar.get("cz", 0.0) if D == 3 else 0.0)],
        dtype=np.float32
    )[:D]

    dyn = cfg.get("dyn", {}) if isinstance(cfg.get("dyn", {}), dict) else {}
    Ad = dyn.get("Ad", None)
    Bd = dyn.get("Bd", None)
    Ad_seq = dyn.get("Ad_seq", None)
    Bd_seq = dyn.get("Bd_seq", None)
    chief_cache = dyn.get("chief_cache", None)

    dyn_type = None

    def _u3(uD: np.ndarray):
        uD = np.asarray(uD, float).reshape(-1)
        if D == 3:
            return uD[:3]
        return np.array([uD[0], uD[1], 0.0], dtype=float)

    def _x6_from_xD(xD: np.ndarray) -> np.ndarray:
        xD = np.asarray(xD, dtype=np.float32).reshape(-1)
        if D == 3:
            return xD.astype(np.float32)
        # D==2: [x,y,vx,vy] -> [x,y,0,vx,vy,0]
        return np.array([xD[0], xD[1], 0.0, xD[2], xD[3], 0.0], dtype=np.float32)

    def _xD_from_x6(x6: np.ndarray) -> np.ndarray:
        x6 = np.asarray(x6, dtype=np.float32).reshape(-1)
        if D == 3:
            return x6.astype(np.float32)
        return np.array([x6[0], x6[1], x6[3], x6[4]], dtype=np.float32)

    if dyn_name == "hcw":
        if Ad is None or Bd is None:
            from dyn_models import hcw_mean_motion, hcw_discrete_mats, as_numpy_const
            n = hcw_mean_motion(cfg.get("hcw", {}))
            Ad_mx, Bd_mx = hcw_discrete_mats(n, dt)
            Ad = as_numpy_const(Ad_mx).astype(np.float32)
            Bd = as_numpy_const(Bd_mx).astype(np.float32)
        dyn_type = "lti"

    elif dyn_name in ("elliptic_ltv", "elliptical_ltv", "th", "tschauner_hempel"):
        if Ad_seq is None or Bd_seq is None or chief_cache is None:
            from dyn_models import chief_orbit_cache_rtn, linearize_two_body_rtn_discrete
            orb = cfg.get("chief_orbit", {})
            chief_cache = chief_orbit_cache_rtn(orb, dt=dt, N=steps)
            Ad_seq, Bd_seq = linearize_two_body_rtn_discrete(chief_cache, dt=dt, eps=1e-5)
            Ad_seq = Ad_seq.astype(np.float32)
            Bd_seq = Bd_seq.astype(np.float32)
        dyn_type = "ltv"

    elif dyn_name in ("two_body", "twobody", "two-body"):
        if chief_cache is None:
            from dyn_models import chief_orbit_cache_rtn
            orb = cfg.get("chief_orbit", {})
            chief_cache = chief_orbit_cache_rtn(orb, dt=dt, N=steps)
        dyn_type = "nonlinear"

    else:
        raise ValueError(f"Unknown dynamics '{cfg.get('dynamics')}'")

    def step_plant_single(xD: np.ndarray, uD: np.ndarray, k: int) -> np.ndarray:
        x6 = _x6_from_xD(xD)
        u3 = _u3(uD)

        if dyn_type == "lti":
            x6n = (Ad @ x6 + Bd @ u3).astype(np.float32)
        elif dyn_type == "ltv":
            Ak = Ad_seq[k]
            Bk = Bd_seq[k]
            x6n = (Ak @ x6 + Bk @ u3).astype(np.float32)
        elif dyn_type == "nonlinear":
            from dyn_models import two_body_step_rtn
            x6n = two_body_step_rtn(x6, u3, k, chief_cache).astype(np.float32)
        else:
            raise RuntimeError(f"dyn_type='{dyn_type}' not recognized")

        return _xD_from_x6(x6n)

    return step_plant_single, center, dyn_type


# ============================================================
# Strategy set (Eq. 15-style neighborhood)
# ============================================================

def _action_grid(u_prev: np.ndarray, du: float, N: int, umax: float, grid_mode: str) -> List[np.ndarray]:
    """
    grid_mode="cartesian": all per-axis combinations u_prev + [i,j,k]*du, i,j,k in [-N..N].
    Returns list of D-vectors clipped to [-umax, +umax].

    Practical note:
      If du is tiny relative to umax, the solver can get "stuck" near zero.
      A reasonable starting point is:
          du ~ 0.25 * umax  and N ~ 2..4
    """
    u_prev = np.asarray(u_prev, float).reshape(-1)
    D = u_prev.size

    if N <= 0 or du <= 0.0:
        return [np.clip(u_prev, -umax, +umax).astype(np.float32)]

    if (grid_mode or "cartesian").lower() != "cartesian":
        grid_mode = "cartesian"

    vals = np.arange(-N, N + 1, dtype=int) * float(du)

    out: List[np.ndarray] = []
    if D == 1:
        for dx in vals:
            u = u_prev + np.array([dx], float)
            out.append(np.clip(u, -umax, +umax).astype(np.float32))
        return out

    if D == 2:
        for dx in vals:
            for dy in vals:
                u = u_prev + np.array([dx, dy], float)
                out.append(np.clip(u, -umax, +umax).astype(np.float32))
        return out

    # D >= 3
    for dx in vals:
        for dy in vals:
            for dz in vals:
                inc = np.array([dx, dy, dz], float)
                if D > 3:
                    inc = np.pad(inc, (0, D - 3), constant_values=0.0)
                u = u_prev + inc
                out.append(np.clip(u, -umax, +umax).astype(np.float32))
    return out


# ============================================================
# Helpers: distances and prediction
# ============================================================

def _dist_metrics(pE: np.ndarray, pP: List[np.ndarray]) -> Tuple[float, float]:
    d_list = [float(np.linalg.norm(pi - pE)) for pi in pP]
    return float(min(d_list)), float(sum(d_list))


def _predict_positions(
    xE: np.ndarray,
    xP: List[np.ndarray],
    uE: np.ndarray,
    uP: List[np.ndarray],
    k0: int,
    n_steps: int,
    step_plant_single,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Predict forward n_steps using constant controls uE, uP (repeated each step).
    Returns (xE_pred, xP_pred_list).
    """
    xE_cur = np.asarray(xE, np.float32).copy()
    xP_cur = [np.asarray(x, np.float32).copy() for x in xP]

    for t in range(n_steps):
        kk = k0 + t
        xE_cur = step_plant_single(xE_cur, uE, kk)
        for i in range(len(xP_cur)):
            xP_cur[i] = step_plant_single(xP_cur[i], uP[i], kk)

    return xE_cur, xP_cur


# ============================================================
# Paper payoff computation (Eqs. 17–22)
# ============================================================

def _paper_payoffs(
    xE: np.ndarray,
    xP: List[np.ndarray],
    uE: np.ndarray,
    uP: List[np.ndarray],
    k: int,
    step_plant_single,
    params: PaperParams,
    umax_e: float,
    umax_p: float,
) -> Tuple[float, List[float]]:
    """
    Returns (J_e, [J_p1, ..., J_pn]).
      - Evader maximizes J_e.
      - Pursuers minimize J_pi independently (non-cooperative between pursuers).

    This matches the structure you implemented from the paper:
      - F1 one-step factor
      - F2 tau-step factor
      - F3 energy
      - safety constraint between pursuers (dsafe)
    """
    D = int(xE.size // 2)
    eps = 1e-12

    pE_k = np.asarray(xE[:D], float)
    pP_k = [np.asarray(x[:D], float) for x in xP]

    dmin_k, dsum_k = _dist_metrics(pE_k, pP_k)
    dmin_k = max(dmin_k, eps)

    # 1-step
    xE_1, xP_1 = _predict_positions(xE, xP, uE, uP, k0=k, n_steps=1, step_plant_single=step_plant_single)
    pE_1 = np.asarray(xE_1[:D], float)
    pP_1 = [np.asarray(x[:D], float) for x in xP_1]
    dmin_1, dsum_1 = _dist_metrics(pE_1, pP_1)

    # tau-step
    tau = max(1, int(params.tau))
    xE_T, xP_T = _predict_positions(xE, xP, uE, uP, k0=k, n_steps=tau, step_plant_single=step_plant_single)
    pE_T = np.asarray(xE_T[:D], float)
    pP_T = [np.asarray(x[:D], float) for x in xP_T]
    dmin_T, dsum_T = _dist_metrics(pE_T, pP_T)

    # F1
    term_min_1 = np.arctan(params.alpha1 * ((dmin_k - dmin_1) / dmin_k))
    term_sum_1 = np.arctan(params.alpha2 * ((dsum_k - dsum_1) / dmin_k))
    F1 = (2.0 / np.pi) * (params.beta1 * term_min_1 + params.beta2 * term_sum_1)

    # F2
    F2_p = (2.0 / np.pi) * np.arctan(params.alpha3 * ((dmin_k - dmin_T) / dmin_k))
    term_min_2e = np.arctan(params.alpha4 * ((dmin_k - dmin_T) / dmin_k))
    term_sum_2e = np.arctan(params.alpha5 * ((dsum_k - dsum_T) / dmin_k))
    F2_e = (2.0 / np.pi) * (params.beta1 * term_min_2e + params.beta2 * term_sum_2e)

    # F3
    F3_e = float(np.linalg.norm(uE) / max(umax_e, eps))
    F3_p_each = [float(np.linalg.norm(ui) / max(umax_p, eps)) for ui in uP]

    # Evader payoff (maximize)
    if np.linalg.norm(uE) <= umax_e + 1e-12:
        J_e = float(-params.c1 * F1 - params.c2 * F2_e - params.c3 * F3_e)
    else:
        J_e = -10.0

    # Pursuer payoffs (minimize), safety constraint at k+1
    J_p: List[float] = []
    for i in range(len(uP)):
        ok_u = (np.linalg.norm(uP[i]) <= umax_p + 1e-12)

        ok_safe = True
        if len(uP) > 1:
            for j in range(len(uP)):
                if j == i:
                    continue
                dij = float(np.linalg.norm(pP_1[i] - pP_1[j]))
                if dij <= float(params.dsafe):
                    ok_safe = False
                    break

        if ok_u and ok_safe:
            Ji = float(-params.c1 * F1 - params.c2 * F2_p + params.c3 * F3_p_each[i])
        else:
            Ji = 10.0
        J_p.append(Ji)

    return J_e, J_p


# ============================================================
# Paper-style "NE" via ARS best-response iterations
# ============================================================

def _solve_one_step_ne_ars(
    xE: np.ndarray,
    xP: List[np.ndarray],
    uE_prev: np.ndarray,
    uP_prev: List[np.ndarray],
    k: int,
    step_plant_single,
    params: PaperParams,
    umax_e: float,
    umax_p: float,
) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, Any]]:
    """
    Discrete action sets around previous controls; Gauss-Seidel best responses:
      pursuers: minimize J_pi independently
      evader:   maximize J_e

    Returns (uE*, [uP*], dbg)
    """
    nP = len(uP_prev)
    uE_prev = np.asarray(uE_prev, float).reshape(-1)
    uP_prev = [np.asarray(u, float).reshape(-1) for u in uP_prev]

    Ue = _action_grid(uE_prev, params.due, params.Nue, umax_e, params.grid_mode)
    Up = [_action_grid(uP_prev[i], params.dup, params.Nup, umax_p, params.grid_mode) for i in range(nP)]

    def _nearest_idx(U: List[np.ndarray], u: np.ndarray) -> int:
        d = [float(np.linalg.norm(ui - u)) for ui in U]
        return int(np.argmin(d))

    idx_e = _nearest_idx(Ue, uE_prev)
    idx_p = [_nearest_idx(Up[i], uP_prev[i]) for i in range(nP)]

    seen = set()

    def _profile_key(ie: int, ip: List[int]) -> Tuple[int, Tuple[int, ...]]:
        return (ie, tuple(int(x) for x in ip))

    def _best_resp_p(i: int, ie: int, ip: List[int]) -> int:
        best_j = None
        best_idx = ip[i]
        for a_idx, ui in enumerate(Up[i]):
            uE = Ue[ie]
            uP = [Up[j][ip[j]] for j in range(nP)]
            uP[i] = ui
            _, Jp = _paper_payoffs(xE, xP, uE, uP, k, step_plant_single, params, umax_e, umax_p)
            Ji = Jp[i]
            if (best_j is None) or (Ji < best_j - params.ne_tol):
                best_j = Ji
                best_idx = a_idx
        return best_idx

    def _best_resp_e(ie: int, ip: List[int]) -> int:
        best_j = None
        best_idx = ie
        for a_idx, ue in enumerate(Ue):
            uP = [Up[j][ip[j]] for j in range(nP)]
            Je, _ = _paper_payoffs(xE, xP, ue, uP, k, step_plant_single, params, umax_e, umax_p)
            if (best_j is None) or (Je > best_j + params.ne_tol):
                best_j = Je
                best_idx = a_idx
        return best_idx

    it_used = 0
    for it in range(params.max_ars_iter):
        it_used = it + 1
        key = _profile_key(idx_e, idx_p)
        if params.break_on_cycle:
            if key in seen:
                break
            seen.add(key)

        changed = False
        for i in range(nP):
            new_i = _best_resp_p(i, idx_e, idx_p)
            if new_i != idx_p[i]:
                idx_p[i] = new_i
                changed = True

        new_e = _best_resp_e(idx_e, idx_p)
        if new_e != idx_e:
            idx_e = new_e
            changed = True

        if not changed:
            break

    uE_star = np.asarray(Ue[idx_e], np.float32)
    uP_star = [np.asarray(Up[i][idx_p[i]], np.float32) for i in range(nP)]
    dbg = {
        "objective": "paper",
        "paper_game_mode": "paper_ne",
        "ars_iters": int(it_used),
        "idx_e": int(idx_e),
        "idx_p": [int(x) for x in idx_p],
        "n_actions_e": int(len(Ue)),
        "n_actions_p": [int(len(Up[i])) for i in range(nP)],
    }
    return uE_star, uP_star, dbg


# ============================================================
# Paper-derived security payoff g + solver (zero-sum)
# ============================================================

def _paper_security_g(
    xE: np.ndarray,
    xP: List[np.ndarray],
    uE: np.ndarray,
    uP: List[np.ndarray],
    k: int,
    step_plant_single,
    params: PaperParams,
    umax_e: float,
    umax_p: float,
) -> float:
    """
    Scalar zero-sum payoff g:
      - evader MAXIMIZES g
      - pursuer team MINIMIZES g

    params.g_mode:
      - "je"       : g = Je (paper)
      - "dmin_1"   : g = min_i ||pPi(k+1) - pE(k+1)||
      - "dmin_tau" : g = min_i ||pPi(k+tau) - pE(k+tau)||

    Tie-breakers:
      g -= w_uE * ||uE||/umax_e
      g += w_uP * mean_i ||uPi||/umax_p
    """
    D = int(xE.size // 2)
    eps = 1e-12

    # hard feasibility
    if np.linalg.norm(uE) > umax_e + 1e-12:
        return -1e9
    for ui in uP:
        if np.linalg.norm(ui) > umax_p + 1e-12:
            return +1e9

    # paper safety constraint at k+1
    if len(uP) > 1 and float(params.dsafe) > 0.0:
        _, xP_1 = _predict_positions(xE, xP, uE, uP, k0=k, n_steps=1, step_plant_single=step_plant_single)
        pP_1 = [np.asarray(x[:D], float) for x in xP_1]
        for i in range(len(pP_1)):
            for j in range(i + 1, len(pP_1)):
                if float(np.linalg.norm(pP_1[i] - pP_1[j])) <= float(params.dsafe):
                    return +1e9

    gm = (params.g_mode or "dmin_tau").lower()
    if gm == "je":
        Je, _ = _paper_payoffs(xE, xP, uE, uP, k, step_plant_single, params, umax_e, umax_p)
        g = float(Je)
    elif gm == "dmin_1":
        xE_1, xP_1 = _predict_positions(xE, xP, uE, uP, k0=k, n_steps=1, step_plant_single=step_plant_single)
        pE_1 = np.asarray(xE_1[:D], float)
        pP_1 = [np.asarray(x[:D], float) for x in xP_1]
        dmin_1, _ = _dist_metrics(pE_1, pP_1)
        g = float(dmin_1)
    else:
        tau = max(1, int(params.tau))
        xE_T, xP_T = _predict_positions(xE, xP, uE, uP, k0=k, n_steps=tau, step_plant_single=step_plant_single)
        pE_T = np.asarray(xE_T[:D], float)
        pP_T = [np.asarray(x[:D], float) for x in xP_T]
        dmin_T, _ = _dist_metrics(pE_T, pP_T)
        g = float(dmin_T)

    # effort tie-breakers
    if float(params.w_uE) != 0.0:
        g -= float(params.w_uE) * float(np.linalg.norm(uE) / max(umax_e, eps))
    if float(params.w_uP) != 0.0 and len(uP) > 0:
        g += float(params.w_uP) * float(np.mean([np.linalg.norm(ui) / max(umax_p, eps) for ui in uP]))

    return float(g)


def _solve_one_step_minimax_security(
    xE: np.ndarray,
    xP: List[np.ndarray],
    uE_prev: np.ndarray,
    uP_prev: List[np.ndarray],
    k: int,
    step_plant_single,
    params: PaperParams,
    umax_e: float,
    umax_p: float,
) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, Any]]:
    """
    Discrete security (maximin):

      uE* = argmax_{uE in Ue}  min_{uP team in Up} g(x,uE,uP)

    - 1v1 supports exact bruteforce over grids (if params.minmax_solver=="bruteforce")
    - 1v2 uses alternating best responses / coordinate descent (fast)
    """
    nP = len(uP_prev)
    uE_prev = np.asarray(uE_prev, float).reshape(-1)
    uP_prev = [np.asarray(u, float).reshape(-1) for u in uP_prev]

    Ue = _action_grid(uE_prev, params.due, params.Nue, umax_e, params.grid_mode)
    Up = [_action_grid(uP_prev[i], params.dup, params.Nup, umax_p, params.grid_mode) for i in range(nP)]

    def _nearest_idx(U: List[np.ndarray], u: np.ndarray) -> int:
        d = [float(np.linalg.norm(ui - u)) for ui in U]
        return int(np.argmin(d))

    idx_e = _nearest_idx(Ue, uE_prev)
    idx_p = [_nearest_idx(Up[i], uP_prev[i]) for i in range(nP)]

    solver = (params.minmax_solver or "ars").lower()
    if solver == "bruteforce" and nP != 1:
        solver = "ars"

    # ----- exact bruteforce (1v1 only) -----
    if solver == "bruteforce":
        best_val = -1e18
        best_ie = idx_e
        best_ip = idx_p[0]

        for ie, uE in enumerate(Ue):
            min_val = +1e18
            min_ip = 0
            for ip, uP0 in enumerate(Up[0]):
                g = _paper_security_g(
                    xE, xP, uE, [uP0], k,
                    step_plant_single, params, umax_e, umax_p
                )
                if g < min_val:
                    min_val = g
                    min_ip = ip
            if min_val > best_val:
                best_val = min_val
                best_ie = ie
                best_ip = min_ip

        uE_star = np.asarray(Ue[best_ie], np.float32)
        uP_star = [np.asarray(Up[0][best_ip], np.float32)]
        dbg = {
            "objective": "paper",
            "paper_game_mode": "minmax",
            "solver": "bruteforce",
            "ars_iters": 0,
            "g_mode": params.g_mode,
            "g_star": float(best_val),
            "idx_e": int(best_ie),
            "idx_p": [int(best_ip)],
            "n_actions_e": int(len(Ue)),
            "n_actions_p": [int(len(Up[0]))],
        }
        return uE_star, uP_star, dbg

    # ----- alternating best responses (fast; works for 1v2) -----
    seen = set()

    def _profile_key(ie: int, ip: List[int]) -> Tuple[int, Tuple[int, ...]]:
        return (ie, tuple(int(x) for x in ip))

    def _g_for_indices(ie: int, ip: List[int]) -> float:
        uE = Ue[ie]
        uP = [Up[i][ip[i]] for i in range(nP)]
        return _paper_security_g(xE, xP, uE, uP, k, step_plant_single, params, umax_e, umax_p)

    def _best_resp_p(i: int, ie: int, ip: List[int]) -> int:
        best_g = None
        best_idx = ip[i]
        for a_idx, ui in enumerate(Up[i]):
            uP = [Up[j][ip[j]] for j in range(nP)]
            uP[i] = ui
            g = _paper_security_g(xE, xP, Ue[ie], uP, k, step_plant_single, params, umax_e, umax_p)
            if (best_g is None) or (g < best_g - params.ne_tol):
                best_g = g
                best_idx = a_idx
        return best_idx

    def _best_resp_e(ie: int, ip: List[int]) -> int:
        best_g = None
        best_idx = ie
        uP = [Up[j][ip[j]] for j in range(nP)]
        for a_idx, ue in enumerate(Ue):
            g = _paper_security_g(xE, xP, ue, uP, k, step_plant_single, params, umax_e, umax_p)
            if (best_g is None) or (g > best_g + params.ne_tol):
                best_g = g
                best_idx = a_idx
        return best_idx

    it_used = 0
    for it in range(params.max_ars_iter):
        it_used = it + 1
        key = _profile_key(idx_e, idx_p)
        if params.break_on_cycle:
            if key in seen:
                break
            seen.add(key)

        changed = False
        for i in range(nP):
            new_i = _best_resp_p(i, idx_e, idx_p)
            if new_i != idx_p[i]:
                idx_p[i] = new_i
                changed = True

        new_e = _best_resp_e(idx_e, idx_p)
        if new_e != idx_e:
            idx_e = new_e
            changed = True

        if not changed:
            break

    uE_star = np.asarray(Ue[idx_e], np.float32)
    uP_star = [np.asarray(Up[i][idx_p[i]], np.float32) for i in range(nP)]
    g_star = float(_g_for_indices(idx_e, idx_p))

    dbg = {
        "objective": "paper",
        "paper_game_mode": "minmax",
        "solver": "ars",
        "ars_iters": int(it_used),
        "g_mode": params.g_mode,
        "g_star": float(g_star),
        "idx_e": int(idx_e),
        "idx_p": [int(x) for x in idx_p],
        "n_actions_e": int(len(Ue)),
        "n_actions_p": [int(len(Up[i])) for i in range(nP)],
    }
    return uE_star, uP_star, dbg


def _solve_controls_paper(
    cfg: Dict[str, Any],
    *,
    xE: np.ndarray,
    xP: List[np.ndarray],
    uE_prev: np.ndarray,
    uP_prev: List[np.ndarray],
    k: int,
    step_plant_single,
    paper_params: PaperParams,
    umax_e: float,
    umax_p: float,
) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, Any]]:
    """
    Paper objective dispatch:
      - paper_ne
      - minmax (security) on paper-derived scalar g
    """
    mode = _pick_paper_game_mode(cfg)
    if mode == "minmax":
        return _solve_one_step_minimax_security(
            xE=xE, xP=xP,
            uE_prev=uE_prev, uP_prev=uP_prev,
            k=k, step_plant_single=step_plant_single,
            params=paper_params, umax_e=umax_e, umax_p=umax_p
        )
    return _solve_one_step_ne_ars(
        xE=xE, xP=xP,
        uE_prev=uE_prev, uP_prev=uP_prev,
        k=k, step_plant_single=step_plant_single,
        params=paper_params, umax_e=umax_e, umax_p=umax_p
    )


# ============================================================
# PPO-like OI security game (defender max, attackers min)
# ============================================================

def _wall_penalty(p: np.ndarray, center: np.ndarray, R: float, soft_wall: float, wallK: float) -> float:
    rho = float(np.linalg.norm(p - center) / (R + 1e-12))
    gap = max(0.0, rho - float(soft_wall))
    return float((gap * gap) * wallK)


def _def_keepout_penalty(p_def: np.ndarray, center: np.ndarray, oi_r: float, buf: float, coef: float) -> float:
    if oi_r <= 0.0 or coef <= 0.0:
        return 0.0
    d = float(np.linalg.norm(p_def - center))
    r_keep = float(oi_r + buf)
    if d >= r_keep:
        return 0.0
    gap = r_keep - d
    return float(coef * (gap * gap))


def _pick_threat_attacker(pA_list: List[np.ndarray], center: np.ndarray, mode: str) -> int:
    mode = (mode or "idx0").lower()
    if mode in ("idx0", "primary", "first"):
        return 0
    d2 = [float(np.dot(p - center, p - center)) for p in pA_list]
    return int(np.argmin(d2))


def _ppo_terminal_adjustment(
    pD: np.ndarray,
    pA_list: List[np.ndarray],
    center: np.ndarray,
    R: float,
    params: PPOObjectiveParams,
    primary_idx: int,
) -> Tuple[float, Dict[str, Any]]:
    eps = 1e-12
    oi_r_norm = float(params.oi_radius_m) / (float(R) + eps) if R > 0.0 else 0.0

    rho_def = float(np.linalg.norm(pD - center) / (float(R) + eps))
    rho_att = float(np.linalg.norm(pA_list[primary_idx] - center) / (float(R) + eps))

    thresh_def = (1.0 + float(params.hit_buffer_def)) * oi_r_norm
    thresh_att = (1.0 + float(params.hit_buffer_att)) * oi_r_norm

    att_hit_target = (oi_r_norm > 0.0) and (rho_att <= thresh_att)
    def_hit_target = (oi_r_norm > 0.0) and (rho_def <= thresh_def)
    hit_target = bool(att_hit_target or def_hit_target)

    collision = False
    if float(params.collision_radius_m) > 0.0:
        for pA in pA_list:
            if float(np.linalg.norm(pA - pD)) <= float(params.collision_radius_m):
                collision = True
                break

    oob_def = bool(rho_def >= float(params.margin))
    oob_att = any(
        float(np.linalg.norm(pA - center) / (float(R) + eps)) >= float(params.margin)
        for pA in pA_list
    )

    done = bool(collision or hit_target or oob_def or oob_att)

    g_term = 0.0
    if done:
        if collision:
            g_term += float(params.collision_penalty)
        elif att_hit_target or def_hit_target:
            g_term -= float(params.target_hit_reward_penalty)
        elif oob_def:
            g_term -= float(params.wall_penalty)
        elif oob_att:
            g_term += float(params.wall_penalty)

    info = {
        "done": done,
        "collision": bool(collision),
        "att_hit_target": bool(att_hit_target),
        "def_hit_target": bool(def_hit_target),
        "oob_def": bool(oob_def),
        "oob_att": bool(oob_att),
        "terminal_g": float(g_term),
        "primary_idx": int(primary_idx),
    }
    return float(g_term), info


def _ppo_security_value_g(
    xD: np.ndarray,
    xA_list: List[np.ndarray],
    uD: np.ndarray,
    uA_list: List[np.ndarray],
    k: int,
    step_plant_single,
    center: np.ndarray,
    R: float,
    params: PPOObjectiveParams,
    umax: float,
) -> float:
    D = int(xD.size // 2)
    eps = 1e-12

    def _ppo_step_reward_g(
        pD_prev: np.ndarray,
        pA_prev: List[np.ndarray],
        pD: np.ndarray,
        pA: List[np.ndarray],
    ) -> float:
        ith = _pick_threat_attacker(pA, center, params.threat_mode)
        d2_prev = float(np.dot(pA_prev[ith] - center, pA_prev[ith] - center)) / (R * R + eps)
        d2 = float(np.dot(pA[ith] - center, pA[ith] - center)) / (R * R + eps)
        dist_rel = float(np.linalg.norm(pA[ith] - pD))

        uD_loc = np.asarray(uD, float).reshape(-1)
        umax2 = float(umax) * float(umax) + eps
        aD_n2 = float(np.dot(uD_loc, uD_loc)) / umax2
        aA_n2 = float(
            np.mean([
                float(np.dot(np.asarray(uA, float).reshape(-1), np.asarray(uA, float).reshape(-1))) / umax2
                for uA in uA_list
            ])
        )

        if params.zero_sum_mode == "legacy_dock":
            dock_gap = max(0.0, dist_rel - float(params.collision_radius_m)) / (float(R) + eps)
            g_step = (
                float(params.k_pos) * d2
                - float(params.k_dock) * dock_gap
                - float(params.lD) * aD_n2
                + float(params.lA) * aA_n2
            )
        else:
            attacker_progress = max(0.0, d2_prev - d2)
            safe_sep = max(float(params.safe_sep_m), float(params.collision_radius_m) + 1e-6)
            unsafe_close = max(0.0, safe_sep - dist_rel) / (safe_sep + eps)
            gate_denom = max(safe_sep - float(params.collision_radius_m), 1e-6)
            progress_gate = np.clip(
                (dist_rel - float(params.collision_radius_m)) / gate_denom,
                0.0,
                1.0,
            )
            progress_gate = progress_gate ** float(params.progress_gate_power)

            g_step = (
                float(params.k_pos) * d2
                - float(params.progress_coef) * progress_gate * attacker_progress
                + float(params.unsafe_close_coef) * (unsafe_close ** float(params.unsafe_close_power))
                - float(params.lD) * aD_n2
                + float(params.lA) * aA_n2
            )

        if params.include_step_walls:
            g_step -= _wall_penalty(pD, center, R, params.soft_wall, params.wallK)

        if params.include_keepout:
            g_step -= _def_keepout_penalty(
                p_def=pD,
                center=center,
                oi_r=params.oi_radius_m,
                buf=params.def_keepout_buffer_m,
                coef=params.def_center_avoid_coef,
            )

        return float(g_step)

    lookahead = max(1, int(params.lookahead))
    objective_mode = (params.objective_mode or "discounted_sum").lower()

    if objective_mode == "terminal_state":
        xD_pred, xA_pred = _predict_positions(
            xE=xD, xP=xA_list,
            uE=uD, uP=uA_list,
            k0=k, n_steps=lookahead,
            step_plant_single=step_plant_single,
        )
        pD_prev = np.asarray(xD[:D], float)
        pA_prev = [np.asarray(x[:D], float) for x in xA_list]
        pD = np.asarray(xD_pred[:D], float)
        pA = [np.asarray(x[:D], float) for x in xA_pred]
        g = _ppo_step_reward_g(pD_prev, pA_prev, pD, pA)
        if params.include_terminal:
            g_term, _ = _ppo_terminal_adjustment(
                pD=pD,
                pA_list=pA,
                center=center,
                R=R,
                params=params,
                primary_idx=0,
            )
            g += g_term
        return float(g)

    gamma = float(params.gamma)
    xD_cur = np.asarray(xD, np.float32).copy()
    xA_cur = [np.asarray(x, np.float32).copy() for x in xA_list]
    total_g = 0.0
    disc = 1.0

    for t in range(lookahead):
        kk = k + t
        pD_prev = np.asarray(xD_cur[:D], float)
        pA_prev = [np.asarray(x[:D], float) for x in xA_cur]
        xD_cur = step_plant_single(xD_cur, uD, kk)
        for i in range(len(xA_cur)):
            xA_cur[i] = step_plant_single(xA_cur[i], uA_list[i], kk)

        pD = np.asarray(xD_cur[:D], float)
        pA = [np.asarray(x[:D], float) for x in xA_cur]

        g_step = _ppo_step_reward_g(pD_prev, pA_prev, pD, pA)
        g_term, term_dbg = _ppo_terminal_adjustment(
            pD=pD,
            pA_list=pA,
            center=center,
            R=R,
            params=params,
            primary_idx=0,
        )
        if params.include_terminal:
            g_step += g_term

        total_g += disc * float(g_step)
        if term_dbg["done"]:
            break
        disc *= gamma

    return float(total_g)


def _solve_one_step_minmax_team_ppo_oi(
    *,
    xD: np.ndarray,
    xA_list: List[np.ndarray],
    uD_prev: np.ndarray,
    uA_prev: List[np.ndarray],
    k: int,
    step_plant_single,
    center: np.ndarray,
    R: float,
    paper_params: PaperParams,
    ppo_params: PPOObjectiveParams,
    umax: float,
) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, Any]]:
    """
    Discrete neighborhood around previous controls (Eq. 15 style),
    solve TEAM minmax:

      uD*  = argmax_{uD in Ud}  min_{uA_i in Ua_i} g(uD, uA_1..uA_n)

    Attackers coordinate-descent to minimize g; defender best-responds to maximize g.
    """
    nA = len(uA_prev)
    uD_prev = np.asarray(uD_prev, float).reshape(-1)
    uA_prev = [np.asarray(u, float).reshape(-1) for u in uA_prev]

    Ud = _action_grid(uD_prev, paper_params.due, paper_params.Nue, umax, paper_params.grid_mode)
    Ua = [_action_grid(uA_prev[i], paper_params.dup, paper_params.Nup, umax, paper_params.grid_mode) for i in range(nA)]

    def _nearest_idx(U, u):
        d = [float(np.linalg.norm(ui - u)) for ui in U]
        return int(np.argmin(d))

    idx_d = _nearest_idx(Ud, uD_prev)
    idx_a = [_nearest_idx(Ua[i], uA_prev[i]) for i in range(nA)]

    seen = set()

    def _key(id_d: int, id_a: List[int]) -> Tuple[int, Tuple[int, ...]]:
        return (int(id_d), tuple(int(x) for x in id_a))

    def _best_resp_att(i: int, id_d: int, id_a: List[int]) -> int:
        best = None
        best_idx = id_a[i]
        for a_idx, ui in enumerate(Ua[i]):
            uD = Ud[id_d]
            uA = [Ua[j][id_a[j]] for j in range(nA)]
            uA[i] = ui
            g = _ppo_security_value_g(
                xD=xD, xA_list=xA_list,
                uD=uD, uA_list=uA,
                k=k, step_plant_single=step_plant_single,
                center=center, R=R, params=ppo_params, umax=umax
            )
            if (best is None) or (g < best - paper_params.ne_tol):
                best = g
                best_idx = a_idx
        return best_idx

    def _best_resp_def(id_d: int, id_a: List[int]) -> int:
        best = None
        best_idx = id_d
        uA = [Ua[j][id_a[j]] for j in range(nA)]
        for d_idx, uD in enumerate(Ud):
            g = _ppo_security_value_g(
                xD=xD, xA_list=xA_list,
                uD=uD, uA_list=uA,
                k=k, step_plant_single=step_plant_single,
                center=center, R=R, params=ppo_params, umax=umax
            )
            if (best is None) or (g > best + paper_params.ne_tol):
                best = g
                best_idx = d_idx
        return best_idx

    it_used = 0
    for it in range(paper_params.max_ars_iter):
        it_used = it + 1
        kk = _key(idx_d, idx_a)
        if paper_params.break_on_cycle:
            if kk in seen:
                break
            seen.add(kk)

        changed = False

        # attackers coordinate updates (minimize g)
        for i in range(nA):
            new_i = _best_resp_att(i, idx_d, idx_a)
            if new_i != idx_a[i]:
                idx_a[i] = new_i
                changed = True

        # defender best response (maximize g)
        new_d = _best_resp_def(idx_d, idx_a)
        if new_d != idx_d:
            idx_d = new_d
            changed = True

        if not changed:
            break

    uD_star = np.asarray(Ud[idx_d], np.float32)
    uA_star = [np.asarray(Ua[i][idx_a[i]], np.float32) for i in range(nA)]

    # compute g_star for debug
    g_star = _ppo_security_value_g(
        xD=xD, xA_list=xA_list,
        uD=uD_star, uA_list=uA_star,
        k=k, step_plant_single=step_plant_single,
        center=center, R=R, params=ppo_params, umax=umax
    )

    dbg = {
        "objective": "ppo_zero_sum",
        "ars_iters": int(it_used),
        "g_star": float(g_star),
        "idx_def": int(idx_d),
        "idx_att": [int(x) for x in idx_a],
        "n_actions_def": int(len(Ud)),
        "n_actions_att": [int(len(Ua[i])) for i in range(nA)],
    }
    return uD_star, uA_star, dbg


# ============================================================
# Rollout helpers (frames_dict styling)
# ============================================================

def _p3(xD: np.ndarray, D: int):
    xD = np.asarray(xD, float).reshape(-1)
    if D == 3:
        return (float(xD[0]), float(xD[1]), float(xD[2]))
    return (float(xD[0]), float(xD[1]), 0.0)


def _identity_R():
    return np.eye(3, dtype=float)


def _pad3(u: np.ndarray, D: int) -> np.ndarray:
    u = np.asarray(u, float).reshape(-1)
    if D == 3:
        return np.array([u[0], u[1], u[2]], float)
    return np.array([u[0], u[1], 0.0], float)


# ============================================================
# Rollout runners
# ============================================================

def run_rhc_with_paper_game_1v1_collect_frames_3d(
    cfg: Dict[str, Any],
    steps: int | None = None,
    turn_len: int | None = None,
):
    """
    1v1 rollout.

    Agent mapping to match your plotting conventions:
      - Agent 1 (exec1) = "evader slot" (your defender slot)
      - Agent 2 (exec2) = "pursuer slot" (your attacker slot)

    Objective switch:
      cfg["paper_baseline"]["objective"] in {"paper","ppo_zero_sum"}

    Paper submode:
      cfg["paper"]["game_mode"] in {"paper_ne","minmax"}
    """
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx = 2 * D
    T_horizon = int(cfg["T"])

    if steps is None:
        steps = int(cfg.get("T_eval", cfg.get("T", cfg.get("steps", 60))))
    if turn_len is None:
        turn_len = 1

    paper_params = _paper_params_from_cfg(cfg)
    obj_mode = _baseline_objective_mode(cfg)

    umax = float(cfg.get("umax", 5e-4))
    umax_e, umax_p = _paper_control_limits_from_cfg(cfg)
    step_plant_single, center, _ = _build_step_plant_single(cfg, steps=steps, D=D)
    R = float(cfg.get("arena", {}).get("r", 30.0))
    paper_check = _paper_parameter_check(cfg, paper_params, turn_len=turn_len)

    # PPO zero-sum objective params (only used if obj_mode == "ppo_zero_sum")
    ppo_obj = _populate_ppo_obj_from_cfg(cfg, _ppo_obj_params_from_cfg(cfg))

    # initial states
    x0 = np.asarray(cfg["x0"], dtype=np.float32)
    xE = x0[0, :nx].copy()
    xP = x0[1, :nx].copy()

    # previous controls (neighborhood centers)
    uE_prev = np.zeros((D,), dtype=np.float32)
    uP_prev = [np.zeros((D,), dtype=np.float32)]
    active_uE = uE_prev.copy()
    active_uP = [u.copy() for u in uP_prev]

    # logs for animator
    plan_hist1, plan_hist2 = [], []
    plan_att1, plan_att2 = [], []
    exec_xyz1, exec_xyz2 = [], []
    exec_att1, exec_att2 = [], []
    phi_hist1, phi_hist2 = [], []
    fov_axis_hist, fov_seen_mask = [], []

    # command logs
    u_cmd_1, u_cmd_2 = [], []
    u_cmd_norm_1, u_cmd_norm_2 = [], []

    # debug logs
    it_hist: List[int] = []
    g_star_hist: List[float] = []
    terminal_hist: List[Dict[str, Any]] = []
    stopped_early = False

    # t=0
    exec_xyz1.append(_p3(xE[:D], D))
    exec_xyz2.append(_p3(xP[:D], D))
    exec_att1.append({"R": _identity_R(), "phi": 0.0}); phi_hist1.append(0.0)
    exec_att2.append({"R": _identity_R(), "phi": 0.0}); phi_hist2.append(0.0)
    fov_axis_hist.append(None); fov_seen_mask.append(False)

    for k in range(steps):
        should_resolve = (k == 0) or ((k % turn_len) == 0)
        if should_resolve:
            if obj_mode == "paper":
                active_uE, active_uP_list, dbg = _solve_controls_paper(
                    cfg,
                    xE=xE, xP=[xP],
                    uE_prev=uE_prev, uP_prev=uP_prev,
                    k=k, step_plant_single=step_plant_single,
                    paper_params=paper_params,
                    umax_e=umax_e,
                    umax_p=umax_p,
                )
                active_uP = [active_uP_list[0]]
            else:
                active_uD, active_uA_list, dbg = _solve_one_step_minmax_team_ppo_oi(
                    xD=xE, xA_list=[xP],
                    uD_prev=uE_prev, uA_prev=uP_prev,
                    k=k,
                    step_plant_single=step_plant_single,
                    center=center, R=R,
                    paper_params=paper_params,
                    ppo_params=ppo_obj,
                    umax=umax,
                )
                active_uE = active_uD
                active_uP = [active_uA_list[0]]
        else:
            dbg = {
                "objective": obj_mode,
                "reused_action": True,
                "ars_iters": 0,
                "g_star": float(g_star_hist[-1]) if g_star_hist else float("nan"),
            }

        uE = active_uE
        uP = active_uP[0]

        # debug traces
        it_hist.append(int(dbg.get("ars_iters", 0)))
        g_star_hist.append(float(dbg.get("g_star", np.nan)))

        # thrust logs
        uE3 = _pad3(uE, D)
        uP3 = _pad3(uP, D)
        u_cmd_1.append(uE3.copy()); u_cmd_norm_1.append(float(np.linalg.norm(uE3)))
        u_cmd_2.append(uP3.copy()); u_cmd_norm_2.append(float(np.linalg.norm(uP3)))

        # plant
        xE = step_plant_single(xE, uE, k)
        xP = step_plant_single(xP, uP, k)

        _, term_dbg = _ppo_terminal_adjustment(
            pD=np.asarray(xE[:D], float),
            pA_list=[np.asarray(xP[:D], float)],
            center=center,
            R=R,
            params=ppo_obj,
            primary_idx=0,
        )
        terminal_hist.append(term_dbg)

        # update neighborhood centers
        uE_prev = uE
        uP_prev = [uP]

        # flat plans like your RL runner
        plan1 = [_p3(xE, D)] * T_horizon
        plan2 = [_p3(xP, D)] * T_horizon
        plan_hist1.append(plan1); plan_hist2.append(plan2)

        I = _identity_R()
        att_stub = [{"R": I, "phi": 0.0} for _ in range(T_horizon)]
        plan_att1.append(att_stub); plan_att2.append(att_stub)

        exec_att1.append({"R": I, "phi": 0.0}); phi_hist1.append(0.0)
        exec_att2.append({"R": I, "phi": 0.0}); phi_hist2.append(0.0)

        exec_xyz1.append(_p3(xE, D))
        exec_xyz2.append(_p3(xP, D))
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

        # debug/meta
        "paper_params": paper_params.__dict__,
        "paper_parameter_check": paper_check,
        "paper_game_mode": _pick_paper_game_mode(cfg),
        "baseline_objective": obj_mode,
        "iters_hist": it_hist,
        "g_star_hist": g_star_hist,
        "terminal_hist": terminal_hist,
        "stopped_early": bool(stopped_early),
    }
    return out


def run_rhc_with_paper_game_1v2_collect_frames_3d(
    cfg: Dict[str, Any],
    steps: int | None = None,
    turn_len: int | None = None,
):
    """
    1v2 rollout.

    Agent mapping:
      - Agent 1 (exec1) = "evader slot" (your defender slot)
      - Agents 2&3      = "pursuer slots" (your attacker slots)

    Objective switch:
      cfg["paper_baseline"]["objective"] in {"paper","ppo_zero_sum"}

    Paper submode:
      cfg["paper"]["game_mode"] in {"paper_ne","minmax"}
    """
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx = 2 * D
    T_horizon = int(cfg["T"])

    if steps is None:
        steps = int(cfg.get("T_eval", cfg.get("T", cfg.get("steps", 60))))
    if turn_len is None:
        turn_len = 1

    paper_params = _paper_params_from_cfg(cfg)
    obj_mode = _baseline_objective_mode(cfg)

    umax = float(cfg.get("umax", 5e-4))
    umax_e, umax_p = _paper_control_limits_from_cfg(cfg)
    step_plant_single, center, _ = _build_step_plant_single(cfg, steps=steps, D=D)
    R = float(cfg.get("arena", {}).get("r", 30.0))
    paper_check = _paper_parameter_check(cfg, paper_params, turn_len=turn_len)

    ppo_obj = _populate_ppo_obj_from_cfg(cfg, _ppo_obj_params_from_cfg(cfg))

    x0 = np.asarray(cfg["x0"], dtype=np.float32)
    xE = x0[0, :nx].copy()
    xP0 = x0[1, :nx].copy()
    if x0.shape[0] >= 3:
        xP1 = x0[2, :nx].copy()
    else:
        xP1 = x0[1, :nx].copy()

    uE_prev = np.zeros((D,), dtype=np.float32)
    uP_prev = [np.zeros((D,), dtype=np.float32), np.zeros((D,), dtype=np.float32)]
    active_uE = uE_prev.copy()
    active_uP = [u.copy() for u in uP_prev]

    plan_hist1, plan_hist2, plan_hist3 = [], [], []
    plan_att1, plan_att2, plan_att3 = [], [], []
    exec_xyz1, exec_xyz2, exec_xyz3 = [], [], []
    exec_att1, exec_att2, exec_att3 = [], [], []
    phi_hist1, phi_hist2, phi_hist3 = [], [], []
    fov_axis_hist, fov_seen_mask = [], []

    u_cmd_1, u_cmd_2, u_cmd_3 = [], [], []
    u_cmd_norm_1, u_cmd_norm_2, u_cmd_norm_3 = [], [], []

    it_hist: List[int] = []
    g_star_hist: List[float] = []
    terminal_hist: List[Dict[str, Any]] = []
    stopped_early = False

    # t=0
    exec_xyz1.append(_p3(xE[:D], D))
    exec_xyz2.append(_p3(xP0[:D], D))
    exec_xyz3.append(_p3(xP1[:D], D))
    I = _identity_R()
    exec_att1.append({"R": I, "phi": 0.0}); phi_hist1.append(0.0)
    exec_att2.append({"R": I, "phi": 0.0}); phi_hist2.append(0.0)
    exec_att3.append({"R": I, "phi": 0.0}); phi_hist3.append(0.0)
    fov_axis_hist.append(None); fov_seen_mask.append(False)

    for k in range(steps):
        should_resolve = (k == 0) or ((k % turn_len) == 0)
        if should_resolve:
            if obj_mode == "paper":
                active_uE, active_uP, dbg = _solve_controls_paper(
                    cfg,
                    xE=xE, xP=[xP0, xP1],
                    uE_prev=uE_prev, uP_prev=uP_prev,
                    k=k, step_plant_single=step_plant_single,
                    paper_params=paper_params,
                    umax_e=umax_e,
                    umax_p=umax_p,
                )
            else:
                active_uD, active_uA, dbg = _solve_one_step_minmax_team_ppo_oi(
                    xD=xE, xA_list=[xP0, xP1],
                    uD_prev=uE_prev, uA_prev=uP_prev,
                    k=k,
                    step_plant_single=step_plant_single,
                    center=center, R=R,
                    paper_params=paper_params,
                    ppo_params=ppo_obj,
                    umax=umax,
                )
                active_uE = active_uD
                active_uP = active_uA
        else:
            dbg = {
                "objective": obj_mode,
                "reused_action": True,
                "ars_iters": 0,
                "g_star": float(g_star_hist[-1]) if g_star_hist else float("nan"),
            }

        uE = active_uE
        uP0, uP1 = active_uP[0], active_uP[1]

        # debug traces
        it_hist.append(int(dbg.get("ars_iters", 0)))
        g_star_hist.append(float(dbg.get("g_star", np.nan)))

        # thrust logs
        uE3 = _pad3(uE, D)
        u03 = _pad3(uP0, D)
        u13 = _pad3(uP1, D)

        u_cmd_1.append(uE3.copy()); u_cmd_norm_1.append(float(np.linalg.norm(uE3)))
        u_cmd_2.append(u03.copy()); u_cmd_norm_2.append(float(np.linalg.norm(u03)))
        u_cmd_3.append(u13.copy()); u_cmd_norm_3.append(float(np.linalg.norm(u13)))

        # plant
        xE = step_plant_single(xE, uE, k)
        xP0 = step_plant_single(xP0, uP0, k)
        xP1 = step_plant_single(xP1, uP1, k)

        _, term_dbg = _ppo_terminal_adjustment(
            pD=np.asarray(xE[:D], float),
            pA_list=[np.asarray(xP0[:D], float), np.asarray(xP1[:D], float)],
            center=center,
            R=R,
            params=ppo_obj,
            primary_idx=0,
        )
        terminal_hist.append(term_dbg)

        # update neighborhood centers
        uE_prev = uE
        uP_prev = [uP0, uP1]

        # flat plans
        plan1 = [_p3(xE, D)] * T_horizon
        plan2 = [_p3(xP0, D)] * T_horizon
        plan3 = [_p3(xP1, D)] * T_horizon
        plan_hist1.append(plan1); plan_hist2.append(plan2); plan_hist3.append(plan3)

        att_stub = [{"R": I, "phi": 0.0} for _ in range(T_horizon)]
        plan_att1.append(att_stub); plan_att2.append(att_stub); plan_att3.append(att_stub)

        exec_att1.append({"R": I, "phi": 0.0}); phi_hist1.append(0.0)
        exec_att2.append({"R": I, "phi": 0.0}); phi_hist2.append(0.0)
        exec_att3.append({"R": I, "phi": 0.0}); phi_hist3.append(0.0)

        exec_xyz1.append(_p3(xE, D))
        exec_xyz2.append(_p3(xP0, D))
        exec_xyz3.append(_p3(xP1, D))
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

        # debug/meta
        "paper_params": paper_params.__dict__,
        "paper_parameter_check": paper_check,
        "paper_game_mode": _pick_paper_game_mode(cfg),
        "baseline_objective": obj_mode,
        "iters_hist": it_hist,
        "g_star_hist": g_star_hist,
        "terminal_hist": terminal_hist,
        "stopped_early": bool(stopped_early),
    }
    return out


__all__ = [
    "run_rhc_with_paper_game_1v1_collect_frames_3d",
    "run_rhc_with_paper_game_1v2_collect_frames_3d",
]
