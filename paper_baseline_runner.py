# paper_baseline_runner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


# -----------------------------
# Paper parameter bundle
# -----------------------------
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

    # Pursuer safety constraint (Eq. 22): dk+1_{pi,p(-i)} > dsafe
    dsafe: float = 1.0

    # ARS / best-response loop controls (Algorithm 1/2 spirit)
    max_ars_iter: int = 30
    ne_tol: float = 0.0          # 0.0 means exact match on discrete grid
    break_on_cycle: bool = True


def _paper_params_from_cfg(cfg: Dict[str, Any]) -> PaperParams:
    """
    Override defaults via cfg["paper"] dict, e.g.
      cfg["paper"] = {"Nup": 3, "dup": 1e-3, "tau": 6, "dsafe": 2.0, ...}
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
    return p


# -----------------------------
# Dynamics adapter (matches your RL runners)
# -----------------------------
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
    ar.setdefault("cx", 0.0); ar.setdefault("cy", 0.0); ar.setdefault("cz", 0.0)
    ar.setdefault("r", 30.0)
    center = np.array([ar["cx"], ar["cy"], (ar.get("cz", 0.0) if D == 3 else 0.0)], dtype=np.float32)[:D]

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


# -----------------------------
# Strategy set (Eq. 15-style neighborhood)
# -----------------------------
def _action_grid(u_prev: np.ndarray, du: float, N: int, umax: float, grid_mode: str) -> List[np.ndarray]:
    """
    grid_mode="cartesian": all per-axis combinations u_prev + [i,j,k]*du, i,j,k in [-N..N].
    Returns list of D-vectors clipped to [-umax, +umax].
    """
    u_prev = np.asarray(u_prev, float).reshape(-1)
    D = u_prev.size

    if N <= 0 or du <= 0.0:
        return [np.clip(u_prev, -umax, +umax).astype(np.float32)]

    if grid_mode != "cartesian":
        # Fallback: cartesian anyway (best coverage; paper text is ambiguous on per-axis vs scalar increment).
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

    # D >= 3 (use first 3 comps as typical RTN)
    for dx in vals:
        for dy in vals:
            for dz in vals:
                inc = np.array([dx, dy, dz], float)
                if D > 3:
                    inc = np.pad(inc, (0, D - 3), constant_values=0.0)
                u = u_prev + inc
                out.append(np.clip(u, -umax, +umax).astype(np.float32))
    return out


# -----------------------------
# Paper payoff computation (Eqs. 17–22)
# -----------------------------
def _dist_metrics(pE: np.ndarray, pP: List[np.ndarray]) -> Tuple[float, float]:
    """Return (d_min, d_sum) in meters."""
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
    Evader maximizes J_e.
    Pursuers minimize J_pi.

    Implements the structure described by the paper:
      - distance factor F1 (one-step)
      - prediction guidance factor F2 (tau-step)
      - energy factor F3
      - performance indices J_e and J_pi with constraints
    """
    D = int(xE.size // 2)
    eps = 1e-12

    # current positions
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

    # --- factors ---
    # F1: one-step distance factor (for both sides; sign handled in J)
    term_min_1 = np.arctan(params.alpha1 * ((dmin_k - dmin_1) / dmin_k))
    term_sum_1 = np.arctan(params.alpha2 * ((dsum_k - dsum_1) / dmin_k))
    F1 = (2.0 / np.pi) * (params.beta1 * term_min_1 + params.beta2 * term_sum_1)

    # F2: prediction guidance factor
    F2_p = (2.0 / np.pi) * np.arctan(params.alpha3 * ((dmin_k - dmin_T) / dmin_k))
    term_min_2e = np.arctan(params.alpha4 * ((dmin_k - dmin_T) / dmin_k))
    term_sum_2e = np.arctan(params.alpha5 * ((dsum_k - dsum_T) / dmin_k))
    F2_e = (2.0 / np.pi) * (params.beta1 * term_min_2e + params.beta2 * term_sum_2e)

    # F3: normalized energy (accel magnitude)
    F3_e = float(np.linalg.norm(uE) / max(umax_e, eps))
    F3_p_each = [float(np.linalg.norm(ui) / max(umax_p, eps)) for ui in uP]

    # --- performance indices (paper’s sign pattern) ---
    # Evader payoff (maximize)
    if np.linalg.norm(uE) <= umax_e + 1e-12:
        J_e = float(-params.c1 * F1 - params.c2 * F2_e - params.c3 * F3_e)
    else:
        J_e = -10.0

    # Pursuer payoffs (minimize), with dsafe constraint checked at k+1
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


# -----------------------------
# One-step NE via ARS (best-response iterations)
# -----------------------------
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
      pursuers: minimize J_pi
      evader:   maximize J_e

    Returns (uE*, [uP1*,..], debug_info)
    """
    nP = len(uP_prev)
    uE_prev = np.asarray(uE_prev, float).reshape(-1)
    uP_prev = [np.asarray(u, float).reshape(-1) for u in uP_prev]

    # Build discrete strategy sets
    Ue = _action_grid(uE_prev, params.due, params.Nue, umax_e, params.grid_mode)
    Up = [_action_grid(uP_prev[i], params.dup, params.Nup, umax_p, params.grid_mode) for i in range(nP)]

    # Helper: nearest index to previous (so we have a stable initial point)
    def _nearest_idx(U: List[np.ndarray], u: np.ndarray) -> int:
        d = [float(np.linalg.norm(ui - u)) for ui in U]
        return int(np.argmin(d))

    idx_e = _nearest_idx(Ue, uE_prev)
    idx_p = [ _nearest_idx(Up[i], uP_prev[i]) for i in range(nP) ]

    seen = set()

    def _profile_key(ie: int, ip: List[int]) -> Tuple[int, Tuple[int, ...]]:
        return (ie, tuple(int(x) for x in ip))

    def _best_resp_p(i: int, ie: int, ip: List[int]) -> int:
        # pursuer i minimizes its J_i, others fixed
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
        # evader maximizes J_e, pursuers fixed
        best_j = None
        best_idx = ie
        for a_idx, ue in enumerate(Ue):
            uP = [Up[j][ip[j]] for j in range(nP)]
            Je, _ = _paper_payoffs(xE, xP, ue, uP, k, step_plant_single, params, umax_e, umax_p)
            if (best_j is None) or (Je > best_j + params.ne_tol):
                best_j = Je
                best_idx = a_idx
        return best_idx

    def _is_ne(ie: int, ip: List[int]) -> bool:
        # Check unilateral deviations in the discrete sets
        # pursuers: no one can reduce J_i
        for i in range(nP):
            br_i = _best_resp_p(i, ie, ip)
            if br_i != ip[i]:
                return False
        # evader: cannot improve
        br_e = _best_resp_e(ie, ip)
        return (br_e == ie)

    # ARS iterations
    it_used = 0
    for it in range(params.max_ars_iter):
        it_used = it + 1

        key = _profile_key(idx_e, idx_p)
        if params.break_on_cycle:
            if key in seen:
                # cycle detected; return current profile (paper does random restart; you can add that later)
                break
            seen.add(key)

        # update pursuers sequentially
        changed = False
        for i in range(nP):
            new_i = _best_resp_p(i, idx_e, idx_p)
            if new_i != idx_p[i]:
                idx_p[i] = new_i
                changed = True

        # update evader
        new_e = _best_resp_e(idx_e, idx_p)
        if new_e != idx_e:
            idx_e = new_e
            changed = True

        if not changed:
            # fixed point of best responses
            break

        # optional full NE check (costly but small sets)
        if _is_ne(idx_e, idx_p):
            break

    uE_star = np.asarray(Ue[idx_e], np.float32)
    uP_star = [np.asarray(Up[i][idx_p[i]], np.float32) for i in range(nP)]

    dbg = {
        "ars_iters": it_used,
        "idx_e": int(idx_e),
        "idx_p": [int(x) for x in idx_p],
        "n_actions_e": int(len(Ue)),
        "n_actions_p": [int(len(Up[i])) for i in range(nP)],
    }
    return uE_star, uP_star, dbg


# -----------------------------
# Rollout helpers (frames_dict styling)
# -----------------------------
def _p3(xD: np.ndarray, D: int):
    xD = np.asarray(xD, float).reshape(-1)
    if D == 3:
        return (float(xD[0]), float(xD[1]), float(xD[2]))
    return (float(xD[0]), float(xD[1]), 0.0)


def _identity_R():
    return np.eye(3, dtype=float)


def run_rhc_with_paper_game_1v1_collect_frames_3d(
    cfg: Dict[str, Any],
    steps: int | None = None,
    turn_len: int | None = None,
):
    """
    Paper baseline: 1 pursuer vs 1 evader
      - Agent 1 (exec1) = evader (your defender slot)
      - Agent 2 (exec2) = pursuer (your attacker slot)
    Output keys match your RL runner format.
    """
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx = 2 * D
    T_horizon = int(cfg["T"])
    dt = float(cfg["dt"])

    if steps is None:
        steps = int(cfg.get("T_eval", cfg.get("steps", 60)))
    if turn_len is None:
        turn_len = 1  # unused; kept for symmetry

    params = _paper_params_from_cfg(cfg)
    umax = float(cfg.get("umax", 5e-4))

    step_plant_single, center, _ = _build_step_plant_single(cfg, steps=steps, D=D)

    x0 = np.asarray(cfg["x0"], dtype=np.float32)
    xE = x0[0, :nx].copy()   # evader (defender slot)
    xP = x0[1, :nx].copy()   # pursuer (attacker slot)

    uE_prev = np.zeros((D,), dtype=np.float32)
    uP_prev = [np.zeros((D,), dtype=np.float32)]

    plan_hist1, plan_hist2 = [], []
    plan_att1, plan_att2 = [], []
    exec_xyz1, exec_xyz2 = [], []
    exec_att1, exec_att2 = [], []
    phi_hist1, phi_hist2 = [], []
    fov_axis_hist, fov_seen_mask = [], []

    u_cmd_1, u_cmd_2 = [], []
    u_cmd_norm_1, u_cmd_norm_2 = [], []

    ars_iters_hist = []

    # t=0
    exec_xyz1.append(_p3(xE, D))
    exec_xyz2.append(_p3(xP, D))
    exec_att1.append({"R": _identity_R(), "phi": 0.0}); phi_hist1.append(0.0)
    exec_att2.append({"R": _identity_R(), "phi": 0.0}); phi_hist2.append(0.0)
    fov_axis_hist.append(None); fov_seen_mask.append(False)

    for k in range(steps):
        # Solve one-step NE (discrete ARS)
        uE, uP_list, dbg = _solve_one_step_ne_ars(
            xE=xE,
            xP=[xP],
            uE_prev=uE_prev,
            uP_prev=uP_prev,
            k=k,
            step_plant_single=step_plant_single,
            params=params,
            umax_e=umax,
            umax_p=umax,
        )
        uP = uP_list[0]
        ars_iters_hist.append(int(dbg["ars_iters"]))

        # Log thrust (pad to 3 for consistency with your plotting)
        uE3 = np.array([uE[0], uE[1], uE[2] if D == 3 else 0.0], float)
        uP3 = np.array([uP[0], uP[1], uP[2] if D == 3 else 0.0], float)
        u_cmd_1.append(uE3.copy()); u_cmd_norm_1.append(float(np.linalg.norm(uE3)))
        u_cmd_2.append(uP3.copy()); u_cmd_norm_2.append(float(np.linalg.norm(uP3)))

        # Plant step
        xE = step_plant_single(xE, uE, k)
        xP = step_plant_single(xP, uP, k)

        # Update "previous" controls (Eq. 15-style neighborhood next step)
        uE_prev = uE
        uP_prev = [uP]

        # Flat plans like your RL runner
        plan1 = [_p3(xE, D)] * T_horizon
        plan2 = [_p3(xP, D)] * T_horizon
        plan_hist1.append(plan1); plan_hist2.append(plan2)

        I = _identity_R()
        att_stub = [{"R": I, "phi": 0.0} for _ in range(T_horizon)]
        plan_att1.append(att_stub); plan_att2.append(att_stub)
        exec_att1.append({"R": I, "phi": 0.0})
        exec_att2.append({"R": I, "phi": 0.0})
        phi_hist1.append(0.0); phi_hist2.append(0.0)

        exec_xyz1.append(_p3(xE, D))
        exec_xyz2.append(_p3(xP, D))
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

        # debug
        "paper_params": params.__dict__,
        "ars_iters_hist": ars_iters_hist,
    }
    return out


def run_rhc_with_paper_game_1v2_collect_frames_3d(
    cfg: Dict[str, Any],
    steps: int | None = None,
    turn_len: int | None = None,
):
    """
    Paper baseline: 2 cooperative pursuers vs 1 evader
      - Agent 1 (exec1) = evader (your defender slot)
      - Agents 2 & 3 (exec2, exec3) = pursuers (your attacker slots)
    Output keys match your 1v2 RL runner style (adds *3 keys).
    """
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx = 2 * D
    T_horizon = int(cfg["T"])

    if steps is None:
        steps = int(cfg.get("T_eval", cfg.get("steps", 60)))
    if turn_len is None:
        turn_len = 1

    params = _paper_params_from_cfg(cfg)
    umax = float(cfg.get("umax", 5e-4))

    step_plant_single, center, _ = _build_step_plant_single(cfg, steps=steps, D=D)

    x0 = np.asarray(cfg["x0"], dtype=np.float32)
    xE = x0[0, :nx].copy()
    xP0 = x0[1, :nx].copy()
    if x0.shape[0] >= 3:
        xP1 = x0[2, :nx].copy()
    else:
        xP1 = x0[1, :nx].copy()

    uE_prev = np.zeros((D,), dtype=np.float32)
    uP_prev = [np.zeros((D,), dtype=np.float32), np.zeros((D,), dtype=np.float32)]

    plan_hist1, plan_hist2, plan_hist3 = [], [], []
    plan_att1, plan_att2, plan_att3 = [], [], []
    exec_xyz1, exec_xyz2, exec_xyz3 = [], [], []
    exec_att1, exec_att2, exec_att3 = [], [], []
    phi_hist1, phi_hist2, phi_hist3 = [], [], []
    fov_axis_hist, fov_seen_mask = [], []

    u_cmd_1, u_cmd_2, u_cmd_3 = [], [], []
    u_cmd_norm_1, u_cmd_norm_2, u_cmd_norm_3 = [], [], []
    ars_iters_hist = []

    # t=0
    exec_xyz1.append(_p3(xE, D))
    exec_xyz2.append(_p3(xP0, D))
    exec_xyz3.append(_p3(xP1, D))
    I = _identity_R()
    exec_att1.append({"R": I, "phi": 0.0}); phi_hist1.append(0.0)
    exec_att2.append({"R": I, "phi": 0.0}); phi_hist2.append(0.0)
    exec_att3.append({"R": I, "phi": 0.0}); phi_hist3.append(0.0)
    fov_axis_hist.append(None); fov_seen_mask.append(False)

    for k in range(steps):
        uE, uP_list, dbg = _solve_one_step_ne_ars(
            xE=xE,
            xP=[xP0, xP1],
            uE_prev=uE_prev,
            uP_prev=uP_prev,
            k=k,
            step_plant_single=step_plant_single,
            params=params,
            umax_e=umax,
            umax_p=umax,
        )
        uP0, uP1 = uP_list[0], uP_list[1]
        ars_iters_hist.append(int(dbg["ars_iters"]))

        # thrust logs (pad)
        def _pad3(u):
            u = np.asarray(u, float).reshape(-1)
            return np.array([u[0], u[1], u[2] if D == 3 else 0.0], float)

        uE3 = _pad3(uE)
        u03 = _pad3(uP0)
        u13 = _pad3(uP1)

        u_cmd_1.append(uE3.copy()); u_cmd_norm_1.append(float(np.linalg.norm(uE3)))
        u_cmd_2.append(u03.copy()); u_cmd_norm_2.append(float(np.linalg.norm(u03)))
        u_cmd_3.append(u13.copy()); u_cmd_norm_3.append(float(np.linalg.norm(u13)))

        # plant
        xE = step_plant_single(xE, uE, k)
        xP0 = step_plant_single(xP0, uP0, k)
        xP1 = step_plant_single(xP1, uP1, k)

        # update prev
        uE_prev = uE
        uP_prev = [uP0, uP1]

        # flat plans
        plan1 = [_p3(xE, D)] * T_horizon
        plan2 = [_p3(xP0, D)] * T_horizon
        plan3 = [_p3(xP1, D)] * T_horizon
        plan_hist1.append(plan1); plan_hist2.append(plan2); plan_hist3.append(plan3)

        I = _identity_R()
        att_stub = [{"R": I, "phi": 0.0} for _ in range(T_horizon)]
        plan_att1.append(att_stub); plan_att2.append(att_stub); plan_att3.append(att_stub)

        exec_att1.append({"R": I, "phi": 0.0}); phi_hist1.append(0.0)
        exec_att2.append({"R": I, "phi": 0.0}); phi_hist2.append(0.0)
        exec_att3.append({"R": I, "phi": 0.0}); phi_hist3.append(0.0)

        exec_xyz1.append(_p3(xE, D))
        exec_xyz2.append(_p3(xP0, D))
        exec_xyz3.append(_p3(xP1, D))
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

        # debug
        "paper_params": params.__dict__,
        "ars_iters_hist": ars_iters_hist,
    }
    return out


__all__ = [
    "run_rhc_with_paper_game_1v1_collect_frames_3d",
    "run_rhc_with_paper_game_1v2_collect_frames_3d",
]