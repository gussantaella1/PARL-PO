"""
matchup_runner.py

Rollout runner for policy-versus-baseline matchups across paper, game-theory, IPOPT, and rule opponents.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np

from core.safety_filter import project_box_halfspace_np, velocity_cbf_halfspace_np
from GameTheory_baseline import _gt_params_from_cfg
from IPOPT_baseline import _ipopt_params_from_cfg, _solve_box_subproblem
from paper_baseline_runner import (
    _action_grid,
    _baseline_objective_mode,
    _build_step_plant_single,
    _identity_R,
    _p3,
    _pad3,
    _paper_control_limits_from_cfg,
    _paper_params_from_cfg,
    _paper_payoffs,
    _paper_security_g,
    _pick_paper_game_mode,
    _populate_ppo_obj_from_cfg,
    _ppo_obj_params_from_cfg,
    _ppo_security_value_g,
)
from rl_infer import RLPolicyDiff
from ukf_estimator import KF_CV
from core.controllers import AttackerRuleController


SUPPORTED_BASELINE_OPPONENTS = ("paper", "game_theory", "ipopt", "rule")


def _v3_from_state(xD: np.ndarray, D: int) -> np.ndarray:
    """Internal helper for v3 from state."""
    xD = np.asarray(xD, float).reshape(-1)
    if D == 3:
        return np.array([xD[3], xD[4], xD[5]], dtype=float)
    return np.array([xD[2], xD[3], 0.0], dtype=float)


def _normalize_kf_action_access(mode: Any) -> str:
    """Normalize kf action access into the canonical representation used here."""
    key = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
    if key == "groundtruth":
        key = "ground_truth"
    if key == "inferred":
        key = "measured"
    valid = {"ground_truth", "measured", "none"}
    if key not in valid:
        raise ValueError(f"Unsupported estimator action_access='{mode}'. Expected one of {sorted(valid)}.")
    return key


def _normalize_kf_control_noise_std(noise_std: Any, default_std: float) -> np.ndarray:
    """Normalize kf control noise std into the canonical representation used here."""
    arr = np.asarray(noise_std if noise_std is not None else default_std, dtype=float)
    if arr.ndim == 0:
        std = np.full(3, float(arr), dtype=float)
    else:
        arr = arr.reshape(-1)
        if arr.size != 3:
            raise ValueError(
                f"Expected estimator action_meas_std to be a scalar or length-3 sequence, got shape {arr.shape}."
            )
        std = arr.astype(float)
    return np.maximum(std, 0.0)


def _normalize_estimator_kind(cfg: Dict[str, Any]) -> str:
    """Normalize estimator kind into the canonical representation used here."""
    kind = cfg.get("estimator_kind", "ukf")
    key = str(kind).strip().lower()
    if key not in {"ukf", "ekf"}:
        raise ValueError(f"Unsupported estimator_kind='{kind}'. Expected 'ukf' or 'ekf'.")
    return key


def _ekf_factory_kwargs(cfg: Dict[str, Any], linearization_group: str) -> Dict[str, Any]:
    """Internal helper for ekf factory kwargs."""
    if _normalize_estimator_kind(cfg) != "ekf":
        return {}
    ukf_cfg = cfg.get("ukf", {})
    return {
        "jacobian_mode": ukf_cfg.get("ekf_jacobian_mode", "exact"),
        "linearization_group": linearization_group,
        "use_torch_backend": bool(ukf_cfg.get("ekf_use_torch", False)),
        "device": cfg.get("device", "cpu"),
    }


def _kf_predict_control(
    action_access: str,
    ground_truth_u: np.ndarray | None,
    control_meas_std: np.ndarray | None = None,
    control_limit: float | None = None,
):
    """Internal helper for kf predict control."""
    if ground_truth_u is None:
        return None, None
    u_true = np.asarray(ground_truth_u, dtype=float).reshape(3)
    if action_access == "ground_truth":
        return u_true, None
    if action_access == "measured":
        std = np.asarray(control_meas_std if control_meas_std is not None else np.zeros(3), dtype=float).reshape(3)
        noise = np.random.normal(loc=0.0, scale=std, size=3)
        u_meas = u_true + noise
        if control_limit is not None and np.isfinite(control_limit):
            u_meas = np.clip(u_meas, -control_limit, control_limit)
        return u_meas, np.diag(std**2)
    return None, None


def _normalize_angle(a: float) -> float:
    """Normalize angle into the canonical representation used here."""
    return (float(a) + np.pi) % (2.0 * np.pi) - np.pi


def _project_box(u: np.ndarray, umax: float) -> np.ndarray:
    """Project box into the constrained space used by the controller."""
    return np.clip(np.asarray(u, dtype=float), -float(umax), float(umax))


def _build_velocity_cbf_filter(
    cfg: Dict[str, Any],
    *,
    D: int,
    dt: float,
    umax: float,
):
    """Build velocity cbf filter for the current workflow."""
    sf_cfg = dict(cfg.get("safety_filter", {}) or {})
    enabled = bool(sf_cfg.get("enabled", False))
    u_lo = -float(umax)
    u_hi = +float(umax)

    if not enabled:
        def _no_filter(p: np.ndarray, v: np.ndarray, u_nom: np.ndarray) -> np.ndarray:
            """Internal helper for no filter."""
            return np.clip(np.asarray(u_nom, dtype=float).reshape(D,), u_lo, u_hi).astype(np.float32)
        return _no_filter

    kind = str(sf_cfg.get("kind", "velocity_cbf_qp")).strip().lower()
    if kind != "velocity_cbf_qp":
        raise ValueError(
            f"Unsupported safety_filter.kind={kind!r}; expected 'velocity_cbf_qp'."
        )

    raw_cbf_vmax = sf_cfg.get("vmax", None)
    if raw_cbf_vmax is None:
        raw_cbf_vmax = cfg.get("vmax", None)
    vmax = np.inf if raw_cbf_vmax is None else float(raw_cbf_vmax)
    alpha = float(sf_cfg.get("alpha", 5.0))
    if not np.isfinite(vmax) or vmax <= 0.0:
        raise ValueError(
            "safety_filter requires a finite positive vmax either in "
            "safety_filter['vmax'] (or legacy top-level cfg['vmax'])."
        )
    if alpha <= 0.0:
        raise ValueError(f"safety_filter['alpha'] must be > 0, got {alpha}.")

    dyn_name = str(cfg.get("dynamics", "hcw")).strip().lower()
    hcw_n = None
    if dyn_name == "hcw":
        hcw_cfg = dict(cfg.get("hcw", {}) or {})
        if "n" in hcw_cfg:
            hcw_n = float(hcw_cfg["n"])
        else:
            mu = float(hcw_cfg.get("mu", 3.986004418e14))
            r0 = float(hcw_cfg["r0"])
            hcw_n = float(np.sqrt(mu / (r0 ** 3)))

    dyn_cfg = dict(cfg.get("dyn", {}) or {})
    Ad = dyn_cfg.get("Ad", None)
    Bd = dyn_cfg.get("Bd", None)
    if Ad is not None:
        Ad = np.asarray(Ad, dtype=float)
    if Bd is not None:
        Bd = np.asarray(Bd, dtype=float)

    def _apply_filter(p: np.ndarray, v: np.ndarray, u_nom: np.ndarray) -> np.ndarray:
        """Apply filter to the current config, state, or rollout data."""
        u_nom = np.asarray(u_nom, dtype=float).reshape(D,)
        a, b = velocity_cbf_halfspace_np(
            np.asarray(p, dtype=float).reshape(D,),
            np.asarray(v, dtype=float).reshape(D,),
            vmax=vmax,
            alpha=alpha,
            dyn_name=dyn_name,
            dt=dt,
            D=D,
            Ad=Ad,
            Bd=Bd,
            hcw_n=hcw_n,
        )
        return project_box_halfspace_np(u_nom, u_lo, u_hi, a, b).astype(np.float32)

    return _apply_filter


def _finite_diff_grad(fun, x: np.ndarray, eps: float) -> np.ndarray:
    """Estimate grad with finite differences."""
    x = np.asarray(x, dtype=float).reshape(-1)
    grad = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (float(fun(xp)) - float(fun(xm))) / (2.0 * eps)
    return grad


def _as_action_list(actions: Any, D: int, n_expected: int) -> List[np.ndarray]:
    """Internal helper for as action list."""
    # Baseline solvers disagree on whether a single action is a vector or a list.
    # Normalize here so downstream 1vN adapters can work with one representation.
    if isinstance(actions, (list, tuple)):
        out = [np.asarray(a, dtype=float).reshape(-1) for a in actions]
    else:
        arr = np.asarray(actions, dtype=float)
        if arr.ndim == 1:
            out = [arr.reshape(-1)]
        elif arr.ndim == 2:
            out = [arr[i].reshape(-1) for i in range(arr.shape[0])]
        else:
            raise ValueError(f"Unsupported action shape {arr.shape} for D={D}, n_expected={n_expected}.")

    if len(out) != n_expected:
        raise ValueError(f"Expected {n_expected} action vectors, got {len(out)}.")
    for a in out:
        if a.size != D:
            raise ValueError(f"Expected action dim {D}, got {a.size}.")
    return [a.astype(np.float32) for a in out]


def _flatten_action_list(actions: List[np.ndarray]) -> np.ndarray:
    """Flatten per-agent action arrays into one optimization vector."""
    if not actions:
        return np.zeros((0,), dtype=np.float32)
    # Box optimizers operate on one vector, while rollout code wants per-agent arrays.
    return np.concatenate([np.asarray(a, dtype=np.float32).reshape(-1) for a in actions], axis=0)


def _unflatten_actions(u_vec: np.ndarray, D: int, n_actions: int) -> List[np.ndarray]:
    """Convert flattened action vectors back into per-agent action arrays."""
    u_vec = np.asarray(u_vec, dtype=float).reshape(-1)
    if u_vec.size != D * n_actions:
        raise ValueError(f"Expected flattened action dim {D * n_actions}, got {u_vec.size}.")
    return [u_vec[i * D:(i + 1) * D].astype(np.float32) for i in range(n_actions)]


def _restore_action_shape(actions: List[np.ndarray]) -> np.ndarray | List[np.ndarray]:
    """Internal helper for restore action shape."""
    if len(actions) == 1:
        return np.asarray(actions[0], dtype=np.float32)
    return [np.asarray(a, dtype=np.float32) for a in actions]


def _paper_best_response(
    *,
    cfg: Dict[str, Any],
    baseline_role: str,
    xD: np.ndarray,
    xA_list: List[np.ndarray],
    fixed_u_policy: np.ndarray | List[np.ndarray],
    u_baseline_prev: np.ndarray | List[np.ndarray],
    k: int,
    step_plant_single,
    center: np.ndarray,
    arena_r: float,
    ppo_obj,
    umax: float,
) -> Tuple[np.ndarray | List[np.ndarray], Dict[str, Any]]:
    """Internal helper for paper best response."""
    paper_params = _paper_params_from_cfg(cfg)
    obj_mode = _baseline_objective_mode(cfg)
    game_mode = _pick_paper_game_mode(cfg)
    umax_e, umax_p = _paper_control_limits_from_cfg(cfg)
    D = int(np.asarray(xD).reshape(-1).size // 2)
    nA = len(xA_list)

    if baseline_role == "def":
        fixed_u_policy_list = _as_action_list(fixed_u_policy, D, nA)
        grid = _action_grid(
            np.asarray(u_baseline_prev, dtype=float).reshape(-1),
            paper_params.due,
            paper_params.Nue,
            umax_e if obj_mode == "paper" else umax,
            paper_params.grid_mode,
        )

        best_val = None
        best_u = np.asarray(u_baseline_prev, dtype=np.float32).copy()
        eval_count = 0

        for u_cand in grid:
            eval_count += 1
            if obj_mode == "paper":
                if game_mode == "paper_ne":
                    val, _ = _paper_payoffs(
                        xE=xD,
                        xP=xA_list,
                        uE=u_cand,
                        uP=fixed_u_policy_list,
                        k=k,
                        step_plant_single=step_plant_single,
                        params=paper_params,
                        umax_e=umax_e,
                        umax_p=umax_p,
                    )
                else:
                    val = _paper_security_g(
                        xE=xD,
                        xP=xA_list,
                        uE=u_cand,
                        uP=fixed_u_policy_list,
                        k=k,
                        step_plant_single=step_plant_single,
                        params=paper_params,
                        umax_e=umax_e,
                        umax_p=umax_p,
                    )
                better = (best_val is None) or (float(val) > float(best_val))
            else:
                val = _ppo_security_value_g(
                    xD=xD,
                    xA_list=xA_list,
                    uD=u_cand,
                    uA_list=fixed_u_policy_list,
                    k=k,
                    step_plant_single=step_plant_single,
                    center=center,
                    R=arena_r,
                    params=ppo_obj,
                    umax=umax,
                )
                better = (best_val is None) or (float(val) > float(best_val))

            if better:
                best_val = float(val)
                best_u = np.asarray(u_cand, dtype=np.float32).copy()

        dbg = {
            "baseline": "paper",
            "baseline_role": baseline_role,
            "objective": obj_mode,
            "paper_game_mode": game_mode if obj_mode == "paper" else None,
            "solver_family": "grid_best_response",
            "eval_count": int(eval_count),
            "objective_value": float(best_val) if best_val is not None else float("nan"),
            "num_attackers": int(nA),
        }
        return best_u, dbg

    fixed_u_def = np.asarray(fixed_u_policy, dtype=float).reshape(-1)
    uA_prev = _as_action_list(u_baseline_prev, D, nA)
    Ua = [
        _action_grid(
            uA_prev[i],
            paper_params.dup,
            paper_params.Nup,
            umax_p if obj_mode == "paper" else umax,
            paper_params.grid_mode,
        )
        for i in range(nA)
    ]

    def objective(uA_list: List[np.ndarray]) -> float:
        """Handle objective for this workflow."""
        if obj_mode == "paper":
            if game_mode == "paper_ne":
                _, Jp = _paper_payoffs(
                    xE=xD,
                    xP=xA_list,
                    uE=fixed_u_def,
                    uP=uA_list,
                    k=k,
                    step_plant_single=step_plant_single,
                    params=paper_params,
                    umax_e=umax_e,
                    umax_p=umax_p,
                )
                return float(np.mean(Jp))
            return float(
                _paper_security_g(
                    xE=xD,
                    xP=xA_list,
                    uE=fixed_u_def,
                    uP=uA_list,
                    k=k,
                    step_plant_single=step_plant_single,
                    params=paper_params,
                    umax_e=umax_e,
                    umax_p=umax_p,
                )
            )
        return float(
            _ppo_security_value_g(
                xD=xD,
                xA_list=xA_list,
                uD=fixed_u_def,
                uA_list=uA_list,
                k=k,
                step_plant_single=step_plant_single,
                center=center,
                R=arena_r,
                params=ppo_obj,
                umax=umax,
            )
        )

    def nearest_idx(U, u):
        """Handle nearest idx for this workflow."""
        return int(np.argmin([float(np.linalg.norm(ui - u)) for ui in U]))

    idx_a = [nearest_idx(Ua[i], uA_prev[i]) for i in range(nA)]
    seen = set()
    it_used = 0

    for it in range(max(1, int(paper_params.max_ars_iter))):
        it_used = it + 1
        key = tuple(int(v) for v in idx_a)
        if paper_params.break_on_cycle:
            if key in seen:
                break
            seen.add(key)

        changed = False
        for i in range(nA):
            best_val = None
            best_idx = idx_a[i]
            for cand_idx, ui in enumerate(Ua[i]):
                uA_cand = [Ua[j][idx_a[j]] for j in range(nA)]
                uA_cand[i] = ui
                val = objective(uA_cand)
                if (best_val is None) or (val < best_val - paper_params.ne_tol):
                    best_val = val
                    best_idx = cand_idx
            if best_idx != idx_a[i]:
                idx_a[i] = best_idx
                changed = True

        if not changed:
            break

    best_u_list = [np.asarray(Ua[i][idx_a[i]], dtype=np.float32) for i in range(nA)]
    best_val = objective(best_u_list)
    dbg = {
        "baseline": "paper",
        "baseline_role": baseline_role,
        "objective": obj_mode,
        "paper_game_mode": game_mode if obj_mode == "paper" else None,
        "solver_family": "coordinate_descent_best_response",
        "iters_used": int(it_used),
        "objective_value": float(best_val),
        "idx_att": [int(v) for v in idx_a],
        "num_attackers": int(nA),
    }
    return _restore_action_shape(best_u_list), dbg


def _gametheory_best_response(
    *,
    cfg: Dict[str, Any],
    baseline_role: str,
    xD: np.ndarray,
    xA_list: List[np.ndarray],
    fixed_u_policy: np.ndarray | List[np.ndarray],
    u_baseline_prev: np.ndarray | List[np.ndarray],
    k: int,
    step_plant_single,
    center: np.ndarray,
    arena_r: float,
    ppo_obj,
    umax: float,
) -> Tuple[np.ndarray | List[np.ndarray], Dict[str, Any]]:
    """Internal helper for gametheory best response."""
    params = _gt_params_from_cfg(cfg)
    D = int(np.asarray(xD).reshape(-1).size // 2)
    nA = len(xA_list)

    if baseline_role == "def":
        z = _project_box(np.asarray(u_baseline_prev, dtype=float).reshape(-1), umax)
        anchor = np.asarray(u_baseline_prev, dtype=float).reshape(-1)
        fixed_u_policy_list = _as_action_list(fixed_u_policy, D, nA)

        def objective(uD_vec: np.ndarray) -> float:
            """Handle objective for this workflow."""
            return float(
                _ppo_security_value_g(
                    xD=xD,
                    xA_list=xA_list,
                    uD=uD_vec,
                    uA_list=fixed_u_policy_list,
                    k=k,
                    step_plant_single=step_plant_single,
                    center=center,
                    R=arena_r,
                    params=ppo_obj,
                    umax=umax,
                )
                - float(params.reg_prev) * float(np.dot(uD_vec - anchor, uD_vec - anchor))
            )

        step_size = float(params.step_size_def)
        direction = +1.0
    else:
        uA_prev = _as_action_list(u_baseline_prev, D, nA)
        z = _project_box(_flatten_action_list(uA_prev), umax)
        anchor = _flatten_action_list(uA_prev)
        fixed_u_def = np.asarray(fixed_u_policy, dtype=float).reshape(-1)

        def objective(uA_vec: np.ndarray) -> float:
            """Handle objective for this workflow."""
            uA_list = _unflatten_actions(_project_box(uA_vec, umax), D, nA)
            return float(
                _ppo_security_value_g(
                    xD=xD,
                    xA_list=xA_list,
                    uD=fixed_u_def,
                    uA_list=uA_list,
                    k=k,
                    step_plant_single=step_plant_single,
                    center=center,
                    R=arena_r,
                    params=ppo_obj,
                    umax=umax,
                )
                + float(params.reg_prev) * float(np.dot(uA_vec - anchor, uA_vec - anchor))
            )

        step_size = float(params.step_size_att)
        direction = -1.0

    grad_norm_hist: List[float] = []
    for _ in range(max(1, int(params.max_iters))):
        grad = _finite_diff_grad(objective, z, params.fd_eps)
        z_next = _project_box(z + direction * step_size * grad, umax)
        grad_norm = float(np.linalg.norm(grad))
        grad_norm_hist.append(grad_norm)
        delta = float(np.linalg.norm(z_next - z))
        z = z_next
        if grad_norm <= float(params.tol) or delta <= float(params.tol):
            break

    obj_val = float(objective(z))
    dbg = {
        "baseline": "game_theory",
        "baseline_role": baseline_role,
        "objective": "ppo_zero_sum",
        "solver_family": "projected_gradient_best_response",
        "iters_used": int(len(grad_norm_hist)),
        "grad_norm_hist": grad_norm_hist,
        "objective_value": obj_val,
        "num_attackers": int(nA),
    }
    if baseline_role == "def":
        return z.astype(np.float32), dbg
    return _restore_action_shape(_unflatten_actions(z, D, nA)), dbg


def _ipopt_best_response(
    *,
    cfg: Dict[str, Any],
    baseline_role: str,
    xD: np.ndarray,
    xA_list: List[np.ndarray],
    fixed_u_policy: np.ndarray | List[np.ndarray],
    u_baseline_prev: np.ndarray | List[np.ndarray],
    k: int,
    step_plant_single,
    center: np.ndarray,
    arena_r: float,
    ppo_obj,
    umax: float,
) -> Tuple[np.ndarray | List[np.ndarray], Dict[str, Any]]:
    """Internal helper for ipopt best response."""
    params = _ipopt_params_from_cfg(cfg)
    D = int(np.asarray(xD).reshape(-1).size // 2)
    nA = len(xA_list)

    if baseline_role == "def":
        x0 = _project_box(np.asarray(u_baseline_prev, dtype=float).reshape(-1), umax)
        anchor = np.asarray(u_baseline_prev, dtype=float).reshape(-1)
        fixed_u_policy_list = _as_action_list(fixed_u_policy, D, nA)

        def objective(uD_vec: np.ndarray) -> float:
            """Handle objective for this workflow."""
            g = _ppo_security_value_g(
                xD=xD,
                xA_list=xA_list,
                uD=_project_box(uD_vec, umax),
                uA_list=fixed_u_policy_list,
                k=k,
                step_plant_single=step_plant_single,
                center=center,
                R=arena_r,
                params=ppo_obj,
                umax=umax,
            )
            reg = float(params.reg_prev) * float(np.dot(uD_vec - anchor, uD_vec - anchor))
            return float(-g + reg)
    else:
        uA_prev = _as_action_list(u_baseline_prev, D, nA)
        x0 = _project_box(_flatten_action_list(uA_prev), umax)
        anchor = _flatten_action_list(uA_prev)
        fixed_u_def = np.asarray(fixed_u_policy, dtype=float).reshape(-1)

        def objective(uA_vec: np.ndarray) -> float:
            """Handle objective for this workflow."""
            g = _ppo_security_value_g(
                xD=xD,
                xA_list=xA_list,
                uD=fixed_u_def,
                uA_list=_unflatten_actions(_project_box(uA_vec, umax), D, nA),
                k=k,
                step_plant_single=step_plant_single,
                center=center,
                R=arena_r,
                params=ppo_obj,
                umax=umax,
            )
            reg = float(params.reg_prev) * float(np.dot(uA_vec - anchor, uA_vec - anchor))
            return float(g + reg)

    sol, info = _solve_box_subproblem(fun=objective, x0=x0, umax=umax, params=params)
    dbg = {
        "baseline": "ipopt",
        "baseline_role": baseline_role,
        "objective": "ppo_zero_sum",
        "solver_family": "single_side_box_optimization",
        "objective_value": float(objective(sol)),
        "num_attackers": int(nA),
        **info,
    }
    if baseline_role == "def":
        return np.asarray(sol, dtype=np.float32), dbg
    return _restore_action_shape(_unflatten_actions(sol, D, nA)), dbg


def _rule_best_response(
    *,
    cfg: Dict[str, Any],
    baseline_role: str,
    xD: np.ndarray,
    xA_list: List[np.ndarray],
) -> Tuple[np.ndarray | List[np.ndarray], Dict[str, Any]]:
    """Internal helper for rule best response."""
    if baseline_role != "att":
        raise ValueError("opponent_baseline='rule' is only supported for attacker-side baselines.")

    ctrl = AttackerRuleController(cfg)
    pD = np.asarray(xD[:ctrl.D], dtype=np.float32)
    vD = np.asarray(xD[ctrl.D:2 * ctrl.D], dtype=np.float32)
    u_list = []
    for xA in xA_list:
        xA = np.asarray(xA, dtype=np.float32)
        pA = xA[:ctrl.D]
        vA = xA[ctrl.D:2 * ctrl.D]
        u_list.append(np.asarray(ctrl.act(pD, vD, pA, vA), dtype=np.float32))

    dbg = {
        "baseline": "rule",
        "baseline_role": baseline_role,
        "objective": "training_rule_controller",
        "solver_family": "closed_form_rule",
        "num_attackers": int(len(u_list)),
    }
    return _restore_action_shape(u_list), dbg


def _solve_baseline_best_response(
    *,
    cfg: Dict[str, Any],
    opponent_baseline: str,
    baseline_role: str,
    xD: np.ndarray,
    xA_list: List[np.ndarray],
    fixed_u_policy: np.ndarray | List[np.ndarray],
    u_baseline_prev: np.ndarray | List[np.ndarray],
    k: int,
    step_plant_single,
    center: np.ndarray,
    arena_r: float,
    ppo_obj,
    umax: float,
) -> Tuple[np.ndarray | List[np.ndarray], Dict[str, Any]]:
    """Internal helper for solve baseline best response."""
    if opponent_baseline == "rule":
        return _rule_best_response(
            cfg=cfg,
            baseline_role=baseline_role,
            xD=xD,
            xA_list=xA_list,
        )
    if opponent_baseline == "paper":
        return _paper_best_response(
            cfg=cfg,
            baseline_role=baseline_role,
            xD=xD,
            xA_list=xA_list,
            fixed_u_policy=fixed_u_policy,
            u_baseline_prev=u_baseline_prev,
            k=k,
            step_plant_single=step_plant_single,
            center=center,
            arena_r=arena_r,
            ppo_obj=ppo_obj,
            umax=umax,
        )
    if opponent_baseline == "game_theory":
        return _gametheory_best_response(
            cfg=cfg,
            baseline_role=baseline_role,
            xD=xD,
            xA_list=xA_list,
            fixed_u_policy=fixed_u_policy,
            u_baseline_prev=u_baseline_prev,
            k=k,
            step_plant_single=step_plant_single,
            center=center,
            arena_r=arena_r,
            ppo_obj=ppo_obj,
            umax=umax,
        )
    if opponent_baseline == "ipopt":
        return _ipopt_best_response(
            cfg=cfg,
            baseline_role=baseline_role,
            xD=xD,
            xA_list=xA_list,
            fixed_u_policy=fixed_u_policy,
            u_baseline_prev=u_baseline_prev,
            k=k,
            step_plant_single=step_plant_single,
            center=center,
            arena_r=arena_r,
            ppo_obj=ppo_obj,
            umax=umax,
        )
    raise ValueError(f"Unsupported opponent_baseline={opponent_baseline!r}")


def _baseline_should_resolve(

    opponent_baseline: str,
    cfg: Dict[str, Any],
    k: int,
    turn_len: int,
) -> bool:
    """Internal helper for baseline should resolve."""
    # Some baselines are expensive, so turn_len lets them hold the previous action
    # between solver calls while the simulation still steps every dt.
    if k == 0:
        return True
    if opponent_baseline == "rule":
        return True
    if opponent_baseline == "paper":
        return (k % turn_len) == 0
    if opponent_baseline == "game_theory":
        params = _gt_params_from_cfg(cfg)
        return (not bool(params.reuse_action_for_turn_len)) or ((k % turn_len) == 0)
    if opponent_baseline == "ipopt":
        params = _ipopt_params_from_cfg(cfg)
        return (not bool(params.reuse_action_for_turn_len)) or ((k % turn_len) == 0)
    raise ValueError(f"Unsupported opponent_baseline={opponent_baseline!r}")




def _pack_multi_agent_rollout(
    *,
    plan_hist_all,
    plan_att_all,
    exec_xyz_all,
    vel_xyz_all,
    exec_att_all,
    phi_hist_all,
    fov_axis_hist,
    fov_seen_mask,
    u_cmd_all,
    u_cmd_norm_all,
    u_real_all,
    u_real_norm_all,
    done_info,
    rollout_timing_sec,
    fuel_frac_all=None,
    thrust_all=None,
    mdot_all=None,
) -> Dict[str, Any]:
    """Pack multi agent rollout into the dictionary format expected by downstream plotting and metrics."""
    out: Dict[str, Any] = {
        "num_attackers": max(0, len(exec_xyz_all) - 1),
        "plan_hist_all": plan_hist_all,
        "plan_att_all": plan_att_all,
        "exec_xyz_all": exec_xyz_all,
        "vel_xyz_all": [np.asarray(v, dtype=float) for v in vel_xyz_all],
        "exec_att_all": exec_att_all,
        "phi_hist_all": phi_hist_all,
        "fov_axis_hist": fov_axis_hist,
        "fov_seen_mask": fov_seen_mask,
        "u_cmd_all": [np.asarray(u, dtype=float) for u in u_cmd_all],
        "u_cmd_norm_all": [np.asarray(u, dtype=float) for u in u_cmd_norm_all],
        "u_real_all": [np.asarray(u, dtype=float) for u in u_real_all],
        "u_real_norm_all": [np.asarray(u, dtype=float) for u in u_real_norm_all],
        "done_info": done_info,
        "rollout_timing_sec": rollout_timing_sec,
    }

    for idx in range(len(exec_xyz_all)):
        key = idx + 1
        out[f"plan_hist{key}"] = plan_hist_all[idx]
        out[f"plan_att{key}"] = plan_att_all[idx]
        out[f"exec{key}_xyz"] = exec_xyz_all[idx]
        out[f"vel{key}_xyz"] = np.asarray(vel_xyz_all[idx], dtype=float)
        out[f"exec_att{key}"] = exec_att_all[idx]
        out[f"phi_hist{key}"] = phi_hist_all[idx]

    if fuel_frac_all is not None:
        out["fuel_frac_all"] = [np.asarray(v, dtype=float) for v in fuel_frac_all]
    if thrust_all is not None:
        out["thrust_all"] = [np.asarray(v, dtype=float) for v in thrust_all]
    if mdot_all is not None:
        out["mdot_all"] = [np.asarray(v, dtype=float) for v in mdot_all]

    return out


def _run_rhc_with_policy_vs_baseline_collect_frames_3d_multi(
    cfg: Dict[str, Any],
    *,
    policy_role: str,
    opponent_baseline: str,
    steps: int | None = None,
    turn_len: int | None = None,
) -> Dict[str, Any]:
    """Run the internal rhc with policy vs baseline collect frames 3d multi implementation used by the public entry point."""
    t_fn0 = time.perf_counter()

    policy_role = str(policy_role).lower()
    if policy_role not in ("def", "att"):
        raise ValueError(f"policy_role must be 'def' or 'att', got {policy_role!r}.")

    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx = 2 * D
    T = int(cfg["T"])
    dt = float(cfg["dt"])
    Na = int(cfg.get("num_attackers", 1))

    if Na <= 1:
        raise ValueError("_run_rhc_with_policy_vs_baseline_collect_frames_3d_multi expects num_attackers > 1.")

    if steps is None:
        steps = int(cfg.get("T_eval", cfg.get("T", cfg.get("steps", 60))))
    if turn_len is None:
        turn_len = 1
    turn_len = max(1, int(turn_len))

    ar = cfg.setdefault("arena", {})
    ar.setdefault("type", "sphere")
    ar.setdefault("cx", 0.0)
    ar.setdefault("cy", 0.0)
    ar.setdefault("cz", 0.0)
    ar.setdefault("r", 30.0)
    if ar["type"] != "sphere":
        raise ValueError("Only spherical arena is supported in this rollout helper.")

    center = np.array(
        [ar["cx"], ar["cy"], (ar.get("cz", 0.0) if D == 3 else 0.0)],
        dtype=np.float32,
    )[:D]
    arena_r = float(ar["r"])

    x0 = np.asarray(cfg["x0"], dtype=np.float32)
    if x0.shape[0] < 1 + Na:
        raise ValueError(f"cfg['x0'] must contain defender plus {Na} attackers for num_attackers={Na}.")

    xD = np.asarray(x0[0, :nx], dtype=np.float32).copy()
    xA_list = [np.asarray(x0[i + 1, :nx], dtype=np.float32).copy() for i in range(Na)]

    deterministic = bool(cfg.get("rl_eval_deterministic", True))
    umax = float(cfg.get("umax", 5e-4))
    debug_actions = bool(cfg.get("debug_actions", False))
    stop_on_done = bool(cfg.get("stop_on_done", True))
    apply_velocity_cbf_filter = _build_velocity_cbf_filter(cfg, D=D, dt=dt, umax=umax)

    use_fuel = bool(cfg.get("fuel", {}).get("enable"))
    obs_expected = (2 + 3 * Na) * D + (2 if use_fuel else 0)
    g0 = 9.80665

    if use_fuel:
        fuel_cfg = cfg["fuel"]
        fdef = fuel_cfg["def"]
        fatt = fuel_cfg["att"]

        m0_def = float(fdef["m0"])
        mdry_def = float(fdef["m_dry"])
        Tmax_def = float(fdef["Tmax"])
        Isp_def = float(fdef["Isp"])

        m0_att = float(fatt["m0"])
        mdry_att = float(fatt["m_dry"])
        Tmax_att = float(fatt["Tmax"])
        Isp_att = float(fatt["Isp"])

        m_def = m0_def
        m_att = [m0_att for _ in range(Na)]
    else:
        m0_def = mdry_def = Tmax_def = Isp_def = None
        m0_att = mdry_att = Tmax_att = Isp_att = None
        m_def = None
        m_att = [None for _ in range(Na)]

    def apply_propulsion(
        a_cmd: np.ndarray,
        m: float,
        m_dry: float,
        Tmax: float,
        Isp: float,
    ):
        """Apply propulsion to the current config, state, or rollout data."""
        a_cmd = np.asarray(a_cmd, dtype=float)

        if m <= m_dry + 1e-9:
            return np.zeros_like(a_cmd, dtype=np.float32), float(m_dry), True, 0.0, 0.0

        F_req = m * a_cmd
        F_req_norm = np.linalg.norm(F_req)
        if F_req_norm > Tmax:
            F = F_req * (Tmax / (F_req_norm + 1e-9))
        else:
            F = F_req

        a_real = F / max(m, 1e-9)
        thrust_norm = float(np.linalg.norm(F))
        mdot = thrust_norm / (Isp * g0)
        m_next = max(m_dry, m - mdot * dt)
        fuel_depleted = bool(m_next <= m_dry + 1e-9)
        return a_real.astype(np.float32), float(m_next), fuel_depleted, thrust_norm, float(mdot)

    def build_train_obs(
        xD_vec: np.ndarray,
        xA_vecs: List[np.ndarray],
        m_def_cur: float | None,
        m_att_cur: List[float | None],
    ) -> np.ndarray:
        """Build train obs for the current workflow."""
        pD = xD_vec[:D]
        vD = xD_vec[D:2 * D]
        parts: List[np.ndarray] = [pD - center]

        for xA_vec in xA_vecs:
            parts.append(xA_vec[:D] - center)
        for xA_vec in xA_vecs:
            parts.append(xA_vec[:D] - pD)

        parts.append(vD)
        for xA_vec in xA_vecs:
            parts.append(xA_vec[D:2 * D])

        if use_fuel:
            fuel_frac_def = (m_def_cur - mdry_def) / (m0_def - mdry_def + 1e-9)
            fuel_frac_att = (m_att_cur[0] - mdry_att) / (m0_att - mdry_att + 1e-9)
            parts.append(np.array([np.clip(fuel_frac_def, 0.0, 1.0)], dtype=np.float32))
            parts.append(np.array([np.clip(fuel_frac_att, 0.0, 1.0)], dtype=np.float32))

        obs = np.concatenate(parts).astype(np.float32)
        return obs

    step_plant_single, center_from_dyn, _ = _build_step_plant_single(cfg, steps=steps, D=D)
    center = np.asarray(center_from_dyn, dtype=np.float32)
    arena_r = float(cfg.get("arena", {}).get("r", 1.0))

    use_kf = bool(cfg.get("use_kf", False))
    estimator_kind = _normalize_estimator_kind(cfg)
    if use_kf:
        print(
            f"[warn] {estimator_kind.upper()} eval path is only implemented for num_attackers == 1 "
            "in matchup_runner; disabling estimator."
        )
        use_kf = False

    pol = RLPolicy_Multi(
        cfg,
        device=cfg.get("device", "cpu"),
        def_ckpt=(cfg.get("def_ckpt_path") if policy_role == "def" else None),
        att_ckpt=(cfg.get("att_ckpt_path") if policy_role == "att" else None),
    )
    din_def, din_att = pol.verify_ckpt_compat()
    if policy_role == "def" and din_def != obs_expected:
        raise RuntimeError(
            f"Defender policy obs dim mismatch: got {din_def}, expected {obs_expected}."
        )
    if policy_role == "att" and din_att != obs_expected:
        raise RuntimeError(
            f"Attacker policy obs dim mismatch: got {din_att}, expected {obs_expected}."
        )

    ppo_obj = _populate_ppo_obj_from_cfg(cfg, _ppo_obj_params_from_cfg(cfg))
    baseline_role = "att" if policy_role == "def" else "def"
    if baseline_role == "def":
        u_baseline_prev: np.ndarray | List[np.ndarray] = np.zeros((D,), dtype=np.float32)
        active_u_baseline: np.ndarray | List[np.ndarray] = np.zeros((D,), dtype=np.float32)
    else:
        u_baseline_prev = [np.zeros((D,), dtype=np.float32) for _ in range(Na)]
        active_u_baseline = [u.copy() for u in u_baseline_prev]
    baseline_dbg_hist: List[Dict[str, Any]] = []

    oi = cfg.get("oi", {}) or {}
    oi_radius = float(oi.get("r", 0.0))
    oi_radius_norm = oi_radius / arena_r if arena_r > 0 else 0.0
    hit_buffer_def = 0.0
    hit_buffer_att = 0.0
    collision_radius_m = float(cfg.get("collision_radius_m"))
    arena_margin = float(cfg.get("arena_terminate_margin"))

    def check_done(
        xD_vec: np.ndarray,
        xA_vecs: List[np.ndarray],
        fuel_depleted_def: bool,
        fuel_depleted_att: List[bool],
    ):
        """Handle check done for this workflow."""
        pD = xD_vec[:D]
        pA_list = [xA[:D] for xA in xA_vecs]

        rhoD = np.linalg.norm(pD - center) / max(arena_r, 1e-9)
        rhoA = [np.linalg.norm(pA - center) / max(arena_r, 1e-9) for pA in pA_list]

        oob_def = rhoD >= arena_margin
        oob_att = [rho >= arena_margin for rho in rhoA]

        def_hit_target = False
        att_hit_target = [False for _ in range(Na)]
        if oi_radius_norm > 0.0:
            def_hit_target = rhoD <= (1.0 + hit_buffer_def) * oi_radius_norm
            att_hit_target = [rho <= (1.0 + hit_buffer_att) * oi_radius_norm for rho in rhoA]

        collision = [False for _ in range(Na)]
        if collision_radius_m > 0.0:
            collision = [np.linalg.norm(pA - pD) <= collision_radius_m for pA in pA_list]

        done = (
            oob_def
            or any(oob_att)
            or def_hit_target
            or any(att_hit_target)
            or any(collision)
            or (use_fuel and (fuel_depleted_def or any(fuel_depleted_att)))
        )

        reason = None
        attacker_idx = -1
        if done:
            if any(collision):
                attacker_idx = int(np.argmax(np.asarray(collision, dtype=int)))
                reason = "collision"
            elif any(att_hit_target):
                attacker_idx = int(np.argmax(np.asarray(att_hit_target, dtype=int)))
                reason = "attacker_hit_target"
            elif def_hit_target:
                reason = "defender_hit_target"
            elif oob_def:
                reason = "defender_oob"
            elif any(oob_att):
                attacker_idx = int(np.argmax(np.asarray(oob_att, dtype=int)))
                reason = "attacker_oob"
            elif use_fuel and fuel_depleted_def:
                reason = "defender_fuel_depleted"
            elif use_fuel and any(fuel_depleted_att):
                attacker_idx = int(np.argmax(np.asarray(fuel_depleted_att, dtype=int)))
                reason = "attacker_fuel_depleted"

        return done, {
            "oob_def": bool(oob_def),
            "oob_att": [bool(v) for v in oob_att],
            "oob_att_any": bool(any(oob_att)),
            "def_hit_target": bool(def_hit_target),
            "att_hit_target": [bool(v) for v in att_hit_target],
            "att_hit_target_any": bool(any(att_hit_target)),
            "collision": [bool(v) for v in collision],
            "collision_any": bool(any(collision)),
            "fuel_depleted_def": bool(fuel_depleted_def),
            "fuel_depleted_att": [bool(v) for v in fuel_depleted_att],
            "fuel_depleted_att_any": bool(any(fuel_depleted_att)),
            "done_reason": reason,
            "done_attacker_idx": int(attacker_idx),
        }

    plan_hist_all = [[] for _ in range(1 + Na)]
    plan_att_all = [[] for _ in range(1 + Na)]
    exec_xyz_all = [[] for _ in range(1 + Na)]
    vel_xyz_all = [[] for _ in range(1 + Na)]
    exec_att_all = [[] for _ in range(1 + Na)]
    phi_hist_all = [[] for _ in range(1 + Na)]
    fov_axis_hist, fov_seen_mask = [], []

    u_cmd_all = [[] for _ in range(1 + Na)]
    u_cmd_norm_all = [[] for _ in range(1 + Na)]
    u_real_all = [[] for _ in range(1 + Na)]
    u_real_norm_all = [[] for _ in range(1 + Na)]
    fuel_frac_all = [[] for _ in range(1 + Na)]
    thrust_all = [[] for _ in range(1 + Na)]
    mdot_all = [[] for _ in range(1 + Na)]
    done_info = None

    I = _identity_R()
    all_states = [xD] + xA_list
    for idx, xs in enumerate(all_states):
        exec_xyz_all[idx].append(_p3(xs, D))
        vel_xyz_all[idx].append(_v3_from_state(xs, D))
        exec_att_all[idx].append({"R": I, "phi": 0.0})
        phi_hist_all[idx].append(0.0)
    fov_axis_hist.append(None)
    fov_seen_mask.append(False)

    if use_fuel:
        fuel_frac_all[0].append(float(np.clip((m_def - mdry_def) / (m0_def - mdry_def + 1e-9), 0.0, 1.0)))
        for j in range(Na):
            fuel_frac_all[1 + j].append(float(np.clip((m_att[j] - mdry_att) / (m0_att - mdry_att + 1e-9), 0.0, 1.0)))

    t_roll0 = time.perf_counter()
    for k in range(steps):
        obs = build_train_obs(xD, xA_list, m_def, m_att)

        if policy_role == "def":
            u_policy_cmd = np.clip(
                np.asarray(pol.act_def_obs(obs, deterministic=deterministic), dtype=np.float32),
                -umax,
                +umax,
            )
        else:
            u_policy_cmd = [
                np.clip(
                    np.asarray(pol.act_att_obs(obs, deterministic=deterministic, attacker_idx=j), dtype=np.float32),
                    -umax,
                    +umax,
                )
                for j in range(Na)
            ]

        if _baseline_should_resolve(opponent_baseline, cfg, k, turn_len):
            u_baseline_cmd, dbg = _solve_baseline_best_response(
                cfg=cfg,
                opponent_baseline=opponent_baseline,
                baseline_role=baseline_role,
                xD=xD,
                xA_list=xA_list,
                fixed_u_policy=u_policy_cmd,
                u_baseline_prev=u_baseline_prev,
                k=k,
                step_plant_single=step_plant_single,
                center=center,
                arena_r=arena_r,
                ppo_obj=ppo_obj,
                umax=umax,
            )
            if baseline_role == "def":
                active_u_baseline = np.asarray(u_baseline_cmd, dtype=np.float32).copy()
                u_baseline_prev = active_u_baseline.copy()
            else:
                active_u_baseline = [u.copy() for u in _as_action_list(u_baseline_cmd, D, Na)]
                u_baseline_prev = [u.copy() for u in active_u_baseline]
        else:
            dbg = {
                "baseline": opponent_baseline,
                "baseline_role": baseline_role,
                "solver_family": "action_reuse",
                "reused_action": True,
                "num_attackers": int(Na),
            }

        baseline_dbg_hist.append(dbg)

        if policy_role == "def":
            uD_cmd = np.asarray(u_policy_cmd, dtype=np.float32)
            uA_cmd = [u.copy() for u in active_u_baseline]
        else:
            uD_cmd = np.asarray(active_u_baseline, dtype=np.float32)
            uA_cmd = [u.copy() for u in u_policy_cmd]
        uD_cmd = apply_velocity_cbf_filter(xD[:D], xD[D:2 * D], uD_cmd)
        uA_cmd = [
            apply_velocity_cbf_filter(xA_list[j][:D], xA_list[j][D:2 * D], uA_cmd[j])
            for j in range(Na)
        ]

        cmd_actions = [uD_cmd] + uA_cmd
        for idx, u_cmd in enumerate(cmd_actions):
            u3 = _pad3(u_cmd, D)
            u_cmd_all[idx].append(u3.copy())
            u_cmd_norm_all[idx].append(float(np.linalg.norm(u3)))

        if debug_actions and k < 3:
            att_norms = [float(np.linalg.norm(u)) for u in uA_cmd]
            print(
                f"[k={k:02d}] ||a_def_cmd||={np.linalg.norm(uD_cmd):.3g}, "
                f"att_cmd_norms={att_norms}, obs[:6]={obs[:6]}"
            )

        if use_fuel:
            uD_real, m_def, fuel_depleted_def, thrust_def, mdot_def = apply_propulsion(
                uD_cmd, m_def, mdry_def, Tmax_def, Isp_def
            )
            uA_real = []
            fuel_depleted_att = []
            thrust_att = []
            mdot_att = []
            for j in range(Na):
                u_real_j, m_att[j], fuel_j, thrust_j, mdot_j = apply_propulsion(
                    uA_cmd[j], m_att[j], mdry_att, Tmax_att, Isp_att
                )
                uA_real.append(u_real_j)
                fuel_depleted_att.append(bool(fuel_j))
                thrust_att.append(float(thrust_j))
                mdot_att.append(float(mdot_j))
        else:
            uD_real = uD_cmd
            uA_real = [u.copy() for u in uA_cmd]
            fuel_depleted_def = False
            fuel_depleted_att = [False for _ in range(Na)]
            thrust_def = 0.0
            mdot_def = 0.0
            thrust_att = [0.0 for _ in range(Na)]
            mdot_att = [0.0 for _ in range(Na)]

        real_actions = [uD_real] + uA_real
        for idx, u_real in enumerate(real_actions):
            u3 = _pad3(u_real, D)
            u_real_all[idx].append(u3.copy())
            u_real_norm_all[idx].append(float(np.linalg.norm(u3)))

        thrust_all[0].append(float(thrust_def))
        mdot_all[0].append(float(mdot_def))
        for j in range(Na):
            thrust_all[1 + j].append(float(thrust_att[j]))
            mdot_all[1 + j].append(float(mdot_att[j]))

        xD = step_plant_single(xD, uD_real, k)
        for j in range(Na):
            xA_list[j] = step_plant_single(xA_list[j], uA_real[j], k)

        if use_fuel:
            fuel_frac_all[0].append(float(np.clip((m_def - mdry_def) / (m0_def - mdry_def + 1e-9), 0.0, 1.0)))
            for j in range(Na):
                fuel_frac_all[1 + j].append(float(np.clip((m_att[j] - mdry_att) / (m0_att - mdry_att + 1e-9), 0.0, 1.0)))

        cur_states = [xD] + xA_list
        for idx, xs in enumerate(cur_states):
            plan_hist_all[idx].append([_p3(xs, D)] * T)
            plan_att_all[idx].append([{"R": I, "phi": 0.0} for _ in range(T)])
            exec_xyz_all[idx].append(_p3(xs, D))
            vel_xyz_all[idx].append(_v3_from_state(xs, D))
            exec_att_all[idx].append({"R": I, "phi": 0.0})
            phi_hist_all[idx].append(0.0)

        fov_axis_hist.append(None)
        fov_seen_mask.append(False)

        done, done_info = check_done(xD, xA_list, fuel_depleted_def, fuel_depleted_att)
        if done and stop_on_done:
            break

    t_fn1 = time.perf_counter()
    timing = {
        "setup": float(t_roll0 - t_fn0),
        "simulation": float(t_fn1 - t_roll0),
        "total": float(t_fn1 - t_fn0),
    }

    out = _pack_multi_agent_rollout(
        plan_hist_all=plan_hist_all,
        plan_att_all=plan_att_all,
        exec_xyz_all=exec_xyz_all,
        vel_xyz_all=vel_xyz_all,
        exec_att_all=exec_att_all,
        phi_hist_all=phi_hist_all,
        fov_axis_hist=fov_axis_hist,
        fov_seen_mask=fov_seen_mask,
        u_cmd_all=u_cmd_all,
        u_cmd_norm_all=u_cmd_norm_all,
        u_real_all=u_real_all,
        u_real_norm_all=u_real_norm_all,
        done_info=done_info,
        rollout_timing_sec=timing,
        fuel_frac_all=(fuel_frac_all if use_fuel else None),
        thrust_all=(thrust_all if use_fuel else None),
        mdot_all=(mdot_all if use_fuel else None),
    )
    out.update({
        "policy_role": policy_role,
        "opponent_baseline": opponent_baseline,
        "baseline_dbg_hist": baseline_dbg_hist,
    })
    return out


def run_rhc_with_policy_vs_baseline_collect_frames_3d(
    cfg: Dict[str, Any],
    *,
    policy_role: str,
    opponent_baseline: str,
    steps: int | None = None,
    turn_len: int | None = None,
) -> Dict[str, Any]:
    """Run rhc with policy vs baseline collect frames 3d and return rollout frames, controls, and summary data."""
    t_fn0 = time.perf_counter()

    policy_role = str(policy_role).lower()
    if policy_role not in ("def", "att"):
        raise ValueError(f"policy_role must be 'def' or 'att', got {policy_role!r}.")

    opponent_baseline = str(opponent_baseline).lower()
    if opponent_baseline not in SUPPORTED_BASELINE_OPPONENTS:
        raise ValueError(
            f"Unsupported opponent_baseline={opponent_baseline!r}; "
            f"expected one of {SUPPORTED_BASELINE_OPPONENTS}."
        )

    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx = 2 * D
    T = int(cfg["T"])
    dt = float(cfg["dt"])

    num_attackers = int(cfg.get("num_attackers", 1))
    if num_attackers > 1:
        raise NotImplementedError("The multi-attacker matchup path has been removed; use num_attackers=1.")

    if steps is None:
        steps = int(cfg.get("T_eval", cfg.get("T", cfg.get("steps", 60))))
    if turn_len is None:
        turn_len = 1
    turn_len = max(1, int(turn_len))

    ar = cfg.setdefault("arena", {})
    ar.setdefault("type", "sphere")
    ar.setdefault("cx", 0.0)
    ar.setdefault("cy", 0.0)
    ar.setdefault("cz", 0.0)
    ar.setdefault("r", 30.0)

    if ar["type"] != "sphere":
        raise ValueError("Only spherical arena is supported in this rollout helper.")

    center = np.array(
        [ar["cx"], ar["cy"], (ar.get("cz", 0.0) if D == 3 else 0.0)],
        dtype=np.float32,
    )[:D]
    arena_r = float(ar["r"])

    x0 = np.asarray(cfg["x0"], dtype=np.float32)
    if x0.shape[0] < 2:
        raise ValueError("cfg['x0'] must contain at least defender and attacker rows.")

    x1 = np.asarray(x0[0, :nx], dtype=np.float32).copy()
    x2 = np.asarray(x0[1, :nx], dtype=np.float32).copy()

    deterministic = bool(cfg.get("rl_eval_deterministic", True))
    umax = float(cfg.get("umax", 5e-4))
    debug_actions = bool(cfg.get("debug_actions", False))
    stop_on_done = bool(cfg.get("stop_on_done", True))
    apply_velocity_cbf_filter = _build_velocity_cbf_filter(cfg, D=D, dt=dt, umax=umax)

    use_fuel = bool(cfg.get("fuel", {}).get("enable"))
    obs_expected = 5 * D + (2 if use_fuel else 0)
    g0 = 9.80665

    if use_fuel:
        fuel_cfg = cfg["fuel"]
        fdef = fuel_cfg["def"]
        fatt = fuel_cfg["att"]

        m0_def = float(fdef["m0"])
        mdry_def = float(fdef["m_dry"])
        Tmax_def = float(fdef["Tmax"])
        Isp_def = float(fdef["Isp"])

        m0_att = float(fatt["m0"])
        mdry_att = float(fatt["m_dry"])
        Tmax_att = float(fatt["Tmax"])
        Isp_att = float(fatt["Isp"])

        m_def = m0_def
        m_att = m0_att
    else:
        m0_def = mdry_def = Tmax_def = Isp_def = None
        m0_att = mdry_att = Tmax_att = Isp_att = None
        m_def = None
        m_att = None

    def apply_propulsion(
        a_cmd: np.ndarray,
        m: float,
        m_dry: float,
        Tmax: float,
        Isp: float,
    ):
        """Apply propulsion to the current config, state, or rollout data."""
        a_cmd = np.asarray(a_cmd, dtype=float)

        if m <= m_dry + 1e-9:
            return np.zeros_like(a_cmd, dtype=np.float32), float(m_dry), True, 0.0, 0.0

        F_req = m * a_cmd
        F_req_norm = np.linalg.norm(F_req)

        if F_req_norm > Tmax:
            F = F_req * (Tmax / (F_req_norm + 1e-9))
        else:
            F = F_req

        a_real = F / max(m, 1e-9)

        thrust_norm = float(np.linalg.norm(F))
        mdot = thrust_norm / (Isp * g0)
        m_next = max(m_dry, m - mdot * dt)
        fuel_depleted = bool(m_next <= m_dry + 1e-9)

        return a_real.astype(np.float32), float(m_next), fuel_depleted, thrust_norm, float(mdot)

    def build_train_obs(
        x1_vec: np.ndarray,
        x2_vec: np.ndarray,
        m_def_cur: float | None,
        m_att_cur: float | None,
    ) -> np.ndarray:
        """Build train obs for the current workflow."""
        p1 = x1_vec[:D]
        v1 = x1_vec[D:2 * D]
        p2 = x2_vec[:D]
        v2 = x2_vec[D:2 * D]

        parts = [
            p1 - center,
            p2 - center,
            p2 - p1,
            v1,
            v2,
        ]

        if use_fuel:
            fuel_frac_def = (m_def_cur - mdry_def) / (m0_def - mdry_def + 1e-9)
            fuel_frac_att = (m_att_cur - mdry_att) / (m0_att - mdry_att + 1e-9)
            parts.append(np.array([np.clip(fuel_frac_def, 0.0, 1.0)], dtype=np.float32))
            parts.append(np.array([np.clip(fuel_frac_att, 0.0, 1.0)], dtype=np.float32))

        obs = np.concatenate(parts).astype(np.float32)
        return obs

    step_plant_single, center_from_dyn, _ = _build_step_plant_single(cfg, steps=steps, D=D)
    center = np.asarray(center_from_dyn, dtype=np.float32)
    arena_r = float(cfg.get("arena", {}).get("r", 1.0))

    def _u3(uD: np.ndarray):
        """Internal helper for u3."""
        uD = np.asarray(uD, float).reshape(-1)
        if D == 3:
            return uD[:3]
        return np.array([uD[0], uD[1], 0.0], dtype=float)

    def _x6_from_xD(xD: np.ndarray) -> np.ndarray:
        """Internal helper for x6 from x d."""
        xD = np.asarray(xD, dtype=np.float32).reshape(-1)
        if D == 3:
            return xD.astype(np.float32)
        return np.array([xD[0], xD[1], 0.0, xD[2], xD[3], 0.0], dtype=np.float32)

    def _xD_from_x6(x6: np.ndarray) -> np.ndarray:
        """Internal helper for x d from x6."""
        x6 = np.asarray(x6, dtype=np.float32).reshape(-1)
        if D == 3:
            return x6.astype(np.float32)
        return np.array([x6[0], x6[1], x6[3], x6[4]], dtype=np.float32)

    use_kf = bool(cfg.get("use_kf", False))
    estimator_kind = _normalize_estimator_kind(cfg)
    dyn_name = str(cfg.get("dynamics", "hcw")).lower()
    if use_kf and dyn_name != "hcw":
        print(f"[warn] {estimator_kind.upper()} in this runner is HCW-only; disabling estimator for non-HCW dynamics.")
        use_kf = False

    # Estimator state is optional and strictly an observation source; the plant
    # still advances with the true simulated states below.
    est_hist = None
    x2_est = x2.copy()
    x1_est = x1.copy()

    if use_kf:
        ukf_cfg = dict(cfg.get("ukf", {}))
        sigma_az = float(ukf_cfg.get("sigma_az", np.deg2rad(0.5)))
        sigma_el = float(ukf_cfg.get("sigma_el", np.deg2rad(0.5)))
        ukf_action_access = _normalize_kf_action_access(ukf_cfg.get("action_access", "ground_truth"))
        ukf_control_meas_std = _normalize_kf_control_noise_std(
            ukf_cfg.get("action_meas_std", 0.1 * max(abs(umax), 1e-3)),
            default_std=0.1 * max(abs(umax), 1e-3),
        )
        ukf_state_dim = 6
        pos_std0 = float(ukf_cfg.get("pos_std0", 0.2 * arena_r))
        vel_std0 = float(ukf_cfg.get("vel_std0", 0.01))
        init_mean_pos_std = float(ukf_cfg.get("init_mean_pos_std", 0.0))
        init_mean_vel_std = float(ukf_cfg.get("init_mean_vel_std", 0.0))
        init_mean_accel_std = float(ukf_cfg.get("init_mean_accel_std", 0.0))
        Q_scale = float(ukf_cfg.get("Q_scale", 1e-5))
        accel_std0 = float(ukf_cfg.get("accel_std0", max(abs(umax), 1e-3)))
        accel_q_scale = float(ukf_cfg.get("accel_Q_scale", Q_scale))

        who = str(ukf_cfg.get("who", "both")).lower()
        meas_every = int(ukf_cfg.get("every", 1))

        if ukf_state_dim == 9:
            P0 = np.diag([pos_std0**2] * 3 + [vel_std0**2] * 3 + [accel_std0**2] * 3)
            Q = np.diag([Q_scale] * 6 + [accel_q_scale] * 3)
        else:
            P0 = np.diag([pos_std0**2] * 3 + [vel_std0**2] * 3)
            Q = Q_scale * np.diag([1, 1, 1, 1, 1, 1])
        Rm = np.diag([sigma_az**2, sigma_el**2])
        hcw_params = cfg.get("hcw", {})

        x2_6 = _x6_from_xD(x2)
        x1_6 = _x6_from_xD(x1)

        x2_6_mean = x2_6.copy()
        x1_6_mean = x1_6.copy()
        x2_6_mean[:3] += np.random.normal(0.0, init_mean_pos_std, size=3)
        x2_6_mean[3:6] += np.random.normal(0.0, init_mean_vel_std, size=3)
        x1_6_mean[:3] += np.random.normal(0.0, init_mean_pos_std, size=3)
        x1_6_mean[3:6] += np.random.normal(0.0, init_mean_vel_std, size=3)
        x2_ukf0 = x2_6_mean
        x1_ukf0 = x1_6_mean

        ukf12 = (
            KF_CV(
                x2_ukf0,
                P0,
                Q,
                Rm,
                dt,
                kind=estimator_kind,
                dyn="hcw",
                hcw=hcw_params,
                **_ekf_factory_kwargs(cfg, linearization_group="observer_1_to_2"),
            )
            if who in ("both", "1->2")
            else None
        )
        ukf21 = (
            KF_CV(
                x1_ukf0,
                P0,
                Q,
                Rm,
                dt,
                kind=estimator_kind,
                dyn="hcw",
                hcw=hcw_params,
                **_ekf_factory_kwargs(cfg, linearization_group="observer_2_to_1"),
            )
            if who in ("both", "2->1")
            else None
        )

        est_hist = {}
        if ukf12 is not None:
            est_hist["est12_xyz"] = [ukf12.x[:3].copy()]
            est_hist["meas12_azel"] = [None]
            est_hist["meas12_innov_sq"] = [float("nan")]
            est_hist["trP12_pos"] = [float(np.trace(ukf12.P[:3, :3]))]
            x2_est = _xD_from_x6(np.r_[ukf12.x[:3], ukf12.x[3:6]]).astype(np.float32)

        if ukf21 is not None:
            est_hist["est21_xyz"] = [ukf21.x[:3].copy()]
            est_hist["meas21_azel"] = [None]
            est_hist["meas21_innov_sq"] = [float("nan")]
            est_hist["trP21_pos"] = [float(np.trace(ukf21.P[:3, :3]))]
            x1_est = _xD_from_x6(np.r_[ukf21.x[:3], ukf21.x[3:6]]).astype(np.float32)

        meas_std_az, meas_std_el = sigma_az, sigma_el
    else:
        ukf12 = ukf21 = None
        ukf_action_access = "ground_truth"
        meas_every = 1
        meas_std_az = meas_std_el = None

    def build_student_sigma_feat():
        """Build student sigma feat for the current workflow."""
        if use_kf and (ukf12 is not None):
            P = np.asarray(ukf12.P, dtype=np.float32)
            P_rel = P[: 2 * D, : 2 * D]
            iu = np.triu_indices(2 * D)
            return P_rel[iu].astype(np.float32)
        return None

    pol = RLPolicyDiff(
        cfg,
        device=cfg.get("device", "cpu"),
        load_defender=(policy_role == "def"),
        load_attacker=(policy_role == "att"),
    )
    din_def, din_att = pol.verify_ckpt_compat()
    if policy_role == "def" and din_def != obs_expected:
        raise RuntimeError(
            f"Defender policy obs dim mismatch: got {din_def}, expected {obs_expected}."
        )
    if policy_role == "att" and din_att != obs_expected:
        raise RuntimeError(
            f"Attacker policy obs dim mismatch: got {din_att}, expected {obs_expected}."
        )

    ppo_obj = _populate_ppo_obj_from_cfg(cfg, _ppo_obj_params_from_cfg(cfg))
    baseline_role = "att" if policy_role == "def" else "def"
    u_baseline_prev = np.zeros((D,), dtype=np.float32)
    active_u_baseline = u_baseline_prev.copy()
    baseline_dbg_hist: List[Dict[str, Any]] = []

    oi = cfg.get("oi", {}) or {}
    oi_radius = float(oi.get("r", 0.0))
    oi_radius_norm = oi_radius / arena_r if arena_r > 0 else 0.0
    hit_buffer_def = 0.0
    hit_buffer_att = 0.0
    collision_radius_m = float(cfg.get("collision_radius_m"))
    arena_margin = float(cfg.get("arena_terminate_margin"))

    def check_done(
        x1_vec: np.ndarray,
        x2_vec: np.ndarray,
        fuel_depleted_def: bool,
        fuel_depleted_att: bool,
    ):
        """Handle check done for this workflow."""
        p1 = x1_vec[:D]
        p2 = x2_vec[:D]

        rho1 = np.linalg.norm(p1 - center) / max(arena_r, 1e-9)
        rho2 = np.linalg.norm(p2 - center) / max(arena_r, 1e-9)

        oob_def = rho1 >= arena_margin
        oob_att = rho2 >= arena_margin

        att_hit_target = False
        def_hit_target = False
        if oi_radius_norm > 0.0:
            def_hit_target = rho1 <= (1.0 + hit_buffer_def) * oi_radius_norm
            att_hit_target = rho2 <= (1.0 + hit_buffer_att) * oi_radius_norm

        collision = False
        if collision_radius_m > 0.0:
            collision = np.linalg.norm(p2 - p1) <= collision_radius_m

        done = (
            oob_def
            or oob_att
            or def_hit_target
            or att_hit_target
            or collision
            or (use_fuel and (fuel_depleted_def or fuel_depleted_att))
        )

        reason = None
        if done:
            if collision:
                reason = "collision"
            elif att_hit_target:
                reason = "attacker_hit_target"
            elif def_hit_target:
                reason = "defender_hit_target"
            elif oob_def:
                reason = "defender_oob"
            elif oob_att:
                reason = "attacker_oob"
            elif use_fuel and fuel_depleted_def:
                reason = "defender_fuel_depleted"
            elif use_fuel and fuel_depleted_att:
                reason = "attacker_fuel_depleted"

        return done, {
            "oob_def": bool(oob_def),
            "oob_att": bool(oob_att),
            "def_hit_target": bool(def_hit_target),
            "att_hit_target": bool(att_hit_target),
            "collision": bool(collision),
            "fuel_depleted_def": bool(fuel_depleted_def),
            "fuel_depleted_att": bool(fuel_depleted_att),
            "done_reason": reason,
        }

    plan_hist1, plan_hist2 = [], []
    plan_att1, plan_att2 = [], []
    exec_xyz1, exec_xyz2 = [], []
    vel_xyz1, vel_xyz2 = [], []
    exec_att1, exec_att2 = [], []
    phi_hist1, phi_hist2 = [], []
    fov_axis_hist, fov_seen_mask = [], []

    u_cmd_1, u_cmd_2 = [], []
    u_cmd_norm_1, u_cmd_norm_2 = [], []
    u_real_1, u_real_2 = [], []
    u_real_norm_1, u_real_norm_2 = [], []
    fuel_frac_def_hist, fuel_frac_att_hist = [], []
    thrust_def_hist, thrust_att_hist = [], []
    mdot_def_hist, mdot_att_hist = [], []
    done_info = None

    exec_xyz1.append(_p3(x1, D))
    exec_xyz2.append(_p3(x2, D))
    vel_xyz1.append(_v3_from_state(x1, D))
    vel_xyz2.append(_v3_from_state(x2, D))
    exec_att1.append({"R": _identity_R(), "phi": 0.0})
    exec_att2.append({"R": _identity_R(), "phi": 0.0})
    phi_hist1.append(0.0)
    phi_hist2.append(0.0)
    fov_axis_hist.append(None)
    fov_seen_mask.append(False)

    if use_fuel:
        fuel_frac_def_hist.append(float(np.clip((m_def - mdry_def) / (m0_def - mdry_def + 1e-9), 0.0, 1.0)))
        fuel_frac_att_hist.append(float(np.clip((m_att - mdry_att) / (m0_att - mdry_att + 1e-9), 0.0, 1.0)))

    t_roll0 = time.perf_counter()
    for k in range(steps):
        # Each side can observe either truth or the latest estimator state, depending
        # on the KF settings chosen for this rollout.
        x2_for_def_obs = x2_est if (use_kf and ukf12 is not None) else x2
        x1_for_att_obs = x1_est if (use_kf and ukf21 is not None) else x1
        obs_def = build_train_obs(x1, x2_for_def_obs, m_def, m_att)
        obs_att = build_train_obs(x1_for_att_obs, x2, m_def, m_att)
        sigma_feat = build_student_sigma_feat()

        if policy_role == "def":
            u_policy_cmd = np.clip(
                np.asarray(
                    pol.act_def_obs(obs_def, deterministic=deterministic, sigma_feat=sigma_feat),
                    dtype=np.float32,
                ),
                -umax,
                +umax,
            )
        else:
            u_policy_cmd = np.clip(
                np.asarray(
                    pol.act_att_obs(obs_att, deterministic=deterministic, sigma_feat=sigma_feat),
                    dtype=np.float32,
                ),
                -umax,
                +umax,
            )

        if _baseline_should_resolve(opponent_baseline, cfg, k, turn_len):
            # The baseline always responds to the current policy action, so the
            # matchup is policy-vs-best-response rather than two independent rollouts.
            u_baseline_cmd, dbg = _solve_baseline_best_response(
                cfg=cfg,
                opponent_baseline=opponent_baseline,
                baseline_role=baseline_role,
                xD=x1,
                xA_list=[x2],
                fixed_u_policy=u_policy_cmd,
                u_baseline_prev=u_baseline_prev,
                k=k,
                step_plant_single=step_plant_single,
                center=center,
                arena_r=arena_r,
                ppo_obj=ppo_obj,
                umax=umax,
            )
            active_u_baseline = np.asarray(u_baseline_cmd, dtype=np.float32).copy()
            u_baseline_prev = active_u_baseline.copy()
        else:
            dbg = {
                "baseline": opponent_baseline,
                "baseline_role": baseline_role,
                "solver_family": "action_reuse",
                "reused_action": True,
            }

        baseline_dbg_hist.append(dbg)

        if policy_role == "def":
            u1_cmd = u_policy_cmd
            u2_cmd = active_u_baseline.copy()
        else:
            u1_cmd = active_u_baseline.copy()
            u2_cmd = u_policy_cmd
        u1_cmd = apply_velocity_cbf_filter(x1[:D], x1[D:2 * D], u1_cmd)
        u2_cmd = apply_velocity_cbf_filter(x2[:D], x2[D:2 * D], u2_cmd)

        u1_cmd_3 = _u3(u1_cmd)
        u2_cmd_3 = _u3(u2_cmd)
        u_cmd_1.append(u1_cmd_3.copy())
        u_cmd_2.append(u2_cmd_3.copy())
        u_cmd_norm_1.append(float(np.linalg.norm(u1_cmd_3)))
        u_cmd_norm_2.append(float(np.linalg.norm(u2_cmd_3)))

        if debug_actions and k < 3:
            print(
                f"[k={k:02d}] ||a_def_cmd||={np.linalg.norm(u1_cmd):.3g}, "
                f"||a_att_cmd||={np.linalg.norm(u2_cmd):.3g}, "
                f"obs_def[:6]={obs_def[:6]}, obs_att[:6]={obs_att[:6]}"
            )

        if use_fuel:
            u1_real, m_def, fuel_depleted_def, thrust_def, mdot_def = apply_propulsion(
                u1_cmd, m_def, mdry_def, Tmax_def, Isp_def
            )
            u2_real, m_att, fuel_depleted_att, thrust_att, mdot_att = apply_propulsion(
                u2_cmd, m_att, mdry_att, Tmax_att, Isp_att
            )
        else:
            u1_real = u1_cmd
            u2_real = u2_cmd
            fuel_depleted_def = False
            fuel_depleted_att = False
            thrust_def = thrust_att = 0.0
            mdot_def = mdot_att = 0.0

        u1_real_3 = _u3(u1_real)
        u2_real_3 = _u3(u2_real)
        u_real_1.append(u1_real_3.copy())
        u_real_2.append(u2_real_3.copy())
        u_real_norm_1.append(float(np.linalg.norm(u1_real_3)))
        u_real_norm_2.append(float(np.linalg.norm(u2_real_3)))

        thrust_def_hist.append(float(thrust_def))
        thrust_att_hist.append(float(thrust_att))
        mdot_def_hist.append(float(mdot_def))
        mdot_att_hist.append(float(mdot_att))

        x1 = step_plant_single(x1, u1_real, k)
        x2 = step_plant_single(x2, u2_real, k)

        if use_kf and est_hist is not None:
            idx = len(exec_xyz1)
            take = (idx % meas_every) == 0

            def _to3_pos(xD_local):
                """Internal helper for to3 pos."""
                p = np.zeros(3, dtype=float)
                p[:D] = np.asarray(xD_local[:D], float)
                return p

            if ukf12 is not None:
                # Prediction can consume true, noisy, or no opponent control depending
                # on ukf.action_access; measurement is az/el only.
                u12_pred, u12_cov = _kf_predict_control(
                    ukf_action_access,
                    _u3(u2_real),
                    control_meas_std=ukf_control_meas_std,
                    control_limit=umax,
                )
                ukf12.predict(dt, u=u12_pred, u_cov=u12_cov)
                innov_sq = float("nan")
                if take:
                    p_obs = _to3_pos(x1)
                    p_tgt = _to3_pos(x2)
                    d = p_tgt - p_obs
                    d /= (np.linalg.norm(d) + 1e-12)
                    az = np.arctan2(d[1], d[0]) + np.random.randn() * meas_std_az
                    el = np.arctan2(d[2], np.sqrt(max(d[0] ** 2 + d[1] ** 2, 1e-18))) + np.random.randn() * meas_std_el
                    z = np.array([az, el], float)
                    z_hat = ukf12.h(ukf12.x.copy(), p_obs=p_obs, R_wb=_identity_R())
                    innov = z - z_hat
                    innov[0] = _normalize_angle(innov[0])
                    innov[1] = _normalize_angle(innov[1])
                    innov_sq = float(innov @ innov)
                    ukf12.update(z, p_obs=p_obs, R_wb=_identity_R())
                    est_hist["meas12_azel"].append((p_obs.copy(), z.copy()))
                else:
                    est_hist["meas12_azel"].append(None)
                est_hist["meas12_innov_sq"].append(innov_sq)
                est_hist["trP12_pos"].append(float(np.trace(ukf12.P[:3, :3])))
                est_hist["est12_xyz"].append(ukf12.x[:3].copy())

            if ukf21 is not None:
                u21_pred, u21_cov = _kf_predict_control(
                    ukf_action_access,
                    _u3(u1_real),
                    control_meas_std=ukf_control_meas_std,
                    control_limit=umax,
                )
                ukf21.predict(dt, u=u21_pred, u_cov=u21_cov)
                innov_sq = float("nan")
                if take:
                    p_obs = _to3_pos(x2)
                    p_tgt = _to3_pos(x1)
                    d = p_tgt - p_obs
                    d /= (np.linalg.norm(d) + 1e-12)
                    az = np.arctan2(d[1], d[0]) + np.random.randn() * meas_std_az
                    el = np.arctan2(d[2], np.sqrt(max(d[0] ** 2 + d[1] ** 2, 1e-18))) + np.random.randn() * meas_std_el
                    z = np.array([az, el], float)
                    z_hat = ukf21.h(ukf21.x.copy(), p_obs=p_obs, R_wb=_identity_R())
                    innov = z - z_hat
                    innov[0] = _normalize_angle(innov[0])
                    innov[1] = _normalize_angle(innov[1])
                    innov_sq = float(innov @ innov)
                    ukf21.update(z, p_obs=p_obs, R_wb=_identity_R())
                    est_hist["meas21_azel"].append((p_obs.copy(), z.copy()))
                else:
                    est_hist["meas21_azel"].append(None)
                est_hist["meas21_innov_sq"].append(innov_sq)
                est_hist["trP21_pos"].append(float(np.trace(ukf21.P[:3, :3])))
                est_hist["est21_xyz"].append(ukf21.x[:3].copy())

            if ukf12 is not None:
                x2_est = _xD_from_x6(np.r_[ukf12.x[:3], ukf12.x[3:6]]).astype(np.float32)
            if ukf21 is not None:
                x1_est = _xD_from_x6(np.r_[ukf21.x[:3], ukf21.x[3:6]]).astype(np.float32)

        if use_fuel:
            fuel_frac_def_hist.append(
                float(np.clip((m_def - mdry_def) / (m0_def - mdry_def + 1e-9), 0.0, 1.0))
            )
            fuel_frac_att_hist.append(
                float(np.clip((m_att - mdry_att) / (m0_att - mdry_att + 1e-9), 0.0, 1.0))
            )

        plan1 = [_p3(x1, D)] * T
        plan2 = [_p3(x2, D)] * T
        plan_hist1.append(plan1)
        plan_hist2.append(plan2)

        I = _identity_R()
        att_stub = [{"R": I, "phi": 0.0} for _ in range(T)]
        plan_att1.append(att_stub)
        plan_att2.append(att_stub)

        exec_att1.append({"R": I, "phi": 0.0})
        exec_att2.append({"R": I, "phi": 0.0})
        phi_hist1.append(0.0)
        phi_hist2.append(0.0)
        exec_xyz1.append(_p3(x1, D))
        exec_xyz2.append(_p3(x2, D))
        vel_xyz1.append(_v3_from_state(x1, D))
        vel_xyz2.append(_v3_from_state(x2, D))
        fov_axis_hist.append(None)
        fov_seen_mask.append(False)

        done, done_info = check_done(x1, x2, fuel_depleted_def, fuel_depleted_att)
        if done and stop_on_done:
            break

    t_fn1 = time.perf_counter()
    timing = {
        "setup": float(t_roll0 - t_fn0),
        "simulation": float(t_fn1 - t_roll0),
        "total": float(t_fn1 - t_fn0),
    }

    out = {
        "plan_hist1": plan_hist1,
        "plan_hist2": plan_hist2,
        "plan_att1": plan_att1,
        "plan_att2": plan_att2,
        "exec1_xyz": exec_xyz1,
        "exec2_xyz": exec_xyz2,
        "vel1_xyz": np.asarray(vel_xyz1, dtype=float),
        "vel2_xyz": np.asarray(vel_xyz2, dtype=float),
        "vel_xyz_all": [
            np.asarray(vel_xyz1, dtype=float),
            np.asarray(vel_xyz2, dtype=float),
        ],
        "exec_att1": exec_att1,
        "exec_att2": exec_att2,
        "phi_hist1": phi_hist1,
        "phi_hist2": phi_hist2,
        "fov_axis_hist": fov_axis_hist,
        "fov_seen_mask": fov_seen_mask,
        "u_cmd_all": [
            np.asarray(u_cmd_1, dtype=float),
            np.asarray(u_cmd_2, dtype=float),
        ],
        "u_cmd_norm_all": [
            np.asarray(u_cmd_norm_1, dtype=float),
            np.asarray(u_cmd_norm_2, dtype=float),
        ],
        "u_real_all": [
            np.asarray(u_real_1, dtype=float),
            np.asarray(u_real_2, dtype=float),
        ],
        "u_real_norm_all": [
            np.asarray(u_real_norm_1, dtype=float),
            np.asarray(u_real_norm_2, dtype=float),
        ],
        "done_info": done_info,
        "policy_role": policy_role,
        "opponent_baseline": opponent_baseline,
        "baseline_dbg_hist": baseline_dbg_hist,
        "rollout_timing_sec": timing,
    }

    if use_fuel:
        out.update(
            {
                "fuel_frac_all": [
                    np.asarray(fuel_frac_def_hist, dtype=float),
                    np.asarray(fuel_frac_att_hist, dtype=float),
                ],
                "thrust_all": [
                    np.asarray(thrust_def_hist, dtype=float),
                    np.asarray(thrust_att_hist, dtype=float),
                ],
                "mdot_all": [
                    np.asarray(mdot_def_hist, dtype=float),
                    np.asarray(mdot_att_hist, dtype=float),
                ],
            }
        )

    if est_hist is not None:
        out.update(est_hist)

    return out


__all__ = [
    "SUPPORTED_BASELINE_OPPONENTS",
    "run_rhc_with_policy_vs_baseline_collect_frames_3d",
]
