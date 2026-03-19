from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from paper_baseline_runner import (
    _build_step_plant_single,
    _identity_R,
    _p3,
    _pad3,
    _populate_ppo_obj_from_cfg,
    _ppo_obj_params_from_cfg,
    _ppo_security_value_g,
    _ppo_terminal_adjustment,
)


@dataclass
class IPOPTBaselineParams:
    br_iters: int = 12
    subproblem_max_iter: int = 100
    tol: float = 1e-4
    fd_eps: float = 1e-6
    reg_prev: float = 1e-3
    reuse_action_for_turn_len: bool = True


def _ipopt_params_from_cfg(cfg: Dict[str, Any]) -> IPOPTBaselineParams:
    p = IPOPTBaselineParams()
    d = cfg.get("ipopt_baseline", {}) or {}
    for k, v in d.items():
        if hasattr(p, k):
            setattr(p, k, v)
    return p


def _project_box(u: np.ndarray, umax: float) -> np.ndarray:
    return np.clip(np.asarray(u, dtype=float), -float(umax), float(umax))


def _flatten_actions(u_list: List[np.ndarray]) -> np.ndarray:
    if not u_list:
        return np.zeros((0,), dtype=float)
    return np.concatenate([np.asarray(u, dtype=float).reshape(-1) for u in u_list], axis=0)


def _unflatten_actions(vec: np.ndarray, D: int, n_players: int) -> List[np.ndarray]:
    arr = np.asarray(vec, dtype=float).reshape(n_players, D)
    return [arr[i].astype(np.float32) for i in range(n_players)]


def _finite_diff_grad(fun, x: np.ndarray, eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    grad = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (float(fun(xp)) - float(fun(xm))) / (2.0 * eps)
    return grad


def _solve_box_subproblem(
    *,
    fun,
    x0: np.ndarray,
    umax: float,
    params: IPOPTBaselineParams,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    x0 = _project_box(x0, umax).astype(float)
    bounds = [(-float(umax), float(umax))] * x0.size
    last_err: str | None = None

    try:
        from cyipopt import minimize_ipopt  # type: ignore

        res = minimize_ipopt(
            fun=lambda z: float(fun(z)),
            x0=x0,
            jac=lambda z: _finite_diff_grad(fun, z, params.fd_eps),
            bounds=bounds,
            options={
                "max_iter": int(params.subproblem_max_iter),
                "tol": float(params.tol),
                "print_level": 0,
            },
        )
        x_opt = _project_box(np.asarray(res.x, dtype=float), umax)
        return x_opt, {
            "backend": "cyipopt",
            "success": bool(getattr(res, "success", True)),
            "status": getattr(res, "status", None),
            "nit": int(getattr(res, "nit", -1)),
        }
    except Exception as e:
        last_err = str(e)

    try:
        from scipy.optimize import minimize  # type: ignore

        res = minimize(
            fun=lambda z: float(fun(z)),
            x0=x0,
            jac=lambda z: _finite_diff_grad(fun, z, params.fd_eps),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(params.subproblem_max_iter), "ftol": float(params.tol)},
        )
        x_opt = _project_box(np.asarray(res.x, dtype=float), umax)
        return x_opt, {
            "backend": "scipy_lbfgsb",
            "success": bool(res.success),
            "status": int(getattr(res, "status", -1)),
            "nit": int(getattr(res, "nit", -1)),
            "fallback_reason": last_err,
        }
    except Exception as e:
        return x0, {
            "backend": "none",
            "success": False,
            "status": None,
            "nit": 0,
            "fallback_reason": last_err or str(e),
        }


def solve_one_step_ipopt_team_ppo_zero_sum(
    *,
    cfg: Dict[str, Any],
    xD: np.ndarray,
    xA_list: List[np.ndarray],
    uD_prev: np.ndarray,
    uA_prev: List[np.ndarray],
    k: int,
    step_plant_single,
    center: np.ndarray,
    R: float,
    ppo_params,
    umax: float,
) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, Any]]:
    params = _ipopt_params_from_cfg(cfg)
    D = int(np.asarray(xD).size // 2)
    nA = len(xA_list)

    uD = _project_box(np.asarray(uD_prev, dtype=float).reshape(-1), umax)
    uA_vec = _project_box(_flatten_actions(uA_prev), umax)

    nit_def_hist: List[int] = []
    nit_att_hist: List[int] = []
    backend_hist: List[str] = []

    for _ in range(max(1, int(params.br_iters))):
        uA_list_cur = _unflatten_actions(uA_vec, D, nA)

        def att_obj(z: np.ndarray) -> float:
            z = _project_box(z, umax)
            uA_list_z = _unflatten_actions(z, D, nA)
            g = _ppo_security_value_g(
                xD=xD,
                xA_list=xA_list,
                uD=uD,
                uA_list=uA_list_z,
                k=k,
                step_plant_single=step_plant_single,
                center=center,
                R=R,
                params=ppo_params,
                umax=umax,
            )
            reg = float(params.reg_prev) * float(np.dot(z - _flatten_actions(uA_prev), z - _flatten_actions(uA_prev)))
            return float(g + reg)

        uA_new, att_info = _solve_box_subproblem(fun=att_obj, x0=uA_vec, umax=umax, params=params)

        def def_obj(z: np.ndarray) -> float:
            z = _project_box(z, umax)
            g = _ppo_security_value_g(
                xD=xD,
                xA_list=xA_list,
                uD=z,
                uA_list=_unflatten_actions(uA_new, D, nA),
                k=k,
                step_plant_single=step_plant_single,
                center=center,
                R=R,
                params=ppo_params,
                umax=umax,
            )
            reg = float(params.reg_prev) * float(np.dot(z - np.asarray(uD_prev, dtype=float), z - np.asarray(uD_prev, dtype=float)))
            return float(-g + reg)

        uD_new, def_info = _solve_box_subproblem(fun=def_obj, x0=uD, umax=umax, params=params)

        step_delta = max(
            float(np.linalg.norm(uD_new - uD)),
            float(np.linalg.norm(uA_new - uA_vec)),
        )
        uD = uD_new
        uA_vec = uA_new
        nit_def_hist.append(int(def_info.get("nit", 0)))
        nit_att_hist.append(int(att_info.get("nit", 0)))
        backend_hist.extend([str(att_info.get("backend", "unknown")), str(def_info.get("backend", "unknown"))])
        if step_delta <= float(params.tol):
            break

    uA_star = _unflatten_actions(uA_vec, D, nA)
    g_star = _ppo_security_value_g(
        xD=xD,
        xA_list=xA_list,
        uD=uD,
        uA_list=uA_star,
        k=k,
        step_plant_single=step_plant_single,
        center=center,
        R=R,
        params=ppo_params,
        umax=umax,
    )
    dbg = {
        "objective": "ppo_zero_sum",
        "solver_family": "ipopt_best_response",
        "g_star": float(g_star),
        "def_backend_hist": backend_hist[1::2],
        "att_backend_hist": backend_hist[0::2],
        "def_nit_hist": nit_def_hist,
        "att_nit_hist": nit_att_hist,
        "br_iters_used": int(len(nit_def_hist)),
    }
    return uD.astype(np.float32), uA_star, dbg


def _run_rollout(
    cfg: Dict[str, Any],
    n_attackers: int,
    steps: int | None,
    turn_len: int | None,
) -> Dict[str, Any]:
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx = 2 * D
    T_horizon = int(cfg["T"])

    if steps is None:
        steps = int(cfg.get("T_eval", cfg.get("T", cfg.get("steps", 60))))

    solver_params = _ipopt_params_from_cfg(cfg)
    if turn_len is None:
        turn_len = 1
    turn_len = max(1, int(turn_len))

    umax = float(cfg.get("umax", 5e-4))
    step_plant_single, center, _ = _build_step_plant_single(cfg, steps=steps, D=D)
    R = float(cfg.get("arena", {}).get("r", 30.0))
    ppo_obj = _populate_ppo_obj_from_cfg(cfg, _ppo_obj_params_from_cfg(cfg))

    x0 = np.asarray(cfg["x0"], dtype=np.float32)
    xD = x0[0, :nx].copy()
    xA_list = [x0[min(i + 1, x0.shape[0] - 1), :nx].copy() for i in range(n_attackers)]

    uD_prev = np.zeros((D,), dtype=np.float32)
    uA_prev = [np.zeros((D,), dtype=np.float32) for _ in range(n_attackers)]
    active_uD = uD_prev.copy()
    active_uA = [u.copy() for u in uA_prev]

    plan_hist: List[List[Tuple[float, float, float]]] = [[] for _ in range(1 + n_attackers)]
    plan_att: List[List[List[Dict[str, Any]]]] = [[] for _ in range(1 + n_attackers)]
    exec_xyz: List[List[Tuple[float, float, float]]] = [[] for _ in range(1 + n_attackers)]
    exec_att: List[List[Dict[str, Any]]] = [[] for _ in range(1 + n_attackers)]
    phi_hist: List[List[float]] = [[] for _ in range(1 + n_attackers)]
    u_cmd: List[List[np.ndarray]] = [[] for _ in range(1 + n_attackers)]
    u_cmd_norm: List[List[float]] = [[] for _ in range(1 + n_attackers)]

    fov_axis_hist: List[Any] = []
    fov_seen_mask: List[bool] = []
    dbg_hist: List[Dict[str, Any]] = []
    g_star_hist: List[float] = []
    terminal_hist: List[Dict[str, Any]] = []
    stopped_early = False

    I = _identity_R()
    all_states = [xD] + xA_list
    for i, xs in enumerate(all_states):
        exec_xyz[i].append(_p3(xs[:D], D))
        exec_att[i].append({"R": I, "phi": 0.0})
        phi_hist[i].append(0.0)
    fov_axis_hist.append(None)
    fov_seen_mask.append(False)

    for k in range(steps):
        should_resolve = (k == 0) or (not solver_params.reuse_action_for_turn_len) or ((k % turn_len) == 0)
        if should_resolve:
            active_uD, active_uA, dbg = solve_one_step_ipopt_team_ppo_zero_sum(
                cfg=cfg,
                xD=xD,
                xA_list=xA_list,
                uD_prev=uD_prev,
                uA_prev=uA_prev,
                k=k,
                step_plant_single=step_plant_single,
                center=center,
                R=R,
                ppo_params=ppo_obj,
                umax=umax,
            )
            uD_prev = active_uD.copy()
            uA_prev = [u.copy() for u in active_uA]
        else:
            dbg = {
                "objective": "ppo_zero_sum",
                "solver_family": "ipopt_best_response_reuse",
                "g_star": float(
                    _ppo_security_value_g(
                        xD=xD,
                        xA_list=xA_list,
                        uD=active_uD,
                        uA_list=active_uA,
                        k=k,
                        step_plant_single=step_plant_single,
                        center=center,
                        R=R,
                        params=ppo_obj,
                        umax=umax,
                    )
                ),
                "br_iters_used": 0,
            }

        dbg_hist.append(dbg)
        g_star_hist.append(float(dbg.get("g_star", np.nan)))

        xD = step_plant_single(xD, active_uD, k)
        for i in range(n_attackers):
            xA_list[i] = step_plant_single(xA_list[i], active_uA[i], k)

        _, term_dbg = _ppo_terminal_adjustment(
            pD=np.asarray(xD[:D], float),
            pA_list=[np.asarray(x[:D], float) for x in xA_list],
            center=center,
            R=R,
            params=ppo_obj,
            primary_idx=0,
        )
        terminal_hist.append(term_dbg)

        cur_states = [xD] + xA_list
        cur_actions = [active_uD] + active_uA
        for i, (xs, ua) in enumerate(zip(cur_states, cur_actions)):
            plan_hist[i].append([_p3(xs, D)] * T_horizon)
            plan_att[i].append([{"R": I, "phi": 0.0} for _ in range(T_horizon)])
            exec_xyz[i].append(_p3(xs, D))
            exec_att[i].append({"R": I, "phi": 0.0})
            phi_hist[i].append(0.0)
            u3 = _pad3(ua, D)
            u_cmd[i].append(u3.copy())
            u_cmd_norm[i].append(float(np.linalg.norm(u3)))

        fov_axis_hist.append(None)
        fov_seen_mask.append(False)

        if term_dbg["done"] and bool(cfg.get("stop_on_done", True)):
            stopped_early = True
            break

    out: Dict[str, Any] = {
        "plan_hist1": plan_hist[0],
        "plan_hist2": plan_hist[1],
        "plan_att1": plan_att[0],
        "plan_att2": plan_att[1],
        "exec1_xyz": exec_xyz[0],
        "exec2_xyz": exec_xyz[1],
        "exec_att1": exec_att[0],
        "exec_att2": exec_att[1],
        "phi_hist1": phi_hist[0],
        "phi_hist2": phi_hist[1],
        "fov_axis_hist": fov_axis_hist,
        "fov_seen_mask": fov_seen_mask,
        "u_cmd_all": [np.asarray(v, float) for v in u_cmd],
        "u_cmd_norm_all": [np.asarray(v, float) for v in u_cmd_norm],
        "ipopt_dbg_hist": dbg_hist,
        "ipopt_params": solver_params.__dict__,
        "baseline_objective": "ppo_zero_sum",
        "g_star_hist": g_star_hist,
        "terminal_hist": terminal_hist,
        "stopped_early": bool(stopped_early),
    }

    if n_attackers == 2:
        out.update(
            {
                "plan_hist3": plan_hist[2],
                "plan_att3": plan_att[2],
                "exec3_xyz": exec_xyz[2],
                "exec_att3": exec_att[2],
                "phi_hist3": phi_hist[2],
            }
        )

    return out


def run_rhc_with_ipopt_game_1v1_collect_frames_3d(
    cfg: Dict[str, Any],
    steps: int | None = None,
    turn_len: int | None = None,
):
    return _run_rollout(cfg=cfg, n_attackers=1, steps=steps, turn_len=turn_len)


def run_rhc_with_ipopt_game_1v2_collect_frames_3d(
    cfg: Dict[str, Any],
    steps: int | None = None,
    turn_len: int | None = None,
):
    return _run_rollout(cfg=cfg, n_attackers=2, steps=steps, turn_len=turn_len)


__all__ = [
    "solve_one_step_ipopt_team_ppo_zero_sum",
    "run_rhc_with_ipopt_game_1v1_collect_frames_3d",
    "run_rhc_with_ipopt_game_1v2_collect_frames_3d",
]
