"""
GameTheory_baseline.py

Game-theory baseline rollout helpers for comparing learned policies against gradient-based one-step best responses.
"""

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
class GameTheoryParams:
    """Configuration values for the gradient-based game-theory baseline."""
    max_iters: int = 50
    step_size_def: float = 5e-5
    step_size_att: float = 5e-5
    fd_eps: float = 1e-6
    tol: float = 1e-5
    reg_prev: float = 1e-3
    reuse_action_for_turn_len: bool = True


def _gt_params_from_cfg(cfg: Dict[str, Any]) -> GameTheoryParams:
    """Internal helper for gt params from cfg."""
    p = GameTheoryParams()
    d = cfg.get("game_theory_baseline", {}) or {}
    for k, v in d.items():
        if hasattr(p, k):
            setattr(p, k, v)
    return p


def _project_box(u: np.ndarray, umax: float) -> np.ndarray:
    """Project box into the constrained space used by the controller."""
    return np.clip(np.asarray(u, dtype=float), -float(umax), float(umax))


def _flatten_actions(u_list: List[np.ndarray]) -> np.ndarray:
    """Flatten per-agent action arrays into one optimization vector."""
    if not u_list:
        return np.zeros((0,), dtype=float)
    return np.concatenate([np.asarray(u, dtype=float).reshape(-1) for u in u_list], axis=0)


def _unflatten_actions(vec: np.ndarray, D: int, n_players: int) -> List[np.ndarray]:
    """Convert flattened action vectors back into per-agent action arrays."""
    arr = np.asarray(vec, dtype=float).reshape(n_players, D)
    return [arr[i].astype(np.float32) for i in range(n_players)]


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


def solve_one_step_extragradient_team_ppo_zero_sum(
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
    """Handle solve one step extragradient team ppo zero sum for this workflow."""
    params = _gt_params_from_cfg(cfg)
    D = int(np.asarray(xD).size // 2)
    nA = len(xA_list)

    z_def = _project_box(np.asarray(uD_prev, dtype=float).reshape(-1), umax)
    z_att = _project_box(_flatten_actions(uA_prev), umax)
    uD_anchor = np.asarray(uD_prev, dtype=float).reshape(-1)
    uA_anchor = _flatten_actions(uA_prev)

    def g_value(uD_vec: np.ndarray, uA_vec: np.ndarray) -> float:
        """Handle g value for this workflow."""
        return float(
            _ppo_security_value_g(
                xD=xD,
                xA_list=xA_list,
                uD=uD_vec,
                uA_list=_unflatten_actions(uA_vec, D, nA),
                k=k,
                step_plant_single=step_plant_single,
                center=center,
                R=R,
                params=ppo_params,
                umax=umax,
            )
            - float(params.reg_prev) * float(np.dot(uD_vec - uD_anchor, uD_vec - uD_anchor))
            + float(params.reg_prev) * float(np.dot(uA_vec - uA_anchor, uA_vec - uA_anchor))
        )

    grad_norm_hist: List[float] = []
    for _ in range(max(1, int(params.max_iters))):
        grad_def = _finite_diff_grad(lambda z: g_value(z, z_att), z_def, params.fd_eps)
        grad_att = _finite_diff_grad(lambda z: g_value(z_def, z), z_att, params.fd_eps)

        mid_def = _project_box(z_def + float(params.step_size_def) * grad_def, umax)
        mid_att = _project_box(z_att - float(params.step_size_att) * grad_att, umax)

        grad_def_mid = _finite_diff_grad(lambda z: g_value(z, mid_att), mid_def, params.fd_eps)
        grad_att_mid = _finite_diff_grad(lambda z: g_value(mid_def, z), mid_att, params.fd_eps)

        z_def_next = _project_box(z_def + float(params.step_size_def) * grad_def_mid, umax)
        z_att_next = _project_box(z_att - float(params.step_size_att) * grad_att_mid, umax)

        grad_norm = float(np.sqrt(np.dot(grad_def_mid, grad_def_mid) + np.dot(grad_att_mid, grad_att_mid)))
        grad_norm_hist.append(grad_norm)
        delta = max(float(np.linalg.norm(z_def_next - z_def)), float(np.linalg.norm(z_att_next - z_att)))
        z_def = z_def_next
        z_att = z_att_next
        if grad_norm <= float(params.tol) or delta <= float(params.tol):
            break

    uA_star = _unflatten_actions(z_att, D, nA)
    g_star = _ppo_security_value_g(
        xD=xD,
        xA_list=xA_list,
        uD=z_def,
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
        "solver_family": "projected_extragradient",
        "iters_used": int(len(grad_norm_hist)),
        "grad_norm_hist": grad_norm_hist,
        "g_star": float(g_star),
    }
    return z_def.astype(np.float32), uA_star, dbg


def _run_rollout(
    cfg: Dict[str, Any],
    n_attackers: int,
    steps: int | None,
    turn_len: int | None,
) -> Dict[str, Any]:
    """Run the internal rollout implementation used by the public entry point."""
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx = 2 * D
    T_horizon = int(cfg["T"])

    if steps is None:
        steps = int(cfg.get("T_eval", cfg.get("T", cfg.get("steps", 60))))

    solver_params = _gt_params_from_cfg(cfg)
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
            active_uD, active_uA, dbg = solve_one_step_extragradient_team_ppo_zero_sum(
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
                "solver_family": "projected_extragradient_reuse",
                "iters_used": 0,
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
        "game_theory_dbg_hist": dbg_hist,
        "game_theory_params": solver_params.__dict__,
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


def run_rhc_with_gametheory_game_1v1_collect_frames_3d(
    cfg: Dict[str, Any],
    steps: int | None = None,
    turn_len: int | None = None,
):
    """Run rhc with gametheory game 1v1 collect frames 3d and return rollout frames, controls, and summary data."""
    return _run_rollout(cfg=cfg, n_attackers=1, steps=steps, turn_len=turn_len)


def run_rhc_with_gametheory_game_1v2_collect_frames_3d(
    cfg: Dict[str, Any],
    steps: int | None = None,
    turn_len: int | None = None,
):
    """Run rhc with gametheory game 1v2 collect frames 3d and return rollout frames, controls, and summary data."""
    return _run_rollout(cfg=cfg, n_attackers=2, steps=steps, turn_len=turn_len)


__all__ = [
    "solve_one_step_extragradient_team_ppo_zero_sum",
    "run_rhc_with_gametheory_game_1v1_collect_frames_3d",
    "run_rhc_with_gametheory_game_1v2_collect_frames_3d",
]
