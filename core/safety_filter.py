"""
core/safety_filter.py

Velocity-cap safety filters that project nominal accelerations into safe box and CBF constraints.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def project_box_halfspace_np(
    u_nom: np.ndarray,
    lo: float | np.ndarray,
    hi: float | np.ndarray,
    a: np.ndarray,
    b: float,
    *,
    tol: float = 1e-9,
    max_bracket_iters: int = 32,
    max_bisect_iters: int = 48,
) -> np.ndarray:
    """
    Solve the projection

      min_u 0.5 ||u - u_nom||^2
      s.t.  a^T u <= b
            lo <= u <= hi

    for a small dense vector using the monotone KKT scalar multiplier.
    """
    u_nom = np.asarray(u_nom, dtype=float).reshape(-1)
    a = np.asarray(a, dtype=float).reshape(-1)
    lo_arr = np.broadcast_to(np.asarray(lo, dtype=float), u_nom.shape)
    hi_arr = np.broadcast_to(np.asarray(hi, dtype=float), u_nom.shape)

    u0 = np.clip(u_nom, lo_arr, hi_arr)
    if float(np.dot(a, u0)) <= float(b) + tol or float(np.dot(a, a)) <= tol:
        return u0

    u_min = np.where(a >= 0.0, lo_arr, hi_arr)
    if float(np.dot(a, u_min)) > float(b) + tol:
        return u_min

    lam_lo = 0.0
    lam_hi = max(1.0, float(np.dot(a, u0) - b) / (float(np.dot(a, a)) + tol))
    for _ in range(max_bracket_iters):
        u_hi = np.clip(u_nom - lam_hi * a, lo_arr, hi_arr)
        if float(np.dot(a, u_hi)) <= float(b) + tol:
            break
        lam_hi *= 2.0

    for _ in range(max_bisect_iters):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        u_mid = np.clip(u_nom - lam_mid * a, lo_arr, hi_arr)
        if float(np.dot(a, u_mid)) > float(b):
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid

    return np.clip(u_nom - lam_hi * a, lo_arr, hi_arr)


def project_box_halfspace_torch(
    u_nom: torch.Tensor,
    lo: float | torch.Tensor,
    hi: float | torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    tol: float = 1e-6,
    max_bracket_iters: int = 24,
    max_bisect_iters: int = 32,
) -> torch.Tensor:
    """
    Batched torch version of `project_box_halfspace_np`.
    """
    lo_t = torch.as_tensor(lo, dtype=u_nom.dtype, device=u_nom.device)
    hi_t = torch.as_tensor(hi, dtype=u_nom.dtype, device=u_nom.device)
    u0 = torch.clamp(u_nom, min=lo_t, max=hi_t)

    dot0 = torch.sum(a * u0, dim=-1)
    a_norm2 = torch.sum(a * a, dim=-1)
    active = (dot0 > (b + tol)) & (a_norm2 > tol)
    if not torch.any(active):
        return u0

    u_min = torch.where(a >= 0.0, lo_t, hi_t)
    min_dot = torch.sum(a * u_min, dim=-1)
    infeasible = active & (min_dot > (b + tol))

    out = u0.clone()
    if torch.any(infeasible):
        out[infeasible] = u_min[infeasible]

    feasible = active & (~infeasible)
    if not torch.any(feasible):
        return out

    lam_lo = torch.zeros_like(b)
    lam_hi = torch.clamp((dot0 - b) / torch.clamp(a_norm2, min=tol), min=0.0)
    lam_hi = torch.maximum(lam_hi, torch.ones_like(lam_hi))

    for _ in range(max_bracket_iters):
        u_hi = torch.clamp(u_nom - lam_hi[:, None] * a, min=lo_t, max=hi_t)
        still_bad = feasible & (torch.sum(a * u_hi, dim=-1) > (b + tol))
        if not torch.any(still_bad):
            break
        lam_hi = torch.where(still_bad, 2.0 * lam_hi, lam_hi)

    for _ in range(max_bisect_iters):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        u_mid = torch.clamp(u_nom - lam_mid[:, None] * a, min=lo_t, max=hi_t)
        mid_bad = feasible & (torch.sum(a * u_mid, dim=-1) > b)
        lam_lo = torch.where(mid_bad, lam_mid, lam_lo)
        lam_hi = torch.where(mid_bad, lam_hi, lam_mid)

    u_star = torch.clamp(u_nom - lam_hi[:, None] * a, min=lo_t, max=hi_t)
    out[feasible] = u_star[feasible]
    return out


def velocity_cbf_halfspace_np(
    p: np.ndarray,
    v: np.ndarray,
    *,
    vmax: float,
    alpha: float,
    dyn_name: str,
    dt: float,
    D: int,
    Ad: np.ndarray | None = None,
    Bd: np.ndarray | None = None,
    hcw_n: float | None = None,
) -> Tuple[np.ndarray, float]:
    """
    Build the single CBF half-space a^T u <= b for

      h(v) = vmax^2 - ||v||^2
      hdot + alpha * h >= 0

    using an exact HCW drift when available and a local control-affine
    approximation from (Ad, Bd) otherwise.
    """
    p = np.asarray(p, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float).reshape(-1)
    h = float(vmax * vmax - np.dot(v, v))

    dyn_key = str(dyn_name).strip().lower()
    if dyn_key == "hcw" and hcw_n is not None:
        n = float(hcw_n)
        drift = np.zeros((D,), dtype=float)
        if D >= 2:
            drift[0] = 3.0 * n * n * p[0] + 2.0 * n * v[1]
            drift[1] = -2.0 * n * v[0]
        if D >= 3:
            drift[2] = -n * n * p[2]
        a = 2.0 * v
        b = float(alpha) * h - 2.0 * float(np.dot(v, drift))
        return a, b

    if Ad is None or Bd is None:
        raise ValueError("Ad and Bd are required for the generic velocity CBF half-space.")

    x = np.concatenate([p, v], axis=0)
    Ac = (np.asarray(Ad, dtype=float) - np.eye(2 * D, dtype=float)) / float(dt)
    Bc = np.asarray(Bd, dtype=float) / float(dt)
    drift = Ac[D : 2 * D, :] @ x
    Gv = Bc[D : 2 * D, :]
    a = 2.0 * (v @ Gv)
    b = float(alpha) * h - 2.0 * float(np.dot(v, drift))
    return np.asarray(a, dtype=float).reshape(-1), b


def velocity_cbf_halfspace_torch(
    x: torch.Tensor,
    *,
    vmax: float,
    alpha: float,
    dyn_name: str,
    dt: float,
    D: int,
    Ad_t: torch.Tensor | None = None,
    Bd_t: torch.Tensor | None = None,
    hcw_n: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched torch version of `velocity_cbf_halfspace_np`.
    """
    p = x[:, :D]
    v = x[:, D : 2 * D]
    h = (float(vmax) * float(vmax)) - torch.sum(v * v, dim=-1)

    dyn_key = str(dyn_name).strip().lower()
    if dyn_key == "hcw" and hcw_n is not None:
        n = float(hcw_n)
        drift = torch.zeros_like(v)
        if D >= 2:
            drift[:, 0] = 3.0 * n * n * p[:, 0] + 2.0 * n * v[:, 1]
            drift[:, 1] = -2.0 * n * v[:, 0]
        if D >= 3:
            drift[:, 2] = -n * n * p[:, 2]
        a = 2.0 * v
        b = float(alpha) * h - 2.0 * torch.sum(v * drift, dim=-1)
        return a, b

    if Ad_t is None or Bd_t is None:
        raise ValueError("Ad_t and Bd_t are required for the generic velocity CBF half-space.")

    if Ad_t.ndim == 2:
        eye = torch.eye(2 * D, dtype=Ad_t.dtype, device=Ad_t.device)
        Ac_t = (Ad_t - eye) / float(dt)
        Bc_t = Bd_t / float(dt)
        drift = torch.matmul(x, Ac_t[D : 2 * D, :].transpose(0, 1))
        Gv = Bc_t[D : 2 * D, :]
        a = 2.0 * torch.matmul(v, Gv)
    elif Ad_t.ndim == 3:
        eye = torch.eye(2 * D, dtype=Ad_t.dtype, device=Ad_t.device).unsqueeze(0)
        Ac_t = (Ad_t - eye) / float(dt)
        Bc_t = Bd_t / float(dt)
        drift = torch.bmm(Ac_t[:, D : 2 * D, :], x.unsqueeze(-1)).squeeze(-1)
        Gv = Bc_t[:, D : 2 * D, :]
        a = 2.0 * torch.bmm(v.unsqueeze(1), Gv).squeeze(1)
    else:
        raise ValueError(f"Expected Ad_t to have ndim 2 or 3, got {Ad_t.ndim}.")
    b = float(alpha) * h - 2.0 * torch.sum(v * drift, dim=-1)
    return a, b
