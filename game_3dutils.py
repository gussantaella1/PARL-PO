# game_3dutils.py
# Dedicated 3D utilities and entrypoints. Depends on game_sharedutils.
from __future__ import annotations
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from dyn_models import (
    dims_from_D,
    hcw_mean_motion, hcw_discrete_mats, as_numpy_const,
    augment_AB_for_att, augment_bounds_with_att, pad_x0_with_att,
    world_to_body_R, frame_from_axis_continuous, apply_roll_about_axis,
    make_bounds,
)



import importlib, game_3dutils, neos_path_game#,rl_infer, game_costs
importlib.reload(game_3dutils)
# importlib.reload(game_costs)
importlib.reload(neos_path_game)
# importlib.reload(rl_infer)

# Keep only module-level imports and always reference via the module
import ukf_estimator, ekf_estimator
import importlib
importlib.reload(ukf_estimator)
importlib.reload(ekf_estimator)

# (Optional) local aliases to current classes after reload
AgentUKF = ukf_estimator.AgentUKF
AgentEKF = ekf_estimator.AgentEKF




importlib.reload(ukf_estimator)
importlib.reload(ekf_estimator)

# from rl_loop_diffgame import AttackerRuleController
# importlib.reload(AttackerRuleController)

__all__ = [
     "run_rhc_and_collect_frames_3d",
    "animate_rollout_3d", "interactive_rollout_3d"
]

try:
    from neos_path_game import (
        build_mcp_two_player_one_shot,
        solve_with_local_path,
        extract_trajectories,
    )


    HAS_PATH = True
except Exception:
    HAS_PATH = False



# ---- PATH/MCP: path-inequality builders (h(k) >= 0) -------------------------
def build_h_builders(cfg, nx, D):
    """
    Return a list of callables h(m,k) >= 0 built from whatever keys
    actually exist in cfg (arena, sep_min/min_sep, vmax/speed_max).
    """
    ar = (cfg.get("arena") or {})
    funcs = []

    def _x(m, agent, k, j):
        return m.x1[k, j] if agent == 1 else m.x2[k, j]

    # -------- arena --------
    # Sphere if given (your CONFIG uses this)
    if {"cx","cy","cz","r"} <= set(ar.keys()) or ar.get("type") == "sphere":
        cx = float(ar.get("cx", 0.0))
        cy = float(ar.get("cy", 0.0))
        cz = float(ar.get("cz", 0.0))
        R2 = float(ar.get("r", 1.0))**2

        def _sphere_h(agent):
            def h(m,k,_a=agent,_cx=cx,_cy=cy,_cz=cz,_R2=R2):
                px = _x(m,_a,k,0) - _cx
                py = _x(m,_a,k,1) - _cy if D >= 2 else 0.0
                pz = _x(m,_a,k,2) - _cz if D == 3 else 0.0
                return _R2 - (px*px + py*py + pz*pz)  # inside sphere
            return h
        funcs += [_sphere_h(1), _sphere_h(2)]

    # -------- min separation --------
    sep = cfg["sep_min"] if "sep_min" in cfg else (cfg["min_sep"] if "min_sep" in cfg else None)
    if sep is not None:
        d2 = float(sep)**2
        def h_sep(m,k,_d2=d2):
            s = 0
            for j in range(D):
                s += (_x(m,1,k,j) - _x(m,2,k,j))**2
            return s - _d2  # >= 0
        funcs.append(h_sep)

    # -------- speed cap --------
    vmax = cfg["vmax"] if "vmax" in cfg else (cfg["speed_max"] if "speed_max" in cfg else None)
    if vmax is not None:
        vmax2 = float(vmax)**2
        vel_idx = list(range(D, 2*D))
        def _speed_h(agent):
            def h(m,k,_a=agent,_v=vel_idx,_v2=vmax2):
                vsq = 0
                for j in _v:
                    vsq += (m.x1[k,j] if _a == 1 else m.x2[k,j])**2
                return _v2 - vsq  # >= 0
            return h
        funcs += [_speed_h(1), _speed_h(2)]

    # -------- object-of-interest (keep-out for selected agents) --------
    # cfg example:
    # cfg["oi"] = {
    #   "enabled": True,
    #   "cx": 0.0, "cy": 0.0, "cz": 0.0,   # cz optional if D==2
    #   "r":  2.0,                         # keep-out radius
    #   "avoid_by": [1]                    # which agents must avoid (defaults to [1])
    # }
    oi = cfg.get("oi", {})
    if bool(oi.get("enabled", False)):
        oc = [float(oi.get(k, 0.0)) for k in (["cx","cy"] if D == 2 else ["cx","cy","cz"])]
        r2 = float(oi.get("r", 1.0))**2
        avoid_by = list(oi.get("avoid_by", [1]))
        for agent in (1, 2):
            if agent not in avoid_by:
                continue
            def _h_oi(m,k,_a=agent,_oc=tuple(oc),_r2=r2):
                s = 0.0
                for j in range(D):
                    s += (_x(m,_a,k,j) - _oc[j])**2
                return s - _r2   # ≥ 0 => outside the object
            funcs.append(_h_oi)

    return funcs


def _filter_kwargs(fn, kwargs):
    """Keep only kwargs that `fn` accepts."""
    import inspect
    sig = inspect.signature(fn)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


class KF_CV:
    """
    Common interface over AgentUKF / AgentEKF.

    - ctor: KF_CV(x0, P0, Q, R, dt, kind='auto'|'ukf'|'ekf', **kwargs)
      UKF-recognized kwargs: alpha, beta, kappa, dyn, hcw
    - predict(dt=None, u=None, **kwargs)
    - update(z, p_obs, R_wb, **kwargs)
    """
    def __init__(self, x0, P0, Q, R, dt, kind='auto', **kwargs):
        kind = (kind or 'auto').lower()

        has_ukf = (AgentUKF is not None)
        has_ekf = (AgentEKF is not None)

        if kind == 'ekf' and has_ekf:
            self._impl = AgentEKF(x0, P0, Q, R, dt)
        elif kind in ('ukf', 'auto') and has_ukf:
            # forward UKF-specific ctor kwargs, incl. dynamics selection
            ukf_ctor_keys = ('alpha', 'beta', 'kappa', 'dyn', 'hcw')
            ukf_ctor = {k: kwargs[k] for k in ukf_ctor_keys if k in kwargs}
            self._impl = AgentUKF(x0, P0, Q, R, dt, **ukf_ctor)
        elif has_ukf:
            self._impl = AgentUKF(x0, P0, Q, R, dt)
        elif has_ekf:
            self._impl = AgentEKF(x0, P0, Q, R, dt)
        else:
            raise RuntimeError("Neither AgentUKF nor AgentEKF is importable.")

    def __getattr__(self, name):
        return getattr(self._impl, name)

    def predict(self, dt=None, u=None, **kwargs):
        f = getattr(self._impl, 'predict')
        return f(dt=dt, u=u, **_filter_kwargs(f, kwargs))

    def update(self, z, p_obs, R_wb, **kwargs):
        f = getattr(self._impl, 'update')
        return f(z, p_obs=p_obs, R_wb=R_wb, **_filter_kwargs(f, kwargs))




# -------------------- FOV artists & cones --------------------



def project_point_pinhole(X_w, x_def, cam_cfg, axis=None, R_wb=None):
    """
    Project a world point into pixel coordinates.
    Returns (u, v, depth, visible_bool).
    depth = Z_cam if align='z', or X_cam if align='x'.
    """
    p_cam = np.asarray(x_def[:3], float)
    align = cam_cfg.get("align", "z")
    R_wc  = np.asarray(R_wb, float) if R_wb is not None else \
            world_to_body_R(_unit(np.asarray(axis,float)), 3, align=align)
    X_c = R_wc @ (np.asarray(X_w, float) - p_cam)

    # world → camera
    if align == "z":
        depth = X_c[2]
        if depth <= 0:
            return None, None, depth, False
        u = cam_cfg["fx"] * (X_c[0] / depth) + cam_cfg["cx"]
        v = cam_cfg["fy"] * (X_c[1] / depth) + cam_cfg["cy"]
    else:  # align == 'x'
        depth = X_c[0]
        if depth <= 0:
            return None, None, depth, False
        u = cam_cfg["fx"] * (X_c[1] / depth) + cam_cfg["cx"]
        v = cam_cfg["fy"] * (X_c[2] / depth) + cam_cfg["cy"]

    W,H   = cam_cfg["W"], cam_cfg["H"]
    near, far = cam_cfg["near"], cam_cfg["far"]

    in_front   = (depth >= near) and (depth <= far)
    in_pixels  = (0 <= u < W) and (0 <= v < H)
    return float(u), float(v), float(depth), (in_front and in_pixels)

def points_in_fov_mask(Xw_list, x_def, axis, cam_cfg):
    """Vector helper: returns a boolean mask for many points."""
    mask = []
    for X in Xw_list:
        _, _, _, ok = project_point_pinhole(X, x_def, axis, cam_cfg)
        mask.append(ok)
    return np.array(mask, dtype=bool)







# -------------------- dims & dynamics (orbital HCW)--------------------

# --- in game_sharedutils.py ---
def as_numpy_const(M):
    """
    Convert a constant CasADi matrix (MX/SX/DM) or array-like to a numpy float array.
    Works across CasADi variants: DM, MX->Function->DM, dict-return Functions.
    """
    import numpy as np
    import casadi as ca

    # Fast path: already numpy
    if isinstance(M, np.ndarray):
        return M.astype(float)

    # If it already behaves like DM (has .full()), use it
    try:
        return np.array(M.full(), dtype=float)
    except Exception:
        pass

    # Evaluate constant MX/SX via a zero-input Function.
    f = ca.Function('const_mat', [], [ca.MX(M)])
    out = f()  # can be DM, or sometimes a dict like {'o0': DM(...)} depending on build

    # Normalize to a DM-like object
    if isinstance(out, dict):
        # take first (and only) value
        out = next(iter(out.values()))
    elif isinstance(out, (list, tuple)):
        out = out[0]

    # Convert to numpy
    try:
        return np.array(out.full(), dtype=float)
    except Exception:
        # Fallback: last resort
        return np.array(out, dtype=float)


# -------------------- frames & FOV --------------------

def att_lin3d_AB(dt: float, att_cfg: dict):
    """
    Linear small-angle attitude model about a reference:
      x_att = [δθ(3); ω(3)],  u_att = τ(3)
      δθ^+ = δθ + dt * (ω - ω_ref)   # set ω_ref=0 to keep strictly linear
      ω^+  = ω  + dt * J^{-1} τ
    We keep ω_ref=0 to stay strictly linear (Ax+Bu); if you want ω_ref≠0, treat it as a known affine term.
    """
    import numpy as np
    J = np.asarray(att_cfg.get("J", [12.0, 10.0, 8.0]), float)
    if J.ndim == 2:   # if full matrix was given, use only the diagonal for B
        Jx, Jy, Jz = np.diag(J)
    else:
        Jx, Jy, Jz = J

    A = np.block([
        [np.eye(3), dt*np.eye(3)],   # δθ^+ = δθ + dt ω
        [np.zeros((3,3)), np.eye(3)] # ω^+  = ω
    ])
    Bin = np.diag([dt/Jx, dt/Jy, dt/Jz])
    B = np.vstack([np.zeros((3,3)), Bin])   # τ only affects ω-row
    return A.astype(float), B.astype(float)


def in_fov(p_t, x_def, axis, fov_cfg, D: int):
    """Check if target p_t is within defender x_def FOV cone/wedge."""
    p_def = np.asarray(x_def[:D], float)
    rel   = np.asarray(p_t[:D], float) - p_def
    dist  = float(np.linalg.norm(rel))
    if dist > float(fov_cfg["range"]):
        return False, dist
    half = 0.5*np.deg2rad(float(fov_cfg["hfov_deg"]))
    cosang = float(np.dot(_unit(rel), _unit(axis)))
    return (cosang >= np.cos(half)), dist

