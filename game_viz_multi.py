# game_viz_multi.py
from __future__ import annotations

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Reuse your geometry helpers
from dyn_models import world_to_body_R, _unit

# Reuse *some* helpers from your original file if you want to import them,
# but this file is self-contained enough to drop in as-is.


__all__ = [
    "animate_rollout_3d_multi",
    "interactive_rollout_3d_multi",
    "plot_rollout_thrust_u_multi",
    "plot_rollout_velocity_multi",
    "plot_rollout_center_distance_multi",
]


# -------------------- small utilities --------------------

def _as3_exec(seq):
    """Coerce list/array of positions to [(x,y,z), ...]. Accept (T,2)->z=0."""
    if seq is None:
        return []
    X = np.asarray(seq, dtype=float)
    if X.ndim != 2 or X.shape[0] < 1:
        return []
    if X.shape[1] == 2:
        X = np.hstack([X, np.zeros((X.shape[0], 1), dtype=float)])
    return [tuple(map(float, row[:3])) for row in X]

def _as3_hist(hist):
    """Coerce plan hist to list of list of 3D tuples."""
    if not hist:
        return []
    out = []
    for fr in hist:
        if fr is None:
            out.append([])
            continue
        A = np.asarray(fr, dtype=float)
        if A.ndim != 2 or A.shape[0] < 1:
            out.append([])
            continue
        if A.shape[1] == 2:
            A = np.hstack([A, np.zeros((A.shape[0], 1), dtype=float)])
        out.append([tuple(map(float, row[:3])) for row in A])
    return out

def _legend_clean(handles):
    pairs = [(h, getattr(h, "get_label", lambda: "")()) for h in handles if h is not None]
    pairs = [(h, lab) for (h, lab) in pairs if lab and not str(lab).startswith('_')]
    return [h for (h, _) in pairs], [lab for (_, lab) in pairs]

def _normalize_oi_list(oi_cfg):
    if not oi_cfg:
        return []
    if isinstance(oi_cfg, (list, tuple)):
        return [d for d in oi_cfg if d and bool(d.get("enabled", True))]
    return [oi_cfg] if bool(oi_cfg.get("enabled", True)) else []

def draw_object_of_interest(ax, oi, D=3, res=24):
    cx = float(oi.get("cx", 0.0))
    cy = float(oi.get("cy", 0.0))
    cz = float(oi.get("cz", 0.0)) if D == 3 else 0.0
    r  = float(oi.get("r", 1.0))

    color     = oi.get("color", "k")
    alpha     = float(oi.get("alpha", 0.15))
    edgecolor = oi.get("edgecolor", "k")

    arts = []
    if D == 3:
        u = np.linspace(0, 2*np.pi, res)
        v = np.linspace(0, np.pi,    res//2 + 1)
        uu, vv = np.meshgrid(u, v)
        X = cx + r * np.cos(uu) * np.sin(vv)
        Y = cy + r * np.sin(uu) * np.sin(vv)
        Z = cz + r * np.cos(vv)
        surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False,
                               color=color, alpha=alpha, shade=True)
        th = np.linspace(0, 2*np.pi, max(32, res))
        xe = cx + r*np.cos(th); ye = cy + r*np.sin(th); ze = cz + 0*th
        (rim,) = ax.plot(xe, ye, ze, color=edgecolor, lw=1.0, alpha=min(1.0, 0.8))
        arts += [surf, rim]
    else:
        th = np.linspace(0, 2*np.pi, max(64, 2*res))
        x = cx + r*np.cos(th); y = cy + r*np.sin(th); z = cz + 0*th
        (ln,) = ax.plot(x, y, z, color=edgecolor, lw=1.5)
        verts = [[(cx, cy, cz), (x[i], y[i], z[i]), (x[(i+1)%len(x)], y[(i+1)%len(y)], z[(i+1)%len(z)])]
                 for i in range(len(x))]
        coll = Poly3DCollection(verts, facecolors=color, edgecolors='none', alpha=alpha)
        ax.add_collection3d(coll)
        arts += [coll, ln]

    for a in arts:
        try: a.set_label("_nolegend_")
        except Exception: pass
    return arts

def _label_axes_3d(ax, scale: float | None = None, unit: str = "m", label_only: bool = True):
    import numpy as np
    from matplotlib.ticker import FuncFormatter

    def _pow10_str(s):
        k = int(round(np.log10(s)))
        return r"$10^{%d}$" % k

    if scale is None or abs(scale - 1.0) < 1e-12:
        ax.set_xlabel(r"$x$ [%s]" % unit)
        ax.set_ylabel(r"$y$ [%s]" % unit)
        ax.set_zlabel(r"$z$ [%s]" % unit)
        return

    if label_only:
        s = _pow10_str(scale)
        ax.set_xlabel(rf"$x$ ({s} {unit})")
        ax.set_ylabel(rf"$y$ ({s} {unit})")
        ax.set_zlabel(rf"$z$ ({s} {unit})")
    else:
        fmt = FuncFormatter(lambda val, pos: f"{val/scale:g}")
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.zaxis.set_major_formatter(fmt)
        ax.set_xlabel(r"$x$ [%s]" % unit)
        ax.set_ylabel(r"$y$ [%s]" % unit)
        ax.set_zlabel(r"$z$ [%s]" % unit)


# -------------------- triads --------------------

def make_body_axes_artists_3d(ax, colors=('tab:red','tab:green','tab:blue'), lw=2, alpha=0.9):
    bx, = ax.plot([], [], [], '-', lw=lw, alpha=alpha, color=colors[0])
    by, = ax.plot([], [], [], '-', lw=lw, alpha=alpha, color=colors[1])
    bz, = ax.plot([], [], [], '-', lw=lw, alpha=alpha, color=colors[2])
    return dict(bx=bx, by=by, bz=bz)

def update_body_axes_artists_3d(lines, p, R_wb, L=(0.4,0.4,0.6)):
    p = np.asarray(p, float)
    x_b, y_b, z_b = R_wb[0], R_wb[1], R_wb[2]
    ends = [p + L[0]*x_b, p + L[1]*y_b, p + L[2]*z_b]
    for (ln, q) in zip((lines['bx'], lines['by'], lines['bz']), ends):
        ln.set_data([p[0], q[0]], [p[1], q[1]])
        ln.set_3d_properties([p[2], q[2]])

def add_triad_legend(ax, colors=('tab:red','tab:green','tab:blue'),
                     labels=('x_b (boresight)','y_b','z_b'),
                     loc='lower left', ncol=1, title='Body axes',
                     keep_legend=None):
    proxies = [Line2D([0], [0], lw=2, color=c, label=lab) for c, lab in zip(colors, labels)]
    leg_axes = ax.legend(proxies, labels, loc=loc, ncol=ncol, title=title)
    if keep_legend is not None:
        ax.add_artist(keep_legend)
    for p in proxies:
        p.set_label('_nolegend_')
    return leg_axes


# -------------------- FOV drawings (cone + pinhole frustum) --------------------

def draw_fov_cone_3d(ax, x_def, fov_cfg, n=24, color='C1', alpha=0.12, align="x", R_wb=None):
    p0 = np.asarray(x_def[:3], float)

    if R_wb is None:
        axis = np.array([1,0,0], float) if align=="x" else np.array([0,0,1], float)
        R = world_to_body_R(axis, 3, align=align)
    else:
        R = np.asarray(R_wb, float)

    rng  = float(fov_cfg["range"])
    half = 0.5*np.deg2rad(float(fov_cfg["hfov_deg"]))
    radius = rng * np.tan(half)

    n = max(int(n), 3)
    ts = np.linspace(0, 2*np.pi, n, endpoint=False)
    if align == "x":
        circ_b = np.vstack([np.full_like(ts, rng),
                            radius*np.cos(ts), radius*np.sin(ts)])
    else:
        circ_b = np.vstack([radius*np.cos(ts),
                            radius*np.sin(ts), np.full_like(ts, rng)])

    circ_w = (R.T @ circ_b).T + p0[None, :]
    verts = [[p0, circ_w[i], circ_w[(i+1) % len(circ_w)]] for i in range(len(circ_w))]
    coll  = Poly3DCollection(verts, facecolors=color, alpha=alpha, edgecolors='none')
    ax.add_collection3d(coll)
    rim_line, = ax.plot(circ_w[:,0], circ_w[:,1], circ_w[:,2], color=color, alpha=0.4, lw=1)
    return coll, rim_line


def draw_camera_frustum_3d(ax, x_def, cam_cfg, color='C2', alpha=0.10,
                           draw_edges=True, draw_rays=True,
                           lw=1.0, rim_alpha=0.55, ray_alpha=0.35,
                           R_wb=None):
    p_cam = np.asarray(x_def[:3], float)
    align = cam_cfg.get("align", "x")

    if R_wb is None:
        axis = np.array([1,0,0], float) if align=="x" else np.array([0,0,1], float)
        R_wc = world_to_body_R(_unit(axis), 3, align=align)
    else:
        R_wc = np.asarray(R_wb, float)

    def cam_to_world(Pc):
        return (R_wc.T @ Pc.T).T + p_cam[None, :]

    W, H = float(cam_cfg["W"]), float(cam_cfg["H"])
    fx, fy = float(cam_cfg["fx"]), float(cam_cfg["fy"])
    cx, cy = float(cam_cfg["cx"]), float(cam_cfg["cy"])
    near, far = float(cam_cfg["near"]), float(cam_cfg["far"])

    corners_px = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=float)
    u = corners_px[:,0]; v = corners_px[:,1]
    if align == "z":
        rays_c = np.stack([(u-cx)/fx, (v-cy)/fy, np.ones_like(u)], axis=1)
        s_near = near / rays_c[:,2]; s_far = far / rays_c[:,2]
    else:
        rays_c = np.stack([np.ones_like(u), (u-cx)/fx, (v-cy)/fy], axis=1)
        s_near = near / rays_c[:,0]; s_far = far / rays_c[:,0]
    near_c = rays_c * s_near[:,None]
    far_c  = rays_c * s_far[:,None]

    near_w = cam_to_world(near_c)
    far_w  = cam_to_world(far_c)

    quads = []
    for i, j in [(0,1),(1,2),(2,3),(3,0)]:
        quads.append([near_w[i], near_w[j], far_w[j], far_w[i]])
    quads.append([near_w[0],near_w[1],near_w[2],near_w[3]])
    quads.append([far_w[0],far_w[1],far_w[2],far_w[3]])

    coll = Poly3DCollection(quads, facecolors=color, alpha=alpha, edgecolors='none')
    ax.add_collection3d(coll)

    edges = []
    if draw_edges:
        (ln_far,)  = ax.plot(*far_w[[0,1,2,3,0]].T,  color=color, alpha=rim_alpha, lw=lw)
        (ln_near,) = ax.plot(*near_w[[0,1,2,3,0]].T, color=color, alpha=rim_alpha*0.9, lw=max(0.8*lw, 0.6))
        edges.extend([ln_far, ln_near])
    if draw_rays:
        for k in range(4):
            (ln,) = ax.plot(*np.vstack([p_cam, far_w[k]]).T, color=color, alpha=ray_alpha, lw=lw)
            edges.append(ln)
    return coll, edges


# -------------------- rollout dict discovery --------------------

def _discover_exec_series(frames_dict: dict):
    """
    Returns:
      exec_all: list of exec trajectories (each is list[(x,y,z), ...])
      N: number of agents
      key_mode: "all" or "numbered"
    """
    if "exec_xyz_all" in frames_dict and frames_dict["exec_xyz_all"] is not None:
        exec_raw = frames_dict["exec_xyz_all"]
        if isinstance(exec_raw, (list, tuple)) and exec_raw and isinstance(exec_raw[0], (list, tuple, np.ndarray)):
            exec_all = [_as3_exec(e) for e in exec_raw]
            return exec_all, len(exec_all), "all"

    # numbered fallback: exec1_xyz, exec2_xyz, exec3_xyz, ...
    exec_all = []
    i = 1
    while True:
        k = f"exec{i}_xyz"
        if k not in frames_dict:
            break
        exec_all.append(_as3_exec(frames_dict.get(k)))
        i += 1
    if not exec_all:
        raise KeyError("Could not find exec trajectories. Expected exec_xyz_all or exec1_xyz/exec2_xyz/... in frames_dict.")
    return exec_all, len(exec_all), "numbered"


def _discover_plan_series(frames_dict: dict, N: int):
    if "plan_hist_all" in frames_dict and frames_dict["plan_hist_all"] is not None:
        plan_raw = frames_dict["plan_hist_all"]
        if isinstance(plan_raw, (list, tuple)) and len(plan_raw) == N:
            return [_as3_hist(p) for p in plan_raw]

    # numbered fallback
    plan_all = []
    for i in range(1, N+1):
        plan_all.append(_as3_hist(frames_dict.get(f"plan_hist{i}", [])))
    return plan_all


def _discover_att_series(frames_dict: dict, N: int):
    """
    Returns list of attitude dict sequences, each of length T-ish or [].
    Each entry: {"R": 3x3, "phi": float} like your runners.
    """
    if "exec_att_all" in frames_dict and frames_dict["exec_att_all"] is not None:
        att_raw = frames_dict["exec_att_all"]
        if isinstance(att_raw, (list, tuple)) and len(att_raw) == N:
            return att_raw

    att_all = []
    for i in range(1, N+1):
        att_all.append(frames_dict.get(f"exec_att{i}", []))
    return att_all


def _discover_est_meas(frames_dict: dict):
    """
    Find any keys of the form:
      est12_xyz, est21_xyz, est31_xyz, ...
      meas12_azel, meas21_azel, ...
    Returns:
      est_series: dict[(i,j)] -> list[(x,y,z),...]
      meas_series: dict[(i,j)] -> list[ (p_obs, z) or None, ... ]
    where i,j are 0-indexed agent ids.
    """
    est_series = {}
    meas_series = {}
    pat_est  = re.compile(r"^est(\d+)(\d+)_xyz$")
    pat_meas = re.compile(r"^meas(\d+)(\d+)_azel$")

    # accept nested frames_dict["est"] if you use that convention later
    merged = dict(frames_dict)
    if isinstance(frames_dict.get("est"), dict):
        merged.update(frames_dict["est"])

    for k, v in merged.items():
        m = pat_est.match(k)
        if m:
            i = int(m.group(1)) - 1
            j = int(m.group(2)) - 1
            est_series[(i, j)] = _as3_exec(v)
        m = pat_meas.match(k)
        if m:
            i = int(m.group(1)) - 1
            j = int(m.group(2)) - 1
            meas_series[(i, j)] = v
    return est_series, meas_series


# -------------------- MULTI animation --------------------

def animate_rollout_3d_multi(frames_dict, save_path="traj_3D.gif", fps=20, cfg=None,
                             show_fov=True, show_axes=True, triads="fov"):
    """
    N-aware rollout animator. Works for 1v2, 1vN, and legacy 1v1.
    - Uses exec_xyz_all/plan_hist_all if present, else exec1_xyz/exec2_xyz/...
    - Draws triads for 'fov' agent or 'all' agents.
    - Overlays any estIJ_xyz / measIJ_azel if present.
    """
    import shutil
    from matplotlib import animation
    from matplotlib.animation import FFMpegWriter, PillowWriter

    if cfg is None:
        raise ValueError("cfg must be provided")

    viz_cfg = cfg.get("viz", {})
    fov_cfg = cfg.get("fov", {})
    att_cfg = cfg.get("att", {})
    D = int(cfg.get("D", 3))

    exec_all, N, _mode = _discover_exec_series(frames_dict)
    plan_all = _discover_plan_series(frames_dict, N)
    att_all  = _discover_att_series(frames_dict, N)

    # frame count
    n_frames = min(len(ex) for ex in exec_all if ex)
    if n_frames < 2:
        raise ValueError("Not enough frames to animate.")

    # axis labels
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.grid(True)

    scale = float(viz_cfg.get("axis_scale", 1.0))
    unit  = str(viz_cfg.get("axis_unit", "m"))
    label_only = bool(viz_cfg.get("axis_label_only", True))
    _label_axes_3d(ax, scale=scale, unit=unit, label_only=label_only)

    # bounds
    ar = cfg.get("arena", {})
    if ar.get("type") == "sphere" and {"cx","cy","cz","r"} <= set(ar.keys()):
        cx, cy, cz, R = float(ar["cx"]), float(ar["cy"]), float(ar["cz"]), float(ar["r"])
        ax.set_xlim(cx-R, cx+R); ax.set_ylim(cy-R, cy+R); ax.set_zlim(cz-R, cz+R)
    elif ar.get("type") == "box" and {"xmin","xmax","ymin","ymax"} <= set(ar.keys()):
        ax.set_xlim(ar["xmin"], ar["xmax"])
        ax.set_ylim(ar["ymin"], ar["ymax"])
        ax.set_zlim(ar.get("zmin", -0.5), ar.get("zmax", 0.5))
    else:
        pts = np.array([p for ex in exec_all for p in ex], float)
        mn = pts.min(0); mx = pts.max(0)
        span = np.maximum(mx-mn, 1e-3); pad = 0.1*span
        lo, hi = mn - pad, mx + pad
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])

    # object(s) of interest
    oi_list = _normalize_oi_list(cfg.get("oi"))
    for oi in oi_list:
        draw_object_of_interest(ax, oi, D=D)

    # colors
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', None) or [
        "tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown",
        "tab:pink","tab:gray","tab:olive","tab:cyan"
    ]

    # artists per agent
    plan_lns, exec_lns, dots = [], [], []
    for a in range(N):
        c = color_cycle[a % len(color_cycle)]
        (pl,) = ax.plot([], [], [], '--', lw=1, alpha=0.65, label=f'Plan P{a+1}', color=c)
        (ex,) = ax.plot([], [], [], '-',  lw=2,              label=f'Exec P{a+1}', color=c)
        (dtp,) = ax.plot([], [], [], 'o', ms=6, color=c, mec='k', mew=0.6, label='_nolegend_')
        plan_lns.append(pl); exec_lns.append(ex); dots.append(dtp)

    # estimates + meas
    est_series, meas_series = _discover_est_meas(frames_dict)
    est_lns = {}
    meas_lns = {}
    show_meas = bool(viz_cfg.get("show_meas", False))

    for (i, j), est in est_series.items():
        c = color_cycle[j % len(color_cycle)]
        (ln,) = ax.plot([], [], [], ':', lw=1.8, marker='x', ms=5.5, mew=1.0,
                        label=f'P{j+1} est by P{i+1}', color=c)
        est_lns[(i, j)] = ln
        if show_meas and (i, j) in meas_series:
            (mln,) = ax.plot([], [], [], '-', lw=0.9, alpha=0.35, color=c, label='_nolegend_')
            meas_lns[(i, j)] = mln

    # legends
    H, L = _legend_clean(plan_lns + exec_lns + list(est_lns.values()))
    leg_main = ax.legend(H, L, loc='upper left') if H else None

    # triads
    triads_mode = (triads or "fov").lower()
    triad_idxs = [] if (not show_axes or triads_mode == "none") else (
        list(range(N)) if triads_mode == "all" else [max(0, int(fov_cfg.get("agent", 1))-1)]
    )
    triad_colors = tuple(viz_cfg.get('triad_colors', ('tab:red','tab:green','tab:blue')))
    triad_labels = tuple(viz_cfg.get('triad_labels', ('x_b (boresight)', 'y_b', 'z_b')))
    triad_leg_loc = viz_cfg.get('triad_leg_loc', 'lower left')
    triad_leg_ncol = int(viz_cfg.get('triad_leg_ncol', 3))
    triad_leg_title = viz_cfg.get('triad_leg_title', 'Body axes')
    L_tri = tuple(viz_cfg.get("triad_len", (0.35, 0.35, 0.55)))

    triad_art = [None for _ in range(N)]
    for a in triad_idxs:
        triad_art[a] = make_body_axes_artists_3d(ax, colors=triad_colors)
        for key in ('bx','by','bz'):
            triad_art[a][key].set_label('_nolegend_')
    if triad_idxs:
        add_triad_legend(ax, colors=triad_colors, labels=triad_labels,
                         loc=triad_leg_loc, ncol=triad_leg_ncol,
                         title=triad_leg_title, keep_legend=leg_main)

    # FOV
    fov_enabled = bool(fov_cfg.get("enabled", False)) and show_fov
    fov_agent = max(1, min(N, int(fov_cfg.get("agent", 1)))) - 1
    seen_mask = list(frames_dict.get("fov_seen_mask", []) or [])
    if seen_mask:
        seen_mask = seen_mask[:n_frames]

    fov_art = {"coll": None, "rim": None, "edges": []}
    def _clear_fov():
        for k in ("coll","rim"):
            art = fov_art.get(k)
            if art is not None:
                try: art.remove()
                except Exception: pass
            fov_art[k] = None
        for ln in fov_art.get("edges", []):
            try: ln.remove()
            except Exception: pass
        fov_art["edges"] = []

    def _R_exec(agent, idx):
        seq = att_all[agent]
        if not seq or idx >= len(seq) or "R" not in seq[idx]:
            return np.eye(3, dtype=float)
        R = np.asarray(seq[idx]["R"], float)
        return R if R.shape == (3,3) else np.eye(3, dtype=float)

    def _set_fov(p, R_wb, idx):
        if not fov_enabled:
            _clear_fov(); return
        col = fov_cfg.get("color","C1")
        if seen_mask and idx < len(seen_mask) and bool(seen_mask[idx]):
            col = "tab:green"
        x_def = np.r_[p, [0,0,0]]

        _clear_fov()
        if fov_cfg.get("type","cone") == "pinhole" and cfg.get("camera") is not None:
            coll, edges = draw_camera_frustum_3d(
                ax, x_def=x_def, cam_cfg=cfg["camera"], color=col,
                alpha=fov_cfg.get("alpha",0.15), R_wb=R_wb
            )
            coll.set_label('_nolegend_')
            for ln in edges:
                ln.set_label('_nolegend_')
            fov_art["coll"], fov_art["rim"], fov_art["edges"] = coll, None, edges
        else:
            coll, rim = draw_fov_cone_3d(
                ax, x_def, fov_cfg, color=col, alpha=fov_cfg.get("alpha",0.15),
                align=att_cfg.get("align","x"), R_wb=R_wb
            )
            coll.set_label('_nolegend_')
            if rim is not None:
                rim.set_label('_nolegend_')
            fov_art["coll"], fov_art["rim"] = coll, rim

    def _set3d(ln, xs, ys, zs):
        ln.set_data(xs, ys); ln.set_3d_properties(zs)

    # meas rays helper
    def _meas_ray(p_obs, R_wb, az, el, Lm):
        c = np.cos(el)
        v_b = np.array([c*np.cos(az), c*np.sin(az), np.sin(el)])
        v_w = R_wb.T @ v_b
        pF = p_obs + Lm * v_w
        return p_obs, pF

    L_meas = float(viz_cfg.get("meas_len", cfg.get("camera", {}).get("far", 15.0)))

    def init():
        for a in range(N):
            plan_lns[a].set_data([], []); plan_lns[a].set_3d_properties([])
            exec_lns[a].set_data([], []); exec_lns[a].set_3d_properties([])
            dots[a].set_data([], []); dots[a].set_3d_properties([])
        for ln in est_lns.values():
            ln.set_data([], []); ln.set_3d_properties([])
        for ln in meas_lns.values():
            ln.set_data([], []); ln.set_3d_properties([])
        for a in triad_idxs:
            L = triad_art[a]
            if L:
                for key in ('bx','by','bz'):
                    L[key].set_data([], []); L[key].set_3d_properties([])
        _clear_fov()
        return []

    def update(f):
        # plans/execs/dots
        for a in range(N):
            if plan_all[a] and f < len(plan_all[a]) and plan_all[a][f]:
                xs, ys, zs = zip(*plan_all[a][f]); _set3d(plan_lns[a], xs, ys, zs)
            else:
                _set3d(plan_lns[a], [], [], [])
            xs = [p[0] for p in exec_all[a][:f+1]]
            ys = [p[1] for p in exec_all[a][:f+1]]
            zs = [p[2] for p in exec_all[a][:f+1]]
            _set3d(exec_lns[a], xs, ys, zs)
            x,y,z = exec_all[a][f]
            dots[a].set_data([x],[y]); dots[a].set_3d_properties([z])

        # triads
        for a in triad_idxs:
            if triad_art[a] is not None:
                x,y,z = exec_all[a][f]
                update_body_axes_artists_3d(triad_art[a], np.array([x,y,z], float), _R_exec(a, f), L=L_tri)

        # estimates
        for (i, j), est in est_series.items():
            ln = est_lns[(i, j)]
            if est:
                xs = [p[0] for p in est[:min(f+1, len(est))]]
                ys = [p[1] for p in est[:min(f+1, len(est))]]
                zs = [p[2] for p in est[:min(f+1, len(est))]]
                _set3d(ln, xs, ys, zs)
            else:
                _set3d(ln, [], [], [])

            if (i, j) in meas_lns:
                me = meas_series[(i, j)]
                if me and f < len(me) and me[f] is not None:
                    p_obs, z = me[f]
                    p_obs = np.asarray(p_obs, float).ravel()[:3]
                    R_i = _R_exec(i, f)
                    p0, pF = _meas_ray(p_obs, R_i, float(z[0]), float(z[1]), L_meas)
                    meas_lns[(i, j)].set_data([p0[0], pF[0]], [p0[1], pF[1]])
                    meas_lns[(i, j)].set_3d_properties([p0[2], pF[2]])
                else:
                    meas_lns[(i, j)].set_data([], []); meas_lns[(i, j)].set_3d_properties([])

        # fov
        if fov_enabled:
            x,y,z = exec_all[fov_agent][f]
            _set_fov(np.array([x,y,z], float), _R_exec(fov_agent, f), f)

        return []

    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=n_frames, interval=int(1000//fps),
                                   blit=False, repeat=False)

    out_path = save_path
    try:
        if out_path.lower().endswith(".mp4") and shutil.which("ffmpeg"):
            writer = FFMpegWriter(fps=fps, codec="libx264",
                                  bitrate=1800, extra_args=["-pix_fmt", "yuv420p"])
            anim.save(out_path, writer=writer, dpi=150)
        else:
            if not out_path.lower().endswith(".gif"):
                out_path = out_path.rsplit(".", 1)[0] + ".gif"
            writer = PillowWriter(fps=fps)
            anim.save(out_path, writer=writer)
    finally:
        plt.close(fig)
    print(f"Saved 3D animation to {out_path}")


# -------------------- MULTI interactive --------------------

def interactive_rollout_3d_multi(frames_dict, cfg, title="Interactive 3D rollout (multi)",
                                 show_fov=True, show_axes=True, triads="fov",
                                 perf_skip_fov_every: int = 1):
    import ipywidgets as W
    from IPython.display import display

    viz_cfg = cfg.get("viz", {})
    fov_cfg = cfg.get("fov", {})
    att_cfg = cfg.get("att", {})
    D = int(cfg.get("D", 3))

    exec_all, N, _mode = _discover_exec_series(frames_dict)
    plan_all = _discover_plan_series(frames_dict, N)
    att_all  = _discover_att_series(frames_dict, N)

    n_frames = min(len(ex) for ex in exec_all if ex)
    if n_frames < 2:
        raise ValueError("No frames to display.")

    est_series, meas_series = _discover_est_meas(frames_dict)
    show_meas_default = bool(viz_cfg.get("show_meas", False)) and bool(meas_series)

    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.grid(True)
    ax.set_box_aspect((1,1,1))

    scale = float(viz_cfg.get("axis_scale", 1.0))
    unit  = str(viz_cfg.get("axis_unit", "m"))
    label_only = bool(viz_cfg.get("axis_label_only", True))
    _label_axes_3d(ax, scale=scale, unit=unit, label_only=label_only)

    ar = cfg.get("arena", {})
    if ar.get("type") == "sphere" and {"cx","cy","cz","r"} <= set(ar.keys()):
        cx, cy, cz, R = float(ar["cx"]), float(ar["cy"]), float(ar["cz"]), float(ar["r"])
        ax.set_xlim(cx-R, cx+R); ax.set_ylim(cy-R, cy+R); ax.set_zlim(cz-R, cz+R)
    elif ar.get("type") == "box" and {"xmin","xmax","ymin","ymax"} <= set(ar.keys()):
        ax.set_xlim(ar["xmin"], ar["xmax"])
        ax.set_ylim(ar["ymin"], ar["ymax"])
        ax.set_zlim(ar.get("zmin", -0.5), ar.get("zmax", 0.5))
    else:
        pts = np.array([p for ex in exec_all for p in ex], float)
        mn = pts.min(0); mx = pts.max(0)
        span = np.maximum(mx-mn, 1e-3); pad = 0.1*span
        lo, hi = mn - pad, mx + pad
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])

    # OI
    oi_list = _normalize_oi_list(cfg.get("oi"))
    for oi in oi_list:
        draw_object_of_interest(ax, oi, D=D)

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', None) or [
        "tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown",
        "tab:pink","tab:gray","tab:olive","tab:cyan"
    ]

    plan_lns, exec_lns, dots = [], [], []
    for a in range(N):
        c = color_cycle[a % len(color_cycle)]
        (pl,) = ax.plot([], [], [], "--", lw=1, alpha=0.65, label=f"Plan P{a+1}", color=c)
        (ex,) = ax.plot([], [], [], "-",  lw=2,              label=f"Exec P{a+1}", color=c)
        (dtp,) = ax.plot([], [], [], "o", ms=6, color=c, mec="k", mew=0.6, label="_nolegend_")
        plan_lns.append(pl); exec_lns.append(ex); dots.append(dtp)

    est_lns, meas_lns = {}, {}
    for (i, j), est in est_series.items():
        c = color_cycle[j % len(color_cycle)]
        (ln,) = ax.plot([], [], [], ":", lw=1.8, marker="x", ms=5.5, mew=1.0,
                        label=f"P{j+1} est by P{i+1}", color=c)
        est_lns[(i, j)] = ln
        if (i, j) in meas_series:
            (mln,) = ax.plot([], [], [], "-", lw=0.9, alpha=0.35, color=c, label="_nolegend_")
            meas_lns[(i, j)] = mln

    H, L = _legend_clean(plan_lns + exec_lns + list(est_lns.values()))
    leg_main = ax.legend(H, L, loc="upper left") if H else None

    triads_mode = (triads or "fov").lower()
    triad_idxs = [] if (not show_axes or triads_mode == "none") else (
        list(range(N)) if triads_mode == "all" else [max(0, int(fov_cfg.get("agent", 1))-1)]
    )
    triad_colors = tuple(viz_cfg.get('triad_colors', ('tab:red','tab:green','tab:blue')))
    triad_labels = tuple(viz_cfg.get('triad_labels', ('x_b (boresight)', 'y_b', 'z_b')))
    triad_leg_loc = viz_cfg.get('triad_leg_loc', 'lower left')
    triad_leg_ncol = int(viz_cfg.get('triad_leg_ncol', 3))
    triad_leg_title = viz_cfg.get('triad_leg_title', 'Body axes')
    L_tri = tuple(viz_cfg.get("triad_len", (0.35, 0.35, 0.55)))

    triad_art = [None for _ in range(N)]
    for a in triad_idxs:
        triad_art[a] = make_body_axes_artists_3d(ax, colors=triad_colors)
        for key in ('bx','by','bz'):
            triad_art[a][key].set_label('_nolegend_')
    if triad_idxs:
        add_triad_legend(ax, colors=triad_colors, labels=triad_labels,
                         loc=triad_leg_loc, ncol=triad_leg_ncol,
                         title=triad_leg_title, keep_legend=leg_main)

    # FOV
    fov_enabled = bool(fov_cfg.get("enabled", False)) and show_fov
    fov_agent = max(1, min(N, int(fov_cfg.get("agent", 1)))) - 1
    seen_mask = list(frames_dict.get("fov_seen_mask", []) or [])
    if seen_mask:
        seen_mask = seen_mask[:n_frames]

    fov_art = {"coll": None, "rim": None, "edges": []}
    def _clear_fov():
        for k in ("coll","rim"):
            art = fov_art.get(k)
            if art is not None:
                try: art.remove()
                except Exception: pass
            fov_art[k] = None
        for ln in fov_art.get("edges", []):
            try: ln.remove()
            except Exception: pass
        fov_art["edges"] = []

    def _R_exec(agent, idx):
        seq = att_all[agent]
        if not seq or idx >= len(seq) or "R" not in seq[idx]:
            return np.eye(3, dtype=float)
        R = np.asarray(seq[idx]["R"], float)
        return R if R.shape == (3,3) else np.eye(3, dtype=float)

    def _set_fov(p, R_wb, idx):
        _clear_fov()
        if not fov_enabled:
            return
        col = fov_cfg.get("color","C1")
        if seen_mask and idx < len(seen_mask) and bool(seen_mask[idx]):
            col = "tab:green"
        x_def = np.r_[p, [0,0,0]]
        if fov_cfg.get("type","cone") == "pinhole" and cfg.get("camera") is not None:
            coll, edges = draw_camera_frustum_3d(
                ax, x_def=x_def, cam_cfg=cfg["camera"], color=col,
                alpha=fov_cfg.get("alpha",0.15), R_wb=R_wb
            )
            fov_art["coll"], fov_art["rim"], fov_art["edges"] = coll, None, edges
        else:
            coll, rim = draw_fov_cone_3d(
                ax, x_def, fov_cfg, color=col, alpha=fov_cfg.get("alpha",0.15),
                align=att_cfg.get("align","x"), R_wb=R_wb
            )
            fov_art["coll"], fov_art["rim"] = coll, rim

    # meas rays helper
    def _meas_ray(p_obs, R_wb, az, el, Lm):
        c = np.cos(el)
        v_b = np.array([c*np.cos(az), c*np.sin(az), np.sin(el)])
        v_w = R_wb.T @ v_b
        pF = p_obs + Lm * v_w
        return p_obs, pF

    L_meas = float(viz_cfg.get("meas_len", cfg.get("camera", {}).get("far", 15.0)))

    # widgets
    s_frame = W.IntSlider(min=0, max=n_frames-1, step=1, value=0, description="frame")
    s_azim  = W.IntSlider(min=-180, max=180, step=1, value=45, description="azim")
    s_elev  = W.IntSlider(min=-10,  max=90,  step=1, value=25, description="elev")
    t_plan  = W.Checkbox(value=True,              description="show plan")
    t_axes  = W.Checkbox(value=show_axes,         description="show axes")
    t_fov   = W.Checkbox(value=fov_enabled,       description="show FOV")
    t_est   = W.Checkbox(value=bool(est_series),  description="show estimates", disabled=not bool(est_series))
    t_meas  = W.Checkbox(value=show_meas_default, description="show meas rays", disabled=not bool(meas_series))

    play = W.Play(min=0, max=n_frames-1, step=1, interval=50, value=0)
    W.jslink((play, "value"), (s_frame, "value"))

    def _set3d(ln, xs, ys, zs):
        ln.set_data(xs, ys); ln.set_3d_properties(zs)

    def redraw(f):
        for a in range(N):
            if t_plan.value and plan_all[a] and f < len(plan_all[a]) and plan_all[a][f]:
                xs, ys, zs = zip(*plan_all[a][f]); _set3d(plan_lns[a], xs, ys, zs)
            else:
                _set3d(plan_lns[a], [], [], [])
            xs = [p[0] for p in exec_all[a][:f+1]]
            ys = [p[1] for p in exec_all[a][:f+1]]
            zs = [p[2] for p in exec_all[a][:f+1]]
            _set3d(exec_lns[a], xs, ys, zs)
            x,y,z = exec_all[a][f]
            dots[a].set_data([x],[y]); dots[a].set_3d_properties([z])

        # triads
        if t_axes.value:
            for a in triad_idxs:
                if triad_art[a] is not None:
                    x,y,z = exec_all[a][f]
                    update_body_axes_artists_3d(triad_art[a], np.array([x,y,z], float), _R_exec(a, f), L=L_tri)
        else:
            for a in triad_idxs:
                L = triad_art[a]
                if L:
                    for key in ('bx','by','bz'):
                        _set3d(L[key], [], [], [])

        # estimates + meas
        for (i, j), est in est_series.items():
            ln = est_lns[(i, j)]
            if t_est.value and est:
                xs = [p[0] for p in est[:min(f+1, len(est))]]
                ys = [p[1] for p in est[:min(f+1, len(est))]]
                zs = [p[2] for p in est[:min(f+1, len(est))]]
                _set3d(ln, xs, ys, zs)
            else:
                _set3d(ln, [], [], [])
            if (i, j) in meas_lns:
                mln = meas_lns[(i, j)]
                me = meas_series.get((i, j), [])
                if t_est.value and t_meas.value and me and f < len(me) and me[f] is not None:
                    p_obs, z = me[f]
                    p_obs = np.asarray(p_obs, float).ravel()[:3]
                    R_i = _R_exec(i, f)
                    p0, pF = _meas_ray(p_obs, R_i, float(z[0]), float(z[1]), L_meas)
                    mln.set_data([p0[0], pF[0]], [p0[1], pF[1]])
                    mln.set_3d_properties([p0[2], pF[2]])
                else:
                    mln.set_data([], []); mln.set_3d_properties([])

        # FOV (perf skip)
        if t_fov.value and fov_enabled:
            if (f % max(1, int(perf_skip_fov_every))) == 0:
                x,y,z = exec_all[fov_agent][f]
                _set_fov(np.array([x,y,z], float), _R_exec(fov_agent, f), f)

        ax.view_init(elev=s_elev.value, azim=s_azim.value)
        fig.canvas.draw_idle()

    s_frame.observe(lambda ch: redraw(ch["new"]), names="value")
    for w in (s_azim, s_elev, t_plan, t_axes, t_fov, t_est, t_meas):
        w.observe(lambda ch: redraw(s_frame.value), names="value")

    redraw(0)
    ui = W.VBox([
        W.HBox([play, s_frame]),
        W.HBox([s_azim, s_elev, t_plan, t_axes, t_fov, t_est, t_meas]),
    ])
    display(ui)


# -------------------- plotting helpers (multi-aware wrappers) --------------------

def plot_rollout_thrust_u_multi(frames_dict, cfg=None, agent: int = 1, show: bool = True):
    # Prefer N-aware arrays in u_cmd_all
    if "u_cmd_all" in frames_dict and frames_dict["u_cmd_all"] is not None:
        u_all = frames_dict["u_cmd_all"]
        if isinstance(u_all, (list, tuple)) and 0 <= agent-1 < len(u_all):
            U = np.asarray(u_all[agent-1], dtype=float)
        else:
            raise KeyError("u_cmd_all present but agent index out of range.")
    else:
        # legacy fallbacks
        k = f"u{agent}_cmd_xyz"
        if k in frames_dict:
            U = np.asarray(frames_dict[k], dtype=float)
        else:
            raise KeyError("Could not find thrust history. Expected u_cmd_all or u{agent}_cmd_xyz.")
    if U.ndim != 2 or U.shape[1] < 3:
        raise ValueError("Thrust array must be (T,3) or (T,>=3).")
    U = U[:, :3]
    dt = float(cfg.get("dt", 1.0)) if cfg is not None else 1.0
    t = dt * np.arange(U.shape[0])
    un = np.linalg.norm(U, axis=1)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(t, U[:,0], label="u_x")
    ax.plot(t, U[:,1], label="u_y")
    ax.plot(t, U[:,2], label="u_z")
    ax.plot(t, un, label="||u||", lw=2)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("commanded thrust")
    ax.grid(True, alpha=0.35)
    ax.set_title(f"Commanded thrust u (agent {agent})")
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_rollout_velocity_multi(frames_dict, cfg=None, agent: int = 1, show: bool = True, title: str | None = None):
    from game_viz import plot_rollout_velocity

    return plot_rollout_velocity(
        frames_dict,
        cfg=cfg,
        agent=agent,
        title=title,
        show=show,
    )


def plot_rollout_center_distance_multi(
    frames_dict,
    cfg=None,
    agents=(1, 2),
    show: bool = True,
    title: str | None = None,
    show_oi_radius: bool = True,
):
    import numpy as np
    import matplotlib.pyplot as plt

    exec_all, N, _ = _discover_exec_series(frames_dict)

    # --- center (prefer arena sphere center; fallback zeros) ---
    ar = cfg.get("arena", {}) if cfg is not None else {}
    center = np.array(
        [float(ar.get("cx", 0.0)), float(ar.get("cy", 0.0)), float(ar.get("cz", 0.0))],
        dtype=float,
    )

    # --- dt ---
    dt = float(cfg.get("dt", 1.0)) if cfg is not None else 1.0

    # --- align T across requested agents ---
    agents = tuple(int(a) for a in agents)
    T = min(len(exec_all[a - 1]) for a in agents)
    if T < 1:
        raise ValueError("No rollout samples found (T < 1).")

    t = dt * np.arange(T)

    # --- plot distances ---
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for a in agents:
        X = np.array(exec_all[a - 1][:T], dtype=float)  # (T,3) list of tuples -> array
        d = np.linalg.norm(X - center[None, :], axis=1)
        ax.plot(t, d, label=f"Agent {a}")

    # --- NEW: OI radius horizontal line(s) ---
    if show_oi_radius and (cfg is not None):
        oi_cfg = cfg.get("oi", None)

        def _normalize_oi_list(oi_cfg):
            if not oi_cfg:
                return []
            if isinstance(oi_cfg, (list, tuple)):
                return [d for d in oi_cfg if d and bool(d.get("enabled", True))]
            return [oi_cfg] if bool(oi_cfg.get("enabled", True)) else []

        oi_list = _normalize_oi_list(oi_cfg)
        for k, oi in enumerate(oi_list):
            if not oi or ("r" not in oi):
                continue
            r = float(oi["r"])
            lab = "OI radius" if len(oi_list) == 1 else f"OI{(k+1)} radius"
            ax.axhline(r, linestyle="--", linewidth=1.5, alpha=0.8, label=lab)

    ax.set_xlabel("time [s]")
    ax.set_ylabel("distance to center")
    ax.grid(True, alpha=0.35)
    ax.set_title(title or "Distance to center vs time")
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax
