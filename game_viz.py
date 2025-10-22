# game_viz.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
from dyn_models import world_to_body_R, _unit, apply_roll_about_axis
import re



# Lazy-import heavy libs to keep core light
def _mpl3d():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.lines import Line2D
    return plt, Poly3DCollection, Line2D


__all__ = [
    "make_body_axes_artists_3d", "update_body_axes_artists_3d",
    "draw_fov_cone_3d", "draw_camera_frustum_3d",
    "add_triad_legend",
    "animate_rollout_3d", "interactive_rollout_3d",
]


def _normalize_oi_list(oi_cfg):
    """Return a list of enabled OI dicts (accepts dict or list of dicts)."""
    if not oi_cfg:
        return []
    if isinstance(oi_cfg, (list, tuple)):
        return [d for d in oi_cfg if d and bool(d.get("enabled", True))]
    return [oi_cfg] if bool(oi_cfg.get("enabled", True)) else []

def draw_object_of_interest(ax, oi, D=3, res=24):
    """
    Draw a sphere (3D) or a circle (2D embedded in z=0) to visualize the object-of-interest.
    oi keys: {cx,cy[,cz], r, color?, alpha?, edgecolor?}
    Returns a list of Matplotlib artists (so caller can hold references if needed).
    """
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    cx = float(oi.get("cx", 0.0))
    cy = float(oi.get("cy", 0.0))
    cz = float(oi.get("cz", 0.0)) if D == 3 else float(oi.get("cz", 0.0) if "cz" in oi else 0.0)
    r  = float(oi.get("r", 1.0))

    color     = oi.get("color", "k")
    alpha     = float(oi.get("alpha", 0.15))
    edgecolor = oi.get("edgecolor", "k")

    arts = []

    if D == 3:
        # sphere surface as a low-res mesh (lightweight)
        u = np.linspace(0, 2*np.pi, res)
        v = np.linspace(0, np.pi,    res//2 + 1)
        uu, vv = np.meshgrid(u, v)
        X = cx + r * np.cos(uu) * np.sin(vv)
        Y = cy + r * np.sin(uu) * np.sin(vv)
        Z = cz + r * np.cos(vv)

        surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False,
                               color=color, alpha=alpha, shade=True)
        # equator ring for clarity
        th = np.linspace(0, 2*np.pi, max(32, res))
        xe = cx + r*np.cos(th); ye = cy + r*np.sin(th); ze = cz + 0*th
        (rim,) = ax.plot(xe, ye, ze, color=edgecolor, lw=1.0, alpha=min(1.0, 0.8))
        arts += [surf, rim]
    else:
        # 2D case: draw a circle in the z=cz plane
        th = np.linspace(0, 2*np.pi, max(64, 2*res))
        x = cx + r*np.cos(th); y = cy + r*np.sin(th); z = cz + 0*th
        (ln,) = ax.plot(x, y, z, color=edgecolor, lw=1.5)
        # very light filled disk as many triangles (optional and cheap)
        verts = [[(cx, cy, cz), (x[i], y[i], z[i]), (x[(i+1)%len(x)], y[(i+1)%len(y)], z[(i+1)%len(z)])]
                 for i in range(len(x))]
        coll = Poly3DCollection(verts, facecolors=color, edgecolors='none', alpha=alpha)
        ax.add_collection3d(coll)
        arts += [coll, ln]

    # keep it out of the legend
    for a in arts:
        try: a.set_label("_nolegend_")
        except Exception: pass
    return arts


# --- artists & legends ---
def make_body_axes_artists_3d(ax, colors=('tab:red','tab:green','tab:blue'), lw=2, alpha=0.9):
    plt, _, _ = _mpl3d()
    bx, = ax.plot([], [], [], '-', lw=lw, alpha=alpha, color=colors[0])
    by, = ax.plot([], [], [], '-', lw=lw, alpha=alpha, color=colors[1])
    bz, = ax.plot([], [], [], '-', lw=lw, alpha=alpha, color=colors[2])
    return dict(bx=bx, by=by, bz=bz)

def update_body_axes_artists_3d(lines, p, R_wb, L=(0.4,0.4,0.6)):
    p = np.asarray(p, float); x_b, y_b, z_b = R_wb[0], R_wb[1], R_wb[2]
    ends = [p + L[0]*x_b, p + L[1]*y_b, p + L[2]*z_b]
    for (ln, q) in zip((lines['bx'], lines['by'], lines['bz']), ends):
        ln.set_data([p[0], q[0]], [p[1], q[1]])
        ln.set_3d_properties([p[2], q[2]])

def add_triad_legend(ax, colors=('tab:red','tab:green','tab:blue'),
                     labels=('x_b (boresight)','y_b','z_b'),
                     loc='lower left', ncol=1, title='Body axes',
                     keep_legend=None):
    plt, _, Line2D = _mpl3d()
    proxies = [Line2D([0], [0], lw=2, color=c, label=lab) for c, lab in zip(colors, labels)]
    leg_axes = ax.legend(proxies, labels, loc=loc, ncol=ncol, title=title)
    if keep_legend is not None:
        ax.add_artist(keep_legend)
    for p in proxies:
        p.set_label('_nolegend_')
    return leg_axes

# --- fov geometry drawing (cones/frustums) ---
def draw_fov_cone_3d(ax, x_def, fov_cfg, n=24, color='C1', alpha=0.12, align="x"):
    plt, Poly3DCollection, _ = _mpl3d()
    p0   = np.asarray(x_def[:3], float)
    axis = _unit(np.asarray(fov_cfg.get("axis_override", [1,0,0]) if False else ax.viewLim, float))  # placeholder to avoid lints
    axis = _unit(np.asarray(fov_cfg.get("axis"), float)) if "axis" in fov_cfg else None
    if axis is None:
        axis = np.array([1,0,0], float) if align=="x" else np.array([0,0,1], float)
    R    = world_to_body_R(axis, 3, align=align)

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


def draw_camera_frustum_3d(ax, x_def, axis=None, cam_cfg=None,
                           color='C2', alpha=0.10,
                           draw_edges=True, draw_rays=True,
                           lw=1.0, rim_alpha=0.55, ray_alpha=0.35,
                           R_wb=None):
    """
    Draw a pinhole camera frustum using intrinsics and near/far planes.

    Args
    ----
    ax      : Matplotlib 3D axes
    x_def   : state vector; camera position at x_def[:3]
    axis    : boresight direction in WORLD coords
    cam_cfg : dict with keys {W,H,fx,fy,cx,cy,near,far,align}
              align is 'x' (x-forward) or 'z' (z-forward)

    Returns
    -------
    coll  : Poly3DCollection for filled frustum (near/far + 4 side quads)
    edges : list of Line3D objects for rims/rays (for later removal)
    """
    # --- pose + rotation conventions ---
    p_cam = np.asarray(x_def[:3], float)
    align = cam_cfg.get("align", "x")  # must match your boresight convention

    # world→camera; use transpose for camera→world
    if R_wb is None:
        axis = _unit(np.asarray(axis, float))
        R_wc = world_to_body_R(axis, 3, align=align)
    else:
        R_wc = np.asarray(R_wb, float)

    def cam_to_world(Pc):
        return (R_wc.T @ Pc.T).T + p_cam[None, :]

    # --- intrinsics / depth bounds ---
    W, H = float(cam_cfg["W"]), float(cam_cfg["H"])
    fx, fy = float(cam_cfg["fx"]), float(cam_cfg["fy"])
    cx, cy = float(cam_cfg["cx"]), float(cam_cfg["cy"])
    near, far = float(cam_cfg["near"]), float(cam_cfg["far"])
    assert near > 0.0 and far > near, "Require 0 < near < far"
    assert fx != 0.0 and fy != 0.0, "fx/fy must be nonzero"

    # --- corner rays (TL, TR, BR, BL) in camera frame ---
    corners_px = _image_corners_px(W, H)
    rays_c     = _corner_rays_camera(corners_px, fx, fy, cx, cy, align=align)

    # scale rays to near/far planes (depth along align axis)
    near_c = _scale_rays_to_plane(rays_c, near, align=align)  # (4,3)
    far_c  = _scale_rays_to_plane(rays_c,  far,  align=align)  # (4,3)

    # transform to world
    near_w = cam_to_world(near_c)
    far_w  = cam_to_world(far_c)

    # --- build faces: 4 side quads + caps ---
    # consistent winding using TL(0)->TR(1)->BR(2)->BL(3)
    quads = []
    for i, j in [(0,1), (1,2), (2,3), (3,0)]:
        quads.append([near_w[i], near_w[j], far_w[j], far_w[i]])   # sides
    quads.append([near_w[0], near_w[1], near_w[2], near_w[3]])     # near cap
    quads.append([far_w[0],  far_w[1],  far_w[2],  far_w[3]])      # far cap

    coll = Poly3DCollection(quads, facecolors=color, alpha=alpha, edgecolors='none')
    ax.add_collection3d(coll)

    # --- optional edges/rays for clarity (return handles so we can remove later) ---
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

def _image_corners_px(W, H):
    # (u, v) pixel corners in order: TL, TR, BR, BL
    return np.array([[0,   0],
                     [W-1, 0],
                     [W-1, H-1],
                     [0,   H-1]], dtype=float)

def _corner_rays_camera(corners_px, fx, fy, cx, cy, align="z"):
    """
    Return 4 direction vectors (not normalized) in camera frame that
    pass through image corners. For align='z', rays are [ (u-cx)/fx, (v-cy)/fy, 1 ].
    For align='x', rays are [ 1, (u-cx)/fx, (v-cy)/fy ].
    """
    u = corners_px[:, 0]
    v = corners_px[:, 1]
    if align == "z":
        return np.stack([(u - cx)/fx, (v - cy)/fy, np.ones_like(u)], axis=1)
    else:  # align == "x"
        return np.stack([np.ones_like(u), (u - cx)/fx, (v - cy)/fy], axis=1)

def _scale_rays_to_plane(rays, depth, align="z"):
    """
    Scale each ray so that it intersects the plane at 'depth':
    If align='z': set Z = depth.
    If align='x': set X = depth.
    """
    rays = np.asarray(rays, float)
    out = rays.copy()
    if align == "z":
        s = depth / rays[:, 2]  # Z component
        out *= s[:, None]
    else:  # align == "x"
        s = depth / rays[:, 0]  # X component
        out *= s[:, None]
    return out


def _legend_clean(handles):
    """Return (handles, labels) with any private/empty labels removed."""
    pairs = [(h, getattr(h, "get_label", lambda: "")()) for h in handles if h is not None]
    pairs = [(h, lab) for (h, lab) in pairs if lab and not str(lab).startswith('_')]
    return [h for (h, _) in pairs], [lab for (_, lab) in pairs]

# ---- axis label helper (unit label + optional tick scaling) ----
def _label_axes_3d(ax, scale: float | None = None, unit: str = "m", label_only: bool = True):
    """
    If scale is None or 1.0: show plain labels like 'x [m]'.
    If scale is e.g. 1e2 and label_only=True: labels show 'x (10^2 m)'.
    If scale is e.g. 1e2 and label_only=False: tick numbers are divided by 1e2
    and labels show 'x [m]'.
    """
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


# -------------------- animation (triads + legend fixes) --------------------


def animate_rollout_3d(frames_dict, save_path="traj_3D.gif", fps=20, cfg=None,
                       show_fov=True, show_axes=True):
    import shutil
    from matplotlib import animation
    from matplotlib.animation import FFMpegWriter, PillowWriter

    if cfg is None:
        raise ValueError("cfg must be provided")

    fov_cfg  = cfg.get("fov", {})
    att_cfg  = cfg.get("att", {})
    viz_cfg  = cfg.get("viz", {})
    agent_id = int(fov_cfg.get("agent", 2))

    triad_colors = tuple(viz_cfg.get('triad_colors', ('tab:red','tab:green','tab:blue')))
    triad_labels = tuple(viz_cfg.get('triad_labels', ('x_b (boresight)', 'y_b', 'z_b')))
    triad_leg_loc = viz_cfg.get('triad_leg_loc', 'lower left')
    triad_leg_ncol = int(viz_cfg.get('triad_leg_ncol', 3))
    triad_leg_title = viz_cfg.get('triad_leg_title', 'Body axes')
    L_tri = tuple(cfg.get("viz", {}).get("triad_len", (0.35, 0.35, 0.55)))

    plan_hist1 = frames_dict.get('plan_hist1', [])
    plan_hist2 = frames_dict.get('plan_hist2', [])
    exec1      = frames_dict.get('exec1_xyz', [])
    exec2      = frames_dict.get('exec2_xyz', [])
    axis_hist  = frames_dict.get('fov_axis_hist', [])
    seen_mask  = frames_dict.get('fov_seen_mask', [])

    # estimation overlays
    est12 = frames_dict.get('est12_xyz', [])
    est21 = frames_dict.get('est21_xyz', [])
    has_est12 = bool(est12) and any(e is not None for e in est12)
    has_est21 = bool(est21) and any(e is not None for e in est21)
    est_enabled = bool(cfg.get('est', {}).get('enabled', False))

    only_est  = bool(cfg.get('viz', {}).get('only_est', False))
    show_meas = bool(cfg.get('viz', {}).get('show_meas', False))

    n_frames = min(len(exec1), len(exec2))
    if n_frames < 2:
        if n_frames == 1:
            exec1 = exec1 + [exec1[-1]]
            exec2 = exec2 + [exec2[-1]]
            plan_hist1 = plan_hist1 + [plan_hist1[-1] if plan_hist1 else []]
            plan_hist2 = plan_hist2 + [plan_hist2[-1] if plan_hist2 else []]
            n_frames = 2
        else:
            raise ValueError("No frames to animate.")

    fig = plt.figure(figsize=(7,6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.grid(True)

    # NEW: axis labels / tick scaling (viz.axis_scale, viz.axis_unit, viz.axis_label_only)
    scale = float(viz_cfg.get("axis_scale", 1.0))        # e.g., 1e2 for “10^2 m”
    unit  = str(viz_cfg.get("axis_unit", "m"))
    label_only = bool(viz_cfg.get("axis_label_only", True))
    _label_axes_3d(ax, scale=scale, unit=unit, label_only=label_only)

    # arena limits
    ar = cfg.get("arena", {})
    if ar.get("type") == "box":
        ax.set_xlim(ar["xmin"], ar["xmax"])
        ax.set_ylim(ar["ymin"], ar["ymax"])
        ax.set_zlim(ar.get("zmin", -0.5), ar.get("zmax", 0.5))
    elif ar.get("type") == "sphere":
        cx, cy, cz, R = ar["cx"], ar["cy"], ar["cz"], ar["r"]
        ax.set_xlim(cx-R, cx+R); ax.set_ylim(cy-R, cy+R); ax.set_zlim(cz-R, cz+R)

    # --- draw object(s) of interest, if any ---
    D = int(cfg.get("D", 3))
    oi_list = _normalize_oi_list(cfg.get("oi"))
    oi_artists = []
    for oi in oi_list:
        oi_artists += draw_object_of_interest(ax, oi, D=D)

    # ---- artists ----
    plan1_ln = plan2_ln = exe1_ln = exe2_ln = dot1 = dot2 = None
    me12_ln = me21_ln = None
    handles_for_legend = []

    if not only_est:
        plan1_ln, = ax.plot([], [], [], '--', lw=1, alpha=0.6, label='Plan P1')
        plan2_ln, = ax.plot([], [], [], '--', lw=1, alpha=0.6, label='Plan P2')
        exe1_ln,  = ax.plot([], [], [], '-',  lw=2, label='Exec P1')
        exe2_ln,  = ax.plot([], [], [], '-',  lw=2, label='Exec P2')
        dot1,     = ax.plot([], [], [], 'o',  ms=6, label='_nolegend_')
        dot2,     = ax.plot([], [], [], 'o',  ms=6, label='_nolegend_')
        handles_for_legend += [plan1_ln, plan2_ln, exe1_ln, exe2_ln]

    est12_ln, = ax.plot([], [], [], ':', lw=1.8, marker='x', ms=6, mew=1.0,
                        label=('P2 est by P1' if (est_enabled and has_est12) else '_nolegend_'))
    est21_ln, = ax.plot([], [], [], ':', lw=1.8, marker='x', ms=6, mew=1.0,
                        label=('P1 est by P2' if (est_enabled and has_est21) else '_nolegend_'))
    if est_enabled and has_est12: handles_for_legend.append(est12_ln)
    if est_enabled and has_est21: handles_for_legend.append(est21_ln)

    if (not only_est) and show_meas:
        me12_ln,  = ax.plot([], [], [], '-',  lw=1.0, alpha=0.45, label='_nolegend_')
        me21_ln,  = ax.plot([], [], [], '-',  lw=1.0, alpha=0.45, label='_nolegend_')

    handles, labels = _legend_clean(handles_for_legend)
    leg_main = ax.legend(handles, labels, loc='upper left') if handles else None

    leg_axes = None
    if show_axes:
        leg_axes = add_triad_legend(ax,
                                    colors=triad_colors,
                                    labels=triad_labels,
                                    loc=triad_leg_loc,
                                    ncol=triad_leg_ncol,
                                    title=triad_leg_title,
                                    keep_legend=leg_main)

    att1_lines = make_body_axes_artists_3d(ax, colors=triad_colors) if show_axes else None
    att2_lines = make_body_axes_artists_3d(ax, colors=triad_colors) if show_axes else None
    for L in (att1_lines, att2_lines):
        if L:
            L['bx'].set_label('_nolegend_'); L['by'].set_label('_nolegend_'); L['bz'].set_label('_nolegend_')

    fov_art = {'coll': None, 'rim': None, 'edges': []}

    def _pos3(p_like):
        p = np.asarray(p_like, float).ravel()
        return p[:3] if p.size >= 3 else np.array([p[0], p[1], 0.0], float)

    def _def_pos(f):
        return exec2[f] if agent_id == 2 else exec1[f]

    def _clear_fov():
        for k in ('coll','rim'):
            art = fov_art.get(k)
            if art is not None:
                try: art.remove()
                except Exception: pass
                fov_art[k] = None
        for ln in fov_art.get('edges', []):
            try: ln.remove()
            except Exception: pass
        fov_art['edges'] = []

    def _R_exec(agent, idx):
        key = "exec_att1" if agent == 1 else "exec_att2"
        L = frames_dict.get(key)
        if not L or idx >= len(L) or "R" not in L[idx]:
            raise KeyError(f"Missing attitude for agent {agent} at frame {idx} (need frames_dict['{key}'][{idx}]['R']).")
        R = np.asarray(L[idx]["R"], float)
        if R.shape != (3, 3):
            raise ValueError(f"Attitude R for agent {agent} at frame {idx} must be 3x3, got {R.shape}.")
        return R


    def _set_fov(p, R_wb, idx):
        if not (show_fov and fov_cfg.get("enabled", False)):
            _clear_fov(); return
        col = fov_cfg.get("color","C1")
        if seen_mask and idx < len(seen_mask) and bool(seen_mask[idx]):
            col = 'tab:green'
        x_def = np.r_[p, [0,0,0]]
        if fov_cfg.get("type","cone") == "pinhole" and cfg.get("camera") is not None:
            coll, edges = draw_camera_frustum_3d(
                ax, x_def=x_def, cam_cfg=cfg["camera"],
                color=col, alpha=fov_cfg.get("alpha",0.15),
                R_wb=R_wb
            )
            coll.set_label('_nolegend_')
            for ln in edges: ln.set_label('_nolegend_')
            fov_art['coll'], fov_art['rim'], fov_art['edges'] = coll, None, edges
        else:
            coll, rim = draw_fov_cone_3d(
                ax, x_def, fov_cfg,
                color=col, alpha=fov_cfg.get("alpha",0.15),
                align=att_cfg.get('align','x'),
                R_wb=R_wb
            )
            coll.set_label('_nolegend_')
            if rim is not None: rim.set_label('_nolegend_')
            fov_art['coll'], fov_art['rim'] = coll, rim

    def init():
        for ln in (plan1_ln, plan2_ln, exe1_ln, exe2_ln, dot1, dot2,
                   est12_ln, est21_ln, me12_ln, me21_ln):
            if ln is not None:
                ln.set_data([], []); ln.set_3d_properties([])
        for L in (att1_lines, att2_lines):
            if L:
                for ln in (L['bx'], L['by'], L['bz']):
                    ln.set_data([], []); ln.set_3d_properties([])
        _clear_fov()
        return []

    def update(f):
        def _set3d(ln, xs, ys, zs):
            if ln is not None:
                ln.set_data(xs, ys); ln.set_3d_properties(zs)

        # estimates
        if est_enabled and has_est12:
            xs, ys, zs = zip(*est12[:f+1]); _set3d(est12_ln, xs, ys, zs)
        else:
            _set3d(est12_ln, [], [], [])
        if est_enabled and has_est21:
            xs, ys, zs = zip(*est21[:f+1]); _set3d(est21_ln, xs, ys, zs)
        else:
            _set3d(est21_ln, [], [], [])

        # plan/exec
        if not only_est:
            if plan_hist1 and f < len(plan_hist1) and plan_hist1[f]:
                xs, ys, zs = zip(*plan_hist1[f]); _set3d(plan1_ln, xs, ys, zs)
            else:
                _set3d(plan1_ln, [], [], [])
            if plan_hist2 and f < len(plan_hist2) and plan_hist2[f]:
                xs, ys, zs = zip(*plan_hist2[f]); _set3d(plan2_ln, xs, ys, zs)
            else:
                _set3d(plan2_ln, [], [], [])
            xs1 = [p[0] for p in exec1[:f+1]]; ys1 = [p[1] for p in exec1[:f+1]]; zs1 = [p[2] for p in exec1[:f+1]]
            xs2 = [p[0] for p in exec2[:f+1]]; ys2 = [p[1] for p in exec2[:f+1]]; zs2 = [p[2] for p in exec2[:f+1]]
            _set3d(exe1_ln, xs1, ys1, zs1); _set3d(exe2_ln, xs2, ys2, zs2)
            if dot1 is not None:
                x1,y1,z1 = exec1[min(f, len(exec1)-1)]
                dot1.set_data([x1],[y1]); dot1.set_3d_properties([z1])
            if dot2 is not None:
                x2,y2,z2 = exec2[min(f, len(exec2)-1)]
                dot2.set_data([x2],[y2]); dot2.set_3d_properties([z2])
        else:
            for ln in (plan1_ln, plan2_ln, exe1_ln, exe2_ln):
                _set3d(ln, [], [], [])
            if dot1 is not None: dot1.set_data([],[]); dot1.set_3d_properties([])
            if dot2 is not None: dot2.set_data([],[]); dot2.set_3d_properties([])

        # triads
        if show_axes:
            x1,y1,z1 = exec1[min(f, len(exec1)-1)]
            x2,y2,z2 = exec2[min(f, len(exec2)-1)]
            if att1_lines: update_body_axes_artists_3d(att1_lines, np.array([x1,y1,z1]), _R_exec(1, f), L=L_tri)
            if att2_lines: update_body_axes_artists_3d(att2_lines, np.array([x2,y2,z2]), _R_exec(2, f), L=L_tri)

        # FOV
        p_def = _pos3(_def_pos(f)); R_def = _R_exec(agent_id, f)
        _clear_fov(); _set_fov(p_def, R_def, f)
        return []

    out_path = save_path
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=n_frames, interval=int(1000//fps),
                                   blit=False, repeat=False)
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


# -------------------- interactive (triads + estimates + meas + legend hygiene) --------------------
def interactive_rollout_3d(
    frames_dict,
    cfg,
    title="Interactive 3D rollout",
    show_fov=True,
    show_axes=True,
    perf_skip_fov_every: int = 1,
):
    import ipywidgets as W
    from IPython.display import display

    def _legend_clean(handles):
        H = []
        for h in handles:
            if h is None:
                continue
            lab = getattr(h, "get_label", lambda: "")()
            if lab and not str(lab).startswith("_"):
                H.append(h)
        return H, [h.get_label() for h in H]

    def _as3_hist(seq):
        if not seq:
            return []
        if seq and seq[0] and len(seq[0][0]) == 2:
            return [[(x, y, 0.0) for (x, y) in fr] for fr in seq]
        return seq

    def _as3_exec(seq):
        if not seq:
            return []
        if seq and len(seq[0]) == 2:
            return [(x, y, 0.0) for (x, y) in seq]
        return seq

    fov_cfg  = cfg.get("fov", {})
    att_cfg  = cfg.get("att", {})
    viz_cfg  = cfg.get("viz", {})
    agent_id = int(fov_cfg.get("agent", 2))

    triad_colors   = tuple(viz_cfg.get("triad_colors", ("tab:red", "tab:green", "tab:blue")))
    triad_labels   = tuple(viz_cfg.get("triad_labels", ("x_b (boresight)", "y_b", "z_b")))
    triad_leg_loc  = viz_cfg.get("triad_leg_loc", "lower left")
    triad_leg_ncol = int(viz_cfg.get("triad_leg_ncol", 3))
    triad_leg_title = viz_cfg.get("triad_leg_title", "Body axes")
    L_tri = tuple(viz_cfg.get("triad_len", (0.35, 0.35, 0.55)))

    plan_hist1 = frames_dict.get("plan_hist1_3d", frames_dict.get("plan_hist1", []))
    plan_hist2 = frames_dict.get("plan_hist2_3d", frames_dict.get("plan_hist2", []))
    exec1      = frames_dict.get("exec1_xyz", frames_dict.get("exec1_xy", []))
    exec2      = frames_dict.get("exec2_xyz", frames_dict.get("exec2_xy", []))
    axis_hist  = frames_dict.get("fov_axis_hist", [])
    seen_mask  = frames_dict.get("fov_seen_mask", [])

    est12 = frames_dict.get("est12_xyz", [])
    est21 = frames_dict.get("est21_xyz", [])
    me12  = frames_dict.get("meas12_azel", [])
    me21  = frames_dict.get("meas21_azel", [])

    do_est_default    = bool(cfg.get("est", {}).get("enabled", False)) or bool(est12 or est21)
    show_meas_default = bool(viz_cfg.get("show_meas", False)) and do_est_default

    plan_hist1 = _as3_hist(plan_hist1)
    plan_hist2 = _as3_hist(plan_hist2)
    exec1      = _as3_exec(exec1)
    exec2      = _as3_exec(exec2)
    n_frames   = max(2, min(len(exec1), len(exec2)))

    L_meas = float(viz_cfg.get("meas_len", cfg.get("camera", {}).get("far", 15.0)))

    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.grid(True)
    ax.set_box_aspect((1, 1, 1))

    # NEW: axis labels / tick scaling (viz.axis_scale, viz.axis_unit, viz.axis_label_only)
    scale = float(viz_cfg.get("axis_scale", 1.0))
    unit  = str(viz_cfg.get("axis_unit", "m"))
    label_only = bool(viz_cfg.get("axis_label_only", True))
    _label_axes_3d(ax, scale=scale, unit=unit, label_only=label_only)

    # bounds
    ar = cfg.get("arena", {})
    if ar.get("type") == "box" or ({"xmin", "xmax", "ymin", "ymax"} <= set(ar.keys())):
        xmin, xmax = ar.get("xmin", -3), ar.get("xmax", 3)
        ymin, ymax = ar.get("ymin", -3), ar.get("ymax", 3)
        if "zmin" in ar and "zmax" in ar:
            zmin, zmax = ar["zmin"], ar["zmax"]
        else:
            zs = [p[2] for p in (exec1 + exec2)] or [0.0]
            zmin, zmax = min(zs) - 0.5, max(zs) + 0.5
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)
    elif ar.get("type") == "sphere" or ({"cx", "cy", "cz", "r"} <= set(ar.keys())):
        cx, cy, cz = ar.get("cx", 0.0), ar.get("cy", 0.0), ar.get("cz", 0.0)
        Rb = ar.get("r", 3.0)
        ax.set_xlim(cx - Rb, cx + Rb); ax.set_ylim(cy - Rb, cy + Rb); ax.set_zlim(cz - Rb, cz + Rb)
    else:
        pts = np.array(exec1 + exec2, float)
        if pts.size > 0:
            mn = pts.min(0); mx = pts.max(0)
            span = np.maximum(mx - mn, 1e-3); pad = 0.1 * span
            lo = mn - pad; hi = mx + pad
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])

    # --- draw object(s) of interest, if any ---
    D = int(cfg.get("D", 3))
    oi_list = _normalize_oi_list(cfg.get("oi"))
    oi_artists = []
    for oi in oi_list:
        oi_artists += draw_object_of_interest(ax, oi, D=D)


    # artists
    plan1_ln, = ax.plot([], [], [], "--", lw=1, alpha=0.6, label="Plan P1", color="blue")
    plan2_ln, = ax.plot([], [], [], "--", lw=1, alpha=0.6, label="Plan P2", color="orange")
    exe1_ln,  = ax.plot([], [], [], "-",  lw=2, label="Exec P1", color="green")
    exe2_ln,  = ax.plot([], [], [], "-",  lw=2, label="Exec P2", color="red")

    dot1, = ax.plot([], [], [], "o", ms=6, color="cyan",  mec="k", mew=0.6, label="_nolegend_")
    dot2, = ax.plot([], [], [], "o", ms=6, color="orange", mec="k", mew=0.6, label="_nolegend_")

    est12_ln, = ax.plot([], [], [], ":", lw=1.8, marker="x", ms=6, mew=1.0, label="P2 est by P1")
    est21_ln, = ax.plot([], [], [], ":", lw=1.8, marker="x", ms=6, mew=1.0, label="P1 est by P2")

    def _blank_line3d(ln):
        # NaNs are the safest "no geometry" sentinel for 3D lines
        ln.set_data([np.nan], [np.nan])
        ln.set_3d_properties([np.nan])

    me12_ln, = ax.plot([], [], [], "-", lw=0.8, alpha=0.25, zorder=-10, label="_nolegend_")
    me21_ln, = ax.plot([], [], [], "-", lw=0.8, alpha=0.25, zorder=-10, label="_nolegend_")

    # start fully blank & hidden unless the checkbox turns them on
    _blank_line3d(me12_ln); me12_ln.set_visible(False)
    _blank_line3d(me21_ln); me21_ln.set_visible(False)


    att1_lines = make_body_axes_artists_3d(ax, colors=triad_colors) if show_axes else None
    att2_lines = make_body_axes_artists_3d(ax, colors=triad_colors) if show_axes else None

    fov_art = {"coll": None, "rim": None, "edges": []}

    def _clear_fov():
        for k in ("coll", "rim"):
            art = fov_art.get(k)
            if art is not None:
                try:
                    art.remove()
                except Exception:
                    pass
                fov_art[k] = None
        for ln in fov_art.get("edges", []):
            try:
                ln.remove()
            except Exception:
                pass
        fov_art["edges"] = []

    def _pos3(p_like):
        p = np.asarray(p_like, float).ravel()
        return p[:3] if p.size >= 3 else np.array([p[0], p[1], 0.0], float)

    def _R_exec(agent, idx):
        key = "exec_att1" if agent == 1 else "exec_att2"
        L = frames_dict.get(key)
        if not L or idx >= len(L) or "R" not in L[idx]:
            raise KeyError(f"Missing attitude for agent {agent} at frame {idx} (need frames_dict['{key}'][{idx}]['R']).")
        R = np.asarray(L[idx]["R"], float)
        if R.shape != (3, 3):
            raise ValueError(f"Attitude R for agent {agent} at frame {idx} must be 3x3, got {R.shape}.")
        return R


    def _set3d(ln, xs, ys, zs):
        if ln is not None:
            ln.set_data(xs, ys); ln.set_3d_properties(zs)

    def _meas_ray(p_obs, R_wb, az, el, L):
        c = np.cos(el)
        v_b = np.array([c * np.cos(az), c * np.sin(az), np.sin(el)])
        v_w = R_wb.T @ v_b
        pF = p_obs + L * v_w
        return p_obs, pF

    def _draw_fov(idx, p, R_wb):
        _clear_fov()
        if not (t_fov.value and fov_cfg.get("enabled", False)):
            return
        col = fov_cfg.get("color", "C1")
        if seen_mask and idx < len(seen_mask) and bool(seen_mask[idx]):
            col = "tab:green"
        x_def = np.r_[p, [0, 0, 0]]
        if fov_cfg.get("type", "cone") == "pinhole" and cfg.get("camera") is not None:
            coll, edges = draw_camera_frustum_3d(
                ax, x_def=x_def, cam_cfg=cfg["camera"], color=col,
                alpha=fov_cfg.get("alpha", 0.15), R_wb=R_wb
            )
            coll.set_label("_nolegend_")
            for ln in edges:
                ln.set_label("_nolegend_")
            fov_art["coll"], fov_art["rim"], fov_art["edges"] = coll, None, edges
        else:
            coll, rim = draw_fov_cone_3d(
                ax, x_def, fov_cfg, color=col, alpha=fov_cfg.get("alpha", 0.15),
                align=att_cfg.get("align", "x"), R_wb=R_wb
            )
            coll.set_label("_nolegend_")
            if rim is not None:
                rim.set_label("_nolegend_")
            fov_art["coll"], fov_art["rim"] = coll, rim

    s_frame = W.IntSlider(min=0, max=n_frames - 1, step=1, value=0, description="frame")
    s_azim  = W.IntSlider(min=-180, max=180, step=1, value=45, description="azim")
    s_elev  = W.IntSlider(min=-10,  max=90,  step=1, value=25, description="elev")

    t_plan  = W.Checkbox(value=True,              description="show plan")
    t_axes  = W.Checkbox(value=show_axes,         description="show axes")
    t_fov   = W.Checkbox(value=show_fov,          description="show FOV")
    t_est   = W.Checkbox(value=do_est_default,    description="show estimates",
                         disabled=not do_est_default)
    t_meas  = W.Checkbox(value=show_meas_default, description="show meas rays",
                         disabled=not do_est_default)

    play = W.Play(min=0, max=n_frames - 1, step=1, interval=50, value=0)
    W.jslink((play, "value"), (s_frame, "value"))

    leg_main = None
    leg_axes = None

    def build_main_legend(include_est=False):
        nonlocal leg_main
        if leg_main is not None:
            try:
                leg_main.remove()
            except Exception:
                pass
            leg_main = None
        handles = [plan1_ln, plan2_ln, exe1_ln, exe2_ln]
        if include_est and do_est_default:
            handles += [est12_ln, est21_ln]
        H, Lbls = _legend_clean(handles)
        leg_main = ax.legend(H, Lbls, loc="upper left") if H else None
        if leg_axes is not None and leg_main is not None:
            ax.add_artist(leg_axes)

    def ensure_triad_legend():
        nonlocal leg_axes
        if leg_axes is None:
            proxies = [Line2D([0], [0], lw=2, color=c, label=lab)
                       for c, lab in zip(triad_colors, triad_labels)]
            leg_axes = ax.legend(
                proxies, triad_labels, loc=triad_leg_loc,
                ncol=triad_leg_ncol, title=triad_leg_title
            )
            for p in proxies:
                p.set_label("_nolegend_")
            if leg_main is not None:
                ax.add_artist(leg_main)

    build_main_legend(include_est=t_est.value)
    if show_axes:
        ensure_triad_legend()

    def redraw(f):
        if t_plan.value and plan_hist1:
            ph1 = plan_hist1[min(f, len(plan_hist1) - 1)]
            if ph1: xs, ys, zs = zip(*ph1); _set3d(plan1_ln, xs, ys, zs)
            else:   _set3d(plan1_ln, [], [], [])
            ph2 = plan_hist2[min(f, len(plan_hist2) - 1)] if plan_hist2 else []
            if ph2: xs, ys, zs = zip(*ph2); _set3d(plan2_ln, xs, ys, zs)
            else:   _set3d(plan2_ln, [], [], [])
        else:
            _set3d(plan1_ln, [], [], [])
            _set3d(plan2_ln, [], [], [])

        xs1 = [p[0] for p in exec1[: f + 1]]; ys1 = [p[1] for p in exec1[: f + 1]]; zs1 = [p[2] for p in exec1[: f + 1]]
        xs2 = [p[0] for p in exec2[: f + 1]]; ys2 = [p[1] for p in exec2[: f + 1]]; zs2 = [p[2] for p in exec2[: f + 1]]
        _set3d(exe1_ln, xs1, ys1, zs1)
        _set3d(exe2_ln, xs2, ys2, zs2)

        p1 = np.asarray(exec1[min(f, len(exec1) - 1)])
        p2 = np.asarray(exec2[min(f, len(exec2) - 1)])
        dot1.set_data([p1[0]], [p1[1]]); dot1.set_3d_properties([p1[2]])
        dot2.set_data([p2[0]], [p2[1]]); dot2.set_3d_properties([p2[2]])

        if t_axes.value:
            if att1_lines: update_body_axes_artists_3d(att1_lines, p1, _R_exec(1, f), L=L_tri)
            if att2_lines: update_body_axes_artists_3d(att2_lines, p2, _R_exec(2, f), L=L_tri)
        else:
            for L in (att1_lines, att2_lines):
                if L:
                    for ln in (L["bx"], L["by"], L["bz"]):
                        _set3d(ln, [], [], [])

        if t_est.value and est12:
            xs, ys, zs = zip(*est12[: f + 1]); _set3d(est12_ln, xs, ys, zs)
        else:
            _set3d(est12_ln, [], [], [])
        if t_est.value and est21:
            xs, ys, zs = zip(*est21[: f + 1]); _set3d(est21_ln, xs, ys, zs)
        else:
            _set3d(est21_ln, [], [], [])

        if perf_skip_fov_every < 1:
            kskip = 1
        else:
            kskip = perf_skip_fov_every
        if f % kskip == 0:
            p_def = np.asarray(exec2[f] if agent_id == 2 else exec1[f])
            R_def = _R_exec(agent_id, f)
            _draw_fov(f, p_def, R_def)

        if t_meas.value and do_est_default:
            if me12 and f < len(me12) and me12[f] is not None:
                p_obs, z = me12[f]
                R1 = _R_exec(1, f)
                p0, pF = _meas_ray(np.asarray(p_obs, float), R1, float(z[0]), float(z[1]), L_meas)
                me12_ln.set_data([p0[0], pF[0]], [p0[1], pF[1]])
                me12_ln.set_3d_properties([p0[2], pF[2]])
            else:
                me12_ln.set_data([], []); me12_ln.set_3d_properties([])

            if me21 and f < len(me21) and me21[f] is not None:
                p_obs, z = me21[f]
                R2 = _R_exec(2, f)
                p0, pF = _meas_ray(np.asarray(p_obs, float), R2, float(z[0]), float(z[1]), L_meas)
                me21_ln.set_data([p0[0], pF[0]], [p0[1], pF[1]])
                me21_ln.set_3d_properties([p0[2], pF[2]])
            else:
                me21_ln.set_data([], []); me21_ln.set_3d_properties([])
        else:
            me12_ln.set_data([], []); me12_ln.set_3d_properties([])
            me21_ln.set_data([], []); me21_ln.set_3d_properties([])

        ax.view_init(elev=s_elev.value, azim=s_azim.value)
        fig.canvas.draw_idle()

    s_frame.observe(lambda ch: redraw(ch["new"]), names="value")
    s_azim.observe(lambda ch: redraw(s_frame.value), names="value")
    s_elev.observe(lambda ch: redraw(s_frame.value), names="value")

    def _rebuild_and_redraw(_=None):
        build_main_legend(include_est=t_est.value)
        redraw(s_frame.value)

    t_plan.observe(_rebuild_and_redraw, names="value")
    t_est.observe(_rebuild_and_redraw,  names="value")
    t_axes.observe(lambda ch: redraw(s_frame.value), names="value")
    t_fov.observe(lambda ch: redraw(s_frame.value), names="value")
    t_meas.observe(lambda ch: redraw(s_frame.value), names="value")

    redraw(0)
    ui = W.VBox([
        W.HBox([play, s_frame]),
        W.HBox([s_azim, s_elev, t_plan, t_axes, t_fov, t_est, t_meas]),
    ])
    display(ui)


# -------------------- N-aware 3D animation --------------------
def animate_rollout_3d_N(frames_dict, save_path="traj_3D.gif", fps=20, cfg=None,
                         show_fov=True, show_axes=True, show_est=True, triads="fov"):
    """
    N-aware animation:
      - reads plan_hist_all / exec_xyz_all / exec_att_all (falls back to numbered keys)
      - draws FOV for a chosen agent/target
      - overlays any est{i}{j}_xyz and meas{i}{j}_azel (supports nested frames_dict['est'])
    triads: "fov" (only defender), "all" (all agents), or "none".
    """
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    import shutil
    from matplotlib import animation
    from matplotlib.animation import FFMpegWriter, PillowWriter

    if cfg is None:
        raise ValueError("cfg must be provided")

    viz_cfg  = cfg.get("viz", {})
    fov_cfg  = cfg.get("fov", {})
    att_cfg  = cfg.get("att", {})
    N        = int(cfg.get("N", max(2, len(frames_dict.get("exec_xyz_all", [])))))

    # -------------------- pull data (N-aware, with fallbacks) --------------------
    plan_all = frames_dict.get("plan_hist_all")
    exec_all = frames_dict.get("exec_xyz_all")
    att_all  = frames_dict.get("exec_att_all")

    if plan_all is None:
        plan_all = [frames_dict.get(f"plan_hist{a+1}", []) for a in range(N)]
    if exec_all is None:
        exec_all = [frames_dict.get(f"exec{a+1}_xyz", []) for a in range(N)]
    if att_all is None:
        att_all = [frames_dict.get(f"exec_att{a+1}", []) for a in range(N)]

    # 2D→3D coercion
    def _as3_hist(seq):
        if not seq: return []
        if seq and seq[0] and len(seq[0][0]) == 2:
            return [[(x, y, 0.0) for (x, y) in fr] for fr in seq]
        return seq
    def _as3_exec(seq):
        if not seq: return []
        if seq and len(seq[0]) == 2:
            return [(x, y, 0.0) for (x, y) in seq]
        return seq
    plan_all = [ _as3_hist(pl) for pl in plan_all ]
    exec_all = [ _as3_exec(ex) for ex in exec_all ]

    # Frame count
    n_frames = min([len(ex) for ex in exec_all if ex]) if any(exec_all) else 0
    if n_frames < 2:
        if n_frames == 1:
            for a in range(len(exec_all)):
                if exec_all[a]:
                    exec_all[a] = exec_all[a] + [exec_all[a][-1]]
                if plan_all[a]:
                    plan_all[a] = plan_all[a] + [plan_all[a][-1]]
            n_frames = 2
        else:
            raise ValueError("No frames to animate.")

    # FOV agent/target
    fov_enabled = bool(fov_cfg.get("enabled", False))
    fov_agent   = max(1, min(N, int(fov_cfg.get("agent", 2)))) - 1
    fov_target  = fov_cfg.get("target")
    if fov_target is None or int(fov_target) < 1 or int(fov_target) > N or (int(fov_target)-1) == fov_agent:
        fov_target = next((j+1 for j in range(N) if j != fov_agent), 1)
    fov_target -= 1

    seen_mask  = list(frames_dict.get("fov_seen_mask", []) or [])
    if seen_mask:
        seen_mask = seen_mask[:n_frames]
    axis_hist  = frames_dict.get("fov_axis_hist", [])

    # ---------------- discover estimate/measurement series (nested-aware) -------
    est_pairs = []      # list[(i,j)]
    est_series = {}     # {(i,j): [(x,y,z), ...]}
    meas_series = {}    # {(i,j): [(p_obs, [az,el]) or None, ...]}
    if show_est:
        pat_est  = re.compile(r"^est(\d+)(\d+)_xyz$")
        pat_meas = re.compile(r"^meas(\d+)(\d+)_azel$")
        merged = dict(frames_dict)
        if isinstance(frames_dict.get("est"), dict):
            merged.update(frames_dict["est"])
        for k, v in merged.items():
            m = pat_est.match(k)
            if m:
                i, j = int(m.group(1))-1, int(m.group(2))-1
                est_pairs.append((i, j))
                est_series[(i, j)] = v
        for k, v in merged.items():
            m = pat_meas.match(k)
            if m:
                i, j = int(m.group(1))-1, int(m.group(2))-1
                meas_series[(i, j)] = v

    # -------------------- figure / axes --------------------
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.grid(True)

    # axis label scaling
    scale = float(viz_cfg.get("axis_scale", 1.0))
    unit  = str(viz_cfg.get("axis_unit", "m"))
    label_only = bool(viz_cfg.get("axis_label_only", True))
    _label_axes_3d(ax, scale=scale, unit=unit, label_only=label_only)

    # arena bounds
    ar = cfg.get("arena", {})
    if ar.get("type") == "box":
        ax.set_xlim(ar["xmin"], ar["xmax"])
        ax.set_ylim(ar["ymin"], ar["ymax"])
        ax.set_zlim(ar.get("zmin", -0.5), ar.get("zmax", 0.5))
    elif ar.get("type") == "sphere":
        cx, cy, cz, R = ar["cx"], ar["cy"], ar["cz"], ar["r"]
        ax.set_xlim(cx-R, cx+R); ax.set_ylim(cy-R, cy+R); ax.set_zlim(cz-R, cz+R)
    else:
        pts = np.array([p for ea in exec_all for p in ea], float)
        if pts.size:
            mn = pts.min(0); mx = pts.max(0)
            span = np.maximum(mx - mn, 1e-3); pad = 0.1 * span
            lo, hi = mn - pad, mx + pad
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])

    # -------------------- artists per agent --------------------
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', None) or [
        "tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown",
        "tab:pink","tab:gray","tab:olive","tab:cyan"
    ]

    plan_lns, exec_lns, dots = [], [], []
    for a in range(N):
        ca = color_cycle[a % len(color_cycle)]
        (pl_ln,) = ax.plot([], [], [], '--', lw=1, alpha=0.65, label=f'Plan P{a+1}', color=ca)
        (ex_ln,) = ax.plot([], [], [], '-',  lw=2,              label=f'Exec P{a+1}', color=ca)
        (dot,)  = ax.plot([], [], [], 'o',  ms=6, color=ca, mec='k', mew=0.6, label='_nolegend_')
        plan_lns.append(pl_ln); exec_lns.append(ex_ln); dots.append(dot)

    # triads
    triads_mode = (triads or "fov").lower()
    triad_idxs = [] if not show_axes or triads_mode == "none" else (
        list(range(N)) if triads_mode == "all" else [fov_agent]
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

    # estimates / meas
    est_lns, meas_lns = {}, {}
    if est_pairs:
        for (i, j) in est_pairs:
            col = color_cycle[j % len(color_cycle)]
            (ln,) = ax.plot([], [], [], ':', lw=1.8, marker='x', ms=5.5, mew=1.0,
                            label=f'P{j+1} est by P{i+1}', color=col)
            est_lns[(i, j)] = ln
            if (i, j) in meas_series and viz_cfg.get("show_meas", False):
                (mln,) = ax.plot([], [], [], '-', lw=0.9, alpha=0.35, color=col, label='_nolegend_')
                meas_lns[(i, j)] = mln

    # legend
    def _legend_clean(H):
        out = []
        for h in H:
            if h is None: continue
            lab = getattr(h, "get_label", lambda: "")()
            if lab and not str(lab).startswith("_"): out.append(h)
        return out, [h.get_label() for h in out]
    H, L = _legend_clean(plan_lns + exec_lns + list(est_lns.values()))
    if H: ax.legend(H, L, loc='upper left')

    # FOV artists
    fov_art = {'coll': None, 'rim': None, 'edges': []}
    def _clear_fov():
        for k in ('coll','rim'):
            art = fov_art.get(k)
            if art is not None:
                try: art.remove()
                except Exception: pass
                fov_art[k] = None
        for ln in fov_art.get('edges', []):
            try: ln.remove()
            except Exception: pass
        fov_art['edges'] = []

    def _R_exec(agent, idx):
        L = att_all[agent]
        if not L or idx >= len(L) or "R" not in L[idx]:
            raise KeyError(f"Missing attitude for agent {agent+1} at frame {idx}")
        R = np.asarray(L[idx]["R"], float)
        if R.shape != (3,3):
            raise ValueError(f"Attitude R for agent {agent+1} at frame {idx} must be 3x3, got {R.shape}.")
        return R

    def _set_fov(p, R_wb, idx):
        if not (show_fov and fov_enabled):
            _clear_fov(); return
        col = fov_cfg.get("color","C1")
        if seen_mask and idx < len(seen_mask) and bool(seen_mask[idx]):
            col = 'tab:green'
        x_def = np.r_[p, [0,0,0]]
        if fov_cfg.get("type","cone") == "pinhole" and cfg.get("camera") is not None:
            coll, edges = draw_camera_frustum_3d(
                ax, x_def=x_def, cam_cfg=cfg["camera"], color=col,
                alpha=fov_cfg.get("alpha",0.15), R_wb=R_wb
            )
            coll.set_label('_nolegend_')
            for ln in edges: ln.set_label('_nolegend_')
            fov_art['coll'], fov_art['rim'], fov_art['edges'] = coll, None, edges
        else:
            coll, rim = draw_fov_cone_3d(
                ax, x_def, fov_cfg, color=col, alpha=fov_cfg.get("alpha",0.15),
                align=att_cfg.get('align','x'), R_wb=R_wb
            )
            coll.set_label('_nolegend_')
            if rim is not None: rim.set_label('_nolegend_')
            fov_art['coll'], fov_art['rim'] = coll, rim

    def _set3d(ln, xs, ys, zs):
        if ln is not None:
            ln.set_data(xs, ys); ln.set_3d_properties(zs)

    def init():
        for ln in plan_lns + exec_lns + list(est_lns.values()) + list(meas_lns.values()):
            if ln is not None:
                ln.set_data([], []); ln.set_3d_properties([])
        for dot in dots:
            dot.set_data([], []); dot.set_3d_properties([])
        for a in triad_idxs:
            L = triad_art[a]
            if L:
                for key in ('bx','by','bz'):
                    L[key].set_data([], []); L[key].set_3d_properties([])
        _clear_fov()
        return []

    def update(f):
        # plans + exec + dots
        for a in range(N):
            if plan_all[a] and f < len(plan_all[a]) and plan_all[a][f]:
                xs, ys, zs = zip(*plan_all[a][f]); _set3d(plan_lns[a], xs, ys, zs)
            else:
                _set3d(plan_lns[a], [], [], [])
            xs = [p[0] for p in exec_all[a][:f+1]]
            ys = [p[1] for p in exec_all[a][:f+1]]
            zs = [p[2] for p in exec_all[a][:f+1]]
            _set3d(exec_lns[a], xs, ys, zs)
            if exec_all[a]:
                x,y,z = exec_all[a][min(f, len(exec_all[a])-1)]
                dots[a].set_data([x],[y]); dots[a].set_3d_properties([z])

        # triads
        for a in triad_idxs:
            if triad_art[a] and att_all[a]:
                x,y,z = exec_all[a][min(f, len(exec_all[a])-1)]
                R = _R_exec(a, f)
                update_body_axes_artists_3d(triad_art[a], np.array([x,y,z], float), R, L=tuple(L_tri))

        # estimates + meas
        if est_pairs:
            def _meas_ray(p_obs, R_wb, az, el, Lm):
                c = np.cos(el)
                v_b = np.array([c*np.cos(az), c*np.sin(az), np.sin(el)])
                v_w = R_wb.T @ v_b
                pF = p_obs + Lm * v_w
                return p_obs, pF
            L_meas = float(viz_cfg.get("meas_len", cfg.get("camera", {}).get("far", 15.0)))
            for (i, j) in est_pairs:
                est = est_series.get((i, j), [])
                if est:
                    xs, ys, zs = zip(*est[:f+1]) if (f+1) <= len(est) else zip(*est)
                    _set3d(est_lns[(i, j)], xs, ys, zs)
                else:
                    _set3d(est_lns[(i, j)], [], [], [])
                if (i, j) in meas_series and (i, j) in meas_lns:
                    me = meas_series[(i, j)]
                    if f < len(me) and me[f] is not None and att_all[i]:
                        p_obs, z = me[f]
                        R_i = _R_exec(i, f)
                        p0, pF = _meas_ray(np.asarray(p_obs, float), R_i, float(z[0]), float(z[1]), L_meas)
                        meas_lns[(i, j)].set_data([p0[0], pF[0]], [p0[1], pF[1]])
                        meas_lns[(i, j)].set_3d_properties([p0[2], pF[2]])
                    else:
                        meas_lns[(i, j)].set_data([], []); meas_lns[(i, j)].set_3d_properties([])

        # FOV
        if fov_enabled and att_all[fov_agent]:
            x,y,z = exec_all[fov_agent][min(f, len(exec_all[fov_agent])-1)]
            R_def = _R_exec(fov_agent, f)
            _clear_fov(); _set_fov(np.array([x,y,z], float), R_def, f)
        else:
            _clear_fov()
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



# -------------------- N-aware interactive viewer --------------------
def interactive_rollout_3d_N(
    frames_dict,
    cfg,
    title="Interactive 3D rollout (N-aware)",
    show_fov=True,
    show_axes=True,
    triads="fov",                # "fov" (only defender), "all", or "none"
    perf_skip_fov_every: int = 1
):
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    import ipywidgets as W
    from IPython.display import display
    from matplotlib.lines import Line2D

    viz_cfg  = cfg.get("viz", {})
    fov_cfg  = cfg.get("fov", {})
    att_cfg  = cfg.get("att", {})
    N        = int(cfg.get("N", max(2, len(frames_dict.get("exec_xyz_all", [])))))

    # ---------- pull data with fallbacks ----------
    plan_all = frames_dict.get("plan_hist_all")
    exec_all = frames_dict.get("exec_xyz_all")
    att_all  = frames_dict.get("exec_att_all")
    if plan_all is None: plan_all = [frames_dict.get(f"plan_hist{a+1}", []) for a in range(N)]
    if exec_all is None: exec_all = [frames_dict.get(f"exec{a+1}_xyz", []) for a in range(N)]
    if att_all  is None: att_all  = [frames_dict.get(f"exec_att{a+1}", []) for a in range(N)]

    # 2D→3D coercion (matches animator)
    def _as3_hist(seq):
        if not seq: return []
        if seq and seq[0] and len(seq[0][0]) == 2:
            return [[(x, y, 0.0) for (x, y) in fr] for fr in seq]
        return seq
    def _as3_exec(seq):
        if not seq: return []
        if seq and len(seq[0]) == 2:
            return [(x, y, 0.0) for (x, y) in seq]
        return seq
    plan_all = [ _as3_hist(pl) for pl in plan_all ]
    exec_all = [ _as3_exec(ex) for ex in exec_all ]

    # frames = min length across agents (matches animator)
    n_frames = min([len(ex) for ex in exec_all if ex]) if any(exec_all) else 0
    if n_frames < 2:
        if n_frames == 1:
            for a in range(N):
                if exec_all[a]:
                    exec_all[a] = exec_all[a] + [exec_all[a][-1]]
                if plan_all[a]:
                    plan_all[a] = plan_all[a] + [plan_all[a][-1]]
            n_frames = 2
        else:
            raise ValueError("No frames to display.")

    seen_mask  = list(frames_dict.get("fov_seen_mask", []) or [])
    if seen_mask:
        seen_mask = seen_mask[:n_frames]

    # ---------- discover any estimates/meas (nested-aware) ----------
    est_pairs, est_series, meas_series = [], {}, {}
    pat_est  = re.compile(r"^est(\d+)(\d+)_xyz$")
    pat_meas = re.compile(r"^meas(\d+)(\d+)_azel$")
    merged = dict(frames_dict)
    if isinstance(frames_dict.get("est"), dict):
        merged.update(frames_dict["est"])
    for k, v in merged.items():
        m = pat_est.match(k)
        if m:
            i, j = int(m.group(1))-1, int(m.group(2))-1
            est_pairs.append((i, j))
            est_series[(i, j)] = v
    for k, v in merged.items():
        m = pat_meas.match(k)
        if m:
            i, j = int(m.group(1))-1, int(m.group(2))-1
            meas_series[(i, j)] = v

    # ---------- figure/axes ----------
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
    if ar.get("type") == "box":
        ax.set_xlim(ar["xmin"], ar["xmax"])
        ax.set_ylim(ar["ymin"], ar["ymax"])
        ax.set_zlim(ar.get("zmin", -0.5), ar.get("zmax", 0.5))
    elif ar.get("type") == "sphere":
        cx, cy, cz, R = ar["cx"], ar["cy"], ar["cz"], ar["r"]
        ax.set_xlim(cx-R, cx+R); ax.set_ylim(cy-R, cy+R); ax.set_zlim(cz-R, cz+R)
    else:
        pts = np.array([p for ea in exec_all for p in ea], float)
        if pts.size:
            mn, mx = pts.min(0), pts.max(0)
            span = np.maximum(mx-mn, 1e-3); pad = 0.1*span
            lo, hi = mn - pad, mx + pad
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])

    # ---------- artists per agent ----------
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', None) or [
        "tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown",
        "tab:pink","tab:gray","tab:olive","tab:cyan"
    ]
    def _set3d(ln, xs, ys, zs):
        if ln is not None:
            ln.set_data(xs, ys); ln.set_3d_properties(zs)

    plan_lns, exec_lns, dots = [], [], []
    for a in range(N):
        ca = color_cycle[a % len(color_cycle)]
        (pl_ln,) = ax.plot([], [], [], "--", lw=1, alpha=0.65, label=f"Plan P{a+1}", color=ca)
        (ex_ln,) = ax.plot([], [], [], "-",  lw=2,              label=f"Exec P{a+1}", color=ca)
        (dot,)  = ax.plot([], [], [], "o",  ms=6, color=ca, mec="k", mew=0.6, label="_nolegend_")
        plan_lns.append(pl_ln); exec_lns.append(ex_ln); dots.append(dot)

    # triads
    triads_mode = (triads or "fov").lower()
    triad_idxs = [] if not show_axes or triads_mode == "none" else (
        list(range(N)) if triads_mode == "all" else [max(0, int(fov_cfg.get("agent",2))-1)]
    )
    triad_colors = tuple(viz_cfg.get("triad_colors", ("tab:red","tab:green","tab:blue")))
    triad_labels = tuple(viz_cfg.get("triad_labels", ("x_b (boresight)","y_b","z_b")))
    triad_leg_loc  = viz_cfg.get("triad_leg_loc", "lower left")
    triad_leg_ncol = int(viz_cfg.get("triad_leg_ncol", 3))
    triad_leg_title= viz_cfg.get("triad_leg_title", "Body axes")
    L_tri = tuple(viz_cfg.get("triad_len", (0.35, 0.35, 0.55)))

    triad_art = [None for _ in range(N)]
    for a in triad_idxs:
        triad_art[a] = make_body_axes_artists_3d(ax, colors=triad_colors)
        for key in ("bx","by","bz"):
            triad_art[a][key].set_label("_nolegend_")
    leg_axes = add_triad_legend(ax, triad_colors, triad_labels, triad_leg_loc, triad_leg_ncol, triad_leg_title) if triad_idxs else None

    # estimates
    est_lns, meas_lns = {}, {}
    if est_pairs:
        for (i, j) in est_pairs:
            col = color_cycle[j % len(color_cycle)]
            (ln,) = ax.plot([], [], [], ":", lw=1.8, marker="x", ms=5.5, mew=1.0,
                            label=f"P{j+1} est by P{i+1}", color=col)
            est_lns[(i, j)] = ln
            if (i, j) in meas_series and viz_cfg.get("show_meas", False):
                (mln,) = ax.plot([], [], [], "-", lw=0.9, alpha=0.35, color=col, label="_nolegend_")
                meas_lns[(i, j)] = mln

    # main legend
    def _legend_clean(H):
        out = []
        for h in H:
            if h is None: continue
            lab = getattr(h, "get_label", lambda: "")()
            if lab and not str(lab).startswith("_"):
                out.append(h)
        return out, [h.get_label() for h in out]
    H, L = _legend_clean(plan_lns + exec_lns + list(est_lns.values()))
    if H: ax.legend(H, L, loc="upper left")

    # FOV setup
    fov_on = bool(fov_cfg.get("enabled", False)) and show_fov
    fov_agent = max(1, min(N, int(fov_cfg.get("agent",2)))) - 1
    fov_target = fov_cfg.get("target")
    if fov_target is None or int(fov_target) < 1 or int(fov_target) > N or (int(fov_target)-1) == fov_agent:
        fov_target = next((j+1 for j in range(N) if j != fov_agent), 1)
    fov_target -= 1

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
        L = att_all[agent]
        if not L or idx >= len(L) or "R" not in L[idx]:
            raise KeyError(f"Missing attitude for agent {agent+1} at frame {idx}")
        R = np.asarray(L[idx]["R"], float)
        if R.shape != (3,3):
            raise ValueError(f"Attitude R for agent {agent+1} at frame {idx} must be 3x3, got {R.shape}.")
        return R

    def _set_fov(p, R_wb, idx):
        _clear_fov()
        if not (fov_on and show_fov): return
        col = fov_cfg.get("color","C1")
        if seen_mask and idx < len(seen_mask) and bool(seen_mask[idx]):
            col = "tab:green"
        x_def = np.r_[p, [0,0,0]]
        if fov_cfg.get("type","cone") == "pinhole" and cfg.get("camera") is not None:
            coll, edges = draw_camera_frustum_3d(ax, x_def=x_def, cam_cfg=cfg["camera"],
                                                 color=col, alpha=fov_cfg.get("alpha",0.15), R_wb=R_wb)
            for ln in edges: ln.set_label("_nolegend_")
            fov_art["coll"], fov_art["rim"], fov_art["edges"] = coll, None, edges
        else:
            coll, rim = draw_fov_cone_3d(ax, x_def, fov_cfg, color=col, alpha=fov_cfg.get("alpha",0.15),
                                         align=att_cfg.get("align","x"), R_wb=R_wb)
            fov_art["coll"], fov_art["rim"] = coll, rim

    # ---------- widgets ----------
    s_frame = W.IntSlider(min=0, max=n_frames-1, step=1, value=0, description="frame")
    s_azim  = W.IntSlider(min=-180, max=180, step=1, value=45, description="azim")
    s_elev  = W.IntSlider(min=-10,  max=90,  step=1, value=25, description="elev")
    t_plan  = W.Checkbox(value=True, description="show plan")
    t_axes  = W.Checkbox(value=show_axes, description="show axes")
    t_fov   = W.Checkbox(value=show_fov and fov_on, description="show FOV")
    t_est   = W.Checkbox(value=bool(est_pairs), description="show estimates", disabled=(len(est_pairs)==0))
    t_meas  = W.Checkbox(value=bool(viz_cfg.get("show_meas", False)) and bool(est_pairs),
                         description="show meas rays", disabled=(len(est_pairs)==0))

    play = W.Play(min=0, max=n_frames-1, step=1, interval=50, value=0)
    W.jslink((play, "value"), (s_frame, "value"))

    def redraw(f):
        # plans + execs + dots
        for a in range(N):
            if t_plan.value and plan_all[a] and f < len(plan_all[a]) and plan_all[a][f]:
                xs, ys, zs = zip(*plan_all[a][f]); _set3d(plan_lns[a], xs, ys, zs)
            else:
                _set3d(plan_lns[a], [], [], [])
            xs = [p[0] for p in exec_all[a][:f+1]]
            ys = [p[1] for p in exec_all[a][:f+1]]
            zs = [p[2] for p in exec_all[a][:f+1]]
            _set3d(exec_lns[a], xs, ys, zs)
            x,y,z = exec_all[a][min(f, len(exec_all[a])-1)]
            dots[a].set_data([x],[y]); dots[a].set_3d_properties([z])

        # triads
        if t_axes.value and triads.lower() != "none":
            triad_idxs_local = list(range(N)) if triads.lower()=="all" else [max(0, int(fov_cfg.get("agent",2))-1)]
            for a in triad_idxs_local:
                if a < len(att_all) and att_all[a] and triad_art[a]:
                    x,y,z = exec_all[a][min(f, len(exec_all[a])-1)]
                    R = _R_exec(a, f)
                    update_body_axes_artists_3d(triad_art[a], np.array([x,y,z], float), R, L=tuple(viz_cfg.get("triad_len", (0.35,0.35,0.55))))
        else:
            for a in range(N):
                if triad_art[a]:
                    for key in ("bx","by","bz"):
                        triad_art[a][key].set_data([], []); triad_art[a][key].set_3d_properties([])

        # estimates + meas
        for (i, j) in est_pairs:
            ln = est_lns[(i, j)]
            if t_est.value and est_series.get((i, j)):
                est = est_series[(i, j)]
                xs, ys, zs = zip(*est[:f+1]) if (f+1) <= len(est) else zip(*est)
                _set3d(ln, xs, ys, zs)
            else:
                _set3d(ln, [], [], [])
            if (i, j) in meas_lns:
                me = meas_series.get((i, j), [])
                if t_est.value and t_meas.value and f < len(me) and me[f] is not None and att_all[i]:
                    p_obs, z = me[f]
                    R_i = _R_exec(i, f)
                    c = np.cos(float(z[1]))
                    v_b = np.array([c*np.cos(float(z[0])), c*np.sin(float(z[0])), np.sin(float(z[1]))])
                    v_w = R_i.T @ v_b
                    Lm  = float(viz_cfg.get("meas_len", cfg.get("camera", {}).get("far", 15.0)))
                    p0 = np.asarray(p_obs, float); pF = p0 + Lm*v_w
                    meas_lns[(i, j)].set_data([p0[0], pF[0]], [p0[1], pF[1]])
                    meas_lns[(i, j)].set_3d_properties([p0[2], pF[2]])
                else:
                    meas_lns[(i, j)].set_data([], []); meas_lns[(i, j)].set_3d_properties([])

        # FOV
        kskip = max(1, int(perf_skip_fov_every))
        if t_fov.value and show_fov and (f % kskip == 0):
            x,y,z = exec_all[fov_agent][min(f, len(exec_all[fov_agent])-1)]
            _set_fov(np.array([x,y,z], float), _R_exec(fov_agent, f), f)
        else:
            _clear_fov()

        ax.view_init(elev=s_elev.value, azim=s_azim.value)
        fig.canvas.draw_idle()

    # ---------- widgets / wiring ----------
    s_frame = W.IntSlider(min=0, max=n_frames-1, step=1, value=0, description="frame")
    s_azim  = W.IntSlider(min=-180, max=180, step=1, value=45, description="azim")
    s_elev  = W.IntSlider(min=-10,  max=90,  step=1, value=25, description="elev")
    t_plan  = W.Checkbox(value=True, description="show plan")
    t_axes  = W.Checkbox(value=show_axes, description="show axes")
    t_fov   = W.Checkbox(value=show_fov and bool(fov_cfg.get("enabled", False)), description="show FOV")
    t_est   = W.Checkbox(value=bool(est_pairs), description="show estimates", disabled=(len(est_pairs)==0))
    t_meas  = W.Checkbox(value=bool(viz_cfg.get("show_meas", False)) and bool(est_pairs),
                         description="show meas rays", disabled=(len(est_pairs)==0))
    play = W.Play(min=0, max=n_frames-1, step=1, interval=50, value=0)
    W.jslink((play, "value"), (s_frame, "value"))

    s_frame.observe(lambda ch: redraw(ch["new"]), names="value")
    for w in (s_azim, s_elev, t_plan, t_axes, t_fov, t_est, t_meas):
        w.observe(lambda ch: redraw(s_frame.value), names="value")

    # UI
    redraw(0)
    ui = W.VBox([
        W.HBox([play, s_frame]),
        W.HBox([s_azim, s_elev, t_plan, t_axes, t_fov, t_est, t_meas]),
    ])
    display(ui)



# -------------------- Triad legend (unchanged API; silences proxies) --------------------
def add_triad_legend(ax, colors=('tab:red','tab:green','tab:blue'),
                     labels=('x_b (boresight)', 'y_b', 'z_b'),
                     loc='lower left', ncol=1, title='Body axes',
                     keep_legend=None):
    proxies = [Line2D([0], [0], lw=2, color=c, label=lab) for c, lab in zip(colors, labels)]
    leg_axes = ax.legend(proxies, labels, loc=loc, ncol=ncol, title=title)
    if keep_legend is not None:
        ax.add_artist(keep_legend)
    # keep proxies out of future legend calls
    for p in proxies:
        p.set_label('_nolegend_')
    return leg_axes