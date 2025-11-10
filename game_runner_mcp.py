import numpy as np
import torch
from dyn_models import hcw_mean_motion, hcw_discrete_mats, as_numpy_const
from rl_infer_mcp import RLPolicy  # or your infer wrapper

def run_rhc_with_rl_and_collect_frames_3d_mcp(cfg, steps=120, deterministic=True):
    D  = int(cfg["D"]); dt = float(cfg["dt"])
    n  = hcw_mean_motion(cfg["hcw"])
    Ad, Bd = hcw_discrete_mats(n, dt)
    Ad = as_numpy_const(Ad); Bd = as_numpy_const(Bd)

    def step(x,u): return Ad @ x + Bd @ u

    # arena / center
    ar = cfg["arena"]
    c = np.array([ar["cx"], ar["cy"], (ar.get("cz",0.0) if D==3 else 0.0)], float)[:D]
    umax = float(cfg["umax"])

    # init states [p,v]
    x1 = np.asarray(cfg["x0"][0], float)[:2*D].copy()
    x2 = np.asarray(cfg["x0"][1], float)[:2*D].copy()

    # ------------- obs packing variants -------------
    def _pack_obs_variant(x1, x2, c, D, variant: str):
        p1, v1 = x1[:D], x1[D:2*D]
        p2, v2 = x2[:D], x2[D:2*D]
        if variant == "train":
            obs = np.concatenate([p1-c, p2-c, (p2-p1), v1, v2])
        elif variant == "swap12":
            obs = np.concatenate([p2-c, p1-c, (p2-p1), v1, v2])
        elif variant == "rel21":
            obs = np.concatenate([p1-c, p2-c, (p1-p2), v1, v2])
        elif variant == "velswap":
            obs = np.concatenate([p1-c, p2-c, (p2-p1), v2, v1])
        else:
            raise ValueError(f"Unknown infer_obs_variant={variant}")
        return obs.astype(np.float32)

    def build_obs(x1, x2):
        variant = cfg.get("infer_obs_variant", "train")
        obs = _pack_obs_variant(x1, x2, c, D, variant)
        if cfg.get("obsnorm", False) and "obs_stats" in cfg:
            m = np.load(cfg["obs_stats"]["mean_path"]); s = np.load(cfg["obs_stats"]["std_path"])
            obs = (obs - m.astype(np.float32)) / (s.astype(np.float32) + 1e-8)
        return obs

    # ------------- policy -------------
    pol = RLPolicy(cfg, device=cfg.get("device","cpu"))  # loads ckpts from cfg

    # -------- quick diagnosis at t=0 --------
    obs0 = build_obs(x1, x2)
    a2_0 = np.asarray(pol.act_att_obs(obs0, deterministic=True), float)
    a2_0 = np.clip(a2_0, -umax, +umax)
    p2_0 = x2[:D]
    grad0 = 2.0 * (p2_0 - c)                      # ∂(d^2)/∂p_att
    print("[diag] t=0  grad·a_att =", float(np.dot(grad0, a2_0)))
    print("[diag] t=0  ||a_att|| =", float(np.linalg.norm(a2_0)))
    print("[diag] t=0  obs[:8] =", obs0[:8].tolist())

    # animator helpers (attitude placeholders)
    def _p3(x): return (float(x[0]), float(x[1]), float(x[2])) if D==3 else (float(x[0]), float(x[1]), 0.0)
    I = np.eye(3)
    plan_hist1, plan_hist2 = [], []
    plan_att1,  plan_att2  = [], []
    exec_xyz1,  exec_xyz2  = [], []
    exec_att1,  exec_att2  = [], []
    phi_hist1,  phi_hist2  = [], []
    fov_axis_hist, fov_seen_mask = [], []

    exec_xyz1.append(_p3(x1)); exec_xyz2.append(_p3(x2))
    exec_att1.append({"R": I, "phi": 0.0}); exec_att2.append({"R": I, "phi": 0.0})
    phi_hist1.append(0.0); phi_hist2.append(0.0)
    fov_axis_hist.append(None); fov_seen_mask.append(False)

    debug = bool(cfg.get("debug_actions", False))
    T_hor = int(cfg["T"])

    # ---------------- rollout ----------------
    d2_log = []
    for k in range(steps):
        obs = build_obs(x1, x2)
        u1 = pol.act_def_obs(obs, deterministic=deterministic)
        u2 = pol.act_att_obs(obs, deterministic=deterministic)
        u1 = np.clip(np.asarray(u1,float), -umax, +umax)
        u2 = np.clip(np.asarray(u2,float), -umax, +umax)

        if debug and k < 5:
            a_att = u2
            sat_att = np.mean(np.abs(a_att) >= umax - 1e-6)
            a_def = u1
            sat_def = np.mean(np.abs(a_def) >= umax - 1e-6)
            print(f"[diag] k={k:02d} ||a_def||={np.linalg.norm(a_def):.3e} "
                  f"||a_att||={np.linalg.norm(a_att):.3e} att_sat_frac={sat_att:.2f} "
                  f"def_sat_frac={sat_def:.2f} a_att={np.round(a_att,3)}")

        # distance^2 for attacker (for debugging trend)
        p2 = x2[:D]
        d2 = float(np.dot(p2 - c, p2 - c))
        d2_log.append(d2)

        x1 = step(x1, u1)
        x2 = step(x2, u2)

        plan_hist1.append([_p3(x1)] * T_hor)
        plan_hist2.append([_p3(x2)] * T_hor)
        att_stub = [{"R": I, "phi": 0.0}] * T_hor
        plan_att1.append(att_stub); plan_att2.append(att_stub)
        exec_att1.append({"R": I, "phi": 0.0}); exec_att2.append({"R": I, "phi": 0.0})
        phi_hist1.append(0.0); phi_hist2.append(0.0)
        exec_xyz1.append(_p3(x1)); exec_xyz2.append(_p3(x2))
        fov_axis_hist.append(None); fov_seen_mask.append(False)

    if debug:
        print("[diag] d2[0..10]:", [round(v,3) for v in d2_log[:11]])
        print("[diag] d2 last:", round(d2_log[-1],3), " Δ:", round(d2_log[-1]-d2_log[0],3))

    return {
        "plan_hist1": plan_hist1, "plan_hist2": plan_hist2,
        "plan_att1":  plan_att1,  "plan_att2":  plan_att2,
        "exec1_xyz":  exec_xyz1,  "exec2_xyz":  exec_xyz2,
        "exec_att1":  exec_att1,  "exec_att2":  exec_att2,
        "phi_hist1":  phi_hist1,  "phi_hist2":  phi_hist2,
        "fov_axis_hist": fov_axis_hist, "fov_seen_mask": fov_seen_mask,
    }