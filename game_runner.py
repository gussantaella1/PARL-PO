# game_runner.py
# RL-only rollout for Diff-Nash policy; supports HCW (LTI), elliptic LTV, and two-body nonlinear.
# No attitude/FOV (identity stubs)

from __future__ import annotations
from typing import Dict, Any, List
import os
import time
import numpy as np
import torch

from core.env import TorchVecEnv
from core.safety_filter import project_box_halfspace_np, velocity_cbf_halfspace_np
from core.utils import set_seed
from paper_baseline_runner import _build_step_plant_single, _identity_R, _p3
from rl_infer import RLPolicyDiff
from ukf_estimator import KF_CV, _azel_from_body_vec, _body_bearing_from_world

_WARN_ONCE_MESSAGES: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg in _WARN_ONCE_MESSAGES:
        return
    _WARN_ONCE_MESSAGES.add(msg)
    print(msg)


def _u3_from_action(u: np.ndarray, D: int) -> np.ndarray:
    u = np.asarray(u, float).reshape(-1)
    if D == 3:
        return u[:3]
    return np.array([u[0], u[1], 0.0], dtype=float)


def _v3_from_state(x: np.ndarray, D: int) -> np.ndarray:
    x = np.asarray(x, float).reshape(-1)
    v = np.zeros(3, dtype=float)
    v[:D] = x[D:2 * D]
    return v


def _build_velocity_cbf_filter(
    cfg: Dict[str, Any],
    *,
    D: int,
    dt: float,
    umax: float,
):
    sf_cfg = dict(cfg.get("safety_filter", {}) or {})
    enabled = bool(sf_cfg.get("enabled", False))
    u_lo = -float(umax)
    u_hi = +float(umax)

    if not enabled:
        def _no_filter(p: np.ndarray, v: np.ndarray, u_nom: np.ndarray) -> np.ndarray:
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


def _normalize_kf_action_access(mode: Any) -> str:
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
    kind = cfg.get("estimator_kind", "ukf")
    key = str(kind).strip().lower()
    if key not in {"ukf", "ekf"}:
        raise ValueError(f"Unsupported estimator_kind='{kind}'. Expected 'ukf' or 'ekf'.")
    return key


def _canonical_estimator_dyn_name(raw_name: Any) -> str:
    key = str(raw_name or "hcw").strip().lower()
    if key in ("elliptic_ltv", "elliptical_ltv", "th", "tschauner_hempel"):
        return "elliptic_ltv"
    return key


def _estimator_supports_dynamics(cfg: Dict[str, Any], dyn_name: str | None = None) -> bool:
    estimator_kind = _normalize_estimator_kind(cfg)
    canonical_dyn = _canonical_estimator_dyn_name(cfg.get("dynamics", "hcw") if dyn_name is None else dyn_name)
    if canonical_dyn == "hcw":
        return True
    if canonical_dyn == "elliptic_ltv":
        return estimator_kind == "ekf"
    return False


def _estimator_factory_args(cfg: Dict[str, Any], linearization_group: str) -> Dict[str, Any]:
    canonical_dyn = _canonical_estimator_dyn_name(cfg.get("dynamics", "hcw"))
    estimator_kind = _normalize_estimator_kind(cfg)
    args: Dict[str, Any] = {}
    if canonical_dyn == "hcw":
        args["dyn"] = "hcw"
        args["hcw"] = cfg.get("hcw", {})
    elif canonical_dyn == "elliptic_ltv" and estimator_kind == "ekf":
        dyn_cfg = dict(cfg.get("dyn", {}) or {})
        Ad_seq = dyn_cfg.get("Ad_seq", None)
        Bd_seq = dyn_cfg.get("Bd_seq", None)
        if Ad_seq is None or Bd_seq is None:
            raise ValueError(
                "EKF with dynamics='elliptic_ltv' requires cfg['dyn']['Ad_seq'] and cfg['dyn']['Bd_seq']."
            )
        args["dyn"] = "elliptic_ltv"
        args["ltv"] = {"Ad_seq": Ad_seq, "Bd_seq": Bd_seq}
    else:
        raise ValueError(
            f"{estimator_kind.upper()} estimator does not support dynamics='{canonical_dyn}' in this runner."
        )
    args.update(_ekf_factory_kwargs(cfg, linearization_group=linearization_group))
    return args


def _ekf_factory_kwargs(cfg: Dict[str, Any], linearization_group: str) -> Dict[str, Any]:
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
    return (float(a) + np.pi) % (2.0 * np.pi) - np.pi


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


def _combine_reproducibility_seed(seeds: List[int]) -> int:
    acc = 2166136261
    for idx, seed in enumerate(seeds):
        mix = (int(seed) + 0x9E3779B9 + idx * 0x85EBCA6B) & 0xFFFFFFFF
        acc ^= mix
        acc = (acc * 16777619) & 0xFFFFFFFF
    return int(acc or 1)


def _configure_cuda_reproducibility(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(int(seed))
    if not torch.cuda.is_available():
        return
    torch.cuda.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def run_batched_rhc_with_rl_and_collect_frames_3d(
    cfgs: List[Dict[str, Any]],
    *,
    steps: int,
    turn_len: int | None = None,
    episode_seeds: List[int] | None = None,
) -> List[Dict[str, Any]]:
    if not cfgs:
        return []

    def _canonical_batched_dyn_name(raw_name: Any) -> str:
        key = str(raw_name or "hcw").strip().lower()
        if key in ("elliptic_ltv", "elliptical_ltv", "th", "tschauner_hempel"):
            return "elliptic_ltv"
        return key

    cfg0 = cfgs[0]
    D = int(cfg0.get("D", np.asarray(cfg0["x0"]).shape[1] // 2))
    num_attackers = int(cfg0.get("num_attackers", 1))
    if num_attackers != 1:
        raise NotImplementedError("Batched eval currently supports num_attackers == 1 only.")

    dyn_name = _canonical_batched_dyn_name(cfg0.get("dynamics", "hcw"))
    if dyn_name not in ("hcw", "elliptic_ltv"):
        raise NotImplementedError("Batched eval currently supports HCW and elliptic_ltv dynamics only.")
    if dyn_name != "hcw" and bool(cfg0.get("use_kf", False)) and not _estimator_supports_dynamics(cfg0, dyn_name):
        raise NotImplementedError("Batched elliptic_ltv eval currently requires estimator_kind=ekf when use_kf=True.")

    for cfg in cfgs:
        if int(cfg.get("num_attackers", 1)) != 1:
            raise NotImplementedError("Batched eval currently supports num_attackers == 1 only.")
        if _canonical_batched_dyn_name(cfg.get("dynamics", "hcw")) != dyn_name:
            raise ValueError("All batched rollout configs must share the same dynamics model.")
        if dyn_name != "hcw" and bool(cfg.get("use_kf", False)) and not _estimator_supports_dynamics(cfg, dyn_name):
            raise NotImplementedError("Batched elliptic_ltv eval currently requires estimator_kind=ekf when use_kf=True.")

    t_fn0 = time.perf_counter()
    batch_size = len(cfgs)
    device = str(cfg0.get("device", "cpu"))
    if episode_seeds is None:
        episode_seeds = [int(cfg.get("seed", idx)) for idx, cfg in enumerate(cfgs)]
    elif len(episode_seeds) != batch_size:
        raise ValueError(
            f"episode_seeds length={len(episode_seeds)} does not match batch_size={batch_size}."
        )
    else:
        episode_seeds = [int(seed) for seed in episode_seeds]
    repro_seed = _combine_reproducibility_seed(episode_seeds)
    if torch.device(device).type == "cuda":
        _configure_cuda_reproducibility(repro_seed)
    else:
        set_seed(repro_seed)
    deterministic = bool(cfg0.get("rl_eval_deterministic", True))
    stop_on_done = bool(cfg0.get("stop_on_done", True))
    use_kf = bool(cfg0.get("use_kf", False))
    use_fuel = bool(cfg0.get("fuel", {}).get("enable"))
    obs_expected = 5 * D + (2 if use_fuel else 0)
    meas_every = max(1, int((cfg0.get("ukf", {}) or {}).get("every", 1)))

    vec = TorchVecEnv(cfgs, num_envs=batch_size, device=device, episode_seeds=episode_seeds)
    pol = RLPolicyDiff(cfg0, device=device)
    din_def, din_att = pol.verify_ckpt_compat()
    if din_def != obs_expected or din_att != obs_expected:
        raise RuntimeError(
            f"Policy obs dims mismatch in batched eval: defender={din_def}, attacker={din_att}, "
            f"expected={obs_expected} (D={D}, use_fuel={use_fuel})."
        )

    def_hidden = pol.def_net.init_hidden(batch_size=batch_size, device=pol.device) if pol.def_is_student else None
    def_u_prev = (
        torch.zeros((batch_size, pol.act_dim), dtype=torch.float32, device=pol.device)
        if pol.def_is_student else None
    )
    att_hidden = pol.att_net.init_hidden(batch_size=batch_size, device=pol.device) if pol.att_is_student else None
    att_u_prev = (
        torch.zeros((batch_size, pol.act_dim), dtype=torch.float32, device=pol.device)
        if pol.att_is_student else None
    )

    def _state_np() -> np.ndarray:
        return vec.state_t.detach().cpu().numpy()

    def _p3_from_slice(x_slice: np.ndarray) -> tuple[float, float, float]:
        return _p3(np.asarray(x_slice, dtype=np.float32), D)

    def _v3_from_slice(x_slice: np.ndarray) -> tuple[float, float, float]:
        return tuple(map(float, _v3_from_state(np.asarray(x_slice, dtype=np.float32), D)))

    def _current_estimator_snapshot():
        if not use_kf:
            return None
        if vec._batched_ekf_enabled:
            est12 = vec._def_ekf_x_t[:, :3].detach().cpu().numpy()
            est21 = vec._att_ekf_x_t[:, :3].detach().cpu().numpy()
            trp12 = (
                torch.diagonal(vec._def_ekf_P_t[:, :3, :3], dim1=-2, dim2=-1)
                .sum(dim=-1)
                .detach()
                .cpu()
                .numpy()
            )
            trp21 = (
                torch.diagonal(vec._att_ekf_P_t[:, :3, :3], dim1=-2, dim2=-1)
                .sum(dim=-1)
                .detach()
                .cpu()
                .numpy()
            )
            return est12, est21, trp12, trp21

        est12_rows, est21_rows, trp12_rows, trp21_rows = [], [], [], []
        for env in vec.envs:
            if env.ukf is None or env.att_ukf is None:
                return None
            est12_rows.append(np.asarray(env.ukf.x[:3], dtype=float))
            est21_rows.append(np.asarray(env.att_ukf.x[:3], dtype=float))
            trp12_rows.append(float(np.trace(env.ukf.P[:3, :3])))
            trp21_rows.append(float(np.trace(env.att_ukf.P[:3, :3])))
        return (
            np.asarray(est12_rows, dtype=float),
            np.asarray(est21_rows, dtype=float),
            np.asarray(trp12_rows, dtype=float),
            np.asarray(trp21_rows, dtype=float),
        )

    exec_xyz1 = [[] for _ in range(batch_size)]
    exec_xyz2 = [[] for _ in range(batch_size)]
    vel_xyz1 = [[] for _ in range(batch_size)]
    vel_xyz2 = [[] for _ in range(batch_size)]
    u_cmd_norm_1 = [[] for _ in range(batch_size)]
    u_cmd_norm_2 = [[] for _ in range(batch_size)]
    u_real_norm_1 = [[] for _ in range(batch_size)]
    u_real_norm_2 = [[] for _ in range(batch_size)]
    fuel_frac_def = [[] for _ in range(batch_size)] if use_fuel else None
    fuel_frac_att = [[] for _ in range(batch_size)] if use_fuel else None
    thrust_def = [[] for _ in range(batch_size)] if use_fuel else None
    thrust_att = [[] for _ in range(batch_size)] if use_fuel else None
    mdot_def = [[] for _ in range(batch_size)] if use_fuel else None
    mdot_att = [[] for _ in range(batch_size)] if use_fuel else None
    done_info_all: List[Dict[str, Any] | None] = [None for _ in range(batch_size)]

    est12_xyz = [[] for _ in range(batch_size)] if use_kf else None
    est21_xyz = [[] for _ in range(batch_size)] if use_kf else None
    meas12_azel = [[None] for _ in range(batch_size)] if use_kf else None
    meas21_azel = [[None] for _ in range(batch_size)] if use_kf else None
    meas12_innov_sq = [[float("nan")] for _ in range(batch_size)] if use_kf else None
    meas21_innov_sq = [[float("nan")] for _ in range(batch_size)] if use_kf else None
    trP12_pos = [[] for _ in range(batch_size)] if use_kf else None
    trP21_pos = [[] for _ in range(batch_size)] if use_kf else None

    state_np = _state_np()
    for i in range(batch_size):
        x1 = state_np[i, : 2 * D]
        x2 = state_np[i, 2 * D : 4 * D]
        exec_xyz1[i].append(_p3_from_slice(x1))
        exec_xyz2[i].append(_p3_from_slice(x2))
        vel_xyz1[i].append(_v3_from_slice(x1))
        vel_xyz2[i].append(_v3_from_slice(x2))

    if use_fuel:
        fuel_def_np, fuel_att_np = [t.detach().cpu().numpy() for t in vec._fuel_fractions()]
        for i in range(batch_size):
            fuel_frac_def[i].append(float(fuel_def_np[i]))
            fuel_frac_att[i].append(float(fuel_att_np[i]))

    if use_kf:
        snapshot = _current_estimator_snapshot()
        if snapshot is not None:
            est12_np, est21_np, trp12_np, trp21_np = snapshot
            for i in range(batch_size):
                est12_xyz[i].append(np.asarray(est12_np[i], dtype=float).copy())
                est21_xyz[i].append(np.asarray(est21_np[i], dtype=float).copy())
                trP12_pos[i].append(float(trp12_np[i]))
                trP21_pos[i].append(float(trp21_np[i]))

    active = torch.ones((batch_size,), dtype=torch.bool, device=vec.device)

    t_roll0 = time.perf_counter()
    for _k in range(int(steps)):
        if stop_on_done and not bool(torch.any(active).item()):
            break

        sigma_feat = vec.get_student_sigma_features()
        a1, def_hidden, def_u_prev = pol.act_def_batch_torch(
            vec.obs_def,
            deterministic=deterministic,
            sigma_feat=sigma_feat,
            hidden=def_hidden,
            u_prev=def_u_prev,
        )
        a2, att_hidden, att_u_prev = pol.act_att_batch_torch(
            vec.obs_att,
            deterministic=deterministic,
            sigma_feat=sigma_feat,
            hidden=att_hidden,
            u_prev=att_u_prev,
        )

        _, _r1, _r2, done_t, infos = vec.step(
            a1,
            a2,
            reward_mode="both",
            active_mask=active,
            auto_reset=False,
        )

        state_np = _state_np()
        done_np = done_t.detach().cpu().numpy().astype(bool)
        active_np = active.detach().cpu().numpy().astype(bool)
        est_snapshot = _current_estimator_snapshot() if use_kf else None

        for i in range(batch_size):
            if not active_np[i]:
                continue

            info = infos[i]
            x1 = state_np[i, : 2 * D]
            x2 = state_np[i, 2 * D : 4 * D]
            exec_xyz1[i].append(_p3_from_slice(x1))
            exec_xyz2[i].append(_p3_from_slice(x2))
            vel_xyz1[i].append(_v3_from_slice(x1))
            vel_xyz2[i].append(_v3_from_slice(x2))
            u_cmd_norm_1[i].append(float(info["u_cmd_norm_def"]))
            u_cmd_norm_2[i].append(float(info["u_cmd_norm_att"]))
            u_real_norm_1[i].append(float(info["u_real_norm_def"]))
            u_real_norm_2[i].append(float(info["u_real_norm_att"]))

            if use_fuel:
                fuel_frac_def[i].append(float(info["fuel_frac_def"]))
                fuel_frac_att[i].append(float(info["fuel_frac_att"]))
                thrust_def[i].append(float(info["thrust_def"]))
                thrust_att[i].append(float(info["thrust_att"]))
                mdot_def[i].append(float(info["mdot_def"]))
                mdot_att[i].append(float(info["mdot_att"]))

            if use_kf and est_snapshot is not None:
                est12_np, est21_np, _trp12_np, _trp21_np = est_snapshot
                meas_taken = bool(info.get("meas_taken", (int(info["t"]) % meas_every) == 0))
                att_meas_taken = bool(info.get("att_meas_taken", meas_taken))
                est12_xyz[i].append(np.asarray(est12_np[i], dtype=float).copy())
                est21_xyz[i].append(np.asarray(est21_np[i], dtype=float).copy())
                meas12_azel[i].append(True if meas_taken else None)
                meas21_azel[i].append(True if att_meas_taken else None)
                meas12_innov_sq[i].append(float(info["meas_innov_sq"]) if meas_taken else float("nan"))
                meas21_innov_sq[i].append(float(info["att_meas_innov_sq"]) if att_meas_taken else float("nan"))
                trP12_pos[i].append(float(info["ukf_trPpos"]))
                trP21_pos[i].append(float(info["att_ukf_trPpos"]))

            if done_np[i]:
                done_info_all[i] = {
                    "t": int(info["t"]),
                    "oob_def": bool(info["oob_def"]),
                    "oob_att": bool(info["oob_att"]),
                    "hit_target": bool(info["hit_target"]),
                    "collision": bool(info["collision"]),
                }

        if stop_on_done:
            active = active & (~done_t.to(dtype=torch.bool))

    t_fn1 = time.perf_counter()
    timing = {
        "setup": float((t_roll0 - t_fn0) / max(1, batch_size)),
        "simulation": float((t_fn1 - t_roll0) / max(1, batch_size)),
        "total": float((t_fn1 - t_fn0) / max(1, batch_size)),
    }

    outs: List[Dict[str, Any]] = []
    for i in range(batch_size):
        out: Dict[str, Any] = {
            "exec1_xyz": np.asarray(exec_xyz1[i], dtype=float),
            "exec2_xyz": np.asarray(exec_xyz2[i], dtype=float),
            "vel1_xyz": np.asarray(vel_xyz1[i], dtype=float),
            "vel2_xyz": np.asarray(vel_xyz2[i], dtype=float),
            "vel_xyz_all": [
                np.asarray(vel_xyz1[i], dtype=float),
                np.asarray(vel_xyz2[i], dtype=float),
            ],
            "u_cmd_norm_all": [
                np.asarray(u_cmd_norm_1[i], dtype=float),
                np.asarray(u_cmd_norm_2[i], dtype=float),
            ],
            "u_real_norm_all": [
                np.asarray(u_real_norm_1[i], dtype=float),
                np.asarray(u_real_norm_2[i], dtype=float),
            ],
            "done_info": done_info_all[i],
            "rollout_timing_sec": dict(timing),
        }
        if use_fuel:
            out.update(
                {
                    "fuel_frac_all": [
                        np.asarray(fuel_frac_def[i], dtype=float),
                        np.asarray(fuel_frac_att[i], dtype=float),
                    ],
                    "thrust_all": [
                        np.asarray(thrust_def[i], dtype=float),
                        np.asarray(thrust_att[i], dtype=float),
                    ],
                    "mdot_all": [
                        np.asarray(mdot_def[i], dtype=float),
                        np.asarray(mdot_att[i], dtype=float),
                    ],
                }
            )
        if use_kf and est12_xyz is not None:
            out.update(
                {
                    "est12_xyz": np.asarray(est12_xyz[i], dtype=float),
                    "est21_xyz": np.asarray(est21_xyz[i], dtype=float),
                    "meas12_azel": list(meas12_azel[i]),
                    "meas21_azel": list(meas21_azel[i]),
                    "meas12_innov_sq": np.asarray(meas12_innov_sq[i], dtype=float),
                    "meas21_innov_sq": np.asarray(meas21_innov_sq[i], dtype=float),
                    "trP12_pos": np.asarray(trP12_pos[i], dtype=float),
                    "trP21_pos": np.asarray(trP21_pos[i], dtype=float),
                }
            )
        outs.append(out)

    return outs


def _run_rhc_with_rl_and_collect_frames_3d_multi(
    cfg: Dict[str, Any],
    steps: int | None = None,
    turn_len: int | None = None,
):
    t_fn0 = time.perf_counter()

    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx = 2 * D
    T = int(cfg["T"])
    dt = float(cfg["dt"])
    Na = int(cfg.get("num_attackers", 1))

    if Na <= 1:
        raise ValueError("_run_rhc_with_rl_and_collect_frames_3d_multi expects num_attackers > 1.")

    if steps is None:
        steps = int(cfg.get("T_eval", cfg.get("T", cfg.get("steps", 60))))
    if turn_len is None:
        turn_len = 1

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

    dyn_name = str(cfg.get("dynamics", "hcw")).lower()
    use_kf = bool(cfg.get("use_kf", False))
    estimator_kind = _normalize_estimator_kind(cfg)
    if use_kf and not _estimator_supports_dynamics(cfg, dyn_name):
        _warn_once(
            f"[warn] {estimator_kind.upper()} in this runner does not support dynamics='{_canonical_estimator_dyn_name(dyn_name)}'; disabling estimator."
        )
        use_kf = False

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

    ukf_cfg = dict(cfg.get("ukf", {}))
    if not use_kf:
        ukf_cfg = {}
    ukf_list = []
    xA_est_list = [xA.copy() for xA in xA_list]
    est_hist = None
    if use_kf:
        sigma_az = float(ukf_cfg.get("sigma_az", np.deg2rad(0.5)))
        sigma_el = float(ukf_cfg.get("sigma_el", np.deg2rad(0.5)))
        meas_every = max(1, int(ukf_cfg.get("every", 1)))
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

        if ukf_state_dim == 9:
            P0 = np.diag([pos_std0**2] * 3 + [vel_std0**2] * 3 + [accel_std0**2] * 3)
            Q = np.diag([Q_scale] * 6 + [accel_q_scale] * 3)
        else:
            P0 = np.diag([pos_std0**2] * 3 + [vel_std0**2] * 3)
            Q = Q_scale * np.diag([1, 1, 1, 1, 1, 1])
        Rm = np.diag([sigma_az**2, sigma_el**2])
        for j in range(Na):
            xA6 = _x6_from_xD(xA_list[j])
            xA6_mean = xA6.copy()
            xA6_mean[:3] += np.random.normal(0.0, init_mean_pos_std, size=3)
            xA6_mean[3:6] += np.random.normal(0.0, init_mean_vel_std, size=3)
            xA0 = xA6_mean
            ukf_j = KF_CV(
                xA0,
                P0,
                Q,
                Rm,
                dt,
                kind=estimator_kind,
                **_estimator_factory_args(cfg, linearization_group=f"defender_to_attacker_{j}"),
            )
            ukf_list.append(ukf_j)
            xA_est_list[j] = _xD_from_x6(np.r_[ukf_j.x[:3], ukf_j.x[3:6]]).astype(np.float32)

        est_hist = {}
        for j, ukf_j in enumerate(ukf_list):
            key = j + 2
            est_hist[f"est1{key}_xyz"] = [ukf_j.x[:3].copy()]
            est_hist[f"meas1{key}_azel"] = [None]
            est_hist[f"meas1{key}_innov_sq"] = [float("nan")]
            est_hist[f"trP1{key}_pos"] = [float(np.trace(ukf_j.P[:3, :3]))]
    else:
        ukf_action_access = "ground_truth"

    pol = RLPolicy_Multi(cfg, device=cfg.get("device", "cpu"))
    din_def, din_att = pol.verify_ckpt_compat()
    if din_def != obs_expected or din_att != obs_expected:
        raise RuntimeError(
            f"Policy obs dims mismatch: defender={din_def}, attacker={din_att}, "
            f"expected={obs_expected} (D={D}, Na={Na}, use_fuel={use_fuel})."
        )

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
        vel_xyz_all[idx].append(_v3_from_state(xs, D).copy())
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
        obs_xA = xA_est_list if (use_kf and ukf_list) else xA_list
        obs = build_train_obs(xD, obs_xA, m_def, m_att)
        uD_cmd = np.clip(
            np.asarray(pol.act_def_obs(obs, deterministic=deterministic), dtype=np.float32),
            -umax,
            +umax,
        )
        uA_cmd = [
            np.clip(
                np.asarray(pol.act_att_obs(obs, deterministic=deterministic, attacker_idx=j), dtype=np.float32),
                -umax,
                +umax,
            )
            for j in range(Na)
        ]
        uD_cmd = apply_velocity_cbf_filter(xD[:D], xD[D:2 * D], uD_cmd)
        uA_cmd = [
            apply_velocity_cbf_filter(xA_list[j][:D], xA_list[j][D:2 * D], uA_cmd[j])
            for j in range(Na)
        ]

        cmd_actions = [uD_cmd] + uA_cmd
        for idx, u_cmd in enumerate(cmd_actions):
            u3 = _u3_from_action(u_cmd, D)
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
            uA_real = list(uA_cmd)
            fuel_depleted_def = False
            fuel_depleted_att = [False for _ in range(Na)]
            thrust_def = 0.0
            mdot_def = 0.0
            thrust_att = [0.0 for _ in range(Na)]
            mdot_att = [0.0 for _ in range(Na)]

        real_actions = [uD_real] + uA_real
        for idx, u_real in enumerate(real_actions):
            u3 = _u3_from_action(u_real, D)
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

        if use_kf and ukf_list:
            p_obs = xD[:D]
            R_wb = np.eye(3)
            for j in range(Na):
                ukf_j = ukf_list[j]
                key = j + 2
                u_pred, u_cov = _kf_predict_control(
                    ukf_action_access,
                    _u3_from_action(uA_real[j], D),
                    control_meas_std=ukf_control_meas_std,
                    control_limit=umax,
                )
                ukf_j.predict(dt=dt, u=u_pred, u_cov=u_cov)
                innov_sq = float("nan")
                if ((k + 1) % meas_every) == 0:
                    pA_true = xA_list[j][:D]
                    v_b = _body_bearing_from_world(p_obs, R_wb, pA_true)
                    az_true, el_true = _azel_from_body_vec(v_b)
                    z_true = np.array([az_true, el_true], float)
                    z_noise = np.random.multivariate_normal(mean=np.zeros(2), cov=ukf_j.R)
                    z = z_true + z_noise
                    z_hat = ukf_j.h(ukf_j.x.copy(), p_obs=p_obs, R_wb=R_wb)
                    innov = z - z_hat
                    innov[0] = _normalize_angle(innov[0])
                    innov[1] = _normalize_angle(innov[1])
                    innov_sq = float(innov @ innov)
                    ukf_j.update(z, p_obs, R_wb)
                    est_hist[f"meas1{key}_azel"].append((np.asarray(p_obs, float).copy(), z.copy()))
                else:
                    est_hist[f"meas1{key}_azel"].append(None)
                est_hist[f"meas1{key}_innov_sq"].append(innov_sq)
                est_hist[f"trP1{key}_pos"].append(float(np.trace(ukf_j.P[:3, :3])))
                est_hist[f"est1{key}_xyz"].append(ukf_j.x[:3].copy())
                xA_est_list[j] = _xD_from_x6(np.r_[ukf_j.x[:3], ukf_j.x[3:6]]).astype(np.float32)

        if use_fuel:
            fuel_frac_all[0].append(float(np.clip((m_def - mdry_def) / (m0_def - mdry_def + 1e-9), 0.0, 1.0)))
            for j in range(Na):
                fuel_frac_all[1 + j].append(float(np.clip((m_att[j] - mdry_att) / (m0_att - mdry_att + 1e-9), 0.0, 1.0)))

        cur_states = [xD] + xA_list
        for idx, xs in enumerate(cur_states):
            plan_hist_all[idx].append([_p3(xs, D)] * T)
            plan_att_all[idx].append([{"R": I, "phi": 0.0} for _ in range(T)])
            exec_xyz_all[idx].append(_p3(xs, D))
            vel_xyz_all[idx].append(_v3_from_state(xs, D).copy())
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
    if est_hist is not None:
        out.update(est_hist)
    return out


def run_rhc_with_rl_and_collect_frames_3d(
    cfg: Dict[str, Any],
    steps: int | None = None,
    turn_len: int | None = None,
):
    t_fn0 = time.perf_counter()
    # -------------------- basics & dims --------------------
    D = int(cfg.get("D", np.asarray(cfg["x0"]).shape[1] // 2))
    nx = 2 * D
    T = int(cfg["T"])
    dt = float(cfg["dt"])

    num_attackers = int(cfg.get("num_attackers", 1))
    if num_attackers > 1:
        raise NotImplementedError("The multi-attacker rollout path has been removed; use num_attackers=1.")

    dyn_name = str(cfg.get("dynamics", "hcw")).lower()

    # -------------------- sim length --------------------
    if steps is None:
        steps = int(cfg.get("T_eval", cfg.get("T", cfg.get("steps", 60))))
    if turn_len is None:
        turn_len = 1  # kept for API symmetry

    # -------------------- arena & center --------------------
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

    # -------------------- initial states --------------------
    x0 = np.asarray(cfg["x0"], dtype=np.float32)
    if x0.shape[0] < 2:
        raise ValueError("cfg['x0'] must contain at least defender and attacker rows.")

    x1 = np.asarray(x0[0, :nx], dtype=np.float32).copy()
    x2 = np.asarray(x0[1, :nx], dtype=np.float32).copy()

    # -------------------- rollout options --------------------
    deterministic = bool(cfg.get("rl_eval_deterministic", True))
    umax = float(cfg.get("umax", 5e-4))
    debug_actions = bool(cfg.get("debug_actions", False))
    stop_on_done = bool(cfg.get("stop_on_done", True))
    apply_velocity_cbf_filter = _build_velocity_cbf_filter(cfg, D=D, dt=dt, umax=umax)

    # -------------------- fuel setup --------------------
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

    def build_student_sigma_feat():
        if use_kf and (ukf12 is not None):
            P = np.asarray(ukf12.P, dtype=np.float32)
            P_rel = P[: 2 * D, : 2 * D]
            iu = np.triu_indices(2 * D)
            return P_rel[iu].astype(np.float32)
        return None

    # -------------------- policy wrapper --------------------
    pol = RLPolicyDiff(cfg, device=cfg.get("device", "cpu"))
    din_def, din_att = pol.verify_ckpt_compat()

    if din_def != obs_expected or din_att != obs_expected:
        raise RuntimeError(
            f"Policy obs dims mismatch: defender={din_def}, attacker={din_att}, "
            f"expected={obs_expected} (D={D}, use_fuel={use_fuel})."
        )

    # -------------------- tiny helpers for animator compatibility --------------------
    def _p3_row(x_row):
        return (
            (float(x_row[0]), float(x_row[1]), float(x_row[2]))
            if D == 3
            else (float(x_row[0]), float(x_row[1]), 0.0)
        )

    def _p3_vec(x_vec):
        return (
            (float(x_vec[0]), float(x_vec[1]), float(x_vec[2]))
            if D == 3
            else (float(x_vec[0]), float(x_vec[1]), 0.0)
        )

    def _v3_vec(x_vec):
        return tuple(map(float, _v3_from_state(x_vec, D)))

    def _identity_R():
        return np.eye(3, dtype=float)

    def _u3(uD: np.ndarray):
        uD = np.asarray(uD, float).reshape(-1)
        if D == 3:
            return uD[:3]
        return np.array([uD[0], uD[1], 0.0], dtype=float)

    arena_r = float(cfg.get("arena", {}).get("r", 1.0))

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

    # -------------------- optional estimators --------------------
    use_kf = bool(cfg.get("use_kf", False))
    estimator_kind = _normalize_estimator_kind(cfg)
    if use_kf and not _estimator_supports_dynamics(cfg, dyn_name):
        _warn_once(
            f"[warn] {estimator_kind.upper()} in this runner does not support dynamics='{_canonical_estimator_dyn_name(dyn_name)}'; disabling estimator."
        )
        use_kf = False

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
        meas_every = max(1, int(ukf_cfg.get("every", 1)))

        if ukf_state_dim == 9:
            P0 = np.diag([pos_std0**2] * 3 + [vel_std0**2] * 3 + [accel_std0**2] * 3)
            Q = np.diag([Q_scale] * 6 + [accel_q_scale] * 3)
        else:
            P0 = np.diag([pos_std0**2] * 3 + [vel_std0**2] * 3)
            Q = Q_scale * np.diag([1, 1, 1, 1, 1, 1])
        Rm = np.diag([sigma_az**2, sigma_el**2])

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
                **_estimator_factory_args(cfg, linearization_group="observer_1_to_2"),
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
                **_estimator_factory_args(cfg, linearization_group="observer_2_to_1"),
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

    # -------------------- dynamics selection (LTI/LTV/nonlinear) --------------------
    dyn = cfg.get("dyn", {}) if isinstance(cfg.get("dyn", {}), dict) else {}

    dyn_type = dyn.get("type", None)
    Ad = dyn.get("Ad", None)
    Bd = dyn.get("Bd", None)
    Ad_seq = dyn.get("Ad_seq", None)
    Bd_seq = dyn.get("Bd_seq", None)
    chief_cache = dyn.get("chief_cache", None)

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

    # -------------------- termination helper --------------------
    oi = cfg.get("oi", {}) or {}
    oi_radius = float(oi.get("r", 0.0))
    oi_radius_norm = oi_radius / arena_r if arena_r > 0 else 0.0
    # hit_buffer_def = float(cfg.get("hit_buffer_def"))
    # hit_buffer_att = float(cfg.get("hit_buffer_att"))
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

    # -------------------- logs for animator --------------------
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

    # t=0 logs
    exec_xyz1.append(_p3_vec(x1))
    exec_xyz2.append(_p3_vec(x2))
    vel_xyz1.append(_v3_vec(x1))
    vel_xyz2.append(_v3_vec(x2))
    exec_att1.append({"R": _identity_R(), "phi": 0.0})
    exec_att2.append({"R": _identity_R(), "phi": 0.0})
    phi_hist1.append(0.0)
    phi_hist2.append(0.0)
    fov_axis_hist.append(None)
    fov_seen_mask.append(False)

    if use_fuel:
        fuel_frac_def_hist.append(float(np.clip((m_def - mdry_def) / (m0_def - mdry_def + 1e-9), 0.0, 1.0)))
        fuel_frac_att_hist.append(float(np.clip((m_att - mdry_att) / (m0_att - mdry_att + 1e-9), 0.0, 1.0)))

    # ==================== ROLLOUT ====================
    t_roll0 = time.perf_counter()
    for k in range(steps):
        # 1) build observations and query policies ONCE
        x2_for_def_obs = x2_est if (use_kf and ukf12 is not None) else x2
        x1_for_att_obs = x1_est if (use_kf and ukf21 is not None) else x1

        obs_def = build_train_obs(x1, x2_for_def_obs, m_def, m_att)
        obs_att = build_train_obs(x1_for_att_obs, x2, m_def, m_att)
        sigma_feat = build_student_sigma_feat()

        u1_cmd = np.clip(
            np.asarray(
                pol.act_def_obs(obs_def, deterministic=deterministic, sigma_feat=sigma_feat),
                dtype=np.float32,
            ),
            -umax,
            +umax,
        )
        u2_cmd = np.clip(
            np.asarray(
                pol.act_att_obs(obs_att, deterministic=deterministic, sigma_feat=sigma_feat),
                dtype=np.float32,
            ),
            -umax,
            +umax,
        )
        u1_cmd = apply_velocity_cbf_filter(x1[:D], x1[D:2 * D], u1_cmd)
        u2_cmd = apply_velocity_cbf_filter(x2[:D], x2[D:2 * D], u2_cmd)

        # 2) log commanded actions
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

        # 3) propulsion / realized acceleration
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

        # 4) plant step
        x1 = step_plant_single(x1, u1_real, k)
        x2 = step_plant_single(x2, u2_real, k)

        # 5) optional estimator update
        if use_kf and est_hist is not None:
            idx = len(exec_xyz1)
            take = (idx % meas_every) == 0

            def _to3_pos(xD):
                p = np.zeros(3, dtype=float)
                p[:D] = np.asarray(xD[:D], float)
                return p

            if ukf12 is not None:
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

        # 6) fuel traces
        if use_fuel:
            fuel_frac_def_hist.append(
                float(np.clip((m_def - mdry_def) / (m0_def - mdry_def + 1e-9), 0.0, 1.0))
            )
            fuel_frac_att_hist.append(
                float(np.clip((m_att - mdry_att) / (m0_att - mdry_att + 1e-9), 0.0, 1.0))
            )

        # 7) logs for animation
        plan1 = [_p3_row(x1)] * T
        plan2 = [_p3_row(x2)] * T
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

        exec_xyz1.append(_p3_vec(x1))
        exec_xyz2.append(_p3_vec(x2))
        vel_xyz1.append(_v3_vec(x1))
        vel_xyz2.append(_v3_vec(x2))
        fov_axis_hist.append(None)
        fov_seen_mask.append(False)

        # 8) optional termination
        done, done_info = check_done(x1, x2, fuel_depleted_def, fuel_depleted_att)
        if done and stop_on_done:
            break

    t_fn1 = time.perf_counter()
    timing = {
        "setup": float(t_roll0 - t_fn0),
        "simulation": float(t_fn1 - t_roll0),
        "total": float(t_fn1 - t_fn0),
    }

    # -------------------- pack results --------------------
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
