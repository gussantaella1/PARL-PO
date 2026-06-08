"""
core/train_eval.py

Training, evaluation, checkpointing, opponent mixing, and distillation orchestration for staged PPO runs.
"""

import copy
import gc
import os
import time
from typing import Any, Dict

import numpy as np
import torch

from config_rl import build_dyn, config_for_train
from core.buffers import RolloutBuffer
from core.distill import distill_from_teacher
from core.env import Env, SubprocVecEnv, TorchVecEnv, VecEnv, collect_ic_history_from_vecenv
from core.intercept_heuristic import annealed_prior_blend
from core.models import ActorCriticDiff
from core.plotting import make_tb_writer
from core.ppo import PPO
from core.utils import set_seed, squash_action
from core.freeze_utils import freeze_module_, snapshot_state_dict, assert_frozen_unchanged, assert_deterministic_action

# =============================================================
# Training & Evaluation
# =============================================================


def _cpu_state_dict(m: torch.nn.Module) -> dict:
    """Internal helper for cpu state dict."""
    return {k: v.detach().cpu() for k, v in m.state_dict().items()}

def _save_role_checkpoint(ppo, train_role: str, path: str):
    """Save role checkpoint into the current output directory."""
    if train_role == "def":
        net = ppo.def_net
    elif train_role == "att":
        net = ppo.att_net
        if net is None:
            raise RuntimeError("Tried to save attacker checkpoint, but ppo.att_net is None.")
    else:
        raise ValueError(f"Unknown train_role={train_role!r}")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(_cpu_state_dict(net), path)


def _normalize_probs(entries):
    """Normalize probs into the canonical representation used here."""
    probs = np.asarray([float(e.get("prob", 0.0)) for e in entries], dtype=float)
    if np.all(probs <= 0.0):
        probs = np.full((len(entries),), 1.0 / max(1, len(entries)), dtype=float)
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("Opponent mix probabilities must sum to a positive value.")
    return probs / total


def _load_frozen_policy(
    *,
    obs_dim: int,
    act_dim: int,
    cfg: Dict[str, Any],
    device: str,
    ckpt_path: str,
):
    """Load frozen policy from disk, a manifest, or a checkpoint payload."""
    net = ActorCriticDiff(obs_dim, act_dim, cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state)
    freeze_module_(net)
    return net


def _build_opponent_policy_mix(
    *,
    cfg: Dict[str, Any],
    obs_dim: int,
    act_dim: int,
    device: str,
    train_role: str,
):
    """Build opponent policy mix for the current workflow."""
    mix = cfg.get("opp_mix", {}) or {}
    policy_entries = list(mix.get("policies", []) or [])
    if not policy_entries:
        return None

    opp_role = "def" if train_role == "att" else "att"
    if opp_role == "att" and str(cfg.get("attacker_mode", "rule")).lower() != "rl":
        raise ValueError("Checkpoint-based opp_mix for defender training requires attacker_mode='rl'.")

    entries = []
    for item in policy_entries:
        name = str(item.get("name", f"{opp_role}_policy_{len(entries)}"))
        path = item.get("path")
        if not path:
            raise ValueError(f"opp_mix policy entry {name!r} is missing 'path'.")

        net = _load_frozen_policy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            cfg=cfg,
            device=device,
            ckpt_path=path,
        )
        entries.append({
            "name": name,
            "net": net,
            "prob": float(item.get("prob", 0.0)),
            "path": path,
            "action_scale": float(item.get("action_scale", 1.0)),
            "noise_std": float(item.get("noise_std", 0.0)),
            "idle_prob": float(item.get("idle_prob", 0.0)),
        })

    probs = _normalize_probs(entries)
    resample = str(mix.get("resample", "episode")).lower()
    if resample not in ("episode", "update", "never"):
        raise ValueError(f"Unsupported opp_mix resample={resample!r}; expected 'episode', 'update', or 'never'.")

    return {
        "opp_role": opp_role,
        "entries": entries,
        "probs": probs,
        "probs_t": torch.as_tensor(probs, dtype=torch.float32, device=device),
        "resample": resample,
        "act_dim": act_dim,
    }


def _sample_opponent_indices(mix_state, size: int):
    """Sample opponent indices for training, evaluation, or rollout initialization."""
    return torch.multinomial(mix_state["probs_t"], num_samples=size, replacement=True).to(dtype=torch.int64)


@torch.no_grad()
def _act_from_opponent_policy_mix(
    obs_batch: torch.Tensor,
    mix_state,
    active_indices,
    act_scale: float,
):
    """Internal helper for act from opponent policy mix."""
    opp_role = mix_state["opp_role"]
    act_dim = int(mix_state["act_dim"])
    out = torch.zeros((obs_batch.shape[0], act_dim), dtype=obs_batch.dtype, device=obs_batch.device)
    active_indices_t = torch.as_tensor(active_indices, dtype=torch.int64, device=obs_batch.device).reshape(-1)

    unique_ids = torch.unique(active_indices_t).detach().cpu().tolist()
    for idx in unique_ids:
        mask = active_indices_t == int(idx)
        entry = mix_state["entries"][int(idx)]
        net = entry["net"]
        dist = net.dist(obs_batch[mask], who=opp_role)
        a = squash_action(dist.mean, act_scale)

        action_scale = float(entry.get("action_scale", 1.0))
        noise_std = float(entry.get("noise_std", 0.0))
        idle_prob = float(entry.get("idle_prob", 0.0))

        if action_scale != 1.0:
            a = action_scale * a

        if noise_std > 0.0:
            a = a + noise_std * torch.randn_like(a)

        if idle_prob > 0.0:
            idle_mask = (torch.rand((a.shape[0], 1), device=a.device) < idle_prob)
            a = torch.where(idle_mask, torch.zeros_like(a), a)

        a = torch.clamp(a, -act_scale, act_scale)
        out[mask] = a
    return out


def _role_obs_batch(role: str, obs_def: torch.Tensor, obs_att: torch.Tensor) -> torch.Tensor:
    """Internal helper for role obs batch."""
    return obs_def if role == "def" else obs_att


def _deep_merge_dict(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Internal helper for deep merge dict."""
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _make_vec_env(cfg: Dict[str, Any], num_envs: int, device: str):
    """Internal helper for make vec env."""
    backend = str(cfg.get("vec_backend", "sync")).lower()
    if backend == "sync":
        return VecEnv(lambda: Env(cfg), num_envs)
    if backend == "subproc":
        return SubprocVecEnv(
            cfg,
            num_envs,
            num_workers=cfg.get("vec_workers"),
            start_method=cfg.get("mp_start_method"),
        )
    if backend == "torch":
        return TorchVecEnv(cfg, num_envs, device=device)
    raise ValueError(f"Unknown vec_backend={backend!r}. Expected 'sync', 'subproc', or 'torch'.")


def _set_vec_attr(vec, name: str, value: Any):
    """Internal helper for set vec attr."""
    if hasattr(vec, "set_attr"):
        vec.set_attr(name, value)
        return
    for env in vec.envs:
        setattr(env, name, value)


def _tb_run_name(cfg: Dict[str, Any], phase_name: str, suffix: str) -> str:
    """Internal helper for tb run name."""
    prefix = cfg.get("tb_run_prefix") or cfg.get("tb_run_name")
    if not prefix:
        checkpoint_dir = cfg.get("checkpoint_dir") or "training_run"
        prefix = os.path.basename(os.path.abspath(checkpoint_dir))
    return f"{prefix}_{phase_name}_{suffix}"

def train(cfg: Dict[str, Any]):
    """Run the configured staged PPO training workflow."""
    set_seed(cfg["seed"])
    if bool(cfg.get("use_kf", False)) and str(cfg.get("estimator_kind", "ukf")).lower() == "ekf":
        ekf_mode = str(cfg.get("ukf", {}).get("ekf_jacobian_mode", "exact")).strip().lower().replace("-", "_")
        if ekf_mode != "exact":
            from ekf_estimator import AgentEKF

            AgentEKF.clear_global_linearization_cache()

    device = cfg["device"]

    writer = None
    tb_logdir = None
    global_env_step = 0

    if cfg.get("use_tensorboard", False):
        writer, tb_logdir = make_tb_writer(cfg)

        # Optional: show model graphs once (often noisy / can break with custom ops)
        # writer.add_text("notes", "PPO Diffgame run", 0)

    train_role = cfg.get("train_role", "def")  # <-- NEW

    if train_role == "def":
        reward_mode = "def"
    elif train_role == "att":
        reward_mode = "att"

    # -------------------------
    # Checkpoint saving config
    # -------------------------
    save_best_ckpt = bool(cfg.get("save_best_ckpt", True))
    save_last_ckpt = bool(cfg.get("save_last_ckpt", True))
    checkpoint_dir = cfg.get("checkpoint_dir", None)
    checkpoint_prefix = cfg.get("checkpoint_prefix", f"{train_role}_teacher")

    tracked_metric_name = "R_def_mean" if train_role == "def" else "R_att_mean"
    best_metric = -float("inf")
    best_update = None
    best_ckpt_path = None
    last_ckpt_path = None

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    num_envs = int(cfg.get("num_envs"))
    steps_per_env = int(cfg.get("steps_per_env"))
    total_updates = int(cfg.get("total_updates"))
    log_every = int(cfg.get("log_every"))

    vec = _make_vec_env(cfg, num_envs, device)
    vec_is_torch = bool(getattr(vec, "torch_backend", False))
    obs_dim = vec.obs_def.shape[1]
    act_dim = int(cfg["D"])

    print(
        "[train] rollout backend="
        f"{str(cfg.get('vec_backend', 'sync')).lower()} "
        f"(num_envs={num_envs}, vec_workers={cfg.get('vec_workers', 'n/a')})"
    )

    ppo = PPO(obs_dim, act_dim, cfg, device=device)

    # Optional: initialize attacker from a BC checkpoint (before PPO training)
    att_init = cfg.get("att_init_path", None)
    if att_init is not None:
        if ppo.att_net is None:
            raise RuntimeError("att_init_path provided but attacker_mode != 'rl'")
        state = torch.load(att_init, map_location=device)
        ppo.att_net.load_state_dict(state)
        print(f"[train] Loaded attacker init from: {att_init}")


    # Optional: load fixed defender
    def_ckpt = cfg.get("def_ckpt_path", None)
    if def_ckpt is not None:
        state = torch.load(def_ckpt, map_location=device)
        ppo.def_net.load_state_dict(state)
        # If defender should be frozen:
        if cfg.get("freeze_defender", False):
            for p in ppo.def_net.parameters():
                p.requires_grad_(False)

    # Optional: load fixed attacker
    att_ckpt = cfg.get("att_ckpt_path", None)
    if att_ckpt is not None and ppo.att_net is not None:
        state = torch.load(att_ckpt, map_location=device)
        ppo.att_net.load_state_dict(state)
        if cfg.get("freeze_attacker", False):
            for p in ppo.att_net.parameters():
                p.requires_grad_(False)

    # Freeze whichever role we do NOT train in this phase
    if cfg.get("freeze_defender", False):
        freeze_module_(ppo.def_net)

    if cfg.get("freeze_attacker", False) and (ppo.att_net is not None):
        freeze_module_(ppo.att_net)

    # =========================================================
    # NEW: Freeze verification snapshots (opponents must not move)
    # =========================================================
    verify_freeze = bool(cfg.get("verify_freeze"))
    freeze_tol = float(cfg.get("freeze_tol"))  # set 0.0 for exact; or 1e-12 if you’re paranoid

    snap_def = None
    snap_att = None

    if verify_freeze and cfg.get("freeze_defender"):
        # defender is frozen opponent in attacker-training
        snap_def = snapshot_state_dict(ppo.def_net)

    if verify_freeze and cfg.get("freeze_attacker") and (ppo.att_net is not None):
        # attacker is frozen opponent in defender-training
        snap_att = snapshot_state_dict(ppo.att_net)

    # NEW: LR schedule config
    lr_schedule = cfg.get("lr_schedule", "none")
    lr_final_factor = float(cfg.get("lr_final_factor", 0.1))

    # ---- NEW: metrics container ----
    metrics = {
        "update": [],
        "R_def_mean": [],
        "R_att_mean": [],
        "muD_abs_mean": [],
        "stdD_mean": [],
        "arena_radius_mean": [],
        "d1_mean": [],          # defender true distance
        "d2_mean": [],          # attacker belief distance (what obs sees)
        "d2_true_mean": [],     # attacker true distance
        "p2_est_err_mean": [],  # defender-side UKF RMS position error
        "p1_est_err_mean": [],  # attacker-side UKF RMS position error
        "meas_innov_mean": [],  # optional
        "ukf_trPpos_mean": [],  # optional
        "lr_pi": [],
        "lr_vf": [],
        "prior_blend_def": [],
        "rollout_time_s": [],
        "optim_time_s": [],
        "update_time_s": [],
        "env_steps_per_sec": [],
    }

    if cfg.get("fuel", {}).get("enable", False):
        metrics["fuel_used_def_mean"] = []
        metrics["fuel_used_att_mean"] = []
        metrics["fuel_frac_def_mean"] = []
        metrics["fuel_frac_att_mean"] = []

        # optional
        metrics["thrust_def_mean"] = []
        metrics["thrust_att_mean"] = []
        metrics["mdot_def_mean"] = []
        metrics["mdot_att_mean"] = []

    opp_policy_mix = _build_opponent_policy_mix(
        cfg=cfg,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        train_role=train_role,
    )
    mix_name_to_metric = None
    mix_active_idx = None
    if opp_policy_mix is not None:
        mix_name_to_metric = {
            entry["name"]: f"opp_mix__{entry['name']}"
            for entry in opp_policy_mix["entries"]
        }
        for metric_name in mix_name_to_metric.values():
            metrics[metric_name] = []
        mix_active_idx = _sample_opponent_indices(opp_policy_mix, num_envs)



    # Optional anneal of defender center tether
    def_center_base = cfg.get("def_center_coef", 0.0)
    min_anneal = float(cfg.get("def_center_min_anneal", 0.5))
    radius_knob = cfg.get("arena_radius_knob", {}) or {}
    use_radius_knob = bool(radius_knob.get("enabled", False))
    prior_type = str(cfg.get("prior_type", "ls"))
    intercept_train_only = cfg.get("intercept_prior_train_only", {}) or {}
    use_intercept_train_only = (
        train_role == "def"
        and prior_type == "intercept"
        and bool(intercept_train_only.get("enabled", False))
    )
    intercept_blend_start = float(intercept_train_only.get("start_blend", 0.0))
    intercept_blend_end = float(intercept_train_only.get("end_blend", 0.0))
    intercept_anneal_fraction = float(intercept_train_only.get("anneal_fraction", 1.0))
    if bool(intercept_train_only.get("enabled", False)) and prior_type != "intercept":
        print("[intercept_prior_train_only] ignored because prior_type!='intercept'")

    for upd in range(1, total_updates + 1):
        if vec_is_torch:
            term_counts = {
                "oob_def": torch.zeros((), dtype=torch.float64, device=device),
                "oob_att": torch.zeros((), dtype=torch.float64, device=device),
                "hit_target": torch.zeros((), dtype=torch.float64, device=device),
                "collision": torch.zeros((), dtype=torch.float64, device=device),
            }
        else:
            term_counts = {"oob_def":0, "oob_att":0, "hit_target":0, "collision":0}
        mix_counts_update = None
        if opp_policy_mix is not None:
            if vec_is_torch:
                mix_counts_update = torch.zeros(
                    (len(opp_policy_mix["entries"]),),
                    dtype=torch.int64,
                    device=device,
                )
            else:
                mix_counts_update = np.zeros((len(opp_policy_mix["entries"]),), dtype=np.int64)

        if (opp_policy_mix is not None) and (opp_policy_mix["resample"] == "update"):
            mix_active_idx = _sample_opponent_indices(opp_policy_mix, num_envs)


        # ---------- optional LR decay (linear) ----------
        if lr_schedule == "linear":
            # frac_lr goes 0 → 1 over training
            frac_lr = (upd - 1) / max(1, total_updates - 1)
            scale = 1.0 - frac_lr * (1.0 - lr_final_factor)

            # defender
            for g, base in zip(ppo.def_opt.param_groups, ppo.def_base_lrs):
                g["lr"] = base * scale

            # attacker RL (if you ever turn it on)
            if ppo.att_opt is not None and getattr(ppo, "att_base_lrs", None) is not None:
                for g, base in zip(ppo.att_opt.param_groups, ppo.att_base_lrs):
                    g["lr"] = base * scale
        # ------------------------------------------------

        if use_radius_knob:
            radius_progress = (upd - 1) / max(1, total_updates - 1)
            _set_vec_attr(vec, "curriculum_progress", radius_progress)

        if use_intercept_train_only:
            blend_progress = (upd - 1) / max(1, total_updates - 1)
            current_prior_blend_def = annealed_prior_blend(
                blend_progress,
                start_blend=intercept_blend_start,
                end_blend=intercept_blend_end,
                anneal_fraction=intercept_anneal_fraction,
            )
            ppo.def_net.set_prior_blend("def", current_prior_blend_def)
        else:
            current_prior_blend_def = ppo.def_net.get_prior_blend("def")

        # Linear anneal multiplier from 1.0 → min_anneal for k_cent
        center_frac = upd / max(1, total_updates)
        k_cent_mul = 1.0 - (1.0 - min_anneal) * center_frac
        _set_vec_attr(vec, "k_cent", def_center_base * k_cent_mul)

        # Buffers
        bufD = RolloutBuffer(obs_dim, act_dim, num_envs, steps_per_env, device)
        rule_att = (cfg.get("attacker_mode", "rule") == "rule")
        if (not rule_att) and (train_role in ("att", "both")):
            bufA = RolloutBuffer(obs_dim, act_dim, num_envs, steps_per_env, device)
        else:
            bufA = None


        if vec_is_torch:
            o_def = vec.obs_def.to(dtype=torch.float32)
            o_att = vec.obs_att.to(dtype=torch.float32)
            ep_ret_def = torch.zeros(num_envs, dtype=torch.float64, device=device)
            ep_ret_att = torch.zeros(num_envs, dtype=torch.float64, device=device)
            d1_true_sq_m_acc = torch.zeros((), dtype=torch.float64, device=device)
            d2_true_sq_m_acc = torch.zeros((), dtype=torch.float64, device=device)
            d2_belief_sq_m_acc = torch.zeros((), dtype=torch.float64, device=device)
            arena_radius_acc = torch.zeros((), dtype=torch.float64, device=device)
            p2_est_err_sq_m_acc = torch.zeros((), dtype=torch.float64, device=device)
            p1_est_err_sq_m_acc = torch.zeros((), dtype=torch.float64, device=device)
            meas_innov_acc = torch.zeros((), dtype=torch.float64, device=device)
            trP_acc = torch.zeros((), dtype=torch.float64, device=device)
        else:
            o_def = torch.as_tensor(vec.obs_def, dtype=torch.float32, device=device)
            o_att = torch.as_tensor(vec.obs_att, dtype=torch.float32, device=device)
            ep_ret_def = np.zeros(num_envs, dtype=np.float64)
            ep_ret_att = np.zeros(num_envs, dtype=np.float64)
            d1_true_sq_m_acc = 0.0
            d2_true_sq_m_acc = 0.0
            d2_belief_sq_m_acc = 0.0
            arena_radius_acc = 0.0
            p2_est_err_sq_m_acc = 0.0
            p1_est_err_sq_m_acc = 0.0
            meas_innov_acc = 0.0
            trP_acc = 0.0
        info_count = 0

        if cfg.get("fuel", {}).get("enable", False):
            if vec_is_torch:
                fuel_used_def_acc = torch.zeros((), dtype=torch.float64, device=device)
                fuel_used_att_acc = torch.zeros((), dtype=torch.float64, device=device)
                fuel_frac_def_acc = torch.zeros((), dtype=torch.float64, device=device)
                fuel_frac_att_acc = torch.zeros((), dtype=torch.float64, device=device)
                thrust_def_acc = torch.zeros((), dtype=torch.float64, device=device)
                thrust_att_acc = torch.zeros((), dtype=torch.float64, device=device)
                mdot_def_acc = torch.zeros((), dtype=torch.float64, device=device)
                mdot_att_acc = torch.zeros((), dtype=torch.float64, device=device)
            else:
                fuel_used_def_acc = 0.0
                fuel_used_att_acc = 0.0
                fuel_frac_def_acc = 0.0
                fuel_frac_att_acc = 0.0
                thrust_def_acc = 0.0
                thrust_att_acc = 0.0
                mdot_def_acc = 0.0
                mdot_att_acc = 0.0
            fuel_info_count = 0

        t_rollout0 = time.perf_counter()
        for _ in range(steps_per_env):
            with torch.no_grad():
                if train_role == "def":
                    a1, lp1, v1 = ppo.act(o_def, who="def", deterministic=False)
                    if opp_policy_mix is not None:
                        a2 = _act_from_opponent_policy_mix(
                            _role_obs_batch(opp_policy_mix["opp_role"], o_def, o_att),
                            opp_policy_mix,
                            mix_active_idx,
                            ppo.act_scale,
                        )
                        lp2 = torch.zeros(num_envs, dtype=o_att.dtype, device=device)
                        v2 = torch.zeros(num_envs, dtype=o_att.dtype, device=device)
                    else:
                        a2, lp2, v2 = ppo.act(o_att, who="att", deterministic=True)
                elif train_role == "att":
                    if opp_policy_mix is not None:
                        a1 = _act_from_opponent_policy_mix(
                            _role_obs_batch(opp_policy_mix["opp_role"], o_def, o_att),
                            opp_policy_mix,
                            mix_active_idx,
                            ppo.act_scale,
                        )
                        lp1 = torch.zeros(num_envs, dtype=o_def.dtype, device=device)
                        v1 = torch.zeros(num_envs, dtype=o_def.dtype, device=device)
                    else:
                        a1, lp1, v1 = ppo.act(o_def, who="def", deterministic=True)
                    a2, lp2, v2 = ppo.act(o_att, who="att", deterministic=False)
                else:
                    raise ValueError(f"Unknown train_role={train_role!r}")

            if vec_is_torch:
                _, r1_step, r2_step, d_step, infos = vec.step(
                    a1,
                    a2,
                    reward_mode=reward_mode,
                    emit_infos=False,
                )
                o2_def = vec.obs_def.to(dtype=torch.float32)
                o2_att = vec.obs_att.to(dtype=torch.float32)
                r1 = r1_step.to(dtype=torch.float32)
                r2 = r2_step.to(dtype=torch.float32)
                d = d_step.to(dtype=torch.float32)
            else:
                a1_np = a1.cpu().numpy()
                a2_np = a2.cpu().numpy()
                # normalize shapes to what Env expects:
                # - if attacker RL: (B, D) ok
                # - if rule attacker: (B, Na, D) ok
                # - if rule attacker but produced (B, D): expand to (B,1,D)
                if a2_np.ndim == 2 and cfg.get("attacker_mode", "rule") == "rule" and cfg.get("num_attackers", 1) == 1:
                    a2_np = a2_np[:, None, :]

                _, r1_np, r2_np, d_np, infos = vec.step(
                    a1_np,
                    a2_np,
                    reward_mode=reward_mode,
                )

                o2_def = torch.as_tensor(vec.obs_def, dtype=torch.float32, device=device)
                o2_att = torch.as_tensor(vec.obs_att, dtype=torch.float32, device=device)
                r1 = torch.as_tensor(r1_np, dtype=torch.float32, device=device)
                r2 = torch.as_tensor(r2_np, dtype=torch.float32, device=device)
                d = torch.as_tensor(d_np, dtype=torch.float32, device=device)

            bufD.add(o_def.detach(), a1.detach(), lp1.detach(), v1.detach(), r1, d)
            if bufA is not None:
                bufA.add(o_att.detach(), a2.detach(), lp2.detach(), v2.detach(), r2, d)


            if train_role == "def":
                if vec_is_torch:
                    ep_ret_def += r1.to(dtype=ep_ret_def.dtype)
                else:
                    ep_ret_def += r1_np

            if train_role == "att":
                if vec_is_torch:
                    ep_ret_att += r2.to(dtype=ep_ret_att.dtype)
                else:
                    ep_ret_att += r2_np
            
            o_def = o2_def
            o_att = o2_att

            if opp_policy_mix is not None:
                if vec_is_torch:
                    mix_counts_update += torch.bincount(
                        mix_active_idx,
                        minlength=len(opp_policy_mix["entries"]),
                    )
                    if opp_policy_mix["resample"] == "episode":
                        done_idx = torch.nonzero(d.to(dtype=torch.bool), as_tuple=False).flatten()
                        if done_idx.numel() > 0:
                            mix_active_idx[done_idx] = _sample_opponent_indices(
                                opp_policy_mix,
                                int(done_idx.numel()),
                            )
                else:
                    mix_counts_update += np.bincount(
                        mix_active_idx.detach().cpu().numpy(),
                        minlength=len(opp_policy_mix["entries"]),
                    )
                    if opp_policy_mix["resample"] == "episode":
                        done_mask = d_np.astype(bool)
                        if np.any(done_mask):
                            done_idx = torch.as_tensor(np.flatnonzero(done_mask), dtype=torch.int64)
                            mix_active_idx[done_idx] = _sample_opponent_indices(
                                opp_policy_mix,
                                int(done_mask.sum()),
                            )

            if vec_is_torch:
                step_stats = getattr(vec, "last_step_stats", None)
                if step_stats is None:
                    raise RuntimeError("TorchVecEnv fast path did not populate last_step_stats.")
                info_count += int(step_stats["info_count"])
                arena_radius_acc += step_stats["arena_radius_sum"]
                d1_true_sq_m_acc += step_stats["d1_true_sq_m_sum"]
                d2_true_sq_m_acc += step_stats["d2_true_sq_m_sum"]
                d2_belief_sq_m_acc += step_stats["d2_belief_sq_m_sum"]
                p2_est_err_sq_m_acc += step_stats["p2_est_err_sq_m_sum"]
                p1_est_err_sq_m_acc += step_stats["p1_est_err_sq_m_sum"]
                meas_innov_acc += step_stats["meas_innov_sum"]
                trP_acc += step_stats["ukf_trPpos_sum"]
                term_counts["oob_def"] += step_stats["oob_def_sum"]
                term_counts["oob_att"] += step_stats["oob_att_sum"]
                term_counts["hit_target"] += step_stats["hit_target_sum"]
                term_counts["collision"] += step_stats["collision_sum"]

                if cfg.get("fuel", {}).get("enable", False):
                    fuel_used_def_acc += step_stats["fuel_used_def_sum"]
                    fuel_used_att_acc += step_stats["fuel_used_att_sum"]
                    fuel_frac_def_acc += step_stats["fuel_frac_def_sum"]
                    fuel_frac_att_acc += step_stats["fuel_frac_att_sum"]
                    thrust_def_acc += step_stats["thrust_def_sum"]
                    thrust_att_acc += step_stats["thrust_att_sum"]
                    mdot_def_acc += step_stats["mdot_def_sum"]
                    mdot_att_acc += step_stats["mdot_att_sum"]
                    fuel_info_count += int(step_stats["info_count"])
            else:
                # ---- accumulate truth / belief metrics from Env.info ----
                for inf in infos:
                    # count every env-step
                    info_count += 1

                    # always accumulate if present
                    radius_m = float(inf.get("arena_radius_m", cfg["arena"]["r"]))
                    radius_sq_m = radius_m * radius_m
                    arena_radius_acc += radius_m

                    if "d1_true_norm" in inf:
                        d1_true_sq_m_acc += float(inf["d1_true_norm"]) * radius_sq_m

                    if "d2_true_norm" in inf:
                        d2_true_sq_m_acc += float(inf["d2_true_norm"]) * radius_sq_m

                    # belief distance: fall back safely
                    if "d2_belief_norm" in inf:
                        d2_belief_sq_m_acc += float(inf["d2_belief_norm"]) * radius_sq_m
                    elif "d2_true_norm" in inf:
                        d2_belief_sq_m_acc += float(inf["d2_true_norm"]) * radius_sq_m

                    if "p2_est_err_norm" in inf:
                        p2_est_err_sq_m_acc += float(inf["p2_est_err_norm"]) * radius_sq_m
                    if "p1_est_err_norm" in inf:
                        p1_est_err_sq_m_acc += float(inf["p1_est_err_norm"]) * radius_sq_m

                    if "meas_innov_sq" in inf:
                        meas_innov_acc += inf["meas_innov_sq"]

                    if "ukf_trPpos" in inf:
                        trP_acc += inf["ukf_trPpos"]

                    if inf.get("oob_def", False): term_counts["oob_def"] += 1
                    if inf.get("oob_att", False): term_counts["oob_att"] += 1
                    if inf.get("hit_target", False): term_counts["hit_target"] += 1
                    if inf.get("collision", False): term_counts["collision"] += 1

                    if cfg.get("fuel", {}).get("enable", False):
                        if "fuel_used_def" in inf:
                            fuel_used_def_acc += inf["fuel_used_def"]
                            fuel_used_att_acc += inf["fuel_used_att"]
                            fuel_frac_def_acc += inf["fuel_frac_def"]
                            fuel_frac_att_acc += inf["fuel_frac_att"]

                            thrust_def_acc += inf.get("thrust_def", 0.0)
                            thrust_att_acc += inf.get("thrust_att", 0.0)
                            mdot_def_acc += inf.get("mdot_def", 0.0)
                            mdot_att_acc += inf.get("mdot_att", 0.0)

                            fuel_info_count += 1


            global_env_step += num_envs

        rollout_time_s = float(time.perf_counter() - t_rollout0)

        with torch.no_grad():
            next_v_def = ppo.def_net.value(o_def)
        bufD.finalize(next_v_def)

        if bufA is not None:
            with torch.no_grad():
                next_v_att = ppo.att_net.value(o_att)
            bufA.finalize(next_v_att)

        # ---- choose what to update ----
        t_optim0 = time.perf_counter()
        if train_role == "def":
            ppo.update_defender_only(bufD)

        elif train_role == "att":
            if bufA is None:
                raise RuntimeError("train_role='att' requires attacker_mode='rl'")
            ppo.update_attacker_only(bufA)

        else:
            raise ValueError(f"Unknown train_role={train_role!r}")
        optim_time_s = float(time.perf_counter() - t_optim0)
        update_time_s = rollout_time_s + optim_time_s
        env_steps_per_sec = float(num_envs * steps_per_env) / max(update_time_s, 1e-9)
        
        #explicit cleanup (place it right here)
        # del bufD
        # try:
        #     del bufA
        # except NameError:
        #     pass

        # gc.collect()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
                
        # =========================================================
        # NEW: Verify frozen opponent did not change
        # =========================================================
        if verify_freeze:
            # If training attacker, defender should be frozen (Phase 1)
            if (train_role == "att") and (snap_def is not None):
                assert_frozen_unchanged(snap_def, ppo.def_net, name="frozen_defender", tol=freeze_tol)

            # If training defender, attacker should be frozen (Phase 2)
            if (train_role == "def") and (snap_att is not None):
                assert_frozen_unchanged(snap_att, ppo.att_net, name="frozen_attacker", tol=freeze_tol)



        if upd % log_every == 0:
            if vec_is_torch:
                R_def_mean = float(ep_ret_def.mean().item())
                R_att_mean = float(ep_ret_att.mean().item())
                arena_radius_acc_value = float(arena_radius_acc.item())
                d1_true_sq_m_acc_value = float(d1_true_sq_m_acc.item())
                d2_true_sq_m_acc_value = float(d2_true_sq_m_acc.item())
                d2_belief_sq_m_acc_value = float(d2_belief_sq_m_acc.item())
                p2_est_err_sq_m_acc_value = float(p2_est_err_sq_m_acc.item())
                p1_est_err_sq_m_acc_value = float(p1_est_err_sq_m_acc.item())
                meas_innov_acc_value = float(meas_innov_acc.item())
                trP_acc_value = float(trP_acc.item())
                term_counts_log = {k: float(v.item()) for k, v in term_counts.items()}
            else:
                R_def_mean = ep_ret_def.mean()
                R_att_mean = ep_ret_att.mean()
                arena_radius_acc_value = arena_radius_acc
                d1_true_sq_m_acc_value = d1_true_sq_m_acc
                d2_true_sq_m_acc_value = d2_true_sq_m_acc
                d2_belief_sq_m_acc_value = d2_belief_sq_m_acc
                p2_est_err_sq_m_acc_value = p2_est_err_sq_m_acc
                p1_est_err_sq_m_acc_value = p1_est_err_sq_m_acc
                meas_innov_acc_value = meas_innov_acc
                trP_acc_value = trP_acc
                term_counts_log = term_counts

            tracked_metric_value = R_def_mean if train_role == "def" else R_att_mean

            if save_best_ckpt and checkpoint_dir is not None:
                if tracked_metric_value > best_metric:
                    best_metric = float(tracked_metric_value)
                    best_update = int(upd)
                    best_ckpt_path = os.path.join(
                        checkpoint_dir,
                        f"{checkpoint_prefix}__best.pt"
                    )
                    _save_role_checkpoint(ppo, train_role, best_ckpt_path)
                    print(
                        f"[checkpoint] new best {tracked_metric_name}={best_metric:+.3f} "
                        f"at update {best_update} -> {best_ckpt_path}"
                    )

            # means over all steps collected this update
            if info_count > 0:
                arena_radius_mean = arena_radius_acc_value / info_count

                d1_true_mean = np.sqrt(d1_true_sq_m_acc_value / info_count)
                d2_true_mean = np.sqrt(d2_true_sq_m_acc_value / info_count)
                d2_belief_mean = np.sqrt(d2_belief_sq_m_acc_value / info_count)
                p2_est_err_mean = np.sqrt(p2_est_err_sq_m_acc_value / info_count)
                p1_est_err_mean = np.sqrt(p1_est_err_sq_m_acc_value / info_count)
                meas_innov_mean = meas_innov_acc_value / info_count
                trP_mean = trP_acc_value / info_count
            else:
                arena_radius_mean = 0.0
                d1_true_mean = d2_true_mean = d2_belief_mean = 0.0
                p2_est_err_mean = p1_est_err_mean = 0.0
                meas_innov_mean = trP_mean = 0.0



            with torch.no_grad():
                flat_obs = bufD.obs.reshape(-1, obs_dim)
                distD = ppo.def_net.dist(flat_obs, who="def")
                muD = distD.mean.abs().mean().item()
                stdD = distD.stddev.mean().item()

                if verify_freeze:
                    # Check determinism of the opponent (not the learner)
                    if train_role == "att":
                        # opponent is defender
                        assert_deterministic_action(ppo, flat_obs[:256], who="def", tol=0.0)
                    if train_role == "def" and (ppo.att_net is not None):
                        # opponent is attacker (RL opponent case)
                        assert_deterministic_action(ppo, flat_obs[:256], who="att", tol=0.0)


                # obs = [p1c, p2c, rel, v1, v2]
                Dcfg = cfg["D"]
                p1c = flat_obs[:, :Dcfg]
                p2c = flat_obs[:, Dcfg:2*Dcfg]

                # -------------------------------
                # #5: opponent policy std logging
                # -------------------------------
                dist_opp = (
                    ppo.def_net.dist(flat_obs, who="def")
                    if train_role == "att"  # training attacker => opponent is defender
                    else ppo.att_net.dist(flat_obs, who="att")
                    if (train_role == "def" and ppo.att_net is not None)  # training defender => opponent is attacker RL
                    else None
                )
                if dist_opp is not None:
                    print("opp std mean:", dist_opp.stddev.mean().item())

                # ... your obs-derived d1/d2 means ...
                Dcfg = cfg["D"]
                p1c = flat_obs[:, :Dcfg]
                p2c = flat_obs[:, Dcfg:2*Dcfg]
                d1_obs_mean = p1c.pow(2).sum(-1).mean().sqrt().item()
                d2_obs_mean = p2c.pow(2).sum(-1).mean().sqrt().item()


            # grab learning rates (assuming two param groups: policy+logstd and value)
            active_opt = ppo.def_opt if train_role == "def" else ppo.att_opt
            lr_pi = active_opt.param_groups[0]["lr"]
            lr_vf = active_opt.param_groups[-1]["lr"]

            if cfg.get("fuel", {}).get("enable", False):
                if fuel_info_count > 0:
                    if vec_is_torch:
                        fuel_used_def_mean = float(fuel_used_def_acc.item()) / fuel_info_count
                        fuel_used_att_mean = float(fuel_used_att_acc.item()) / fuel_info_count
                        fuel_frac_def_mean = float(fuel_frac_def_acc.item()) / fuel_info_count
                        fuel_frac_att_mean = float(fuel_frac_att_acc.item()) / fuel_info_count
                        thrust_def_mean = float(thrust_def_acc.item()) / fuel_info_count
                        thrust_att_mean = float(thrust_att_acc.item()) / fuel_info_count
                        mdot_def_mean = float(mdot_def_acc.item()) / fuel_info_count
                        mdot_att_mean = float(mdot_att_acc.item()) / fuel_info_count
                    else:
                        fuel_used_def_mean = fuel_used_def_acc / fuel_info_count
                        fuel_used_att_mean = fuel_used_att_acc / fuel_info_count
                        fuel_frac_def_mean = fuel_frac_def_acc / fuel_info_count
                        fuel_frac_att_mean = fuel_frac_att_acc / fuel_info_count
                        thrust_def_mean = thrust_def_acc / fuel_info_count
                        thrust_att_mean = thrust_att_acc / fuel_info_count
                        mdot_def_mean = mdot_def_acc / fuel_info_count
                        mdot_att_mean = mdot_att_acc / fuel_info_count
                else:
                    fuel_used_def_mean = fuel_used_att_mean = 0.0
                    fuel_frac_def_mean = fuel_frac_att_mean = 0.0
                    thrust_def_mean = thrust_att_mean = 0.0
                    mdot_def_mean = mdot_att_mean = 0.0

                metrics["fuel_used_def_mean"].append(fuel_used_def_mean)
                metrics["fuel_used_att_mean"].append(fuel_used_att_mean)
                metrics["fuel_frac_def_mean"].append(fuel_frac_def_mean)
                metrics["fuel_frac_att_mean"].append(fuel_frac_att_mean)

                metrics["thrust_def_mean"].append(thrust_def_mean)
                metrics["thrust_att_mean"].append(thrust_att_mean)
                metrics["mdot_def_mean"].append(mdot_def_mean)
                metrics["mdot_att_mean"].append(mdot_att_mean)


            # ---- store in metrics ----
            metrics["update"].append(upd)
            metrics["R_def_mean"].append(R_def_mean)
            metrics["R_att_mean"].append(R_att_mean)
            metrics["muD_abs_mean"].append(muD)
            metrics["stdD_mean"].append(stdD)
            metrics["arena_radius_mean"].append(arena_radius_mean)

            metrics["d1_mean"].append(d1_true_mean)
            metrics["d2_mean"].append(d2_belief_mean)
            metrics["d2_true_mean"].append(d2_true_mean)
            metrics["p2_est_err_mean"].append(p2_est_err_mean)
            metrics["p1_est_err_mean"].append(p1_est_err_mean)
            metrics["meas_innov_mean"].append(meas_innov_mean)
            metrics["ukf_trPpos_mean"].append(trP_mean)

            metrics["lr_pi"].append(lr_pi)
            metrics["lr_vf"].append(lr_vf)
            metrics["prior_blend_def"].append(current_prior_blend_def)
            metrics["rollout_time_s"].append(rollout_time_s)
            metrics["optim_time_s"].append(optim_time_s)
            metrics["update_time_s"].append(update_time_s)
            metrics["env_steps_per_sec"].append(env_steps_per_sec)
            if opp_policy_mix is not None:
                total_mix = (
                    max(1, int(mix_counts_update.sum().item()))
                    if vec_is_torch
                    else max(1, int(mix_counts_update.sum()))
                )
                mix_summary = []
                for idx, entry in enumerate(opp_policy_mix["entries"]):
                    frac = (
                        float(mix_counts_update[idx].item()) / total_mix
                        if vec_is_torch
                        else float(mix_counts_update[idx]) / total_mix
                    )
                    metrics[mix_name_to_metric[entry["name"]]].append(frac)
                    mix_summary.append(f"{entry['name']}={frac:.2f}")
                print("opp mix usage:", ", ".join(mix_summary))

            print(f"[update {upd:05d}] R_def_mean={R_def_mean:+.3f}  R_att_mean={R_att_mean:+.3f}  (batch={num_envs*steps_per_env})")
            print(
                f"   [{train_role}] |mu|_mean={muD:.3e}  std_mean={stdD:.3e}  "
                f"prior_blend_def={current_prior_blend_def:.3f}"
            )
            print(f"   mean arena radius ≈ {arena_radius_mean:.3f} m")
            print(
                f"   timing: rollout={rollout_time_s:.3f}s  "
                f"optim={optim_time_s:.3f}s  total={update_time_s:.3f}s  "
                f"steps/s={env_steps_per_sec:.1f}"
            )
            print(f"   approx true <||p1-center||> ≈ {d1_true_mean:.3f}")
            print(f"   approx true <||p2-center||> ≈ {d2_true_mean:.3f}")
            if cfg.get("use_kf", False):
                estimator_label = str(cfg.get("estimator_kind", "ukf")).upper()
                print(f"   approx belief <||p2-center||> ≈ {d2_belief_mean:.3f}")
                print(f"   {estimator_label} RMS pos err: def->att={p2_est_err_mean:.3f}, att->def={p1_est_err_mean:.3f}")
                print(f"   meas_innov_mean={meas_innov_mean:.3e},  trPpos_mean={trP_mean:.3e}")

            if cfg.get("fuel", {}).get("enable", False):
                print(
                    f"   fuel used: def={fuel_used_def_mean:.6f}, att={fuel_used_att_mean:.6f}   "
                    f"fuel remaining: def={fuel_frac_def_mean:.6f}, att={fuel_frac_att_mean:.6f}"
                )
                print(
                    f"   thrust mean: def={thrust_def_mean:.6e}, att={thrust_att_mean:.6e}   "
                    f"mdot mean: def={mdot_def_mean:.6e}, att={mdot_att_mean:.6e}"
                )

            
            if writer is not None:
                gs = global_env_step  # x-axis = env steps

                # ===== Returns =====
                writer.add_scalar("returns/def_mean", R_def_mean, gs)
                writer.add_scalar("returns/att_mean", R_att_mean, gs)
                writer.add_scalar("arena/radius_mean_m", arena_radius_mean, gs)

                # ===== Distances (meters) =====
                writer.add_scalar("dist/def_true_p1_to_center_m", d1_true_mean, gs)
                writer.add_scalar("dist/att_true_p2_to_center_m", d2_true_mean, gs)
                writer.add_scalar("dist/att_belief_p2_to_center_m", d2_belief_mean, gs)
                writer.add_scalar("ukf/p2_est_err_rms_m", p2_est_err_mean, gs)
                writer.add_scalar("ukf/p1_est_err_rms_m", p1_est_err_mean, gs)

                # ===== Policy stats =====
                writer.add_scalar("policy/def_mu_abs_mean", muD, gs)
                writer.add_scalar("policy/def_std_mean", stdD, gs)
                writer.add_scalar("policy/prior_blend_def", current_prior_blend_def, gs)

                # ===== Learning rates =====
                writer.add_scalar("lr/def_policy", lr_pi, gs)
                writer.add_scalar("lr/def_value",  lr_vf, gs)
                writer.add_scalar(f"lr/{train_role}_policy", lr_pi, gs)
                writer.add_scalar(f"lr/{train_role}_value",  lr_vf, gs)
                writer.add_scalar("lr/policy", lr_pi, gs)
                writer.add_scalar("lr/value", lr_vf, gs)

                # ===== Throughput =====
                writer.add_scalar("timing/rollout_s", rollout_time_s, gs)
                writer.add_scalar("timing/optim_s", optim_time_s, gs)
                writer.add_scalar("timing/update_s", update_time_s, gs)
                writer.add_scalar("timing/env_steps_per_sec", env_steps_per_sec, gs)

                # ===== UKF stats (if enabled) =====
                if cfg.get("use_kf", False):
                    writer.add_scalar("ukf/meas_innov_sq_mean", meas_innov_mean, gs)
                    writer.add_scalar("ukf/trP_pos_mean", trP_mean, gs)

                if info_count > 0:
                    term_rates = {k: v / info_count for k, v in term_counts_log.items()}
                else:
                    term_rates = {k: 0.0 for k in term_counts_log}


                writer.add_scalar("term_rate/oob_def", term_rates["oob_def"], gs)
                writer.add_scalar("term_rate/oob_att", term_rates["oob_att"], gs)
                writer.add_scalar("term_rate/hit_target", term_rates["hit_target"], gs)
                writer.add_scalar("term_rate/collision", term_rates["collision"], gs)

                writer.add_scalar("act/def_abs_mean", a1.abs().mean().item(), global_env_step)
                writer.add_scalar("act/def_abs_max",  a1.abs().max().item(),  global_env_step)

                if cfg.get("fuel", {}).get("enable", False):
                    writer.add_scalar("fuel/used_def_mean", fuel_used_def_mean, gs)
                    writer.add_scalar("fuel/used_att_mean", fuel_used_att_mean, gs)
                    writer.add_scalar("fuel/remaining_def_mean", fuel_frac_def_mean, gs)
                    writer.add_scalar("fuel/remaining_att_mean", fuel_frac_att_mean, gs)

                    writer.add_scalar("fuel/thrust_def_mean", thrust_def_mean, gs)
                    writer.add_scalar("fuel/thrust_att_mean", thrust_att_mean, gs)
                    writer.add_scalar("fuel/mdot_def_mean", mdot_def_mean, gs)
                    writer.add_scalar("fuel/mdot_att_mean", mdot_att_mean, gs)



    ic_used_path = None
    if cfg.get("record_ic_history", False) and checkpoint_dir is not None:
        def_used, att_used = collect_ic_history_from_vecenv(vec)
        ic_used_path = os.path.join(checkpoint_dir, f"ic_samples_{checkpoint_prefix}.npz")
        np.savez(
            ic_used_path,
            def_pos=def_used,
            att_pos=att_used,
        )
        print(f"[ic] saved actual training IC samples -> {ic_used_path}")

            
            
    # ---- end-of-train cleanup ----
    try:
        del bufD
    except: pass
    try:
        del bufA
    except: pass
    try:
        vec.close()
    except Exception:
        pass
    try:
        del vec
    except: pass


    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if writer is not None:
        writer.flush()
        writer.close()

    if use_intercept_train_only:
        ppo.def_net.set_prior_blend("def", intercept_blend_end)

    if save_last_ckpt and checkpoint_dir is not None:
        last_ckpt_path = os.path.join(
            checkpoint_dir,
            f"{checkpoint_prefix}.pt"
        )
        _save_role_checkpoint(ppo, train_role, last_ckpt_path)
        print(f"[checkpoint] saved last checkpoint -> {last_ckpt_path}")

    ckpt_info = {
        "tracked_metric_name": tracked_metric_name,
        "best_metric": best_metric,
        "best_update": best_update,
        "best_ckpt_path": best_ckpt_path,
        "last_ckpt_path": last_ckpt_path,
        "ic_used_path": ic_used_path,

    }

    print("Training finished.")
    return ppo, metrics, ckpt_info


def evaluate(ppo: PPO, cfg: Dict[str, Any], episodes: int = 2):
    """Run a policy evaluation rollout and return summary metrics."""
    def _pad_xyz(series: np.ndarray) -> np.ndarray:
        """Internal helper for pad xyz."""
        arr = np.asarray(series, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"Expected state series with shape (T, D), got {arr.shape}.")
        if arr.shape[1] == 2:
            return np.hstack([arr, np.zeros((arr.shape[0], 1), dtype=float)])
        return arr[:, :3]

    env = Env(cfg)
    trajs = []
    for _ in range(episodes):
        obs = env.reset()
        states = [env.state.copy()]
        actions = []
        infos = []

        done = False
        while not done:
            o_t = torch.as_tensor(obs[None, :], dtype=torch.float32, device=ppo.def_net.logstd.device)
            with torch.no_grad():
                a1, _, _ = ppo.act(o_t, who="def")
                a2, _, _ = ppo.act(o_t, who="att")
            a1_np = a1.squeeze(0).cpu().numpy()
            a2_np = a2.squeeze(0).cpu().numpy()
            obs, r1, r2, done, info = env.step(a1_np, a2_np)
            states.append(env.state.copy())
            actions.append((a1_np.copy(), a2_np.copy()))
            infos.append(info)
        states_arr = np.stack(states)
        D = int(cfg.get("D", 3))
        exec_xyz_all = []
        vel_xyz_all = []
        for idx in range(2):
            base = 2 * idx * D
            exec_xyz_all.append(_pad_xyz(states_arr[:, base : base + D]))
            vel_xyz_all.append(_pad_xyz(states_arr[:, base + D : base + (2 * D)]))

        traj = {
            "states": states_arr,
            "actions": actions,
            "infos": infos,
            "exec_xyz_all": exec_xyz_all,
            "vel_xyz_all": vel_xyz_all,
        }
        for idx, (P, V) in enumerate(zip(exec_xyz_all, vel_xyz_all), start=1):
            traj[f"exec{idx}_xyz"] = P
            traj[f"vel{idx}_xyz"] = V
        trajs.append(traj)
    return trajs

    # ---------------------------------------------------------
    # Helper: train defender teacher + distill to estimator-backed student
    # ---------------------------------------------------------

def train_with_distill(
    phase_name: str,
    attacker_mode: str,
    train_role: str,
    out_dir: str,
    extra_train_cfg: Dict[str, Any] | None = None,
):
    """
    Unified wrapper for:
      - teacher training (full-state)
      - optional estimator-backed student distillation

    Replaces both:
      - train_defender_with_distill(...)
      - train_attacker_with_distill(...)

    Args:
        phase_name: label used for checkpoints / metrics filenames
        attacker_mode: "rule" or "rl"
        train_role: "def" or "att"
        extra_train_cfg: optional overrides merged into both teacher and student cfgs

    Returns
    -------
    teacher_ckpt : str
        Path to best teacher checkpoint (or last if best missing)
    student_ckpt : str | None
        Path to distilled student checkpoint if distill=True, else None
    """
    if train_role not in ("def", "att"):
        raise ValueError(f"train_role must be 'def' or 'att', got {train_role!r}")

    role_upper = "DEFENDER" if train_role == "def" else "ATTACKER"
    role_lower = "defender" if train_role == "def" else "attacker"

    # =========================================================
    # TEACHER
    # =========================================================
    cfg_teacher = config_for_train(
        attacker_mode=attacker_mode,
        train_role=train_role,
    )
    if extra_train_cfg is not None:
        cfg_teacher = _deep_merge_dict(cfg_teacher, extra_train_cfg)

    # if cfg_teacher["train_ic_mode"] == "random_shell_advantage":
    #     if train_role == "att": 
    #         cfg_teacher["r_att_min"] = 0.0
    #         cfg_teacher["train_ic_mode"] = "random_shell"

    DISTILL = bool(cfg_teacher.get("distill", False))
    DISTILL_METHOD = str(cfg_teacher.get("distill_method", "modern"))

    build_dyn(cfg_teacher)

    # dynamics_config = cfg_teacher["dyn"]
    # print(dynamics_config["Ad"])
    # print(dynamics_config["Bd"])

    # raise("Debug")

    if cfg_teacher["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"[{phase_name.upper()} TEACHER] device='cuda' but CUDA not available."
        )

    print(f"[{phase_name.upper()} TEACHER] Using device: {cfg_teacher['device']}")
    print(f"[{phase_name.upper()}] train_role={train_role}  distill={DISTILL}  distill_method={DISTILL_METHOD}")

    cfg_teacher["checkpoint_dir"] = out_dir
    cfg_teacher["checkpoint_prefix"] = phase_name + "_teacher"
    if cfg_teacher.get("use_tensorboard", False):
        cfg_teacher["tb_run_name"] = _tb_run_name(cfg_teacher, phase_name, "teacher")

    ppo_teacher, metrics_teacher, ckpt_info = train(cfg_teacher)

    teacher_ckpt = ckpt_info["last_ckpt_path"]
    teacher_ckpt_best = ckpt_info["best_ckpt_path"]

    print(f"[{phase_name.upper()} TEACHER] Using {role_lower} checkpoint: {teacher_ckpt}")
    print(
        f"[{phase_name.upper()} TEACHER] "
        f"best {ckpt_info['tracked_metric_name']}={ckpt_info['best_metric']:+.3f} "
        f"at update {ckpt_info['best_update']}"
    )

    metrics_path = os.path.join(out_dir, f"train_metrics_{phase_name}_teacher.npz")
    np.savez(metrics_path, **metrics_teacher)
    print(f"[{phase_name.upper()} TEACHER] Saved metrics to {metrics_path}")

    try:
        del ppo_teacher, metrics_teacher
    except Exception:
        pass

    # =========================================================
    # STUDENT (estimator-observation) via configurable distillation method
    # =========================================================
    student_out = None
    distill_duration_s = None

    if DISTILL:
        cfg_student = config_for_train(
            attacker_mode=attacker_mode,
            train_role=train_role,
        )
        cfg_student["use_kf"] = True
        cfg_student["seed"] = cfg_teacher["seed"] + 1

        if extra_train_cfg is not None:
            cfg_student = _deep_merge_dict(cfg_student, extra_train_cfg)

        student_estimator_kind = str(
            cfg_teacher.get("estimator_kind", cfg_student.get("estimator_kind", "ukf"))
        ).lower()
        cfg_student["estimator_kind"] = student_estimator_kind

        build_dyn(cfg_student)

        if cfg_student["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"[{phase_name.upper()} STUDENT] device='cuda' but CUDA not available."
            )

        print(f"[{phase_name.upper()} STUDENT] Using device: {cfg_student['device']}")
        if cfg_student.get("use_tensorboard", False):
            cfg_student["tb_run_name"] = _tb_run_name(cfg_student, phase_name, "student")

        student_out = os.path.join(out_dir, f"{phase_name}_kf_student.pt")
        distill_t0 = time.perf_counter()

        student, metrics_student = distill_from_teacher(
            cfg_student,
            teacher_ckpt,
            out_path=student_out,
        )
        distill_duration_s = float(time.perf_counter() - distill_t0)

        print(
            f"[{phase_name.upper()} STUDENT] "
            f"Distilled {student_estimator_kind.upper()} student saved to {student_out}"
        )
        print(f"[{phase_name.upper()} STUDENT] Distillation time: {distill_duration_s:.3f} s")

        distill_metrics_path = os.path.join(
            out_dir, f"distill_metrics_{phase_name}_student.npz"
        )
        np.savez(distill_metrics_path, **metrics_student)
        print(f"[{phase_name.upper()} STUDENT] Saved distillation metrics to {distill_metrics_path}")

        try:
            del student, metrics_student
        except Exception:
            pass

    meta = {
        "train_role": train_role,
        "teacher_ckpt": teacher_ckpt,
        "teacher_best_metric_name": ckpt_info["tracked_metric_name"],
        "teacher_best_metric": ckpt_info["best_metric"],
        "teacher_best_update": ckpt_info["best_update"],
        "teacher_best_ckpt_path": ckpt_info["best_ckpt_path"],
        "teacher_last_ckpt_path": ckpt_info["last_ckpt_path"],
        "student_ckpt": student_out,
        "distill_enabled": DISTILL,
        "distill_method": DISTILL_METHOD,
        "distill_duration_s": distill_duration_s,
    }

    return teacher_ckpt, student_out, meta




# =============================================================
# End phase cleanup
# =============================================================

def end_phase_cleanup(
    tag: str = "",
    *,
    clear_cuda: bool = True,
    clear_ipc: bool = True,
    clear_mps: bool = True,
    clear_matplotlib: bool = True,
    sleep_s: float = 0.0,
):
    """
    Best-effort memory cleanup after a training phase.

    - CPU RAM: delete references + gc.collect()
    - GPU VRAM (CUDA): empty_cache(), ipc_collect()
    - MPS (Apple): empty_cache()
    - Matplotlib: close figures so they don't accumulate
    """
    print(f"\n[cleanup] {tag} ...")

    # ---- close any lingering matplotlib figures (common silent RAM leak) ----
    if clear_matplotlib:
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass

    # ---- Python heap cleanup ----
    gc.collect()

    # ---- PyTorch device-specific cleanup ----
    if clear_cuda and torch.cuda.is_available():
        # Clears cached blocks held by the CUDA allocator (does not free tensors you still reference).
        torch.cuda.empty_cache()

        if clear_ipc:
            # Helps in some multi-process / DataLoader / vector-env setups.
            torch.cuda.ipc_collect()

        # Optional: if you're debugging fragmentation, you can print stats:
        # print(torch.cuda.memory_summary())

    if clear_mps and hasattr(torch, "mps") and torch.mps.is_available():
        # Apple Silicon
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    if sleep_s > 0:
        time.sleep(sleep_s)

    print(f"[cleanup] {tag} done.")


def rollout_metrics(states: np.ndarray, center: np.ndarray, R: float):
    """
    states: [T+1, 12] for D=3 => [p1(3), v1(3), p2(3), v2(3)]
    center: (D,)
    R: arena radius (m)
    """
    D = center.shape[0]
    p1 = states[:, 0:D];      v1 = states[:, D:2*D]
    p2 = states[:, 2*D:3*D];  v2 = states[:, 3*D:4*D]

    d1 = np.sum((p1 - center)**2, axis=1) / (R*R)
    d2 = np.sum((p2 - center)**2, axis=1) / (R*R)
    rel2 = np.sum((p2 - p1)**2, axis=1) / (R*R)
    d2_delta = np.diff(d2, prepend=d2[:1])

    return {"d1_norm": d1, "d2_norm": d2, "rel2_norm": rel2, "d2_delta": d2_delta}
