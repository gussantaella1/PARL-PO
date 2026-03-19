from __future__ import annotations

import gc
import os
import time
from collections import Counter
from typing import Any, Dict

import numpy as np
import torch

from config_rl import build_dyn, config_for_train
from core.distill import distill_from_teacher
from core_1v2.buffers import RolloutBuffer
from core_1v2.env import Env, VecEnv, collect_ic_history_from_vecenv
from core_1v2.freeze_utils import (
    assert_deterministic_action,
    assert_frozen_unchanged,
    freeze_module_,
    snapshot_state_dict,
)
from core_1v2.models import ActorCriticDiff
from core_1v2.plotting import make_tb_writer
from core_1v2.ppo import PPO
from core_1v2.utils import permute_obs_for_attacker, set_seed, squash_action


def _cpu_state_dict(m: torch.nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in m.state_dict().items()}


def _save_role_checkpoint(ppo, train_role: str, path: str):
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
        entries.append(
            {
                "name": name,
                "net": net,
                "prob": float(item.get("prob", 0.0)),
                "path": path,
                "action_scale": float(item.get("action_scale", 1.0)),
                "noise_std": float(item.get("noise_std", 0.0)),
                "idle_prob": float(item.get("idle_prob", 0.0)),
            }
        )

    probs = _normalize_probs(entries)
    resample = str(mix.get("resample", "episode")).lower()
    if resample not in ("episode", "update", "never"):
        raise ValueError(
            f"Unsupported opp_mix resample={resample!r}; expected 'episode', 'update', or 'never'."
        )

    return {
        "opp_role": opp_role,
        "entries": entries,
        "probs": probs,
        "resample": resample,
        "act_dim": act_dim,
        "num_attackers": int(cfg.get("num_attackers", 1)),
        "D": int(cfg["D"]),
    }


def _sample_opponent_indices(mix_state, size: int):
    return np.random.choice(len(mix_state["entries"]), size=size, p=mix_state["probs"]).astype(np.int64)


@torch.no_grad()
def _act_from_opponent_policy_mix(
    obs_batch: torch.Tensor,
    mix_state,
    active_indices: np.ndarray,
    act_scale: float,
):
    opp_role = mix_state["opp_role"]
    act_dim = int(mix_state["act_dim"])
    Na = int(mix_state["num_attackers"])
    D = int(mix_state["D"])

    if opp_role == "att" and Na > 1:
        out = torch.zeros((obs_batch.shape[0], Na, act_dim), dtype=obs_batch.dtype, device=obs_batch.device)
    else:
        out = torch.zeros((obs_batch.shape[0], act_dim), dtype=obs_batch.dtype, device=obs_batch.device)

    unique_ids = np.unique(active_indices)
    for idx in unique_ids:
        mask_np = active_indices == idx
        mask = torch.as_tensor(mask_np, dtype=torch.bool, device=obs_batch.device)
        entry = mix_state["entries"][int(idx)]
        net = entry["net"]

        if opp_role == "att" and Na > 1:
            a_list = []
            for k in range(Na):
                obs_k = permute_obs_for_attacker(obs_batch[mask], k, D, Na)
                dist = net.dist(obs_k, who="att")
                a_list.append(squash_action(dist.mean, act_scale))
            a = torch.stack(a_list, dim=1)
        else:
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
            idle_shape = (a.shape[0], 1, 1) if (opp_role == "att" and Na > 1) else (a.shape[0], 1)
            idle_mask = torch.rand(idle_shape, device=a.device) < idle_prob
            a = torch.where(idle_mask, torch.zeros_like(a), a)

        a = torch.clamp(a, -act_scale, act_scale)
        out[mask] = a

    return out


def train(cfg: Dict[str, Any]):
    set_seed(cfg["seed"])
    device = cfg["device"]

    writer = None
    tb_logdir = None
    global_env_step = 0

    if cfg.get("use_tensorboard", False):
        writer, tb_logdir = make_tb_writer(cfg)

    train_role = cfg.get("train_role", "def")
    reward_mode = "def" if train_role == "def" else "att"

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

    def make_env():
        return Env(cfg)

    num_envs = int(cfg.get("num_envs"))
    steps_per_env = int(cfg.get("steps_per_env"))
    total_updates = int(cfg.get("total_updates"))
    log_every = int(cfg.get("log_every"))
    Na = int(cfg.get("num_attackers", 1))

    vec = VecEnv(make_env, num_envs)
    obs_dim = vec.obs.shape[1]
    act_dim = int(cfg["D"])

    ppo = PPO(obs_dim, act_dim, cfg, device=device)

    att_init = cfg.get("att_init_path", None)
    if att_init is not None:
        if ppo.att_net is None:
            raise RuntimeError("att_init_path provided but attacker_mode != 'rl'")
        state = torch.load(att_init, map_location=device)
        ppo.att_net.load_state_dict(state)
        print(f"[train] Loaded attacker init from: {att_init}")

    def_ckpt = cfg.get("def_ckpt_path", None)
    if def_ckpt is not None:
        state = torch.load(def_ckpt, map_location=device)
        ppo.def_net.load_state_dict(state)
        if cfg.get("freeze_defender", False):
            for p in ppo.def_net.parameters():
                p.requires_grad_(False)

    att_ckpt = cfg.get("att_ckpt_path", None)
    if att_ckpt is not None and ppo.att_net is not None:
        state = torch.load(att_ckpt, map_location=device)
        ppo.att_net.load_state_dict(state)
        if cfg.get("freeze_attacker", False):
            for p in ppo.att_net.parameters():
                p.requires_grad_(False)

    if cfg.get("freeze_defender", False):
        freeze_module_(ppo.def_net)
    if cfg.get("freeze_attacker", False) and (ppo.att_net is not None):
        freeze_module_(ppo.att_net)

    verify_freeze = bool(cfg.get("verify_freeze"))
    freeze_tol = float(cfg.get("freeze_tol"))
    snap_def = snapshot_state_dict(ppo.def_net) if verify_freeze and cfg.get("freeze_defender") else None
    snap_att = (
        snapshot_state_dict(ppo.att_net)
        if verify_freeze and cfg.get("freeze_attacker") and (ppo.att_net is not None)
        else None
    )

    lr_schedule = cfg.get("lr_schedule", "none")
    lr_final_factor = float(cfg.get("lr_final_factor", 0.1))

    metrics = {
        "update": [],
        "R_def_mean": [],
        "R_att_mean": [],
        "muD_abs_mean": [],
        "stdD_mean": [],
        "d1_mean": [],
        "d2_mean": [],
        "d2_true_mean": [],
        "meas_innov_mean": [],
        "ukf_trPpos_mean": [],
        "lr_pi": [],
        "lr_vf": [],
    }

    if cfg.get("fuel", {}).get("enable", False):
        metrics["fuel_used_def_mean"] = []
        metrics["fuel_used_att_mean"] = []
        metrics["fuel_frac_def_mean"] = []
        metrics["fuel_frac_att_mean"] = []
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
            entry["name"]: f"opp_mix__{entry['name']}" for entry in opp_policy_mix["entries"]
        }
        for metric_name in mix_name_to_metric.values():
            metrics[metric_name] = []
        mix_active_idx = _sample_opponent_indices(opp_policy_mix, num_envs)

    def_center_base = cfg.get("def_center_coef", 0.0)
    min_anneal = float(cfg.get("def_center_min_anneal", 0.5))

    for upd in range(1, total_updates + 1):
        term_counts = {"oob_def": 0, "oob_att": 0, "hit_target": 0, "collision": 0}
        mix_counts_update = Counter()

        if opp_policy_mix is not None and opp_policy_mix["resample"] == "update":
            mix_active_idx = _sample_opponent_indices(opp_policy_mix, num_envs)

        if lr_schedule == "linear":
            frac_lr = (upd - 1) / max(1, total_updates - 1)
            scale = 1.0 - frac_lr * (1.0 - lr_final_factor)
            for g, base in zip(ppo.def_opt.param_groups, ppo.def_base_lrs):
                g["lr"] = base * scale
            if ppo.att_opt is not None and getattr(ppo, "att_base_lrs", None) is not None:
                for g, base in zip(ppo.att_opt.param_groups, ppo.att_base_lrs):
                    g["lr"] = base * scale

        center_frac = upd / max(1, total_updates)
        k_cent_mul = 1.0 - (1.0 - min_anneal) * center_frac
        for e in vec.envs:
            e.k_cent = def_center_base * k_cent_mul

        bufD = RolloutBuffer(obs_dim, act_dim, num_envs, steps_per_env, device)
        rule_att = cfg.get("attacker_mode", "rule") == "rule"
        if (not rule_att) and train_role in ("att", "both"):
            bufA_envs = num_envs * Na if Na > 1 else num_envs
            bufA = RolloutBuffer(obs_dim, act_dim, bufA_envs, steps_per_env, device)
        else:
            bufA = None

        o = torch.as_tensor(vec.obs, dtype=torch.float32, device=device)
        ep_ret_def = np.zeros(num_envs, dtype=np.float64)
        ep_ret_att = np.zeros(num_envs, dtype=np.float64)

        d1_true_acc = 0.0
        d2_true_acc = 0.0
        d2_belief_acc = 0.0
        meas_innov_acc = 0.0
        trP_acc = 0.0
        info_count = 0

        if cfg.get("fuel", {}).get("enable", False):
            fuel_used_def_acc = 0.0
            fuel_used_att_acc = 0.0
            fuel_frac_def_acc = 0.0
            fuel_frac_att_acc = 0.0
            thrust_def_acc = 0.0
            thrust_att_acc = 0.0
            mdot_def_acc = 0.0
            mdot_att_acc = 0.0
            fuel_info_count = 0

        for _ in range(steps_per_env):
            with torch.no_grad():
                if train_role == "def":
                    a1, lp1, v1 = ppo.act(o, who="def", deterministic=False)
                    if opp_policy_mix is not None:
                        a2 = _act_from_opponent_policy_mix(o, opp_policy_mix, mix_active_idx, ppo.act_scale)
                    else:
                        a2, _, _ = ppo.act(o, who="att", deterministic=True)
                    lp2 = None
                    v2 = None
                elif train_role == "att":
                    if opp_policy_mix is not None:
                        a1 = _act_from_opponent_policy_mix(o, opp_policy_mix, mix_active_idx, ppo.act_scale)
                    else:
                        a1, _, _ = ppo.act(o, who="def", deterministic=True)
                    a2, lp2, v2 = ppo.act(o, who="att", deterministic=False)
                    lp1 = None
                    v1 = None
                else:
                    raise ValueError(f"Unknown train_role={train_role!r}")

            a1_np = a1.cpu().numpy()
            a2_np = a2.cpu().numpy()
            if a2_np.ndim == 2 and cfg.get("attacker_mode", "rule") == "rule" and Na == 1:
                a2_np = a2_np[:, None, :]

            o2_np, r1_np, r2_np, d_np, infos = vec.step(a1_np, a2_np, reward_mode=reward_mode)

            o2 = torch.as_tensor(o2_np, dtype=torch.float32, device=device)
            r1 = torch.as_tensor(r1_np, dtype=torch.float32, device=device)
            r2 = torch.as_tensor(r2_np, dtype=torch.float32, device=device)
            d = torch.as_tensor(d_np, dtype=torch.float32, device=device)

            logp1_store = lp1.detach() if lp1 is not None else torch.zeros_like(r1)
            val1_store = v1.detach() if v1 is not None else torch.zeros_like(r1)
            bufD.add(o.detach(), a1.detach(), logp1_store, val1_store, r1, d)

            if bufA is not None:
                if Na == 1:
                    bufA.add(o.detach(), a2.detach(), lp2.detach(), v2.detach(), r2, d)
                else:
                    o_perm = [permute_obs_for_attacker(o, k, cfg["D"], Na) for k in range(Na)]
                    o_att_step = torch.cat(o_perm, dim=0)
                    a_att_step = torch.cat([a2[:, k, :] for k in range(Na)], dim=0)
                    lp_att_step = torch.cat([lp2[:, k] for k in range(Na)], dim=0)
                    v_att_step = torch.cat([v2[:, k] for k in range(Na)], dim=0)
                    if infos and all("r_att_each" in inf for inf in infos):
                        rA_np = np.stack([inf["r_att_each"] for inf in infos], axis=0).astype(np.float32)
                    else:
                        # The shared env currently exposes only the team attacker reward.
                        # Reuse that scalar for each shared-policy attacker sample.
                        rA_np = np.repeat(r2_np[:, None].astype(np.float32), Na, axis=1)
                    r_att_step = torch.as_tensor(
                        np.concatenate([rA_np[:, k] for k in range(Na)], axis=0),
                        dtype=torch.float32,
                        device=device,
                    )
                    d_att_step = d.repeat(Na)
                    bufA.add(
                        o_att_step.detach(),
                        a_att_step.detach(),
                        lp_att_step.detach(),
                        v_att_step.detach(),
                        r_att_step,
                        d_att_step,
                    )

            if train_role == "def":
                ep_ret_def += r1_np
            if train_role == "att":
                ep_ret_att += r2_np

            o = o2

            if opp_policy_mix is not None:
                mix_counts_update.update(int(i) for i in mix_active_idx.tolist())
                if opp_policy_mix["resample"] == "episode":
                    done_mask = d_np.astype(bool)
                    if np.any(done_mask):
                        mix_active_idx[done_mask] = _sample_opponent_indices(
                            opp_policy_mix, int(done_mask.sum())
                        )

            for inf in infos:
                info_count += 1
                if "d1_true_norm" in inf:
                    d1_true_acc += inf["d1_true_norm"]
                if "d2_true_norm" in inf:
                    d2_true_acc += inf["d2_true_norm"]
                if "d2_belief_norm" in inf:
                    d2_belief_acc += inf["d2_belief_norm"]
                elif "d2_true_norm" in inf:
                    d2_belief_acc += inf["d2_true_norm"]
                elif "d2_norm" in inf:
                    d2_belief_acc += inf["d2_norm"]
                if "meas_innov_sq" in inf:
                    meas_innov_acc += inf["meas_innov_sq"]
                if "ukf_trPpos" in inf:
                    trP_acc += inf["ukf_trPpos"]
                if inf.get("oob_def", False):
                    term_counts["oob_def"] += 1
                if inf.get("oob_att", False):
                    term_counts["oob_att"] += 1
                if inf.get("hit_target", False):
                    term_counts["hit_target"] += 1
                if inf.get("collision", False):
                    term_counts["collision"] += 1

                if cfg.get("fuel", {}).get("enable", False) and "fuel_used_def" in inf:
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

        with torch.no_grad():
            next_v_def = ppo.def_net.value(o)
        bufD.finalize(next_v_def)

        if bufA is not None:
            with torch.no_grad():
                if Na == 1:
                    next_v_att = ppo.att_net.value(o)
                else:
                    next_v_list = [
                        ppo.att_net.value(permute_obs_for_attacker(o, k, cfg["D"], Na)) for k in range(Na)
                    ]
                    next_v_att = torch.cat(next_v_list, dim=0)
            bufA.finalize(next_v_att)

        if train_role == "def":
            ppo.update_defender_only(bufD)
        elif train_role == "att":
            if bufA is None:
                raise RuntimeError("train_role='att' requires attacker_mode='rl'")
            ppo.update_attacker_only(bufA)
        else:
            raise ValueError(f"Unknown train_role={train_role!r}")

        if verify_freeze:
            if train_role == "att" and (snap_def is not None):
                assert_frozen_unchanged(snap_def, ppo.def_net, name="frozen_defender", tol=freeze_tol)
            if train_role == "def" and (snap_att is not None):
                assert_frozen_unchanged(snap_att, ppo.att_net, name="frozen_attacker", tol=freeze_tol)

        if upd % log_every == 0:
            R_def_mean = ep_ret_def.mean()
            R_att_mean = ep_ret_att.mean()
            tracked_metric_value = R_def_mean if train_role == "def" else R_att_mean

            if save_best_ckpt and checkpoint_dir is not None and tracked_metric_value > best_metric:
                best_metric = float(tracked_metric_value)
                best_update = int(upd)
                best_ckpt_path = os.path.join(checkpoint_dir, f"{checkpoint_prefix}__best.pt")
                _save_role_checkpoint(ppo, train_role, best_ckpt_path)
                print(
                    f"[checkpoint] new best {tracked_metric_name}={best_metric:+.3f} "
                    f"at update {best_update} -> {best_ckpt_path}"
                )

            if info_count > 0:
                R = cfg["arena"]["r"]
                d1_true_mean = np.sqrt(d1_true_acc / info_count) * R
                d2_true_mean = np.sqrt(d2_true_acc / info_count) * R
                d2_belief_mean = np.sqrt(d2_belief_acc / info_count) * R
                meas_innov_mean = meas_innov_acc / info_count
                trP_mean = trP_acc / info_count
            else:
                d1_true_mean = d2_true_mean = d2_belief_mean = 0.0
                meas_innov_mean = trP_mean = 0.0

            with torch.no_grad():
                flat_obs = bufD.obs.reshape(-1, obs_dim)
                distD = ppo.def_net.dist(flat_obs, who="def")
                muD = distD.mean.abs().mean().item()
                stdD = distD.stddev.mean().item()

                if verify_freeze:
                    if train_role == "att":
                        assert_deterministic_action(ppo, flat_obs[:256], who="def", tol=0.0)
                    if train_role == "def" and (ppo.att_net is not None):
                        assert_deterministic_action(ppo, flat_obs[:256], who="att", tol=0.0)

                Dcfg = cfg["D"]
                p1c = flat_obs[:, :Dcfg]
                p2c = flat_obs[:, Dcfg:2 * Dcfg]
                d1_obs_mean = p1c.pow(2).sum(-1).mean().sqrt().item()
                d2_obs_mean = p2c.pow(2).sum(-1).mean().sqrt().item()

            lr_pi = ppo.def_opt.param_groups[0]["lr"]
            lr_vf = ppo.def_opt.param_groups[-1]["lr"]

            if cfg.get("fuel", {}).get("enable", False):
                if fuel_info_count > 0:
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

            metrics["update"].append(upd)
            metrics["R_def_mean"].append(R_def_mean)
            metrics["R_att_mean"].append(R_att_mean)
            metrics["muD_abs_mean"].append(muD)
            metrics["stdD_mean"].append(stdD)
            metrics["d1_mean"].append(d1_true_mean)
            metrics["d2_mean"].append(d2_belief_mean)
            metrics["d2_true_mean"].append(d2_true_mean)
            metrics["meas_innov_mean"].append(meas_innov_mean)
            metrics["ukf_trPpos_mean"].append(trP_mean)
            metrics["lr_pi"].append(lr_pi)
            metrics["lr_vf"].append(lr_vf)

            if opp_policy_mix is not None:
                total_mix = max(1, sum(mix_counts_update.values()))
                mix_summary = []
                for idx, entry in enumerate(opp_policy_mix["entries"]):
                    frac = float(mix_counts_update.get(idx, 0)) / total_mix
                    metrics[mix_name_to_metric[entry["name"]]].append(frac)
                    mix_summary.append(f"{entry['name']}={frac:.2f}")
                print("opp mix usage:", ", ".join(mix_summary))

            print(
                f"[update {upd:05d}] R_def_mean={R_def_mean:+.3f}  "
                f"R_att_mean={R_att_mean:+.3f}  (batch={num_envs*steps_per_env})"
            )
            print(f"   [def] |mu|_mean={muD:.3e}  std_mean={stdD:.3e}")
            print(f"   approx true <||p1-center||> ≈ {d1_true_mean:.3f}")
            print(f"   approx true <||p2-center||> ≈ {d2_true_mean:.3f}")
            if cfg.get("use_ukf", False):
                print(f"   approx belief <||p2-center||> ≈ {d2_belief_mean:.3f}")
                print(f"   meas_innov_mean={meas_innov_mean:.3e},  trPpos_mean={trP_mean:.3e}")

            if writer is not None:
                gs = global_env_step
                writer.add_scalar("returns/def_mean", R_def_mean, gs)
                writer.add_scalar("returns/att_mean", R_att_mean, gs)
                writer.add_scalar("dist/def_true_p1_to_center_m", d1_true_mean, gs)
                writer.add_scalar("dist/att_true_p2_to_center_m", d2_true_mean, gs)
                writer.add_scalar("dist/att_belief_p2_to_center_m", d2_belief_mean, gs)
                writer.add_scalar("policy/def_mu_abs_mean", muD, gs)
                writer.add_scalar("policy/def_std_mean", stdD, gs)
                writer.add_scalar("lr/def_policy", lr_pi, gs)
                writer.add_scalar("lr/def_value", lr_vf, gs)
                if cfg.get("use_ukf", False):
                    writer.add_scalar("ukf/meas_innov_sq_mean", meas_innov_mean, gs)
                    writer.add_scalar("ukf/trP_pos_mean", trP_mean, gs)

                term_rates = {k: (v / info_count if info_count > 0 else 0.0) for k, v in term_counts.items()}
                writer.add_scalar("term_rate/oob_def", term_rates["oob_def"], gs)
                writer.add_scalar("term_rate/oob_att", term_rates["oob_att"], gs)
                writer.add_scalar("term_rate/hit_target", term_rates["hit_target"], gs)
                writer.add_scalar("term_rate/collision", term_rates["collision"], gs)
                writer.add_scalar("act/def_abs_mean", a1.abs().mean().item(), gs)
                writer.add_scalar("act/def_abs_max", a1.abs().max().item(), gs)

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
        np.savez(ic_used_path, def_pos=def_used, att_pos=att_used)
        print(f"[ic] saved actual training IC samples -> {ic_used_path}")

    try:
        del bufD
    except Exception:
        pass
    try:
        del bufA
    except Exception:
        pass
    try:
        del vec
    except Exception:
        pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if writer is not None:
        writer.flush()
        writer.close()

    if save_last_ckpt and checkpoint_dir is not None:
        last_ckpt_path = os.path.join(checkpoint_dir, f"{checkpoint_prefix}.pt")
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
        trajs.append({"states": np.stack(states), "actions": actions, "infos": infos})
    return trajs


def train_with_distill(
    phase_name: str,
    attacker_mode: str,
    train_role: str,
    out_dir: str,
    extra_train_cfg: Dict[str, Any] | None = None,
):
    if train_role not in ("def", "att"):
        raise ValueError(f"train_role must be 'def' or 'att', got {train_role!r}")

    role_lower = "defender" if train_role == "def" else "attacker"
    cfg_teacher = config_for_train(attacker_mode=attacker_mode, train_role=train_role)
    cfg_teacher["use_ukf"] = False

    if extra_train_cfg is not None:
        cfg_teacher.update(extra_train_cfg)

    DISTILL = bool(cfg_teacher.get("distill", False)) and int(cfg_teacher.get("num_attackers", 1)) == 1
    DISTILL_METHOD = str(cfg_teacher.get("distill_method", "modern"))

    build_dyn(cfg_teacher)

    if cfg_teacher["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"[{phase_name.upper()} TEACHER] device='cuda' but CUDA not available.")

    print(f"[{phase_name.upper()} TEACHER] Using device: {cfg_teacher['device']}")
    print(
        f"[{phase_name.upper()}] train_role={train_role}  "
        f"distill={DISTILL}  distill_method={DISTILL_METHOD}"
    )

    cfg_teacher["checkpoint_dir"] = out_dir
    cfg_teacher["checkpoint_prefix"] = phase_name + "_teacher"

    ppo_teacher, metrics_teacher, ckpt_info = train(cfg_teacher)
    teacher_ckpt = ckpt_info["last_ckpt_path"]

    print(f"[{phase_name.upper()} TEACHER] Using {role_lower} checkpoint: {teacher_ckpt}")
    print(
        f"[{phase_name.upper()} TEACHER] best {ckpt_info['tracked_metric_name']}="
        f"{ckpt_info['best_metric']:+.3f} at update {ckpt_info['best_update']}"
    )

    metrics_path = os.path.join(out_dir, f"train_metrics_{phase_name}_teacher.npz")
    np.savez(metrics_path, **metrics_teacher)
    print(f"[{phase_name.upper()} TEACHER] Saved metrics to {metrics_path}")

    try:
        del ppo_teacher, metrics_teacher
    except Exception:
        pass

    student_out = None
    distill_duration_s = None

    if DISTILL:
        cfg_student = config_for_train(attacker_mode=attacker_mode, train_role=train_role)
        cfg_student["use_ukf"] = True
        cfg_student["seed"] = cfg_teacher["seed"] + 1
        if extra_train_cfg is not None:
            cfg_student.update(extra_train_cfg)

        build_dyn(cfg_student)

        if cfg_student["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"[{phase_name.upper()} STUDENT] device='cuda' but CUDA not available.")

        print(f"[{phase_name.upper()} STUDENT] Using device: {cfg_student['device']}")
        student_out = os.path.join(out_dir, f"{phase_name}_ukf_student.pt")
        distill_t0 = time.perf_counter()
        student, metrics_student = distill_from_teacher(cfg_student, teacher_ckpt, out_path=student_out)
        distill_duration_s = float(time.perf_counter() - distill_t0)

        print(f"[{phase_name.upper()} STUDENT] Distilled UKF student saved to {student_out}")
        print(f"[{phase_name.upper()} STUDENT] Distillation time: {distill_duration_s:.3f} s")

        distill_metrics_path = os.path.join(out_dir, f"distill_metrics_{phase_name}_student.npz")
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


def end_phase_cleanup(
    tag: str = "",
    *,
    clear_cuda: bool = True,
    clear_ipc: bool = True,
    clear_mps: bool = True,
    clear_matplotlib: bool = True,
    sleep_s: float = 0.0,
):
    print(f"\n[cleanup] {tag} ...")

    if clear_matplotlib:
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except Exception:
            pass

    gc.collect()

    if clear_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
        if clear_ipc:
            torch.cuda.ipc_collect()

    if clear_mps and hasattr(torch, "mps") and torch.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    if sleep_s > 0:
        time.sleep(sleep_s)

    print(f"[cleanup] {tag} done.")


__all__ = ["train", "evaluate", "train_with_distill", "end_phase_cleanup"]
