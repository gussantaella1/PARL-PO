from __future__ import annotations

import copy
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.utils import set_seed
from core.models import ActorCriticDiff as ActorCriticDiffSingle


def _build_dyn_for_cfg(cfg: Dict[str, Any]):
    if int(cfg.get("num_attackers", 1)) > 1:
        from config_rl_1v2 import build_dyn as build_dyn_impl
    else:
        from config_rl import build_dyn as build_dyn_impl
    return build_dyn_impl(cfg)

from core_1v2.models import ActorCriticDiff as ActorCriticDiffMulti
from core.controllers import AttackerRuleController
from core_1v2.utils import _obs_offsets, permute_obs_for_attacker


def _env_cls_for_cfg(cfg: Dict[str, Any]):
    if int(cfg.get("num_attackers", 1)) > 1:
        from core_1v2.env import Env as EnvImpl
    else:
        from core.env import Env as EnvImpl
    return EnvImpl


def _num_attackers(cfg: Dict[str, Any]) -> int:
    return max(1, int(cfg.get("num_attackers", 1)))


def _actor_critic_cls(cfg: Dict[str, Any]):
    return ActorCriticDiffMulti if _num_attackers(cfg) > 1 else ActorCriticDiffSingle


def _obs_dim_from_cfg(cfg: Dict[str, Any]) -> int:
    D = int(cfg["D"])
    Na = _num_attackers(cfg)
    fuel_dim = 2 if cfg.get("fuel", {}).get("enable", False) else 0
    return (2 + 3 * Na) * D + fuel_dim


def _permute_obs_np(obs: np.ndarray, D: int, Na: int, attacker_idx: int) -> np.ndarray:
    obs_np = np.asarray(obs, dtype=np.float32)
    if Na <= 1 or attacker_idx == 0:
        return obs_np
    obs_t = torch.as_tensor(obs_np[None, :], dtype=torch.float32)
    obs_perm = permute_obs_for_attacker(obs_t, attacker_idx, D, Na)
    return obs_perm.squeeze(0).cpu().numpy().astype(np.float32)


def _relative_state_from_obs(obs: np.ndarray, D: int, Na: int) -> np.ndarray:
    obs_np = np.asarray(obs, dtype=np.float32).reshape(-1)
    _off_p1, _off_pA, off_rel, off_v1, off_vA = _obs_offsets(D, Na)
    rel_blocks = [obs_np[off_rel + k * D : off_rel + (k + 1) * D] for k in range(Na)]
    v1 = obs_np[off_v1 : off_v1 + D]
    dv_blocks = []
    for k in range(Na):
        vA_k = obs_np[off_vA + k * D : off_vA + (k + 1) * D]
        dv_blocks.append(vA_k - v1)
    return np.concatenate(rel_blocks + dv_blocks).astype(np.float32)


def _load_checkpoint_payload(path: str, map_location, label: str):
    try:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError as exc:
        if "Weights only load failed" not in str(exc):
            raise
        print(
            f"[distill] {label}: legacy checkpoint format detected; "
            "retrying torch.load(..., weights_only=False)."
        )
        return torch.load(path, map_location=map_location, weights_only=False)


# Assumes these already exist in your codebase:
# - Env
# - ActorCriticDiff
# - AttackerRuleController
# - set_seed
# - build_dyn


# =========================================================
# Paper-style modules
# =========================================================

class PrivilegedFutureIntentEncoder(nn.Module):
    """
    Paper-style teacher intent encoder E*:
      z_t = E*( future relative trajectory )

    We encode a flattened future relative-state trajectory of length H+1:
      [x_rel_t, x_rel_{t+1}, ..., x_rel_{t+H}]
    where x_rel can be chosen as [p_rel, v_rel] in your HCW setting.

    Architecture follows the paper spirit:
      3-layer MLP with [512, 256, 128] hidden units -> latent z (default 8D).
    """
    def __init__(self, traj_dim: int, latent_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(traj_dim, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, future_rel_traj: torch.Tensor) -> torch.Tensor:
        # future_rel_traj: (B, H+1, rel_dim)
        B = future_rel_traj.shape[0]
        x = future_rel_traj.reshape(B, -1)
        return self.net(x)


class PartialObsStudentPolicy(nn.Module):
    """
    Paper-style student:
      - recurrent encoder E over history of
          (xhat_rel_t, Sigma_t, u_{t-1})
      - policy pi_p conditioned on
          (xhat_rel_t, zhat_t)

    This is much closer to the paper than reusing ActorCriticDiff directly.
    """
    def __init__(
        self,
        xhat_dim: int,
        sigma_dim: int,
        act_dim: int,
        latent_dim: int = 8,
        lstm_hidden: int = 256,
        act_scale: float = 1.0,
    ):
        super().__init__()
        self.xhat_dim = xhat_dim
        self.sigma_dim = sigma_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim
        self.lstm_hidden = lstm_hidden
        self.act_scale = float(act_scale)

        enc_in = xhat_dim + sigma_dim + act_dim
        self.encoder = nn.LSTM(
            input_size=enc_in,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.z_head = nn.Linear(lstm_hidden, latent_dim)

        pol_in = xhat_dim + latent_dim
        self.policy = nn.Sequential(
            nn.Linear(pol_in, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
        )
        self.mu_raw = nn.Linear(128, act_dim)

    def init_hidden(self, batch_size: int, device: torch.device):
        h0 = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        c0 = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        return (h0, c0)

    def step(
        self,
        xhat_rel_t: torch.Tensor,   # (B, xhat_dim)
        sigma_t: torch.Tensor,      # (B, sigma_dim)
        u_prev_t: torch.Tensor,     # (B, act_dim)
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ):
        enc_in = torch.cat([xhat_rel_t, sigma_t, u_prev_t], dim=-1)   # (B, *)
        enc_in = enc_in.unsqueeze(1)                                  # (B, 1, *)
        enc_out, hidden = self.encoder(enc_in, hidden)                # (B, 1, H)
        h_t = enc_out[:, 0, :]
        zhat_t = self.z_head(h_t)

        pol_in = torch.cat([xhat_rel_t, zhat_t], dim=-1)
        h = self.policy(pol_in)
        u_raw = self.mu_raw(h)
        a_env = torch.tanh(u_raw) * self.act_scale
        return a_env, zhat_t, hidden

    def forward_chunk(
        self,
        xhat_rel_seq: torch.Tensor,  # (B, T, xhat_dim)
        sigma_seq: torch.Tensor,     # (B, T, sigma_dim)
        u_prev_seq: torch.Tensor,    # (B, T, act_dim)
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ):
        enc_in = torch.cat([xhat_rel_seq, sigma_seq, u_prev_seq], dim=-1)
        enc_out, hidden = self.encoder(enc_in, hidden)
        zhat_seq = self.z_head(enc_out)

        pol_in = torch.cat([xhat_rel_seq, zhat_seq], dim=-1)
        h = self.policy(pol_in)
        u_raw = self.mu_raw(h)
        a_env = torch.tanh(u_raw) * self.act_scale
        return a_env, zhat_seq, hidden


# =========================================================
# Episode container + chunking
# =========================================================

@dataclass
class DAggerEpisode:
    xhat_rel: torch.Tensor   # (T, xhat_dim)
    sigma: torch.Tensor      # (T, sigma_dim)
    u_prev: torch.Tensor     # (T, act_dim)
    u_teacher: torch.Tensor  # (T, act_dim)    teacher env-action labels
    z_teacher: torch.Tensor  # (T, latent_dim) privileged latent labels


def _iter_tbptt_chunks(ep: DAggerEpisode, chunk_len: int):
    T = ep.xhat_rel.shape[0]
    for t0 in range(0, T, chunk_len):
        t1 = min(T, t0 + chunk_len)
        yield (
            ep.xhat_rel[t0:t1],
            ep.sigma[t0:t1],
            ep.u_prev[t0:t1],
            ep.u_teacher[t0:t1],
            ep.z_teacher[t0:t1],
        )


@dataclass
class RecurrentDistillEpisode:
    xhat_rel: torch.Tensor   # (T, xhat_dim)
    sigma: torch.Tensor      # (T, sigma_dim)
    u_prev: torch.Tensor     # (T, act_dim)
    u_teacher: torch.Tensor  # (T, act_dim) teacher env-action labels
    rel_true: torch.Tensor   # (T, rel_dim) true relative state
    v_teacher: torch.Tensor  # (T, 1) teacher value labels


def _iter_recurrent_chunks(ep: RecurrentDistillEpisode, chunk_len: int):
    T = ep.xhat_rel.shape[0]
    for t0 in range(0, T, chunk_len):
        t1 = min(T, t0 + chunk_len)
        yield (
            ep.xhat_rel[t0:t1],
            ep.sigma[t0:t1],
            ep.u_prev[t0:t1],
            ep.u_teacher[t0:t1],
            ep.rel_true[t0:t1],
            ep.v_teacher[t0:t1],
        )


def _future_window_tensor(rel_seq: torch.Tensor, lookahead_H: int) -> torch.Tensor:
    """
    Build per-timestep future windows from a realized relative-state trajectory.
    Output shape: (T, H+1, rel_dim), padded with the terminal state.
    """
    T, rel_dim = rel_seq.shape
    future = torch.empty((T, lookahead_H + 1, rel_dim), dtype=rel_seq.dtype, device=rel_seq.device)
    last = rel_seq[-1]

    for t in range(T):
        max_len = min(lookahead_H + 1, T - t)
        future[t, :max_len] = rel_seq[t : t + max_len]
        if max_len < (lookahead_H + 1):
            future[t, max_len:] = last

    return future


# =========================================================
# Checkpoint helpers
# =========================================================

def _extract_state_dict(payload: Any, allowed_keys: List[str]) -> Optional[Dict[str, torch.Tensor]]:
    if not isinstance(payload, dict):
        return None

    # direct/raw state_dict
    if all(isinstance(k, str) for k in payload.keys()) and any(isinstance(v, torch.Tensor) for v in payload.values()):
        return payload

    for k in allowed_keys:
        if k in payload and isinstance(payload[k], dict):
            return payload[k]

    return None


def _load_teacher_bundle(
    ckpt_path: str,
    obs_dim: int,
    act_dim: int,
    cfg: Dict[str, Any],
    device: torch.device,
    lookahead_H: int,
    rel_dim: int,
    latent_dim: int,
):
    """
    Expects a richer checkpoint than your current raw ActorCriticDiff .pt.

    Required checkpoint payload keys:
      - one policy state dict under one of:
          ["policy_state_dict", "state_dict", "model", "net", "def_net", "actor_critic"]
      - one privileged intent encoder state dict under one of:
          ["intent_encoder_state_dict", "teacher_intent_encoder_state_dict", "privileged_encoder_state_dict"]

    If the intent encoder is missing, we can optionally fall back to handcrafted
    simulation-derived intent labels (cfg["allow_sim_intent_fallback"]=True).
    """
    payload = _load_checkpoint_payload(ckpt_path, map_location=device, label="teacher")

    policy_sd = _extract_state_dict(
        payload,
        ["policy_state_dict", "state_dict", "model", "net", "def_net", "actor_critic"],
    )
    if policy_sd is None:
        raise RuntimeError(
            f"Could not find teacher policy weights in {ckpt_path}."
        )

    intent_sd = None
    if isinstance(payload, dict):
        for k in [
            "intent_encoder_state_dict",
            "teacher_intent_encoder_state_dict",
            "privileged_encoder_state_dict",
        ]:
            if k in payload and isinstance(payload[k], dict):
                intent_sd = payload[k]
                break

    allow_sim_intent_fallback = bool(cfg.get("allow_sim_intent_fallback", True))

    if intent_sd is None and not allow_sim_intent_fallback:
        raise RuntimeError(
            "This checkpoint does not contain a privileged teacher intent encoder. "
            "Either save E* alongside the teacher policy checkpoint or set "
            "cfg['allow_sim_intent_fallback']=True to use simulation-derived intent labels."
        )

    teacher_policy = _actor_critic_cls(cfg)(obs_dim, act_dim, cfg).to(device)
    teacher_policy.load_state_dict(policy_sd, strict=False)
    teacher_policy.eval()
    for p in teacher_policy.parameters():
        p.requires_grad_(False)

    teacher_intent = None
    if intent_sd is not None:
        traj_dim = (lookahead_H + 1) * rel_dim
        teacher_intent = PrivilegedFutureIntentEncoder(traj_dim=traj_dim, latent_dim=latent_dim).to(device)
        teacher_intent.load_state_dict(intent_sd, strict=True)
        teacher_intent.eval()
        for p in teacher_intent.parameters():
            p.requires_grad_(False)

    return teacher_policy, teacher_intent


def _sim_intent_from_future_rel_traj(future_rel: np.ndarray, latent_dim: int) -> np.ndarray:
    """
    Handcrafted intent fallback from a simulated future relative trajectory.
    This is not paper-faithful E*, but it uses the same privileged rollout signal.
    """
    rel = np.asarray(future_rel, dtype=np.float32)
    pos = rel[:, : rel.shape[1] // 2]
    vel = rel[:, rel.shape[1] // 2 :]
    dist = np.linalg.norm(pos, axis=1)

    feats = [
        pos[0],
        vel[0],
        pos[-1],
        vel[-1],
        pos[-1] - pos[0],
        np.array([np.min(dist), np.argmin(dist) / max(1, len(dist) - 1)], dtype=np.float32),
        np.array([dist[-1] - dist[0]], dtype=np.float32),
    ]
    flat = np.concatenate([f.reshape(-1) for f in feats]).astype(np.float32)

    z = np.zeros((latent_dim,), dtype=np.float32)
    n = min(latent_dim, flat.shape[0])
    z[:n] = flat[:n]
    return z


def _load_policy_if_needed(
    ckpt_path: Optional[str],
    obs_dim: int,
    act_dim: int,
    cfg: Dict[str, Any],
    device: torch.device,
    missing_msg: str,
):
    if ckpt_path is None:
        raise RuntimeError(missing_msg)

    payload = _load_checkpoint_payload(ckpt_path, map_location=device, label="opponent")
    sd = _extract_state_dict(
        payload,
        ["policy_state_dict", "state_dict", "model", "net", "def_net", "att_net", "actor_critic"],
    )
    if sd is None:
        if isinstance(payload, dict) and any(k.startswith("att_net.") for k in payload.keys()):
            sd = {k[len("att_net."):]: v for k, v in payload.items() if k.startswith("att_net.")}
        elif isinstance(payload, dict) and any(k.startswith("def_net.") for k in payload.keys()):
            sd = {k[len("def_net."):]: v for k, v in payload.items() if k.startswith("def_net.")}
        else:
            raise RuntimeError(f"Could not parse policy checkpoint: {ckpt_path}")

    policy = _actor_critic_cls(cfg)(obs_dim, act_dim, cfg).to(device)
    policy.load_state_dict(sd, strict=False)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy


# =========================================================
# Env feature builders
# =========================================================


def _full_obs_from_env(env) -> np.ndarray:
    """
    True full-state teacher observation with the same layout as Env._obs(),
    but built from the underlying state rather than the UKF estimate.
    """
    p1, v1, pA_list, vA_list = env._unpack(env.state)
    c = env.center
    pos_scale = 1.0 / max(float(env.radius), 1e-9) if bool(getattr(env, "normalize_pos_obs", False)) else 1.0
    vel_scale = float(getattr(env, "_vel_obs_scale", 1.0))

    parts = [(p1 - c) * pos_scale]
    for pA in pA_list:
        parts.append((pA - c) * pos_scale)
    for pA in pA_list:
        parts.append((pA - p1) * pos_scale)
    parts.append(v1 * vel_scale)
    for vA in vA_list:
        parts.append(vA * vel_scale)

    if getattr(env, "use_fuel", False):
        fdef = (env.m_def - env.mdry_def) / (env.m0_def - env.mdry_def + 1e-9)
        fatt = (env.m_att[0] - env.mdry_att) / (env.m0_att - env.mdry_att + 1e-9)
        parts.append(np.array([np.clip(fdef, 0.0, 1.0)], dtype=np.float32))
        parts.append(np.array([np.clip(fatt, 0.0, 1.0)], dtype=np.float32))

    return np.concatenate(parts).astype(np.float32)


def _true_relative_state_from_env(
    env,
    *,
    distill_role: str = "def",
    attacker_idx: int = 0,
) -> np.ndarray:
    """
    Relative physical state used for privileged future-trajectory labels.
    For multi-attacker settings we concatenate [rel_k, dv_k] for all attackers.
    For attacker distillation, the observation is permuted so attacker idx occupies
    slot 0, matching the shared attacker policy convention.
    """
    D = int(env.D)
    Na = int(getattr(env, "num_attackers", 1))
    full_obs = _full_obs_from_env(env)
    if distill_role == "att":
        full_obs = _permute_obs_np(full_obs, D, Na, attacker_idx)
    return _relative_state_from_obs(full_obs, D, Na)


def _kf_student_features_from_env(
    env,
    u_prev: np.ndarray,
    *,
    distill_role: str = "def",
    attacker_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Student-side input at one timestep:
      xhat_rel_t, Sigma_t, u_{t-1}

    For num_attackers>1 the env does not provide a single-agent estimator, so we fall back to the
    current observation with zero covariance features.
    """
    if distill_role == "att" and hasattr(env, "_obs_att"):
        obs = env._obs_att().astype(np.float32)
    else:
        obs = env._obs().astype(np.float32)
    D = int(env.D)
    Na = int(getattr(env, "num_attackers", 1))

    if distill_role == "att" and Na > 1:
        obs = _permute_obs_np(obs, D, Na, attacker_idx)

    xhat_rel = _relative_state_from_obs(obs, D, Na)
    sigma_dim = xhat_rel.size * (xhat_rel.size + 1) // 2

    if getattr(env, "use_kf", False) and getattr(env, "ukf", None) is not None and Na == 1:
        P = np.asarray(env.ukf.P, dtype=np.float32)
        P_rel = P[: 2 * D, : 2 * D]
        iu = np.triu_indices(2 * D)
        sigma_feat = P_rel[iu].astype(np.float32)
    else:
        sigma_feat = np.zeros((sigma_dim,), dtype=np.float32)

    u_prev = np.asarray(u_prev, dtype=np.float32).reshape(-1)
    return xhat_rel, sigma_feat, u_prev


# =========================================================
# Opponent + teacher rollout helpers
# =========================================================


def _policy_action_from_obs(
    policy,
    obs: np.ndarray,
    device: torch.device,
    act_scale: float,
    who: str,
) -> np.ndarray:
    o = torch.as_tensor(np.asarray(obs, dtype=np.float32)[None, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        dist = policy.dist(o, who=who)
        a_env = torch.tanh(dist.mean) * act_scale
    return a_env[0].detach().cpu().numpy().astype(np.float32)


def _teacher_action_env(
    teacher_policy,
    full_obs: np.ndarray,
    device: torch.device,
    act_scale: float,
    distill_role: str,
    *,
    D: int,
    Na: int,
    attacker_idx: int = 0,
) -> np.ndarray:
    obs_policy = np.asarray(full_obs, dtype=np.float32)
    who = "def"
    if distill_role == "att":
        obs_policy = _permute_obs_np(obs_policy, D, Na, attacker_idx)
        who = "att"
    return _policy_action_from_obs(teacher_policy, obs_policy, device, act_scale, who=who)


def _teacher_attacker_team_actions_env(
    teacher_policy,
    full_obs: np.ndarray,
    device: torch.device,
    act_scale: float,
    *,
    D: int,
    Na: int,
) -> np.ndarray:
    acts = [
        _policy_action_from_obs(
            teacher_policy,
            _permute_obs_np(full_obs, D, Na, k),
            device,
            act_scale,
            who="att",
        )
        for k in range(Na)
    ]
    if Na == 1:
        return acts[0]
    return np.stack(acts, axis=0).astype(np.float32)


def _opponent_action_env(
    env,
    distill_role: str,
    attacker_mode: str,
    rule_ctrl,
    opponent_policy,
    device: torch.device,
    act_scale: float,
    *,
    D: int,
    Na: int,
) -> np.ndarray:
    if distill_role == "def" and attacker_mode == "rule":
        if rule_ctrl is None:
            raise RuntimeError("Rule-based attacker opponent requested but rule controller is not initialized.")
        p1, v1, pA_list, vA_list = env._unpack(env.state)
        acts = [rule_ctrl.act(p1, v1, pA_list[k], vA_list[k]).astype(np.float32) for k in range(Na)]
        if Na == 1:
            return acts[0]
        return np.stack(acts, axis=0).astype(np.float32)

    if opponent_policy is None:
        raise RuntimeError("Frozen opponent policy is required for distillation in this configuration.")

    full_obs = _full_obs_from_env(env)
    if distill_role == "def":
        acts = [
            _policy_action_from_obs(
                opponent_policy,
                _permute_obs_np(full_obs, D, Na, k),
                device,
                act_scale,
                who="att",
            )
            for k in range(Na)
        ]
        if Na == 1:
            return acts[0]
        return np.stack(acts, axis=0).astype(np.float32)

    return _policy_action_from_obs(opponent_policy, full_obs, device, act_scale, who="def")


# =========================================================
# Paper-style recurrent distillation path
# =========================================================


def _future_rel_traj_from_env(
    env,
    *,
    teacher_policy,
    distill_role: str,
    attacker_mode: str,
    rule_ctrl,
    opponent_policy,
    device: torch.device,
    act_scale: float,
    lookahead_H: int,
    reward_mode_for_step: str,
    D: int,
    Na: int,
    attacker_idx: int = 0,
) -> np.ndarray:
    try:
        env_roll = copy.deepcopy(env)
    except Exception:
        base = _true_relative_state_from_env(env, distill_role=distill_role, attacker_idx=attacker_idx)
        return np.repeat(base[None, :], lookahead_H + 1, axis=0).astype(np.float32)

    rel_hist = [_true_relative_state_from_env(env_roll, distill_role=distill_role, attacker_idx=attacker_idx)]
    for _ in range(lookahead_H):
        full_obs_np = _full_obs_from_env(env_roll)
        if distill_role == "def":
            a1_env = _teacher_action_env(
                teacher_policy=teacher_policy,
                full_obs=full_obs_np,
                device=device,
                act_scale=act_scale,
                distill_role=distill_role,
                D=D,
                Na=Na,
            )
            aA_env = _opponent_action_env(
                env=env_roll,
                distill_role=distill_role,
                attacker_mode=attacker_mode,
                rule_ctrl=rule_ctrl,
                opponent_policy=opponent_policy,
                device=device,
                act_scale=act_scale,
                D=D,
                Na=Na,
            )
        else:
            a1_env = _opponent_action_env(
                env=env_roll,
                distill_role=distill_role,
                attacker_mode=attacker_mode,
                rule_ctrl=rule_ctrl,
                opponent_policy=opponent_policy,
                device=device,
                act_scale=act_scale,
                D=D,
                Na=Na,
            )
            aA_env = _teacher_attacker_team_actions_env(
                teacher_policy=teacher_policy,
                full_obs=full_obs_np,
                device=device,
                act_scale=act_scale,
                D=D,
                Na=Na,
            )

        _, _, _, done, _ = env_roll.step(
            a1_env=a1_env,
            aA_env=aA_env,
            reward_mode=reward_mode_for_step,
        )
        rel_hist.append(_true_relative_state_from_env(env_roll, distill_role=distill_role, attacker_idx=attacker_idx))
        if done:
            break

    rel = np.stack(rel_hist, axis=0).astype(np.float32)
    if rel.shape[0] < (lookahead_H + 1):
        pad = np.repeat(rel[-1:, :], (lookahead_H + 1) - rel.shape[0], axis=0)
        rel = np.concatenate([rel, pad], axis=0)
    return rel


def _teacher_intent_label(
    *,
    teacher_intent,
    future_rel_np: np.ndarray,
    latent_dim: int,
    device: torch.device,
) -> np.ndarray:
    if teacher_intent is None:
        return _sim_intent_from_future_rel_traj(future_rel_np, latent_dim=latent_dim)

    future_rel_t = torch.as_tensor(future_rel_np[None, :, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        z = teacher_intent(future_rel_t)[0]
    return z.detach().cpu().numpy().astype(np.float32)

def distill_from_teacher_paper_recurrent(
    cfg: Dict[str, Any],
    teacher_ckpt_path: str,
    out_path: str = "ppo_def_kf_distilled.pt",
):

    cfg = copy.deepcopy(cfg)
    Na = _num_attackers(cfg)
    cfg["use_kf"] = (Na == 1)
    estimator_label = str(cfg.get("estimator_kind", "ukf")).upper()
    if Na > 1:
        print(
            f"[distill] num_attackers>1: {estimator_label} is unavailable, so distillation uses direct env observations "
            "with zero covariance features."
        )

    if ("dyn" not in cfg) or (cfg["dyn"].get("Ad") is None) or (cfg["dyn"].get("Bd") is None):
        _build_dyn_for_cfg(cfg)

    set_seed(int(cfg["seed"]))

    device = torch.device(cfg["device"])
    act_scale = float(cfg["umax"])
    D = int(cfg["D"])
    act_dim = D
    xhat_dim = 2 * Na * D
    sigma_dim = xhat_dim * (xhat_dim + 1) // 2
    obs_dim = _obs_dim_from_cfg(cfg)
    distill_role = str(cfg.get("train_role", "def")).lower()
    if distill_role not in {"def", "att"}:
        raise ValueError(f"Unsupported train_role for distillation: {distill_role!r}")


    episodes_per_iter = int(cfg.get("episodes_per_iter", 8))
    max_steps = int(cfg.get("max_steps", cfg.get("T", 300)))
    iters = int(cfg.get("iters", 100))
    lookahead_H = int(cfg.get("lookahead_H", 15))
    tbptt_chunk_len = int(cfg.get("distill_paper_tbptt_chunk_len", cfg.get("tbptt_chunk_len", 40)))
    lambda_intent = float(cfg.get("distill_paper_lambda_intent", cfg.get("lambda_intent", 1.0)))
    dagger_beta_start = float(cfg.get("dagger_beta_start", 1.0))
    dagger_beta_end = float(cfg.get("dagger_beta_end", 0.0))
    dagger_decay_iters = int(cfg.get("dagger_decay_iters", 50))
    reward_mode_for_step = str(cfg.get("reward_mode_for_step", "both"))
    max_dataset_episodes = cfg.get("max_dataset_episodes", 512)
    if max_dataset_episodes is not None:
        max_dataset_episodes = int(max_dataset_episodes)
    train_episode_cap = cfg.get("distill_paper_train_episodes_per_iter", 64)
    if train_episode_cap is not None:
        train_episode_cap = int(train_episode_cap)
        if train_episode_cap <= 0:
            train_episode_cap = None
    grad_clip_norm = cfg.get("distill_paper_grad_clip_norm", cfg.get("max_grad_norm", 1.0))
    if grad_clip_norm is not None:
        grad_clip_norm = float(grad_clip_norm)

    action_loss = str(cfg.get("distill_paper_action_loss", "mse")).lower()
    intent_loss = str(cfg.get("distill_paper_intent_loss", "mse")).lower()
    latent_dim = int(cfg.get("distill_latent_dim", 8))
    lstm_hidden = int(cfg.get("distill_paper_student_lstm_hidden", 256))
    student_lr = float(cfg.get("distill_lr", cfg.get("policy_lr", 3e-4)))
    log_every = int(cfg.get("distill_paper_log_every", cfg.get("log_every", 10)))
    method_label = str(cfg.get("_distill_save_method", "paper_recurrent")).lower()
    cfg["allow_sim_intent_fallback"] = bool(
        cfg.get("distill_paper_allow_sim_intent_fallback", cfg.get("allow_sim_intent_fallback", True))
    )

    teacher_policy, teacher_intent = _load_teacher_bundle(
        ckpt_path=teacher_ckpt_path,
        obs_dim=obs_dim,
        act_dim=act_dim,
        cfg=cfg,
        device=device,
        lookahead_H=lookahead_H,
        rel_dim=xhat_dim,
        latent_dim=latent_dim,
    )

    attacker_mode = str(cfg.get("attacker_mode", "rule")).lower()
    rule_ctrl = AttackerRuleController(cfg) if (distill_role == "def" and attacker_mode == "rule") else None
    if distill_role == "def" and attacker_mode == "rule":
        opponent_policy = None
    elif distill_role == "def":
        opponent_policy = _load_policy_if_needed(
            cfg.get("att_ckpt_path")
            or cfg.get("att_ckpt")
            or cfg.get("att_policy_path")
            or cfg.get("attacker_ckpt_path"),
            obs_dim,
            act_dim,
            cfg,
            device,
            "Defender distillation with attacker_mode='rl' requires an attacker checkpoint path.",
        )
    else:
        opponent_policy = _load_policy_if_needed(
            cfg.get("def_ckpt_path")
            or cfg.get("def_ckpt")
            or cfg.get("def_policy_path")
            or cfg.get("defender_ckpt_path"),
            obs_dim,
            act_dim,
            cfg,
            device,
            "Attacker distillation requires a defender checkpoint path for the frozen opponent.",
        )

    student = PartialObsStudentPolicy(
        xhat_dim=xhat_dim,
        sigma_dim=sigma_dim,
        act_dim=act_dim,
        latent_dim=latent_dim,
        lstm_hidden=lstm_hidden,
        act_scale=act_scale,
    ).to(device)
    optimizer = optim.Adam(student.parameters(), lr=student_lr)

    if action_loss == "mse":
        act_crit = nn.MSELoss()
    elif action_loss == "huber":
        act_crit = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported distill_paper_action_loss: {action_loss}")

    if intent_loss == "mse":
        z_crit = nn.MSELoss()
    elif intent_loss == "huber":
        z_crit = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported distill_paper_intent_loss: {intent_loss}")

    def beta_at(iter_idx: int) -> float:
        if dagger_decay_iters <= 0:
            return float(dagger_beta_end)
        alpha = min(1.0, max(0.0, iter_idx / float(dagger_decay_iters)))
        return float((1.0 - alpha) * dagger_beta_start + alpha * dagger_beta_end)

    dataset: List[DAggerEpisode] = []
    metrics = {
        "iter": [],
        "dagger_beta": [],
        "loss": [],
        "loss_action": [],
        "loss_intent": [],
        "dataset_episodes": [],
        "train_dataset_episodes": [],
    }

    print(f"=== Distillation: {method_label} recurrent {estimator_label} student with latent intent supervision ===")
    print(f"teacher_ckpt={teacher_ckpt_path}")
    print(f"distill_role={distill_role}")
    print(f"attacker_mode={attacker_mode}")
    print(
        f"student_arch=PartialObsStudentPolicy xhat_dim={xhat_dim} sigma_dim={sigma_dim} "
        f"act_dim={act_dim} latent_dim={latent_dim} lstm_hidden={lstm_hidden}"
    )
    print(
        f"lookahead_H={lookahead_H}  lambda_intent={lambda_intent}  "
        f"tbptt_chunk_len={tbptt_chunk_len}  train_episode_cap={train_episode_cap}"
    )

    for it in range(iters):
        beta = beta_at(it)
        new_episodes: List[DAggerEpisode] = []


        for _ in range(episodes_per_iter):
            env = _env_cls_for_cfg(cfg)(cfg)
            env.reset()

            role_indices = [0] if distill_role == "def" else list(range(Na))
            xhat_lists: Dict[int, List[np.ndarray]] = {idx: [] for idx in role_indices}
            sig_lists: Dict[int, List[np.ndarray]] = {idx: [] for idx in role_indices}
            u_prev_lists: Dict[int, List[np.ndarray]] = {idx: [] for idx in role_indices}
            u_teacher_lists: Dict[int, List[np.ndarray]] = {idx: [] for idx in role_indices}
            z_teacher_lists: Dict[int, List[np.ndarray]] = {idx: [] for idx in role_indices}
            u_prev_exec: Dict[int, np.ndarray] = {idx: np.zeros((act_dim,), dtype=np.float32) for idx in role_indices}
            hidden_states = {idx: student.init_hidden(batch_size=1, device=device) for idx in role_indices}

            for _t in range(max_steps):
                full_obs_np = _full_obs_from_env(env)

                if distill_role == "def":
                    idx = 0
                    xhat_rel_np, sigma_np, u_prev_feat_np = _kf_student_features_from_env(
                        env,
                        u_prev_exec[idx],
                        distill_role=distill_role,
                        attacker_idx=idx,
                    )
                    u_teacher_np = _teacher_action_env(
                        teacher_policy=teacher_policy,
                        full_obs=full_obs_np,
                        device=device,
                        act_scale=act_scale,
                        distill_role=distill_role,
                        D=D,
                        Na=Na,
                        attacker_idx=idx,
                    )
                    future_rel_np = _future_rel_traj_from_env(
                        env,
                        teacher_policy=teacher_policy,
                        distill_role=distill_role,
                        attacker_mode=attacker_mode,
                        rule_ctrl=rule_ctrl,
                        opponent_policy=opponent_policy,
                        device=device,
                        act_scale=act_scale,
                        lookahead_H=lookahead_H,
                        reward_mode_for_step=reward_mode_for_step,
                        D=D,
                        Na=Na,
                        attacker_idx=idx,
                    )
                    z_teacher_np = _teacher_intent_label(
                        teacher_intent=teacher_intent,
                        future_rel_np=future_rel_np,
                        latent_dim=latent_dim,
                        device=device,
                    )

                    xhat_rel_t = torch.as_tensor(xhat_rel_np, dtype=torch.float32, device=device).view(1, -1)
                    sigma_t = torch.as_tensor(sigma_np, dtype=torch.float32, device=device).view(1, -1)
                    u_prev_t = torch.as_tensor(u_prev_feat_np, dtype=torch.float32, device=device).view(1, -1)
                    with torch.no_grad():
                        u_pred_t, _z_hat_t, hidden_next = student.step(
                            xhat_rel_t,
                            sigma_t,
                            u_prev_t,
                            hidden_states[idx],
                        )
                    u_pred_np = u_pred_t[0].detach().cpu().numpy().astype(np.float32)
                    hidden_states[idx] = hidden_next

                    use_teacher = random.random() < beta
                    a_exec = u_teacher_np if use_teacher else u_pred_np
                    a_opp_exec = _opponent_action_env(
                        env=env,
                        distill_role=distill_role,
                        attacker_mode=attacker_mode,
                        rule_ctrl=rule_ctrl,
                        opponent_policy=opponent_policy,
                        device=device,
                        act_scale=act_scale,
                        D=D,
                        Na=Na,
                    )

                    _, _, _, done, _ = env.step(
                        a1_env=a_exec,
                        aA_env=a_opp_exec,
                        reward_mode=reward_mode_for_step,
                    )

                    xhat_lists[idx].append(np.asarray(xhat_rel_np, dtype=np.float32).reshape(-1))
                    sig_lists[idx].append(np.asarray(sigma_np, dtype=np.float32).reshape(-1))
                    u_prev_lists[idx].append(np.asarray(u_prev_feat_np, dtype=np.float32).reshape(-1))
                    u_teacher_lists[idx].append(np.asarray(u_teacher_np, dtype=np.float32).reshape(-1))
                    z_teacher_lists[idx].append(np.asarray(z_teacher_np, dtype=np.float32).reshape(-1))
                    u_prev_exec[idx] = np.asarray(a_exec, dtype=np.float32).reshape(-1)
                else:
                    a_def_exec = _opponent_action_env(
                        env=env,
                        distill_role=distill_role,
                        attacker_mode=attacker_mode,
                        rule_ctrl=rule_ctrl,
                        opponent_policy=opponent_policy,
                        device=device,
                        act_scale=act_scale,
                        D=D,
                        Na=Na,
                    )
                    a_att_exec = np.zeros((Na, act_dim), dtype=np.float32)

                    for idx in role_indices:
                        xhat_rel_np, sigma_np, u_prev_feat_np = _kf_student_features_from_env(
                            env,
                            u_prev_exec[idx],
                            distill_role=distill_role,
                            attacker_idx=idx,
                        )
                        u_teacher_np = _teacher_action_env(
                            teacher_policy=teacher_policy,
                            full_obs=full_obs_np,
                            device=device,
                            act_scale=act_scale,
                            distill_role=distill_role,
                            D=D,
                            Na=Na,
                            attacker_idx=idx,
                        )
                        future_rel_np = _future_rel_traj_from_env(
                            env,
                            teacher_policy=teacher_policy,
                            distill_role=distill_role,
                            attacker_mode=attacker_mode,
                            rule_ctrl=rule_ctrl,
                            opponent_policy=opponent_policy,
                            device=device,
                            act_scale=act_scale,
                            lookahead_H=lookahead_H,
                            reward_mode_for_step=reward_mode_for_step,
                            D=D,
                            Na=Na,
                            attacker_idx=idx,
                        )
                        z_teacher_np = _teacher_intent_label(
                            teacher_intent=teacher_intent,
                            future_rel_np=future_rel_np,
                            latent_dim=latent_dim,
                            device=device,
                        )

                        xhat_rel_t = torch.as_tensor(xhat_rel_np, dtype=torch.float32, device=device).view(1, -1)
                        sigma_t = torch.as_tensor(sigma_np, dtype=torch.float32, device=device).view(1, -1)
                        u_prev_t = torch.as_tensor(u_prev_feat_np, dtype=torch.float32, device=device).view(1, -1)
                        with torch.no_grad():
                            u_pred_t, _z_hat_t, hidden_next = student.step(
                                xhat_rel_t,
                                sigma_t,
                                u_prev_t,
                                hidden_states[idx],
                            )
                        u_pred_np = u_pred_t[0].detach().cpu().numpy().astype(np.float32)
                        hidden_states[idx] = hidden_next

                        use_teacher = random.random() < beta
                        a_exec = u_teacher_np if use_teacher else u_pred_np
                        a_att_exec[idx] = a_exec

                        xhat_lists[idx].append(np.asarray(xhat_rel_np, dtype=np.float32).reshape(-1))
                        sig_lists[idx].append(np.asarray(sigma_np, dtype=np.float32).reshape(-1))
                        u_prev_lists[idx].append(np.asarray(u_prev_feat_np, dtype=np.float32).reshape(-1))
                        u_teacher_lists[idx].append(np.asarray(u_teacher_np, dtype=np.float32).reshape(-1))
                        z_teacher_lists[idx].append(np.asarray(z_teacher_np, dtype=np.float32).reshape(-1))

                    step_att = a_att_exec[0] if Na == 1 else a_att_exec
                    _, _, _, done, _ = env.step(
                        a1_env=a_def_exec,
                        aA_env=step_att,
                        reward_mode=reward_mode_for_step,
                    )
                    for idx in role_indices:
                        u_prev_exec[idx] = np.asarray(a_att_exec[idx], dtype=np.float32).reshape(-1)

                if done:
                    break

            for idx in role_indices:
                if xhat_lists[idx]:
                    new_episodes.append(
                        DAggerEpisode(
                            xhat_rel=torch.as_tensor(np.stack(xhat_lists[idx]), dtype=torch.float32, device=device),
                            sigma=torch.as_tensor(np.stack(sig_lists[idx]), dtype=torch.float32, device=device),
                            u_prev=torch.as_tensor(np.stack(u_prev_lists[idx]), dtype=torch.float32, device=device),
                            u_teacher=torch.as_tensor(np.stack(u_teacher_lists[idx]), dtype=torch.float32, device=device),
                            z_teacher=torch.as_tensor(np.stack(z_teacher_lists[idx]), dtype=torch.float32, device=device),
                        )
                    )

        dataset.extend(new_episodes)
        if max_dataset_episodes is not None and len(dataset) > max_dataset_episodes:
            dataset = dataset[-max_dataset_episodes:]

        student.train()
        total_loss = 0.0
        total_act = 0.0
        total_z = 0.0
        n_chunks = 0

        if train_episode_cap is None or len(dataset) <= train_episode_cap:
            train_dataset = list(dataset)
        else:
            train_dataset = random.sample(dataset, train_episode_cap)

        optimizer.zero_grad(set_to_none=True)
        for ep in train_dataset:
            hidden = student.init_hidden(batch_size=1, device=device)
            for xhat_seq, sig_seq, uprev_seq, ustar_seq, zstar_seq in _iter_tbptt_chunks(ep, tbptt_chunk_len):
                if hidden is not None:
                    hidden = tuple(h.detach() for h in hidden)

                u_preds_b, z_hats_b, hidden = student.forward_chunk(
                    xhat_seq.unsqueeze(0),
                    sig_seq.unsqueeze(0),
                    uprev_seq.unsqueeze(0),
                    hidden,
                )
                u_preds_t = u_preds_b.squeeze(0)
                z_hats_t = z_hats_b.squeeze(0)

                loss_act = act_crit(u_preds_t, ustar_seq)
                loss_z = z_crit(z_hats_t, zstar_seq)
                loss = loss_act + lambda_intent * loss_z
                loss.backward()

                total_loss += float(loss.detach().cpu())
                total_act += float(loss_act.detach().cpu())
                total_z += float(loss_z.detach().cpu())
                n_chunks += 1

        if n_chunks > 0:
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip_norm)
            optimizer.step()

        if ((it % log_every) == 0) or (it == iters - 1):
            mean_loss = total_loss / max(1, n_chunks)
            mean_act = total_act / max(1, n_chunks)
            mean_z = total_z / max(1, n_chunks)
            train_dataset_eps = len(train_dataset)
            print(
                f"[distill {method_label} {it:04d}] beta={beta:.3f}  "
                f"loss={mean_loss:.3e}  action={mean_act:.3e}  intent={mean_z:.3e}  "
                f"dataset_episodes={len(dataset)}  train_eps={train_dataset_eps}"
            )
            metrics["iter"].append(float(it))
            metrics["dagger_beta"].append(float(beta))
            metrics["loss"].append(mean_loss)
            metrics["loss_action"].append(mean_act)
            metrics["loss_intent"].append(mean_z)
            metrics["dataset_episodes"].append(float(len(dataset)))
            metrics["train_dataset_episodes"].append(float(train_dataset_eps))

    if out_path is not None:
        out_path = str(out_path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "student_state_dict": student.state_dict(),
            "cfg": cfg,
            "metrics": metrics,
            "meta": {
                "distill_method": method_label,
                "xhat_dim": xhat_dim,
                "sigma_dim": sigma_dim,
                "act_dim": act_dim,
                "latent_dim": latent_dim,
                "lstm_hidden": lstm_hidden,
                "iters": iters,
                "episodes_per_iter": episodes_per_iter,
                "max_steps": max_steps,
                "lookahead_H": lookahead_H,
                "lambda_intent": lambda_intent,
                "dagger_beta_start": dagger_beta_start,
                "dagger_beta_end": dagger_beta_end,
                "dagger_decay_iters": dagger_decay_iters,
                "tbptt_chunk_len": tbptt_chunk_len,
                "distill_paper_train_episodes_per_iter": train_episode_cap,
                "grad_clip_norm": grad_clip_norm,
                "action_loss": action_loss,
                "intent_loss": intent_loss,
                "reward_mode_for_step": reward_mode_for_step,
                "max_dataset_episodes": max_dataset_episodes,
            },
        }
        torch.save(payload, out_path)
        print(f"[distill] saved {method_label} student checkpoint -> {out_path}")

    return student, metrics


# =========================================================
# Public distillation entry point
# =========================================================

def distill_from_teacher_modern(
    cfg: Dict[str, Any],
    teacher_ckpt_path: str,
    out_path: str = "ppo_def_kf_distilled.pt",
):
    """
    Paper-faithful modern distillation path.

    This follows the UC Berkeley student setup:
      - estimator relative-state mean + covariance inputs
      - previous executed action as part of the recurrent encoder input
      - 1-layer LSTM latent-intent encoder
      - teacher latent supervision from future relative trajectories
      - DAgger-style rollout aggregation

    """
    cfg_mod = copy.deepcopy(cfg)
    cfg_mod["use_kf"] = (_num_attackers(cfg_mod) == 1)
    cfg_mod["_distill_save_method"] = "modern"
    cfg_mod.setdefault("distill_paper_action_loss", "mse")
    cfg_mod.setdefault("distill_paper_intent_loss", "mse")
    cfg_mod.setdefault("distill_paper_student_lstm_hidden", 256)
    cfg_mod.setdefault(
        "distill_paper_log_every",
        int(cfg_mod.get("log_every", 10)),
    )
    cfg_mod.setdefault(
        "distill_paper_allow_sim_intent_fallback",
        bool(cfg_mod.get("allow_sim_intent_fallback", True)),
    )

    requested_collection_mode = str(cfg_mod.get("distill_collection_mode", "dagger")).lower()
    if requested_collection_mode != "dagger":
        print(
            "[distill] modern mode is paper-faithful DAgger; "
            f"overriding distill_collection_mode={requested_collection_mode!r} -> 'dagger'."
        )
    cfg_mod["distill_collection_mode"] = "dagger"

    estimator_label = str(cfg_mod.get("estimator_kind", "ukf")).upper()
    print(f"[distill] modern mode: paper-faithful recurrent {estimator_label} student with latent intent supervision.")
    return distill_from_teacher_paper_recurrent(
        cfg_mod,
        teacher_ckpt_path,
        out_path=out_path,
    )


def distill_from_teacher(
    cfg: Dict[str, Any],
    teacher_ckpt_path: str,
    out_path: str = "ppo_def_kf_distilled.pt",
):
    method = str(cfg.get("distill_method", "modern")).lower()
    if method in {"modern", "default", "paper_modern", "berkeley"}:
        return distill_from_teacher_modern(
            cfg,
            teacher_ckpt_path,
            out_path=out_path,
        )
    if method in {"paper_recurrent", "paper"}:
        return distill_from_teacher_paper_recurrent(
            cfg,
            teacher_ckpt_path,
            out_path=out_path,
        )
    raise ValueError(
        f"Unsupported distill_method={method!r}. Expected one of "
        "'modern' or 'paper_recurrent'."
    )
