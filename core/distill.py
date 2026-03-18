from __future__ import annotations

import copy
import math
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

from core.utils import logprob_squashed, set_seed, squash_action
from config_rl import build_dyn
from core.models import ActorCriticDiff
from core.controllers import AttackerRuleController
from core.env import Env



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
# - squash_action
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

    teacher_policy = ActorCriticDiff(obs_dim, act_dim, cfg).to(device)
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

    policy = ActorCriticDiff(obs_dim, act_dim, cfg).to(device)
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
    True full-state teacher observation:
      [p1-c, p2-c, (p2-p1), v1, v2, (optional fuel fracs)]
    """
    p1, v1, pA_list, vA_list = env._unpack(env.state)
    p2 = pA_list[0]
    v2 = vA_list[0]
    c = env.center

    parts = [
        p1 - c,
        p2 - c,
        p2 - p1,
        v1,
        v2,
    ]

    if getattr(env, "use_fuel", False):
        fdef = (env.m_def - env.mdry_def) / (env.m0_def - env.mdry_def + 1e-9)
        fatt = (env.m_att[0] - env.mdry_att) / (env.m0_att - env.mdry_att + 1e-9)
        parts.append(np.array([np.clip(fdef, 0.0, 1.0)], dtype=np.float32))
        parts.append(np.array([np.clip(fatt, 0.0, 1.0)], dtype=np.float32))

    return np.concatenate(parts).astype(np.float32)


def _true_relative_state_from_env(env) -> np.ndarray:
    """
    Relative physical state used for privileged future-trajectory labels.
    In your HCW setting we use [p_rel, v_rel].
    """
    p1, v1, pA_list, vA_list = env._unpack(env.state)
    p2 = pA_list[0]
    v2 = vA_list[0]
    return np.concatenate([p2 - p1, v2 - v1]).astype(np.float32)


def _ukf_student_features_from_env(env, u_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Student-side input at one timestep:
      xhat_rel_t, Sigma_t, u_{t-1}

    We derive xhat_rel_t from the env's current observation, which already uses UKF/KF estimates
    when use_ukf=True. Sigma_t comes from the estimator covariance if available.
    """
    obs = env._obs().astype(np.float32)
    D = int(env.D)

    rel = obs[2 * D : 3 * D]
    v1  = obs[3 * D : 4 * D]
    v2  = obs[4 * D : 5 * D]
    xhat_rel = np.concatenate([rel, v2 - v1]).astype(np.float32)

    if getattr(env, "use_ukf", False) and getattr(env, "ukf", None) is not None:
        P = np.asarray(env.ukf.P, dtype=np.float32)
        P_rel = P[: 2 * D, : 2 * D]
        iu = np.triu_indices(2 * D)
        sigma_feat = P_rel[iu].astype(np.float32)
    else:
        sigma_feat = np.zeros((2 * D) * (2 * D + 1) // 2, dtype=np.float32)

    u_prev = np.asarray(u_prev, dtype=np.float32).reshape(-1)
    return xhat_rel, sigma_feat, u_prev


# =========================================================
# Opponent + teacher rollout helpers
# =========================================================

def _teacher_action_env(
    teacher_policy,
    full_obs: np.ndarray,
    device: torch.device,
    act_scale: float,
    distill_role: str,
) -> np.ndarray:
    o = torch.as_tensor(full_obs[None, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        dist = teacher_policy.dist(o, who=distill_role)
        a_env = torch.tanh(dist.mean) * act_scale
    return a_env[0].detach().cpu().numpy().astype(np.float32)


def _opponent_action_env(
    env,
    distill_role: str,
    attacker_mode: str,
    rule_ctrl,
    opponent_policy,
    device: torch.device,
    act_scale: float,
) -> np.ndarray:
    if distill_role == "def" and attacker_mode == "rule":
        p1, v1, pA_list, vA_list = env._unpack(env.state)
        p2 = pA_list[0]
        v2 = vA_list[0]
        return rule_ctrl.act(p1, v1, p2, v2).astype(np.float32)

    full_obs = _full_obs_from_env(env)
    o = torch.as_tensor(full_obs[None, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        who = "att" if distill_role == "def" else "def"
        dist = opponent_policy.dist(o, who=who)
        a_env = torch.tanh(dist.mean) * act_scale
    return a_env[0].detach().cpu().numpy().astype(np.float32)


# =========================================================
# Replay buffer for modern distillation
# =========================================================

class _DistillReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        *,
        store_teacher_obs: bool,
        store_teacher_value: bool,
    ):
        self.capacity = int(capacity)
        self.obs_student = torch.empty((self.capacity, obs_dim), dtype=torch.float32, device="cpu")
        self.teacher_raw = torch.empty((self.capacity, act_dim), dtype=torch.float32, device="cpu")
        self.teacher_obs = (
            torch.empty((self.capacity, obs_dim), dtype=torch.float32, device="cpu")
            if store_teacher_obs else None
        )
        self.teacher_value = (
            torch.empty((self.capacity, 1), dtype=torch.float32, device="cpu")
            if store_teacher_value else None
        )
        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs_student: torch.Tensor,
        teacher_raw: torch.Tensor,
        *,
        teacher_obs: Optional[torch.Tensor] = None,
        teacher_value: Optional[torch.Tensor] = None,
    ) -> None:
        obs_student = obs_student.detach().to("cpu", dtype=torch.float32)
        teacher_raw = teacher_raw.detach().to("cpu", dtype=torch.float32)
        if teacher_obs is not None:
            teacher_obs = teacher_obs.detach().to("cpu", dtype=torch.float32)
        if teacher_value is not None:
            teacher_value = teacher_value.detach().to("cpu", dtype=torch.float32).reshape(-1, 1)

        n = int(obs_student.shape[0])
        if n <= 0:
            return

        if n >= self.capacity:
            obs_student = obs_student[-self.capacity :]
            teacher_raw = teacher_raw[-self.capacity :]
            if teacher_obs is not None:
                teacher_obs = teacher_obs[-self.capacity :]
            if teacher_value is not None:
                teacher_value = teacher_value[-self.capacity :]
            n = self.capacity

        first = min(self.capacity - self.ptr, n)
        second = n - first

        self.obs_student[self.ptr : self.ptr + first].copy_(obs_student[:first])
        self.teacher_raw[self.ptr : self.ptr + first].copy_(teacher_raw[:first])
        if self.teacher_obs is not None and teacher_obs is not None:
            self.teacher_obs[self.ptr : self.ptr + first].copy_(teacher_obs[:first])
        if self.teacher_value is not None and teacher_value is not None:
            self.teacher_value[self.ptr : self.ptr + first].copy_(teacher_value[:first])

        if second > 0:
            self.obs_student[:second].copy_(obs_student[first:])
            self.teacher_raw[:second].copy_(teacher_raw[first:])
            if self.teacher_obs is not None and teacher_obs is not None:
                self.teacher_obs[:second].copy_(teacher_obs[first:])
            if self.teacher_value is not None and teacher_value is not None:
                self.teacher_value[:second].copy_(teacher_value[first:])

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        if self.size <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        idx = torch.randint(0, self.size, (int(batch_size),), device="cpu")
        obs_student = self.obs_student[idx].to(device)
        teacher_raw = self.teacher_raw[idx].to(device)
        teacher_obs = self.teacher_obs[idx].to(device) if self.teacher_obs is not None else None
        teacher_value = self.teacher_value[idx].to(device) if self.teacher_value is not None else None
        return obs_student, teacher_raw, teacher_obs, teacher_value


# =========================================================
# Legacy/original distillation path
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
) -> np.ndarray:
    try:
        env_roll = copy.deepcopy(env)
    except Exception:
        base = _true_relative_state_from_env(env)
        return np.repeat(base[None, :], lookahead_H + 1, axis=0).astype(np.float32)

    rel_hist = [_true_relative_state_from_env(env_roll)]
    for _ in range(lookahead_H):
        full_obs_np = _full_obs_from_env(env_roll)
        a_role = _teacher_action_env(
            teacher_policy=teacher_policy,
            full_obs=full_obs_np,
            device=device,
            act_scale=act_scale,
            distill_role=distill_role,
        )
        a_opp = _opponent_action_env(
            env=env_roll,
            distill_role=distill_role,
            attacker_mode=attacker_mode,
            rule_ctrl=rule_ctrl,
            opponent_policy=opponent_policy,
            device=device,
            act_scale=act_scale,
        )
        if distill_role == "def":
            a1_env, aA_env = a_role, a_opp
        else:
            a1_env, aA_env = a_opp, a_role

        _, _, _, done, _ = env_roll.step(
            a1_env=a1_env,
            aA_env=aA_env,
            reward_mode=reward_mode_for_step,
        )
        rel_hist.append(_true_relative_state_from_env(env_roll))
        if done:
            break

    rel = np.stack(rel_hist, axis=0).astype(np.float32)
    if rel.shape[0] < (lookahead_H + 1):
        pad = np.repeat(rel[-1: , :], (lookahead_H + 1) - rel.shape[0], axis=0)
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


def distill_from_teacher_original(
    cfg: Dict[str, Any],
    teacher_ckpt_path: str,
    out_path: str = "ppo_def_ukf_distilled.pt",
):
    cfg = copy.deepcopy(cfg)
    cfg["use_ukf"] = True

    if ("dyn" not in cfg) or (cfg["dyn"].get("Ad") is None) or (cfg["dyn"].get("Bd") is None):
        build_dyn(cfg)

    set_seed(int(cfg["seed"]))

    device = torch.device(cfg["device"])
    act_scale = float(cfg["umax"])
    D = int(cfg["D"])
    act_dim = D
    xhat_dim = 2 * D
    sigma_dim = (2 * D) * (2 * D + 1) // 2
    obs_dim = 5 * D + (2 if cfg.get("fuel", {}).get("enable", False) else 0)
    distill_role = str(cfg.get("train_role", "def")).lower()
    if distill_role not in {"def", "att"}:
        raise ValueError(f"Unsupported train_role for distillation: {distill_role!r}")
    if distill_role == "att":
        print(
            "[distill] warning: attacker distillation still uses the shared defender-centric "
            "observation layout; no attacker-specific UKF belief model is implemented."
        )

    episodes_per_iter = int(cfg.get("episodes_per_iter", 8))
    max_steps = int(cfg.get("max_steps", cfg.get("T", 300)))
    iters = int(cfg.get("iters", 100))
    lookahead_H = int(cfg.get("lookahead_H", 15))
    tbptt_chunk_len = int(cfg.get("tbptt_chunk_len", 40))
    lambda_intent = float(cfg.get("lambda_intent", 1.0))
    dagger_beta_start = float(cfg.get("dagger_beta_start", 1.0))
    dagger_beta_end = float(cfg.get("dagger_beta_end", 0.0))
    dagger_decay_iters = int(cfg.get("dagger_decay_iters", 50))
    reward_mode_for_step = str(cfg.get("reward_mode_for_step", "both"))
    max_dataset_episodes = cfg.get("max_dataset_episodes", 512)
    if max_dataset_episodes is not None:
        max_dataset_episodes = int(max_dataset_episodes)
    grad_clip_norm = cfg.get("distill_original_grad_clip_norm", cfg.get("max_grad_norm", 1.0))
    if grad_clip_norm is not None:
        grad_clip_norm = float(grad_clip_norm)

    action_loss = str(cfg.get("distill_original_action_loss", "mse")).lower()
    intent_loss = str(cfg.get("distill_original_intent_loss", "mse")).lower()
    latent_dim = int(cfg.get("distill_latent_dim", 8))
    lstm_hidden = int(cfg.get("distill_original_student_lstm_hidden", 256))
    student_lr = float(cfg.get("distill_lr", cfg.get("policy_lr", 3e-4)))
    log_every = int(cfg.get("distill_original_log_every", cfg.get("log_every", 10)))
    cfg["allow_sim_intent_fallback"] = bool(
        cfg.get(
            "distill_original_allow_sim_intent_fallback",
            cfg.get("allow_sim_intent_fallback", True),
        )
    )

    teacher_policy, teacher_intent = _load_teacher_bundle(
        ckpt_path=teacher_ckpt_path,
        obs_dim=obs_dim,
        act_dim=act_dim,
        cfg=cfg,
        device=device,
        lookahead_H=lookahead_H,
        rel_dim=2 * D,
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
        raise ValueError(f"Unsupported distill_original_action_loss: {action_loss}")

    if intent_loss == "mse":
        z_crit = nn.MSELoss()
    elif intent_loss == "huber":
        z_crit = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported distill_original_intent_loss: {intent_loss}")

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
    }

    print("=== Distillation: original recurrent student path ===")
    print(f"teacher_ckpt={teacher_ckpt_path}")
    print(f"distill_role={distill_role}")
    print(f"attacker_mode={attacker_mode}")
    print(f"student_arch=PartialObsStudentPolicy xhat_dim={xhat_dim} sigma_dim={sigma_dim} act_dim={act_dim}")

    for it in range(iters):
        beta = beta_at(it)
        new_episodes: List[DAggerEpisode] = []

        for _ in range(episodes_per_iter):
            env = Env(cfg)
            env.reset()

            xhat_list: List[np.ndarray] = []
            sig_list: List[np.ndarray] = []
            u_prev_list: List[np.ndarray] = []
            u_teacher_list: List[np.ndarray] = []
            z_teacher_list: List[np.ndarray] = []

            u_prev_np = np.zeros((act_dim,), dtype=np.float32)
            hidden = student.init_hidden(batch_size=1, device=device)

            for _t in range(max_steps):
                xhat_rel_np, sigma_np, u_prev_feat_np = _ukf_student_features_from_env(env, u_prev_np)
                full_obs_np = _full_obs_from_env(env)
                u_teacher_np = _teacher_action_env(
                    teacher_policy=teacher_policy,
                    full_obs=full_obs_np,
                    device=device,
                    act_scale=act_scale,
                    distill_role=distill_role,
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
                    u_pred_t, _z_hat_t, hidden_next = student.step(xhat_rel_t, sigma_t, u_prev_t, hidden)
                u_pred_np = u_pred_t[0].detach().cpu().numpy().astype(np.float32)
                hidden = hidden_next

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
                )
                if distill_role == "def":
                    a1_env, aA_env = a_exec, a_opp_exec
                else:
                    a1_env, aA_env = a_opp_exec, a_exec

                _, _, _, done, _ = env.step(
                    a1_env=a1_env,
                    aA_env=aA_env,
                    reward_mode=reward_mode_for_step,
                )

                xhat_list.append(np.asarray(xhat_rel_np, dtype=np.float32).reshape(-1))
                sig_list.append(np.asarray(sigma_np, dtype=np.float32).reshape(-1))
                u_prev_list.append(np.asarray(u_prev_feat_np, dtype=np.float32).reshape(-1))
                u_teacher_list.append(np.asarray(u_teacher_np, dtype=np.float32).reshape(-1))
                z_teacher_list.append(np.asarray(z_teacher_np, dtype=np.float32).reshape(-1))

                u_prev_np = np.asarray(a_exec, dtype=np.float32).reshape(-1)

                if done:
                    break

            if xhat_list:
                new_episodes.append(
                    DAggerEpisode(
                        xhat_rel=torch.as_tensor(np.stack(xhat_list), dtype=torch.float32, device=device),
                        sigma=torch.as_tensor(np.stack(sig_list), dtype=torch.float32, device=device),
                        u_prev=torch.as_tensor(np.stack(u_prev_list), dtype=torch.float32, device=device),
                        u_teacher=torch.as_tensor(np.stack(u_teacher_list), dtype=torch.float32, device=device),
                        z_teacher=torch.as_tensor(np.stack(z_teacher_list), dtype=torch.float32, device=device),
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

        optimizer.zero_grad(set_to_none=True)
        for ep in dataset:
            hidden = student.init_hidden(batch_size=1, device=device)
            for xhat_seq, sig_seq, uprev_seq, ustar_seq, zstar_seq in _iter_tbptt_chunks(ep, tbptt_chunk_len):
                if hidden is not None:
                    hidden = tuple(h.detach() for h in hidden)

                u_preds = []
                z_hats = []
                for k in range(xhat_seq.shape[0]):
                    u_pred, z_hat, hidden = student.step(
                        xhat_seq[k].view(1, -1),
                        sig_seq[k].view(1, -1),
                        uprev_seq[k].view(1, -1),
                        hidden,
                    )
                    u_preds.append(u_pred)
                    z_hats.append(z_hat)

                u_preds_t = torch.cat(u_preds, dim=0)
                z_hats_t = torch.cat(z_hats, dim=0)

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
            print(
                f"[distill original {it:04d}] beta={beta:.3f}  "
                f"loss={mean_loss:.3e}  action={mean_act:.3e}  intent={mean_z:.3e}  "
                f"dataset_episodes={len(dataset)}"
            )
            metrics["iter"].append(float(it))
            metrics["dagger_beta"].append(float(beta))
            metrics["loss"].append(mean_loss)
            metrics["loss_action"].append(mean_act)
            metrics["loss_intent"].append(mean_z)
            metrics["dataset_episodes"].append(float(len(dataset)))

    if out_path is not None:
        out_path = str(out_path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "student_state_dict": student.state_dict(),
            "cfg": cfg,
            "metrics": metrics,
            "meta": {
                "distill_method": "original",
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
                "grad_clip_norm": grad_clip_norm,
                "action_loss": action_loss,
                "intent_loss": intent_loss,
                "reward_mode_for_step": reward_mode_for_step,
                "max_dataset_episodes": max_dataset_episodes,
            },
        }
        torch.save(payload, out_path)
        print(f"[distill] saved original student checkpoint -> {out_path}")

    return student, metrics


# =========================================================
# Main rewrite
# =========================================================

def distill_from_teacher_modern(
    cfg: Dict[str, Any],
    teacher_ckpt_path: str,
    out_path: str = "ppo_def_ukf_distilled.pt",
):
    """
    Distill a full-state teacher into an ActorCriticDiff student that consumes the
    same observation interface as PPO, but under cfg["use_ukf"]=True.

    The student architecture intentionally matches fully-observable PPO so that:
      - checkpoints are plain ActorCriticDiff state_dicts
      - inference stays on the same code path as teacher checkpoints
      - defender/attacker evaluation remains apples-to-apples

    Distillation target:
      - teacher mean action from the true full-state observation

    Student input:
      - env._obs() under use_ukf=True, i.e. the observation stream already produced
        by the environment / filter stack
    """
    cfg = copy.deepcopy(cfg)

    # Student observes the partial-observation stream produced by Env._obs().
    cfg["use_ukf"] = True

    # Ensure dynamics exist
    if ("dyn" not in cfg) or (cfg["dyn"].get("Ad") is None) or (cfg["dyn"].get("Bd") is None):
        build_dyn(cfg)

    set_seed(int(cfg["seed"]))

    device = torch.device(cfg["device"])
    act_scale = float(cfg["umax"])
    D = int(cfg["D"])
    act_dim = D
    obs_dim = 5 * D + (2 if cfg.get("fuel", {}).get("enable", False) else 0)
    distill_role = str(cfg.get("train_role", "def")).lower()
    if distill_role not in {"def", "att"}:
        raise ValueError(f"Unsupported train_role for distillation: {distill_role!r}")
    if distill_role == "att":
        print(
            "[distill] warning: attacker distillation still uses the shared defender-centric "
            "observation layout; no attacker-specific UKF belief model is implemented."
        )

    # DAgger hyperparams
    episodes_per_iter = int(cfg.get("episodes_per_iter", 8))
    max_steps         = int(cfg.get("max_steps", cfg.get("T", 300)))
    iters             = int(cfg.get("iters", 100))

    beta_start = float(cfg.get("dagger_beta_start", 1.0))
    beta_end   = float(cfg.get("dagger_beta_end", 0.0))
    beta_decay = int(cfg.get("dagger_decay_iters", 50))

    student_lr = float(cfg.get("distill_lr", cfg.get("policy_lr", 3e-4)))
    grad_clip_norm = float(cfg.get("max_grad_norm", 1.0))
    batch_size = int(cfg.get("distill_batch_size", cfg.get("distill_mb_size", 2048)))
    distill_epochs = int(cfg.get("distill_epochs", 4))
    log_every = int(cfg.get("distill_log_every", cfg.get("log_every", 10)))
    teacher_label_mode = str(cfg.get("distill_teacher_label", "mean")).lower()
    if teacher_label_mode not in {"mean", "sample"}:
        raise ValueError(
            f"Unsupported distill_teacher_label={teacher_label_mode!r}. Expected 'mean' or 'sample'."
        )

    w_nll = float(cfg.get("distill_w_nll", 1.0))
    w_kl = float(cfg.get("distill_w_kl", 0.0))
    w_mse = float(cfg.get("distill_w_mse", 0.0))
    w_v = float(cfg.get("distill_w_v", 0.0))
    warm_start = bool(cfg.get("distill_warm_start_from_teacher", True))
    train_logstd = bool(cfg.get("distill_train_logstd", False))
    logstd_min = cfg.get("distill_logstd_min", None)
    logstd_max = cfg.get("distill_logstd_max", None)
    if logstd_min is not None:
        logstd_min = float(logstd_min)
    if logstd_max is not None:
        logstd_max = float(logstd_max)
    collection_mode = str(cfg.get("distill_collection_mode", "dagger")).lower()
    if collection_mode not in {"dagger", "teacher_forced", "intervention"}:
        raise ValueError(
            f"Unsupported distill_collection_mode={collection_mode!r}. "
            "Expected 'dagger', 'teacher_forced', or 'intervention'."
        )
    intervention_action_l2_thresh = cfg.get("distill_intervention_action_l2_thresh", None)
    if intervention_action_l2_thresh is None:
        intervention_action_l2_thresh = 0.5 * act_scale * math.sqrt(float(act_dim))
    intervention_action_l2_thresh = float(intervention_action_l2_thresh)
    intervention_oi_margin_m = float(cfg.get("distill_intervention_oi_margin_m", 1.0))
    intervention_collision_margin_m = float(cfg.get("distill_intervention_collision_margin_m", 1.0))

    # Teacher bundle
    teacher_policy, _teacher_intent = _load_teacher_bundle(
        ckpt_path=teacher_ckpt_path,
        obs_dim=obs_dim,
        act_dim=act_dim,
        cfg=cfg,
        device=device,
        lookahead_H=1,
        rel_dim=2 * D,
        latent_dim=int(cfg.get("distill_latent_dim", 8)),
    )

    # Opponent during distillation
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

    # Student uses the exact PPO policy class.
    student = ActorCriticDiff(obs_dim, act_dim, cfg).to(device)
    if warm_start:
        miss = student.load_state_dict(teacher_policy.state_dict(), strict=False)
        if miss.missing_keys or miss.unexpected_keys:
            print(
                "[distill] warm-start load mismatch: "
                f"missing={miss.missing_keys} unexpected={miss.unexpected_keys}"
            )

    opt_params = []
    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        if (name == "logstd") and (not train_logstd):
            param.requires_grad_(False)
            continue
        opt_params.append(param)
    optimizer = optim.Adam(opt_params, lr=student_lr)

    def beta_at(iter_idx: int) -> float:
        if beta_decay <= 0:
            return beta_end
        alpha = min(1.0, max(0.0, iter_idx / float(beta_decay)))
        return float((1.0 - alpha) * beta_start + alpha * beta_end)

    max_dataset_eps = cfg.get("max_dataset_episodes", 512)
    buffer_capacity = cfg.get("distill_buffer_capacity", None)
    if buffer_capacity is None:
        if max_dataset_eps is None:
            buffer_capacity = 50 * episodes_per_iter * max_steps
        else:
            buffer_capacity = int(max_dataset_eps) * max_steps
    buffer_capacity = int(buffer_capacity)
    replay = _DistillReplayBuffer(
        capacity=buffer_capacity,
        obs_dim=obs_dim,
        act_dim=act_dim,
        store_teacher_obs=(w_kl != 0.0 or w_v != 0.0),
        store_teacher_value=(w_v != 0.0),
    )
    rollout_samples_nominal = max(1, episodes_per_iter * max_steps)

    metrics = {
        "iter": [],
        "dagger_beta": [],
        "loss": [],
        "loss_nll": [],
        "loss_mse": [],
        "loss_kl": [],
        "loss_v": [],
        "replay_size": [],
    }

    print("=== Distillation: full-state teacher -> UKF-observation ActorCriticDiff student ===")
    print(f"teacher_ckpt={teacher_ckpt_path}")
    print(f"distill_role={distill_role}")
    print(f"attacker_mode={attacker_mode}")
    print(f"student_arch=ActorCriticDiff  obs_dim={obs_dim}  act_dim={act_dim}")
    print(
        f"warm_start={warm_start}  teacher_label_mode={teacher_label_mode}  "
        f"w_nll={w_nll}  w_kl={w_kl}  w_mse={w_mse}  w_v={w_v}  "
        f"train_logstd={train_logstd}  collection_mode={collection_mode}  "
        f"buffer_capacity={buffer_capacity}"
    )

    for it in range(iters):
        beta_exec = beta_at(it)
        obs_student_new: List[torch.Tensor] = []
        teacher_raw_new: List[torch.Tensor] = []
        teacher_obs_new: List[torch.Tensor] = []
        teacher_value_new: List[torch.Tensor] = []
        teacher_exec_count = 0
        intervention_count = 0
        rollout_step_count = 0

        # -------------------------------------------------
        # 1) DAgger collection
        # -------------------------------------------------
        for _ in range(episodes_per_iter):
            env = Env(cfg)
            obs_student = env.reset()

            for _t in range(max_steps):
                # teacher labels from full state
                full_obs_np = _full_obs_from_env(env)
                with torch.no_grad():
                    o_teacher_t = torch.as_tensor(
                        full_obs_np[None, :], dtype=torch.float32, device=device
                    )
                    dist_teacher = teacher_policy.dist(o_teacher_t, who=distill_role)
                    if teacher_label_mode == "mean":
                        u_teacher_raw_t = dist_teacher.mean
                    else:
                        u_teacher_raw_t = dist_teacher.rsample()
                    a_teacher_t = squash_action(u_teacher_raw_t, act_scale)

                    o_student_t = torch.as_tensor(
                        obs_student[None, :], dtype=torch.float32, device=device
                    )
                    dist_student = student.dist(o_student_t, who=distill_role)
                    a_student_t = squash_action(dist_student.mean, act_scale)

                    if w_v != 0.0:
                        v_teacher_t = teacher_policy.value(o_teacher_t).reshape(1, 1)
                    else:
                        v_teacher_t = None

                u_teacher_np = a_teacher_t[0].detach().cpu().numpy().astype(np.float32)
                a_student_np = a_student_t[0].detach().cpu().numpy().astype(np.float32)

                # Rollout action selection
                scheduled_teacher = (random.random() < beta_exec)
                use_teacher = scheduled_teacher
                intervention_triggered = False
                if collection_mode == "teacher_forced":
                    use_teacher = True
                elif collection_mode == "intervention" and (not use_teacher):
                    if np.linalg.norm(a_student_np - u_teacher_np) > intervention_action_l2_thresh:
                        use_teacher = True
                        intervention_triggered = True

                    p1, _v1, pA_list, _vA_list = env._unpack(env.state)
                    p_ctrl = p1 if distill_role == "def" else pA_list[0]

                    if (not use_teacher) and (env.oi_radius > 0.0):
                        dist_to_oi = float(np.linalg.norm(p_ctrl - env.center))
                        if dist_to_oi <= (env.oi_radius + intervention_oi_margin_m):
                            use_teacher = True
                            intervention_triggered = True

                    if (not use_teacher) and (env.collision_radius_m > 0.0):
                        dist_pair = float(np.linalg.norm(pA_list[0] - p1))
                        if dist_pair <= (env.collision_radius_m + intervention_collision_margin_m):
                            use_teacher = True
                            intervention_triggered = True

                a_exec = u_teacher_np if use_teacher else a_student_np
                rollout_step_count += 1
                if use_teacher:
                    teacher_exec_count += 1
                if intervention_triggered:
                    intervention_count += 1

                # attacker/opponent action
                a_opp_exec = _opponent_action_env(
                    env=env,
                    distill_role=distill_role,
                    attacker_mode=attacker_mode,
                    rule_ctrl=rule_ctrl,
                    opponent_policy=opponent_policy,
                    device=device,
                    act_scale=act_scale,
                )

                if distill_role == "def":
                    a1_env, aA_env = a_exec, a_opp_exec
                else:
                    a1_env, aA_env = a_opp_exec, a_exec

                obs_next, _, _, done, _ = env.step(
                    a1_env=a1_env,
                    aA_env=aA_env,
                    reward_mode="both",
                )

                obs_student_new.append(
                    torch.as_tensor(obs_student.copy(), dtype=torch.float32)
                )
                teacher_raw_new.append(
                    u_teacher_raw_t[0].detach().cpu().to(dtype=torch.float32)
                )
                if replay.teacher_obs is not None:
                    teacher_obs_new.append(
                        torch.as_tensor(full_obs_np.copy(), dtype=torch.float32)
                    )
                if replay.teacher_value is not None and v_teacher_t is not None:
                    teacher_value_new.append(
                        v_teacher_t[0].detach().cpu().to(dtype=torch.float32)
                    )
                obs_student = obs_next

                if done:
                    break

        if obs_student_new:
            replay.add(
                torch.stack(obs_student_new, dim=0),
                torch.stack(teacher_raw_new, dim=0),
                teacher_obs=torch.stack(teacher_obs_new, dim=0) if teacher_obs_new else None,
                teacher_value=torch.stack(teacher_value_new, dim=0) if teacher_value_new else None,
            )

        # -------------------------------------------------
        # 2) Supervised update on aggregated dataset
        # -------------------------------------------------
        student.train()

        total_loss = 0.0
        total_nll = 0.0
        total_mse = 0.0
        total_kl = 0.0
        total_v = 0.0
        n_batches = 0

        if replay.size > 0:
            num_mb = max(1, rollout_samples_nominal // max(1, batch_size))
            for _epoch in range(distill_epochs):
                for _mb in range(num_mb):
                    obs_batch, teacher_raw_batch, teacher_obs_batch, teacher_value_batch = replay.sample(
                        batch_size=min(batch_size, replay.size),
                        device=device,
                    )

                    dist_student = student.dist(obs_batch, who=distill_role)

                    if teacher_obs_batch is not None:
                        with torch.no_grad():
                            dist_teacher_batch = teacher_policy.dist(teacher_obs_batch, who=distill_role)
                    else:
                        dist_teacher_batch = None

                    nll = -logprob_squashed(dist_student, teacher_raw_batch).mean()
                    mse = ((dist_student.mean - teacher_raw_batch) ** 2).mean()

                    if dist_teacher_batch is not None:
                        kl = torch.distributions.kl_divergence(dist_teacher_batch, dist_student).sum(-1).mean()
                    else:
                        kl = torch.zeros((), dtype=torch.float32, device=device)

                    if teacher_value_batch is not None:
                        v_student = student.value(obs_batch).reshape(-1, 1)
                        v_mse = ((v_student - teacher_value_batch) ** 2).mean()
                    else:
                        v_mse = torch.zeros((), dtype=torch.float32, device=device)

                    loss = (w_nll * nll) + (w_mse * mse) + (w_kl * kl) + (w_v * v_mse)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(student.parameters(), grad_clip_norm)
                    optimizer.step()
                    if train_logstd and ((logstd_min is not None) or (logstd_max is not None)):
                        with torch.no_grad():
                            lo = logstd_min if logstd_min is not None else -float("inf")
                            hi = logstd_max if logstd_max is not None else float("inf")
                            student.logstd.clamp_(min=lo, max=hi)

                    total_loss += float(loss.detach().cpu())
                    total_nll += float(nll.detach().cpu())
                    total_mse += float(mse.detach().cpu())
                    total_kl += float(kl.detach().cpu())
                    total_v += float(v_mse.detach().cpu())
                    n_batches += 1

        # -------------------------------------------------
        # 3) Logging
        # -------------------------------------------------
        if (it % log_every == 0) or (it == iters - 1):
            mean_loss = total_loss / max(1, n_batches)
            mean_nll = total_nll / max(1, n_batches)
            mean_mse = total_mse / max(1, n_batches)
            mean_kl = total_kl / max(1, n_batches)
            mean_v = total_v / max(1, n_batches)
            std_mean = float(student.logstd.detach().exp().mean().cpu())
            teacher_exec_frac = float(teacher_exec_count) / max(1, rollout_step_count)
            intervention_frac = float(intervention_count) / max(1, rollout_step_count)

            print(
                f"[distill {it:04d}] beta={beta_exec:.3f}  "
                f"loss={mean_loss:.3e}  nll={mean_nll:.3e}  mse={mean_mse:.3e}  "
                f"kl={mean_kl:.3e}  v={mean_v:.3e}  std={std_mean:.3e}  "
                f"teacher_exec={teacher_exec_frac:.3f}  intervene={intervention_frac:.3f}  "
                f"replay={replay.size}"
            )

            metrics["iter"].append(float(it))
            metrics["dagger_beta"].append(float(beta_exec))
            metrics["loss"].append(mean_loss)
            metrics["loss_nll"].append(mean_nll)
            metrics["loss_mse"].append(mean_mse)
            metrics["loss_kl"].append(mean_kl)
            metrics["loss_v"].append(mean_v)
            metrics.setdefault("student_std_mean", []).append(std_mean)
            metrics.setdefault("teacher_exec_frac", []).append(teacher_exec_frac)
            metrics.setdefault("intervention_frac", []).append(intervention_frac)
            metrics["replay_size"].append(float(replay.size))

    # Save a plain ActorCriticDiff checkpoint so inference uses the same path as PPO.
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(student.state_dict(), out_path)
    print(f"[distill] saved student checkpoint -> {out_path}")

    return student, metrics


def distill_from_teacher_teacher_forced(
    cfg: Dict[str, Any],
    teacher_ckpt_path: str,
    out_path: str = "ppo_def_ukf_distilled.pt",
):
    """
    Expert-rollout UKF behavior cloning baseline.

    This reuses the modern ActorCriticDistillation machinery, but forces the
    teacher to execute every rollout action during data collection instead of
    mixing in student actions DAgger-style.
    """
    cfg_tf = copy.deepcopy(cfg)
    cfg_tf["dagger_beta_start"] = 1.0
    cfg_tf["dagger_beta_end"] = 1.0
    cfg_tf["dagger_decay_iters"] = 0
    cfg_tf["distill_collection_mode"] = "teacher_forced"

    print("[distill] teacher_forced mode: executing teacher for all rollout actions.")
    return distill_from_teacher_modern(
        cfg_tf,
        teacher_ckpt_path,
        out_path=out_path,
    )


def distill_from_teacher_intervention(
    cfg: Dict[str, Any],
    teacher_ckpt_path: str,
    out_path: str = "ppo_def_ukf_distilled.pt",
):
    """
    Intervention-DAgger baseline.

    The student is allowed to act, but the teacher overrides execution when the
    student deviates too far in action space or gets too close to protected
    geometry.
    """
    cfg_int = copy.deepcopy(cfg)
    cfg_int["distill_collection_mode"] = "intervention"
    print("[distill] intervention mode: teacher overrides unsafe/divergent student actions.")
    return distill_from_teacher_modern(
        cfg_int,
        teacher_ckpt_path,
        out_path=out_path,
    )


def distill_from_teacher(
    cfg: Dict[str, Any],
    teacher_ckpt_path: str,
    out_path: str = "ppo_def_ukf_distilled.pt",
):
    method = str(cfg.get("distill_method", "modern")).lower()
    if method in {"modern", "default", "actor_critic_diff"}:
        return distill_from_teacher_modern(
            cfg,
            teacher_ckpt_path,
            out_path=out_path,
        )
    if method in {"teacher_forced", "teacher_forcing", "expert_rollout", "bc"}:
        return distill_from_teacher_teacher_forced(
            cfg,
            teacher_ckpt_path,
            out_path=out_path,
        )
    if method in {"intervention", "safe_dagger", "intervene"}:
        return distill_from_teacher_intervention(
            cfg,
            teacher_ckpt_path,
            out_path=out_path,
        )
    if method in {"original", "legacy", "paper"}:
        return distill_from_teacher_original(
            cfg,
            teacher_ckpt_path,
            out_path=out_path,
        )
    raise ValueError(
        f"Unsupported distill_method={method!r}. Expected one of "
        "'modern', 'teacher_forced', 'intervention', or 'original'."
    )
