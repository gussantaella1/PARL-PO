from __future__ import annotations

import copy
import os
import pickle
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.utils import set_seed
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
# Main rewrite
# =========================================================

def distill_from_teacher(
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

    # DAgger hyperparams
    episodes_per_iter = int(cfg.get("episodes_per_iter", 8))
    max_steps         = int(cfg.get("max_steps", cfg.get("T", 300)))
    iters             = int(cfg.get("iters", 100))

    beta_start = float(cfg.get("dagger_beta_start", 1.0))
    beta_end   = float(cfg.get("dagger_beta_end", 0.0))
    beta_decay = int(cfg.get("dagger_decay_iters", 50))

    student_lr       = float(cfg.get("distill_lr", cfg.get("policy_lr", 3e-4)))
    grad_clip_norm   = float(cfg.get("max_grad_norm", 1.0))
    max_dataset_eps  = int(cfg.get("max_dataset_episodes", 512))
    log_every        = int(cfg.get("log_every", 10))

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

    optimizer = optim.Adam(student.parameters(), lr=student_lr)
    act_loss_fn = nn.MSELoss()

    def beta_at(iter_idx: int) -> float:
        if beta_decay <= 0:
            return beta_end
        alpha = min(1.0, max(0.0, iter_idx / float(beta_decay)))
        return float((1.0 - alpha) * beta_start + alpha * beta_end)

    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []

    metrics = {
        "iter": [],
        "dagger_beta": [],
        "loss": [],
        "loss_action": [],
        "dataset_episodes": [],
    }

    print("=== Distillation: full-state teacher -> UKF-observation ActorCriticDiff student ===")
    print(f"teacher_ckpt={teacher_ckpt_path}")
    print(f"distill_role={distill_role}")
    print(f"attacker_mode={attacker_mode}")
    print(f"student_arch=ActorCriticDiff  obs_dim={obs_dim}  act_dim={act_dim}")

    for it in range(iters):
        beta_exec = beta_at(it)
        new_eps: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # -------------------------------------------------
        # 1) DAgger collection
        # -------------------------------------------------
        for _ in range(episodes_per_iter):
            env = Env(cfg)
            obs_student = env.reset()

            for _t in range(max_steps):
                # teacher labels from full state + privileged future
                full_obs_np = _full_obs_from_env(env)
                u_teacher_np = _teacher_action_env(
                    teacher_policy=teacher_policy,
                    full_obs=full_obs_np,
                    device=device,
                    act_scale=act_scale,
                    distill_role=distill_role,
                )

                with torch.no_grad():
                    o_student_t = torch.as_tensor(
                        obs_student[None, :], dtype=torch.float32, device=device
                    )
                    dist_student = student.dist(o_student_t, who=distill_role)
                    a_student_t = torch.tanh(dist_student.mean) * act_scale
                    a_student_np = a_student_t[0].detach().cpu().numpy().astype(np.float32)

                # DAgger execution mixture
                use_teacher = (random.random() < beta_exec)
                a_exec = u_teacher_np if use_teacher else a_student_np

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

                # store supervised sample
                new_eps.append(
                    (
                        torch.as_tensor(obs_student.copy(), dtype=torch.float32, device=device),
                        torch.as_tensor(u_teacher_np.copy(), dtype=torch.float32, device=device),
                    )
                )
                obs_student = obs_next

                if done:
                    break

        dataset.extend(new_eps)
        if len(dataset) > max_dataset_eps:
            dataset = dataset[-max_dataset_eps:]

        # -------------------------------------------------
        # 2) Supervised TBPTT on aggregated dataset
        # -------------------------------------------------
        student.train()

        total_loss = 0.0
        total_act  = 0.0
        n_batches  = 0

        if dataset:
            batch_size = int(cfg.get("distill_batch_size", min(512, max(32, len(dataset)))))
            order = torch.randperm(len(dataset), device=device)
            obs_all = torch.stack([x for x, _ in dataset], dim=0)
            act_all = torch.stack([u for _, u in dataset], dim=0)

            for start in range(0, len(dataset), batch_size):
                idx = order[start : start + batch_size]
                obs_batch = obs_all[idx]
                act_batch = act_all[idx]

                dist = student.dist(obs_batch, who=distill_role)
                pred_actions = torch.tanh(dist.mean) * act_scale

                loss_act = act_loss_fn(pred_actions, act_batch)
                loss = loss_act

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), grad_clip_norm)
                optimizer.step()

                total_loss += float(loss.detach().cpu())
                total_act += float(loss_act.detach().cpu())
                n_batches += 1

        # -------------------------------------------------
        # 3) Logging
        # -------------------------------------------------
        if (it % log_every == 0) or (it == iters - 1):
            mean_loss = total_loss / max(1, n_batches)
            mean_act  = total_act / max(1, n_batches)

            print(
                f"[distill {it:04d}] beta={beta_exec:.3f}  "
                f"loss={mean_loss:.3e}  action={mean_act:.3e}  "
                f"dataset_samples={len(dataset)}"
            )

            metrics["iter"].append(float(it))
            metrics["dagger_beta"].append(float(beta_exec))
            metrics["loss"].append(mean_loss)
            metrics["loss_action"].append(mean_act)
            metrics["dataset_episodes"].append(float(len(dataset)))

    # Save a plain ActorCriticDiff checkpoint so inference uses the same path as PPO.
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(student.state_dict(), out_path)
    print(f"[distill] saved student checkpoint -> {out_path}")

    return student, metrics
