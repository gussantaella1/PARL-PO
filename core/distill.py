from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
    payload = torch.load(ckpt_path, map_location=device)

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


def _load_attacker_policy_if_needed(
    cfg: Dict[str, Any],
    obs_dim: int,
    act_dim: int,
    device: torch.device,
):
    attacker_mode = str(cfg.get("attacker_mode", "rule")).lower()
    if attacker_mode == "rule":
        return None

    att_ckpt = (
        cfg.get("att_ckpt_path")
        or cfg.get("att_ckpt")
        or cfg.get("att_policy_path")
        or cfg.get("attacker_ckpt_path")
    )
    if att_ckpt is None:
        raise RuntimeError("attacker_mode='rl' but no attacker checkpoint path was provided.")

    payload = torch.load(att_ckpt, map_location=device)
    att_sd = _extract_state_dict(
        payload,
        ["policy_state_dict", "state_dict", "model", "net", "att_net", "actor_critic"],
    )
    if att_sd is None:
        if isinstance(payload, dict) and any(k.startswith("att_net.") for k in payload.keys()):
            att_sd = {k[len("att_net."):]: v for k, v in payload.items() if k.startswith("att_net.")}
        else:
            raise RuntimeError(f"Could not parse attacker checkpoint: {att_ckpt}")

    attacker = ActorCriticDiff(obs_dim, act_dim, cfg).to(device)
    attacker.load_state_dict(att_sd, strict=False)
    attacker.eval()
    for p in attacker.parameters():
        p.requires_grad_(False)
    return attacker


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

def _teacher_def_action_env(teacher_policy, full_obs: np.ndarray, device: torch.device, act_scale: float) -> np.ndarray:
    o = torch.as_tensor(full_obs[None, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        dist = teacher_policy.dist(o, who="def")
        a_env = torch.tanh(dist.mean) * act_scale
    return a_env[0].detach().cpu().numpy().astype(np.float32)


def _attacker_action_env(
    env,
    attacker_mode: str,
    rule_ctrl,
    attacker_policy,
    device: torch.device,
    act_scale: float,
) -> np.ndarray:
    if attacker_mode == "rule":
        p1, v1, pA_list, vA_list = env._unpack(env.state)
        p2 = pA_list[0]
        v2 = vA_list[0]
        return rule_ctrl.act(p1, v1, p2, v2).astype(np.float32)

    full_obs = _full_obs_from_env(env)
    o = torch.as_tensor(full_obs[None, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        dist = attacker_policy.dist(o, who="att")
        a_env = torch.tanh(dist.mean) * act_scale
    return a_env[0].detach().cpu().numpy().astype(np.float32)


def _teacher_future_rel_traj(
    env,
    teacher_policy,
    attacker_mode: str,
    rule_ctrl,
    attacker_policy,
    lookahead_H: int,
    device: torch.device,
    act_scale: float,
) -> np.ndarray:
    """
    Roll a cloned env forward under teacher defender + current attacker model
    to produce the privileged future relative trajectory label.
    """
    sim = copy.deepcopy(env)
    rel_seq = []

    for _ in range(lookahead_H + 1):
        rel_seq.append(_true_relative_state_from_env(sim))

        full_obs = _full_obs_from_env(sim)
        a_def = _teacher_def_action_env(teacher_policy, full_obs, device, act_scale)
        a_att = _attacker_action_env(
            sim,
            attacker_mode=attacker_mode,
            rule_ctrl=rule_ctrl,
            attacker_policy=attacker_policy,
            device=device,
            act_scale=act_scale,
        )

        _, _, _, done, _ = sim.step(a_def, a_att, reward_mode="both")
        if done:
            break

    # pad if episode terminated early
    while len(rel_seq) < (lookahead_H + 1):
        rel_seq.append(rel_seq[-1].copy())

    return np.stack(rel_seq, axis=0).astype(np.float32)


# =========================================================
# Main rewrite
# =========================================================

def distill_from_teacher(
    cfg: Dict[str, Any],
    teacher_ckpt_path: str,
    out_path: str = "ppo_def_ukf_distilled.pt",
):
    """
    Paper-style teacher -> partially-observable student distillation.

    What this implements:
      - fully-observable teacher action labels from the true state
      - privileged teacher latent labels z_t from FUTURE relative trajectory
      - partially-observable student with recurrent encoder over
          (xhat_rel_t, Sigma_t, u_{t-1})
      - DAgger rollout collection
      - supervised losses on BOTH:
          (1) action imitation
          (2) latent intent imitation

    Important:
      Preferred: teacher checkpoint contains BOTH policy and privileged intent encoder E*.
      Optional fallback: if E* is missing and cfg["allow_sim_intent_fallback"] is True,
      we synthesize intent labels from mini simulation rollouts of teacher+attacker.
      This preserves the rollout logic but is less paper-faithful than learned E*.
    """
    cfg = copy.deepcopy(cfg)

    # Student must be partial-observable
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

    # DAgger / TBPTT hyperparams
    episodes_per_iter = int(cfg.get("episodes_per_iter", 8))
    max_steps         = int(cfg.get("max_steps", cfg.get("T", 300)))
    iters             = int(cfg.get("iters", 100))
    lookahead_H       = int(cfg.get("lookahead_H", 15))
    tbptt_chunk_len   = int(cfg.get("tbptt_chunk_len", 40))

    beta_start = float(cfg.get("dagger_beta_start", 1.0))
    beta_end   = float(cfg.get("dagger_beta_end", 0.0))
    beta_decay = int(cfg.get("dagger_decay_iters", 50))

    latent_dim       = int(cfg.get("distill_latent_dim", 8))
    lambda_intent    = float(cfg.get("lambda_intent", 1.0))
    student_lr       = float(cfg.get("distill_lr", cfg.get("policy_lr", 3e-4)))
    grad_clip_norm   = float(cfg.get("max_grad_norm", 1.0))
    max_dataset_eps  = int(cfg.get("max_dataset_episodes", 512))
    log_every        = int(cfg.get("log_every", 10))

    # Student feature sizes
    xhat_dim = 2 * D
    sigma_dim = (2 * D) * (2 * D + 1) // 2
    rel_dim = 2 * D

    # Teacher bundle
    teacher_policy, teacher_intent = _load_teacher_bundle(
        ckpt_path=teacher_ckpt_path,
        obs_dim=obs_dim,
        act_dim=act_dim,
        cfg=cfg,
        device=device,
        lookahead_H=lookahead_H,
        rel_dim=rel_dim,
        latent_dim=latent_dim,
    )

    # Opponent during distillation
    attacker_mode = str(cfg.get("attacker_mode", "rule")).lower()
    rule_ctrl = AttackerRuleController(cfg) if attacker_mode == "rule" else None
    attacker_policy = _load_attacker_policy_if_needed(cfg, obs_dim, act_dim, device) if attacker_mode == "rl" else None

    # Student
    student = PartialObsStudentPolicy(
        xhat_dim=xhat_dim,
        sigma_dim=sigma_dim,
        act_dim=act_dim,
        latent_dim=latent_dim,
        lstm_hidden=256,
        act_scale=act_scale,
    ).to(device)

    optimizer = optim.Adam(student.parameters(), lr=student_lr)
    act_loss_fn = nn.MSELoss()
    z_loss_fn   = nn.MSELoss()

    def beta_at(iter_idx: int) -> float:
        if beta_decay <= 0:
            return beta_end
        alpha = min(1.0, max(0.0, iter_idx / float(beta_decay)))
        return float((1.0 - alpha) * beta_start + alpha * beta_end)

    dataset: List[DAggerEpisode] = []

    metrics = {
        "iter": [],
        "dagger_beta": [],
        "loss": [],
        "loss_action": [],
        "loss_intent": [],
        "dataset_episodes": [],
    }

    print("=== Paper-style distillation: full-state teacher -> UKF partial-observable student ===")
    print(f"teacher_ckpt={teacher_ckpt_path}")
    print(f"attacker_mode={attacker_mode}")
    print(f"lookahead_H={lookahead_H}, latent_dim={latent_dim}, tbptt_chunk_len={tbptt_chunk_len}")
    print(f"intent_source={'teacher_encoder' if teacher_intent is not None else 'sim_fallback'}")

    for it in range(iters):
        beta_exec = beta_at(it)
        new_eps: List[DAggerEpisode] = []

        # -------------------------------------------------
        # 1) DAgger collection
        # -------------------------------------------------
        for _ in range(episodes_per_iter):
            env = Env(cfg)
            obs = env.reset()

            hidden = student.init_hidden(batch_size=1, device=device)
            u_prev_exec = np.zeros((act_dim,), dtype=np.float32)

            xhat_hist = []
            sigma_hist = []
            uprev_hist = []
            uteach_hist = []
            zteach_hist = []

            for _t in range(max_steps):
                # student inputs from UKF / filter belief
                xhat_rel_np, sigma_np, u_prev_np = _ukf_student_features_from_env(env, u_prev_exec)

                # teacher labels from full state + privileged future
                full_obs_np = _full_obs_from_env(env)
                u_teacher_np = _teacher_def_action_env(
                    teacher_policy=teacher_policy,
                    full_obs=full_obs_np,
                    device=device,
                    act_scale=act_scale,
                )

                future_rel_np = _teacher_future_rel_traj(
                    env=env,
                    teacher_policy=teacher_policy,
                    attacker_mode=attacker_mode,
                    rule_ctrl=rule_ctrl,
                    attacker_policy=attacker_policy,
                    lookahead_H=lookahead_H,
                    device=device,
                    act_scale=act_scale,
                )

                if teacher_intent is not None:
                    with torch.no_grad():
                        future_rel_t = torch.as_tensor(
                            future_rel_np[None, :, :], dtype=torch.float32, device=device
                        )
                        z_teacher_np = teacher_intent(future_rel_t)[0].detach().cpu().numpy().astype(np.float32)
                else:
                    z_teacher_np = _sim_intent_from_future_rel_traj(future_rel_np, latent_dim=latent_dim)

                # student prediction
                xhat_t = torch.as_tensor(xhat_rel_np[None, :], dtype=torch.float32, device=device)
                sigma_t = torch.as_tensor(sigma_np[None, :], dtype=torch.float32, device=device)
                uprev_t = torch.as_tensor(u_prev_np[None, :], dtype=torch.float32, device=device)

                with torch.no_grad():
                    a_student_t, zhat_t, hidden = student.step(xhat_t, sigma_t, uprev_t, hidden)
                    a_student_np = a_student_t[0].detach().cpu().numpy().astype(np.float32)

                # DAgger execution mixture
                use_teacher = (random.random() < beta_exec)
                a_def_exec = u_teacher_np if use_teacher else a_student_np

                # attacker/opponent action
                a_att_exec = _attacker_action_env(
                    env=env,
                    attacker_mode=attacker_mode,
                    rule_ctrl=rule_ctrl,
                    attacker_policy=attacker_policy,
                    device=device,
                    act_scale=act_scale,
                )

                _, _, _, done, _ = env.step(
                    a1_env=a_def_exec,
                    aA_env=a_att_exec,
                    reward_mode="both",
                )

                # store supervised sample
                xhat_hist.append(xhat_rel_np.copy())
                sigma_hist.append(sigma_np.copy())
                uprev_hist.append(u_prev_np.copy())
                uteach_hist.append(u_teacher_np.copy())
                zteach_hist.append(z_teacher_np.copy())

                u_prev_exec = a_def_exec.copy()

                if done:
                    break

            ep = DAggerEpisode(
                xhat_rel=torch.as_tensor(np.stack(xhat_hist), dtype=torch.float32, device=device),
                sigma=torch.as_tensor(np.stack(sigma_hist), dtype=torch.float32, device=device),
                u_prev=torch.as_tensor(np.stack(uprev_hist), dtype=torch.float32, device=device),
                u_teacher=torch.as_tensor(np.stack(uteach_hist), dtype=torch.float32, device=device),
                z_teacher=torch.as_tensor(np.stack(zteach_hist), dtype=torch.float32, device=device),
            )
            new_eps.append(ep)

        dataset.extend(new_eps)
        if len(dataset) > max_dataset_eps:
            dataset = dataset[-max_dataset_eps:]

        # -------------------------------------------------
        # 2) Supervised TBPTT on aggregated dataset
        # -------------------------------------------------
        student.train()

        total_loss = 0.0
        total_act  = 0.0
        total_z    = 0.0
        n_chunks   = 0

        for ep in dataset:
            hidden = student.init_hidden(batch_size=1, device=device)

            for xhat_seq, sigma_seq, uprev_seq, uteach_seq, zteach_seq in _iter_tbptt_chunks(ep, tbptt_chunk_len):
                if isinstance(hidden, tuple):
                    hidden = tuple(h.detach() for h in hidden)
                else:
                    hidden = hidden.detach()

                pred_actions = []
                pred_latents = []

                Tchunk = xhat_seq.shape[0]
                for k in range(Tchunk):
                    a_pred, z_pred, hidden = student.step(
                        xhat_seq[k].unsqueeze(0),
                        sigma_seq[k].unsqueeze(0),
                        uprev_seq[k].unsqueeze(0),
                        hidden,
                    )
                    pred_actions.append(a_pred)
                    pred_latents.append(z_pred)

                pred_actions = torch.cat(pred_actions, dim=0)
                pred_latents = torch.cat(pred_latents, dim=0)

                loss_act = act_loss_fn(pred_actions, uteach_seq)
                loss_z   = z_loss_fn(pred_latents, zteach_seq)
                loss = loss_act + lambda_intent * loss_z

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), grad_clip_norm)
                optimizer.step()

                total_loss += float(loss.detach().cpu())
                total_act  += float(loss_act.detach().cpu())
                total_z    += float(loss_z.detach().cpu())
                n_chunks   += 1

        # -------------------------------------------------
        # 3) Logging
        # -------------------------------------------------
        if (it % log_every == 0) or (it == iters - 1):
            mean_loss = total_loss / max(1, n_chunks)
            mean_act  = total_act / max(1, n_chunks)
            mean_z    = total_z / max(1, n_chunks)

            print(
                f"[distill {it:04d}] beta={beta_exec:.3f}  "
                f"loss={mean_loss:.3e}  action={mean_act:.3e}  latent={mean_z:.3e}  "
                f"dataset_eps={len(dataset)}"
            )

            metrics["iter"].append(float(it))
            metrics["dagger_beta"].append(float(beta_exec))
            metrics["loss"].append(mean_loss)
            metrics["loss_action"].append(mean_act)
            metrics["loss_intent"].append(mean_z)
            metrics["dataset_episodes"].append(float(len(dataset)))

    # Save a richer student checkpoint
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(
        {
            "student_state_dict": student.state_dict(),
            "cfg": cfg,
            "meta": {
                "type": "paper_style_partial_obs_student",
                "uses_ukf": True,
                "latent_dim": latent_dim,
                "lookahead_H": lookahead_H,
                "tbptt_chunk_len": tbptt_chunk_len,
                "teacher_ckpt_path": teacher_ckpt_path,
            },
            "metrics": metrics,
        },
        out_path,
    )
    print(f"[distill] saved student checkpoint -> {out_path}")

    return student, metrics
