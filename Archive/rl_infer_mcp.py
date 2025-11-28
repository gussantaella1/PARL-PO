# rl_infer_mcp.py
import os
import numpy as np
import torch
import torch.nn as nn

# Minimal ActorCritic must match training architecture (hidden sizes, etc.)
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.logstd = nn.Parameter(torch.full((act_dim,), -1.0))
        self.vf = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def dist(self, obs: torch.Tensor):
        h = self.pi(obs)
        mu = self.mu(h)
        std = self.logstd.exp()
        return torch.distributions.Normal(mu, std)

class RLPolicy:
    """
    Loads defender/attacker PPO nets and exposes obs-based act APIs:
      - act_def_obs(obs, deterministic=True/False)
      - act_att_obs(obs, deterministic=True/False)

    Assumes obs contract is the same one used in training (length = 5*D).
    """
    def __init__(self, cfg, device="cpu", def_ckpt=None, att_ckpt=None):
        self.cfg = cfg
        self.D = int(cfg["D"])
        self.obs_dim = 5 * self.D
        self.act_dim = self.D
        self.device = torch.device(device)

        self.pi_def = ActorCritic(self.obs_dim, self.act_dim).to(self.device)
        self.pi_att = ActorCritic(self.obs_dim, self.act_dim).to(self.device)

        def_ckpt = def_ckpt or cfg.get("ckpt_def", "ppo_def.pt")
        att_ckpt = att_ckpt or cfg.get("ckpt_att", "ppo_att.pt")
        print("[infer] loading ckpts:", def_ckpt, att_ckpt)

        sd_def = torch.load(def_ckpt, map_location=self.device)
        sd_att = torch.load(att_ckpt, map_location=self.device)

        miss_def = self.pi_def.load_state_dict(sd_def, strict=False)
        miss_att = self.pi_att.load_state_dict(sd_att, strict=False)
        print("[infer] missing/unexpected (def):", miss_def)
        print("[infer] missing/unexpected (att):", miss_att)

        self.act_scale = float(cfg.get("umax", 1.0))

    @torch.no_grad()
    def _act(self, net: ActorCritic, obs: np.ndarray, deterministic: bool):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, obs_dim)
        dist = net.dist(obs_t)
        if deterministic:
            u_raw = dist.mean
        else:
            u_raw = dist.rsample()
        u = torch.tanh(u_raw) * self.act_scale
        return u.squeeze(0).cpu().numpy()

    def act_def_obs(self, obs: np.ndarray, deterministic: bool = True):
        return self._act(self.pi_def, obs, deterministic)

    def act_att_obs(self, obs: np.ndarray, deterministic: bool = True):
        return self._act(self.pi_att, obs, deterministic)

    # Optional raw-state API (if you had an older runner that packed states differently)
    @staticmethod
    def pack_state(p1, v1, p2, v2):
        return np.concatenate([p1, v1, p2, v2]).astype(np.float32)
