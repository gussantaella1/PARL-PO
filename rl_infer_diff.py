# rl_infer_diff.py
# Inference helpers for Diff-Nash policy (5D observation = 15 for D=3)

from __future__ import annotations
from typing import Dict, Any, Tuple
import os
import numpy as np
import torch

# IMPORTANT: import the SAME class you used during training
# If your file/class name differs, update this import accordingly.
from rl_loop_diffgame import ActorCriticDiff  # must exist in your project


def _to_t32(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


class RLPolicyDiff:
    """
    Loads Diff-Nash defender/attacker nets and exposes:
      - act_def_obs(obs15), act_att_obs(obs15)
    where obs15 = [p1-c, p2-c, (p2-p1), v1, v2] in float32.
    """
    def __init__(self,
                 cfg: Dict[str, Any],
                 device: str = "cpu",
                 def_ckpt: str | None = None,
                 att_ckpt: str | None = None):
        self.cfg = cfg
        self.device = device

        D = int(cfg.get("D", 3))
        self.D = D
        self.obs_dim = 5 * D
        self.act_dim = D
        self.umax = float(cfg.get("umax", 0.02))

        # Build the SAME nets used for training
        self.pi_def = ActorCriticDiff(self.obs_dim, self.act_dim, cfg).to(self.device)
        self.pi_att = ActorCriticDiff(self.obs_dim, self.act_dim, cfg).to(self.device)

        # Resolve checkpoints
        def_ckpt = def_ckpt or cfg.get("ckpt_def", "ppo_def_diff.pt")
        att_ckpt = att_ckpt or cfg.get("ckpt_att", "ppo_att_diff.pt")

        # Load strictly (shape must match)
        sd_def = torch.load(def_ckpt, map_location=self.device)
        sd_att = torch.load(att_ckpt, map_location=self.device)
        self.pi_def.load_state_dict(sd_def, strict=True)
        self.pi_att.load_state_dict(sd_att, strict=True)
        self.pi_def.eval()
        self.pi_att.eval()

    # ------------- Public API (obs-aware) -------------
    @torch.no_grad()
    def act_def_obs(self, obs_15: np.ndarray | torch.Tensor, deterministic: bool = True) -> np.ndarray:
        o = _to_t32(obs_15, self.device).unsqueeze(0)  # (1, obs_dim)
        dist = self.pi_def.dist(o, who="def")
        u_raw = dist.mean if deterministic else dist.rsample()
        a = torch.tanh(u_raw) * self.umax
        return a.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def act_att_obs(self, obs_15: np.ndarray | torch.Tensor, deterministic: bool = True) -> np.ndarray:
        o = _to_t32(obs_15, self.device).unsqueeze(0)
        dist = self.pi_att.dist(o, who="att")
        u_raw = dist.mean if deterministic else dist.rsample()
        a = torch.tanh(u_raw) * self.umax
        return a.squeeze(0).cpu().numpy()

    # ------------- Optional helpers -------------
    @staticmethod
    def build_train_obs(p1, v1, p2, v2, center) -> np.ndarray:
        """
        Returns float32 array with shape (5D,)
        [p1-c, p2-c, (p2-p1), v1, v2]
        """
        p1 = np.asarray(p1, dtype=np.float32)
        v1 = np.asarray(v1, dtype=np.float32)
        p2 = np.asarray(p2, dtype=np.float32)
        v2 = np.asarray(v2, dtype=np.float32)
        c  = np.asarray(center, dtype=np.float32)
        return np.concatenate([p1 - c, p2 - c, (p2 - p1), v1, v2], axis=-1).astype(np.float32)

    def verify_ckpt_compat(self) -> Tuple[int, int]:
        """
        Quick check to ensure model's first linear layer matches obs_dim.
        Returns (def_in_features, att_in_features).
        """
        def_in = self.pi_def.vf[0].in_features  # expects obs_dim
        att_in = self.pi_att.vf[0].in_features
        return def_in, att_in
