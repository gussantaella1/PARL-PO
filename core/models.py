from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from core.utils import logprob_squashed, squash_action
# =============================================================
# DiffLS Layer & Actor-Critic
# =============================================================
def _obs_extra_dim_from_cfg(cfg: Dict[str, Any]) -> int:
    radius_knob = cfg.get("arena_radius_knob", {}) or {}
    return 1 if bool(radius_knob.get("append_to_obs", radius_knob.get("enabled", False))) else 0


class DiffLSLayer(nn.Module):
    """
    Analytic one-step ridge prior using the SAME discrete dynamics as the env:

        x_{k+1} = Ad x_k + Bd u_k

    For each agent, define position selector P so that:
        p_{k+1} = P Ad x + P Bd u = E + F u

    Then solve:
        min_u ||E + F u||^2 + ridge * ||u||^2

    which gives:
        u* = -(F^T F + ridge I)^(-1) F^T E

    Observation format assumed (single-attacker case):
        obs = [p1c, p2c, rel, v1, v2]                 if fuel disabled
        obs = [p1c, p2c, rel, v1, v2, f_def, f_att]   if fuel enabled

    where:
        p1c = p1 - center
        p2c = p2 - center
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.D = int(cfg["D"])
        self.ridge = float(cfg.get("prior_ridge", 1e-2))
        self.use_fuel = bool(cfg.get("fuel", {}).get("enable", False))
        self.num_attackers = int(cfg.get("num_attackers", 1))
        self.extra_obs_dim = _obs_extra_dim_from_cfg(cfg)

        if self.num_attackers != 1:
            raise NotImplementedError(
                "DiffLSLayer currently supports only num_attackers=1."
            )

        Ad = np.asarray(cfg["dyn"]["Ad"], dtype=np.float32)
        Bd = np.asarray(cfg["dyn"]["Bd"], dtype=np.float32)

        if Ad.shape != (2 * self.D, 2 * self.D):
            raise ValueError(
                f"Expected Ad shape {(2*self.D, 2*self.D)}, got {Ad.shape}"
            )
        if Bd.shape != (2 * self.D, self.D):
            raise ValueError(
                f"Expected Bd shape {(2*self.D, self.D)}, got {Bd.shape}"
            )

        self.register_buffer("Ad", torch.tensor(Ad, dtype=torch.float32))
        self.register_buffer("Bd", torch.tensor(Bd, dtype=torch.float32))

        # Position selector P: x = [p, v] -> P x = p
        P = np.hstack([
            np.eye(self.D, dtype=np.float32),
            np.zeros((self.D, self.D), dtype=np.float32),
        ])
        self.register_buffer("P", torch.tensor(P, dtype=torch.float32))

        # F = P Bd
        F = P @ Bd                                # (D, D)
        M = F.T @ F + self.ridge * np.eye(self.D, dtype=np.float32)

        self.register_buffer("F", torch.tensor(F, dtype=torch.float32))
        self.register_buffer("K", torch.tensor(np.linalg.solve(M, F.T), dtype=torch.float32))
        # K shape: (D, D), so u* = -K E

        self.feature_dim = 4 * self.D + (2 if self.use_fuel else 0) + self.extra_obs_dim

    def _split_obs(self, obs: torch.Tensor):
        """
        Single-attacker observation parser.

        Returns:
            p1c, p2c, rel, v1, v2, fdef, fatt, extras
        where fdef/fatt are None if fuel is disabled.
        """
        D = self.D
        base_dim = 5 * D
        expected = base_dim + (2 if self.use_fuel else 0) + self.extra_obs_dim
        if obs.shape[-1] != expected:
            raise ValueError(
                f"Expected obs dim {expected}, got {obs.shape[-1]}. "
                f"Check prior layer vs env observation layout."
            )

        p1c = obs[:, 0:D]
        p2c = obs[:, D:2*D]
        rel = obs[:, 2*D:3*D]
        v1  = obs[:, 3*D:4*D]
        v2  = obs[:, 4*D:5*D]

        if self.use_fuel:
            fdef = obs[:, base_dim:base_dim+1]
            fatt = obs[:, base_dim+1:base_dim+2]
            extras = obs[:, base_dim+2:]
        else:
            fdef = None
            fatt = None
            extras = obs[:, base_dim:]

        return p1c, p2c, rel, v1, v2, fdef, fatt, extras

    def _one_step_prior(self, p_c: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Build x = [p_centered, v], then solve one-step ridge prior
        using the actual discrete dynamics.
        """
        x = torch.cat([p_c, v], dim=-1)           # (B, 2D)
        E = x @ self.Ad.T @ self.P.T              # (B, D), same as P Ad x
        u = -(E @ self.K.T)                       # (B, D)
        return u

    def forward(self, obs: torch.Tensor, who: str):
        p1c, p2c, rel, v1, v2, fdef, fatt, extras = self._split_obs(obs)

        u_def_prior = self._one_step_prior(p1c, v1)
        u_att_prior = self._one_step_prior(p2c, v2)
        u_prior = u_def_prior if who == "def" else u_att_prior

        if self.use_fuel:
            feats = torch.cat([p1c, p2c, v1, v2, fdef, fatt, extras], dim=-1)
        else:
            feats = torch.cat([p1c, p2c, v1, v2, extras], dim=-1)

        return feats, u_prior


class NoPriorLayer(nn.Module):
    """
    No analytic prior: just repackage the observation into features
    and return u_prior = 0.

    Observation format assumed (single-attacker case):
        obs = [p1c, p2c, rel, v1, v2]                 if fuel disabled
        obs = [p1c, p2c, rel, v1, v2, f_def, f_att]   if fuel enabled
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.D = int(cfg["D"])
        self.use_fuel = bool(cfg.get("fuel", {}).get("enable", False))
        self.num_attackers = int(cfg.get("num_attackers", 1))
        self.extra_obs_dim = _obs_extra_dim_from_cfg(cfg)

        if self.num_attackers != 1:
            raise NotImplementedError(
                "NoPriorLayer currently supports only num_attackers=1."
            )

        self.feature_dim = 4 * self.D + (2 if self.use_fuel else 0) + self.extra_obs_dim

    def _split_obs(self, obs: torch.Tensor):
        D = self.D
        base_dim = 5 * D
        expected = base_dim + (2 if self.use_fuel else 0) + self.extra_obs_dim
        if obs.shape[-1] != expected:
            raise ValueError(
                f"Expected obs dim {expected}, got {obs.shape[-1]}. "
                f"Check prior layer vs env observation layout."
            )

        p1c = obs[:, 0:D]
        p2c = obs[:, D:2*D]
        rel = obs[:, 2*D:3*D]
        v1  = obs[:, 3*D:4*D]
        v2  = obs[:, 4*D:5*D]

        if self.use_fuel:
            fdef = obs[:, base_dim:base_dim+1]
            fatt = obs[:, base_dim+1:base_dim+2]
            extras = obs[:, base_dim+2:]
        else:
            fdef = None
            fatt = None
            extras = obs[:, base_dim:]

        return p1c, p2c, rel, v1, v2, fdef, fatt, extras

    def forward(self, obs: torch.Tensor, who: str):
        B, D = obs.shape[0], self.D
        device, dtype = obs.device, obs.dtype

        p1c, p2c, rel, v1, v2, fdef, fatt, extras = self._split_obs(obs)

        if self.use_fuel:
            feats = torch.cat([p1c, p2c, v1, v2, fdef, fatt, extras], dim=-1)
        else:
            feats = torch.cat([p1c, p2c, v1, v2, extras], dim=-1)

        u_prior = torch.zeros((B, D), device=device, dtype=dtype)
        return feats, u_prior




class ActorCriticDiff(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg: Dict[str, Any]):
        super().__init__()
        hidden = 128

        # Choose which prior layer to use
        prior_type = cfg.get("prior_type", "ls")  # "ls", "nash", or "none"
        if prior_type == "ls":
            self.layer = DiffLSLayer(cfg)
        elif prior_type == "none":
            self.layer = NoPriorLayer(cfg)
        else:
            raise ValueError(
                f"Unknown prior_type={prior_type!r}, expected 'ls', 'nash', or 'none'."
            )

        # Policy (residual over prior)
        feat_dim = self.layer.feature_dim

        self.pi = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu_res = nn.Linear(hidden, act_dim)
        self.logstd = nn.Parameter(torch.full((act_dim,), -1.0))  # std ~ 0.37

        # Value function
        self.vf = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        # How strongly to trust the prior in μ = μ_res + blend * u_prior
        # self.prior_blend_def = float(cfg.get("prior_blend_def"))
        # self.prior_blend_att = float(cfg.get("prior_blend_att"))

        # self.prior_blend_def = 0.5
        # self.prior_blend_att = 0.5

        # self.prior_blend_def = 0.1
        # self.prior_blend_att = 0.5

        self.prior_blend_def = 0.0
        self.prior_blend_att = 0.0

    def dist(self, obs: torch.Tensor, who: str):
        feats, u_prior = self.layer(obs, who)
        h = self.pi(feats)
        mu_res = self.mu_res(h)

        blend = self.prior_blend_def if who == "def" else self.prior_blend_att
        mu = mu_res + blend * u_prior

        std = self.logstd.exp()
        # std = self.logstd.clamp(-5.0, 1.0).exp()   # e.g. std in ~[0.0067, 2.7]

        return torch.distributions.Normal(mu, std)

    def value(self, obs: torch.Tensor):
        return self.vf(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, who: str, act_scale: float):
        dist = self.dist(obs, who)
        u_raw = dist.rsample()
        a_env = squash_action(u_raw, act_scale)
        logp  = logprob_squashed(dist, u_raw)
        val   = self.value(obs)
        return a_env, logp, val
