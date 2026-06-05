"""
Actor-critic policy networks and optional differentiable priors used by PPO and inference.
"""

import math
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from core.intercept_heuristic import clamp_intercept_mix
from core.utils import logprob_squashed, squash_action
# =============================================================
# DiffLS Layer & Actor-Critic
# =============================================================
def _obs_extra_dim_from_cfg(cfg: Dict[str, Any]) -> int:
    """Internal helper for obs extra dim from cfg."""
    return 0


class _SingleAttackerObsLayer(nn.Module):
    """Feature adapter that splits single-attacker observations into policy-friendly tensors."""
    def __init__(self, cfg: Dict[str, Any]):
        """Store configuration and initialize the runtime state for this object."""
        super().__init__()
        self.D = int(cfg["D"])
        self.use_fuel = bool(cfg.get("fuel", {}).get("enable", False))
        self.num_attackers = int(cfg.get("num_attackers", 1))
        self.extra_obs_dim = _obs_extra_dim_from_cfg(cfg)

        if self.num_attackers != 1:
            raise NotImplementedError(
                f"{self.__class__.__name__} currently supports only num_attackers=1."
            )

        self.feature_dim = 4 * self.D + (2 if self.use_fuel else 0) + self.extra_obs_dim

    def _split_obs(self, obs: torch.Tensor):
        """Internal helper for split obs."""
        D = self.D
        base_dim = 5 * D
        expected = base_dim + (2 if self.use_fuel else 0) + self.extra_obs_dim
        if obs.shape[-1] != expected:
            raise ValueError(
                f"Expected obs dim {expected}, got {obs.shape[-1]}. "
                f"Check prior layer vs env observation layout."
            )

        p1c = obs[:, 0:D]
        p2c = obs[:, D:2 * D]
        rel = obs[:, 2 * D:3 * D]
        v1 = obs[:, 3 * D:4 * D]
        v2 = obs[:, 4 * D:5 * D]

        if self.use_fuel:
            fdef = obs[:, base_dim:base_dim + 1]
            fatt = obs[:, base_dim + 1:base_dim + 2]
            extras = obs[:, base_dim + 2:]
        else:
            fdef = None
            fatt = None
            extras = obs[:, base_dim:]

        return p1c, p2c, rel, v1, v2, fdef, fatt, extras

    def _pack_feats(
        self,
        p1c: torch.Tensor,
        p2c: torch.Tensor,
        v1: torch.Tensor,
        v2: torch.Tensor,
        fdef: torch.Tensor | None,
        fatt: torch.Tensor | None,
        extras: torch.Tensor,
    ) -> torch.Tensor:
        """Pack feats into the dictionary format expected by downstream plotting and metrics."""
        if self.use_fuel:
            return torch.cat([p1c, p2c, v1, v2, fdef, fatt, extras], dim=-1)
        return torch.cat([p1c, p2c, v1, v2, extras], dim=-1)


class DiffLSLayer(_SingleAttackerObsLayer):
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
        """Store configuration and initialize the runtime state for this object."""
        super().__init__(cfg)
        self.ridge = float(cfg.get("prior_ridge", 1e-2))

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
        """Run the module forward pass."""
        p1c, p2c, rel, v1, v2, fdef, fatt, extras = self._split_obs(obs)

        u_def_prior = self._one_step_prior(p1c, v1)
        u_att_prior = self._one_step_prior(p2c, v2)
        u_prior = u_def_prior if who == "def" else u_att_prior

        return self._pack_feats(p1c, p2c, v1, v2, fdef, fatt, extras), u_prior


class InterceptPriorLayer(_SingleAttackerObsLayer):
    """
    Direct intercept heuristic for the defender:
      1. Predict a coasting attacker position after a lookahead horizon.
      2. Blend that with the attacker's current position.
      3. Point the defender toward that intercept target with fixed raw gain.

    Observation format assumed (single-attacker case):
        obs = [p1c, p2c, rel, v1, v2]                 if fuel disabled
        obs = [p1c, p2c, rel, v1, v2, f_def, f_att]   if fuel enabled
    """
    def __init__(self, cfg: Dict[str, Any]):
        """Store configuration and initialize the runtime state for this object."""
        super().__init__(cfg)
        Ad = np.asarray(cfg["dyn"]["Ad"], dtype=np.float32)
        if Ad.shape != (2 * self.D, 2 * self.D):
            raise ValueError(
                f"Expected Ad shape {(2*self.D, 2*self.D)}, got {Ad.shape}"
            )
        self.register_buffer("Ad", torch.tensor(Ad, dtype=torch.float32))

        intercept_cfg = dict(cfg.get("intercept_prior", {}) or {})
        self.intercept_lookahead_steps = max(0.0, float(intercept_cfg.get("lookahead_steps", 0.0)))
        self.intercept_mix = clamp_intercept_mix(intercept_cfg.get("mix", 1.0))
        self.intercept_gain = float(intercept_cfg.get("gain", 2.0))

    def _rollout_coasting_state(self, x: torch.Tensor, steps: float) -> torch.Tensor:
        """Internal helper for rollout coasting state."""
        steps_f = max(0.0, float(steps))
        lo = int(math.floor(steps_f))
        hi = int(math.ceil(steps_f))

        x_lo = x
        for _ in range(lo):
            x_lo = x_lo @ self.Ad.T

        if hi == lo:
            return x_lo

        x_hi = x_lo @ self.Ad.T
        alpha = steps_f - float(lo)
        return (1.0 - alpha) * x_lo + alpha * x_hi

    def _intercept_target(self, p_att_c: torch.Tensor, v_att: torch.Tensor) -> torch.Tensor:
        """Internal helper for intercept target."""
        if self.intercept_mix <= 0.0:
            return p_att_c

        x_att = torch.cat([p_att_c, v_att], dim=-1)
        x_pred = self._rollout_coasting_state(x_att, self.intercept_lookahead_steps)
        p_pred = x_pred[:, : self.D]
        if self.intercept_mix >= 1.0:
            return p_pred
        mix = float(self.intercept_mix)
        return (1.0 - mix) * p_att_c + mix * p_pred

    def _defender_intercept_prior(
        self,
        p_def_c: torch.Tensor,
        p_att_c: torch.Tensor,
        v_att: torch.Tensor,
    ) -> torch.Tensor:
        """Internal helper for defender intercept prior."""
        target = self._intercept_target(p_att_c, v_att)
        delta = target - p_def_c
        delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return (self.intercept_gain * delta) / delta_norm

    def forward(self, obs: torch.Tensor, who: str):
        """Run the module forward pass."""
        p1c, p2c, rel, v1, v2, fdef, fatt, extras = self._split_obs(obs)
        u_def_prior = self._defender_intercept_prior(p1c, p2c, v2)
        u_att_prior = torch.zeros_like(u_def_prior)
        u_prior = u_def_prior if who == "def" else u_att_prior
        return self._pack_feats(p1c, p2c, v1, v2, fdef, fatt, extras), u_prior


class NoPriorLayer(_SingleAttackerObsLayer):
    """
    No analytic prior: just repackage the observation into features
    and return u_prior = 0.

    Observation format assumed (single-attacker case):
        obs = [p1c, p2c, rel, v1, v2]                 if fuel disabled
        obs = [p1c, p2c, rel, v1, v2, f_def, f_att]   if fuel enabled
    """
    def __init__(self, cfg: Dict[str, Any]):
        """Store configuration and initialize the runtime state for this object."""
        super().__init__(cfg)

    def forward(self, obs: torch.Tensor, who: str):
        """Run the module forward pass."""
        B, D = obs.shape[0], self.D
        device, dtype = obs.device, obs.dtype

        p1c, p2c, rel, v1, v2, fdef, fatt, extras = self._split_obs(obs)

        u_prior = torch.zeros((B, D), device=device, dtype=dtype)
        return self._pack_feats(p1c, p2c, v1, v2, fdef, fatt, extras), u_prior




class ActorCriticDiff(nn.Module):
    """Actor-critic network with optional differentiable prior layers for action means."""
    def __init__(self, obs_dim: int, act_dim: int, cfg: Dict[str, Any]):
        """Store configuration and initialize the runtime state for this object."""
        super().__init__()
        hidden = 128

        # Choose which prior layer to use
        prior_type = cfg.get("prior_type", "ls")  # "ls", "intercept", or "none"
        if prior_type == "ls":
            self.layer = DiffLSLayer(cfg)
        elif prior_type == "intercept":
            self.layer = InterceptPriorLayer(cfg)
        elif prior_type == "none":
            self.layer = NoPriorLayer(cfg)
        else:
            raise ValueError(
                f"Unknown prior_type={prior_type!r}, expected 'ls', 'intercept', or 'none'."
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
        self.prior_blend_def = float(cfg.get("prior_blend_def", 0.0))
        self.prior_blend_att = float(cfg.get("prior_blend_att", 0.0))

    def set_prior_blend(self, who: str, value: float) -> None:
        """Handle set prior blend for this workflow."""
        blend = clamp_intercept_mix(value)
        if who == "def":
            self.prior_blend_def = blend
            return
        if who == "att":
            self.prior_blend_att = blend
            return
        raise ValueError(f"Unknown role for prior blend: {who!r}")

    def get_prior_blend(self, who: str) -> float:
        """Handle get prior blend for this workflow."""
        if who == "def":
            return float(self.prior_blend_def)
        if who == "att":
            return float(self.prior_blend_att)
        raise ValueError(f"Unknown role for prior blend: {who!r}")

    def dist(self, obs: torch.Tensor, who: str):
        """Build the policy action distribution for a batch of observations."""
        feats, u_prior = self.layer(obs, who)
        h = self.pi(feats)
        mu_res = self.mu_res(h)

        blend = self.prior_blend_def if who == "def" else self.prior_blend_att
        mu = mu_res + blend * u_prior

        std = self.logstd.exp()
        # std = self.logstd.clamp(-5.0, 1.0).exp()   # e.g. std in ~[0.0067, 2.7]

        return torch.distributions.Normal(mu, std)

    def value(self, obs: torch.Tensor):
        """Estimate the value function for a batch of observations."""
        return self.vf(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, who: str, act_scale: float):
        """Choose an action for the current observation or state."""
        dist = self.dist(obs, who)
        u_raw = dist.rsample()
        a_env = squash_action(u_raw, act_scale)
        logp  = logprob_squashed(dist, u_raw)
        val   = self.value(obs)
        return a_env, logp, val
