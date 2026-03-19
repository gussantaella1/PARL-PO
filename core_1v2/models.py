from __future__ import annotations

import importlib
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from core_1v2.utils import _obs_offsets
from core.utils import logprob_squashed, squash_action


class DiffLSLayer(nn.Module):
    """
    Analytic one-step ridge prior using the discrete dynamics in cfg["dyn"].

    For attacker policies in the 1v2 case, the observation is assumed to be
    attacker-ego permuted so attacker slot 0 corresponds to "me".
    """

    def __init__(self, obs_dim: int, cfg: Dict[str, Any]):
        super().__init__()
        self.D = int(cfg["D"])
        self.Na = int(cfg.get("num_attackers", 1))
        self.ridge = float(cfg.get("prior_ridge", 1e-2))
        self.use_fuel = bool(cfg.get("fuel", {}).get("enable", False))

        Ad = np.asarray(cfg["dyn"]["Ad"], dtype=np.float32)
        Bd = np.asarray(cfg["dyn"]["Bd"], dtype=np.float32)
        if Ad.shape != (2 * self.D, 2 * self.D):
            raise ValueError(f"Expected Ad shape {(2*self.D, 2*self.D)}, got {Ad.shape}")
        if Bd.shape != (2 * self.D, self.D):
            raise ValueError(f"Expected Bd shape {(2*self.D, self.D)}, got {Bd.shape}")

        self.register_buffer("Ad", torch.tensor(Ad, dtype=torch.float32))
        self.register_buffer("Bd", torch.tensor(Bd, dtype=torch.float32))

        P = np.hstack(
            [
                np.eye(self.D, dtype=np.float32),
                np.zeros((self.D, self.D), dtype=np.float32),
            ]
        )
        self.register_buffer("P", torch.tensor(P, dtype=torch.float32))

        F = P @ Bd
        M = F.T @ F + self.ridge * np.eye(self.D, dtype=np.float32)
        self.register_buffer("K", torch.tensor(np.linalg.solve(M, F.T), dtype=torch.float32))

        self.feature_dim = 4 * self.D + (2 if self.use_fuel else 0) if self.Na == 1 else obs_dim

    def _split_obs(self, obs: torch.Tensor):
        D, Na = self.D, self.Na
        base_dim = (2 + 3 * Na) * D
        expected = base_dim + (2 if self.use_fuel else 0)
        if obs.shape[-1] != expected:
            raise ValueError(
                f"Expected obs dim {expected}, got {obs.shape[-1]}. "
                "Check prior layer vs env observation layout."
            )

        off_p1, off_pA, off_rel, off_v1, off_vA = _obs_offsets(D, Na)
        p1c = obs[:, off_p1:off_p1 + D]
        p2c = obs[:, off_pA:off_pA + D]
        v1 = obs[:, off_v1:off_v1 + D]
        v2 = obs[:, off_vA:off_vA + D]

        if self.use_fuel:
            fdef = obs[:, base_dim:base_dim + 1]
            fatt = obs[:, base_dim + 1:base_dim + 2]
        else:
            fdef = None
            fatt = None
        return p1c, p2c, v1, v2, fdef, fatt

    def _one_step_prior(self, p_c: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        x = torch.cat([p_c, v], dim=-1)
        E = x @ self.Ad.T @ self.P.T
        return -(E @ self.K.T)

    def forward(self, obs: torch.Tensor, who: str):
        p1c, p2c, v1, v2, fdef, fatt = self._split_obs(obs)
        u_def_prior = self._one_step_prior(p1c, v1)
        u_att_prior = self._one_step_prior(p2c, v2)
        u_prior = u_def_prior if who == "def" else u_att_prior

        if self.Na == 1:
            feats = torch.cat([p1c, p2c, v1, v2], dim=-1)
            if self.use_fuel:
                feats = torch.cat([feats, fdef, fatt], dim=-1)
        else:
            feats = obs
        return feats, u_prior


class DiffNashLayer(nn.Module):
    """
    Legacy Nash prior retained for single-attacker compatibility.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.D = int(cfg["D"])
        self.Na = int(cfg.get("num_attackers", 1))
        if self.Na != 1:
            raise ValueError("prior_type='nash' only supports num_attackers=1")

        self.register_buffer(
            "Ad",
            torch.as_tensor(np.asarray(cfg["dyn"]["Ad"], np.float32), dtype=torch.float32),
        )
        self.register_buffer(
            "Bd",
            torch.as_tensor(np.asarray(cfg["dyn"]["Bd"], np.float32), dtype=torch.float32),
        )

        ar = cfg["arena"]
        c = np.array(
            [ar["cx"], ar["cy"], (ar["cz"] if self.D == 3 else 0.0)],
            dtype=np.float32,
        )[: self.D]
        self.register_buffer("center", torch.tensor(c, dtype=torch.float32))

        solver_cfg = cfg.get("nash_solver", {})
        module_name = solver_cfg.get("module", "nash_ipopt_solver")
        fn_name = solver_cfg.get("fn", "solve_nash_ipopt")
        self.solver_params = solver_cfg.get("params", {})

        mod = importlib.import_module(module_name)
        self.nash_solve = getattr(mod, fn_name)
        self.feature_dim = 4 * self.D

    def forward(self, obs: torch.Tensor, who: str):
        B, D = obs.shape[0], self.D
        device = obs.device
        dtype = obs.dtype

        center = self.center.to(dtype=dtype, device=device)
        p1c = obs[:, 0:D]
        p2c = obs[:, D:2 * D]
        v1 = obs[:, 3 * D:4 * D]
        v2 = obs[:, 4 * D:5 * D]

        x1 = torch.cat([p1c + center, v1], dim=-1)
        x2 = torch.cat([p2c + center, v2], dim=-1)

        x1_np = x1.detach().cpu().numpy()
        x2_np = x2.detach().cpu().numpy()

        u1_list = []
        u2_list = []
        for i in range(B):
            u1_i, u2_i = self.nash_solve(x1_np[i], x2_np[i], self.solver_params)
            u1_list.append(np.asarray(u1_i, dtype=np.float32))
            u2_list.append(np.asarray(u2_i, dtype=np.float32))

        u1_prior = torch.as_tensor(np.stack(u1_list, axis=0), device=device, dtype=dtype)
        u2_prior = torch.as_tensor(np.stack(u2_list, axis=0), device=device, dtype=dtype)
        feats = torch.cat([p1c, p2c, v1, v2], dim=-1)
        return feats, u1_prior if who == "def" else u2_prior


class NoPriorLayer(nn.Module):
    def __init__(self, obs_dim: int, cfg: Dict[str, Any]):
        super().__init__()
        self.D = int(cfg["D"])
        self.Na = int(cfg.get("num_attackers", 1))
        self.use_fuel = bool(cfg.get("fuel", {}).get("enable", False))
        self.feature_dim = 4 * self.D + (2 if self.use_fuel else 0) if self.Na == 1 else obs_dim

    def _split_obs(self, obs: torch.Tensor):
        D, Na = self.D, self.Na
        base_dim = (2 + 3 * Na) * D
        expected = base_dim + (2 if self.use_fuel else 0)
        if obs.shape[-1] != expected:
            raise ValueError(
                f"Expected obs dim {expected}, got {obs.shape[-1]}. "
                "Check prior layer vs env observation layout."
            )

        off_p1, off_pA, off_rel, off_v1, off_vA = _obs_offsets(D, Na)
        p1c = obs[:, off_p1:off_p1 + D]
        p2c = obs[:, off_pA:off_pA + D]
        v1 = obs[:, off_v1:off_v1 + D]
        v2 = obs[:, off_vA:off_vA + D]

        if self.use_fuel:
            fdef = obs[:, base_dim:base_dim + 1]
            fatt = obs[:, base_dim + 1:base_dim + 2]
        else:
            fdef = None
            fatt = None
        return p1c, p2c, v1, v2, fdef, fatt

    def forward(self, obs: torch.Tensor, who: str):
        B = obs.shape[0]
        p1c, p2c, v1, v2, fdef, fatt = self._split_obs(obs)
        if self.Na == 1:
            feats = torch.cat([p1c, p2c, v1, v2], dim=-1)
            if self.use_fuel:
                feats = torch.cat([feats, fdef, fatt], dim=-1)
        else:
            feats = obs

        u_prior = torch.zeros((B, self.D), device=obs.device, dtype=obs.dtype)
        return feats, u_prior


class ActorCriticDiff(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg: Dict[str, Any]):
        super().__init__()
        hidden = 128

        prior_type = cfg.get("prior_type", "ls")
        if prior_type == "ls":
            self.layer = DiffLSLayer(obs_dim, cfg)
        elif prior_type == "nash":
            self.layer = DiffNashLayer(cfg)
        elif prior_type == "none":
            self.layer = NoPriorLayer(obs_dim, cfg)
        else:
            raise ValueError(
                f"Unknown prior_type={prior_type!r}, expected 'ls', 'nash', or 'none'."
            )

        feat_dim = self.layer.feature_dim
        self.pi = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu_res = nn.Linear(hidden, act_dim)
        self.logstd = nn.Parameter(torch.full((act_dim,), -1.0))
        self.vf = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        self.prior_blend_def = float(cfg.get("prior_blend_def", 0.0))
        self.prior_blend_att = float(cfg.get("prior_blend_att", 0.0))

    def dist(self, obs: torch.Tensor, who: str):
        feats, u_prior = self.layer(obs, who)
        h = self.pi(feats)
        mu_res = self.mu_res(h)
        blend = self.prior_blend_def if who == "def" else self.prior_blend_att
        mu = mu_res + blend * u_prior
        return torch.distributions.Normal(mu, self.logstd.exp())

    def value(self, obs: torch.Tensor):
        return self.vf(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, who: str, act_scale: float):
        dist = self.dist(obs, who)
        u_raw = dist.rsample()
        a_env = squash_action(u_raw, act_scale)
        logp = logprob_squashed(dist, u_raw)
        val = self.value(obs)
        return a_env, logp, val


__all__ = ["ActorCriticDiff", "DiffLSLayer", "DiffNashLayer", "NoPriorLayer"]
