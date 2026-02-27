#!/usr/bin/env python3
"""
rl_loop.py
===================================
Single-file training & evaluation where **only the defender is learned (PPO)**
and the **attacker is a deterministic rule-based controller** that (a) drives to the
center and (b) repels away from the defender under HCW dynamics.

Key points
----------
- Single source of truth for config via: from config_rl import config_for_train, config_for_eval, build_dyn
- Clean attacker swap: set cfg["attacker_mode"] to "rule" (default) or "rl"
- Differentiable one-step ridge prior (DiffLS-style) blended into actor mean
- Minimal, single-process VecEnv for reproducible, fixed-length rollouts

PATCH NOTE (1v2):
-----------------
This file is adapted to 1 defender vs N attackers (N=2 is your target), where:
- attackers share the same policy (shared weights)
- training logic/print statements/optimizer logic remain the same
- only the multi-attacker plumbing is changed:
  * Env rewards computed per-attacker (attacker reward averaged for backward-compat r2)
  * defender shaping uses the "most dangerous" attacker = closest to center
  * PPO attacker action returns (B, Na, D) when Na>1
  * attacker training aggregates experience across all attackers via obs permutation

Set cfg["num_attackers"]=2 to enable 1v2.
"""

import importlib
import os
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import gc
import time

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import math
import random
from dataclasses import dataclass

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import json

from pathlib import Path

# --- Single source of truth for config & dynamics ---
import config_rl
importlib.reload(config_rl)
from config_rl import config_for_train, config_for_eval, build_dyn

from ukf_estimator import AgentUKF, _body_bearing_from_world, _azel_from_body_vec


# =============================================================
# Utils
# =============================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def atanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -0.999999, 0.999999)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def squash_action(u_raw: torch.Tensor, act_scale: float) -> torch.Tensor:
    return torch.tanh(u_raw) * act_scale


def logprob_squashed(dist: torch.distributions.Normal, u_raw: torch.Tensor) -> torch.Tensor:
    # log p(tanh(u)) = log p(u) - sum log(1 - tanh(u)^2)
    logp = dist.log_prob(u_raw).sum(-1)
    correction = torch.log1p(-torch.tanh(u_raw).pow(2) + 1e-8).sum(-1)
    return logp - correction


# =============================================================
# NEW (1vN): Obs parsing + attacker-ego permutation
# =============================================================

def _obs_offsets(D: int, Na: int):
    """
    Obs layout (Env._obs):
      [ p1c,
        pA_c[0..Na-1],
        rel[0..Na-1],
        v1,
        vA[0..Na-1] ]
    Total dim = (2 + 3*Na)*D
    """
    off_p1 = 0
    off_pA = D
    off_rel = D + Na * D
    off_v1 = D + 2 * Na * D
    off_vA = off_v1 + D
    return off_p1, off_pA, off_rel, off_v1, off_vA


def permute_obs_for_attacker(obs: torch.Tensor, idx: int, D: int, Na: int) -> torch.Tensor:
    """
    Returns an obs where attacker `idx` is placed into attacker-slot 0
    (and all other attackers keep relative order after it).

    This lets one shared attacker policy treat "me" as attacker0.
    """
    if Na <= 1 or idx == 0:
        return obs

    off_p1, off_pA, off_rel, off_v1, off_vA = _obs_offsets(D, Na)

    p1c = obs[:, off_p1:off_p1 + D]

    pA = [obs[:, off_pA + k * D: off_pA + (k + 1) * D] for k in range(Na)]
    rel = [obs[:, off_rel + k * D: off_rel + (k + 1) * D] for k in range(Na)]

    v1 = obs[:, off_v1:off_v1 + D]
    vA = [obs[:, off_vA + k * D: off_vA + (k + 1) * D] for k in range(Na)]

    order = [idx] + [k for k in range(Na) if k != idx]

    parts = [p1c]
    parts += [pA[k] for k in order]
    parts += [rel[k] for k in order]
    parts += [v1]
    parts += [vA[k] for k in order]

    return torch.cat(parts, dim=-1)


# =============================================================
# Environment (1 defender + Na attackers under HCW)
# =============================================================

class Env:
    """
    State per agent: [px,py,pz,vx,vy,vz]; s = concat(defender, attackers...)
    Observation: [p1-center, pA-center, (pA-p1), v1, vA] stacked over attackers.

    Rewards:
      - Defender shaping uses the "most dangerous" attacker = closest to center.
      - Attacker reward is computed per attacker; step() returns r2 = mean(r_att_each)
        for backward compatibility with existing logging/training loop.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.num_attackers = int(cfg.get("num_attackers", 1))
        Na = self.num_attackers

        self.D = int(cfg["D"])
        self.dt = float(cfg["dt"])
        self.T = int(cfg["T"])

        ar = cfg["arena"]
        if ar["type"] != "sphere":
            raise ValueError("Only 'sphere' arena is implemented.")
        self.center = np.array([ar["cx"], ar["cy"], (ar["cz"] if self.D == 3 else 0.0)], dtype=np.float32)[:self.D]
        self.radius = float(ar["r"])

        self.def_keepout_buffer_m = float(cfg.get("def_keepout_buffer_m", 0.0))

        umax = float(cfg["umax"])
        self.u_lo, self.u_hi = -umax, +umax

        Ad = cfg["dyn"]["Ad"]
        Bd = cfg["dyn"]["Bd"]
        if Ad is None or Bd is None:
            raise ValueError("cfg['dyn']['Ad'] and ['Bd'] must be provided (call build_dyn(cfg)).")
        self.Ad = np.asarray(Ad, dtype=np.float32)
        self.Bd = np.asarray(Bd, dtype=np.float32)

        self.nx_agent = 2 * self.D
        self.nx_total = (1 + Na) * self.nx_agent
        self.act_dim = self.D

        x0 = np.asarray(cfg["x0"], dtype=float)
        if x0.shape[0] == 2 and self.num_attackers > 1:
            base_def = x0[0]
            base_att = x0[1]
            x0_full = np.zeros((1 + Na, 2 * self.D), dtype=float)
            x0_full[0] = base_def
            x0_full[1] = base_att
        elif x0.shape[0] == 1 + self.num_attackers:
            x0_full = x0
        else:
            raise ValueError(f"x0 shape {x0.shape} incompatible with num_attackers={self.num_attackers}")
        self._x0 = x0_full

        # reward params
        self.alpha = float(cfg["dense_coef"])
        self.beta = float(cfg["term_coef"])
        self.k_pos = float(cfg.get("step_pos_coef", 0.0))
        self.k_rel = float(cfg.get("step_rel_coef", 0.0))
        self.k_cent_base = float(cfg.get("def_center_coef", 0.0))
        self.k_cent = self.k_cent_base
        self.k_vel = float(cfg.get("step_vel_coef", 0.0))
        self.lD = float(cfg["effort_def"])
        self.lA = float(cfg["effort_att"])
        self.wallK = float(cfg["wall_penalty"])
        self.soft_wall = float(cfg.get("soft_wall_start", 0.7))
        self.margin = float(cfg["arena_terminate_margin"])

        self.def_center_safe_radius = float(cfg.get("def_center_safe_radius", 0.05))
        self.def_center_avoid_coef = float(cfg.get("def_center_avoid_coef", 10.0))

        oi = cfg.get("oi", {})
        self.oi_radius = float(oi.get("r", 0.0))
        self.oi_radius_norm = self.oi_radius / self.radius if self.radius > 0 else 0.0

        self.att_target_hit_radius = float(cfg.get("att_target_hit_radius", 0.0))
        self.att_target_hit_penalty_def = float(cfg.get("att_target_hit_penalty_def", 0.0))
        self.att_target_hit_reward_att = float(cfg.get("att_target_hit_reward_att", 0.0))

        self.def_oi_safety_buffer = float(cfg.get("def_oi_safety_buffer", 0.0))
        self.def_target_hit_penalty_def = float(cfg.get("def_target_hit_penalty_def", 0.0))
        self.def_target_hit_reward_att = float(cfg.get("def_target_hit_reward_att", 0.0))

        self.collision_radius_m = float(cfg.get("collision_radius_m", 0.2))
        self.collision_penalty_def = float(cfg.get("collision_penalty_def", 0.0))
        self.collision_penalty_att = float(cfg.get("collision_penalty_att", 0.0))

        # ---- UKF / measurement model knobs ----
        self.use_ukf = bool(cfg.get("use_ukf", False))
        self.use_meas_reward = bool(cfg.get("use_meas_reward", False))
        self.meas_innov_coef = float(cfg.get("meas_innov_coef", 0.0))
        self.meas_cov_coef = float(cfg.get("meas_cov_coef", 0.0))

        if self.use_ukf and self.D != 3:
            raise ValueError("UKF / bearing-only measurement currently implemented for D=3 only.")

        self.ukf = None
        self._latest_meas_innov = 0.0
        self._latest_meas_trP = 0.0

        if self.use_ukf:
            ukf_cfg = cfg.get("ukf", {})

            pos_std0 = float(ukf_cfg.get("pos_std0", 0.2 * self.radius))
            vel_std0 = float(ukf_cfg.get("vel_std0", 0.01))
            P0 = np.diag(
                [pos_std0**2, pos_std0**2, pos_std0**2,
                 vel_std0**2, vel_std0**2, vel_std0**2]
            )

            q_scale = float(ukf_cfg.get("Q_scale", 1e-5))
            Q = q_scale * np.eye(6, dtype=float)

            sigma_az = float(ukf_cfg.get("sigma_az", np.deg2rad(0.5)))
            sigma_el = float(ukf_cfg.get("sigma_el", np.deg2rad(0.5)))
            Rm = np.diag([sigma_az**2, sigma_el**2])

            self._ukf_P0 = P0
            self._ukf_Q = Q
            self._ukf_R = Rm

            self._ukf_init_pos_std = float(ukf_cfg.get("init_pos_std", pos_std0))
            self._ukf_init_vel_std = float(ukf_cfg.get("init_vel_std", vel_std0))

        # Training-only initial-condition randomization
        self.train_ic_mode = cfg.get("train_ic_mode", "fixed")
        self.train_ic_vmax = float(cfg.get("train_ic_vmax", 0.05))
        self.train_min_sep = float(cfg.get("train_min_sep", 1.0))

        self.state = None
        self.t = 0

        # (1vN) per-attacker d2_prev
        self._d2_prev = np.zeros((self.num_attackers,), dtype=np.float32)

        # attacker reward knobs
        att = cfg.get("att_reward", {})
        att_rule = cfg.get("att_rule", {})
        self.k_att_prog = float(att.get("k_prog", 2.0))
        self.k_att_cent = float(att.get("k_cent", 0.0))
        self.k_att_close = float(att.get("k_close", 2.0))
        self.att_min_sep = float(att.get("min_sep", att_rule.get("min_sep", 3.0)))
        self.k_att_vrad = float(att.get("k_vrad", 0.5))
        self.k_att_wall = float(att.get("k_wall", self.wallK))
        self.att_wall_power = float(att.get("wall_power", 4.0))

        self.hit_buffer_def = float(self.def_oi_safety_buffer)
        self.hit_buffer_att = float(self.att_target_hit_radius)

    def reset(self) -> np.ndarray:
        self.t = 0
        mode = self.train_ic_mode
        Na = self.num_attackers

        if mode == "fixed":
            x0 = self._x0.copy()
            jit = self.cfg.get("x0_jitter", None)
            if jit:
                jp = float(jit.get("pos", 0.0))
                jv = float(jit.get("vel", 0.0))
                x0[0, 0:self.D] += np.random.uniform(-jp, jp, size=(self.D,))
                x0[0, self.D:2*self.D] += np.random.uniform(-jv, jv, size=(self.D,))
                for k in range(Na):
                    idx = 1 + k
                    x0[idx, 0:self.D] += np.random.uniform(-jp, jp, size=(self.D,))
                    x0[idx, self.D:2*self.D] += np.random.uniform(-jv, jv, size=(self.D,))

        elif mode == "random_shell":
            R = self.radius
            v_max = self.train_ic_vmax
            min_sep = self.train_min_sep

            def sample_in_ball(r_min, r_max):
                d = np.random.normal(size=(self.D,))
                d /= (np.linalg.norm(d) + 1e-9)
                u = np.random.rand()
                r = (r_min**3 + (r_max**3 - r_min**3) * u) ** (1.0 / 3.0)
                return self.center + r * d

            r_def_min, r_def_max = 0.0, 0.5 * R
            r_att_min, r_att_max = 0.4 * R, 0.95 * R

            p1 = sample_in_ball(r_def_min, r_def_max)
            v1 = np.random.uniform(-v_max, v_max, size=(self.D,))

            pA = []
            vA = []
            for k in range(Na):
                for _ in range(1000):
                    pk = sample_in_ball(r_att_min, r_att_max)
                    if np.linalg.norm(pk - p1) < min_sep:
                        continue
                    if any(np.linalg.norm(pk - pj) < min_sep for pj in pA):
                        continue
                    break
                pA.append(pk)
                vA.append(np.random.uniform(-v_max, v_max, size=(self.D,)))

            x0 = np.zeros_like(self._x0)
            x0[0, 0:self.D] = p1
            x0[0, self.D:2*self.D] = v1
            for k in range(Na):
                idx = 1 + k
                x0[idx, 0:self.D] = pA[k]
                x0[idx, self.D:2*self.D] = vA[k]

        else:
            raise ValueError(f"Unknown train_ic_mode='{mode}'")

        self.state = x0.reshape(-1)

        p1, v1, pA_list, vA_list = self._unpack(self.state)

        # ---- UKF init (only supported for Na==1 in this codebase) ----
        if self.use_ukf and self.num_attackers != 1:
            raise NotImplementedError("UKF currently only implemented for num_attackers=1")

        if self.use_ukf:
            p2 = pA_list[0]
            v2 = vA_list[0]

            pos_noise = np.random.normal(scale=self._ukf_init_pos_std, size=p2.shape)
            vel_noise = np.random.normal(scale=self._ukf_init_vel_std, size=v2.shape)
            p2_est = p2 + pos_noise
            v2_est = v2 + vel_noise
            x0_ukf = np.concatenate([p2_est, v2_est])

            self.ukf = AgentUKF(
                x0=x0_ukf,
                P0=self._ukf_P0.copy(),
                Q=self._ukf_Q.copy(),
                R=self._ukf_R.copy(),
                dt=self.dt,
                dyn='hcw',
                hcw=self.cfg.get("hcw", {}),
            )

            self._latest_meas_innov = 0.0
            self._latest_meas_trP = float(np.trace(self.ukf.P[0:3, 0:3]))
        else:
            self.ukf = None
            self._latest_meas_innov = 0.0
            self._latest_meas_trP = 0.0

        # ---- initialize d2_prev per attacker ----
        if self.use_ukf and (self.ukf is not None):
            p2_geom = self.ukf.x[:self.D].copy()
            d2 = float(np.dot(p2_geom - self.center, p2_geom - self.center)) / (self.radius**2)
            self._d2_prev[:] = d2
        else:
            d2_each = []
            for pA in pA_list:
                d2k = float(np.dot(pA - self.center, pA - self.center)) / (self.radius**2)
                d2_each.append(d2k)
            self._d2_prev[:] = np.asarray(d2_each, dtype=np.float32)

        return self._obs()

    def step(self, a1_env: np.ndarray, aA_env: np.ndarray, reward_mode: str = "both"):
        """
        a1_env: (D,)
        aA_env: (Na, D) or (D,) if Na=1
        reward_mode: "def", "att", or "both"
        """
        need_def = reward_mode in ("def", "both")
        need_att = reward_mode in ("att", "both")

        a1 = np.clip(np.asarray(a1_env, float).reshape(self.D,), self.u_lo, self.u_hi)

        aA = np.asarray(aA_env, float)
        if aA.ndim == 1:
            aA = aA.reshape(1, self.D)
        else:
            aA = aA.reshape(self.num_attackers, self.D)
        aA = np.clip(aA, self.u_lo, self.u_hi)

        # propagate true state
        self.state = self._plant_step(self.state, a1, aA)
        self.t += 1

        p1, v1, pA_list, vA_list = self._unpack(self.state)

        # ---- UKF (still only Na==1 supported) ----
        meas_innov_sq = 0.0
        meas_trPpos = 0.0

        if self.use_ukf:
            p2 = pA_list[0]
            self.ukf.predict(dt=self.dt, u=None, u_cov=None)
            p_obs = p1
            if self.D != 3:
                raise RuntimeError("UKF bearing logic assumes D=3.")
            R_wb = np.eye(3)

            v_b = _body_bearing_from_world(p_obs, R_wb, p2)
            az_true, el_true = _azel_from_body_vec(v_b)
            z_true = np.array([az_true, el_true], float)

            z_noise = np.random.multivariate_normal(mean=np.zeros(2), cov=self._ukf_R)
            z_meas = z_true + z_noise

            z_hat_prior = self.ukf.h(self.ukf.x.copy(), p_obs, R_wb)
            innov = z_meas - z_hat_prior
            innov[0] = (innov[0] + np.pi) % (2*np.pi) - np.pi
            innov[1] = (innov[1] + np.pi) % (2*np.pi) - np.pi
            meas_innov_sq = float(innov @ innov)

            self.ukf.update(z_meas, p_obs, R_wb)

            meas_trPpos = float(np.trace(self.ukf.P[0:3, 0:3]))
            self._latest_meas_innov = meas_innov_sq
            self._latest_meas_trP = meas_trPpos
        else:
            self._latest_meas_innov = 0.0
            self._latest_meas_trP = 0.0

        # ---- geometry for all attackers ----
        R2 = (self.radius ** 2)
        v_scale = self.radius / self.dt

        d2_each = np.array(
            [float(np.dot(pA - self.center, pA - self.center)) / R2 for pA in pA_list],
            dtype=np.float32
        )
        rel_each = np.array(
            [float(np.dot(pA - p1, pA - p1)) / R2 for pA in pA_list],
            dtype=np.float32
        )

        # "most dangerous" attacker: closest to center
        k_star = int(np.argmin(d2_each))
        d2 = float(d2_each[k_star])
        rel2 = float(rel_each[k_star])

        # delta for that same attacker index (keeps continuity)
        delta_d2 = float(d2_each[k_star] - self._d2_prev[k_star])

        # defender distance (only if needed)
        if need_def:
            d1_raw = float(np.dot(p1 - self.center, p1 - self.center))
            d1 = d1_raw / R2
        else:
            d1 = 0.0

        # ---- wall / keepout / effort ----
        rho1 = np.linalg.norm(p1 - self.center) / self.radius
        wall1 = ((max(0.0, rho1 - self.soft_wall))**2) * self.wallK if need_def else 0.0

        center_keepout = 0.0
        if need_def and self.oi_radius > 0.0:
            r_keepout_m = self.oi_radius + self.def_keepout_buffer_m
            d1_m = float(np.linalg.norm(p1 - self.center))
            if r_keepout_m > 0.0 and d1_m < r_keepout_m:
                gap_m = (r_keepout_m - d1_m)
                center_keepout = self.def_center_avoid_coef * (gap_m * gap_m)

        a1n2 = float(np.dot(a1, a1)) / (self.u_hi**2) if need_def else 0.0

        aA_n2 = (np.sum(aA * aA, axis=1).astype(np.float32) / (self.u_hi**2))  # (Na,)
        rhoA = np.array([np.linalg.norm(pA - self.center) / self.radius for pA in pA_list], dtype=np.float32)
        wallA = ((np.maximum(0.0, rhoA - self.soft_wall)**2) * self.wallK).astype(np.float32) if need_att else np.zeros_like(rhoA)

        # ---- rewards ----
        r1 = 0.0
        r_att_each = np.zeros((self.num_attackers,), dtype=np.float32)

        if need_def:
            r1 = (
                self.alpha * delta_d2
                + self.k_pos * d2
                - self.lD * a1n2
                - wall1
                - center_keepout
            )

        if need_att:
            progress_each = (self._d2_prev - d2_each).astype(np.float32)  # inward progress
            dist_each = np.array([np.linalg.norm(pA_list[k] - p1) for k in range(self.num_attackers)], dtype=np.float32)
            x = dist_each / (float(self.att_min_sep) + 1e-9)
            close_pen_each = (np.maximum(0.0, 1.0 - x) ** 2).astype(np.float32)

            r_att_each = (
                + self.k_att_prog * progress_each
                - self.k_att_close * close_pen_each
                - self.lA * aA_n2
                - wallA
            ).astype(np.float32)

        r2 = float(r_att_each.mean()) if need_att else 0.0

        # ---- termination (TRUE state) ----
        rho_def = np.linalg.norm(p1 - self.center) / self.radius

        # hit target if ANY attacker hits, or defender hits
        thresh_def = (1.0 + self.hit_buffer_def) * self.oi_radius_norm
        thresh_att = (1.0 + self.hit_buffer_att) * self.oi_radius_norm

        def_hit_target = (self.oi_radius_norm > 0.0) and (rho_def <= thresh_def)

        att_hit_any = False
        for pA_true in pA_list:
            rhoA_true = np.linalg.norm(pA_true - self.center) / self.radius
            if (self.oi_radius_norm > 0.0) and (rhoA_true <= thresh_att):
                att_hit_any = True
                break

        hit_target = bool(att_hit_any or def_hit_target)

        collision = False
        if self.collision_radius_m > 0.0:
            for pA_true in pA_list:
                if np.linalg.norm(pA_true - p1) <= self.collision_radius_m:
                    collision = True
                    break

        oob1 = (rho1 >= self.margin)

        oob2_any = False
        for pA_true in pA_list:
            rhoA_true = np.linalg.norm(pA_true - self.center) / self.radius
            if rhoA_true >= self.margin:
                oob2_any = True
                break

        done = (oob1 or oob2_any or hit_target or collision)

        if need_def and done:
            if collision:
                r1 -= self.collision_penalty_def
            if oob1:
                r1 -= self.wallK
            if att_hit_any:
                r1 -= self.def_target_hit_penalty_def
            if def_hit_target:
                r1 -= self.att_target_hit_penalty_def

        if need_att and done:
            if collision:
                # penalize each attacker the same (keeps old single-scalar return behavior)
                r_att_each -= self.collision_penalty_att
                r2 = float(r_att_each.mean())
            if oob2_any:
                r_att_each -= self.wallK
                r2 = float(r_att_each.mean())
            if att_hit_any:
                r_att_each += self.att_target_hit_reward_att
                r2 = float(r_att_each.mean())

        # update per-attacker d2_prev
        self._d2_prev[:] = d2_each

        # ---- info for logging ----
        d1_true_norm = float(np.dot(p1 - self.center, p1 - self.center)) / (self.radius**2)
        d2_true_norm = float(d2_each[k_star])  # closest-to-center attacker as scalar

        info = {
            "t": self.t,
            "d2_norm": float(d2_each[k_star]),
            "rel2_norm": float(rel_each[k_star]),
            "oob_def": bool(oob1),
            "oob_att": bool(oob2_any),
            "hit_target": bool(hit_target),
            "d1_true_norm": d1_true_norm,
            "d2_true_norm": d2_true_norm,
            "collision": bool(collision),

            # NEW multi-attacker diagnostics (used for attacker training aggregation)
            "closest_attacker_idx": int(k_star),
            "d2_true_norm_each": d2_each.tolist(),
            "rel2_norm_each": rel_each.tolist(),
            "r_att_each": r_att_each.tolist(),
        }

        if self.use_ukf:
            info["meas_innov_sq"] = meas_innov_sq
            info["ukf_trPpos"] = meas_trPpos

        return self._obs(), float(r1), float(r2), bool(done), info

    def _obs(self) -> np.ndarray:
        p1, v1, pA_list, vA_list = self._unpack(self.state)
        Na = self.num_attackers

        if self.use_ukf and (self.ukf is not None) and self.num_attackers == 1:
            p2_obs = self.ukf.x[:self.D]
            v2_obs = self.ukf.x[self.D:2*self.D]
            pA_obs = [p2_obs]
            vA_obs = [v2_obs]
        else:
            pA_obs = pA_list
            vA_obs = vA_list

        p1c = p1 - self.center
        parts = [p1c]

        for pA in pA_obs:
            parts.append(pA - self.center)

        for pA in pA_obs:
            parts.append(pA - p1)

        parts.append(v1)

        for vA in vA_obs:
            parts.append(vA)

        obs = np.concatenate(parts).astype(np.float32)
        return obs

    def _plant_step(self, s: np.ndarray, a1: np.ndarray, aA: np.ndarray) -> np.ndarray:
        D = self.D
        Na = self.num_attackers

        p1, v1, pA_list, vA_list = self._unpack(s)

        x1 = np.concatenate([p1, v1])
        x1n = self.Ad @ x1 + self.Bd @ a1
        p1n, v1n = x1n[:D], x1n[D:]

        aA = np.asarray(aA, float)
        if aA.ndim == 1:
            aA = aA.reshape(1, D)
        else:
            aA = aA.reshape(Na, D)

        pA_new = []
        vA_new = []
        for k in range(Na):
            p2 = pA_list[k]
            v2 = vA_list[k]
            x2 = np.concatenate([p2, v2])
            x2n = self.Ad @ x2 + self.Bd @ aA[k]
            pA_new.append(x2n[:D])
            vA_new.append(x2n[D:])

        parts = [p1n, v1n]
        for p2n, v2n in zip(pA_new, vA_new):
            parts.append(p2n)
            parts.append(v2n)

        return np.concatenate(parts, axis=0)

    def _unpack(self, s: np.ndarray):
        D = self.D
        Na = self.num_attackers

        p1 = s[0:D]
        v1 = s[D:2*D]

        pA_list = []
        vA_list = []

        off = 2 * D
        for k in range(Na):
            pA = s[off + 2*k*D: off + (2*k+1)*D]
            vA = s[off + (2*k+1)*D: off + (2*k+2)*D]
            pA_list.append(pA)
            vA_list.append(vA)

        return p1, v1, pA_list, vA_list


# =============================================================
# Vectorized env (single-process)
# =============================================================

class VecEnv:
    def __init__(self, make_env: Callable[[], Env], num_envs: int):
        self.envs: List[Env] = [make_env() for _ in range(num_envs)]
        self.num_envs = num_envs
        o = [e.reset() for e in self.envs]
        self.obs = np.stack(o, axis=0)

    def reset(self):
        o = [e.reset() for e in self.envs]
        self.obs = np.stack(o, axis=0)
        return self.obs

    def step(self, a1_env: np.ndarray, aA_env: np.ndarray, reward_mode: str = "both"):
        obs_next = []
        r1, r2, done, info = [], [], [], []
        for i, e in enumerate(self.envs):
            o, R1, R2, d, inf = e.step(a1_env[i], aA_env[i], reward_mode=reward_mode)
            if d:
                o = e.reset()
            obs_next.append(o)
            r1.append(R1)
            r2.append(R2)
            done.append(d)
            info.append(inf)
        self.obs = np.stack(obs_next, axis=0)
        return self.obs, np.array(r1), np.array(r2), np.array(done, dtype=np.float32), info


# =============================================================
# PPO Storage & Advantage
# =============================================================

class RolloutBuffer:
    def __init__(self, obs_dim, act_dim, num_envs, horizon, device):
        self.N, self.T = num_envs, horizon
        self.device = device
        self.obs = torch.zeros(self.T, self.N, obs_dim, device=device)
        self.act = torch.zeros(self.T, self.N, act_dim, device=device)
        self.logp = torch.zeros(self.T, self.N, device=device)
        self.val = torch.zeros(self.T, self.N, device=device)
        self.rew = torch.zeros(self.T, self.N, device=device)
        self.done = torch.zeros(self.T, self.N, device=device)
        self.next_val = torch.zeros(self.N, device=device)
        self.ptr = 0

    def add(self, obs, act, logp, val, rew, done):
        t = self.ptr
        self.obs[t] = obs.detach()
        self.act[t] = act.detach()
        self.logp[t] = logp.detach()
        self.val[t] = val.detach()
        self.rew[t] = rew
        self.done[t] = done
        self.ptr += 1

    def finalize(self, next_val):
        self.next_val = next_val

    def get(self):
        B = self.T * self.N
        obs = self.obs.reshape(B, -1)
        act = self.act.reshape(B, -1)
        logp = self.logp.reshape(B)
        val = self.val.reshape(B)
        rew = self.rew.reshape(B)
        done = self.done.reshape(B)
        return obs, act, logp, val, rew, done


def compute_gae_from_buffer(buf: RolloutBuffer, gamma: float, lam: float):
    T, N = buf.T, buf.N
    device = buf.device
    adv = torch.zeros(T, N, device=device)
    lastgaelam = torch.zeros(N, device=device)
    next_val = buf.next_val
    for t in reversed(range(T)):
        nonterminal = 1.0 - buf.done[t]
        next_v = next_val if t == T - 1 else buf.val[t + 1]
        delta = buf.rew[t] + gamma * next_v * nonterminal - buf.val[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = adv + buf.val
    A = adv.reshape(T * N)
    R = ret.reshape(T * N)
    A = (A - A.mean()) / (A.std() + 1e-8)
    return A, R


# =============================================================
# Rule-based Attacker Controller
# =============================================================

class AttackerRuleController:
    """
    u = sat_umax( w_center * u_center + w_avoid * u_repulse - w_damp * v2 )
    """

    def __init__(self, cfg: Dict[str, Any]):
        D = int(cfg["D"])
        self.D = D
        ar = cfg["arena"]
        self.center = np.array(
            [ar["cx"], ar["cy"], (ar["cz"] if D == 3 else 0.0)],
            dtype=np.float32
        )[:D]

        self.Ad = np.asarray(cfg["dyn"]["Ad"], dtype=np.float32)
        self.Bd = np.asarray(cfg["dyn"]["Bd"], dtype=np.float32)

        P = np.hstack([np.eye(D, dtype=np.float32), np.zeros((D, D), dtype=np.float32)])
        self.P = P
        self.F = (self.Bd.T @ P.T).T
        FtF = self.F.T @ self.F

        rule = dict(
            ridge=1e-2,
            w_center=0.5,
            w_avoid=2.0,
            w_damp=0.3,
            min_sep=3.0,
            repulse_gain=10.0,
        )
        rule.update(cfg.get("att_rule", {}))

        self.lam = float(rule["ridge"])
        self.w_center = float(rule["w_center"])
        self.w_avoid = float(rule["w_avoid"])
        self.w_damp = float(rule["w_damp"])
        self.min_sep = float(rule["min_sep"])
        self.repulse_gain = float(rule["repulse_gain"])
        self.umax = float(cfg["umax"])

        self.K = np.linalg.solve(FtF + self.lam * np.eye(D, dtype=np.float32), self.F.T)

    def u_center(self, p2: np.ndarray, v2: np.ndarray) -> np.ndarray:
        x2 = np.concatenate([p2, v2])
        E2 = (self.Ad @ x2)[:self.D] - self.center
        return -(self.K @ E2)

    def u_repulse(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        r = p2 - p1
        dist = float(np.linalg.norm(r)) + 1e-9
        r_hat = r / dist
        if dist < self.min_sep:
            return self.umax * r_hat
        mag = self.repulse_gain / (dist**2)
        return mag * r_hat

    def act(self, p1: np.ndarray, v1: np.ndarray, p2: np.ndarray, v2: np.ndarray) -> np.ndarray:
        uc = self.u_center(p2, v2)
        ur = self.u_repulse(p1, p2)

        dist = float(np.linalg.norm(p2 - p1))
        if dist < self.min_sep:
            w_center_eff = 0.0
        else:
            w_center_eff = self.w_center

        u = w_center_eff * uc + self.w_avoid * ur - self.w_damp * v2
        return np.clip(u, -self.umax, +self.umax)

    def act_multi(
        self,
        p1: np.ndarray, v1: np.ndarray,
        pA_list: List[np.ndarray], vA_list: List[np.ndarray],
    ) -> np.ndarray:
        u_list = []
        for p2, v2 in zip(pA_list, vA_list):
            u_list.append(self.act(p1, v1, p2, v2))
        return np.stack(u_list, axis=0).astype(np.float32)


# =============================================================
# DiffLS / Priors + Actor-Critic (patched for Na>1)
# =============================================================

class DiffLSLayer(nn.Module):
    def __init__(self, obs_dim: int, cfg: Dict[str, Any]):
        super().__init__()
        self.D = int(cfg["D"])
        self.Na = int(cfg.get("num_attackers", 1))
        self.ridge = float(cfg.get("prior_ridge", 1e-2))
        self.dt = float(cfg.get("dt", cfg.get("h", cfg.get("dyn", {}).get("dt", 1.0))))

        # Na==1: same as old feature dim
        self.feat_dim = (4 * self.D) if (self.Na == 1) else obs_dim

    def forward(self, obs: torch.Tensor, who: str):
        B, D, Na = obs.shape[0], self.D, self.Na
        dt = self.dt

        off_p1, off_pA, off_rel, off_v1, off_vA = _obs_offsets(D, Na)

        p1c = obs[:, off_p1:off_p1 + D]
        p2c = obs[:, off_pA:off_pA + D]      # attacker-slot 0 (ego for permuted attacker obs)
        v1 = obs[:, off_v1:off_v1 + D]
        v2 = obs[:, off_vA:off_vA + D]       # attacker-slot 0 velocity

        e1 = p1c + dt * v1
        e2 = p2c + dt * v2

        alpha = 0.5 * (dt * dt)
        k = alpha / (alpha * alpha + self.ridge)

        u_def_prior = -k * e1
        u_att_prior = -k * e2
        u_prior = u_def_prior if who == "def" else u_att_prior

        if Na == 1:
            feats = torch.cat([p1c, p2c, v1, v2], dim=-1)  # 4D (old behavior)
        else:
            feats = obs  # full obs so policy sees both attackers

        return feats, u_prior


class DiffNashLayer(nn.Module):
    # unchanged behavior; only supports Na==1
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.D = int(cfg["D"])
        self.dt = float(cfg["dt"])

        self.register_buffer(
            "Ad",
            torch.as_tensor(np.asarray(cfg["dyn"]["Ad"], np.float32), dtype=torch.float32)
        )
        self.register_buffer(
            "Bd",
            torch.as_tensor(np.asarray(cfg["dyn"]["Bd"], np.float32), dtype=torch.float32)
        )

        ar = cfg["arena"]
        c = np.array(
            [ar["cx"], ar["cy"], (ar["cz"] if self.D == 3 else 0.0)],
            dtype=np.float32
        )[: self.D]
        self.register_buffer("center", torch.tensor(c, dtype=torch.float32))

        solver_cfg = cfg.get("nash_solver", {})
        module_name = solver_cfg.get("module", "nash_ipopt_solver")
        fn_name = solver_cfg.get("fn", "solve_nash_ipopt")
        self.solver_params = solver_cfg.get("params", {})

        mod = importlib.import_module(module_name)
        self.nash_solve = getattr(mod, fn_name)

        # for Na==1: features are 4D
        self.feat_dim = 4 * self.D

    def forward(self, obs: torch.Tensor, who: str):
        B, D = obs.shape[0], self.D
        device = obs.device
        dtype = obs.dtype

        center = self.center.to(dtype=dtype, device=device)

        p1c = obs[:, 0:D]
        p2c = obs[:, D:2*D]
        v1 = obs[:, 3*D:4*D]
        v2 = obs[:, 4*D:5*D]

        p1 = p1c + center
        p2 = p2c + center

        x1 = torch.cat([p1, v1], dim=-1)
        x2 = torch.cat([p2, v2], dim=-1)

        x1_np = x1.detach().cpu().numpy()
        x2_np = x2.detach().cpu().numpy()

        u1_list = []
        u2_list = []
        for i in range(B):
            u1_i, u2_i = self.nash_solve(x1_np[i], x2_np[i], self.solver_params)
            u1_list.append(np.asarray(u1_i, dtype=np.float32))
            u2_list.append(np.asarray(u2_i, dtype=np.float32))

        u1_np = np.stack(u1_list, axis=0)
        u2_np = np.stack(u2_list, axis=0)

        u1_prior = torch.as_tensor(u1_np, device=device, dtype=dtype)
        u2_prior = torch.as_tensor(u2_np, device=device, dtype=dtype)
        u_prior = u1_prior if who == "def" else u2_prior

        feats = torch.cat([p1c, p2c, v1, v2], dim=-1)
        return feats, u_prior


class NoPriorLayer(nn.Module):
    def __init__(self, obs_dim: int, cfg: Dict[str, Any]):
        super().__init__()
        self.D = int(cfg["D"])
        self.Na = int(cfg.get("num_attackers", 1))
        self.feat_dim = (4 * self.D) if (self.Na == 1) else obs_dim

    def forward(self, obs: torch.Tensor, who: str):
        D, Na = self.D, self.Na
        off_p1, off_pA, off_rel, off_v1, off_vA = _obs_offsets(D, Na)

        p1c = obs[:, off_p1:off_p1 + D]
        p2c = obs[:, off_pA:off_pA + D]
        v1 = obs[:, off_v1:off_v1 + D]
        v2 = obs[:, off_vA:off_vA + D]

        if Na == 1:
            feats = torch.cat([p1c, p2c, v1, v2], dim=-1)
        else:
            feats = obs

        u_prior = torch.zeros((obs.shape[0], D), device=obs.device, dtype=obs.dtype)
        return feats, u_prior


class ActorCriticDiff(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg: Dict[str, Any]):
        super().__init__()
        hidden = 128

        prior_type = cfg.get("prior_type", "none")

        if prior_type == "ls":
            self.layer = DiffLSLayer(obs_dim, cfg)
        elif prior_type == "nash":
            # only safe for Na==1; you probably won't use Nash for 1v2 anyway
            if int(cfg.get("num_attackers", 1)) != 1:
                raise ValueError("prior_type='nash' only supported for num_attackers=1")
            self.layer = DiffNashLayer(cfg)
        elif prior_type == "none":
            self.layer = NoPriorLayer(obs_dim, cfg)
        else:
            raise ValueError(f"Unknown prior_type={prior_type!r}, expected 'ls', 'nash', or 'none'.")

        feat_dim = int(getattr(self.layer, "feat_dim", 4 * cfg["D"]))

        self.pi = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu_res = nn.Linear(hidden, act_dim)
        self.logstd = nn.Parameter(torch.full((act_dim,), -1.0))

        self.vf = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        self.prior_blend_def = float(cfg.get("prior_blend_def", 0.5))
        self.prior_blend_att = float(cfg.get("prior_blend_att", 1.0))

    def dist(self, obs: torch.Tensor, who: str):
        feats, u_prior = self.layer(obs, who)
        h = self.pi(feats)
        mu_res = self.mu_res(h)

        blend = self.prior_blend_def if who == "def" else self.prior_blend_att
        mu = mu_res + blend * u_prior

        std = self.logstd.exp()
        return torch.distributions.Normal(mu, std)

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


# =============================================================
# PPO Core
# =============================================================

class PPO:
    def __init__(self, obs_dim: int, act_dim: int, cfg: Dict[str, Any], device="cpu"):
        self.cfg = cfg
        self.device = device
        self.clip_eps = cfg["clip_eps"]
        self.ent_coef = cfg["entropy_coef"]
        self.vf_coef = cfg["value_coef"]
        self.max_grad = cfg["max_grad_norm"]
        self.epochs = cfg["train_epochs"]
        self.mb_size = cfg["minibatch_size"]
        self.gamma = cfg["gamma"]
        self.lam = cfg["gae_lambda"]
        self.act_scale = float(cfg["umax"])
        self.v_clip_eps = 0.2

        self.attacker_mode = cfg.get("attacker_mode", "rule")

        self.freeze_defender = bool(cfg.get("freeze_defender", False))
        self.freeze_attacker = bool(cfg.get("freeze_attacker", False))

        self.D = int(cfg["D"])
        self.Na = int(cfg.get("num_attackers", 1))

        self.def_net = ActorCriticDiff(obs_dim, act_dim, cfg).to(device)
        self.def_opt = optim.Adam([
            {"params": list(self.def_net.pi.parameters()) + list(self.def_net.mu_res.parameters()), "lr": cfg["policy_lr"]},
            {"params": [self.def_net.logstd], "lr": cfg["policy_lr"] * 0.5},
            {"params": self.def_net.vf.parameters(), "lr": cfg["value_lr"]},
        ])
        self.def_base_lrs = [g["lr"] for g in self.def_opt.param_groups]

        if self.attacker_mode == "rl":
            self.att_net = ActorCriticDiff(obs_dim, act_dim, cfg).to(device)
            self.att_opt = optim.Adam([
                {"params": list(self.att_net.pi.parameters()) + list(self.att_net.mu_res.parameters()), "lr": cfg["policy_lr"]},
                {"params": [self.att_net.logstd], "lr": cfg["policy_lr"] * 0.5},
                {"params": self.att_net.vf.parameters(), "lr": cfg["value_lr"]},
            ])
            self.att_base_lrs = [g["lr"] for g in self.att_opt.param_groups]
            self.rule_ctrl = None
        else:
            self.att_net = None
            self.att_opt = None
            self.att_base_lrs = None
            self.rule_ctrl = AttackerRuleController(cfg)

    @torch.no_grad()
    def act(self, obs_batch: torch.Tensor, who: str, deterministic: bool = False):
        """
        Defender:
          a_env: (B, D), logp: (B,), val: (B,)
        Attacker:
          if Na==1 -> same
          if Na>1  -> a_env: (B, Na, D), logp: (B, Na), val: (B, Na)
        """
        B = obs_batch.shape[0]
        D = self.D
        Na = self.Na

        # -------- RULE ATTACKER --------
        if who == "att" and self.attacker_mode == "rule":
            if self.rule_ctrl is None:
                raise RuntimeError("rule_ctrl missing for rule attacker_mode")

            ob = obs_batch.detach().cpu().numpy()
            off_p1, off_pA, off_rel, off_v1, off_vA = _obs_offsets(D, Na)

            p1c = ob[:, off_p1:off_p1 + D]
            pA_c = ob[:, off_pA:off_pA + Na * D].reshape(B, Na, D)
            v1 = ob[:, off_v1:off_v1 + D]
            vA = ob[:, off_vA:off_vA + Na * D].reshape(B, Na, D)

            center = self.rule_ctrl.center
            acts = np.zeros((B, Na, D), dtype=np.float32)
            for i in range(B):
                p1 = p1c[i] + center
                for k in range(Na):
                    p2 = pA_c[i, k] + center
                    acts[i, k] = self.rule_ctrl.act(p1, v1[i], p2, vA[i, k]).astype(np.float32)

            a_t = torch.as_tensor(acts, dtype=obs_batch.dtype, device=obs_batch.device)

            if Na == 1:
                a_t = a_t[:, 0, :]
                zero = torch.zeros(B, dtype=obs_batch.dtype, device=obs_batch.device)
                return a_t, zero, zero

            zero2 = torch.zeros((B, Na), dtype=obs_batch.dtype, device=obs_batch.device)
            return a_t, zero2, zero2

        # -------- RL POLICY --------
        net = self.def_net if who == "def" else self.att_net
        if net is None:
            raise RuntimeError("att_net is None but attacker_mode='rl' required")

        # Defender path (unchanged shape)
        if who == "def" or Na == 1:
            if deterministic:
                dist = net.dist(obs_batch, who=who)
                u_raw = dist.mean
                a_env = squash_action(u_raw, self.act_scale)
                logp = torch.zeros(B, dtype=obs_batch.dtype, device=obs_batch.device)
                val = net.value(obs_batch)
                return a_env, logp, val
            return net.act(obs_batch, who, self.act_scale)

        # Attacker path: Na>1, shared policy applied per attacker via permutation
        a_list, lp_list, v_list = [], [], []
        for k in range(Na):
            obs_k = permute_obs_for_attacker(obs_batch, k, D, Na)
            if deterministic:
                dist = net.dist(obs_k, who="att")
                u_raw = dist.mean
                a_env = squash_action(u_raw, self.act_scale)
                logp = torch.zeros(B, dtype=obs_batch.dtype, device=obs_batch.device)
                val = net.value(obs_k)
            else:
                a_env, logp, val = net.act(obs_k, who="att", act_scale=self.act_scale)
            a_list.append(a_env)
            lp_list.append(logp)
            v_list.append(val)

        a_env = torch.stack(a_list, dim=1)     # (B, Na, D)
        logp = torch.stack(lp_list, dim=1)     # (B, Na)
        val = torch.stack(v_list, dim=1)       # (B, Na)
        return a_env, logp, val

    def _update_one(self, net: ActorCriticDiff, opt: optim.Optimizer,
                    obs: torch.Tensor, act_env: torch.Tensor,
                    old_logp: torch.Tensor, old_val: torch.Tensor,
                    adv: torch.Tensor, ret: torch.Tensor, who: str):
        B = obs.shape[0]
        for _ in range(self.epochs):
            idx = torch.randperm(B, device=obs.device)
            for st in range(0, B, self.mb_size):
                j = idx[st:st + self.mb_size]
                o = obs[j]
                a = act_env[j]
                lp_old = old_logp[j]
                v_old = old_val[j]
                A = adv[j]
                R = ret[j]

                assert not lp_old.requires_grad
                assert not v_old.requires_grad

                dist = net.dist(o, who)
                u_raw = atanh(torch.clamp(a / self.act_scale, -0.999999, 0.999999))
                lp = logprob_squashed(dist, u_raw)
                ratio = (lp - lp_old).exp()

                surr1 = ratio * A
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * A
                pi_loss = -torch.min(surr1, surr2).mean()

                v_pred = net.value(o)
                v_clipped = v_old + (v_pred - v_old).clamp(-self.v_clip_eps, self.v_clip_eps)
                v_loss = torch.max((v_pred - R).pow(2), (v_clipped - R).pow(2)).mean()

                ent = dist.entropy().sum(-1).mean()
                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), self.max_grad)
                opt.step()

    def update_defender_only(self, buf_def: RolloutBuffer):
        advD, retD = compute_gae_from_buffer(buf_def, self.gamma, self.lam)
        oD, aD, lpD, vD, _, _ = buf_def.get()
        self._update_one(self.def_net, self.def_opt, oD, aD, lpD, vD, advD, retD, who="def")

    def update_attacker_only(self, buf_att: RolloutBuffer):
        advA, retA = compute_gae_from_buffer(buf_att, self.gamma, self.lam)
        oA, aA, lpA, vA, _, _ = buf_att.get()
        self._update_one(self.att_net, self.att_opt, oA, aA, lpA, vA, advA, retA, who="att")


# =============================================================
# Distillation helpers
# =============================================================

def build_full_obs_from_envs(vec: VecEnv, device: str) -> torch.Tensor:
    """
    Teacher full-state obs for defender.
    For Na>1 we use the "most dangerous" attacker (closest to center) so it matches
    defender reward shaping.
      o_full = [p1-center, p2-center, (p2-p1), v1, v2] (5*D)
    """
    obs_list = []
    for e in vec.envs:
        p1, v1, pA_list, vA_list = e._unpack(e.state)

        # choose closest-to-center attacker
        d2_each = [float(np.dot(pA - e.center, pA - e.center)) for pA in pA_list]
        k_star = int(np.argmin(d2_each))
        p2 = pA_list[k_star]
        v2 = vA_list[k_star]

        center = e.center
        p1c = p1 - center
        p2c = p2 - center
        rel = p2 - p1

        obs_full = np.concatenate([p1c, p2c, rel, v1, v2]).astype(np.float32)
        obs_list.append(obs_full)

    obs_full_np = np.stack(obs_list, axis=0)
    return torch.as_tensor(obs_full_np, dtype=torch.float32, device=device)


# =============================================================
# Training & Evaluation
# =============================================================

def freeze_module_(m: torch.nn.Module):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)


@torch.no_grad()
def snapshot_state_dict(m: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


@torch.no_grad()
def max_state_dict_diff(snap: dict, m: torch.nn.Module) -> float:
    cur = m.state_dict()
    diffs = []
    for k, v0 in snap.items():
        v = cur[k].detach().cpu()
        diffs.append((v - v0).abs().max().item())
    return float(max(diffs) if diffs else 0.0)


@torch.no_grad()
def assert_frozen_unchanged(snap: dict, m: torch.nn.Module, name: str, tol: float = 0.0):
    d = max_state_dict_diff(snap, m)
    if d > tol:
        raise RuntimeError(f"[FREEZE CHECK FAILED] {name} changed! max|Δ|={d:.3e} > tol={tol:.3e}")
    return d


@torch.no_grad()
def assert_deterministic_action(ppo, obs_batch: torch.Tensor, who: str, tol: float = 0.0):
    a1, _, _ = ppo.act(obs_batch, who=who, deterministic=True)
    a2, _, _ = ppo.act(obs_batch, who=who, deterministic=True)
    d = (a1 - a2).abs().max().item()
    if d > tol:
        raise RuntimeError(f"[DETERMINISM CHECK FAILED] {who} deterministic act differs: max|Δa|={d:.3e}")
    return d


def train(cfg: Dict[str, Any]):
    set_seed(cfg["seed"])
    device = cfg["device"]

    writer = None
    tb_logdir = None
    global_env_step = 0

    if cfg.get("use_tensorboard", False):
        writer, tb_logdir = make_tb_writer(cfg)

    train_role = cfg.get("train_role", "def")

    if train_role == "def":
        reward_mode = "def"
    elif train_role == "att":
        reward_mode = "att"

    def make_env():
        return Env(cfg)

    num_envs = int(cfg.get("num_envs", 8))
    steps_per_env = int(cfg.get("steps_per_env", 256))
    total_updates = int(cfg.get("total_updates", 300))
    log_every = int(cfg.get("log_every", 10))

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

    verify_freeze = bool(cfg.get("verify_freeze", True))
    freeze_tol = float(cfg.get("freeze_tol", 0.0))

    snap_def = None
    snap_att = None

    if verify_freeze and cfg.get("freeze_defender", False):
        snap_def = snapshot_state_dict(ppo.def_net)
    if verify_freeze and cfg.get("freeze_attacker", False) and (ppo.att_net is not None):
        snap_att = snapshot_state_dict(ppo.att_net)

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

    def_center_base = cfg.get("def_center_coef", 0.0)
    min_anneal = float(cfg.get("def_center_min_anneal", 0.5))

    for upd in range(1, total_updates + 1):
        term_counts = {"oob_def": 0, "oob_att": 0, "hit_target": 0, "collision": 0}

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

        # Buffers
        bufD = RolloutBuffer(obs_dim, act_dim, num_envs, steps_per_env, device)

        rule_att = (cfg.get("attacker_mode", "rule") == "rule")
        if (not rule_att) and (train_role == "att"):
            # (1vN) attacker learns from all attackers => effective envs = num_envs * Na
            bufA = RolloutBuffer(obs_dim, act_dim, num_envs * Na, steps_per_env, device)
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

        for _ in range(steps_per_env):
            with torch.no_grad():
                det_def = (train_role != "def")
                det_att = (train_role != "att")

                a1, lp1, v1 = ppo.act(o, who="def", deterministic=det_def)
                a2, lp2, v2 = ppo.act(o, who="att", deterministic=det_att)

            a1_np = a1.cpu().numpy()

            a2_np = a2.cpu().numpy()
            # Env expects (B,Na,D) for Na>1; for Na==1, either (B,D) or (B,1,D) is fine
            if a2_np.ndim == 2 and cfg.get("attacker_mode", "rule") == "rule" and Na == 1:
                a2_np = a2_np[:, None, :]

            o2_np, r1_np, r2_np, d_np, infos = vec.step(
                a1_np,
                a2_np,
                reward_mode=reward_mode,
            )

            o2 = torch.as_tensor(o2_np, dtype=torch.float32, device=device)
            r1 = torch.as_tensor(r1_np, dtype=torch.float32, device=device)
            r2 = torch.as_tensor(r2_np, dtype=torch.float32, device=device)
            d = torch.as_tensor(d_np, dtype=torch.float32, device=device)

            # defender buffer (unchanged)
            bufD.add(o.detach(), a1.detach(), lp1.detach(), v1.detach(), r1, d)

            # attacker buffer (1vN aggregation) only when training attacker with RL attacker_mode
            if bufA is not None:
                if Na == 1:
                    bufA.add(o.detach(), a2.detach(), lp2.detach(), v2.detach(), r2, d)
                else:
                    # build per-attacker samples at this step
                    o_perm = [permute_obs_for_attacker(o, k, cfg["D"], Na) for k in range(Na)]
                    o_att_step = torch.cat(o_perm, dim=0)  # (B*Na, obs_dim)

                    a_att_step = torch.cat([a2[:, k, :] for k in range(Na)], dim=0)  # (B*Na, D)
                    lp_att_step = torch.cat([lp2[:, k] for k in range(Na)], dim=0)    # (B*Na,)
                    v_att_step = torch.cat([v2[:, k] for k in range(Na)], dim=0)      # (B*Na,)

                    # per-attacker rewards from info
                    rA_np = np.stack([inf["r_att_each"] for inf in infos], axis=0).astype(np.float32)  # (B,Na)
                    r_att_step = torch.as_tensor(
                        np.concatenate([rA_np[:, k] for k in range(Na)], axis=0),
                        dtype=torch.float32,
                        device=device,
                    )

                    d_att_step = d.repeat(Na)
                    bufA.add(o_att_step.detach(), a_att_step.detach(), lp_att_step.detach(), v_att_step.detach(), r_att_step, d_att_step)

            if train_role == "def":
                ep_ret_def += r1_np
            if train_role == "att":
                ep_ret_att += r2_np

            o = o2

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

            global_env_step += num_envs

        with torch.no_grad():
            next_v_def = ppo.def_net.value(o)
        bufD.finalize(next_v_def)

        if bufA is not None:
            with torch.no_grad():
                if Na == 1:
                    next_v_att = ppo.att_net.value(o)
                else:
                    next_v_list = [ppo.att_net.value(permute_obs_for_attacker(o, k, cfg["D"], Na)) for k in range(Na)]
                    next_v_att = torch.cat(next_v_list, dim=0)  # (B*Na,)
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
            if (train_role == "att") and (snap_def is not None):
                assert_frozen_unchanged(snap_def, ppo.def_net, name="frozen_defender", tol=freeze_tol)
            if (train_role == "def") and (snap_att is not None):
                assert_frozen_unchanged(snap_att, ppo.att_net, name="frozen_attacker", tol=freeze_tol)

        if upd % log_every == 0:
            R_def_mean = ep_ret_def.mean()
            R_att_mean = ep_ret_att.mean()

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

                dist_opp = (
                    ppo.def_net.dist(flat_obs, who="def")
                    if train_role == "att"
                    else ppo.att_net.dist(flat_obs, who="att")
                    if (train_role == "def" and ppo.att_net is not None)
                    else None
                )
                if dist_opp is not None:
                    print("opp std mean:", dist_opp.stddev.mean().item())

                Dcfg = cfg["D"]
                p1c = flat_obs[:, :Dcfg]
                p2c = flat_obs[:, Dcfg:2*Dcfg]
                d1_obs_mean = p1c.pow(2).sum(-1).mean().sqrt().item()
                d2_obs_mean = p2c.pow(2).sum(-1).mean().sqrt().item()

            lr_pi = ppo.def_opt.param_groups[0]["lr"]
            lr_vf = ppo.def_opt.param_groups[-1]["lr"]

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

            print(f"[update {upd:05d}] R_def_mean={R_def_mean:+.3f}  R_att_mean={R_att_mean:+.3f}  (batch={num_envs*steps_per_env})")
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

                if info_count > 0:
                    term_rates = {k: v / info_count for k, v in term_counts.items()}
                else:
                    term_rates = {k: 0.0 for k in term_counts}

                writer.add_scalar("term_rate/oob_def", term_rates["oob_def"], gs)
                writer.add_scalar("term_rate/oob_att", term_rates["oob_att"], gs)
                writer.add_scalar("term_rate/hit_target", term_rates["hit_target"], gs)
                writer.add_scalar("term_rate/collision", term_rates["collision"], gs)

                writer.add_scalar("act/def_abs_mean", a1.abs().mean().item(), global_env_step)
                writer.add_scalar("act/def_abs_max", a1.abs().max().item(), global_env_step)

    try:
        del bufD
    except:
        pass
    try:
        del bufA
    except:
        pass
    try:
        del vec
    except:
        pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if writer is not None:
        writer.flush()
        writer.close()

    print("Training finished.")
    return ppo, metrics


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
            # if Na>1, a2_np is (Na,D) already; Env.step accepts (Na,D)
            obs, r1, r2, done, info = env.step(a1_np, a2_np)
            states.append(env.state.copy())
            actions.append((a1_np.copy(), a2_np.copy()))
            infos.append(info)
        trajs.append({"states": np.stack(states), "actions": actions, "infos": infos})
    return trajs


# =============================================================
# Plotting / TB writer (unchanged)
# =============================================================

def load_npz_metrics(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Metrics file not found: {path}")
    data = np.load(path, allow_pickle=True)
    metrics = {}
    for k in data.files:
        metrics[k] = data[k]
    return metrics


def plot_training_metrics(
    metrics: dict,
    title: str = "",
    smooth: str = "none",
    smooth_param: float = 0.2,
    show: bool = False,
    out_dir: str | None = None,
    save_prefix: str | None = None,
    dpi: int = 200,
    close: bool = True,
):
    def _as_1d(x):
        x = np.asarray(x)
        if x.ndim == 0:
            return x.reshape(1).astype(float)
        if x.ndim > 1:
            x = x.reshape(-1)
        return x.astype(float)

    def _smooth_series(y, method="none", param=0.2):
        y = _as_1d(y)
        if method is None or method == "none":
            return y
        if method == "ema":
            alpha = float(param)
            if not (0.0 < alpha <= 1.0):
                raise ValueError("EMA alpha must be in (0, 1].")
            out = np.empty_like(y)
            out[0] = y[0]
            for i in range(1, len(y)):
                out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]
            return out
        if method == "ma":
            win = int(param)
            if win <= 1:
                return y
            pad = win - 1
            ypad = np.pad(y, (pad, 0), mode="edge")
            kernel = np.ones(win, dtype=float) / win
            return np.convolve(ypad, kernel, mode="valid")
        raise ValueError(f"Unknown smoothing method: {method!r}")

    if "update" in metrics:
        x = _as_1d(metrics["update"])
    else:
        for k in ["R_def_mean", "R_att_mean", "d1_mean", "d2_mean", "lr_pi"]:
            if k in metrics:
                x = np.arange(len(metrics[k]), dtype=float)
                break
        else:
            raise ValueError("No recognizable metrics keys found to infer x-axis.")

    def _plot_if(ax, key, label=None):
        if key not in metrics:
            return False
        y = _smooth_series(metrics[key], method=smooth, param=smooth_param)
        ax.plot(x, y, label=(label or key))
        return True

    figs = []
    saved_paths = []

    if save_prefix is None:
        save_prefix = title.strip() if title.strip() else "run"

    if ("R_def_mean" in metrics) or ("R_att_mean" in metrics):
        fig = plt.figure()
        ax = plt.gca()
        any_plotted = False
        any_plotted |= _plot_if(ax, "R_def_mean", "R_def_mean (per update)")
        any_plotted |= _plot_if(ax, "R_att_mean", "R_att_mean (per update)")
        if any_plotted:
            ax.set_xlabel("Update")
            ax.set_ylabel("Mean return (sum over rollout)")
            ax.set_title(f"{title} — Returns" if title else "Returns")
            ax.grid(True)
            ax.legend()
            figs.append(("returns", fig))

    dist_keys = [k for k in ["d1_mean", "d2_mean", "d2_true_mean"] if k in metrics]
    if dist_keys:
        fig = plt.figure()
        ax = plt.gca()
        _plot_if(ax, "d1_mean", "Defender true ⟨||p1-center||⟩ (m)")
        _plot_if(ax, "d2_true_mean", "Attacker true ⟨||p2-center||⟩ (m)")
        _plot_if(ax, "d2_mean", "Attacker belief ⟨||p2-center||⟩ (m)")
        ax.set_xlabel("Update")
        ax.set_ylabel("Distance to center (m)")
        ax.set_title(f"{title} — Distances" if title else "Distances")
        ax.grid(True)
        ax.legend()
        figs.append(("distances", fig))

    if ("lr_pi" in metrics) or ("lr_vf" in metrics):
        fig = plt.figure()
        ax = plt.gca()
        _plot_if(ax, "lr_pi", "lr_pi")
        _plot_if(ax, "lr_vf", "lr_vf")
        ax.set_xlabel("Update")
        ax.set_ylabel("Learning rate")
        ax.set_title(f"{title} — Learning rates" if title else "Learning rates")
        ax.grid(True)
        ax.legend()
        figs.append(("lrs", fig))

    if ("muD_abs_mean" in metrics) or ("stdD_mean" in metrics):
        fig = plt.figure()
        ax = plt.gca()
        _plot_if(ax, "muD_abs_mean", "|mu| mean (def)")
        _plot_if(ax, "stdD_mean", "std mean (def)")
        ax.set_xlabel("Update")
        ax.set_ylabel("Value")
        ax.set_title(f"{title} — Policy stats" if title else "Policy stats")
        ax.grid(True)
        ax.legend()
        figs.append(("policy_stats", fig))

    if ("meas_innov_mean" in metrics) or ("ukf_trPpos_mean" in metrics):
        fig = plt.figure()
        ax = plt.gca()
        _plot_if(ax, "meas_innov_mean", "meas_innov_mean (E[||innov||^2])")
        _plot_if(ax, "ukf_trPpos_mean", "ukf_trPpos_mean (trace(P_pos))")
        ax.set_xlabel("Update")
        ax.set_ylabel("Value")
        ax.set_title(f"{title} — UKF stats" if title else "UKF stats")
        ax.grid(True)
        ax.legend()
        figs.append(("ukf_stats", fig))

    if not figs:
        raise ValueError("None of the expected keys were present; nothing to plot.")

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        for tag, fig in figs:
            fname = f"{save_prefix}__{tag}.png"
            fpath = os.path.join(out_dir, fname)
            fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
            saved_paths.append(fpath)

    if show:
        plt.show()

    if close:
        for _, fig in figs:
            plt.close(fig)

    return [fig for _, fig in figs], saved_paths

def plot_compare_phases(
    labeled_paths,
    metric: str,
    ylabel: str = None,
    title: str = None,
    smooth: str = "none",
    smooth_param: float = 0.2,
    show: bool = False,

    # NEW saving knobs
    out_dir: str | None = None,
    filename: str | None = None,
    dpi: int = 200,
    close: bool = True,
):
    """
    Compare a single metric across multiple runs and optionally save it.
    """
    fig = plt.figure()
    ax = plt.gca()

    def _as_1d(x):
        x = np.asarray(x)
        if x.ndim == 0:
            return x.reshape(1).astype(float)
        if x.ndim > 1:
            x = x.reshape(-1)
        return x.astype(float)

    def _smooth_series(y, method="none", param=0.2):
        y = _as_1d(y)
        if method is None or method == "none":
            return y
        if method == "ema":
            alpha = float(param)
            out = np.empty_like(y)
            out[0] = y[0]
            for i in range(1, len(y)):
                out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]
            return out
        if method == "ma":
            win = int(param)
            if win <= 1:
                return y
            pad = win - 1
            ypad = np.pad(y, (pad, 0), mode="edge")
            kernel = np.ones(win, dtype=float) / win
            return np.convolve(ypad, kernel, mode="valid")
        raise ValueError(f"Unknown smoothing method: {method!r}")

    for label, path in labeled_paths:
        m = load_npz_metrics(path)
        if metric not in m:
            print(f"[plot_compare_phases] skipping {label}: missing key {metric!r}")
            continue
        x = _as_1d(m["update"]) if "update" in m else np.arange(len(m[metric]))
        y = _smooth_series(m[metric], method=smooth, param=smooth_param)
        ax.plot(x, y, label=label)

    ax.set_xlabel("Update")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or f"Compare: {metric}")
    ax.grid(True)
    ax.legend()

    saved_path = None
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        if filename is None:
            safe_metric = metric.replace("/", "_")
            filename = f"compare__{safe_metric}.png"
        saved_path = os.path.join(out_dir, filename)
        fig.savefig(saved_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    if close:
        plt.close(fig)

    return fig, saved_path


def make_tb_writer(cfg: dict, run_name: str | None = None):
    root = cfg.get("tb_logdir", "runs")
    if run_name is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = cfg.get("tb_run_name", f"diffgame_{stamp}")

    logdir = os.path.join(root, run_name)
    os.makedirs(logdir, exist_ok=True)

    try:
        with open(os.path.join(logdir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2, default=str)
    except Exception as e:
        print("[tb] could not write config.json:", e)

    writer = SummaryWriter(log_dir=logdir)
    print(f"[tb] logging to: {logdir}")
    return writer, logdir


# =============================================================
# End phase cleanup (unchanged)
# =============================================================

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


# =============================================================
# Main (kept as in your script; just remember cfg["num_attackers"]=2)
# =============================================================

# if __name__ == "__main__":
#     OUT_DIR = "Training_Policy"
#     os.makedirs(OUT_DIR, exist_ok=True)

#     # Example: set num_attackers=2 in the config builder you use
#     # (You can do this inside config_for_train if you prefer.)
#     cfg = config_for_train(attacker_mode="rl", train_role="def")
#     cfg["num_attackers"] = 2
#     build_dyn(cfg)
#     ppo, metrics = train(cfg)

#     print("This script is now 1 defender vs N attackers (shared attacker policy).")
#     print("Set cfg['num_attackers']=2 and run your existing phase logic as before.")


# =============================================================
# (1v2) FULL PHASED TRAINING CYCLE: Def0 -> Att1 -> Def1
# Drop this into your patched rl_loop.py (below make_tb_writer / plotting helpers),
# and replace your current __main__ with the __main__ at the bottom of this block.
#
# IMPORTANT:
# - This preserves the *same* phase print statements and the same PPO training logic.
# - The ONLY changes are the 1v2 plumbing: setting cfg["num_attackers"]=2 everywhere,
#   and making attacker BC-pretrain work with Na>1 (shared attacker policy) the same
#   way PPO does (obs permutation + stacking).
# - UKF distillation is still 1-attacker only in your code; for Na=2 we simply skip
#   distillation by forcing DISTILL=False in the wrappers (no extra prints).
# =============================================================

# ---- global knob for 1 defender vs 2 attackers ----
NUM_ATTACKERS = 2


# =============================================================
# BC PRETRAIN (patched for Na>=1, shared attacker policy)
# =============================================================
def pretrain_attacker_from_rule(
    cfg: Dict[str, Any],
    out_path: str,
    steps_per_env: int = 256,
    total_updates: int = 200,
    bc_epochs: int = 4,
    bc_mb_size: int = 2048,
):
    """
    Behavior clone the rule-based attacker into ppo.att_net (attacker_mode='rl').

    Na==1: identical behavior to your old function.
    Na>1 : collect (obs_perm_k, a_rule_k) for each attacker k using:
             - obs permutation (ego attacker in slot 0)
             - rule labels generated from TRUE state for each attacker
           and train shared attacker policy on the stacked dataset.
    """
    assert cfg.get("attacker_mode", "rule") == "rl", "BC pretrain requires attacker_mode='rl'"

    cfg = dict(cfg)
    cfg["num_attackers"] = int(cfg.get("num_attackers", NUM_ATTACKERS))

    set_seed(cfg["seed"])
    device = cfg["device"]

    def make_env():
        return Env(cfg)

    num_envs = int(cfg.get("num_envs", 8))
    vec = VecEnv(make_env, num_envs)
    obs_dim = vec.obs.shape[1]
    act_dim = int(cfg["D"])
    Na = int(cfg.get("num_attackers", 1))

    # PPO container (we'll only use its networks)
    ppo = PPO(obs_dim, act_dim, cfg, device=device)

    # Optional: load a defender checkpoint to make the data distribution realistic
    def_ckpt = cfg.get("def_ckpt_path", None)
    if def_ckpt is not None:
        state = torch.load(def_ckpt, map_location=device)
        ppo.def_net.load_state_dict(state)
        ppo.def_net.eval()
        for p in ppo.def_net.parameters():
            p.requires_grad_(False)

    # Rule controller to generate labels
    rule_ctrl = AttackerRuleController(cfg)

    # Optimizer for BC (attacker policy head only)
    bc_lr = float(cfg.get("att_bc_lr", cfg.get("policy_lr", 3e-4)))
    bc_opt = optim.Adam(
        list(ppo.att_net.pi.parameters()) + list(ppo.att_net.mu_res.parameters()) + [ppo.att_net.logstd],
        lr=bc_lr
    )

    print("=== BC PRETRAIN: attacker net imitates rule controller ===")
    print(f"Saving attacker BC checkpoint to: {out_path}")
    print(f"updates={total_updates}, steps_per_env={steps_per_env}, num_envs={num_envs}")

    for upd in range(1, total_updates + 1):
        # collect a batch
        obs_buf = torch.zeros(steps_per_env, num_envs, obs_dim, device=device)
        act_buf = torch.zeros(steps_per_env, num_envs, Na, act_dim, device=device)  # per-attacker labels

        for t in range(steps_per_env):
            o = torch.as_tensor(vec.obs, dtype=torch.float32, device=device)

            # Defender action: deterministic if frozen, else sample
            with torch.no_grad():
                det_def = (def_ckpt is not None)
                a1, _, _ = ppo.act(o, who="def", deterministic=det_def)

            # Rule attacker labels from TRUE state: one label per attacker
            aA_list = []
            for e in vec.envs:
                p1, v1, pA_list, vA_list = e._unpack(e.state)
                uA = rule_ctrl.act_multi(p1, v1, pA_list, vA_list).astype(np.float32)  # (Na, D)
                aA_list.append(uA)
            aA_np = np.stack(aA_list, axis=0)  # (B, Na, D)

            # Step env using defender action + rule attacker action
            o2_np, _, _, _, _ = vec.step(a1.cpu().numpy(), aA_np, reward_mode="both")

            obs_buf[t] = o
            act_buf[t] = torch.as_tensor(aA_np, dtype=torch.float32, device=device)

            vec.obs = o2_np

        # ---- flatten & (if Na>1) expand per-attacker samples via permutation ----
        B = steps_per_env * num_envs
        obs_flat = obs_buf.reshape(B, obs_dim)
        act_flat = act_buf.reshape(B, Na, act_dim)

        if Na == 1:
            # old shape
            obs_train = obs_flat
            act_train = act_flat[:, 0, :]
        else:
            # stack per attacker with permuted obs
            obs_list = []
            act_list = []
            for k in range(Na):
                obs_list.append(permute_obs_for_attacker(obs_flat, k, cfg["D"], Na))
                act_list.append(act_flat[:, k, :])
            obs_train = torch.cat(obs_list, dim=0)  # (B*Na, obs_dim)
            act_train = torch.cat(act_list, dim=0)  # (B*Na, D)
            B = B * Na

        # BC epochs
        for _ in range(bc_epochs):
            perm = torch.randperm(B, device=device)
            for st in range(0, B, bc_mb_size):
                idx = perm[st:st + bc_mb_size]
                o_mb = obs_train[idx]
                a_mb = act_train[idx]

                dist = ppo.att_net.dist(o_mb, who="att")
                mu = dist.mean

                bc_loss = ((mu - a_mb) ** 2).mean()

                bc_opt.zero_grad(set_to_none=True)
                bc_loss.backward()
                nn.utils.clip_grad_norm_(ppo.att_net.parameters(), cfg.get("max_grad_norm", 0.5))
                bc_opt.step()

        if upd % int(cfg.get("log_every", 10)) == 0:
            with torch.no_grad():
                take = min(B, 4096)
                dist_dbg = ppo.att_net.dist(obs_train[:take], who="att")
                mse_dbg = ((dist_dbg.mean - act_train[:take]) ** 2).mean().item()
            print(f"[att_bc {upd:05d}] BC MSE (dbg) = {mse_dbg:.3e}")

    torch.save(ppo.att_net.state_dict(), out_path)
    print(f"BC pretrain finished. Saved: {out_path}")
    return out_path


# =============================================================
# Phase wrappers (same prints/flow as your original)
# =============================================================

def train_defender_with_distill(
    phase_name: str,
    attacker_mode: str,
    extra_train_cfg: Dict[str, Any] | None = None,
):
    """
    Same structure as your original wrapper, with 1v2 added via cfg["num_attackers"]=NUM_ATTACKERS.

    NOTE: UKF distillation is not implemented for Na>1 in your codebase.
    For Na>1 we force DISTILL=False (no extra prints).
    """
    cfg_teacher = config_for_train(
        attacker_mode=attacker_mode,
        train_role="def",
    )
    cfg_teacher["num_attackers"] = int(cfg_teacher.get("num_attackers", NUM_ATTACKERS))
    cfg_teacher["use_ukf"] = False  # teacher is full-state

    if extra_train_cfg is not None:
        cfg_teacher.update(extra_train_cfg)
        cfg_teacher["num_attackers"] = int(cfg_teacher.get("num_attackers", NUM_ATTACKERS))

    # Force distill off if Na>1 (UKF only supports Na==1)
    DISTILL = bool(cfg_teacher.get("distill", False)) and (int(cfg_teacher["num_attackers"]) == 1)

    build_dyn(cfg_teacher)

    if cfg_teacher["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"[{phase_name.upper()} TEACHER] device='cuda' but CUDA not available.")
    print(f"[{phase_name.upper()} TEACHER] Using device: {cfg_teacher['device']}")
    print(f"[{phase_name.upper()}] distill={DISTILL}")

    ppo_def, metrics_def = train(cfg_teacher)

    def_teacher_ckpt = os.path.join(OUT_DIR, f"{phase_name}_teacher.pt")
    torch.save(ppo_def.def_net.state_dict(), def_teacher_ckpt)
    print(f"[{phase_name.upper()} TEACHER] Saved defender teacher to {def_teacher_ckpt}")

    metrics_path = os.path.join(OUT_DIR, f"train_metrics_{phase_name}_teacher.npz")
    np.savez(metrics_path, **metrics_def)
    print(f"[{phase_name.upper()} TEACHER] Saved metrics to {metrics_path}")

    try:
        del ppo_def, metrics_def
    except:
        pass

    # Student distill (skipped for Na>1 by DISTILL flag)
    student_out = None
    if DISTILL:
        cfg_student = config_for_train(
            attacker_mode=attacker_mode,
            train_role="def",
        )
        cfg_student["num_attackers"] = int(cfg_teacher["num_attackers"])  # =1 in this branch
        cfg_student["use_ukf"] = True
        cfg_student["seed"] = cfg_teacher["seed"] + 1

        if extra_train_cfg is not None:
            cfg_student.update(extra_train_cfg)
            cfg_student["num_attackers"] = int(cfg_teacher["num_attackers"])  # keep 1

        build_dyn(cfg_student)

        if cfg_student["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"[{phase_name.upper()} STUDENT] device='cuda' but CUDA not available.")
        print(f"[{phase_name.upper()} STUDENT] Using device: {cfg_student['device']}")

        student_out = os.path.join(OUT_DIR, f"{phase_name}_ukf_student.pt")
        student, metrics_student = distill_from_teacher(
            cfg_student, def_teacher_ckpt, out_path=student_out,
        )
        print(f"[{phase_name.upper()} STUDENT] Distilled UKF student saved to {student_out}")

        distill_metrics_path = os.path.join(OUT_DIR, f"distill_metrics_{phase_name}_student.npz")
        np.savez(distill_metrics_path, **metrics_student)
        print(f"[{phase_name.upper()} STUDENT] Saved distillation metrics to {distill_metrics_path}")

        try:
            del student, metrics_student
        except:
            pass

    return def_teacher_ckpt, student_out


def train_attacker_with_distill(
    phase_name: str,
    attacker_mode: str,
    extra_train_cfg: Dict[str, Any] | None = None,
):
    """
    Same structure as your original wrapper, with 1v2 added via cfg["num_attackers"]=NUM_ATTACKERS.

    NOTE:
    - We keep your BC pretrain -> PPO attacker training structure.
    - Distillation of attacker is still not done (same as your original note).
    """
    cfg_teacher = config_for_train(
        attacker_mode=attacker_mode,
        train_role="att",
    )
    cfg_teacher["num_attackers"] = int(cfg_teacher.get("num_attackers", NUM_ATTACKERS))
    cfg_teacher["use_ukf"] = False  # teacher is full-state
    cfg_teacher["def_ckpt_path"] = def0_teacher_ckpt
    cfg_teacher["freeze_defender"] = True

    if extra_train_cfg is not None:
        cfg_teacher.update(extra_train_cfg)
        cfg_teacher["num_attackers"] = int(cfg_teacher.get("num_attackers", NUM_ATTACKERS))

    # Attacker distill still off (as in your original)
    DISTILL = bool(cfg_teacher.get("distill", False)) and (int(cfg_teacher["num_attackers"]) == 1)

    build_dyn(cfg_teacher)

    # --- BC pretrain attacker from rule controller (now supports Na>1) ---
    att_bc_ckpt = os.path.join(OUT_DIR, "att1_bc_init.pt")
    cfg_for_bc = cfg_teacher.copy()
    cfg_for_bc["seed"] = cfg_teacher["seed"] + 123

    pretrain_attacker_from_rule(
        cfg_for_bc,
        out_path=att_bc_ckpt,
        steps_per_env=int(cfg_for_bc.get("steps_per_env", 256)),
        total_updates=int(cfg_for_bc.get("att_bc_updates", 200)),
        bc_epochs=int(cfg_for_bc.get("att_bc_epochs", 4)),
        bc_mb_size=int(cfg_for_bc.get("att_bc_mb_size", 2048)),
    )

    cfg_teacher["att_init_path"] = att_bc_ckpt

    # Re-read DISTILL after updates (same idea as your original)
    DISTILL = bool(cfg_teacher.get("distill", False)) and (int(cfg_teacher["num_attackers"]) == 1)

    if cfg_teacher["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"[{phase_name.upper()} TEACHER] device='cuda' but CUDA not available.")
    print(f"[{phase_name.upper()} TEACHER] Using device: {cfg_teacher['device']}")
    print(f"[{phase_name.upper()}] distill={DISTILL}")

    ppo_att, metrics_att = train(cfg_teacher)

    att_teacher_ckpt = os.path.join(OUT_DIR, f"{phase_name}_teacher.pt")
    torch.save(ppo_att.att_net.state_dict(), att_teacher_ckpt)
    print(f"[{phase_name.upper()} TEACHER] Saved attacker teacher to {att_teacher_ckpt}")

    metrics_path = os.path.join(OUT_DIR, f"train_metrics_{phase_name}_teacher.npz")
    np.savez(metrics_path, **metrics_att)
    print(f"[{phase_name.upper()} TEACHER] Saved metrics to {metrics_path}")

    try:
        del ppo_att, metrics_att
    except:
        pass

    student_out = None
    if DISTILL:
        # This branch is effectively unreachable for Na=2 (forced off),
        # but left structurally identical.
        cfg_student = config_for_train(
            attacker_mode=attacker_mode,
            train_role="att",
        )
        cfg_student["num_attackers"] = int(cfg_teacher["num_attackers"])  # would be 1 here
        cfg_student["use_ukf"] = True
        cfg_student["seed"] = cfg_teacher["seed"] + 1

        if extra_train_cfg is not None:
            cfg_student.update(extra_train_cfg)
            cfg_student["num_attackers"] = int(cfg_teacher["num_attackers"])

        build_dyn(cfg_student)

        if cfg_student["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"[{phase_name.upper()} STUDENT] device='cuda' but CUDA not available.")
        print(f"[{phase_name.upper()} STUDENT] Using device: {cfg_student['device']}")

        student_out = os.path.join(OUT_DIR, f"{phase_name}_ukf_student.pt")
        student, metrics_student = distill_from_teacher(
            cfg_student, att_teacher_ckpt, out_path=student_out,
        )
        print(f"[{phase_name.upper()} STUDENT] Distilled UKF student saved to {student_out}")

        distill_metrics_path = os.path.join(OUT_DIR, f"distill_metrics_{phase_name}_student.npz")
        np.savez(distill_metrics_path, **metrics_student)
        print(f"[{phase_name.upper()} STUDENT] Saved distillation metrics to {distill_metrics_path}")

        try:
            del student, metrics_student
        except:
            pass

    return att_teacher_ckpt, student_out


# =============================================================
# Main: stepped training (Def₀ -> Att₁ -> Def₁) + defender distillation
# (identical prints + flow; only adds num_attackers=2 via wrappers)
# =============================================================
if __name__ == "__main__":
    OUT_DIR = "Training_Policy"
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg_distillation = config_for_train(
        attacker_mode="rl",   # attacker is RL now
        train_role="att",     # PPO will only update attacker
    )
    cfg_distillation["num_attackers"] = int(cfg_distillation.get("num_attackers", NUM_ATTACKERS))
    DISTILL = cfg_distillation["distill"]

    # =========================================================
    # PHASE 0: Defender₀ vs rule-based attacker (teacher + distill)
    # =========================================================
    print("\n===== PHASE 0: Train DEFENDER_0 vs RULE attacker =====")
    def0_teacher_ckpt, def0_student_ckpt = train_defender_with_distill(
        phase_name="def0",
        attacker_mode="rule",
        extra_train_cfg={
            "num_attackers": NUM_ATTACKERS,
        },
    )
    end_phase_cleanup("Cleanup after PHASE 0")

    # =========================================================
    # PHASE 1: Attacker₁ vs fixed Defender₀ (teacher only, for now)
    # =========================================================
    print("\n===== PHASE 1: Train ATTACKER_1 vs frozen DEFENDER_0 =====")
    att1_teacher_ckpt, att1_student_ckpt = train_attacker_with_distill(
        phase_name="att1",
        attacker_mode="rl",
        extra_train_cfg={
            "def_ckpt_path": def0_teacher_ckpt,
            "num_attackers": NUM_ATTACKERS,
            # "freeze_attacker": True,
        },
    )
    end_phase_cleanup("Cleanup after PHASE 1")

    # =========================================================
    # PHASE 2: Defender₁ vs frozen Attacker₁ (teacher + distill)
    # =========================================================
    print("\n===== PHASE 2: Train DEFENDER_1 vs frozen ATTACKER_1 =====")
    def1_teacher_ckpt, def1_student_ckpt = train_defender_with_distill(
        phase_name="def1",
        attacker_mode="rl",
        extra_train_cfg={
            "att_ckpt_path": att1_teacher_ckpt,
            "freeze_attacker": True,
            "num_attackers": NUM_ATTACKERS,
        },
    )
    end_phase_cleanup("Cleanup after PHASE 2")

    PLOTS_ROOT = os.path.join(OUT_DIR, "Plots")

    # stage folders
    PLOTS_DEF0 = os.path.join(PLOTS_ROOT, "def0")
    PLOTS_ATT1 = os.path.join(PLOTS_ROOT, "att1")
    PLOTS_DEF1 = os.path.join(PLOTS_ROOT, "def1")
    PLOTS_COMP = os.path.join(PLOTS_ROOT, "comparisons")

    for d in [PLOTS_DEF0, PLOTS_ATT1, PLOTS_DEF1, PLOTS_COMP]:
        os.makedirs(d, exist_ok=True)

    # ---- def0 ----
    m_def0_teacher = load_npz_metrics(os.path.join(OUT_DIR, "train_metrics_def0_teacher.npz"))
    plot_training_metrics(
        m_def0_teacher,
        title="def0_teacher",
        smooth="ema",
        smooth_param=0.2,
        show=False,
        out_dir=PLOTS_DEF0,
        save_prefix="def0_teacher",
    )

    # ---- att1 ----
    m_att1_teacher = load_npz_metrics(os.path.join(OUT_DIR, "train_metrics_att1_teacher.npz"))
    plot_training_metrics(
        m_att1_teacher,
        title="att1_teacher",
        smooth="ema",
        smooth_param=0.2,
        show=False,
        out_dir=PLOTS_ATT1,
        save_prefix="att1_teacher",
    )

    # ---- def1 ----
    m_def1_teacher = load_npz_metrics(os.path.join(OUT_DIR, "train_metrics_def1_teacher.npz"))
    plot_training_metrics(
        m_def1_teacher,
        title="def1_teacher",
        smooth="ema",
        smooth_param=0.2,
        show=False,
        out_dir=PLOTS_DEF1,
        save_prefix="def1_teacher",
    )

    # ---- comparisons ----
    plot_compare_phases(
        [
            ("def0_teacher", os.path.join(OUT_DIR, "train_metrics_def0_teacher.npz")),
            ("def1_teacher", os.path.join(OUT_DIR, "train_metrics_def1_teacher.npz")),
        ],
        metric="R_def_mean",
        ylabel="Mean defender return",
        title="Defender return across phases",
        smooth="ema",
        smooth_param=0.2,
        show=False,
        out_dir=PLOTS_COMP,
        filename="compare__R_def_mean__def0_vs_def1.png",
    )

    print("Saved plots under:", PLOTS_ROOT)
    print(" -", PLOTS_DEF0)
    print(" -", PLOTS_ATT1)
    print(" -", PLOTS_DEF1)
    print(" -", PLOTS_COMP)

    print("\n===== ALL PHASES COMPLETE =====")
    print(f"Defender_0 teacher:  {def0_teacher_ckpt if def0_teacher_ckpt else '(skipped)'}")
    print(f"Defender_0 student:  {def0_student_ckpt if def0_student_ckpt else '(skipped)'}")

    print(f"Attacker_1 teacher:  {att1_teacher_ckpt if att1_teacher_ckpt else '(skipped)'}")
    print(f"Attacker_1 student:  {att1_student_ckpt if att1_student_ckpt else '(skipped)'}")

    print(f"Defender_1 teacher:  {def1_teacher_ckpt if def1_teacher_ckpt else '(skipped)'}")
    print(f"Defender_1 student:  {def1_student_ckpt if def1_student_ckpt else '(skipped)'}")