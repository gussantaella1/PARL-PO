"""
rl_loop_multi.py
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

# from __future__ import annotations

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
# Environment (2 agents under HCW; attacker may be rule-based)
# =============================================================
class Env:
    """
    State per agent: [px,py,pz,vx,vy,vz]; s = concat(defender, attacker).
    Observation: [p1-center, p2-center, (p2-p1), v1, v2]  (size = 5*D)

    Per-step rewards (distances normalized by R^2):
      r_def = +αΔd2 + k_pos d2 - k_rel rel2 - k_cent d1 - k_vel||v1||^2 - k_vrad*vrad1^2 - λD||a1||^2 - wall1
      r_att = -αΔd2 - k_pos d2 + k_rel rel2             - k_vel||v2||^2                           - wall2 - λA||a2||^2

    Terminal bonus at done: r_def += β d2 - 0.10 d1, r_att -= β d2.
    """
    def __init__(self, cfg: Dict[str, Any]):
        # self.scale_invariant = bool(cfg["scale_invariant"])

        self.cfg = cfg
        self.num_attackers = int(cfg.get("num_attackers", 1))
        self.D = int(cfg["D"])
        self.dt = float(cfg["dt"])
        self.T  = int(cfg["T"])

        self.num_attackers = int(cfg.get("num_attackers", 1))  # NEW
        Na = self.num_attackers

        ar = cfg["arena"]
        if ar["type"] != "sphere":
            raise ValueError("Only 'sphere' arena is implemented.")
        self.center = np.array([ar["cx"], ar["cy"], (ar["cz"] if self.D == 3 else 0.0)], dtype=np.float32)[:self.D]
        self.radius = float(ar["r"])


        self.def_keepout_buffer_m = float(cfg.get("def_keepout_buffer_m", 0.0))
        # self.def_target_hit_buffer_frac = float(cfg.get("def_target_hit_buffer_frac", 0.0))


        umax = float(cfg["umax"]) ; self.u_lo, self.u_hi = -umax, +umax

        Ad = cfg["dyn"]["Ad"]; Bd = cfg["dyn"]["Bd"]
        if Ad is None or Bd is None:
            raise ValueError("cfg['dyn']['Ad'] and ['Bd'] must be provided (call build_dyn(cfg)).")
        self.Ad = np.asarray(Ad, dtype=np.float32)
        self.Bd = np.asarray(Bd, dtype=np.float32)

        self.nx_agent = 2 * self.D
        self.nx_total = (1 + Na) * self.nx_agent   # use num_attackers
        self.act_dim = self.D

        x0 = np.asarray(cfg["x0"], dtype=float)
        # Allow x0 with either 2 rows (def + one att) or (1+Na) rows
        if x0.shape[0] == 2 and self.num_attackers > 1:
            # row 0: defender, row 1: “prototype” attacker
            base_def = x0[0]
            base_att = x0[1]
            x0_full = np.zeros((1 + Na, 2*self.D), dtype=float)
            x0_full[0] = base_def
            x0_full[1] = base_att
            # other attackers will be randomized at reset()
        elif x0.shape[0] == 1 + self.num_attackers:
            x0_full = x0
        else:
            raise ValueError(
                f"x0 shape {x0.shape} incompatible with num_attackers={self.num_attackers}"
            )
        self._x0 = x0_full



        # reward params
        self.alpha = float(cfg["dense_coef"])  # Δd2 stability term
        self.beta  = float(cfg["term_coef"])   # terminal ±d2
        self.k_pos = float(cfg.get("step_pos_coef", 0.0))
        self.k_rel = float(cfg.get("step_rel_coef", 0.0))
        self.k_cent_base = float(cfg.get("def_center_coef", 0.0))
        self.k_cent = self.k_cent_base         # may be annealed during training
        self.k_vel = float(cfg.get("step_vel_coef", 0.0))
        self.lD    = float(cfg["effort_def"])
        self.lA    = float(cfg["effort_att"])
        self.wallK = float(cfg["wall_penalty"])
        self.soft_wall = float(cfg.get("soft_wall_start", 0.7))
        self.margin = float(cfg["arena_terminate_margin"])  # 1.0 = at radius

        # NEW: defender center keep-out (normalized radius wrt arena R)
        self.def_center_safe_radius = float(cfg.get("def_center_safe_radius", 0.05))
        self.def_center_avoid_coef  = float(cfg.get("def_center_avoid_coef", 10.0))

        # NEW: attacker "hit object" termination around center (normalized wrt arena R)

        oi = cfg.get("oi", {})


        self.oi_radius = float(oi.get("r", 0.0))
        self.oi_radius_norm = self.oi_radius / self.radius if self.radius > 0 else 0.0

        self.att_target_hit_radius = float(cfg.get("att_target_hit_radius", 0.0))
        self.att_target_hit_penalty_def = float(cfg.get("att_target_hit_penalty_def", 0.0))
        self.att_target_hit_reward_att  = float(cfg.get("att_target_hit_reward_att", 0.0))

        self.def_oi_safety_buffer = float(cfg.get("def_oi_safety_buffer", 0.0))
        self.def_target_hit_penalty_def = float(cfg.get("def_target_hit_penalty_def", 0.0))
        self.def_target_hit_reward_att  = float(cfg.get("def_target_hit_reward_att", 0.0))


        # NEW: collision termination (defender vs any attacker)
        self.collision_radius_m    = float(cfg.get("collision_radius_m", 0.2))  # meters; 0 disables
        self.collision_penalty_def = float(cfg.get("collision_penalty_def", 0.0))
        self.collision_penalty_att = float(cfg.get("collision_penalty_att", 0.0))


        # ---- UKF / measurement model knobs ----
        self.use_ukf          = bool(cfg.get("use_ukf", False))
        self.use_meas_reward  = bool(cfg.get("use_meas_reward", False))
        self.meas_innov_coef  = float(cfg.get("meas_innov_coef", 0.0))  # weight on innovation^2
        self.meas_cov_coef    = float(cfg.get("meas_cov_coef", 0.0))    # weight on trace(P_pos)

        if self.use_ukf and self.D != 3:
            raise ValueError("UKF / bearing-only measurement currently implemented for D=3 only.")

        self.ukf = None
        self._latest_meas_innov = 0.0
        self._latest_meas_trP   = 0.0

        if self.use_ukf:
            ukf_cfg = cfg.get("ukf", {})

            # Initial covariance P0
            pos_std0 = float(ukf_cfg.get("pos_std0", 0.2 * self.radius))
            vel_std0 = float(ukf_cfg.get("vel_std0", 0.01))
            P0 = np.diag(
                [pos_std0**2, pos_std0**2, pos_std0**2,
                 vel_std0**2, vel_std0**2, vel_std0**2]
            )

            # Process noise Q (simple isotropic default)
            q_scale = float(ukf_cfg.get("Q_scale", 1e-5))
            Q = q_scale * np.eye(6, dtype=float)

            # Measurement noise R (az, el)
            sigma_az = float(ukf_cfg.get("sigma_az", np.deg2rad(0.5)))
            sigma_el = float(ukf_cfg.get("sigma_el", np.deg2rad(0.5)))
            Rm = np.diag([sigma_az**2, sigma_el**2])

            self._ukf_P0 = P0
            self._ukf_Q  = Q
            self._ukf_R  = Rm

            # Keep some init stds around for reset-time randomization
            self._ukf_init_pos_std = float(ukf_cfg.get("init_pos_std", pos_std0))
            self._ukf_init_vel_std = float(ukf_cfg.get("init_vel_std", vel_std0))


        # Training-only initial-condition randomization
        # Defaults to "fixed" if keys are absent (e.g., eval config)
        self.train_ic_mode = cfg.get("train_ic_mode", "fixed")
        self.train_ic_vmax = float(cfg.get("train_ic_vmax", 0.05))
        self.train_min_sep = float(cfg.get("train_min_sep", 1.0))


        self.state = None
        self.t = 0
        self._d2_prev = None  # will become np.ndarray shape (Na,)

        self._d2_prev_def = None  # scalar: defender shaping uses "threat attacker" only


        # =========================
        # Attacker reward knobs (read once from cfg)
        # =========================
        att = cfg.get("att_reward", {})
        att_rule = cfg.get("att_rule", {})

        self.k_att_prog     = float(att.get("k_prog", 2.0))
        self.k_att_cent     = float(att.get("k_cent", 0.0))
        self.k_att_close    = float(att.get("k_close", 2.0))

        # IMPORTANT: this fixes your old crash too
        self.att_min_sep    = float(att.get("min_sep", att_rule.get("min_sep", 3.0)))

        self.k_att_vrad     = float(att.get("k_vrad", 0.5))
        self.k_att_wall     = float(att.get("k_wall", self.wallK))
        self.att_wall_power = float(att.get("wall_power", 4.0))


        self.hit_buffer_def = float(self.def_oi_safety_buffer)
        self.hit_buffer_att = float(self.att_target_hit_radius)




    def reset(self) -> np.ndarray:
        self.t = 0
        mode = self.train_ic_mode
        Na = self.num_attackers

        if mode == "fixed":
            # Base ICs from __init__
            x0 = self._x0.copy()
            jit = self.cfg.get("x0_jitter", None)
            if jit:
                jp = float(jit.get("pos", 0.0))
                jv = float(jit.get("vel", 0.0))
                # defender
                x0[0, 0:self.D]        += np.random.uniform(-jp, jp, size=(self.D,))
                x0[0, self.D:2*self.D] += np.random.uniform(-jv, jv, size=(self.D,))
                # attackers
                for k in range(Na):
                    idx = 1 + k
                    x0[idx, 0:self.D]        += np.random.uniform(-jp, jp, size=(self.D,))
                    x0[idx, self.D:2*self.D] += np.random.uniform(-jv, jv, size=(self.D,))

        elif mode == "random_shell":
            # Sample defender near center, attackers near outer shell
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

            # defender
            p1 = sample_in_ball(r_def_min, r_def_max)
            v1 = np.random.uniform(-v_max, v_max, size=(self.D,))

            # attackers: enforce min_sep to defender and between attackers
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
            x0[0, 0:self.D]        = p1
            x0[0, self.D:2*self.D] = v1
            for k in range(Na):
                idx = 1 + k
                x0[idx, 0:self.D]        = pA[k]
                x0[idx, self.D:2*self.D] = vA[k]

        else:
            raise ValueError(f"Unknown train_ic_mode='{mode}'")

        # ---- flatten to state (all agents) ----
        self.state = x0.reshape(-1)

        # With multi-attacker _unpack, we get lists:
        p1, v1, pA_list, vA_list = self._unpack(self.state)
        # For UKF and rewards we still only use the *first* attacker
        p2 = pA_list[0]
        v2 = vA_list[0]

        # ---- UKF init (only supported for 1 attacker right now) ----
        if self.use_ukf and self.num_attackers != 1:
            raise NotImplementedError("UKF currently only implemented for num_attackers=1")

        if self.use_ukf:
            pos_noise = np.random.normal(
                scale=self._ukf_init_pos_std,
                size=p2.shape
            )
            vel_noise = np.random.normal(
                scale=self._ukf_init_vel_std,
                size=v2.shape
            )
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
            self._latest_meas_trP   = float(np.trace(self.ukf.P[0:3, 0:3]))
        else:
            self.ukf = None
            self._latest_meas_innov = 0.0
            self._latest_meas_trP   = 0.0

        # ---- initialize d2_prev using belief if UKF is on ----
        if self.use_ukf and (self.ukf is not None):
            p2_geom = self.ukf.x[:self.D].copy()

            r_est = np.linalg.norm(p2_geom - self.center)
            R = self.radius
            clip_factor = float(self.cfg.get("belief_clip_factor", 2.0))
            r_max = clip_factor * R

            if (not np.isfinite(r_est)) or (r_est > r_max):
                if np.isfinite(r_est) and r_est > 1e-9:
                    direction = (p2_geom - self.center) / r_est
                    p2_geom = self.center + direction * r_max
                else:
                    p2_geom = p2.copy()

                if self.cfg.get("reset_ukf_on_diverge", True):
                    self.ukf.x[:self.D] = p2
                    self.ukf.x[self.D:2*self.D] = v2
                    self.ukf.P = self._ukf_P0.copy()
        else:
            p2_geom = p2

        p1, v1, pA_list, vA_list = self._unpack(self.state)

        d2_prev = np.zeros((self.num_attackers,), dtype=np.float32)
        for k, pA in enumerate(pA_list):
            d2_raw = float(np.dot(pA - self.center, pA - self.center))
            d2_prev[k] = d2_raw / (self.radius**2)
        self._d2_prev = d2_prev

        # defender shaping baseline = min attacker d2 (threat attacker)
        self._d2_prev_def = float(np.min(d2_prev)) if d2_prev.size > 0 else 0.0

        return self._obs()


    def step(self, a1_env: np.ndarray, aA_env: np.ndarray, reward_mode: str = "both"):
        """
        a1_env: (D,)
        aA_env: (Na, D) or (D,) if Na=1
        reward_mode: "def", "att", or "both"
        """
        need_def = reward_mode in ("def", "both")
        need_att = reward_mode in ("att", "both")

        # Defender action
        a1 = np.clip(np.asarray(a1_env, float).reshape(self.D,), self.u_lo, self.u_hi)

        # Attacker actions => force (Na, D)
        aA = np.asarray(aA_env, float)
        if aA.ndim == 1:
            aA = aA.reshape(1, self.D)
        else:
            aA = aA.reshape(self.num_attackers, self.D)
        aA = np.clip(aA, self.u_lo, self.u_hi)

        # For convenience in reward calc we reference the "primary" attacker as index 0
        a2 = aA[0]


        # propagate true state
        self.state = self._plant_step(self.state, a1, aA)
        self.t += 1

        # Unpack new state
        p1, v1, pA_list, vA_list = self._unpack(self.state)
        p2 = pA_list[0]
        v2 = vA_list[0]

        # --- NEW: threat attacker for defender shaping (closest to center) ---
        d2_all = np.array(
            [float(np.dot(pA - self.center, pA - self.center)) / (self.radius**2) for pA in pA_list],
            dtype=np.float32,
        )
        k_threat = int(np.argmin(d2_all)) if d2_all.size > 0 else 0
        d2_threat = float(d2_all[k_threat]) if d2_all.size > 0 else 0.0

        if self._d2_prev_def is None:
            self._d2_prev_def = d2_threat
        delta_d2_threat = d2_threat - float(self._d2_prev_def)



        # ---- UKF (only if enabled; you can also gate this further if you want) ----
        meas_innov_sq = 0.0
        meas_trPpos   = 0.0

        if self.use_ukf:
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
            self._latest_meas_trP   = meas_trPpos
        else:
            self._latest_meas_innov = 0.0
            self._latest_meas_trP   = 0.0

        # # ---- choose geometry p2_geom (belief if UKF on, else truth) ----
        # if self.use_ukf and (self.ukf is not None):
        #     p2_geom = self.ukf.x[:self.D].copy()
        #     # (keep your sanity clip logic here if you want)
        # else:
        #     p2_geom = p2

        # ---- shared geometry needed by whichever reward(s) we compute ----
        # d2 and rel2 are used by both rewards
        d2_raw = float(np.dot(p2 - self.center, p2 - self.center))
        d2 = d2_raw / (self.radius**2)
        d2_prev0 = float(self._d2_prev[0] if self._d2_prev is not None else d2)
        delta_d2 = d2 - d2_prev0

        rel2 = float(np.dot((p2 - p1), (p2 - p1))) / (self.radius**2)

        # d1 only needed for defender reward (step/terminal), but cheap; compute only if needed
        if need_def:
            d1_raw = float(np.dot(p1 - self.center, p1 - self.center))
            d1 = d1_raw / (self.radius**2)
        else:
            d1 = 0.0  # placeholder

        # ---- wall penalties: compute only what you need ----
        wall1 = 0.0
        wall2 = 0.0
        center_keepout = 0.0
        vrad1 = 0.0
        k_vrad = self.k_vel * 3.0

        rho1 = np.linalg.norm(p1 - self.center)/ self.radius
        rho2 = np.linalg.norm(p2 - self.center) / self.radius

        v_scale = self.radius / self.dt


        if need_def:
            
            # rho1_rel_to_center = np.linalg.norm(p1 - self.center)
            wall1 = ((max(0.0, rho1 - self.soft_wall))**2) * self.wallK

            # defender keep-out (METERS): keep defender outside (oi_radius + buffer)
            center_keepout = 0.0
            if self.oi_radius > 0.0:
                r_keepout_m = self.oi_radius + self.def_keepout_buffer_m
                d1_m = float(np.linalg.norm(p1 - self.center))  # meters

                if r_keepout_m > 0.0 and d1_m < r_keepout_m:
                    gap_m = (r_keepout_m - d1_m)  # meters inside keepout
                    # normalize gap by arena radius so penalty scale is comparable across different R
                    gap = gap_m / (self.radius + 1e-9)
                    center_keepout = self.def_center_avoid_coef * (gap_m * gap_m)


            # defender radial velocity
            rhat1 = (p1 - self.center)
            rnorm = np.linalg.norm(rhat1) + 1e-9

            v1n2 = float(np.dot(v1, v1)) / (v_scale**2)
            a1n2 = float(np.dot(a1, a1)) / (self.u_hi**2)

            vrad1 = float(np.dot(v1, rhat1 / rnorm)) / v_scale  # dimensionless

        if need_att:
            v2n2 = float(np.dot(v2, v2)) / (v_scale**2)
            a2n2 = float(np.dot(a2, a2)) / (self.u_hi**2)
            
            wall2 = ((max(0.0, rho2 - self.soft_wall))**2) * self.wallK

            # attacker wall penalties etc should also be per attacker
            wall2_vec = np.zeros((self.num_attackers,), dtype=np.float32)
            a2n2_vec  = np.zeros((self.num_attackers,), dtype=np.float32)

            for k in range(self.num_attackers):
                p2k = pA_list[k]
                v2k = vA_list[k]
                a2k = aA[k]

                rho2k = np.linalg.norm(p2k - self.center) / self.radius
                wall2_vec[k] = ((max(0.0, rho2k - self.soft_wall))**2) * self.wallK

                a2n2_vec[k] = float(np.dot(a2k, a2k)) / (self.u_hi**2)

        # ---- compute only requested reward(s) ----
        r1 = 0.0
        r2 = 0.0

        if need_def:
            r1 = (
                # self.alpha * delta_d2
                # + self.k_pos * d2

                self.alpha * delta_d2_threat
                + self.k_pos * d2_threat
                # - self.k_rel * rel2
                # - self.k_vel * v1n2
                # - k_vrad * (vrad1**2)
                - self.lD * a1n2
                - wall1
                - center_keepout
            )

        # if need_att:
        #     # normalized squared distance to center (you already computed d2)
        #     # d2 = ||p2-center||^2 / R^2

        #     # normalized effort (you already computed a2n2)
        #     # a2n2 = ||a2||^2 / umax^2

        #     r2 = (
        #         - self.k_att_cent * d2
        #         - self.lA         * a2n2
        #         - wall2
        #     )


        r2_vec = np.zeros((self.num_attackers,), dtype=np.float32)

        if need_att: 
            # compute r2 for each attacker
            for k in range(self.num_attackers):
                p2k = pA_list[k]
                d2k_raw = float(np.dot(p2k - self.center, p2k - self.center))
                d2k = d2k_raw / (self.radius**2)

                dist = float(np.linalg.norm(p2k - p1))  # meters
                x = dist / (float(self.att_min_sep) + 1e-9)
                close_pen = max(0.0, 1.0 - x) ** 2

                d2_prev_k = float(self._d2_prev[k] if self._d2_prev is not None else d2k)
                progress = d2_prev_k - d2k

                r2_vec[k] = (
                    + self.k_att_prog * progress
                    - self.k_att_close * close_pen
                    - self.lA * a2n2_vec[k]
                    - wall2_vec[k]
                )

        # if need_att:
        #     # NOTE: we still use d2_prev stored as scalar currently (from attacker0).
        #     # For multi-attacker RL, we should track per-attacker d2_prev. We'll fix that below.
        #     pass


        






        # if need_att:
        #     R = self.radius

        #     # --- Progress reward (positive when moving toward center) ---
        #     d2_prev = (self._d2_prev if self._d2_prev is not None else d2)
        #     progress = (d2_prev - d2)  # >0 means closer to center

        #     # --- Center distance term (optional) ---
        #     cent_pen = d2  # normalized squared

        #     # --- Keep-out penalty (ONLY if too close to defender) ---
        #     dist = float(np.linalg.norm(p2_geom - p1))  # meters
        #     close_pen = 0.0
        #     if dist < self.att_min_sep:
        #         gap = (self.att_min_sep - dist) / R
        #         close_pen = gap * gap

        #     # --- Radial-speed penalty (reduces "fly through center then slam wall") ---
        #     rhat2 = (p2_geom - self.center)
        #     rnorm2 = np.linalg.norm(rhat2) + 1e-9
        #     vrad2 = float(np.dot(v2, rhat2 / rnorm2)) / R  # normalized radial speed

        #     # --- Wall barrier (starts at soft_wall) ---
        #     rho2 = np.linalg.norm(p2_geom - self.center) / R
        #     wall_gap = max(0.0, rho2 - self.soft_wall)
        #     wall2_att = (wall_gap ** self.att_wall_power) * self.k_att_wall

        #     # --- Regularization (same style as your old r2) ---
        #     effort_pen = float(np.dot(a2, a2))
        #     speed_pen  = float(np.dot(v2, v2))

        #     r2 = (
        #         + self.k_att_prog * progress
        #         - self.k_att_cent * cent_pen
        #         - self.k_att_close * close_pen
        #         - self.k_att_vrad * (vrad2 ** 2)
        #         - wall2_att
        #         - self.lA * effort_pen
        #         - self.k_vel * speed_pen
        #     )



        # ---- termination scenariosalways uses TRUE state ----


        # # target hit: (keep your current meaning = first attacker hits target)

        hit_target = False
        # p2_true = pA_list[0]
        # # rho2_true = np.linalg.norm(p2_true - self.center) / self.radius
        # rho2_rel_to_target = np.linalg.norm(p2_true - self.center)

        # hit_target = False
        # if self.att_target_hit_radius > 0.0 and rho2_rel_to_target <= (1 + self.att_target_hit_radius)*self.oi_radius:
        #     hit_target = True

        # attacker target hit: (keep your current meaning = first attacker hits target)

        # -------------------------
        # termination (TRUE state)
        # -------------------------

        rho_def = np.linalg.norm(p1 - self.center) / self.radius

        thresh_def = (1.0 + self.hit_buffer_def) * self.oi_radius_norm
        thresh_att = (1.0 + self.hit_buffer_att) * self.oi_radius_norm

        # Defender hits target?
        def_hit_target = (self.oi_radius_norm > 0.0) and (rho_def <= thresh_def)

        # Any attacker hits target?
        att_hit_target = False
        att_hit_idx = -1
        if self.oi_radius_norm > 0.0:
            for k, pA_true in enumerate(pA_list):
                rhoA = np.linalg.norm(pA_true - self.center) / self.radius
                if rhoA <= thresh_att:
                    att_hit_target = True
                    att_hit_idx = k
                    break

        hit_target = att_hit_target or def_hit_target

        # collision: defender within collision_radius_m of ANY attacker (TRUE distance)
        collision = False
        if self.collision_radius_m > 0.0:
            for pA_true in pA_list:
                if np.linalg.norm(pA_true - p1) <= self.collision_radius_m:
                    collision = True
                    break

        # out-of-bounds for defender
        oob1 = (rho1 >= self.margin)

        # out-of-bounds for ANY attacker
        oob2_any = False
        for pA_true in pA_list:
            rhoA_true = np.linalg.norm(pA_true - self.center) / self.radius
            if rhoA_true >= self.margin:
                oob2_any = True
                break

        done = (oob1 or oob2_any or hit_target or collision)

        # -------------------------
        # terminal shaping
        # -------------------------
        if need_def and done:
            if collision:
                r1 -= self.collision_penalty_def
            if oob1:
                r1 -= self.wallK
            if att_hit_target:
                r1 -= self.def_target_hit_penalty_def
            if def_hit_target:
                r1 -= self.att_target_hit_penalty_def

        if need_att and done:
            if collision:
                r2_vec[:] -= self.collision_penalty_att
            if oob2_any:
                r2_vec[:] -= self.wallK

            # reward only the attacker that actually hit (att_hit_idx), if any
            if att_hit_target and (att_hit_idx >= 0):
                r2_vec[:] += self.att_target_hit_reward_att


            # if done and def_hit_target: r2 += self.def_target_hit_reward_att

        # if need_att and done:
        #     # reward only if attacker ended near target
        #     # (oi_radius_norm > 0 means you actually defined a target)
        #     if self.oi_radius_norm > 0.0:
        #         if att_hit_target:
        #             r2 += float(self.beta)   # e.g. beta = 1.0 or 2.0
        #         else:
        #             r2 -= float(self.beta)   # optional: discourage ending without hitting





        # track d2_prev based on the geometry used for reward (same as your current logic)
        d2_now = np.zeros((self.num_attackers,), dtype=np.float32)
        for k, pA in enumerate(pA_list):
            d2_raw = float(np.dot(pA - self.center, pA - self.center))
            d2_now[k] = d2_raw / (self.radius**2)
        self._d2_prev = d2_now
        # keep defender baseline consistent with threat definition
        self._d2_prev_def = float(np.min(d2_now))  # or d2_threat if you computed it

        # info: you can also gate what you store here if you want
        # ---- always compute these for logging ----
        d1_true_norm = float(np.dot(p1 - self.center, p1 - self.center)) / (self.radius**2)
        # d2_true_norm = float(np.dot(p2 - self.center, p2 - self.center)) / (self.radius**2)

        d2_true_norm = d2_threat
        # d2_belief_norm = float(np.dot(p2_geom - self.center, p2_geom - self.center)) / (self.radius**2)

        info = {
            "t": self.t,
            "d2_norm": d2,                 # whatever you used for reward (belief if UKF)
            "rel2_norm": rel2,
            "oob_def": bool(oob1),
            "oob_att": bool(oob2_any),
            "hit_target": bool(hit_target),

            # NEW: what your logger expects
            "d1_true_norm": d1_true_norm,
            "d2_true_norm": d2_true_norm,
            # "d2_belief_norm": d2_belief_norm,

            "collision": bool(collision),

        }

        if self.use_ukf:
            info["meas_innov_sq"] = meas_innov_sq
            info["ukf_trPpos"] = meas_trPpos

        info["threat_idx"] = k_threat


        r2_out = r2_vec if need_att else np.zeros((self.num_attackers,), dtype=np.float32)
        return self._obs(), float(r1), r2_out, bool(done), info




    def _obs(self) -> np.ndarray:
        p1, v1, pA_list, vA_list = self._unpack(self.state)

        # For now, rule attackers don't use obs; obs is defender-centric.
        # We stack all attacker info.
        if self.use_ukf and (self.ukf is not None) and self.num_attackers == 1:
            # existing UKF path for single attacker
            p2_obs = self.ukf.x[:self.D]
            v2_obs = self.ukf.x[self.D:2*self.D]
            pA_obs = [p2_obs]
            vA_obs = [v2_obs]
        else:
            pA_obs = pA_list
            vA_obs = vA_list

        # --- NEW: reorder attackers so index 0 is the most threatening (closest to center) ---
        if len(pA_obs) > 1:
            # sort by distance-to-center (squared) ascending
            d2_list = [float(np.dot(pA - self.center, pA - self.center)) for pA in pA_obs]
            order = np.argsort(d2_list)
            pA_obs = [pA_obs[i] for i in order]
            vA_obs = [vA_obs[i] for i in order]

        # build obs = [p1c, pA1c, ..., pANc, rel1, ..., relN, v1, vA1, ..., vAN]
        p1c = p1 - self.center
        parts = [p1c]

        # positions (centered)
        for pA in pA_obs:
            parts.append(pA - self.center)

        # relative positions
        for pA in pA_obs:
            parts.append(pA - p1)

        # defender vel
        parts.append(v1)

        # attacker vels
        for vA in vA_obs:
            parts.append(vA)

        obs = np.concatenate(parts).astype(np.float32)
        return obs
    
    def obs_att_k(self, k: int) -> np.ndarray:
        """
        Per-attacker observation (5D):
        [p1-center, pAk-center, (pAk-p1), v1, vAk]
        Uses UKF only if enabled AND Na==1 AND k==0 (consistent with your current UKF support).
        """
        p1, v1, pA_list, vA_list = self._unpack(self.state)

        if self.use_ukf and (self.ukf is not None) and self.num_attackers == 1 and k == 0:
            p2 = self.ukf.x[:self.D]
            v2 = self.ukf.x[self.D:2*self.D]
        else:
            p2 = pA_list[k]
            v2 = vA_list[k]

        c = self.center
        p1c = p1 - c
        p2c = p2 - c
        rel = p2 - p1
        return np.concatenate([p1c, p2c, rel, v1, v2]).astype(np.float32)


    def obs_att_all(self) -> np.ndarray:
        """
        Returns (Na, 5D) per-attacker observations.
        """
        return np.stack([self.obs_att_k(k) for k in range(self.num_attackers)], axis=0)



    def _plant_step(self, s: np.ndarray, a1: np.ndarray, aA: np.ndarray) -> np.ndarray:
        """
        Propagate defender and all attackers one step.

        s   : flattened state
        a1  : (D,) defender action
        aA  : (Na, D) attacker actions, or (D,) if Na=1
        """
        D  = self.D
        Na = self.num_attackers

        p1, v1, pA_list, vA_list = self._unpack(s)

        # Defender
        x1  = np.concatenate([p1, v1])
        x1n = self.Ad @ x1 + self.Bd @ a1
        p1n, v1n = x1n[:D], x1n[D:]

        # Ensure aA is (Na, D)
        aA = np.asarray(aA, float)
        if aA.ndim == 1:
            aA = aA.reshape(1, D)
        else:
            aA = aA.reshape(Na, D)

        # Attackers
        pA_new = []
        vA_new = []
        for k in range(Na):
            p2 = pA_list[k]
            v2 = vA_list[k]
            x2  = np.concatenate([p2, v2])
            x2n = self.Ad @ x2 + self.Bd @ aA[k]
            pA_new.append(x2n[:D])
            vA_new.append(x2n[D:])

        # Re-flatten to match _unpack layout: [p1, v1, pA0, vA0, pA1, vA1, ...]
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
            pA = s[off + 2*k*D     : off + (2*k+1)*D]
            vA = s[off + (2*k+1)*D : off + (2*k+2)*D]
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
        r1_list, r2_list, done_list, info_list = [], [], [], []

        for i, e in enumerate(self.envs):
            o, R1, R2, d, inf = e.step(a1_env[i], aA_env[i], reward_mode=reward_mode)
            if d:
                o = e.reset()
            obs_next.append(o)
            r1_list.append(R1)
            r2_list.append(R2)   # R2 is now (Na,)
            done_list.append(d)
            info_list.append(inf)

        self.obs = np.stack(obs_next, axis=0)

        r1 = np.array(r1_list, dtype=np.float32)                 # (B,)
        r2 = np.stack(r2_list, axis=0).astype(np.float32)        # (B, Na)
        done = np.array(done_list, dtype=np.float32)             # (B,)

        return self.obs, r1, r2, done, info_list
    
    def get_att_obs(self) -> np.ndarray:
        """
        Returns attacker observations:
        (B, Na, 5D)
        """
        obsA = []
        for e in self.envs:
            obsA.append(e.obs_att_all())
        return np.stack(obsA, axis=0).astype(np.float32)






# =============================================================
# PPO Storage & Advantage
# =============================================================
class RolloutBuffer:
    def __init__(self, obs_dim, act_dim, num_envs, horizon, device):
        self.N, self.T = num_envs, horizon
        self.device = device
        self.obs  = torch.zeros(self.T, self.N, obs_dim, device=device)
        self.act  = torch.zeros(self.T, self.N, act_dim, device=device)
        self.logp = torch.zeros(self.T, self.N, device=device)
        self.val  = torch.zeros(self.T, self.N, device=device)
        self.rew  = torch.zeros(self.T, self.N, device=device)
        self.done = torch.zeros(self.T, self.N, device=device)
        self.next_val = torch.zeros(self.N, device=device)
        self.ptr = 0

    def add(self, obs, act, logp, val, rew, done):
        t = self.ptr

        # Make rollout data "dead" (no grad graph can leak into PPO update)
        self.obs[t]  = obs.detach()
        self.act[t]  = act.detach()
        self.logp[t] = logp.detach()
        self.val[t]  = val.detach()

        self.rew[t]  = rew
        self.done[t] = done
        self.ptr += 1


    def finalize(self, next_val):
        self.next_val = next_val

    def get(self):
        B = self.T * self.N
        obs  = self.obs.reshape(B, -1)
        act  = self.act.reshape(B, -1)
        logp = self.logp.reshape(B)
        val  = self.val.reshape(B)
        rew  = self.rew.reshape(B)
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
        next_v = next_val if t == T-1 else buf.val[t+1]
        delta = buf.rew[t] + gamma * next_v * nonterminal - buf.val[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = adv + buf.val
    A = adv.reshape(T*N)
    R = ret.reshape(T*N)
    A = (A - A.mean()) / (A.std() + 1e-8)
    return A, R


# =============================================================
# Rule-based Attacker Controller
# =============================================================
class AttackerRuleController:
    """
    u = sat_umax( w_center * u_center + w_avoid * u_repulse - w_damp * v2 )

    - u_center: one-step ridge minimizer to drive p2 -> center under (Ad, Bd)
    - u_repulse: repulsion away from defender with a *hard* keep-out:
        * if dist(p2, p1) < min_sep  -> full thrust directly away
        * else                       -> smooth inverse-square repulsion
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

        # Position selection matrix: P * [p, v] = p
        P = np.hstack([
            np.eye(D, dtype=np.float32),
            np.zeros((D, D), dtype=np.float32),
        ])
        self.P = P
        self.F = (self.Bd.T @ P.T).T  # (D, D)
        FtF = self.F.T @ self.F

        # Safe defaults; allow overrides via cfg["att_rule"]
        rule = dict(
            ridge=1e-2,         # ridge for one-step center pull
            w_center=0.5,       # weight for center attraction
            w_avoid=2.0,        # weight for repulsion from defender
            w_damp=0.3,         # linear velocity damping
            min_sep=3.0,        # meters; hard keep-out radius
            repulse_gain=10.0,  # strength of repulsion outside min_sep
        )
        rule.update(cfg.get("att_rule", {}))

        self.lam          = float(rule["ridge"])
        self.w_center     = float(rule["w_center"])
        self.w_avoid      = float(rule["w_avoid"])
        self.w_damp       = float(rule["w_damp"])
        self.min_sep      = float(rule["min_sep"])
        self.repulse_gain = float(rule["repulse_gain"])
        self.umax         = float(cfg["umax"])

        # One-step ridge solution gain for u_center
        self.K = np.linalg.solve(
            FtF + self.lam * np.eye(D, dtype=np.float32),
            self.F.T
        )

    def u_center(self, p2: np.ndarray, v2: np.ndarray) -> np.ndarray:
        x2 = np.concatenate([p2, v2])
        # Next-step position error relative to center (E2)
        E2 = (self.Ad @ x2)[:self.D] - self.center
        return -(self.K @ E2)

    def u_repulse(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """
        Hard keep-out:
          - If dist < min_sep: full thrust directly away from defender.
          - Else: smooth inverse-square repulsion.
        """
        r = p2 - p1
        dist = float(np.linalg.norm(r)) + 1e-9
        r_hat = r / dist

        # Inside keep-out zone: *max thrust* directly away
        if dist < self.min_sep:
            return self.umax * r_hat

        # Outside keep-out: smoother inverse-square repulsion
        # magnitude ≈ repulse_gain / dist^2 (then clipped in act())
        mag = self.repulse_gain / (dist**2)
        return mag * r_hat

    # def act(self,
    #         p1: np.ndarray, v1: np.ndarray,
    #         p2: np.ndarray, v2: np.ndarray) -> np.ndarray:
    #     # Center pull and repulsion
    #     uc = self.u_center(p2, v2)
    #     ur = self.u_repulse(p1, p2)

    #     # Compute separation to optionally *down-weight center* close in
    #     dist = float(np.linalg.norm(p2 - p1))
    #     if dist < self.min_sep:
    #         # Near defender: ignore center attraction
    #         w_center_eff = 0.0
    #     else:
    #         w_center_eff = self.w_center

    #     u = w_center_eff * uc + self.w_avoid * ur - self.w_damp * v2
    #     return np.clip(u, -self.umax, +self.umax)

    def act(self,
                p1: np.ndarray, v1: np.ndarray,
                p2: np.ndarray, v2: np.ndarray) -> np.ndarray:
            """
            Single-attacker control law.

            If you ever have multiple attackers per env, you just call this in a loop
            from the env / PPO:
                u_list = [ctrl.act(p1, v1, pA[k], vA[k]) for k in range(Na)]
            """
            # Center pull and repulsion
            uc = self.u_center(p2, v2)
            ur = self.u_repulse(p1, p2)

            # Compute separation to optionally *down-weight center* when close in
            dist = float(np.linalg.norm(p2 - p1))
            if dist < self.min_sep:
                # Near defender: ignore center attraction
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
        """
        Returns: (Na, D) actions
        """
        u_list = []
        for p2, v2 in zip(pA_list, vA_list):
            u_list.append(self.act(p1, v1, p2, v2))
        return np.stack(u_list, axis=0).astype(np.float32)


    # def act_single(self,
    #                p1: np.ndarray, v1: np.ndarray,
    #                p2: np.ndarray, v2: np.ndarray) -> np.ndarray:
    #     uc = self.u_center(p2, v2)
    #     ur = self.u_repulse(p1, p2)

    #     dist = float(np.linalg.norm(p2 - p1))
    #     if dist < self.min_sep:
    #         w_center_eff = 0.0
    #     else:
    #         w_center_eff = self.w_center

    #     u = w_center_eff * uc + self.w_avoid * ur - self.w_damp * v2
    #     return np.clip(u, -self.umax, +self.umax)

    # def act_multi(self,
    #               p1: np.ndarray, v1: np.ndarray,
    #               pA_list: list[np.ndarray], vA_list: list[np.ndarray]) -> list[np.ndarray]:
    #     u_list = []
    #     for p2, v2 in zip(pA_list, vA_list):
    #         u_list.append(self.act_single(p1, v1, p2, v2))
    #     return u_list
    
    



# =============================================================
# DiffLS Layer & Actor-Critic
# =============================================================

    

class NoPriorLayer(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.D = int(cfg["D"])
        ar = cfg["arena"]
        c = np.array([ar["cx"], ar["cy"], (ar["cz"] if self.D == 3 else 0.0)], dtype=np.float32)[: self.D]
        self.register_buffer("center", torch.tensor(c, dtype=torch.float32))

    def forward(self, obs: torch.Tensor, who: str):
        """
        Supports both:
          - legacy Na=1: obs_dim = 5*D
          - multiattacker: obs_dim = (2 + 3*Na)*D, where:
              [p1c, pA_c (Na*D), rel (Na*D), v1, vA (Na*D)]
        We use attacker 0 as the "primary" attacker for features.
        """
        B, obs_dim = obs.shape
        D = self.D
        device, dtype = obs.device, obs.dtype

        if obs_dim % D != 0:
            raise RuntimeError(f"obs_dim={obs_dim} not divisible by D={D}")

        blocks = obs_dim // D

        # legacy case: 5 blocks
        if blocks == 5:
            p1c = obs[:, 0:D]
            p2c = obs[:, D:2*D]
            v1  = obs[:, 3*D:4*D]
            v2  = obs[:, 4*D:5*D]
        else:
            # multiattacker layout: blocks = 2 + 3*Na  => Na = (blocks - 2)//3
            if (blocks - 2) % 3 != 0:
                raise RuntimeError(f"Unexpected obs layout: blocks={blocks} not of form 2+3*Na")
            Na = (blocks - 2) // 3

            off = 0
            p1c = obs[:, off:off + D]; off += D

            # pA_c: Na blocks, take attacker0
            p2c = obs[:, off:off + D]  # attacker 0
            off += Na * D

            # rel: skip Na blocks
            off += Na * D

            # v1
            v1 = obs[:, off:off + D]; off += D

            # vA: Na blocks, take attacker0
            v2 = obs[:, off:off + D]

        feats = torch.cat([p1c, p2c, v1, v2], dim=-1)         # (B, 4D)
        u_prior = torch.zeros((B, D), device=device, dtype=dtype)
        return feats, u_prior





class ActorCriticDiff(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg: Dict[str, Any]):
        super().__init__()
        hidden = 128

        # Choose which prior layer to use
        prior_type = cfg.get("prior_type", "none")  # "ls", "nash", or "none"

        print(prior_type)
        raise("Debug (loop multi)")
        if prior_type == "none":
            self.layer = NoPriorLayer(cfg)
        else:
            raise ValueError(
                f"Unknown prior_type={prior_type!r}, expected 'ls', 'nash', or 'none'."
            )

        # Policy (residual over prior)
        self.pi = nn.Sequential(
            nn.Linear(4 * cfg["D"], hidden), nn.Tanh(),
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
        self.prior_blend_def = float(cfg.get("prior_blend_def", 0.5))
        self.prior_blend_att = float(cfg.get("prior_blend_att", 1.0))

    def dist(self, obs: torch.Tensor, who: str):
        feats, u_prior = self.layer(obs, who)
        h = self.pi(feats)
        mu_res = self.mu_res(h)

        blend = self.prior_blend_def if who == "def" else self.prior_blend_att
        mu = mu_res + blend * u_prior

        # std = self.logstd.exp()
        std = self.logstd.clamp(-5.0, 0.5).exp()
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




# =============================================================
# PPO Core (defender learns; attacker optionally rule-based)
# =============================================================
class PPO:
    def __init__(self, obs_dim_def: int, act_dim: int, cfg: Dict[str, Any], device="cpu", obs_dim_att: int | None = None):

        self.cfg = cfg
        self.num_attackers = int(cfg.get("num_attackers", 1))
        self.D = int(cfg["D"])

        if obs_dim_att is None:
            obs_dim_att = obs_dim_def
        self.obs_dim_def = obs_dim_def
        self.obs_dim_att = obs_dim_att

        self.device    = device
        self.clip_eps  = cfg["clip_eps"]
        self.ent_coef  = cfg["entropy_coef"]
        self.vf_coef   = cfg["value_coef"]
        self.max_grad  = cfg["max_grad_norm"]
        self.epochs    = cfg["train_epochs"]
        self.mb_size   = cfg["minibatch_size"]
        self.gamma     = cfg["gamma"]
        self.lam       = cfg["gae_lambda"]
        self.act_scale = float(cfg["umax"])
        self.v_clip_eps = 0.2

        self.attacker_mode = cfg.get("attacker_mode", "rule")

        self.freeze_defender = bool(cfg.get("freeze_defender", False))
        self.freeze_attacker = bool(cfg.get("freeze_attacker", False))  # optional symmetry


        # Defender (always RL)
        self.def_net = ActorCriticDiff(obs_dim_def, act_dim, cfg).to(device)

        self.def_opt = optim.Adam([
            {"params": list(self.def_net.pi.parameters()) + list(self.def_net.mu_res.parameters()), "lr": cfg["policy_lr"]},
            {"params": [self.def_net.logstd], "lr": cfg["policy_lr"] * 0.5},
            {"params": self.def_net.vf.parameters(), "lr": cfg["value_lr"]},
        ])
        # NEW: remember initial defender LRs
        self.def_base_lrs = [g["lr"] for g in self.def_opt.param_groups]

        # Attacker: rule or RL
        if self.attacker_mode == "rl":
            self.att_net = ActorCriticDiff(obs_dim_att, act_dim, cfg).to(device)
            self.att_opt = optim.Adam([
                {
                    "params": list(self.att_net.pi.parameters()) + list(self.att_net.mu_res.parameters()),
                    "lr": cfg["policy_lr"],
                },
                {
                    "params": [self.att_net.logstd],
                    "lr": cfg["policy_lr"] * 0.5,
                },
                {
                    "params": self.att_net.vf.parameters(),
                    "lr": cfg["value_lr"],
                },
            ])
            # NEW: remember initial attacker LRs
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
        Returns:
        a_env:   (B, D) for defender
                (B, D) or (B, Na, D) for attacker depending on cfg["num_attackers"]
        logp:    (B,)
        val:     (B,)
        """
        # ---- NEW: multi-attacker RL ----
        if who == "att" and self.attacker_mode == "rl" and obs_batch.ndim == 3:
            # obs_batch: (B, Na, obs_dim_att)
            B, Na, od = obs_batch.shape
            obs_flat = obs_batch.reshape(B * Na, od)

            if deterministic:
                dist = self.att_net.dist(obs_flat, who="att")
                u_raw = dist.mean
                a_flat = squash_action(u_raw, self.act_scale)
                lp_flat = torch.zeros(B * Na, dtype=obs_batch.dtype, device=obs_batch.device)
                v_flat  = self.att_net.value(obs_flat)
            else:
                a_flat, lp_flat, v_flat = self.att_net.act(obs_flat, who="att", act_scale=self.act_scale)

            a = a_flat.reshape(B, Na, self.D)
            lp = lp_flat.reshape(B, Na)
            v  = v_flat.reshape(B, Na)
            return a, lp, v


        B = obs_batch.shape[0]
        D = int(self.act_scale * 0 + obs_batch.shape[-1] // 5)  # not used except sanity
        D = obs_batch.shape[-1] // 5  # true D from obs layout
        Na = int(getattr(self, "num_attackers", 1) or 1)
        # Better: read Na from cfg once and store it:
        # in PPO.__init__: self.num_attackers = int(cfg.get("num_attackers", 1))
        # then use that here. We'll fallback if missing.
        if hasattr(self, "cfg"):
            Na = int(self.cfg.get("num_attackers", Na))
        elif hasattr(self, "num_attackers"):
            Na = int(self.num_attackers)

        # -------- RULE ATTACKER --------
        if who == "att" and self.attacker_mode == "rule":
            D  = self.D
            Na = self.num_attackers
            ob = obs_batch.detach().cpu().numpy()  # (B, obs_dim)
            B  = ob.shape[0]

            expected_dim = (2 + 3 * Na) * D
            if ob.shape[1] != expected_dim:
                raise RuntimeError(
                    f"[rule_att] obs_dim mismatch: got {ob.shape[1]}, expected {(2 + 3*Na)*D} "
                    f"(D={D}, Na={Na}). If you changed _obs(), update this parser."
                )

            center = self.rule_ctrl.center

            # offsets
            off = 0
            p1c = ob[:, off:off + D]; off += D

            pA_c = []
            for k in range(Na):
                pA_c.append(ob[:, off:off + D]); off += D

            # rel blocks (we actually don't need these for rule controller)
            off += Na * D

            v1 = ob[:, off:off + D]; off += D

            vA = []
            for k in range(Na):
                vA.append(ob[:, off:off + D]); off += D

            acts = np.zeros((B, Na, D), dtype=np.float32)
            for i in range(B):
                p1 = p1c[i] + center
                pA_list = [pA_c[k][i] + center for k in range(Na)]
                vA_list = [vA[k][i] for k in range(Na)]
                acts[i] = self.rule_ctrl.act_multi(p1, v1[i], pA_list, vA_list)  # (Na, D)

            a_t = torch.as_tensor(acts, dtype=obs_batch.dtype, device=obs_batch.device)
            zero = torch.zeros(B, dtype=obs_batch.dtype, device=obs_batch.device)
            return a_t, zero, zero

        # -------- RL POLICY (defender OR attacker) --------
        net = self.def_net if who == "def" else self.att_net

        if deterministic:
            dist = net.dist(obs_batch, who=who)
            u_raw = dist.mean
            a_env = squash_action(u_raw, self.act_scale)
            logp = torch.zeros(B, dtype=obs_batch.dtype, device=obs_batch.device)
            val  = net.value(obs_batch)
            return a_env, logp, val

        return net.act(obs_batch, who, self.act_scale)


    def _update_one(self,
                    net: ActorCriticDiff, opt: optim.Optimizer,
                    obs: torch.Tensor, act_env: torch.Tensor,
                    old_logp: torch.Tensor, old_val: torch.Tensor,
                    adv: torch.Tensor, ret: torch.Tensor, who: str):
        B = obs.shape[0]
        for _ in range(self.epochs):
            idx = torch.randperm(B, device=obs.device)  # <-- shuffle indices on same device
            for st in range(0, B, self.mb_size):
                j = idx[st:st+self.mb_size]
                o = obs[j]; a = act_env[j]; lp_old = old_logp[j]; v_old = old_val[j]; A = adv[j]; R = ret[j]

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
# Distillation helpers (teacher uses full-state; student sees belief)
# =============================================================

def build_full_obs_from_envs(vec: VecEnv, device: str) -> torch.Tensor:
    """
    Build 'full-state' observations for the teacher from each Env in vec.envs:
      o_full = [p1 - c, p2 - c, (p2 - p1), v1, v2]
    using TRUE p2, v2 (not UKF belief).
    Shape: [num_envs, 5*D]
    """
    obs_list = []
    for e in vec.envs:
        # NEW
        p1, v1, pA_list, vA_list = e._unpack(e.state)
        p2 = pA_list[0]
        v2 = vA_list[0]
        center = e.center
        p1c = p1 - center
        p2c = p2 - center
        rel = p2 - p1

        obs_full = np.concatenate([p1c, p2c, rel, v1, v2]).astype(np.float32)
        obs_list.append(obs_full)
    obs_full_np = np.stack(obs_list, axis=0)
    return torch.as_tensor(obs_full_np, dtype=torch.float32, device=device)


#Distillation helpers. 
# Distillation based on https://arxiv.org/pdf/2308.16185

# -----------------------------
# Data container (one episode)
# -----------------------------
@dataclass
class DistillEpisode:
    # Student inputs (what the partially-observable policy is allowed to use)
    xhat_rel: torch.Tensor       # (T, Dx)   estimated relative state from KF
    sigma: torch.Tensor          # (T, Ds)   KF covariance features (flattened or compressed)
    u_prev: torch.Tensor         # (T, Du)   previous pursuer action applied (u_{t-1}), with u_prev[0]=0

    # Teacher labels
    u_star: torch.Tensor         # (T, Du)   teacher action label
    z_star: torch.Tensor         # (T, Dz)   teacher intent label


# ---------------------------------------
# Helper: chunk an episode for TBPTT
# ---------------------------------------
def _iter_tbptt_chunks(ep: DistillEpisode, chunk_len: int):
    T = ep.xhat_rel.shape[0]
    for t0 in range(0, T, chunk_len):
        t1 = min(T, t0 + chunk_len)
        yield (
            ep.xhat_rel[t0:t1],
            ep.sigma[t0:t1],
            ep.u_prev[t0:t1],
            ep.u_star[t0:t1],
            ep.z_star[t0:t1],
        )


# ---------------------------------------
# Main distillation function (paper-style)
# ---------------------------------------

def get_true_rel_state_from_env(env, attacker_idx: int = 0) -> np.ndarray:
    """
    x_rel = [p_att - p_def, v_att - v_def]  (Dx = 2*D)
    Uses TRUE state (not UKF belief).
    """
    p1, v1, pA_list, vA_list = env._unpack(env.state)
    p2 = pA_list[attacker_idx]
    v2 = vA_list[attacker_idx]
    x_rel = np.concatenate([p2 - p1, v2 - v1]).astype(np.float32)
    return x_rel


def build_true_teacher_obs_from_env(env, attacker_idx: int = 0) -> np.ndarray:
    """
    Teacher-style observation for *one* attacker:
      o_full = [p1-center, p2-center, (p2-p1), v1, v2]  (shape = 5*D)
    Uses TRUE p2, v2 (not UKF belief).
    """
    p1, v1, pA_list, vA_list = env._unpack(env.state)
    p2 = pA_list[attacker_idx]
    v2 = vA_list[attacker_idx]
    c = env.center

    p1c = p1 - c
    p2c = p2 - c
    rel = p2 - p1
    return np.concatenate([p1c, p2c, rel, v1, v2]).astype(np.float32)

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import random
import numpy as np
import torch
import torch.nn as nn


# def distill_from_teacher(
#     cfg_student,
#     def_teacher_ckpt,
#     *,
#     out_path=None,
# ) -> Tuple[nn.Module, Dict[str, float]]:

#     # -------------------------
#     # Resolve cfg + device
#     # -------------------------
#     def _extract_cfg_from_teacher_ckpt(ckpt_path: str) -> Dict[str, Any]:
#         ckpt = torch.load(str(ckpt_path), map_location="cpu")
#         cfg = ckpt.get("cfg", None) if isinstance(ckpt, dict) else None
#         if isinstance(cfg, dict):
#             return cfg
#         raise ValueError("Teacher checkpoint missing ['cfg'] dict.")

#     def _extract_cfg() -> Dict[str, Any]:
#         # 1) Preferred: cfg_student.cfg
#         cfg = getattr(cfg_student, "cfg", None)
#         if isinstance(cfg, dict):
#             return cfg

#         # 2) Allow cfg_student to be dict-like
#         if isinstance(cfg_student, dict):
#             return cfg_student.get("cfg", cfg_student)

#         # 3) Fallback: teacher ckpt
#         if def_teacher_ckpt is not None:
#             return _extract_cfg_from_teacher_ckpt(def_teacher_ckpt)

#         raise ValueError("Could not resolve cfg dict for distillation.")

#     def _ensure_dyn(cfg: Dict[str, Any]) -> Dict[str, Any]:
#         dyn = cfg.setdefault("dyn", {})
#         if dyn.get("Ad", None) is None or dyn.get("Bd", None) is None:
#             build_dyn(cfg)  # mutates cfg
#         return cfg

#     cfg = _ensure_dyn(_extract_cfg())

#     device = getattr(cfg_student, "device", None) or cfg.get("device", None)
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Construct env AFTER dyn is built
#     env = Env(cfg)

#     # -------------------------
#     # Build student + required hooks
#     # -------------------------
#     cfg_student = config_for_train(
#             attacker_mode=attacker_mode,
#             train_role="def",
#         )
#     optimizer = cfg_student.make_optimizer(student)
#     detector_fn = cfg_student.make_detector()
#     kf_step_fn = cfg_student.make_kf(env)
#     teacher_label_fn = cfg_student.make_teacher_label_fn(def_teacher_ckpt)

#     get_x_rel_t_fn = getattr(cfg_student, "get_x_rel_t", None)
#     get_x_rel_future_fn = getattr(cfg_student, "get_x_rel_future", None)

#     attacker_action_fn = getattr(cfg_student, "attacker_action_fn", None)

#     student.to(device)
#     student.train()


#     metrics_last: Dict[str, float] = {}

#     # ----- losses -----
#     if action_loss == "mse":
#         act_crit = nn.MSELoss()
#     elif action_loss == "huber":
#         act_crit = nn.SmoothL1Loss()
#     else:
#         raise ValueError(f"Unsupported action_loss: {action_loss}")

#     if intent_loss == "mse":
#         z_crit = nn.MSELoss()
#     elif intent_loss == "huber":
#         z_crit = nn.SmoothL1Loss()
#     else:
#         raise ValueError(f"Unsupported intent_loss: {intent_loss}")

#     def beta_at(iter_idx: int) -> float:
#         if dagger_decay_iters <= 0:
#             return float(dagger_beta_end)
#         a = min(1.0, max(0.0, iter_idx / float(dagger_decay_iters)))
#         return float((1.0 - a) * dagger_beta_start + a * dagger_beta_end)

#     # ----- attacker action default (rule if available, else zeros) -----
#     rule_ctrl = None
#     if attacker_action_fn is None:
#         cfg = getattr(env, "cfg", {})
#         mode = cfg.get("attacker_mode", "rule") if isinstance(cfg, dict) else "rule"

#         if mode == "rule" and isinstance(cfg, dict) and ("dyn" in cfg):
#             # If your project defines this elsewhere, keep it.
#             rule_ctrl = AttackerRuleController(cfg)

#         def attacker_action_fn(env_local):
#             D = int(env_local.D)
#             Na = int(getattr(env_local, "num_attackers", 1))

#             if rule_ctrl is not None:
#                 p1, v1, pA_list, vA_list = env_local._unpack(env_local.state)
#                 uA = rule_ctrl.act_multi(p1, v1, pA_list, vA_list)  # (Na, D)
#                 return uA[0] if Na == 1 else uA

#             return np.zeros((D,), dtype=np.float32) if Na == 1 else np.zeros((Na, D), dtype=np.float32)

#     # -------------------------
#     # 1) (Optional but loyal) DAgger dataset aggregation
#     # -------------------------
#     dataset: List[DistillEpisode] = []

#     # ============================
#     # Main DAgger distill loop
#     # ============================
#     for it in range(iters):
#         beta = beta_at(it)

#         # -----------------------------
#         # 1a) Collect DAgger rollouts (new episodes this iter)
#         # -----------------------------
#         new_episodes: List[DistillEpisode] = []

#         for _ in range(episodes_per_iter):
#             obs = env.reset()

#             x_rel_hist: List[np.ndarray] = []
#             u_hist: List[np.ndarray] = []

#             xhat_list: List[np.ndarray] = []
#             sig_list: List[np.ndarray] = []
#             u_prev_list: List[np.ndarray] = []

#             u_star_list: List[np.ndarray] = []
#             z_star_list: List[np.ndarray] = []

#             u_prev = None

#             hidden = None
#             if hasattr(student, "init_hidden"):
#                 hidden = student.init_hidden(batch_size=1, device=device)

#             for t in range(max_steps):
#                 # ----- student pipeline: detector -> KF -> (xhat_rel, Sigma) -----
#                 y_t = detector_fn(obs)
#                 xhat_rel_t_np, Sigma_t_np = kf_step_fn(env, y_t)

#                 # ----- privileged teacher inputs -----
#                 x_rel_t_np = get_x_rel_t_fn(env)
#                 x_rel_hist.append(x_rel_t_np)

#                 x_rel_future_np = get_x_rel_future_fn(env, t, lookahead_H)

#                 teacher_in = {
#                     "t": t,
#                     "x_rel_t": x_rel_t_np,
#                     "x_rel_hist": x_rel_hist,
#                     "u_hist": u_hist,
#                     "x_rel_future": x_rel_future_np,
#                     "env_info": {},
#                 }
#                 with torch.no_grad():
#                     u_star_t_np, z_star_t_np = teacher_label_fn(teacher_in)

#                 # ----- student forward -----
#                 xhat_rel_t = torch.as_tensor(xhat_rel_t_np, dtype=torch.float32, device=device).view(-1)
#                 sigma_t    = torch.as_tensor(Sigma_t_np,    dtype=torch.float32, device=device).view(-1)

#                 u_prev_np = np.zeros_like(u_star_t_np, dtype=np.float32) if u_prev is None else u_prev
#                 u_prev_t  = torch.as_tensor(u_prev_np, dtype=torch.float32, device=device).view(-1)

#                 u_pred_t, z_hat_t, hidden = student.step(xhat_rel_t, sigma_t, u_prev_t, hidden)

#                 # ----- DAgger mixture -----
#                 use_teacher = (random.random() < beta)
#                 u_apply_np = u_star_t_np if use_teacher else u_pred_t.detach().cpu().numpy()

#                 # ----- attacker action & step env -----
#                 aA_apply_np = attacker_action_fn(env)

#                 obs, r1, r2, done, info = env.step(
#                     a1_env=u_apply_np,
#                     aA_env=aA_apply_np,
#                     reward_mode=reward_mode_for_step,
#                 )
#                 env.info = info

#                 # ----- log supervised data (teacher-labeled) -----
#                 xhat_list.append(np.asarray(xhat_rel_t_np, dtype=np.float32).reshape(-1))
#                 sig_list.append(np.asarray(Sigma_t_np, dtype=np.float32).reshape(-1))
#                 u_prev_list.append(np.asarray(u_prev_np, dtype=np.float32).reshape(-1))

#                 u_star_list.append(np.asarray(u_star_t_np, dtype=np.float32).reshape(-1))
#                 z_star_list.append(np.asarray(z_star_t_np, dtype=np.float32).reshape(-1))

#                 # histories should match executed action distribution
#                 u_hist.append(np.asarray(u_apply_np, dtype=np.float32).reshape(-1))
#                 u_prev = np.asarray(u_apply_np, dtype=np.float32).reshape(-1)

#                 if done:
#                     break

#             ep = DistillEpisode(
#                 xhat_rel=torch.as_tensor(np.stack(xhat_list), device=device),
#                 sigma=torch.as_tensor(np.stack(sig_list), device=device),
#                 u_prev=torch.as_tensor(np.stack(u_prev_list), device=device),
#                 u_star=torch.as_tensor(np.stack(u_star_list), device=device),
#                 z_star=torch.as_tensor(np.stack(z_star_list), device=device),
#             )
#             new_episodes.append(ep)

#         # Aggregate (loyal DAgger)
#         dataset.extend(new_episodes)
#         if max_dataset_episodes is not None and len(dataset) > int(max_dataset_episodes):
#             dataset = dataset[-int(max_dataset_episodes):]

#         # -----------------------------
#         # 1b) Supervised update (TBPTT) over aggregated dataset
#         # -----------------------------
#         total_loss = 0.0
#         total_act  = 0.0
#         total_z    = 0.0
#         n_chunks   = 0

#         optimizer.zero_grad(set_to_none=True)

#         for ep in dataset:
#             hidden = None
#             if hasattr(student, "init_hidden"):
#                 hidden = student.init_hidden(batch_size=1, device=device)

#             for (xhat_seq, sig_seq, uprev_seq, ustar_seq, zstar_seq) in _iter_tbptt_chunks(ep, tbptt_chunk_len):
#                 # detach hidden between chunks
#                 if hidden is not None:
#                     if isinstance(hidden, (tuple, list)):
#                         hidden = tuple(h.detach() for h in hidden)
#                     else:
#                         hidden = hidden.detach()

#                 u_preds = []
#                 z_hats  = []

#                 Tchunk = xhat_seq.shape[0]
#                 for k in range(Tchunk):
#                     u_pred, z_hat, hidden = student.step(
#                         xhat_seq[k].view(-1),
#                         sig_seq[k].view(-1),
#                         uprev_seq[k].view(-1),
#                         hidden,
#                     )
#                     u_preds.append(u_pred.view(1, -1))
#                     z_hats.append(z_hat.view(1, -1))

#                 u_preds = torch.cat(u_preds, dim=0)
#                 z_hats  = torch.cat(z_hats,  dim=0)

#                 loss_act = act_crit(u_preds, ustar_seq)
#                 loss_z   = z_crit(z_hats,  zstar_seq)
#                 loss     = loss_act + lambda_intent * loss_z

#                 loss.backward()

#                 total_loss += float(loss.detach().cpu())
#                 total_act  += float(loss_act.detach().cpu())
#                 total_z    += float(loss_z.detach().cpu())
#                 n_chunks   += 1

#         if grad_clip_norm is not None and grad_clip_norm > 0:
#             torch.nn.utils.clip_grad_norm_(student.parameters(), float(grad_clip_norm))

#         optimizer.step()

#         if n_chunks > 0:
#             metrics_last = {
#                 "iter": float(it),
#                 "dagger_beta": float(beta),
#                 "loss": total_loss / n_chunks,
#                 "loss_action": total_act / n_chunks,
#                 "loss_intent": total_z / n_chunks,
#                 "dataset_episodes": float(len(dataset)),
#             }
#             if log_fn is not None:
#                 log_fn(metrics_last)

#     # ============================
#     # Save distilled student
#     # ============================
#     if out_path is not None:
#         out_path = str(out_path)
#         Path(out_path).parent.mkdir(parents=True, exist_ok=True)

#         payload = {
#             "state_dict": student.state_dict(),
#             "cfg": getattr(env, "cfg", None),
#             "metrics": metrics_last,
#             "meta": {
#                 "iters": int(iters),
#                 "episodes_per_iter": int(episodes_per_iter),
#                 "max_steps": int(max_steps),
#                 "lookahead_H": int(lookahead_H),
#                 "lambda_intent": float(lambda_intent),
#                 "dagger_beta_start": float(dagger_beta_start),
#                 "dagger_beta_end": float(dagger_beta_end),
#                 "dagger_decay_iters": int(dagger_decay_iters),
#                 "tbptt_chunk_len": int(tbptt_chunk_len),
#                 "grad_clip_norm": float(grad_clip_norm) if grad_clip_norm is not None else None,
#                 "action_loss": str(action_loss),
#                 "intent_loss": str(intent_loss),
#                 "reward_mode_for_step": str(reward_mode_for_step),
#                 "max_dataset_episodes": int(max_dataset_episodes) if max_dataset_episodes is not None else None,
#             },
#         }
#         torch.save(payload, out_path)
#         print(f"[distill] saved student checkpoint -> {out_path}", flush=True)

#     return student, metrics_last

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def distill_from_teacher(
    cfg: Dict[str, Any],
    teacher_ckpt_path: str,
    out_path: str = "ppo_def_ukf_distilled.pt",
):
    """
    DAgger-style distillation: aggregate teacher-labeled data across updates.

    - Student observes UKF/belief obs (vec.obs).
    - Teacher acts on full-state obs we construct from TRUE env state.
    - During rollout, we execute teacher w.p. beta_exec, else execute student.
    - Regardless of what we execute, we ALWAYS query teacher to label the visited state.
    - We AGGREGATE all (student_obs, teacher_obs, teacher_u_raw) into a replay buffer
      and train on that buffer (classic DAgger dataset aggregation).

    Returns
    -------
    student : ActorCriticDiff
    metrics : Dict[str, List[float]]
    """

    set_seed(cfg["seed"])
    device = cfg["device"]

    # -----------------------
    # Hyperparams
    # -----------------------
    num_envs      = int(cfg.get("num_envs", 8))
    steps_per_env = int(cfg.get("steps_per_env", 256))
    total_updates = int(cfg.get("total_updates", 300))
    log_every     = int(cfg.get("log_every", 10))

    # DAgger schedule (prob of executing TEACHER)
    beta0       = float(cfg.get("distill_beta0", 1.0))
    beta_final  = float(cfg.get("distill_beta_final", 0.05))
    beta_decay  = cfg.get("distill_beta_decay", "linear")  # "linear" or "exp"
    beta_exp_k  = float(cfg.get("distill_beta_exp_k", 5.0))

    # Supervised opt
    bc_lr     = float(cfg.get("distill_lr", cfg.get("policy_lr", 3e-4)))
    mb_size   = int(cfg.get("distill_mb_size", 2048))
    bc_epochs = int(cfg.get("distill_epochs", 4))

    # Loss weights
    w_nll = float(cfg.get("distill_w_nll", 1.0))   # -log pi_s(u_teacher_raw) with squash correction
    w_kl  = float(cfg.get("distill_w_kl", 0.0))    # KL(teacher||student) in raw Normal space
    w_mse = float(cfg.get("distill_w_mse", 0.0))   # MSE(student_raw_mean, teacher_raw)
    w_v   = float(cfg.get("distill_w_v", 0.0))     # MSE(V_student(ukf_obs), V_teacher(full_obs))

    teacher_label_mode = cfg.get("distill_teacher_label", "mean")  # "mean" or "sample"
    assert teacher_label_mode in ("mean", "sample")

    act_scale = float(cfg["umax"])

    # Replay / aggregation capacity (DAgger dataset aggregation)
    B_per_update = steps_per_env * num_envs
    buf_capacity = int(cfg.get("distill_buffer_capacity", 50 * B_per_update))  # ~50 rollouts

    # -----------------------
    # Build env
    # -----------------------
    def make_env():
        return Env(cfg)

    vec = VecEnv(make_env, num_envs)
    obs_dim = vec.obs.shape[1]
    act_dim = int(cfg["D"])

    # -----------------------
    # Load teacher
    # -----------------------
    teacher = ActorCriticDiff(obs_dim, act_dim, cfg).to(device)

    def _load_state_dict_robust(model: torch.nn.Module, path: str):
        payload = torch.load(path, map_location=device)

        if isinstance(payload, dict) and all(isinstance(k, str) for k in payload.keys()):
            # try direct
            try:
                model.load_state_dict(payload, strict=True)
                return
            except Exception:
                pass

            # try nested
            for key in ["state_dict", "model", "policy", "net", "def_net", "actor_critic"]:
                if key in payload and isinstance(payload[key], dict):
                    try:
                        model.load_state_dict(payload[key], strict=True)
                        return
                    except Exception:
                        pass

            # PPO wrapper prefix "def_net."
            if any(k.startswith("def_net.") for k in payload.keys()):
                stripped = {k[len("def_net."):]: v for k, v in payload.items() if k.startswith("def_net.")}
                model.load_state_dict(stripped, strict=True)
                return

        raise RuntimeError(f"Could not load teacher checkpoint from {path}")

    _load_state_dict_robust(teacher, teacher_ckpt_path)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # -----------------------
    # Student
    # -----------------------
    student = ActorCriticDiff(obs_dim, act_dim, cfg).to(device)
    student_opt = optim.Adam(
        list(student.pi.parameters()) + list(student.mu_res.parameters()) + [student.logstd],
        lr=bc_lr,
    )

    # Attacker rule-based
    rule_ctrl = AttackerRuleController(cfg)

    # -----------------------
    # Metrics
    # -----------------------
    metrics = {
        "update": [],
        "dbg_mse_mean": [],
        "dbg_nll": [],
        "dbg_kl": [],
        "dbg_v_mse": [],
        "beta_exec": [],
        "replay_size": [],
    }

    print("=== Distillation (DAgger dataset aggregation) : teacher -> UKF student ===")
    print(f"Teacher checkpoint: {teacher_ckpt_path}")
    print(f"Saving distilled student to: {out_path}")
    print(f"w_nll={w_nll}, w_kl={w_kl}, w_mse={w_mse}, w_v={w_v}")
    print(f"beta0={beta0}, beta_final={beta_final}, beta_decay={beta_decay}")
    print(f"buffer_capacity={buf_capacity}")

    # -----------------------
    # Beta schedule
    # -----------------------
    def _beta_for_update(upd: int) -> float:
        if total_updates <= 1:
            return beta_final
        frac = (upd - 1) / (total_updates - 1)
        if beta_decay == "linear":
            return float(beta0 + (beta_final - beta0) * frac)
        elif beta_decay == "exp":
            return float(beta_final + (beta0 - beta_final) * np.exp(-beta_exp_k * frac))
        else:
            raise ValueError(f"Unknown distill_beta_decay={beta_decay!r}")

    # -----------------------
    # Ring buffer for DAgger aggregation
    # Store on CPU to avoid GPU memory blow-up.
    # -----------------------
    class _RingBuffer:
        def __init__(self, cap: int, obs_dim_: int, act_dim_: int):
            self.cap = cap
            self.obs_s = torch.empty((cap, obs_dim_), dtype=torch.float32, device="cpu")
            self.obs_t = torch.empty((cap, obs_dim_), dtype=torch.float32, device="cpu")
            self.u_t   = torch.empty((cap, act_dim_), dtype=torch.float32, device="cpu")  # teacher RAW action
            self.v_t   = torch.empty((cap, 1), dtype=torch.float32, device="cpu") if (w_v != 0.0) else None

            self.ptr = 0
            self.size = 0

        def add(self, obs_s_b, obs_t_b, u_t_b, v_t_b=None):
            n = obs_s_b.shape[0]
            obs_s_b = obs_s_b.detach().to("cpu")
            obs_t_b = obs_t_b.detach().to("cpu")
            u_t_b   = u_t_b.detach().to("cpu")
            if v_t_b is not None:
                v_t_b = v_t_b.detach().to("cpu")
                if v_t_b.ndim == 1:
                    v_t_b = v_t_b[:, None]

            for i in range(n):
                j = self.ptr
                self.obs_s[j].copy_(obs_s_b[i])
                self.obs_t[j].copy_(obs_t_b[i])
                self.u_t[j].copy_(u_t_b[i])
                if self.v_t is not None and v_t_b is not None:
                    self.v_t[j].copy_(v_t_b[i])

                self.ptr = (self.ptr + 1) % self.cap
                self.size = min(self.size + 1, self.cap)

        def sample(self, batch_size: int, device_: str):
            assert self.size > 0
            idx = torch.randint(0, self.size, (batch_size,), device="cpu")
            o_s = self.obs_s[idx].to(device_)
            o_t = self.obs_t[idx].to(device_)
            u_t = self.u_t[idx].to(device_)
            v_t = self.v_t[idx].to(device_) if self.v_t is not None else None
            return o_s, o_t, u_t, v_t

    replay = _RingBuffer(buf_capacity, obs_dim, act_dim)

    # -----------------------
    # Rollout + train loop
    # -----------------------
    for upd in range(1, total_updates + 1):
        beta_exec = _beta_for_update(upd)

        # Collect one rollout batch (on GPU), then add to replay (CPU)
        obs_student_buf = torch.zeros(steps_per_env, num_envs, obs_dim, device=device)
        obs_teacher_buf = torch.zeros(steps_per_env, num_envs, obs_dim, device=device)
        u_teacher_buf   = torch.zeros(steps_per_env, num_envs, act_dim, device=device)  # RAW action
        if w_v != 0.0:
            v_teacher_buf = torch.zeros(steps_per_env, num_envs, 1, device=device)

        for t in range(steps_per_env):
            o_student = torch.as_tensor(vec.obs, dtype=torch.float32, device=device)
            o_teacher = build_full_obs_from_envs(vec, device)

            with torch.no_grad():
                # Teacher label on visited state
                dist_t = teacher.dist(o_teacher, who="def")
                if teacher_label_mode == "mean":
                    u_teacher = dist_t.mean
                else:
                    u_teacher = dist_t.rsample()

                a_teacher_env = squash_action(u_teacher, act_scale)

                # Student action for possible execution (use mean for stable rollouts)
                dist_s_exec = student.dist(o_student, who="def")
                u_student_exec = dist_s_exec.mean
                a_student_env  = squash_action(u_student_exec, act_scale)

                # Optional teacher value target (privileged)
                if w_v != 0.0:
                    v_t = teacher.value(o_teacher).reshape(-1, 1)

            # Execute teacher/student mixture in environment (DAgger)
            if np.random.rand() < beta_exec:
                a1_exec = a_teacher_env
            else:
                a1_exec = a_student_env

            # Rule attacker from TRUE state
            acts_att = []
            for e in vec.envs:
                p1, v1, pA_list, vA_list = e._unpack(e.state)
                p2 = pA_list[0]
                v2 = vA_list[0]
                a2 = rule_ctrl.act(p1, v1, p2, v2).astype(np.float32)
                acts_att.append(a2)
            a2_env = np.stack(acts_att, axis=0)

            # Step env
            o2_np, _, done, trunc, _ = vec.step(
                a1_exec.detach().cpu().numpy(),
                a2_env,
            )
            vec.obs = o2_np

            # Store teacher-labeled data (label regardless of executed action)
            obs_student_buf[t] = o_student
            obs_teacher_buf[t] = o_teacher
            u_teacher_buf[t]   = u_teacher
            if w_v != 0.0:
                v_teacher_buf[t] = v_t

            # Reset handling (best-effort; depends on your VecEnv implementation)
            if isinstance(done, (list, np.ndarray)) and np.any(done):
                if hasattr(vec, "reset_done"):
                    vec.reset_done(done, trunc)
                elif hasattr(vec, "reset"):
                    vec.reset()

        # Add rollout to replay (DAgger aggregation)
        B = steps_per_env * num_envs
        obs_s_new = obs_student_buf.reshape(B, obs_dim)
        obs_t_new = obs_teacher_buf.reshape(B, obs_dim)
        u_t_new   = u_teacher_buf.reshape(B, act_dim)
        v_t_new   = v_teacher_buf.reshape(B, 1) if (w_v != 0.0) else None
        replay.add(obs_s_new, obs_t_new, u_t_new, v_t_b=v_t_new)

        # Train on aggregated dataset
        # Roughly do ~one rollout-worth of samples per epoch
        num_mb = max(1, B_per_update // mb_size)
        for _ in range(bc_epochs):
            for _mb in range(num_mb):
                o_s_mb, o_t_mb, u_t_mb, v_t_mb = replay.sample(mb_size, device)

                dist_s = student.dist(o_s_mb, who="def")
                with torch.no_grad():
                    dist_t_mb = teacher.dist(o_t_mb, who="def")

                # (1) NLL of teacher RAW action under student (with squash correction)
                nll = -logprob_squashed(dist_s, u_t_mb).mean()

                # (2) Optional raw-mean MSE (consistent space)
                mse = ((dist_s.mean - u_t_mb) ** 2).mean()

                # (3) Optional KL in raw Gaussian space
                kl = torch.distributions.kl_divergence(dist_t_mb, dist_s).sum(-1).mean()

                # (4) Optional value distillation
                if w_v != 0.0:
                    v_s = student.value(o_s_mb).reshape(-1, 1)
                    v_mse = ((v_s - v_t_mb) ** 2).mean()
                else:
                    v_mse = torch.zeros((), device=device)

                loss = (w_nll * nll) + (w_mse * mse) + (w_kl * kl) + (w_v * v_mse)

                student_opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), cfg.get("max_grad_norm", 0.5))
                student_opt.step()

        # Logging on a debug sample from replay
        if upd % log_every == 0:
            with torch.no_grad():
                dbg_B = min(replay.size, 4096)
                o_s_dbg, o_t_dbg, u_t_dbg, v_t_dbg = replay.sample(dbg_B, device)

                dist_s_dbg = student.dist(o_s_dbg, who="def")
                dist_t_dbg = teacher.dist(o_t_dbg, who="def")

                dbg_mse = ((dist_s_dbg.mean - u_t_dbg) ** 2).mean().item()
                dbg_nll = (-logprob_squashed(dist_s_dbg, u_t_dbg)).mean().item()
                dbg_kl  = torch.distributions.kl_divergence(dist_t_dbg, dist_s_dbg).sum(-1).mean().item()

                if w_v != 0.0:
                    v_s = student.value(o_s_dbg).reshape(-1, 1)
                    dbg_v_mse = ((v_s - v_t_dbg) ** 2).mean().item()
                else:
                    dbg_v_mse = 0.0

            print(
                f"[distill {upd:05d}] beta_exec={beta_exec:.3f}  "
                f"MSE(rawmean)={dbg_mse:.3e}  NLL={dbg_nll:.3e}  KL={dbg_kl:.3e}  V_MSE={dbg_v_mse:.3e}  "
                f"replay={replay.size}"
            )

            metrics["update"].append(upd)
            metrics["dbg_mse_mean"].append(dbg_mse)
            metrics["dbg_nll"].append(dbg_nll)
            metrics["dbg_kl"].append(dbg_kl)
            metrics["dbg_v_mse"].append(dbg_v_mse)
            metrics["beta_exec"].append(beta_exec)
            metrics["replay_size"].append(replay.size)

    torch.save(student.state_dict(), out_path)
    print(f"Distillation finished. Saved student defender to '{out_path}'.")

    return student, metrics


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

    Now supports num_attackers >= 1 by treating each attacker as a supervised sample
    for a single shared policy.
    """
    assert cfg.get("attacker_mode", "rule") == "rl", "BC pretrain requires attacker_mode='rl'"

    set_seed(cfg["seed"])
    device = cfg["device"]

    Na = int(cfg.get("num_attackers", 1))
    D  = int(cfg["D"])

    # build_dyn(cfg)             # <-- ADD THIS (or ensure_dyn(cfg))

    def make_env():
        return Env(cfg)

    num_envs = int(cfg.get("num_envs", 8))
    vec = VecEnv(make_env, num_envs)

    obs_dim_def = vec.obs.shape[1]
    obs_dim_att = 5 * D  # per-attacker obs
    act_dim = D

    # PPO container (we'll only use its networks)
    ppo = PPO(obs_dim_def, act_dim, cfg, device=device, obs_dim_att=obs_dim_att)

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
    print(f"updates={total_updates}, steps_per_env={steps_per_env}, num_envs={num_envs}, Na={Na}")

    for upd in range(1, total_updates + 1):
        # collect a batch of per-attacker data
        # obs_buf: (T, B, Na, obs_dim_att)
        # act_buf: (T, B, Na, D)
        obs_buf = torch.zeros(steps_per_env, num_envs, Na, obs_dim_att, device=device)
        act_buf = torch.zeros(steps_per_env, num_envs, Na, act_dim, device=device)

        for t in range(steps_per_env):
            o_def = torch.as_tensor(vec.obs, dtype=torch.float32, device=device)

            # Defender action: deterministic if frozen, else sample
            with torch.no_grad():
                det_def = (cfg.get("def_ckpt_path", None) is not None)
                a1, _, _ = ppo.act(o_def, who="def", deterministic=det_def)

            # Build per-attacker obs from envs: (B, Na, 5D)
            o_att_np = vec.get_att_obs()
            o_att = torch.as_tensor(o_att_np, dtype=torch.float32, device=device)

            # Rule attacker labels from TRUE state: (B, Na, D)
            aA_list = []
            for e in vec.envs:
                p1, v1, pA_list, vA_list = e._unpack(e.state)
                aA = rule_ctrl.act_multi(p1, v1, pA_list, vA_list).astype(np.float32)  # (Na, D)
                aA_list.append(aA)
            aA_np = np.stack(aA_list, axis=0).astype(np.float32)  # (B, Na, D)

            # Step env using defender action + rule attacker actions
            o2_np, _, _, _, _ = vec.step(a1.cpu().numpy(), aA_np, reward_mode="both")
            vec.obs = o2_np

            obs_buf[t] = o_att
            act_buf[t] = torch.as_tensor(aA_np, dtype=torch.float32, device=device)

        # Flatten per-attacker samples: (T*B*Na, ...)
        Btotal = steps_per_env * num_envs * Na
        obs_flat = obs_buf.reshape(Btotal, obs_dim_att)
        act_flat = act_buf.reshape(Btotal, act_dim)

        # BC epochs
        for _ in range(bc_epochs):
            perm = torch.randperm(Btotal, device=device)
            for st in range(0, Btotal, bc_mb_size):
                idx = perm[st:st + bc_mb_size]
                o_mb = obs_flat[idx]
                a_mb = act_flat[idx]

                dist = ppo.att_net.dist(o_mb, who="att")
                mu = dist.mean
                bc_loss = ((mu - a_mb) ** 2).mean()

                bc_opt.zero_grad(set_to_none=True)
                bc_loss.backward()
                nn.utils.clip_grad_norm_(ppo.att_net.parameters(), cfg.get("max_grad_norm", 0.5))
                bc_opt.step()

        if upd % int(cfg.get("log_every", 10)) == 0:
            with torch.no_grad():
                dbgN = min(Btotal, 4096)
                dist_dbg = ppo.att_net.dist(obs_flat[:dbgN], who="att")
                mse_dbg = ((dist_dbg.mean - act_flat[:dbgN]) ** 2).mean().item()
            print(f"[att_bc {upd:05d}] BC MSE (dbg) = {mse_dbg:.3e}")

    torch.save(ppo.att_net.state_dict(), out_path)
    print(f"BC pretrain finished. Saved: {out_path}")
    return out_path




# =============================================================
# Training & Evaluation
# =============================================================
def train(cfg: Dict[str, Any]):



    set_seed(cfg["seed"])
    device = cfg["device"]

    writer = None
    tb_logdir = None
    global_env_step = 0

    if cfg.get("use_tensorboard", False):
        writer, tb_logdir = make_tb_writer(cfg)

        # Optional: show model graphs once (often noisy / can break with custom ops)
        # writer.add_text("notes", "PPO Diffgame run", 0)



    train_role = cfg.get("train_role", "def")  # <-- NEW

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

    vec = VecEnv(make_env, num_envs)

    obs_dim_def = vec.obs.shape[1]
    act_dim = int(cfg["D"])

    Na = int(cfg.get("num_attackers", 1))
    obs_dim_att = 5 * act_dim  # per-attacker obs

    ppo = PPO(obs_dim_def, act_dim, cfg, device=device, obs_dim_att=obs_dim_att)

    # Optional: initialize attacker from a BC checkpoint (before PPO training)
    att_init = cfg.get("att_init_path", None)
    if att_init is not None:
        if ppo.att_net is None:
            raise RuntimeError("att_init_path provided but attacker_mode != 'rl'")
        state = torch.load(att_init, map_location=device)
        ppo.att_net.load_state_dict(state)
        print(f"[train] Loaded attacker init from: {att_init}")


    # Optional: load fixed defender
    def_ckpt = cfg.get("def_ckpt_path", None)
    if def_ckpt is not None:
        state = torch.load(def_ckpt, map_location=device)
        ppo.def_net.load_state_dict(state)
        # If defender should be frozen:
        if cfg.get("freeze_defender", False):
            for p in ppo.def_net.parameters():
                p.requires_grad_(False)

    # Optional: load fixed attacker
    att_ckpt = cfg.get("att_ckpt_path", None)
    if att_ckpt is not None and ppo.att_net is not None:
        state = torch.load(att_ckpt, map_location=device)
        ppo.att_net.load_state_dict(state)
        if cfg.get("freeze_attacker", False):
            for p in ppo.att_net.parameters():
                p.requires_grad_(False)

    # Freeze whichever role we do NOT train in this phase
    if cfg.get("freeze_defender", False):
        freeze_module_(ppo.def_net)

    if cfg.get("freeze_attacker", False) and (ppo.att_net is not None):
        freeze_module_(ppo.att_net)

    # =========================================================
    # NEW: Freeze verification snapshots (opponents must not move)
    # =========================================================
    verify_freeze = bool(cfg.get("verify_freeze", True))
    freeze_tol = float(cfg.get("freeze_tol", 0.0))  # set 0.0 for exact; or 1e-12 if you’re paranoid

    snap_def = None
    snap_att = None

    if verify_freeze and cfg.get("freeze_defender", False):
        # defender is frozen opponent in attacker-training
        snap_def = snapshot_state_dict(ppo.def_net)

    if verify_freeze and cfg.get("freeze_attacker", False) and (ppo.att_net is not None):
        # attacker is frozen opponent in defender-training
        snap_att = snapshot_state_dict(ppo.att_net)




    # NEW: LR schedule config
    lr_schedule = cfg.get("lr_schedule", "none")
    lr_final_factor = float(cfg.get("lr_final_factor", 0.1))

    # ---- NEW: metrics container ----
    metrics = {
        "update": [],
        "R_def_mean": [],
        "R_att_mean": [],
        "muD_abs_mean": [],
        "stdD_mean": [],
        "d1_mean": [],          # defender true distance
        "d2_mean": [],          # attacker belief distance (what obs sees)
        "d2_true_mean": [],     # attacker true distance
        "meas_innov_mean": [],  # optional
        "ukf_trPpos_mean": [],  # optional
        "lr_pi": [],
        "lr_vf": [],
    }



    # Optional anneal of defender center tether
    def_center_base = cfg.get("def_center_coef", 0.0)
    min_anneal = float(cfg.get("def_center_min_anneal", 0.5))

    for upd in range(1, total_updates + 1):
        term_counts = {"oob_def":0, "oob_att":0, "hit_target":0, "collision":0}


        # ---------- optional LR decay (linear) ----------
        if lr_schedule == "linear":
            # frac_lr goes 0 → 1 over training
            frac_lr = (upd - 1) / max(1, total_updates - 1)
            scale = 1.0 - frac_lr * (1.0 - lr_final_factor)

            # defender
            for g, base in zip(ppo.def_opt.param_groups, ppo.def_base_lrs):
                g["lr"] = base * scale

            # attacker RL (if you ever turn it on)
            if ppo.att_opt is not None and getattr(ppo, "att_base_lrs", None) is not None:
                for g, base in zip(ppo.att_opt.param_groups, ppo.att_base_lrs):
                    g["lr"] = base * scale
        # ------------------------------------------------

        # Linear anneal multiplier from 1.0 → min_anneal for k_cent
        center_frac = upd / max(1, total_updates)
        k_cent_mul = 1.0 - (1.0 - min_anneal) * center_frac
        for e in vec.envs:
            e.k_cent = def_center_base * k_cent_mul

        # Buffers
        bufD = RolloutBuffer(obs_dim_def, act_dim, num_envs, steps_per_env, device)
        rule_att = (cfg.get("attacker_mode", "rule") == "rule")
        if (not rule_att) and (train_role == "att"):
            bufA = RolloutBuffer(obs_dim_att, act_dim, num_envs * Na, steps_per_env, device)
        else:
            bufA = None


        o = torch.as_tensor(vec.obs, dtype=torch.float32, device=device)
        ep_ret_def = np.zeros(num_envs, dtype=np.float64)
        ep_ret_att = np.zeros(num_envs, dtype=np.float64)

        # accumulators for metrics over this update
        d1_true_acc = 0.0
        d2_true_acc = 0.0
        d2_belief_acc = 0.0
        meas_innov_acc = 0.0
        trP_acc = 0.0
        info_count = 0

        for _ in range(steps_per_env):
            with torch.no_grad():
                det_def = (train_role != "def")   # defender is opponent unless training defender
                det_att = (train_role != "att")   # attacker is opponent unless training attacker

                a1, lp1, v1 = ppo.act(o, who="def", deterministic=det_def)
                if cfg.get("attacker_mode", "rule") == "rl":
                    # attacker obs: (B, Na, 5D)
                    o_att_np = vec.get_att_obs()
                    o_att = torch.as_tensor(o_att_np, dtype=torch.float32, device=device)
                    a2, lp2, v2 = ppo.act(o_att, who="att", deterministic=det_att)  # (B,Na,D), (B,Na), (B,Na)
                else:
                    a2, lp2, v2 = ppo.act(o, who="att", deterministic=det_att)     # rule path already returns (B,Na,D)


            a1_np = a1.cpu().numpy()

            a2_np = a2.cpu().numpy()
            # normalize shapes to what Env expects:
            # - if attacker RL: (B, D) ok
            # - if rule attacker: (B, Na, D) ok
            # - if rule attacker but produced (B, D): expand to (B,1,D)
            Na = int(cfg.get("num_attackers", 1))
            D  = int(cfg["D"])

            # ✅ Always pass attacker actions as (B, Na, D)
            if a2_np.ndim == 2:
                # (B, D) -> (B, 1, D)
                a2_np = a2_np[:, None, :]
            elif a2_np.ndim == 3:
                pass
            else:
                raise RuntimeError(f"Unexpected attacker action shape: {a2_np.shape}")

            if a2_np.shape[1] != Na or a2_np.shape[2] != D:
                raise RuntimeError(f"Attacker action shape mismatch: got {a2_np.shape}, expected (B,{Na},{D})")

            o2_np, r1_np, r2_np, d_np, infos = vec.step(
                a1_np,
                a2_np,
                reward_mode=reward_mode,
            )


            o2 = torch.as_tensor(o2_np, dtype=torch.float32, device=device)
            r1 = torch.as_tensor(r1_np, dtype=torch.float32, device=device)
            r2 = torch.as_tensor(r2_np, dtype=torch.float32, device=device)
            d  = torch.as_tensor(d_np,  dtype=torch.float32, device=device)

            bufD.add(o.detach(), a1.detach(), lp1.detach(), v1.detach(), r1, d)
            if bufA is not None:
                # o_att: (B, Na, obs_dim_att)
                B = o_att.shape[0]
                o_att_flat = o_att.reshape(B * Na, obs_dim_att)

                a2_flat  = a2.reshape(B * Na, act_dim)
                lp2_flat = lp2.reshape(B * Na)
                v2_flat  = v2.reshape(B * Na)

                r2_flat = r2.reshape(B * Na)

                # done is per-env; replicate for each attacker
                d_flat = d.repeat_interleave(Na)

                bufA.add(o_att_flat.detach(), a2_flat.detach(), lp2_flat.detach(), v2_flat.detach(), r2_flat, d_flat)


            if train_role == "def":
                ep_ret_def += r1_np

            if train_role == "att":
                # ep_ret_att += r2_np.mean(axis=1)   # shape (B,)
                ep_ret_att += r2_np.sum(axis=1)
            
            o = o2

            # ---- accumulate truth / belief metrics from Env.info ----
            for inf in infos:
                # count every env-step
                info_count += 1

                # always accumulate if present
                if "d1_true_norm" in inf:
                    d1_true_acc += inf["d1_true_norm"]

                if "d2_true_norm" in inf:
                    d2_true_acc += inf["d2_true_norm"]

                # belief distance: fall back safely
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

                if inf.get("oob_def", False): term_counts["oob_def"] += 1
                if inf.get("oob_att", False): term_counts["oob_att"] += 1
                if inf.get("hit_target", False): term_counts["hit_target"] += 1
                if inf.get("collision", False): term_counts["collision"] += 1


            global_env_step += num_envs




        with torch.no_grad():
            next_v_def = ppo.def_net.value(o)
        bufD.finalize(next_v_def)

        if bufA is not None:
            with torch.no_grad():
                o_att_np = vec.get_att_obs()
                o_att = torch.as_tensor(o_att_np, dtype=torch.float32, device=device)
                o_att_flat = o_att.reshape(num_envs * Na, obs_dim_att)
                next_v_att = ppo.att_net.value(o_att_flat)
            bufA.finalize(next_v_att)

        # ---- choose what to update ----
        if train_role == "def":
            ppo.update_defender_only(bufD)

        elif train_role == "att":
            if bufA is None:
                raise RuntimeError("train_role='att' requires attacker_mode='rl'")
            ppo.update_attacker_only(bufA)

        else:
            raise ValueError(f"Unknown train_role={train_role!r}")
        
        #explicit cleanup (place it right here)
        # del bufD
        # try:
        #     del bufA
        # except NameError:
        #     pass

        # gc.collect()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
                
        # =========================================================
        # NEW: Verify frozen opponent did not change
        # =========================================================
        if verify_freeze:
            # If training attacker, defender should be frozen (Phase 1)
            if (train_role == "att") and (snap_def is not None):
                assert_frozen_unchanged(snap_def, ppo.def_net, name="frozen_defender", tol=freeze_tol)

            # If training defender, attacker should be frozen (Phase 2)
            if (train_role == "def") and (snap_att is not None):
                assert_frozen_unchanged(snap_att, ppo.att_net, name="frozen_attacker", tol=freeze_tol)



        if upd % log_every == 0:
            R_def_mean = ep_ret_def.mean()
            R_att_mean = ep_ret_att.mean()

            # means over all steps collected this update
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
                flat_obs = bufD.obs.reshape(-1, obs_dim_def)
                distD = ppo.def_net.dist(flat_obs, who="def")
                muD = distD.mean.abs().mean().item()
                stdD = distD.stddev.mean().item()

                # if verify_freeze:
                #     # Check determinism of the opponent (not the learner)
                #     if train_role == "att":
                #         # opponent is defender
                #         assert_deterministic_action(ppo, flat_obs[:256], who="def", tol=0.0)
                #     if train_role == "def" and (ppo.att_net is not None):
                #         # opponent is attacker (RL opponent case)
                #         assert_deterministic_action(ppo, flat_obs[:256], who="att", tol=0.0)


                # obs = [p1c, p2c, rel, v1, v2]
                Dcfg = cfg["D"]
                p1c = flat_obs[:, :Dcfg]
                p2c = flat_obs[:, Dcfg:2*Dcfg]

                # -------------------------------
                # #5: opponent policy std logging
                # -------------------------------
                dist_opp = (
                    ppo.def_net.dist(flat_obs, who="def")
                    if train_role == "att"  # training attacker => opponent is defender
                    else ppo.att_net.dist(flat_obs, who="att")
                    if (train_role == "def" and ppo.att_net is not None)  # training defender => opponent is attacker RL
                    else None
                )
                if dist_opp is not None:
                    print("opp std mean:", dist_opp.stddev.mean().item())

                # ... your obs-derived d1/d2 means ...
                Dcfg = cfg["D"]
                p1c = flat_obs[:, :Dcfg]
                p2c = flat_obs[:, Dcfg:2*Dcfg]
                d1_obs_mean = p1c.pow(2).sum(-1).mean().sqrt().item()
                d2_obs_mean = p2c.pow(2).sum(-1).mean().sqrt().item()


            # grab learning rates (assuming two param groups: policy+logstd and value)
            lr_pi = ppo.def_opt.param_groups[0]["lr"]
            lr_vf = ppo.def_opt.param_groups[-1]["lr"]

            # ---- store in metrics ----
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
                gs = global_env_step  # x-axis = env steps

                # ===== Returns =====
                writer.add_scalar("returns/def_mean", R_def_mean, gs)
                writer.add_scalar("returns/att_mean", R_att_mean, gs)

                # ===== Distances (meters) =====
                writer.add_scalar("dist/def_true_p1_to_center_m", d1_true_mean, gs)
                writer.add_scalar("dist/att_true_p2_to_center_m", d2_true_mean, gs)
                writer.add_scalar("dist/att_belief_p2_to_center_m", d2_belief_mean, gs)

                # ===== Policy stats =====
                writer.add_scalar("policy/def_mu_abs_mean", muD, gs)
                writer.add_scalar("policy/def_std_mean", stdD, gs)

                # ===== Learning rates =====
                writer.add_scalar("lr/def_policy", lr_pi, gs)
                writer.add_scalar("lr/def_value",  lr_vf, gs)

                # ===== UKF stats (if enabled) =====
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
                writer.add_scalar("act/def_abs_max",  a1.abs().max().item(),  global_env_step)




            
            
    # ---- end-of-train cleanup ----
    try:
        del bufD
    except: pass
    try:
        del bufA
    except: pass
    try:
        del vec
    except: pass


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

                # attacker obs for this single env
                obsA = np.stack([env.obs_att_k(k) for k in range(env.num_attackers)], axis=0)  # (Na,5D)
                o_att = torch.as_tensor(obsA[None, :, :], dtype=torch.float32, device=ppo.def_net.logstd.device)  # (1,Na,5D)
                a2, _, _ = ppo.act(o_att, who="att")
            a1_np = a1.squeeze(0).cpu().numpy()
            a2_np = a2.squeeze(0).cpu().numpy()  # (Na,D)

            if a2_np.ndim == 1:
                a2_np = a2_np[None, :]          # -> (1, D)
            obs, r1, r2, done, info = env.step(a1_np, a2_np)
            states.append(env.state.copy())
            actions.append((a1_np.copy(), a2_np.copy()))
            infos.append(info)
        trajs.append({"states": np.stack(states), "actions": actions, "infos": infos})
    return trajs


def rollout_metrics(states: np.ndarray, center: np.ndarray, R: float):
    """
    states: [T+1, 12] for D=3 => [p1(3), v1(3), p2(3), v2(3)]
    center: (D,)
    R: arena radius (m)
    """
    D = center.shape[0]
    p1 = states[:, 0:D];      v1 = states[:, D:2*D]
    p2 = states[:, 2*D:3*D];  v2 = states[:, 3*D:4*D]

    d1 = np.sum((p1 - center)**2, axis=1) / (R*R)
    d2 = np.sum((p2 - center)**2, axis=1) / (R*R)
    rel2 = np.sum((p2 - p1)**2, axis=1) / (R*R)
    d2_delta = np.diff(d2, prepend=d2[:1])

    return {"d1_norm": d1, "d2_norm": d2, "rel2_norm": rel2, "d2_delta": d2_delta}

# =============================================================
# Plotting scripts
# =============================================================

def load_npz_metrics(path: str) -> dict:
    """
    Load a .npz metrics file saved via: np.savez(metrics_path, **metrics)

    Returns
    -------
    metrics : dict[str, np.ndarray]
        Keys map to 1D numpy arrays (or arrays as saved).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Metrics file not found: {path}")

    data = np.load(path, allow_pickle=True)
    metrics = {}
    for k in data.files:
        metrics[k] = data[k]
    return metrics


def _as_1d(x):
    """Best-effort convert to 1D float array (for plotting)."""
    x = np.asarray(x)
    if x.ndim == 0:
        return x.reshape(1).astype(float)
    if x.ndim > 1:
        # if it's a column vector or similar, flatten
        x = x.reshape(-1)
    return x.astype(float)


def _smooth_series(y, method="none", param=0.2):
    """
    Smooth a 1D series.

    method:
      - "none": no smoothing
      - "ema": exponential moving average with alpha=param (0<alpha<=1)
      - "ma":  moving average with window=int(param)
    """
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
        # pad at left so length stays the same
        pad = win - 1
        ypad = np.pad(y, (pad, 0), mode="edge")
        kernel = np.ones(win, dtype=float) / win
        return np.convolve(ypad, kernel, mode="valid")

    raise ValueError(f"Unknown smoothing method: {method!r}")


def plot_training_metrics(
    metrics: dict,
    title: str = "",
    smooth: str = "none",         # "none" | "ema" | "ma"
    smooth_param: float = 0.2,    # ema alpha OR ma window
    show: bool = False,

    # NEW saving knobs
    out_dir: str | None = None,
    save_prefix: str | None = None,
    dpi: int = 200,
    close: bool = True,
):
    """
    Plot the metrics saved by your train() loop.

    If out_dir is provided, saves figures there (one PNG per figure) and
    does not require showing.

    Returns
    -------
    figs : list[matplotlib.figure.Figure]
    saved_paths : list[str]  (empty if out_dir is None)
    """
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

    # x-axis
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

    # pick a reasonable prefix for filenames
    if save_prefix is None:
        save_prefix = title.strip() if title.strip() else "run"

    # 1) Returns
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

    # 2) Distances
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

    # 3) Learning rates
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

    # 4) Policy stats
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

    # 5) UKF stats
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

    # ---- save if requested ----
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        for tag, fig in figs:
            fname = f"{save_prefix}__{tag}.png"
            fpath = os.path.join(out_dir, fname)
            fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
            saved_paths.append(fpath)

    # ---- show / close behavior ----
    if show:
        plt.show()

    if close:
        for _, fig in figs:
            plt.close(fig)

    # return plain fig list for compatibility + saved paths
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
    """
    Creates a TensorBoard SummaryWriter under:
      runs/<run_name or timestamped name>/

    Also writes config.json for reproducibility.
    """
    root = cfg.get("tb_logdir", "runs")
    if run_name is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = cfg.get("tb_run_name", f"diffgame_{stamp}")

    logdir = os.path.join(root, run_name)
    os.makedirs(logdir, exist_ok=True)

    # Save config snapshot next to TB logs (nice for later)
    try:
        with open(os.path.join(logdir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2, default=str)
    except Exception as e:
        print("[tb] could not write config.json:", e)

    writer = SummaryWriter(log_dir=logdir)
    print(f"[tb] logging to: {logdir}")
    return writer, logdir


    # ---------------------------------------------------------
    # Helper: train defender teacher + distill to UKF student
    # ---------------------------------------------------------
def train_defender_with_distill(
    phase_name: str,
    attacker_mode: str,
    extra_train_cfg: Dict[str, Any] | None = None,
):
    """
    Always trains a full-state TEACHER defender.
    If cfg_teacher["distill"] == True, then runs a UKF STUDENT distillation phase.
    If False, skips distillation and returns student_ckpt=None.
    """

    # -------- TEACHER (full-state) --------

    cfg_teacher = config_for_train(
            attacker_mode=attacker_mode,
            train_role="def",
        )
    
    BUILD_TEACHER = True
    DISTILL = bool(cfg_teacher.get("distill", False))

    if BUILD_TEACHER: 
        cfg_teacher = config_for_train(
            attacker_mode=attacker_mode,
            train_role="def",
        )
        cfg_teacher["use_ukf"] = False  # teacher is full-state

        if extra_train_cfg is not None:
            cfg_teacher.update(extra_train_cfg)

        # IMPORTANT: read the flag AFTER updates, and default to False if absent

        build_dyn(cfg_teacher)

        if cfg_teacher["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"[{phase_name.upper()} TEACHER] device='cuda' but CUDA not available.")
        print(f"[{phase_name.upper()} TEACHER] Using device: {cfg_teacher['device']}")
        print(f"[{phase_name.upper()}] distill={DISTILL}")

        ppo_def, metrics_def = train(cfg_teacher)

        # --- Save defender teacher checkpoint ---
        def_teacher_ckpt = os.path.join(OUT_DIR, f"{phase_name}_teacher.pt")
        torch.save(ppo_def.def_net.state_dict(), def_teacher_ckpt)
        print(f"[{phase_name.upper()} TEACHER] Saved defender teacher to {def_teacher_ckpt}")

        # --- Save teacher metrics ---
        metrics_path = os.path.join(OUT_DIR, f"train_metrics_{phase_name}_teacher.npz")
        np.savez(metrics_path, **metrics_def)
        print(f"[{phase_name.upper()} TEACHER] Saved metrics to {metrics_path}")

    try:
        del ppo_def, metrics_def
    except: pass

    # def_teacher_ckpt = "Training_Policy/def0_def_teacher.pt"

    # -------- STUDENT (UKF) via distillation (optional) --------
    student_out = None
    if DISTILL:
        cfg_student = config_for_train(
            attacker_mode=attacker_mode,
            train_role="def",
        )

        cfg_student["use_ukf"] = True
        cfg_student["seed"] = cfg_teacher["seed"] + 1

        # If you want the student to inherit any relevant training knobs, do it here:
        # (e.g., same attacker opponent checkpoint/freeze settings)
        if extra_train_cfg is not None:
            cfg_student.update(extra_train_cfg)

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
        except: pass

    #Clearing cache
    # del ppo_def
    # del env
    # gc.collect()
    # torch.cuda.empty_cache()

    return def_teacher_ckpt, student_out

def train_attacker_with_distill(
    phase_name: str,
    attacker_mode: str,
    extra_train_cfg: Dict[str, Any] | None = None,
):
    """
    Wrapper that (1) optionally rule-pretrains the attacker, then (2) trains with PPO (teacher),
    then (3) distills (student). Adjust the distill part to match your pipeline.

    Args:
        phase_name: label for logging/checkpoints
        attacker_mode: e.g. "rl"
        extra_train_cfg: overrides merged into cfg_teacher
        do_rule_pretrain: if True, call pretrain_attacker_from_rule before PPO
        rule_pretrain_kwargs: forwarded into pretrain_attacker_from_rule(...)
    """
    # -------- TEACHER (full-state) --------
    cfg_teacher = config_for_train(
        attacker_mode=attacker_mode,
        train_role="att",
    )
    cfg_teacher["use_ukf"] = False  # teacher is full-state
    cfg_teacher["def_ckpt_path"] = def0_teacher_ckpt   # load defender₀ as opponent
    cfg_teacher["freeze_defender"] = True              # keep defender fixed

    DISTILL = bool(cfg_teacher.get("distill", False))

    build_dyn(cfg_teacher)

    # --- NEW: BC pretrain attacker from rule controller ---
    att_bc_ckpt = os.path.join(OUT_DIR, "att1_bc_init.pt")
    cfg_for_bc = cfg_teacher.copy()

    # During BC we still want an RL attacker net to exist,
    # but labels come from the rule controller.
    # Make sure attacker_mode is 'rl' (it already is here).
    cfg_for_bc["seed"] = cfg_teacher["seed"] + 123  # any offset you want

    pretrain_attacker_from_rule(
        cfg_for_bc,
        out_path=att_bc_ckpt,
        steps_per_env=int(cfg_for_bc.get("steps_per_env", 256)),
        total_updates=int(cfg_for_bc.get("att_bc_updates", 200)),
        bc_epochs=int(cfg_for_bc.get("att_bc_epochs", 4)),
        bc_mb_size=int(cfg_for_bc.get("att_bc_mb_size", 2048)),
    )

    # Now train attacker with PPO starting from the BC init
    cfg_teacher["att_init_path"] = att_bc_ckpt

    if extra_train_cfg is not None:
        cfg_teacher.update(extra_train_cfg)

    # IMPORTANT: read the flag AFTER updates, and default to False if absent
    DISTILL = bool(cfg_teacher.get("distill", False))

    # build_dyn(cfg_teacher)

    if cfg_teacher["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"[{phase_name.upper()} TEACHER] device='cuda' but CUDA not available.")
    print(f"[{phase_name.upper()} TEACHER] Using device: {cfg_teacher['device']}")
    print(f"[{phase_name.upper()}] distill={DISTILL}")

    ppo_att, metrics_att = train(cfg_teacher)

    # --- Save attacker teacher checkpoint ---
    att_teacher_ckpt = os.path.join(OUT_DIR, f"{phase_name}_teacher.pt")
    torch.save(ppo_att.att_net.state_dict(), att_teacher_ckpt)
    print(f"[{phase_name.upper()} TEACHER] Saved attacker teacher to {att_teacher_ckpt}")

    # --- Save teacher metrics ---
    metrics_path = os.path.join(OUT_DIR, f"train_metrics_{phase_name}_teacher.npz")
    np.savez(metrics_path, **metrics_att)
    print(f"[{phase_name.upper()} TEACHER] Saved metrics to {metrics_path}")

    # --- Optional: evaluate teacher (full-state) ---
    # cfg_eval = config_for_eval(
    #     attacker_mode=cfg_teacher.get("attacker_mode", attacker_mode),
    #     umax=cfg_teacher["umax"],
    #     T=cfg_teacher["T"],
    # )
    # cfg_eval["use_ukf"] = False
    # build_dyn(cfg_eval)
    # trajs = evaluate(ppo_att, cfg_eval, episodes=2)

    # ar = cfg_eval["arena"]
    # D = cfg_eval["D"]
    # center = np.array([ar["cx"], ar["cy"], (ar["cz"] if D == 3 else 0.0)], dtype=float)[:D]
    # R = float(ar["r"])
    # m = rollout_metrics(trajs[0]["states"], center, R)
    # print(
    #     f"[{phase_name.upper()} TEACHER metrics] "
    #     f"d2_T={m['d2_norm'][-1]:.3f}  d1_med={np.median(m['d1_norm']):.3f}  rel2_med={np.median(m['rel2_norm']):.3f}"
    # )

    try:
        del ppo_att, metrics_att
    except: pass

    # -------- STUDENT (UKF) via distillation (optional) --------
    student_out = None

    if DISTILL:
        cfg_student = config_for_train(
            attacker_mode=attacker_mode,
            train_role="att",
        )

        cfg_student["use_ukf"] = True
        cfg_student["seed"] = cfg_teacher["seed"] + 1

        # If you want the student to inherit any relevant training knobs, do it here:
        # (e.g., same attacker opponent checkpoint/freeze settings)
        if extra_train_cfg is not None:
            cfg_student.update(extra_train_cfg)

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
        except: pass

    #Clearing cache
    # del ppo_def
    # del env
    # gc.collect()
    # torch.cuda.empty_cache()

    return att_teacher_ckpt, student_out



# =============================================================
# Freeze / verification helpers
# =============================================================

def freeze_module_(m: torch.nn.Module):
    """Hard-freeze: eval() + requires_grad_(False) on all params."""
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)

@torch.no_grad()
def snapshot_state_dict(m: torch.nn.Module) -> dict:
    """CPU clone of state_dict tensors for exact comparison."""
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}

@torch.no_grad()
def max_state_dict_diff(snap: dict, m: torch.nn.Module) -> float:
    """Max |param - snap| over all tensors in state_dict."""
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



# =============================================================
# End phase cleanup
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
    """
    Best-effort memory cleanup after a training phase.

    - CPU RAM: delete references + gc.collect()
    - GPU VRAM (CUDA): empty_cache(), ipc_collect()
    - MPS (Apple): empty_cache()
    - Matplotlib: close figures so they don't accumulate
    """
    print(f"\n[cleanup] {tag} ...")

    # ---- close any lingering matplotlib figures (common silent RAM leak) ----
    if clear_matplotlib:
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass

    # ---- Python heap cleanup ----
    gc.collect()

    # ---- PyTorch device-specific cleanup ----
    if clear_cuda and torch.cuda.is_available():
        # Clears cached blocks held by the CUDA allocator (does not free tensors you still reference).
        torch.cuda.empty_cache()

        if clear_ipc:
            # Helps in some multi-process / DataLoader / vector-env setups.
            torch.cuda.ipc_collect()

        # Optional: if you're debugging fragmentation, you can print stats:
        # print(torch.cuda.memory_summary())

    if clear_mps and hasattr(torch, "mps") and torch.mps.is_available():
        # Apple Silicon
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    if sleep_s > 0:
        time.sleep(sleep_s)

    print(f"[cleanup] {tag} done.")


# =============================================================
# Main: stepped training (Def₀ -> Att₁ -> Def₁) + defender distillation
# =============================================================
if __name__ == "__main__":
    OUT_DIR = "Training_Policy"
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg_distillation = config_for_train(
        attacker_mode="rl",   # attacker is RL now
        train_role="att",     # PPO will only update attacker
    )
    DISTILL = cfg_distillation["distill"]


    # =========================================================
    # PHASE 0: Defender₀ vs rule-based attacker (teacher + distill)
    # =========================================================
    print("\n===== PHASE 0: Train DEFENDER_0 vs RULE attacker =====")
    def0_teacher_ckpt, def0_student_ckpt = train_defender_with_distill(
        phase_name="def0",
        attacker_mode="rule",
        extra_train_cfg=None,  # training vs the built-in rule-based attacker
    )
    end_phase_cleanup("Cleanup after PHASE 0")

    # def0_teacher_ckpt = "Training_Policy/def0_teacher.pt"


    # =========================================================
    # PHASE 1: Attacker₁ vs fixed Defender₀ (teacher only, for now)
    # =========================================================
    print("\n===== PHASE 1: Train ATTACKER_1 vs frozen DEFENDER_0 =====")


    att1_teacher_ckpt, att1_student_ckpt = train_attacker_with_distill(
        phase_name="att1",
        attacker_mode="rl",
        extra_train_cfg={
            "def_ckpt_path": def0_teacher_ckpt,
            # "freeze_attacker": True,   # keep attacker₁ fixed during defender₁ training
        },       
    )
    end_phase_cleanup("Cleanup after PHASE 1")


    # NOTE: we are *not* yet distilling the attacker here.
    # To distill the attacker, we'd need a clear attacker-side sensor / partial
    # observation model (UKF or otherwise). Once that's designed, we can add a
    # distillation routine analogous to distill_from_teacher but with who='att'.

    # =========================================================
    # PHASE 2: Defender₁ vs frozen Attacker₁ (teacher + distill)
    # =========================================================
    print("\n===== PHASE 2: Train DEFENDER_1 vs frozen ATTACKER_1 =====")

    def1_teacher_ckpt, def1_student_ckpt = train_defender_with_distill(
        phase_name="def1",
        attacker_mode="rl",   # attacker is RL now
        extra_train_cfg={
            "att_ckpt_path": att1_teacher_ckpt,
            "freeze_attacker": True,   # keep attacker₁ fixed during defender₁ training
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

