"""
rl_loop_diffgame.py
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
from typing import Dict, Any, Tuple, Callable, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt



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
        self.def_target_hit_buffer_frac = float(cfg.get("def_target_hit_buffer_frac", 0.0))


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


        self.def_target_hit_penalty_def = float(cfg.get("def_target_hit_penalty_def", 0.0))
        self.def_target_hit_reward_att  = float(cfg.get("def_target_hit_reward_att", 0.0))


        # NEW: collision termination (defender vs any attacker)
        self.collision_radius_m    = float(cfg.get("collision_radius_m", 0.0))  # meters; 0 disables
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
        self._d2_prev = None

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


        self.hit_buffer_def = float(self.def_target_hit_buffer_frac)
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

        d2_raw = float(np.dot(p2_geom - self.center, p2_geom - self.center))
        self._d2_prev = d2_raw / (self.radius**2)

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
        delta_d2 = d2 - (self._d2_prev if self._d2_prev is not None else d2)

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


                # # normalized distance from defender to center
                # rho1 = np.linalg.norm(p1 - self.center) / self.radius  # already computed above

                # # keepout radius: object radius + safety buffer, all normalized by arena radius
                # r_keepout = self.oi_radius_norm + self.def_center_safe_radius  # if you interpret def_center_safe_radius as BUFFER

                # eps = 1e-6
                # inside = max(0.0, r_keepout - rho1)   # >0 only if inside keepout

                # # Barrier that grows as you go deeper inside keepout
                # center_keepout = self.k_cent * (inside**2) / (eps + (rho1 - r_keepout)**2)
                # # simpler variant:
                # # center_keepout = self.k_cent * (inside**2) / (eps + inside**2)   # bounded 0..k_cent



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

        # ---- compute only requested reward(s) ----
        r1 = 0.0
        r2 = 0.0

        if need_def:
            r1 = (
                self.alpha * delta_d2
                + self.k_pos * d2
                - self.k_rel * rel2
                - self.k_vel * v1n2
                - k_vrad * (vrad1**2)
                - self.lD * a1n2
                - wall1
                - center_keepout
            )

        # if need_att:
        #     R = float(self.radius)

        #     # --- progress toward center (positive is good) ---
        #     # d2 is normalized squared radius; use radius = sqrt(d2) for better scaling
        #     d2_prev = float(self._d2_prev if self._d2_prev is not None else d2)

        #     r_prev = np.sqrt(max(d2_prev, 0.0))
        #     r_now  = np.sqrt(max(float(d2), 0.0))
        #     progress_r = (r_prev - r_now)  # bigger, better-shaped signal than (d2_prev - d2)

        #     # --- smooth keep-out penalty near defender ---
        #     # Penalize *approaching* min_sep smoothly (not just violating it).
        #     dist = float(np.linalg.norm(p2 - p1))  # meters
        #     x = dist / (float(self.att_min_sep) + 1e-9)  # 1.0 at the boundary
        #     close_pen = max(0.0, 1.0 - x) ** 2           # ramps up as dist -> min_sep

        #     # --- optional: "defender blocks direct path to center" shaping ---
        #     # Encourages attacker to approach center from angles that avoid defender being "in front".
        #     block_pen = 0.0
        #     if getattr(self, "k_att_block", 0.0) != 0.0:
        #         u = (self.center - p2)  # attacker -> center
        #         w = (p1 - p2)           # attacker -> defender
        #         u_norm = float(np.linalg.norm(u))
        #         w_norm = float(np.linalg.norm(w))
        #         if u_norm > 1e-9 and w_norm > 1e-9:
        #             u_hat = u / u_norm
        #             # positive if defender lies roughly along the approach direction to center
        #             proj = float(np.dot(w, u_hat)) / (w_norm + 1e-9)
        #             block_pen = max(0.0, proj) ** 2

        #     # wall penalty (already computed wall2)
        #     # speed/effort (already computed v2n2, a2n2)

        #     r2 = (
        #         + self.k_att_prog  * progress_r
        #         - self.k_att_cent  * r_now
        #         - self.k_att_close * close_pen
        #         - getattr(self, "k_att_block", 0.0) * block_pen
        #         - self.k_vel       * v2n2
        #         - self.lA          * a2n2
        #         - wall2
        #     )

        if need_att:
            # normalized squared distance to center (you already computed d2)
            # d2 = ||p2-center||^2 / R^2

            # normalized effort (you already computed a2n2)
            # a2n2 = ||a2||^2 / umax^2

            r2 = (
                - self.k_att_cent * d2
                - self.lA         * a2n2
                - wall2
            )




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



        # ---- termination always uses TRUE state ----


        # # target hit: (keep your current meaning = first attacker hits target)

        hit_target = False
        # p2_true = pA_list[0]
        # # rho2_true = np.linalg.norm(p2_true - self.center) / self.radius
        # rho2_rel_to_target = np.linalg.norm(p2_true - self.center)

        # hit_target = False
        # if self.att_target_hit_radius > 0.0 and rho2_rel_to_target <= (1 + self.att_target_hit_radius)*self.oi_radius:
        #     hit_target = True

        # attacker target hit: (keep your current meaning = first attacker hits target)



        rho_att = np.linalg.norm(p2 - self.center) / self.radius
        rho_def = np.linalg.norm(p1 - self.center) / self.radius

        thresh_def = (1.0 + self.hit_buffer_def) * self.oi_radius_norm
        thresh_att = (1.0 + self.hit_buffer_att) * self.oi_radius_norm


        att_hit_target = (self.oi_radius_norm > 0.0) and (rho_att <= thresh_att)
        def_hit_target = (self.oi_radius_norm > 0.0) and (rho_def <= thresh_def)


        hit_target = att_hit_target or def_hit_target


        # collision: defender within collision_radius_m of ANY attacker (TRUE distance)
        collision = False
        if self.collision_radius_m > 0.0:
            for pA_true in pA_list:
                if np.linalg.norm(pA_true - p1) <= self.collision_radius_m:
                    collision = True
                    break

        oob1 = (rho1 >= self.margin)

        # oob for ANY attacker
        oob2_any = False
        for pA_true in pA_list:
            rhoA_true = np.linalg.norm(pA_true - self.center) / self.radius
            if rhoA_true >= self.margin:
                oob2_any = True
                break

        done = (oob1 or oob2_any or hit_target or collision)

        if collision:
            r1 -= self.collision_penalty_def
            r2 -= self.collision_penalty_att

        if need_def:
            if oob1: r1 -= self.wallK
            if att_hit_target: r1 -= self.def_target_hit_penalty_def
            if def_hit_target: r1 -= self.att_target_hit_penalty_def

        if need_att:
            if oob2_any: r2 -= self.wallK
            if att_hit_target: r2 += self.att_target_hit_reward_att
            if def_hit_target: r2 += self.def_target_hit_reward_att

        if need_att and done:
            # reward only if attacker ended near target
            # (oi_radius_norm > 0 means you actually defined a target)
            if self.oi_radius_norm > 0.0:
                if att_hit_target:
                    r2 += float(self.beta)   # e.g. beta = 1.0 or 2.0
                else:
                    r2 -= float(self.beta)   # optional: discourage ending without hitting





        # track d2_prev based on the geometry used for reward (same as your current logic)
        self._d2_prev = d2

        # info: you can also gate what you store here if you want
        # ---- always compute these for logging ----
        d1_true_norm = float(np.dot(p1 - self.center, p1 - self.center)) / (self.radius**2)
        d2_true_norm = float(np.dot(p2 - self.center, p2 - self.center)) / (self.radius**2)
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

        return self._obs(), float(r1), float(r2), bool(done), info




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
        r1, r2, done, info = [], [], [], []
        for i, e in enumerate(self.envs):
            o, R1, R2, d, inf = e.step(a1_env[i], aA_env[i], reward_mode=reward_mode)
            if d:
                o = e.reset()
            obs_next.append(o)
            r1.append(R1); r2.append(R2); done.append(d); info.append(inf)
        self.obs = np.stack(obs_next, axis=0)
        return self.obs, np.array(r1), np.array(r2), np.array(done, dtype=np.float32), info






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
class DiffLSLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.D = int(cfg["D"])
        self.ridge = float(cfg.get("prior_ridge", 1e-2))

        # robust dt fetch (adjust if your config uses a different key)
        self.dt = float(cfg.get("dt", cfg.get("h", cfg.get("dyn", {}).get("dt", 1.0))))

    def forward(self, obs: torch.Tensor, who: str):
        B, D = obs.shape[0], self.D
        dt = self.dt

        # obs = [p1c, p2c, rel, v1, v2]  (5D dims total; 15 when D=3)
        p1c = obs[:, 0:D]
        p2c = obs[:, D:2*D]
        v1  = obs[:, 3*D:4*D]
        v2  = obs[:, 4*D:5*D]

        # one-step position error to center (in centered coordinates)
        e1 = p1c + dt * v1   # (B, D)
        e2 = p2c + dt * v2   # (B, D)

        alpha = 0.5 * (dt * dt)
        k = alpha / (alpha * alpha + self.ridge)  # scalar

        u_def_prior = -k * e1
        u_att_prior = -k * e2
        u_prior = u_def_prior if who == "def" else u_att_prior

        # IMPORTANT: feats must be 4D = 12 when D=3
        feats = torch.cat([p1c, p2c, v1, v2], dim=-1)
        return feats, u_prior


class DiffNashLayer(nn.Module):
    """
    One-step Nash prior via an external IPOPT-based game solver.

    - Reconstructs (x1, x2) = [p1, v1], [p2, v2] from the observation.
    - Calls an external 'Nash game solver' (IPOPT-based) that returns (u1*, u2*).
    - Returns u1* as prior if who='def', u2* if who='att'.

    Notes:
    - This layer *does not* backprop through the IPOPT solver; u_prior is treated
      as a fixed, non-differentiable prior. Gradients only flow through the
      learned residual policy on top of u_prior.
    - The actual game definition (costs, constraints) is inside the external
      solver you provide (e.g. nash_ipopt_solver.solve_nash_ipopt).
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.D = int(cfg["D"])
        self.dt = float(cfg["dt"])

        # Dynamics, center, etc., same geometry as DiffLS
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

        # Where to find the IPOPT game solver
        # Example expected structure in cfg:
        # cfg["nash_solver"] = {
        #     "module": "nash_ipopt_solver",
        #     "fn":     "solve_nash_ipopt",
        #     "params": {...}   # optional dict passed to solver
        # }
        solver_cfg = cfg.get("nash_solver", {})
        module_name = solver_cfg.get("module", "nash_ipopt_solver")
        fn_name     = solver_cfg.get("fn", "solve_nash_ipopt")
        self.solver_params = solver_cfg.get("params", {})

        mod = importlib.import_module(module_name)
        self.nash_solve = getattr(mod, fn_name)

    def forward(self, obs: torch.Tensor, who: str):
        """
        obs:  [B, 5*D] = [p1-center, p2-center, (p2-p1), v1, v2]
        who:  'def' or 'att'
        """
        B, D = obs.shape[0], self.D
        device = obs.device
        dtype  = obs.dtype

        center = self.center.to(dtype=dtype, device=device)

        # Decompose obs into pieces
        p1c = obs[:, 0:D]          # defender pos (centered)
        p2c = obs[:, D:2*D]        # attacker pos (centered)
        v1  = obs[:, 3*D:4*D]
        v2  = obs[:, 4*D:5*D]

        # Recover absolute positions
        p1 = p1c + center          # [B, D]
        p2 = p2c + center          # [B, D]

        # Construct state vectors x1, x2 = [p, v]
        x1 = torch.cat([p1, v1], dim=-1)   # [B, 2D]
        x2 = torch.cat([p2, v2], dim=-1)   # [B, 2D]

        # We will call the IPOPT solver in numpy-space (no grads)
        x1_np = x1.detach().cpu().numpy()
        x2_np = x2.detach().cpu().numpy()

        u1_list = []
        u2_list = []

        # Call solver for each batch element
        for i in range(B):
            u1_i, u2_i = self.nash_solve(x1_np[i], x2_np[i], self.solver_params)
            # Expect u1_i, u2_i to be numpy arrays of shape (D,)
            u1_list.append(np.asarray(u1_i, dtype=np.float32))
            u2_list.append(np.asarray(u2_i, dtype=np.float32))

        u1_np = np.stack(u1_list, axis=0)   # [B, D]
        u2_np = np.stack(u2_list, axis=0)   # [B, D]

        u1_prior = torch.as_tensor(u1_np, device=device, dtype=dtype)
        u2_prior = torch.as_tensor(u2_np, device=device, dtype=dtype)
        u_prior  = u1_prior if who == "def" else u2_prior

        # As in DiffLS, the "features" are [p1c, p2c, v1, v2]
        feats = torch.cat([p1c, p2c, v1, v2], dim=-1)  # [B, 4D]
        return feats, u_prior
    

class NoPriorLayer(nn.Module):
    """
    No analytic prior: just repackage the observation into features
    and return u_prior = 0.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.D = int(cfg["D"])
        ar = cfg["arena"]
        c = np.array(
            [ar["cx"], ar["cy"], (ar["cz"] if self.D == 3 else 0.0)],
            dtype=np.float32
        )[: self.D]
        self.register_buffer("center", torch.tensor(c, dtype=torch.float32))

    def forward(self, obs: torch.Tensor, who: str):
        """
        obs: [B, 5*D] = [p1-center, p2-center, (p2-p1), v1, v2]
        Return:
          feats: [B, 4*D] = [p1c, p2c, v1, v2]
          u_prior: [B, D] = 0
        """
        B, D = obs.shape[0], self.D
        device, dtype = obs.device, obs.dtype

        p1c = obs[:, 0:D]
        p2c = obs[:, D:2*D]
        v1  = obs[:, 3*D:4*D]
        v2  = obs[:, 4*D:5*D]

        feats = torch.cat([p1c, p2c, v1, v2], dim=-1)  # (B, 4D)
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
        elif prior_type == "nash":
            self.layer = DiffNashLayer(cfg)
        elif prior_type == "none":
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




# =============================================================
# PPO Core (defender learns; attacker optionally rule-based)
# =============================================================
class PPO:
    def __init__(self, obs_dim: int, act_dim: int, cfg: Dict[str, Any], device="cpu"):
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
        self.def_net = ActorCriticDiff(obs_dim, act_dim, cfg).to(device)
        self.def_opt = optim.Adam([
            {"params": list(self.def_net.pi.parameters()) + list(self.def_net.mu_res.parameters()), "lr": cfg["policy_lr"]},
            {"params": [self.def_net.logstd], "lr": cfg["policy_lr"] * 0.5},
            {"params": self.def_net.vf.parameters(), "lr": cfg["value_lr"]},
        ])
        # NEW: remember initial defender LRs
        self.def_base_lrs = [g["lr"] for g in self.def_opt.param_groups]

        # Attacker: rule or RL
        if self.attacker_mode == "rl":
            self.att_net = ActorCriticDiff(obs_dim, act_dim, cfg).to(device)
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
            # obs layout: [p1c, pA1c..pANc, rel1..relN, v1, vA1..vAN]
            # We must parse based on Na.
            D = int(self.act_scale * 0 + obs_batch.shape[-1])  # noop keep torch happy
            D = int((obs_batch.shape[1]) // (2 + 2*Na + 1 + Na))  # not reliable
            # Instead use cfg D (robust):
            D = int(self.def_net.layer.D) if hasattr(self.def_net, "layer") else int(obs_batch.shape[1] // 5)

            ob = obs_batch.detach().cpu().numpy()
            # parse p1c
            p1c = ob[:, 0:D]
            off = D

            # pA centered (Na blocks)
            pA_c = []
            for k in range(Na):
                pA_c.append(ob[:, off:off + D])
                off += D

            # rel blocks (Na)
            rels = []
            for k in range(Na):
                rels.append(ob[:, off:off + D])
                off += D

            # v1
            v1 = ob[:, off:off + D]
            off += D

            # vA blocks (Na)
            vA = []
            for k in range(Na):
                vA.append(ob[:, off:off + D])
                off += D

            center = self.rule_ctrl.center

            acts = []
            for i in range(B):
                p1 = p1c[i] + center
                # recover absolute attackers:
                pA_list = [pA_c[k][i] + center for k in range(Na)]
                vA_list = [vA[k][i] for k in range(Na)]
                uA = self.rule_ctrl.act_multi(p1, v1[i], pA_list, vA_list)  # (Na, D)
                acts.append(uA)

            a_np = np.stack(acts, axis=0)  # (B, Na, D)
            a_t = torch.as_tensor(a_np, dtype=obs_batch.dtype, device=obs_batch.device)

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

def distill_from_teacher(
    cfg: Dict[str, Any],
    teacher_ckpt_path: str,
    out_path: str = "ppo_def_ukf_distilled.pt",
):
    """
    Distillation phase:
      - Env runs with UKF / partial observation (cfg['use_ukf']=True).
      - Defender actions come from a frozen full-state teacher policy.
      - Student defender sees *belief-based* obs and imitates teacher actions
        via supervised MSE on the mean action (behavior cloning).

    Returns
    -------
    student : ActorCriticDiff
        The distilled student network.
    metrics : Dict[str, List[float]]
        Simple metrics over distillation, currently:
            - "update": list of update indices where we logged
            - "bc_mse_dbg": behavior cloning MSE on a debug batch
    """
    set_seed(cfg["seed"])
    device = cfg["device"]

    def make_env():
        return Env(cfg)

    num_envs      = int(cfg.get("num_envs", 8))
    steps_per_env = int(cfg.get("steps_per_env", 256))
    total_updates = int(cfg.get("total_updates", 300))
    log_every     = int(cfg.get("log_every", 10))

    # Vectorized UKF env (student view)
    vec = VecEnv(make_env, num_envs)
    obs_dim = vec.obs.shape[1]
    act_dim = int(cfg["D"])

    # ---------- Teacher (full-state policy) ----------
    teacher = ActorCriticDiff(obs_dim, act_dim, cfg).to(device)
    teacher_state = torch.load(teacher_ckpt_path, map_location=device)
    teacher.load_state_dict(teacher_state)
    teacher.eval()   # freeze
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ---------- Student (belief-based policy) ----------
    student = ActorCriticDiff(obs_dim, act_dim, cfg).to(device)
    bc_lr = float(cfg.get("distill_lr", cfg["policy_lr"]))
    student_opt = optim.Adam(
        list(student.pi.parameters()) + list(student.mu_res.parameters()) + [student.logstd],
        lr=bc_lr,
    )

    # Attacker remains the same rule-based controller
    rule_ctrl = AttackerRuleController(cfg)

    mb_size    = int(cfg.get("distill_mb_size", 1024))
    bc_epochs  = int(cfg.get("distill_epochs", 4))

    # --- metrics container for distillation ---
    metrics = {
        "update": [],
        "bc_mse_dbg": [],
    }

    print("=== Distillation: teacher -> UKF student ===")
    print(f"Teacher checkpoint: {teacher_ckpt_path}")
    print(f"Saving distilled student to: {out_path}")

    for upd in range(1, total_updates + 1):
        # Storage for this "update"
        obs_buf = torch.zeros(steps_per_env, num_envs, obs_dim, device=device)
        act_buf = torch.zeros(steps_per_env, num_envs, act_dim, device=device)

        # --------- Data collection (teacher drives, student observes) ---------
        for t in range(steps_per_env):
            # Student obs (belief-based) from UKF env
            o_student = torch.as_tensor(vec.obs, dtype=torch.float32, device=device)

            # Build full-state obs for teacher from TRUE state
            o_teacher = build_full_obs_from_envs(vec, device)

            with torch.no_grad():
                # Teacher action (env-scaled)
                a_teacher, _, _ = teacher.act(o_teacher, who="def", act_scale=float(cfg["umax"]))

            # Attacker actions from rule controller
            acts_att = []
            for e in vec.envs:
                p1, v1, pA_list, vA_list = e._unpack(e.state)
                # For now we assume num_attackers == 1
                p2 = pA_list[0]
                v2 = vA_list[0]
                a2 = rule_ctrl.act(p1, v1, p2, v2).astype(np.float32)
                acts_att.append(a2)
            a2_env = np.stack(acts_att, axis=0)  # shape [num_envs, D]

            # Step env using teacher actions for defender
            o2_np, _, _, _, _ = vec.step(
                a_teacher.cpu().numpy(), a2_env
            )


            # Store belief-obs + teacher action for BC
            obs_buf[t] = o_student
            act_buf[t] = a_teacher

            vec.obs = o2_np

        # --------- Behavior cloning update on collected batch ---------
        B = steps_per_env * num_envs
        obs_flat = obs_buf.reshape(B, obs_dim)
        act_flat = act_buf.reshape(B, act_dim)

        # ✅ Option C cleanup at the end of distillation
        # del obs_buf, act_buf
        # gc.collect()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        for _ in range(bc_epochs):
            perm = torch.randperm(B, device=device)
            for start in range(0, B, mb_size):
                idx = perm[start:start + mb_size]
                o = obs_flat[idx]
                target = act_flat[idx]

                dist = student.dist(o, who="def")
                mu = dist.mean
                bc_loss = ((mu - target) ** 2).mean()

                student_opt.zero_grad(set_to_none=True)
                bc_loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), cfg["max_grad_norm"])
                student_opt.step()

        # --- logging on a debug batch ---
        if upd % log_every == 0:
            with torch.no_grad():
                dist_dbg = student.dist(obs_flat[:min(B, 2048)], who="def")
                mse_dbg  = ((dist_dbg.mean - act_flat[:min(B, 2048)])**2).mean().item()
            print(f"[distill {upd:05d}] BC MSE (dbg batch) = {mse_dbg:.3e}")

            metrics["update"].append(upd)
            metrics["bc_mse_dbg"].append(mse_dbg)

    # Save student weights
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

    Collect (obs, a_rule) by rolling the env with:
      - defender policy (either random-ish PPO init, or optional frozen ckpt if provided)
      - attacker actions from AttackerRuleController

    Then train attacker network to predict a_rule via MSE on dist.mean.
    """
    assert cfg.get("attacker_mode", "rule") == "rl", "BC pretrain requires attacker_mode='rl'"
    assert int(cfg.get("num_attackers", 1)) == 1, "BC pretrain currently assumes num_attackers=1"

    set_seed(cfg["seed"])
    device = cfg["device"]

    build_dyn(cfg)             # <-- ADD THIS (or ensure_dyn(cfg))

    def make_env():
        return Env(cfg)

    num_envs = int(cfg.get("num_envs", 8))
    vec = VecEnv(make_env, num_envs)
    obs_dim = vec.obs.shape[1]
    act_dim = int(cfg["D"])

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
        act_buf = torch.zeros(steps_per_env, num_envs, act_dim, device=device)

        for t in range(steps_per_env):
            o = torch.as_tensor(vec.obs, dtype=torch.float32, device=device)

            # Defender action: deterministic if frozen, else sample
            with torch.no_grad():
                det_def = (def_ckpt is not None)  # if we loaded a defender, keep it deterministic
                a1, _, _ = ppo.act(o, who="def", deterministic=det_def)

            # Rule attacker labels from TRUE state (more stable than obs-derived)
            a2_list = []
            for e in vec.envs:
                p1, v1, pA_list, vA_list = e._unpack(e.state)
                p2 = pA_list[0]; v2 = vA_list[0]
                a2 = rule_ctrl.act(p1, v1, p2, v2).astype(np.float32)
                a2_list.append(a2)
            a2_np = np.stack(a2_list, axis=0)  # (N, D)

            # Step env using defender action + rule attacker action
            o2_np, _, _, _, _ = vec.step(a1.cpu().numpy(), a2_np, reward_mode="both")

            obs_buf[t] = o
            act_buf[t] = torch.as_tensor(a2_np, dtype=torch.float32, device=device)

            vec.obs = o2_np

        # Flatten for supervised learning
        B = steps_per_env * num_envs
        obs_flat = obs_buf.reshape(B, obs_dim)
        act_flat = act_buf.reshape(B, act_dim)

        # ✅ Option C cleanup at the end of distillation
        # del obs_buf, act_buf
        # gc.collect()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        # BC epochs
        for _ in range(bc_epochs):
            perm = torch.randperm(B, device=device)
            for st in range(0, B, bc_mb_size):
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
                dist_dbg = ppo.att_net.dist(obs_flat[:min(B, 4096)], who="att")
                mse_dbg = ((dist_dbg.mean - act_flat[:min(B, 4096)]) ** 2).mean().item()
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
    obs_dim = vec.obs.shape[1]
    act_dim = int(cfg["D"])

    ppo = PPO(obs_dim, act_dim, cfg, device=device)

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
        bufD = RolloutBuffer(obs_dim, act_dim, num_envs, steps_per_env, device)
        rule_att = (cfg.get("attacker_mode", "rule") == "rule")
        if (not rule_att) and (train_role in ("att", "both")):
            bufA = RolloutBuffer(obs_dim, act_dim, num_envs, steps_per_env, device)
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
                a2, lp2, v2 = ppo.act(o, who="att", deterministic=det_att)


            a1_np = a1.cpu().numpy()

            a2_np = a2.cpu().numpy()
            # normalize shapes to what Env expects:
            # - if attacker RL: (B, D) ok
            # - if rule attacker: (B, Na, D) ok
            # - if rule attacker but produced (B, D): expand to (B,1,D)
            if a2_np.ndim == 2 and cfg.get("attacker_mode", "rule") == "rule" and cfg.get("num_attackers", 1) == 1:
                a2_np = a2_np[:, None, :]

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
                bufA.add(o.detach(), a2.detach(), lp2.detach(), v2.detach(), r2, d)


            if train_role == "def":
                ep_ret_def += r1_np

            if train_role == "att":
                ep_ret_att += r2_np
            
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



        with torch.no_grad():
            next_v_def = ppo.def_net.value(o)
        bufD.finalize(next_v_def)

        if bufA is not None:
            with torch.no_grad():
                next_v_att = ppo.att_net.value(o)
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
                flat_obs = bufD.obs.reshape(-1, obs_dim)
                distD = ppo.def_net.dist(flat_obs, who="def")
                muD = distD.mean.abs().mean().item()
                stdD = distD.stddev.mean().item()

                if verify_freeze:
                    # Check determinism of the opponent (not the learner)
                    if train_role == "att":
                        # opponent is defender
                        assert_deterministic_action(ppo, flat_obs[:256], who="def", tol=0.0)
                    if train_role == "def" and (ppo.att_net is not None):
                        # opponent is attacker (RL opponent case)
                        assert_deterministic_action(ppo, flat_obs[:256], who="att", tol=0.0)


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
    cfg_teacher["use_ukf"] = False  # teacher is full-state

    if extra_train_cfg is not None:
        cfg_teacher.update(extra_train_cfg)

    # IMPORTANT: read the flag AFTER updates, and default to False if absent
    DISTILL = bool(cfg_teacher.get("distill", False))

    build_dyn(cfg_teacher)

    if cfg_teacher["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"[{phase_name.upper()} TEACHER] device='cuda' but CUDA not available.")
    print(f"[{phase_name.upper()} TEACHER] Using device: {cfg_teacher['device']}")
    print(f"[{phase_name.upper()}] distill={DISTILL}")

    ppo_def, metrics_def = train(cfg_teacher)

    # --- Save defender teacher checkpoint ---
    def_teacher_ckpt = os.path.join(OUT_DIR, f"{phase_name}_def_teacher.pt")
    torch.save(ppo_def.def_net.state_dict(), def_teacher_ckpt)
    print(f"[{phase_name.upper()} TEACHER] Saved defender teacher to {def_teacher_ckpt}")

    # --- Save teacher metrics ---
    metrics_path = os.path.join(OUT_DIR, f"train_metrics_{phase_name}_teacher.npz")
    np.savez(metrics_path, **metrics_def)
    print(f"[{phase_name.upper()} TEACHER] Saved metrics to {metrics_path}")

    # --- Optional: evaluate teacher (full-state) ---
    cfg_eval = config_for_eval(
        attacker_mode=cfg_teacher.get("attacker_mode", attacker_mode),
        umax=cfg_teacher["umax"],
        T=cfg_teacher["T"],
    )
    cfg_eval["use_ukf"] = False
    build_dyn(cfg_eval)
    trajs = evaluate(ppo_def, cfg_eval, episodes=2)

    ar = cfg_eval["arena"]
    D = cfg_eval["D"]
    center = np.array([ar["cx"], ar["cy"], (ar["cz"] if D == 3 else 0.0)], dtype=float)[:D]
    R = float(ar["r"])
    m = rollout_metrics(trajs[0]["states"], center, R)
    print(
        f"[{phase_name.upper()} TEACHER metrics] "
        f"d2_T={m['d2_norm'][-1]:.3f}  d1_med={np.median(m['d1_norm']):.3f}  rel2_med={np.median(m['rel2_norm']):.3f}"
    )

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

        student_out = os.path.join(OUT_DIR, f"{phase_name}_def_ukf_student.pt")
        student, metrics_student = distill_from_teacher(
            cfg_student, def_teacher_ckpt, out_path=student_out
        )
        print(f"[{phase_name.upper()} STUDENT] Distilled UKF student saved to {student_out}")

        distill_metrics_path = os.path.join(OUT_DIR, f"distill_metrics_{phase_name}_student.npz")
        np.savez(distill_metrics_path, **metrics_student)
        print(f"[{phase_name.upper()} STUDENT] Saved distillation metrics to {distill_metrics_path}")

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

    build_dyn(cfg_teacher)

    if cfg_teacher["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"[{phase_name.upper()} TEACHER] device='cuda' but CUDA not available.")
    print(f"[{phase_name.upper()} TEACHER] Using device: {cfg_teacher['device']}")
    print(f"[{phase_name.upper()}] distill={DISTILL}")

    ppo_att, metrics_att = train(cfg_teacher)

    # --- Save defender teacher checkpoint ---
    att_teacher_ckpt = os.path.join(OUT_DIR, f"{phase_name}_def_teacher.pt")
    torch.save(ppo_att.def_net.state_dict(), att_teacher_ckpt)
    print(f"[{phase_name.upper()} TEACHER] Saved attacker teacher to {att_teacher_ckpt}")

    # --- Save teacher metrics ---
    metrics_path = os.path.join(OUT_DIR, f"train_metrics_{phase_name}_teacher.npz")
    np.savez(metrics_path, **metrics_att)
    print(f"[{phase_name.upper()} TEACHER] Saved metrics to {metrics_path}")

    # --- Optional: evaluate teacher (full-state) ---
    cfg_eval = config_for_eval(
        attacker_mode=cfg_teacher.get("attacker_mode", attacker_mode),
        umax=cfg_teacher["umax"],
        T=cfg_teacher["T"],
    )
    cfg_eval["use_ukf"] = False
    build_dyn(cfg_eval)
    trajs = evaluate(ppo_att, cfg_eval, episodes=2)

    ar = cfg_eval["arena"]
    D = cfg_eval["D"]
    center = np.array([ar["cx"], ar["cy"], (ar["cz"] if D == 3 else 0.0)], dtype=float)[:D]
    R = float(ar["r"])
    m = rollout_metrics(trajs[0]["states"], center, R)
    print(
        f"[{phase_name.upper()} TEACHER metrics] "
        f"d2_T={m['d2_norm'][-1]:.3f}  d1_med={np.median(m['d1_norm']):.3f}  rel2_med={np.median(m['rel2_norm']):.3f}"
    )

    # -------- STUDENT (UKF) via distillation (optional) --------
    student_out = None
    if DISTILL:
        cfg_student = config_for_train(
            attacker_mode="rl",
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

        student_out = os.path.join(OUT_DIR, f"{phase_name}_att_ukf_student.pt")
        student, metrics_student = distill_from_teacher(
            cfg_student, att_teacher_ckpt, out_path=student_out
        )
        print(f"[{phase_name.upper()} STUDENT] Distilled UKF student saved to {student_out}")

        distill_metrics_path = os.path.join(OUT_DIR, f"distill_metrics_{phase_name}_student.npz")
        np.savez(distill_metrics_path, **metrics_student)
        print(f"[{phase_name.upper()} STUDENT] Saved distillation metrics to {distill_metrics_path}")

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
    # def0_teacher_ckpt, def0_student_ckpt = train_defender_with_distill(
    #     phase_name="def0",
    #     attacker_mode="rule",
    #     extra_train_cfg=None,  # training vs the built-in rule-based attacker
    # )

    def0_teacher_ckpt = os.path.join(OUT_DIR, "def0_def_teacher.pt")

    # =========================================================
    # PHASE 1: Attacker₁ vs fixed Defender₀ (teacher only, for now)
    # =========================================================
    print("\n===== PHASE 1: Train ATTACKER_1 vs frozen DEFENDER_0 =====")


    att1_teacher_ckpt, att1_student_ckpt = train_attacker_with_distill(
        phase_name="att1",
        attacker_mode="rl",
        extra_train_cfg=None,  # training vs the built-in rule-based attacker
    )

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
    print(f"Defender_0 teacher:  {def0_teacher_ckpt}")
    print(f"Defender_0 student:  {def0_student_ckpt if def0_student_ckpt else '(skipped)'}")

    print(f"Attacker_1 teacher:  {att1_teacher_ckpt}")
    print(f"Defender_1 teacher:  {def1_teacher_ckpt}")
    print(f"Defender_1 student:  {def1_student_ckpt if def1_student_ckpt else '(skipped)'}")

