from typing import Any, Callable, Dict, List

import copy
import numpy as np
import torch
import os

from ukf_estimator import AgentUKF, _body_bearing_from_world, _azel_from_body_vec
from config_rl_1v2 import build_dyn
from core.utils import set_seed

# =============================================================
# Environment (1 defender vs N attackers)
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
        self.k_pos = float(cfg.get("k_pos"))
        self.k_dock = float(cfg.get("k_dock"))


        self.lD    = float(cfg["effort_def"])
        self.lA    = float(cfg["effort_att"])
        self.wallK = float(cfg["wall_penalty"])
        self.soft_wall = float(cfg.get("soft_wall_start"))
        self.margin = float(cfg["arena_terminate_margin"])  # 1.0 = at radius

        # NEW: attacker "hit object" termination around center (normalized wrt arena R)

        oi = cfg.get("oi", {})


        self.oi_radius = float(oi.get("r", 0.0))
        self.oi_radius_norm = self.oi_radius / self.radius if self.radius > 0 else 0.0


        self.hit_buffer_def = float(cfg.get("hit_buffer_def"))
        self.hit_buffer_att = float(cfg.get("hit_buffer_att"))

        self.target_hit_reward_penalty = float(cfg.get("target_hit_reward_penalty"))
        self.collision_penalty = float(cfg.get("collision_penalty"))
        self.wall_penalty = float(cfg.get("wall_penalty"))

        self.fuel_depletion_penalty = float(cfg.get("fuel_depletion_penalty"))


        # NEW: collision termination (defender vs any attacker)
        self.collision_radius_m    = float(cfg.get("collision_radius_m"))  # meters; 0 disables


        # ---- UKF / measurement model knobs ----
        self.use_ukf          = bool(cfg.get("use_ukf", False))
        self.use_meas_reward  = bool(cfg.get("use_meas_reward", False))
        self.meas_innov_coef  = float(cfg.get("meas_innov_coef"))  # weight on innovation^2
        self.meas_cov_coef    = float(cfg.get("meas_cov_coef"))    # weight on trace(P_pos)
        self.reward_from_belief = bool(cfg.get("reward_from_belief", False))
        self.belief_clip_factor = float(cfg.get("belief_clip_factor", 2.0))
        self.reset_ukf_on_diverge = bool(cfg.get("reset_ukf_on_diverge", True))

        #Fuel config
        fuel = cfg.get("fuel", {})
        self.use_fuel = bool(fuel.get("enable"))

        if self.use_ukf and self.D != 3:
            raise ValueError("UKF / bearing-only measurement currently implemented for D=3 only.")

        self.ukf = None
        self._latest_meas_innov = 0.0
        self._latest_meas_trP   = 0.0

        self.record_ic_history = bool(cfg.get("record_ic_history", False))
        self.max_ic_history = int(cfg.get("max_ic_history", 200000))

        self.ic_history_def = []
        self.ic_history_att = []

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

        if self.use_fuel:
            self.k_eff_def = float(cfg.get("k_eff_def"))
            self.k_eff_att = float(cfg.get("k_eff_att"))


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

        # --- Opponent-domain randomization (used when training attacker) ---
        self.opp_domain = "def0"  # default
        mix = cfg.get("opp_mix", {}) or {}
        self.opp_resample = str(mix.get("resample", "episode"))
        self.weak_scale = float(mix.get("weak_scale", 0.2))
        self.weak_noise_std = float(mix.get("weak_noise_std", 0.0))



        fuel = cfg.get("fuel", {})
        self.use_fuel = bool(fuel.get("enable", False))

        if self.use_fuel:
            fdef = fuel.get("def", {})
            fatt = fuel.get("att", {})

            self.g0 = 9.80665  # m/s^2

            self.m0_def   = float(fdef.get("m0", 1.0))
            self.mdry_def = float(fdef.get("m_dry", 1.0))
            self.Tmax_def = float(fdef.get("Tmax", 0.0))
            self.Isp_def  = float(fdef.get("Isp", 0.0))

            self.m0_att   = float(fatt.get("m0", 1.0))
            self.mdry_att = float(fatt.get("m_dry", 1.0))
            self.Tmax_att = float(fatt.get("Tmax", 0.0))
            self.Isp_att  = float(fatt.get("Isp", 0.0))

            if self.m0_def < self.mdry_def:
                raise ValueError("fuel.def: m0 must be >= m_dry")
            if self.m0_att < self.mdry_att:
                raise ValueError("fuel.att: m0 must be >= m_dry")
            if self.Tmax_def <= 0.0 or self.Isp_def <= 0.0:
                raise ValueError("fuel.def: Tmax and Isp must be > 0")
            if self.Tmax_att <= 0.0 or self.Isp_att <= 0.0:
                raise ValueError("fuel.att: Tmax and Isp must be > 0")

    def set_opp_domain(self, mode: str):
        self.opp_domain = str(mode)

    def _belief_state(self, p2_true: np.ndarray, v2_true: np.ndarray):
        if (not self.use_ukf) or (self.ukf is None):
            return p2_true, v2_true

        p2_bel = self.ukf.x[:self.D].copy()
        v2_bel = self.ukf.x[self.D:2*self.D].copy()

        r_est = np.linalg.norm(p2_bel - self.center)
        r_max = self.belief_clip_factor * self.radius

        if (not np.isfinite(r_est)) or (r_est > r_max):
            if np.isfinite(r_est) and r_est > 1e-9:
                direction = (p2_bel - self.center) / r_est
                p2_bel = self.center + direction * r_max
            else:
                p2_bel = p2_true.copy()
                v2_bel = v2_true.copy()

            if self.reset_ukf_on_diverge:
                self.ukf.x[:self.D] = p2_true
                self.ukf.x[self.D:2*self.D] = v2_true
                self.ukf.P = self._ukf_P0.copy()
                p2_bel = p2_true.copy()
                v2_bel = v2_true.copy()

        return p2_bel, v2_bel

    def _select_reward_attacker(self, pA_list, vA_list):
        center_dists = np.asarray([np.linalg.norm(pA - self.center) for pA in pA_list], dtype=float)
        if center_dists.size == 0:
            raise RuntimeError("1v2 Env expected at least one attacker.")
        idx = int(np.argmin(center_dists))
        return idx, pA_list[idx], vA_list[idx], center_dists

    def _split_team_attacker_reward(self, team_reward: float):
        share = float(team_reward) / float(max(1, self.num_attackers))
        return np.full((self.num_attackers,), share, dtype=np.float32)

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
                r = (r_min**3 + (r_max**3 - r_min**3) * u) ** (1.0 / self.D)
                return self.center + r * d

            # r_def_min, r_def_max = 0.0, 0.5 * R
            # r_att_min, r_att_max = 0.4 * R, 0.95 * R

            r_def_min = float(self.cfg.get("r_def_min")) * R  
            r_def_max = float(self.cfg.get("r_def_max")) * R

            r_att_min = float(self.cfg.get("r_att_min")) * R
            r_att_max = float(self.cfg.get("r_att_max")) * R 


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

        elif mode == "random_shell_advantage":
            R = self.radius
            v_max = self.train_ic_vmax
            min_sep = self.train_min_sep

            def sample_in_shell(r_min, r_max):
                if r_max < r_min:
                    raise ValueError(f"Invalid shell: r_min={r_min} > r_max={r_max}")

                d = np.random.normal(size=(self.D,))
                d /= (np.linalg.norm(d) + 1e-9)

                # correct for generic dimension D
                u = np.random.rand()
                r = (r_min**self.D + (r_max**self.D - r_min**self.D) * u) ** (1.0 / self.D)
                return self.center + r * d

            # Recommended: margin in METERS, based on OI radius.
            oi_r = float(self.cfg.get("oi", {}).get("r"))

            percent_advantage_defender = self.cfg.get("percent_advantage_defender", 0.75)
            radial_margin = float(percent_advantage_defender*np.pi*2*oi_r) 

            r_def_min = float(self.cfg.get("r_def_min")) * R
            r_def_max = float(self.cfg.get("r_def_max")) * R

            r_att_min = float(self.cfg.get("r_att_min")) * R
            r_att_max = float(self.cfg.get("r_att_max")) * R

            r_att_min = max(r_att_min, radial_margin)

            if radial_margin < 0.0:
                raise ValueError(f"percent_advantage_defender must be >= 0, got {radial_margin}")

            if r_def_min > r_def_max:
                raise ValueError(f"Invalid defender shell: [{r_def_min}, {r_def_max}]")
            if r_att_min > r_att_max:
                raise ValueError(f"Invalid attacker shell: [{r_att_min}, {r_att_max}]")

            # Quick feasibility sanity check
            if r_def_min > (r_att_max - radial_margin):
                raise ValueError(
                    "Infeasible radial shells: defender cannot be at least "
                    f"{radial_margin:.3f} m closer to center than attacker. "
                    f"Got r_def_min={r_def_min:.3f}, r_att_max={r_att_max:.3f}."
                )

            placed = False

            # Outer loop: resample whole scene until all constraints are satisfied
            for _scene_try in range(2000):
                # -------------------------------------------------
                # 1) Sample all attackers first
                # -------------------------------------------------
                pA = []
                vA = []
                rA = []

                attackers_ok = True
                for k in range(Na):
                    found_att = False
                    for _ in range(1000):
                        pk = sample_in_shell(r_att_min, r_att_max)
                        rk = np.linalg.norm(pk - self.center)

                        # enforce attacker-attacker min separation
                        if any(np.linalg.norm(pk - pj) < min_sep for pj in pA):
                            continue

                        pA.append(pk)
                        rA.append(rk)
                        vA.append(np.random.uniform(-v_max, v_max, size=(self.D,)))
                        found_att = True
                        break

                    if not found_att:
                        attackers_ok = False
                        break

                if not attackers_ok:
                    continue

                # -------------------------------------------------
                # 2) Defender must be closer than EVERY attacker
                #    by at least radial_margin
                # -------------------------------------------------
                r_att_nearest = min(rA)
                r_def_max_eff = min(r_def_max, r_att_nearest - radial_margin)

                if r_def_max_eff < r_def_min:
                    continue

                found_def = False
                for _ in range(1000):
                    p1 = sample_in_shell(r_def_min, r_def_max_eff)

                    # enforce defender-attacker Euclidean min separation too
                    if any(np.linalg.norm(p1 - pk) < min_sep for pk in pA):
                        continue

                    found_def = True
                    break

                if not found_def:
                    continue

                v1 = np.random.uniform(-v_max, v_max, size=(self.D,))
                placed = True
                break

            if not placed:
                raise RuntimeError(
                    "random_shell: could not sample a feasible initial condition after many attempts. "
                    "Try relaxing r_def_max, increasing r_att_min/r_att_max, reducing min_sep, "
                    "or reducing percent_advantage_defender."
                )

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
        self._record_ic_scene(x0)
        self.state = x0.reshape(-1)

        # With multi-attacker _unpack, choose the current most threatening attacker
        # for the scalar team-level zero-sum reward state.
        p1, v1, pA_list, vA_list = self._unpack(self.state)
        threat_idx, p2, v2, att_center_dists = self._select_reward_attacker(pA_list, vA_list)

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

        # ---- initialize d2_prev using the same geometry mode as step rewards ----
        if self.reward_from_belief and self.use_ukf and (self.ukf is not None):
            p2_geom, _ = self._belief_state(p2, v2)
        else:
            p2_geom = p2

        d2_raw = float(np.dot(p2_geom - self.center, p2_geom - self.center))
        self._d2_prev = d2_raw / (self.radius**2)

        if self.use_fuel:
            self.m_def = self.m0_def
            self.m_att = np.full((self.num_attackers,), self.m0_att, dtype=float)

        return self._obs()


    def step(self, a1_env: np.ndarray, aA_env: np.ndarray, reward_mode: str = "both"):
        """
        a1_env: (D,)
        aA_env: (Na, D) or (D,) if Na=1
        reward_mode: "def", "att", or "both"
        """
        need_def = reward_mode in ("def", "both")
        need_att = reward_mode in ("att", "both")

        # Defender commanded action
        a1_cmd = np.asarray(a1_env, float).reshape(self.D,)

        if self.opp_domain == "none":
            a1_cmd[:] = 0.0
        elif self.opp_domain == "weak":
            a1_cmd = self.weak_scale * a1_cmd
            if self.weak_noise_std > 0:
                a1_cmd = a1_cmd + np.random.normal(0.0, self.weak_noise_std, size=(self.D,))

        a1_cmd = np.clip(a1_cmd, self.u_lo, self.u_hi)

        # Attacker commanded actions
        aA_cmd = np.asarray(aA_env, float)
        if aA_cmd.ndim == 1:
            aA_cmd = aA_cmd.reshape(1, self.D)
        else:
            aA_cmd = aA_cmd.reshape(self.num_attackers, self.D)
        aA_cmd = np.clip(aA_cmd, self.u_lo, self.u_hi)

        # -------------------------------------------------
        # Convert commanded actions into realized dynamics
        # -------------------------------------------------
        fuel_depleted_def = False
        fuel_depleted_att = False

        if self.use_fuel:
            a1_real, self.m_def, fuel_depleted_def, thrust_def, mdot_def = self._apply_propulsion(
                a_cmd=a1_cmd,
                m=self.m_def,
                m_dry=self.mdry_def,
                Tmax=self.Tmax_def,
                Isp=self.Isp_def,
            )

            aA_real = np.zeros_like(aA_cmd, dtype=float)
            thrust_att_all = np.zeros((self.num_attackers,), dtype=float)
            mdot_att_all = np.zeros((self.num_attackers,), dtype=float)

            for k in range(self.num_attackers):
                aA_real[k], self.m_att[k], fuel_k, thrust_k, mdot_k = self._apply_propulsion(
                    a_cmd=aA_cmd[k],
                    m=self.m_att[k],
                    m_dry=self.mdry_att,
                    Tmax=self.Tmax_att,
                    Isp=self.Isp_att,
                )
                thrust_att_all[k] = thrust_k
                mdot_att_all[k] = mdot_k
                fuel_depleted_att = fuel_depleted_att or fuel_k
        else:
            a1_real = a1_cmd
            aA_real = aA_cmd
            thrust_def = 0.0
            mdot_def = 0.0
            thrust_att_all = np.zeros((self.num_attackers,), dtype=float)
            mdot_att_all = np.zeros((self.num_attackers,), dtype=float)

        # propagate true state using REALIZED accelerations
        self.state = self._plant_step(self.state, a1_real, aA_real)
        self.t += 1

        # Unpack new state and choose the current most threatening attacker for
        # the scalar team-level zero-sum reward state.
        p1, v1, pA_list, vA_list = self._unpack(self.state)
        threat_idx, p2, v2, att_center_dists = self._select_reward_attacker(pA_list, vA_list)
        a2_cmd = aA_cmd[threat_idx]
        a2_real = aA_real[threat_idx]

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

        if self.reward_from_belief and self.use_ukf and (self.ukf is not None):
            p2_reward, v2_reward = self._belief_state(p2, v2)
        else:
            p2_reward, v2_reward = p2, v2

        # ---- shared geometry needed by whichever reward(s) we compute ----
        # d2 and rel2 are used by both rewards
        d2_raw = float(np.dot(p2_reward - self.center, p2_reward - self.center))
        d2 = d2_raw / (self.radius**2)
        delta_d2 = d2 - (self._d2_prev if self._d2_prev is not None else d2)

        rel2 = float(np.dot((p2_reward - p1), (p2_reward - p1))) / (self.radius**2)

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

        rho1 = np.linalg.norm(p1 - self.center)/ self.radius
        rho2 = np.linalg.norm(p2_reward - self.center) / self.radius

        v_scale = self.radius / self.dt

        #Defender vars
        #             
        # rho1_rel_to_center = np.linalg.norm(p1 - self.center)
        wall1 = ((max(0.0, rho1 - self.soft_wall))**2) * self.wallK

        # defender radial velocity
        rhat1 = (p1 - self.center)
        rnorm = np.linalg.norm(rhat1) + 1e-9

        v1n2 = float(np.dot(v1, v1)) / (v_scale**2)
        a1n2 = float(np.dot(a1_cmd, a1_cmd)) / (self.u_hi**2)

        vrad1 = float(np.dot(v1, rhat1 / rnorm)) / v_scale  # dimensionless

        #Attacker vars

        v2n2 = float(np.dot(v2_reward, v2_reward)) / (v_scale**2)
        a2n2 = float(np.dot(a2_cmd, a2_cmd)) / (self.u_hi**2)

        wall2 = ((max(0.0, rho2 - self.soft_wall))**2) * self.wallK

        # ---- termination scenariosalways uses TRUE state ----

        # 1) Hitting target
        hit_target = False

        rho_att = float(att_center_dists[threat_idx]) / self.radius
        rho_def = np.linalg.norm(p1 - self.center) / self.radius

        thresh_def = (1.0 + self.hit_buffer_def) * self.oi_radius_norm
        thresh_att = (1.0 + self.hit_buffer_att) * self.oi_radius_norm

        att_rho_all = att_center_dists / self.radius
        att_hit_target = (self.oi_radius_norm > 0.0) and bool(np.any(att_rho_all <= thresh_att))
        def_hit_target = (self.oi_radius_norm > 0.0) and (rho_def <= thresh_def)


        hit_target = att_hit_target or def_hit_target

        # 2) collision: defender within collision_radius_m of ANY attacker (TRUE distance)
        collision = False
        if self.collision_radius_m > 0.0:
            for pA_true in pA_list:
                if np.linalg.norm(pA_true - p1) <= self.collision_radius_m:
                    collision = True
                    break



        # 3) Exiting the arena

        # oob for ANY attacker
        oob1 = (rho1 >= self.margin)
        oob2_any = False
        for pA_true in pA_list:
            rhoA_true = np.linalg.norm(pA_true - self.center) / self.radius
            if rhoA_true >= self.margin:
                oob2_any = True
                break

        if self.opp_domain == "none":
            # do NOT let defender OOB / defender-hit / collisions terminate training
            oob1 = False
            def_hit_target = False
            collision = False
            hit_target = att_hit_target  # only attacker matters

        # 4) Fuel logic (optional)

        done = (oob1 or oob2_any or hit_target or collision)

        if self.use_fuel == True:
            done = done or fuel_depleted_att or fuel_depleted_def          



        # ---- compute only requested reward(s) ----
        r1 = 0.0
        r2 = 0.0

        use_security = False
        use_zero_sum = True


        #Visualize what strategies 
        # Off policy methods 

        if use_security:

            k_time = 0.001

            g = (
                #Both agents: TBoth terms
                self.k_pos * d2
                # + k_time
            )

            if self.use_fuel:
                # eff_def = thrust_def / (self.Tmax_def + 1e-9)
                # eff_att = thrust_att_all[0] / (self.Tmax_att + 1e-9)
                # g += - self.k_eff_def * eff_def + self.k_eff_att * eff_att
                burn_frac_def = (mdot_def * self.dt) / (self.m0_def - self.mdry_def + 1e-9)
                burn_frac_att = float(np.mean(mdot_att_all) * self.dt) / (self.m0_att - self.mdry_att + 1e-9)
                g += - self.k_eff_def * burn_frac_def + self.k_eff_att * burn_frac_att

            # terminal handling must also be zero-sum. Colissions make it symmetrical

            shared_termination = 0.0

            if done:
                # Example: encode hit_target / collision / oob into g so the sign flip is consistent
                # if collision:
                #     g += self.collision_penalty_def
                if att_hit_target or def_hit_target:
                    g -= self.target_hit_reward_penalty
                elif collision:
                    shared_termination += self.collision_penalty
                elif oob1:
                    g -= self.wall_penalty
                elif oob2_any:
                    g += self.wall_penalty

                elif self.use_fuel:
                    if fuel_depleted_def:
                        g -= self.fuel_depletion_penalty
                    if fuel_depleted_att:
                        g += self.fuel_depletion_penalty

                # If you want attacker “success” to matter, it should reduce defender g,
                # which automatically increases attacker reward via -g.


            # if done and collision:
            #     shared_termination += self.collision_penalty

            if need_def:
                r1 = g - shared_termination
            if need_att:
                r2 = -g - shared_termination

            # if done and oob1:
            #     if need_def: 
            #         r1 -= self.wallK

        elif use_zero_sum:

            dist_rel = np.linalg.norm(p2_reward - p1)
            dock_gap = max(0.0, dist_rel - self.collision_radius_m) / self.radius

            g = (
                self.k_pos * d2
                # - (self.k_dock) * dock_gap                
                # - (self.k_pos/1.5) * rel2
                # + k_time
            )

            # Control effort: defender pays for actuation directly in g, attacker sees the
            # sign-flipped version through r_att = -g.
            # g += - self.lD * a1n2 + self.lA * a2n2

            # if self.use_ukf and self.use_meas_reward:
            #     g -= (self.meas_innov_coef * meas_innov_sq) + (self.meas_cov_coef * meas_trPpos)

            if self.use_fuel:
                # eff_def = thrust_def / (self.Tmax_def + 1e-9)
                # eff_att = thrust_att_all[0] / (self.Tmax_att + 1e-9)
                # g += - self.k_eff_def * eff_def + self.k_eff_att * eff_att
                burn_frac_def = (mdot_def * self.dt) / (self.m0_def - self.mdry_def + 1e-9)
                burn_frac_att = float(np.mean(mdot_att_all) * self.dt) / (self.m0_att - self.mdry_att + 1e-9)
                g += - self.k_eff_def * burn_frac_def + self.k_eff_att * burn_frac_att


            if done:
                # Example: encode hit_target / collision / oob into g so the sign flip is consistent
                # if collision:
                #     g += self.collision_penalty_def
                if collision:
                    g += self.collision_penalty
                elif att_hit_target or def_hit_target:
                    g -= self.target_hit_reward_penalty
                elif oob1:
                    g -= self.wall_penalty
                elif oob2_any:
                    g += self.wall_penalty

                elif self.use_fuel:
                    if fuel_depleted_def:
                        g -= self.fuel_depletion_penalty
                    if fuel_depleted_att:
                        g += self.fuel_depletion_penalty

                # If you want attacker “success” to matter, it should reduce defender g,
                # which automatically increases attacker reward via -g.


            # if done and collision:
            #     shared_termination += self.collision_penalty

            if need_def:
                r1 = g 
            if need_att:
                r2 = -g


            

        else:
            raise("Failed")


        # track d2_prev based on the geometry used for reward (same as your current logic)
        self._d2_prev = d2

        # info: you can also gate what you store here if you want
        # ---- always compute these for logging ----
        d1_true_norm = float(np.dot(p1 - self.center, p1 - self.center)) / (self.radius**2)
        d2_true_norm = float(np.dot(p2 - self.center, p2 - self.center)) / (self.radius**2)
        d2_belief_norm = None
        if self.use_ukf and (self.ukf is not None):
            p2_belief, _ = self._belief_state(p2, v2)
            d2_belief_norm = float(np.dot(p2_belief - self.center, p2_belief - self.center)) / (self.radius**2)

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

            "collision": bool(collision),
            "threat_attacker_idx": int(threat_idx),
            "r_att_each": self._split_team_attacker_reward(r2).tolist(),

        }

        if self.use_ukf:
            info["meas_innov_sq"] = meas_innov_sq
            info["ukf_trPpos"] = meas_trPpos
            if d2_belief_norm is not None:
                info["d2_belief_norm"] = d2_belief_norm

        if self.use_fuel:
            fuel_frac_def = (self.m_def - self.mdry_def) / (self.m0_def - self.mdry_def + 1e-9)
            fuel_frac_att_all = (self.m_att - self.mdry_att) / (self.m0_att - self.mdry_att + 1e-9)
            fuel_frac_att_mean = float(np.mean(np.clip(fuel_frac_att_all, 0.0, 1.0)))

            info["fuel_frac_def"] = float(np.clip(fuel_frac_def, 0.0, 1.0))
            info["fuel_frac_att"] = fuel_frac_att_mean
            info["fuel_used_def"] = 1.0 - info["fuel_frac_def"]
            info["fuel_used_att"] = 1.0 - info["fuel_frac_att"]

            # optional, nice to have
            info["thrust_def"] = float(thrust_def)
            info["thrust_att"] = float(np.mean(thrust_att_all))
            info["mdot_def"] = float(mdot_def)
            info["mdot_att"] = float(np.mean(mdot_att_all))

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

        # Defender and attacker fuel:

        if self.use_fuel:
            fuel_frac_def = (self.m_def - self.mdry_def) / (self.m0_def - self.mdry_def + 1e-9)
            fuel_frac_att = float(np.mean((self.m_att - self.mdry_att) / (self.m0_att - self.mdry_att + 1e-9)))

            parts.append(np.array([np.clip(fuel_frac_def, 0.0, 1.0)], dtype=np.float32))
            parts.append(np.array([np.clip(fuel_frac_att, 0.0, 1.0)], dtype=np.float32))

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

        if self.opp_domain == "none":
            # freeze defender in place for "no defender" domain
            x1n = x1
        else:
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
    
    def _apply_propulsion(
        self,
        a_cmd: np.ndarray,
        m: float,
        m_dry: float,
        Tmax: float,
        Isp: float,
    ):
        """
        Convert commanded acceleration into realized acceleration using
        thrust saturation and propellant consumption.

        a_cmd : commanded acceleration [m/s^2]
        m     : current mass [kg]
        Tmax  : max thrust magnitude [N]
        Isp   : specific impulse [s]
        """
        a_cmd = np.asarray(a_cmd, dtype=float)

        if m <= m_dry + 1e-9:
            return np.zeros_like(a_cmd), m_dry, True, 0.0, 0.0

        # Required thrust to realize the commanded acceleration
        F_req = m * a_cmd
        F_req_norm = np.linalg.norm(F_req)

        # Saturate by max available thrust
        if F_req_norm > Tmax:
            F = F_req * (Tmax / (F_req_norm + 1e-9))
        else:
            F = F_req

        # Realized acceleration
        a_real = F / max(m, 1e-9)

        # Rocket equation mass flow
        thrust_norm = np.linalg.norm(F)
        mdot = thrust_norm / (Isp * self.g0)   # kg/s
        m_next = max(m_dry, m - mdot * self.dt)
        fuel_depleted = (m_next <= m_dry + 1e-9)

        return a_real, m_next, fuel_depleted, thrust_norm, mdot
    
    def _record_ic_scene(self, x0: np.ndarray):
        """
        Record initial positions actually used at reset().
        Stores positions only, not velocities.
        """
        if not self.record_ic_history:
            return

        x0 = np.asarray(x0, dtype=np.float32)

        # defender
        self.ic_history_def.append(x0[0, 0:self.D].copy())

        # attackers
        for k in range(self.num_attackers):
            idx = 1 + k
            self.ic_history_att.append(x0[idx, 0:self.D].copy())

        # keep memory bounded
        if len(self.ic_history_def) > self.max_ic_history:
            extra = len(self.ic_history_def) - self.max_ic_history
            self.ic_history_def = self.ic_history_def[extra:]

        if len(self.ic_history_att) > self.max_ic_history * self.num_attackers:
            extra = len(self.ic_history_att) - self.max_ic_history * self.num_attackers
            self.ic_history_att = self.ic_history_att[extra:]


    def get_ic_history_arrays(self):
        """
        Returns
        -------
        def_pos : (Ndef, D)
        att_pos : (Natt, D)
        """
        if len(self.ic_history_def) == 0:
            def_pos = np.zeros((0, self.D), dtype=np.float32)
        else:
            def_pos = np.stack(self.ic_history_def, axis=0).astype(np.float32)

        if len(self.ic_history_att) == 0:
            att_pos = np.zeros((0, self.D), dtype=np.float32)
        else:
            att_pos = np.stack(self.ic_history_att, axis=0).astype(np.float32)

        return def_pos, att_pos



# =============================================================
# Vectorized env (single-process)
# =============================================================
class VecEnv:
    def __init__(self, make_env: Callable[[], Env], num_envs: int):
        self.envs: List[Env] = [make_env() for _ in range(num_envs)]
        self.num_envs = num_envs


        # NEW: pick opponent domain per env (only matters in attacker-training)
        for e in self.envs:
            e.set_opp_domain(_sample_opp_domain(e.cfg))

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
                # NEW: resample opponent domain at episode boundary
                if e.opp_resample == "episode":
                    e.set_opp_domain(_sample_opp_domain(e.cfg))
                o = e.reset()
            obs_next.append(o)
            r1.append(R1); r2.append(R2); done.append(d); info.append(inf)
        self.obs = np.stack(obs_next, axis=0)
        return self.obs, np.array(r1), np.array(r2), np.array(done, dtype=np.float32), info


def collect_ic_history_from_vecenv(vec: VecEnv):
    """
    Aggregate initial-condition history from all sub-envs.

    Returns
    -------
    def_pos : (N, D)
    att_pos : (M, D)
    """
    def_all = []
    att_all = []

    for e in vec.envs:
        d, a = e.get_ic_history_arrays()
        if d.shape[0] > 0:
            def_all.append(d)
        if a.shape[0] > 0:
            att_all.append(a)

    D = vec.envs[0].D
    def_pos = np.concatenate(def_all, axis=0) if len(def_all) > 0 else np.zeros((0, D), dtype=np.float32)
    att_pos = np.concatenate(att_all, axis=0) if len(att_all) > 0 else np.zeros((0, D), dtype=np.float32)

    return def_pos, att_pos


def sample_ic_support(
    cfg: Dict[str, Any],
    n_scenes: int = 20000,
    seed: int = 123,
):
    """
    Approximate the feasible initial-condition support by repeatedly
    calling Env.reset() on a fresh env.

    Returns
    -------
    def_pos : (n_scenes, D)
    att_pos : (n_scenes * num_attackers, D)
    """
    cfg_probe = copy.deepcopy(cfg)
    cfg_probe["record_ic_history"] = False  # do not accumulate in the probe env

    build_dyn(cfg_probe)
    set_seed(seed)

    env = Env(cfg_probe)

    def_all = []
    att_all = []

    for _ in range(n_scenes):
        env.reset()
        p1, v1, pA_list, vA_list = env._unpack(env.state)

        def_all.append(np.asarray(p1, dtype=np.float32).copy())
        for pA in pA_list:
            att_all.append(np.asarray(pA, dtype=np.float32).copy())

    def_pos = np.stack(def_all, axis=0).astype(np.float32)
    att_pos = np.stack(att_all, axis=0).astype(np.float32)

    return def_pos, att_pos


def _sample_opp_domain(cfg: Dict[str, Any]) -> str:
    mix = cfg.get("opp_mix", None)
    if not mix:
        return "def0"
    modes = list(mix.get("modes", ["def0"]))
    probs = mix.get("probs")
    if probs is None:
        # uniform
        return np.random.choice(modes)
    probs = np.asarray(probs, float)

    total = probs.sum()
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(
            f"opp_mix['probs'] must sum to 1.0, got sum={total:.12f} and probs={probs.tolist()}"
        )

    probs = probs / (probs.sum() + 1e-12)
    return str(np.random.choice(modes, p=probs))


__all__ = ["Env", "VecEnv", "collect_ic_history_from_vecenv", "sample_ic_support"]
