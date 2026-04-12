from typing import Any, Callable, Dict, List

import copy
import multiprocessing as mp
import numpy as np
import torch
import os

from ukf_estimator import KF_CV, _body_bearing_from_world, _azel_from_body_vec
from config_rl import build_dyn
from core.utils import set_seed

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
        self.normalize_pos_obs = bool(cfg.get("normalize_pos_obs", False))
        self._pos_obs_scale = (1.0 / max(self.radius, 1e-9)) if self.normalize_pos_obs else 1.0
        self._vel_obs_scale = (self.dt / max(self.radius, 1e-9)) if self.normalize_pos_obs else 1.0


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
        reward_type = str(
            cfg.get(
                "reward_type",
                "zero_sum_kf" if bool(cfg.get("use_zero_sum_kf", False)) else "zero_sum",
            )
        ).strip().lower()
        if reward_type not in {"zero_sum", "zero_sum_kf"}:
            raise ValueError(
                f"Unsupported reward_type='{reward_type}'. Expected 'zero_sum' or 'zero_sum_kf'."
            )
        self.reward_type = reward_type
        self.use_zero_sum_kf = (self.reward_type == "zero_sum_kf")

        self.use_kf           = bool(cfg.get("use_kf", False))
        self.meas_innov_coef  = float(cfg.get("meas_innov_coef"))  # weight on innovation^2
        self.meas_cov_coef    = float(cfg.get("meas_cov_coef"))    # weight on trace(P_pos)
        # `use_kf` is the estimator toggle.
        self.belief_clip_factor = float(cfg.get("belief_clip_factor", 2.0))
        self.reset_ukf_on_diverge = bool(cfg.get("reset_ukf_on_diverge", True))
        self._estimator_cfg = dict(cfg.get("ukf", {}))
        self.estimator_kind = self._normalize_estimator_kind(cfg.get("estimator_kind", "ukf"))
        self.estimator_label = self.estimator_kind.upper()
        self._estimator_dyn = None
        self._ekf_jacobian_mode = str(self._estimator_cfg.get("ekf_jacobian_mode", "exact"))

        #Fuel config
        fuel = cfg.get("fuel", {})
        self.use_fuel = bool(fuel.get("enable"))

        if self.use_kf and self.D != 3:
            raise ValueError("Bearing-only estimator currently implemented for D=3 only.")

        self.ukf = None
        self.att_ukf = None
        self._latest_meas_innov = 0.0
        self._latest_meas_trP   = 0.0
        self._latest_att_meas_innov = 0.0
        self._latest_att_meas_trP   = 0.0
        self._ukf_action_access = "ground_truth"
        self._kf_control_meas_std = np.zeros(3, dtype=float)
        self._ukf_state_dim = 6

        self.record_ic_history = bool(cfg.get("record_ic_history", False))
        self.max_ic_history = int(cfg.get("max_ic_history", 200000))

        self.ic_history_def = []
        self.ic_history_att = []

        if self.use_kf:
            ukf_cfg = self._estimator_cfg
            self._ukf_every = max(1, int(ukf_cfg.get("every", 1)))
            self._estimator_dyn = self._resolve_estimator_dynamics()

            # Initial covariance P0
            pos_std0 = float(ukf_cfg.get("pos_std0", 0.2 * self.radius))
            vel_std0 = float(ukf_cfg.get("vel_std0", 0.01))
            q_scale = float(ukf_cfg.get("Q_scale", 1e-5))

            # Measurement noise R (az, el)
            sigma_az = float(ukf_cfg.get("sigma_az", np.deg2rad(0.5)))
            sigma_el = float(ukf_cfg.get("sigma_el", np.deg2rad(0.5)))
            Rm = np.diag([sigma_az**2, sigma_el**2])

            # Legacy init stds are kept for compatibility, but the UKF mean
            # now uses separate init_mean_* knobs so P0 and mean perturbation
            # are not conflated.
            self._ukf_init_pos_std = float(ukf_cfg.get("init_pos_std", pos_std0))
            self._ukf_init_vel_std = float(ukf_cfg.get("init_vel_std", vel_std0))
            self._ukf_init_mean_pos_std = float(ukf_cfg.get("init_mean_pos_std", 0.0))
            self._ukf_init_mean_vel_std = float(ukf_cfg.get("init_mean_vel_std", 0.0))
            self._ukf_action_access = self._normalize_kf_action_access(
                ukf_cfg.get("action_access", "ground_truth")
            )
            self._kf_control_meas_std = self._normalize_kf_control_noise_std(
                ukf_cfg.get("action_meas_std", 0.1 * max(abs(float(cfg.get("umax", 1.0))), 1e-3))
            )
            self._ukf_state_dim = 6
            accel_std0 = float(ukf_cfg.get("accel_std0", max(abs(float(cfg.get("umax", 1.0))), 1e-3)))
            accel_q_scale = float(ukf_cfg.get("accel_Q_scale", q_scale))
            self._ukf_init_mean_accel_std = float(ukf_cfg.get("init_mean_accel_std", 0.0))

            if self._ukf_state_dim == 9:
                P0 = np.diag(
                    [pos_std0**2, pos_std0**2, pos_std0**2,
                     vel_std0**2, vel_std0**2, vel_std0**2,
                     accel_std0**2, accel_std0**2, accel_std0**2]
                )
                Q = np.diag(
                    [q_scale, q_scale, q_scale,
                     q_scale, q_scale, q_scale,
                     accel_q_scale, accel_q_scale, accel_q_scale]
                )
            else:
                P0 = np.diag(
                    [pos_std0**2, pos_std0**2, pos_std0**2,
                     vel_std0**2, vel_std0**2, vel_std0**2]
                )
                Q = q_scale * np.eye(6, dtype=float)

            self._ukf_P0 = P0
            self._ukf_Q  = Q
            self._ukf_R  = Rm

        if self.use_zero_sum_kf and not self.use_kf:
            raise ValueError("reward_type='zero_sum_kf' requires the estimator to be enabled.")

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
        if (not self.use_kf) or (self.ukf is None):
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
                if self.ukf.x.size > 2 * self.D:
                    self.ukf.x[2*self.D:] = 0.0
                self.ukf.P = self._ukf_P0.copy()
                p2_bel = p2_true.copy()
                v2_bel = v2_true.copy()

        return p2_bel, v2_bel

    def _attacker_belief_state(self, p1_true: np.ndarray, v1_true: np.ndarray):
        if (not self.use_kf) or (self.att_ukf is None):
            return p1_true, v1_true

        p1_bel = self.att_ukf.x[:self.D].copy()
        v1_bel = self.att_ukf.x[self.D:2*self.D].copy()

        r_est = np.linalg.norm(p1_bel - self.center)
        r_max = self.belief_clip_factor * self.radius

        if (not np.isfinite(r_est)) or (r_est > r_max):
            if np.isfinite(r_est) and r_est > 1e-9:
                direction = (p1_bel - self.center) / r_est
                p1_bel = self.center + direction * r_max
            else:
                p1_bel = p1_true.copy()
                v1_bel = v1_true.copy()

            if self.reset_ukf_on_diverge:
                self.att_ukf.x[:self.D] = p1_true
                self.att_ukf.x[self.D:2*self.D] = v1_true
                if self.att_ukf.x.size > 2 * self.D:
                    self.att_ukf.x[2*self.D:] = 0.0
                self.att_ukf.P = self._ukf_P0.copy()
                p1_bel = p1_true.copy()
                v1_bel = v1_true.copy()

        return p1_bel, v1_bel

    def _center_belief(self) -> np.ndarray:
        return self.center.copy()

    def _normalize_kf_action_access(self, mode: Any) -> str:
        key = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
        if key == "groundtruth":
            key = "ground_truth"
        if key == "inferred":
            key = "measured"
        valid = {"ground_truth", "measured", "none"}
        if key not in valid:
            raise ValueError(f"Unsupported estimator action_access='{mode}'. Expected one of {sorted(valid)}.")
        return key

    def _normalize_kf_control_noise_std(self, noise_std: Any) -> np.ndarray:
        arr = np.asarray(noise_std, dtype=float)
        if arr.ndim == 0:
            std = np.full(3, float(arr), dtype=float)
        else:
            arr = arr.reshape(-1)
            if arr.size != 3:
                raise ValueError(
                    f"Expected estimator action_meas_std to be a scalar or length-3 sequence, got shape {arr.shape}."
                )
            std = arr.astype(float)
        return np.maximum(std, 0.0)

    def _normalize_estimator_kind(self, kind: Any) -> str:
        key = str(kind).strip().lower()
        valid = {"ukf", "ekf"}
        if key not in valid:
            raise ValueError(f"Unsupported estimator_kind='{kind}'. Expected one of {sorted(valid)}.")
        return key

    def _resolve_estimator_dynamics(self) -> str:
        dyn_name = str(self.cfg.get("dynamics", "hcw")).strip().lower()
        if dyn_name != "hcw":
            raise ValueError(
                f"{self.estimator_label} estimator currently requires cfg['dynamics']='hcw'; got '{dyn_name}'."
            )
        return "hcw"

    def _make_estimator(self, x0_est: np.ndarray, linearization_group: str = "default"):
        estimator_kwargs = {}
        if self.estimator_kind == "ekf":
            estimator_kwargs["jacobian_mode"] = self._ekf_jacobian_mode
            estimator_kwargs["linearization_group"] = linearization_group
        return KF_CV(
            x0=x0_est,
            P0=self._ukf_P0.copy(),
            Q=self._ukf_Q.copy(),
            R=self._ukf_R.copy(),
            dt=self.dt,
            kind=self.estimator_kind,
            dyn=self._estimator_dyn,
            hcw=self.cfg.get("hcw", {}),
            **estimator_kwargs,
        )

    def _kf_predict_control(self, ground_truth_u: np.ndarray | None):
        if ground_truth_u is None:
            return None, None
        u_true = np.asarray(ground_truth_u, dtype=float).reshape(3)
        if self._ukf_action_access == "ground_truth":
            return u_true, None
        if self._ukf_action_access == "measured":
            noise = np.random.normal(loc=0.0, scale=self._kf_control_meas_std, size=3)
            u_meas = u_true + noise
            umax = float(self.cfg.get("umax", np.inf))
            if np.isfinite(umax):
                u_meas = np.clip(u_meas, -umax, umax)
            return u_meas, np.diag(self._kf_control_meas_std**2)
        return None, None

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

        # With multi-attacker _unpack, we get lists:
        p1, v1, pA_list, vA_list = self._unpack(self.state)
        # For UKF and rewards we still only use the *first* attacker
        p2 = pA_list[0]
        v2 = vA_list[0]

        # ---- estimator init (only supported for 1 attacker right now) ----
        if self.use_kf and self.num_attackers != 1:
            raise NotImplementedError(f"{self.estimator_label} currently only implemented for num_attackers=1")

        if self.use_kf:
            pos_noise = np.random.normal(
                scale=self._ukf_init_mean_pos_std,
                size=p2.shape
            )
            vel_noise = np.random.normal(
                scale=self._ukf_init_mean_vel_std,
                size=v2.shape
            )
            p2_est = p2 + pos_noise
            v2_est = v2 + vel_noise
            if self._ukf_state_dim == 9:
                accel_noise = np.random.normal(
                    scale=self._ukf_init_mean_accel_std,
                    size=(3,),
                )
                x0_ukf = np.concatenate([p2_est, v2_est, accel_noise])
            else:
                x0_ukf = np.concatenate([p2_est, v2_est])

            self.ukf = self._make_estimator(x0_ukf, linearization_group="defender_observer")

            p1_pos_noise = np.random.normal(
                scale=self._ukf_init_mean_pos_std,
                size=p1.shape
            )
            p1_vel_noise = np.random.normal(
                scale=self._ukf_init_mean_vel_std,
                size=v1.shape
            )
            if self._ukf_state_dim == 9:
                p1_accel_noise = np.random.normal(
                    scale=self._ukf_init_mean_accel_std,
                    size=(3,),
                )
                x0_att_ukf = np.concatenate([p1 + p1_pos_noise, v1 + p1_vel_noise, p1_accel_noise])
            else:
                x0_att_ukf = np.concatenate([p1 + p1_pos_noise, v1 + p1_vel_noise])
            self.att_ukf = self._make_estimator(x0_att_ukf, linearization_group="attacker_observer")

            self._latest_meas_innov = 0.0
            self._latest_meas_trP   = float(np.trace(self.ukf.P[0:3, 0:3]))
            self._latest_att_meas_innov = 0.0
            self._latest_att_meas_trP = float(np.trace(self.att_ukf.P[0:3, 0:3]))

        else:
            self.ukf = None
            self.att_ukf = None
            self._latest_meas_innov = 0.0
            self._latest_meas_trP   = 0.0
            self._latest_att_meas_innov = 0.0
            self._latest_att_meas_trP = 0.0

        # ---- initialize d2_prev using the same geometry mode as step rewards ----
        if self.use_kf and (self.ukf is not None):
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

        # primary attacker convenience
        a2_cmd = aA_cmd[0]
        a2_real = aA_real[0]

        # propagate true state using REALIZED accelerations
        self.state = self._plant_step(self.state, a1_real, aA_real)
        self.t += 1

        # Unpack new state
        p1, v1, pA_list, vA_list = self._unpack(self.state)
        p2 = pA_list[0]
        v2 = vA_list[0]

        # ---- UKF (only if enabled; you can also gate this further if you want) ----
        meas_innov_sq = 0.0
        meas_trPpos   = 0.0
        att_meas_innov_sq = 0.0
        att_meas_trPpos = 0.0

        if self.use_kf:
            u12_predict, u12_cov = self._kf_predict_control(a2_real)
            self.ukf.predict(dt=self.dt, u=u12_predict, u_cov=u12_cov)
            if (self.t % self._ukf_every) == 0:
                p_obs = p1
                if self.D != 3:
                    raise RuntimeError("UKF bearing logic assumes D=3.")
                R_wb = np.eye(3)

                v_b = _body_bearing_from_world(p_obs, R_wb, p2)
                az_true, el_true = _azel_from_body_vec(v_b)
                z_true = np.array([az_true, el_true], float)

                z_noise = np.random.multivariate_normal(mean=np.zeros(2), cov=self._ukf_R)
                z_meas = z_true + z_noise

                z_hat_prior = (
                    self.ukf.measurement_prediction(p_obs, R_wb)
                    if hasattr(self.ukf, "measurement_prediction")
                    else self.ukf.h(self.ukf.x.copy(), p_obs, R_wb)
                )
                innov = z_meas - z_hat_prior
                innov[0] = (innov[0] + np.pi) % (2*np.pi) - np.pi
                innov[1] = (innov[1] + np.pi) % (2*np.pi) - np.pi
                meas_innov_sq = float(innov @ innov)

                self.ukf.update(z_meas, p_obs, R_wb)

            meas_trPpos = float(np.trace(self.ukf.P[0:3, 0:3]))
            self._latest_meas_innov = meas_innov_sq
            self._latest_meas_trP   = meas_trPpos

            if self.att_ukf is not None:
                u21_predict, u21_cov = self._kf_predict_control(a1_real)
                self.att_ukf.predict(dt=self.dt, u=u21_predict, u_cov=u21_cov)
                if (self.t % self._ukf_every) == 0:
                    p_obs = p2
                    if self.D != 3:
                        raise RuntimeError("UKF bearing logic assumes D=3.")
                    R_wb = np.eye(3)

                    v_b_att = _body_bearing_from_world(p_obs, R_wb, p1)
                    az_att_true, el_att_true = _azel_from_body_vec(v_b_att)
                    z_att_true = np.array([az_att_true, el_att_true], float)

                    z_att_noise = np.random.multivariate_normal(mean=np.zeros(2), cov=self._ukf_R)
                    z_att_meas = z_att_true + z_att_noise

                    z_att_hat_prior = (
                        self.att_ukf.measurement_prediction(p_obs, R_wb)
                        if hasattr(self.att_ukf, "measurement_prediction")
                        else self.att_ukf.h(self.att_ukf.x.copy(), p_obs, R_wb)
                    )
                    att_innov = z_att_meas - z_att_hat_prior
                    att_innov[0] = (att_innov[0] + np.pi) % (2*np.pi) - np.pi
                    att_innov[1] = (att_innov[1] + np.pi) % (2*np.pi) - np.pi
                    att_meas_innov_sq = float(att_innov @ att_innov)

                    self.att_ukf.update(z_att_meas, p_obs, R_wb)

                att_meas_trPpos = float(np.trace(self.att_ukf.P[0:3, 0:3]))
                self._latest_att_meas_innov = att_meas_innov_sq
                self._latest_att_meas_trP = att_meas_trPpos
            else:
                self._latest_att_meas_innov = 0.0
                self._latest_att_meas_trP = 0.0
        else:
            self._latest_meas_innov = 0.0
            self._latest_meas_trP   = 0.0
            self._latest_att_meas_innov = 0.0
            self._latest_att_meas_trP = 0.0

        use_zero_sum_kf = (self.reward_type == "zero_sum_kf")
        use_zero_sum = (self.reward_type == "zero_sum")

        # Keep the original game objective for the UKF zero-sum mode: the policy acts
        # from belief-space observations, but the task reward is still computed on truth.
        if use_zero_sum_kf:
            p2_reward, v2_reward = p2, v2
        else:
            p2_reward, v2_reward = p2, v2

        center_reward = self.center

        # ---- shared geometry needed by whichever reward(s) we compute ----
        # d2 and rel2 are used by both rewards
        d2_raw = float(np.dot(p2_reward - self.center, p2_reward - self.center))
        d2 = d2_raw / (self.radius**2)

        rel2 = float(np.dot((p2_reward - p1), (p2_reward - p1))) / (self.radius**2)

        # d1 only needed for defender reward (step/terminal), but cheap; compute only if needed
        if need_def:
            d1_raw = float(np.dot(p1 - self.center, p1 - self.center))
            d1 = d1_raw / (self.radius**2)
        else:
            d1 = 0.0  # placeholder

        # ---- wall penalties: compute only what you need ----

        rho1 = np.linalg.norm(p1 - self.center)/ self.radius
        rho2 = np.linalg.norm(p2_reward - self.center) / self.radius

        v_scale = self.radius / self.dt

        #Defender vars             

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

        rho_att = np.linalg.norm(p2 - self.center) / self.radius
        rho_def = np.linalg.norm(p1 - self.center) / self.radius

        thresh_def = (1.0 + self.hit_buffer_def) * self.oi_radius_norm
        thresh_att = (1.0 + self.hit_buffer_att) * self.oi_radius_norm


        att_hit_target = (self.oi_radius_norm > 0.0) and (rho_att <= thresh_att)
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

        #Visualize what strategies 
        # Off policy methods 

        if use_zero_sum:
            if self.use_kf:
                raise RuntimeError("use_zero_sum requires use_kf=False.")
            g = (
                self.k_pos * d2
            )

            # Control effort: defender pays for actuation directly in g, attacker sees the
            # sign-flipped version through r_att = -g.
            # g += - self.lD * a1n2 + self.lA * a2n2

            if self.use_fuel:
                # eff_def = thrust_def / (self.Tmax_def + 1e-9)
                # eff_att = thrust_att_all[0] / (self.Tmax_att + 1e-9)
                # g += - self.k_eff_def * eff_def + self.k_eff_att * eff_att
                burn_frac_def = (mdot_def * self.dt) / (self.m0_def - self.mdry_def + 1e-9)
                burn_frac_att = (mdot_att_all[0] * self.dt) / (self.m0_att - self.mdry_att + 1e-9)
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

        elif use_zero_sum_kf:
            if not self.use_kf:
                raise RuntimeError("use_zero_sum_kf requires use_kf=True.")
            g = (
                self.k_pos * d2
            )

            # Control effort: defender pays for actuation directly in g, attacker sees the
            # sign-flipped version through r_att = -g.
            # g += - self.lD * a1n2 + self.lA * a2n2

            if self.use_fuel:
                # eff_def = thrust_def / (self.Tmax_def + 1e-9)
                # eff_att = thrust_att_all[0] / (self.Tmax_att + 1e-9)
                # g += - self.k_eff_def * eff_def + self.k_eff_att * eff_att
                burn_frac_def = (mdot_def * self.dt) / (self.m0_def - self.mdry_def + 1e-9)
                burn_frac_att = (mdot_att_all[0] * self.dt) / (self.m0_att - self.mdry_att + 1e-9)
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
            raise RuntimeError("Unsupported reward configuration.")


        # track d2_prev based on the geometry used for reward (same as your current logic)
        self._d2_prev = d2

        # info: you can also gate what you store here if you want
        # ---- always compute these for logging ----
        d1_true_norm = float(np.dot(p1 - self.center, p1 - self.center)) / (self.radius**2)
        d2_true_norm = float(np.dot(p2 - self.center, p2 - self.center)) / (self.radius**2)
        d2_belief_norm = None
        p2_est_err_norm = None
        if self.use_kf and (self.ukf is not None):
            p2_belief, _ = self._belief_state(p2, v2)
            d2_belief_norm = float(np.dot(p2_belief - self.center, p2_belief - self.center)) / (self.radius**2)
            p2_est_err_norm = float(np.dot(p2_belief - p2, p2_belief - p2)) / (self.radius**2)
        d1_belief_norm = None
        p1_est_err_norm = None
        if self.use_kf and (self.att_ukf is not None):
            p1_belief, _ = self._attacker_belief_state(p1, v1)
            d1_belief_norm = float(np.dot(p1_belief - self.center, p1_belief - self.center)) / (self.radius**2)
            p1_est_err_norm = float(np.dot(p1_belief - p1, p1_belief - p1)) / (self.radius**2)
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

        }

        if self.use_kf:
            info["meas_innov_sq"] = meas_innov_sq
            info["ukf_trPpos"] = meas_trPpos
            info["att_meas_innov_sq"] = att_meas_innov_sq
            info["att_ukf_trPpos"] = att_meas_trPpos
            if d2_belief_norm is not None:
                info["d2_belief_norm"] = d2_belief_norm
            if p2_est_err_norm is not None:
                info["p2_est_err_norm"] = p2_est_err_norm
            if d1_belief_norm is not None:
                info["d1_belief_norm"] = d1_belief_norm
            if p1_est_err_norm is not None:
                info["p1_est_err_norm"] = p1_est_err_norm

        if self.use_fuel:
            fuel_frac_def = (self.m_def - self.mdry_def) / (self.m0_def - self.mdry_def + 1e-9)
            fuel_frac_att = (self.m_att[0] - self.mdry_att) / (self.m0_att - self.mdry_att + 1e-9)

            info["fuel_frac_def"] = float(np.clip(fuel_frac_def, 0.0, 1.0))
            info["fuel_frac_att"] = float(np.clip(fuel_frac_att, 0.0, 1.0))
            info["fuel_used_def"] = 1.0 - info["fuel_frac_def"]
            info["fuel_used_att"] = 1.0 - info["fuel_frac_att"]

            # optional, nice to have
            info["thrust_def"] = float(thrust_def)
            info["thrust_att"] = float(thrust_att_all[0])
            info["mdot_def"] = float(mdot_def)
            info["mdot_att"] = float(mdot_att_all[0])

        return self._obs(), float(r1), float(r2), bool(done), info




    def _obs_def(self) -> np.ndarray:
        p1, v1, pA_list, vA_list = self._unpack(self.state)

        if self.use_kf and (self.ukf is not None) and self.num_attackers == 1:
            p2_obs = self.ukf.x[:self.D]
            v2_obs = self.ukf.x[self.D:2*self.D]
            pA_obs = [p2_obs]
            vA_obs = [v2_obs]
        else:
            pA_obs = pA_list
            vA_obs = vA_list

        center_obs = self.center.copy()
        pos_scale = self._pos_obs_scale
        vel_scale = self._vel_obs_scale

        # build obs = [p1c, pA1c, ..., pANc, rel1, ..., relN, v1, vA1, ..., vAN]
        p1c = (p1 - center_obs) * pos_scale
        parts = [p1c]

        # positions (centered)
        for pA in pA_obs:
            parts.append((pA - center_obs) * pos_scale)

        # relative positions
        for pA in pA_obs:
            parts.append((pA - p1) * pos_scale)

        # defender vel
        parts.append(v1 * vel_scale)

        # attacker vels
        for vA in vA_obs:
            parts.append(vA * vel_scale)

        # Defender and attacker fuel:

        if self.use_fuel:
            fuel_frac_def = (self.m_def - self.mdry_def) / (self.m0_def - self.mdry_def + 1e-9)
            fuel_frac_att = (self.m_att[0] - self.mdry_att) / (self.m0_att - self.mdry_att + 1e-9)

            parts.append(np.array([np.clip(fuel_frac_def, 0.0, 1.0)], dtype=np.float32))
            parts.append(np.array([np.clip(fuel_frac_att, 0.0, 1.0)], dtype=np.float32))

        obs = np.concatenate(parts).astype(np.float32)
        return obs

    def _obs_att(self) -> np.ndarray:
        p1, v1, pA_list, vA_list = self._unpack(self.state)
        p2 = pA_list[0]
        v2 = vA_list[0]

        if self.use_kf and (self.att_ukf is not None) and self.num_attackers == 1:
            p1_obs, v1_obs = self._attacker_belief_state(p1, v1)
        else:
            p1_obs, v1_obs = p1, v1

        center_obs = self.center.copy()
        pos_scale = self._pos_obs_scale
        vel_scale = self._vel_obs_scale
        p1c = (p1_obs - center_obs) * pos_scale
        parts = [p1c, (p2 - center_obs) * pos_scale, (p2 - p1_obs) * pos_scale, v1_obs * vel_scale, v2 * vel_scale]

        if self.use_fuel:
            fuel_frac_def = (self.m_def - self.mdry_def) / (self.m0_def - self.mdry_def + 1e-9)
            fuel_frac_att = (self.m_att[0] - self.mdry_att) / (self.m0_att - self.mdry_att + 1e-9)
            parts.append(np.array([np.clip(fuel_frac_def, 0.0, 1.0)], dtype=np.float32))
            parts.append(np.array([np.clip(fuel_frac_att, 0.0, 1.0)], dtype=np.float32))

        return np.concatenate(parts).astype(np.float32)

    def get_obs_pair(self):
        return self._obs_def(), self._obs_att()

    def _obs(self) -> np.ndarray:
        return self._obs_def()



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
    torch_backend = False

    def __init__(self, make_env: Callable[[], Env], num_envs: int):
        self.envs: List[Env] = [make_env() for _ in range(num_envs)]
        self.num_envs = num_envs


        # NEW: pick opponent domain per env (only matters in attacker-training)
        for e in self.envs:
            e.set_opp_domain(_sample_opp_domain(e.cfg))

        self._refresh_obs(reset_envs=True)

    def _refresh_obs(self, reset_envs: bool = False):
        obs_def = []
        obs_att = []
        for e in self.envs:
            if reset_envs:
                e.reset()
            o_def, o_att = e.get_obs_pair()
            obs_def.append(o_def)
            obs_att.append(o_att)
        self.obs_def = np.stack(obs_def, axis=0)
        self.obs_att = np.stack(obs_att, axis=0)
        self.obs = self.obs_def

    def reset(self):
        self._refresh_obs(reset_envs=True)
        return self.obs

    def set_attr(self, name: str, value: Any):
        for e in self.envs:
            setattr(e, name, value)

    def step(self, a1_env: np.ndarray, aA_env: np.ndarray, reward_mode: str = "both"):
        obs_next = []
        obs_att_next = []
        r1, r2, done, info = [], [], [], []
        for i, e in enumerate(self.envs):
            o, R1, R2, d, inf = e.step(a1_env[i], aA_env[i], reward_mode=reward_mode)
            if d:
                # NEW: resample opponent domain at episode boundary
                if e.opp_resample == "episode":
                    e.set_opp_domain(_sample_opp_domain(e.cfg))
                o = e.reset()
            o_def, o_att = e.get_obs_pair()
            obs_next.append(o_def)
            obs_att_next.append(o_att)
            r1.append(R1); r2.append(R2); done.append(d); info.append(inf)
        self.obs_def = np.stack(obs_next, axis=0)
        self.obs_att = np.stack(obs_att_next, axis=0)
        self.obs = self.obs_def
        return self.obs, np.array(r1), np.array(r2), np.array(done, dtype=np.float32), info

    def close(self):
        return None


def _vec_chunk_sizes(num_envs: int, num_workers: int) -> List[int]:
    num_workers = max(1, min(int(num_workers), int(num_envs)))
    base, rem = divmod(int(num_envs), num_workers)
    sizes = []
    for idx in range(num_workers):
        size = base + (1 if idx < rem else 0)
        if size > 0:
            sizes.append(size)
    return sizes


def _subproc_start_method(start_method: str | None) -> str:
    if start_method is not None:
        return str(start_method)

    available = mp.get_all_start_methods()
    if "fork" in available:
        return "fork"
    if "forkserver" in available:
        return "forkserver"
    return "spawn"


def _subproc_vec_worker(conn, cfg: Dict[str, Any], num_envs: int, worker_idx: int):
    vec = None
    try:
        cfg_local = copy.deepcopy(cfg)
        base_seed = int(cfg_local.get("seed", 0))
        set_seed(base_seed + 100_003 * int(worker_idx) + 17)
        vec = VecEnv(lambda: Env(cfg_local), int(num_envs))

        while True:
            cmd, payload = conn.recv()

            if cmd == "get_obs":
                conn.send((vec.obs_def, vec.obs_att))
            elif cmd == "reset":
                vec.reset()
                conn.send((vec.obs_def, vec.obs_att))
            elif cmd == "step_full":
                a1_env, aA_env, reward_mode = payload
                _, r1, r2, done, info = vec.step(a1_env, aA_env, reward_mode=reward_mode)
                conn.send((vec.obs_def, vec.obs_att, r1, r2, done, info))
            elif cmd == "set_attr":
                name, value = payload
                vec.set_attr(name, value)
                conn.send(True)
            elif cmd == "get_ic_history":
                conn.send(collect_ic_history_from_vecenv(vec))
            elif cmd == "close":
                conn.send(True)
                break
            else:
                raise ValueError(f"Unknown SubprocVecEnv worker command: {cmd!r}")
    except EOFError:
        pass
    finally:
        try:
            if vec is not None:
                vec.close()
        finally:
            conn.close()


class SubprocVecEnv:
    torch_backend = False

    def __init__(
        self,
        cfg: Dict[str, Any],
        num_envs: int,
        num_workers: int | None = None,
        start_method: str | None = None,
    ):
        self.cfg = copy.deepcopy(cfg)
        self.num_envs = int(num_envs)
        requested_workers = num_workers
        if requested_workers is None:
            cpu_count = os.cpu_count() or 1
            requested_workers = min(self.num_envs, max(1, cpu_count // 2))
        self.chunk_sizes = _vec_chunk_sizes(self.num_envs, int(requested_workers))
        self.num_workers = len(self.chunk_sizes)
        self.start_method = _subproc_start_method(start_method)
        self._closed = False

        ctx = mp.get_context(self.start_method)
        self._conns = []
        self._procs = []

        for worker_idx, chunk_size in enumerate(self.chunk_sizes):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_subproc_vec_worker,
                args=(child_conn, self.cfg, chunk_size, worker_idx),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self._conns.append(parent_conn)
            self._procs.append(proc)

        self._refresh_obs(reset_envs=False)

    def _refresh_obs(self, reset_envs: bool = False):
        cmd = "reset" if reset_envs else "get_obs"
        for conn in self._conns:
            conn.send((cmd, None))

        obs_pairs = [conn.recv() for conn in self._conns]
        self.obs_def = np.concatenate([pair[0] for pair in obs_pairs], axis=0)
        self.obs_att = np.concatenate([pair[1] for pair in obs_pairs], axis=0)
        self.obs = self.obs_def
        return self.obs

    def reset(self):
        return self._refresh_obs(reset_envs=True)

    def set_attr(self, name: str, value: Any):
        for conn in self._conns:
            conn.send(("set_attr", (name, value)))
        for conn in self._conns:
            conn.recv()

    def step(self, a1_env: np.ndarray, aA_env: np.ndarray, reward_mode: str = "both"):
        if self._closed:
            raise RuntimeError("SubprocVecEnv is already closed.")

        split_idx = np.cumsum(self.chunk_sizes[:-1], dtype=int)
        a1_chunks = np.split(np.asarray(a1_env), split_idx, axis=0)
        aA_chunks = np.split(np.asarray(aA_env), split_idx, axis=0)

        for conn, a1_chunk, aA_chunk in zip(self._conns, a1_chunks, aA_chunks):
            conn.send(("step_full", (a1_chunk, aA_chunk, reward_mode)))

        results = [conn.recv() for conn in self._conns]
        self.obs_def = np.concatenate([res[0] for res in results], axis=0)
        self.obs_att = np.concatenate([res[1] for res in results], axis=0)
        self.obs = self.obs_def

        r1 = np.concatenate([res[2] for res in results], axis=0)
        r2 = np.concatenate([res[3] for res in results], axis=0)
        done = np.concatenate([res[4] for res in results], axis=0)
        info = [item for res in results for item in res[5]]
        return self.obs, r1, r2, done, info

    def collect_ic_history(self):
        for conn in self._conns:
            conn.send(("get_ic_history", None))

        collected = [conn.recv() for conn in self._conns]
        D = int(self.cfg.get("D", 3))

        def_all = [d for d, _ in collected if d.shape[0] > 0]
        att_all = [a for _, a in collected if a.shape[0] > 0]

        def_pos = np.concatenate(def_all, axis=0) if def_all else np.zeros((0, D), dtype=np.float32)
        att_pos = np.concatenate(att_all, axis=0) if att_all else np.zeros((0, D), dtype=np.float32)
        return def_pos, att_pos

    def close(self):
        if self._closed:
            return None

        for conn in self._conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass

        for conn in self._conns:
            try:
                conn.recv()
            except (BrokenPipeError, EOFError):
                pass
            finally:
                conn.close()

        for proc in self._procs:
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)

        self._closed = True
        return None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _opp_mode_code(mode: str) -> int:
    key = str(mode)
    if key == "none":
        return 1
    if key == "weak":
        return 2
    return 0


class TorchVecEnv:
    """
    Batched single-process torch rollout backend.

    The main dynamics, rewards, terminations, and observation assembly stay batched
    on torch tensors. Optional per-env reset sampling and estimator updates still use
    the existing Env helpers so behavior remains aligned with the scalar environment.
    """

    torch_backend = True

    def __init__(self, cfg: Dict[str, Any], num_envs: int, device: str | torch.device):
        self.cfg = copy.deepcopy(cfg)
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.envs: List[Env] = [Env(copy.deepcopy(cfg)) for _ in range(self.num_envs)]

        if not self.envs:
            raise ValueError("TorchVecEnv requires at least one environment.")

        base = self.envs[0]
        self.D = int(base.D)
        self.num_attackers = int(base.num_attackers)
        if self.num_attackers != 1:
            raise NotImplementedError("TorchVecEnv currently supports num_attackers=1 only.")

        self.dtype = torch.float32
        self.radius = float(base.radius)
        self.dt = float(base.dt)
        self.margin = float(base.margin)
        self.soft_wall = float(base.soft_wall)
        self.k_pos = float(base.k_pos)
        self.u_lo = float(base.u_lo)
        self.u_hi = float(base.u_hi)
        self.oi_radius_norm = float(base.oi_radius_norm)
        self.hit_buffer_def = float(base.hit_buffer_def)
        self.hit_buffer_att = float(base.hit_buffer_att)
        self.collision_radius_m = float(base.collision_radius_m)
        self.wall_penalty = float(base.wall_penalty)
        self.target_hit_reward_penalty = float(base.target_hit_reward_penalty)
        self.collision_penalty = float(base.collision_penalty)
        self.fuel_depletion_penalty = float(base.fuel_depletion_penalty)
        self.use_kf = bool(base.use_kf)
        self.use_fuel = bool(base.use_fuel)
        self.reward_type = str(base.reward_type)
        self.pos_obs_scale = float(base._pos_obs_scale)
        self.vel_obs_scale = float(base._vel_obs_scale)
        self.reset_ukf_on_diverge = bool(base.reset_ukf_on_diverge)

        self.center_t = torch.as_tensor(base.center, dtype=self.dtype, device=self.device)
        self.Ad_t = torch.as_tensor(base.Ad, dtype=self.dtype, device=self.device)
        self.Bd_t = torch.as_tensor(base.Bd, dtype=self.dtype, device=self.device)

        if self.use_fuel:
            self.k_eff_def = float(base.k_eff_def)
            self.k_eff_att = float(base.k_eff_att)
            self.g0 = float(base.g0)
            self.m0_def = float(base.m0_def)
            self.mdry_def = float(base.mdry_def)
            self.Tmax_def = float(base.Tmax_def)
            self.Isp_def = float(base.Isp_def)
            self.m0_att = float(base.m0_att)
            self.mdry_att = float(base.mdry_att)
            self.Tmax_att = float(base.Tmax_att)
            self.Isp_att = float(base.Isp_att)

        self.state_t = torch.zeros((self.num_envs, 4 * self.D), dtype=self.dtype, device=self.device)
        self._t_steps = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)
        self._opp_mode_codes = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)
        self._weak_scale_t = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
        self._weak_noise_std_t = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)

        if self.use_fuel:
            self.m_def_t = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
            self.m_att_t = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)

        for env in self.envs:
            env.set_opp_domain(_sample_opp_domain(env.cfg))

        self.obs_def = None
        self.obs_att = None
        self.obs = None
        self.reset()

    def _sync_slot_from_env(self, idx: int):
        env = self.envs[idx]
        self.state_t[idx] = torch.as_tensor(np.asarray(env.state, dtype=np.float32), dtype=self.dtype, device=self.device)
        self._t_steps[idx] = int(env.t)
        self._opp_mode_codes[idx] = _opp_mode_code(env.opp_domain)
        self._weak_scale_t[idx] = float(env.weak_scale)
        self._weak_noise_std_t[idx] = float(env.weak_noise_std)

        if self.use_fuel:
            self.m_def_t[idx] = float(env.m_def)
            self.m_att_t[idx] = float(env.m_att[0])

    def _sync_from_envs(self, indices: List[int] | None = None):
        if indices is None:
            indices = list(range(self.num_envs))
        for idx in indices:
            self._sync_slot_from_env(int(idx))

    def _split_state(self):
        D = self.D
        x1 = self.state_t[:, : 2 * D]
        x2 = self.state_t[:, 2 * D : 4 * D]
        return x1[:, :D], x1[:, D:], x2[:, :D], x2[:, D:]

    def _fuel_fractions(self):
        fuel_frac_def = (self.m_def_t - self.mdry_def) / (self.m0_def - self.mdry_def + 1e-9)
        fuel_frac_att = (self.m_att_t - self.mdry_att) / (self.m0_att - self.mdry_att + 1e-9)
        return fuel_frac_def.clamp(0.0, 1.0), fuel_frac_att.clamp(0.0, 1.0)

    def _refresh_obs(self):
        p1, v1, p2, v2 = self._split_state()

        if self.use_kf:
            p1_np = p1.detach().cpu().numpy()
            v1_np = v1.detach().cpu().numpy()
            p2_np = p2.detach().cpu().numpy()
            v2_np = v2.detach().cpu().numpy()

            p2_obs_np = np.empty_like(p2_np)
            v2_obs_np = np.empty_like(v2_np)
            p1_obs_np = np.empty_like(p1_np)
            v1_obs_np = np.empty_like(v1_np)

            for i, env in enumerate(self.envs):
                p2_obs_np[i], v2_obs_np[i] = env._belief_state(p2_np[i], v2_np[i])
                p1_obs_np[i], v1_obs_np[i] = env._attacker_belief_state(p1_np[i], v1_np[i])

            p2_obs = torch.as_tensor(p2_obs_np, dtype=self.dtype, device=self.device)
            v2_obs = torch.as_tensor(v2_obs_np, dtype=self.dtype, device=self.device)
            p1_obs = torch.as_tensor(p1_obs_np, dtype=self.dtype, device=self.device)
            v1_obs = torch.as_tensor(v1_obs_np, dtype=self.dtype, device=self.device)
        else:
            p2_obs, v2_obs = p2, v2
            p1_obs, v1_obs = p1, v1

        p1c_def = (p1 - self.center_t) * self.pos_obs_scale
        p2c_def = (p2_obs - self.center_t) * self.pos_obs_scale
        rel_def = (p2_obs - p1) * self.pos_obs_scale
        obs_def_parts = [
            p1c_def,
            p2c_def,
            rel_def,
            v1 * self.vel_obs_scale,
            v2_obs * self.vel_obs_scale,
        ]

        p1c_att = (p1_obs - self.center_t) * self.pos_obs_scale
        p2c_att = (p2 - self.center_t) * self.pos_obs_scale
        rel_att = (p2 - p1_obs) * self.pos_obs_scale
        obs_att_parts = [
            p1c_att,
            p2c_att,
            rel_att,
            v1_obs * self.vel_obs_scale,
            v2 * self.vel_obs_scale,
        ]

        if self.use_fuel:
            fuel_frac_def, fuel_frac_att = self._fuel_fractions()
            obs_def_parts.extend([fuel_frac_def[:, None], fuel_frac_att[:, None]])
            obs_att_parts.extend([fuel_frac_def[:, None], fuel_frac_att[:, None]])

        self.obs_def = torch.cat(obs_def_parts, dim=-1).to(dtype=self.dtype)
        self.obs_att = torch.cat(obs_att_parts, dim=-1).to(dtype=self.dtype)
        self.obs = self.obs_def
        return self.obs

    def reset(self):
        for env in self.envs:
            env.reset()
        self._sync_from_envs()
        self._refresh_obs()
        return self.obs

    def set_attr(self, name: str, value: Any):
        for env in self.envs:
            setattr(env, name, value)

    def _apply_propulsion_batch(
        self,
        a_cmd: torch.Tensor,
        m: torch.Tensor,
        m_dry: float,
        Tmax: float,
        Isp: float,
    ):
        alive = m > (m_dry + 1e-9)
        F_req = a_cmd * m[:, None]
        F_req_norm = torch.linalg.vector_norm(F_req, dim=-1)

        scale = torch.ones_like(F_req_norm)
        over = F_req_norm > Tmax
        scale = torch.where(over, Tmax / (F_req_norm + 1e-9), scale)
        F = F_req * scale[:, None]
        F = torch.where(alive[:, None], F, torch.zeros_like(F))

        a_real = F / torch.clamp(m[:, None], min=1e-9)
        thrust_norm = torch.linalg.vector_norm(F, dim=-1)
        mdot = thrust_norm / (Isp * self.g0)
        m_next = torch.where(alive, torch.clamp(m - mdot * self.dt, min=m_dry), torch.full_like(m, m_dry))
        fuel_depleted = m_next <= (m_dry + 1e-9)
        return a_real, m_next, fuel_depleted, thrust_norm, mdot

    def step(self, a1_env, aA_env, reward_mode: str = "both"):
        need_def = reward_mode in ("def", "both")
        need_att = reward_mode in ("att", "both")

        a1_cmd = torch.as_tensor(a1_env, dtype=self.dtype, device=self.device).reshape(self.num_envs, self.D)
        aA_cmd = torch.as_tensor(aA_env, dtype=self.dtype, device=self.device)
        if aA_cmd.ndim == 3:
            if aA_cmd.shape[1] != 1:
                raise NotImplementedError("TorchVecEnv currently supports a single attacker only.")
            a2_cmd = aA_cmd[:, 0, :]
        else:
            a2_cmd = aA_cmd.reshape(self.num_envs, self.D)

        none_mask = self._opp_mode_codes == 1
        weak_mask = self._opp_mode_codes == 2

        a1_cmd = a1_cmd.clone()
        if torch.any(none_mask):
            a1_cmd[none_mask] = 0.0
        if torch.any(weak_mask):
            a1_cmd[weak_mask] = self._weak_scale_t[weak_mask, None] * a1_cmd[weak_mask]
            weak_noise = self._weak_noise_std_t[weak_mask]
            if torch.any(weak_noise > 0.0):
                a1_cmd[weak_mask] = a1_cmd[weak_mask] + weak_noise[:, None] * torch.randn_like(a1_cmd[weak_mask])

        a1_cmd = torch.clamp(a1_cmd, self.u_lo, self.u_hi)
        a2_cmd = torch.clamp(a2_cmd, self.u_lo, self.u_hi)

        if self.use_fuel:
            a1_real, self.m_def_t, fuel_depleted_def, thrust_def, mdot_def = self._apply_propulsion_batch(
                a1_cmd, self.m_def_t, self.mdry_def, self.Tmax_def, self.Isp_def
            )
            a2_real, self.m_att_t, fuel_depleted_att, thrust_att, mdot_att = self._apply_propulsion_batch(
                a2_cmd, self.m_att_t, self.mdry_att, self.Tmax_att, self.Isp_att
            )
        else:
            a1_real = a1_cmd
            a2_real = a2_cmd
            fuel_depleted_def = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            fuel_depleted_att = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            thrust_def = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
            thrust_att = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
            mdot_def = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)
            mdot_att = torch.zeros((self.num_envs,), dtype=self.dtype, device=self.device)

        x1 = self.state_t[:, : 2 * self.D]
        x2 = self.state_t[:, 2 * self.D : 4 * self.D]
        x1n = torch.matmul(x1, self.Ad_t.transpose(0, 1)) + torch.matmul(a1_real, self.Bd_t.transpose(0, 1))
        if torch.any(none_mask):
            x1n = torch.where(none_mask[:, None], x1, x1n)
        x2n = torch.matmul(x2, self.Ad_t.transpose(0, 1)) + torch.matmul(a2_real, self.Bd_t.transpose(0, 1))
        self.state_t = torch.cat([x1n, x2n], dim=-1)
        self._t_steps = self._t_steps + 1

        p1, v1, p2, v2 = self._split_state()

        meas_innov = np.zeros((self.num_envs,), dtype=np.float32)
        meas_trP = np.zeros((self.num_envs,), dtype=np.float32)
        att_meas_innov = np.zeros((self.num_envs,), dtype=np.float32)
        att_meas_trP = np.zeros((self.num_envs,), dtype=np.float32)

        if self.use_kf:
            p1_np = p1.detach().cpu().numpy()
            v1_np = v1.detach().cpu().numpy()
            p2_np = p2.detach().cpu().numpy()
            v2_np = v2.detach().cpu().numpy()
            a1_real_np = a1_real.detach().cpu().numpy()
            a2_real_np = a2_real.detach().cpu().numpy()
            t_np = self._t_steps.detach().cpu().numpy()

            for i, env in enumerate(self.envs):
                env.t = int(t_np[i])
                u12_predict, u12_cov = env._kf_predict_control(a2_real_np[i])
                env.ukf.predict(dt=env.dt, u=u12_predict, u_cov=u12_cov)
                if (env.t % env._ukf_every) == 0:
                    p_obs = p1_np[i]
                    R_wb = np.eye(3)
                    v_b = _body_bearing_from_world(p_obs, R_wb, p2_np[i])
                    az_true, el_true = _azel_from_body_vec(v_b)
                    z_true = np.array([az_true, el_true], float)
                    z_noise = np.random.multivariate_normal(mean=np.zeros(2), cov=env._ukf_R)
                    z_meas = z_true + z_noise
                    z_hat_prior = (
                        env.ukf.measurement_prediction(p_obs, R_wb)
                        if hasattr(env.ukf, "measurement_prediction")
                        else env.ukf.h(env.ukf.x.copy(), p_obs, R_wb)
                    )
                    innov = z_meas - z_hat_prior
                    innov[0] = (innov[0] + np.pi) % (2 * np.pi) - np.pi
                    innov[1] = (innov[1] + np.pi) % (2 * np.pi) - np.pi
                    meas_innov[i] = float(innov @ innov)
                    env.ukf.update(z_meas, p_obs, R_wb)
                meas_trP[i] = float(np.trace(env.ukf.P[0:3, 0:3]))
                env._latest_meas_innov = float(meas_innov[i])
                env._latest_meas_trP = float(meas_trP[i])

                if env.att_ukf is not None:
                    u21_predict, u21_cov = env._kf_predict_control(a1_real_np[i])
                    env.att_ukf.predict(dt=env.dt, u=u21_predict, u_cov=u21_cov)
                    if (env.t % env._ukf_every) == 0:
                        p_obs = p2_np[i]
                        R_wb = np.eye(3)
                        v_b_att = _body_bearing_from_world(p_obs, R_wb, p1_np[i])
                        az_att_true, el_att_true = _azel_from_body_vec(v_b_att)
                        z_att_true = np.array([az_att_true, el_att_true], float)
                        z_att_noise = np.random.multivariate_normal(mean=np.zeros(2), cov=env._ukf_R)
                        z_att_meas = z_att_true + z_att_noise
                        z_att_hat_prior = (
                            env.att_ukf.measurement_prediction(p_obs, R_wb)
                            if hasattr(env.att_ukf, "measurement_prediction")
                            else env.att_ukf.h(env.att_ukf.x.copy(), p_obs, R_wb)
                        )
                        att_innov = z_att_meas - z_att_hat_prior
                        att_innov[0] = (att_innov[0] + np.pi) % (2 * np.pi) - np.pi
                        att_innov[1] = (att_innov[1] + np.pi) % (2 * np.pi) - np.pi
                        att_meas_innov[i] = float(att_innov @ att_innov)
                        env.att_ukf.update(z_att_meas, p_obs, R_wb)
                    att_meas_trP[i] = float(np.trace(env.att_ukf.P[0:3, 0:3]))
                    env._latest_att_meas_innov = float(att_meas_innov[i])
                    env._latest_att_meas_trP = float(att_meas_trP[i])
                else:
                    env._latest_att_meas_innov = 0.0
                    env._latest_att_meas_trP = 0.0

        center_delta_def = p1 - self.center_t
        center_delta_att = p2 - self.center_t
        d1_true_norm = center_delta_def.square().sum(dim=-1) / (self.radius ** 2)
        d2_true_norm = center_delta_att.square().sum(dim=-1) / (self.radius ** 2)
        rel2 = (p2 - p1).square().sum(dim=-1) / (self.radius ** 2)

        rho1 = torch.linalg.vector_norm(center_delta_def, dim=-1) / self.radius
        rho2 = torch.linalg.vector_norm(center_delta_att, dim=-1) / self.radius

        thresh_def = (1.0 + self.hit_buffer_def) * self.oi_radius_norm
        thresh_att = (1.0 + self.hit_buffer_att) * self.oi_radius_norm
        att_hit_target = (self.oi_radius_norm > 0.0) & (rho2 <= thresh_att)
        def_hit_target = (self.oi_radius_norm > 0.0) & (rho1 <= thresh_def)
        hit_target = att_hit_target | def_hit_target

        collision = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        if self.collision_radius_m > 0.0:
            collision = torch.linalg.vector_norm(p2 - p1, dim=-1) <= self.collision_radius_m

        oob1 = rho1 >= self.margin
        oob2_any = rho2 >= self.margin

        if torch.any(none_mask):
            oob1 = torch.where(none_mask, torch.zeros_like(oob1), oob1)
            def_hit_target = torch.where(none_mask, torch.zeros_like(def_hit_target), def_hit_target)
            collision = torch.where(none_mask, torch.zeros_like(collision), collision)
            hit_target = torch.where(none_mask, att_hit_target, hit_target)

        done = oob1 | oob2_any | hit_target | collision
        if self.use_fuel:
            done = done | fuel_depleted_att | fuel_depleted_def

        if self.reward_type == "zero_sum" and self.use_kf:
            raise RuntimeError("use_zero_sum requires use_kf=False.")
        if self.reward_type == "zero_sum_kf" and not self.use_kf:
            raise RuntimeError("use_zero_sum_kf requires use_kf=True.")

        g = self.k_pos * d2_true_norm
        if self.use_fuel:
            burn_frac_def = (mdot_def * self.dt) / (self.m0_def - self.mdry_def + 1e-9)
            burn_frac_att = (mdot_att * self.dt) / (self.m0_att - self.mdry_att + 1e-9)
            g = g - self.k_eff_def * burn_frac_def + self.k_eff_att * burn_frac_att

        if torch.any(done):
            terminal_delta = torch.zeros_like(g)
            terminal_delta = torch.where(collision, terminal_delta + self.collision_penalty, terminal_delta)
            still_open = ~collision
            terminal_delta = torch.where(still_open & (att_hit_target | def_hit_target), terminal_delta - self.target_hit_reward_penalty, terminal_delta)
            still_open = still_open & ~(att_hit_target | def_hit_target)
            terminal_delta = torch.where(still_open & oob1, terminal_delta - self.wall_penalty, terminal_delta)
            still_open = still_open & ~oob1
            terminal_delta = torch.where(still_open & oob2_any, terminal_delta + self.wall_penalty, terminal_delta)
            if self.use_fuel:
                still_open = still_open & ~oob2_any
                terminal_delta = torch.where(still_open & fuel_depleted_def, terminal_delta - self.fuel_depletion_penalty, terminal_delta)
                still_open = still_open & ~fuel_depleted_def
                terminal_delta = torch.where(still_open & fuel_depleted_att, terminal_delta + self.fuel_depletion_penalty, terminal_delta)
            g = g + terminal_delta

        r1 = g if need_def else torch.zeros_like(g)
        r2 = -g if need_att else torch.zeros_like(g)

        d2_belief_norm = None
        p2_est_err_norm = None
        d1_belief_norm = None
        p1_est_err_norm = None
        if self.use_kf:
            p1_np = p1.detach().cpu().numpy()
            v1_np = v1.detach().cpu().numpy()
            p2_np = p2.detach().cpu().numpy()
            v2_np = v2.detach().cpu().numpy()
            d2_belief_norm = np.zeros((self.num_envs,), dtype=np.float32)
            p2_est_err_norm = np.zeros((self.num_envs,), dtype=np.float32)
            d1_belief_norm = np.zeros((self.num_envs,), dtype=np.float32)
            p1_est_err_norm = np.zeros((self.num_envs,), dtype=np.float32)
            for i, env in enumerate(self.envs):
                p2_belief, _ = env._belief_state(p2_np[i], v2_np[i])
                d2_belief_norm[i] = float(np.dot(p2_belief - env.center, p2_belief - env.center)) / (env.radius ** 2)
                p2_est_err_norm[i] = float(np.dot(p2_belief - p2_np[i], p2_belief - p2_np[i])) / (env.radius ** 2)
                if env.att_ukf is not None:
                    p1_belief, _ = env._attacker_belief_state(p1_np[i], v1_np[i])
                    d1_belief_norm[i] = float(np.dot(p1_belief - env.center, p1_belief - env.center)) / (env.radius ** 2)
                    p1_est_err_norm[i] = float(np.dot(p1_belief - p1_np[i], p1_belief - p1_np[i])) / (env.radius ** 2)

        d2_np = d2_true_norm.detach().cpu().numpy()
        rel2_np = rel2.detach().cpu().numpy()
        d1_true_np = d1_true_norm.detach().cpu().numpy()
        d2_true_np = d2_true_norm.detach().cpu().numpy()
        oob1_np = oob1.detach().cpu().numpy()
        oob2_np = oob2_any.detach().cpu().numpy()
        hit_target_np = hit_target.detach().cpu().numpy()
        collision_np = collision.detach().cpu().numpy()
        t_np = self._t_steps.detach().cpu().numpy()

        fuel_frac_def_np = None
        fuel_frac_att_np = None
        thrust_def_np = None
        thrust_att_np = None
        mdot_def_np = None
        mdot_att_np = None
        if self.use_fuel:
            fuel_frac_def_np = self._fuel_fractions()[0].detach().cpu().numpy()
            fuel_frac_att_np = self._fuel_fractions()[1].detach().cpu().numpy()
            thrust_def_np = thrust_def.detach().cpu().numpy()
            thrust_att_np = thrust_att.detach().cpu().numpy()
            mdot_def_np = mdot_def.detach().cpu().numpy()
            mdot_att_np = mdot_att.detach().cpu().numpy()

        infos = []
        for i in range(self.num_envs):
            info = {
                "t": int(t_np[i]),
                "d2_norm": float(d2_np[i]),
                "rel2_norm": float(rel2_np[i]),
                "oob_def": bool(oob1_np[i]),
                "oob_att": bool(oob2_np[i]),
                "hit_target": bool(hit_target_np[i]),
                "d1_true_norm": float(d1_true_np[i]),
                "d2_true_norm": float(d2_true_np[i]),
                "collision": bool(collision_np[i]),
            }
            if self.use_kf:
                info["meas_innov_sq"] = float(meas_innov[i])
                info["ukf_trPpos"] = float(meas_trP[i])
                info["att_meas_innov_sq"] = float(att_meas_innov[i])
                info["att_ukf_trPpos"] = float(att_meas_trP[i])
                if d2_belief_norm is not None:
                    info["d2_belief_norm"] = float(d2_belief_norm[i])
                if p2_est_err_norm is not None:
                    info["p2_est_err_norm"] = float(p2_est_err_norm[i])
                if d1_belief_norm is not None:
                    info["d1_belief_norm"] = float(d1_belief_norm[i])
                if p1_est_err_norm is not None:
                    info["p1_est_err_norm"] = float(p1_est_err_norm[i])
            if self.use_fuel:
                info["fuel_frac_def"] = float(np.clip(fuel_frac_def_np[i], 0.0, 1.0))
                info["fuel_frac_att"] = float(np.clip(fuel_frac_att_np[i], 0.0, 1.0))
                info["fuel_used_def"] = 1.0 - info["fuel_frac_def"]
                info["fuel_used_att"] = 1.0 - info["fuel_frac_att"]
                info["thrust_def"] = float(thrust_def_np[i])
                info["thrust_att"] = float(thrust_att_np[i])
                info["mdot_def"] = float(mdot_def_np[i])
                info["mdot_att"] = float(mdot_att_np[i])
            infos.append(info)

        done_indices = torch.nonzero(done, as_tuple=False).flatten().detach().cpu().tolist()
        for idx in done_indices:
            env = self.envs[idx]
            if env.opp_resample == "episode":
                env.set_opp_domain(_sample_opp_domain(env.cfg))
            env.reset()
            self._sync_slot_from_env(idx)

        self._refresh_obs()
        return self.obs, r1, r2, done.to(dtype=self.dtype), infos

    def close(self):
        return None


def collect_ic_history_from_vecenv(vec: VecEnv):
    """
    Aggregate initial-condition history from all sub-envs.

    Returns
    -------
    def_pos : (N, D)
    att_pos : (M, D)
    """
    if hasattr(vec, "envs"):
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

    if hasattr(vec, "collect_ic_history"):
        return vec.collect_ic_history()

    raise TypeError(f"Unsupported vec env type: {type(vec)!r}")


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
