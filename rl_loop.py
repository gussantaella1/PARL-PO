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
"""

import importlib
import os
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import gc
import time
import sys
import platform
from datetime import datetime, timezone
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

# from __future__ import annotations

from dataclasses import dataclass


from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import json

from pathlib import Path


#Logger utils:

def _utc_now():
    return datetime.now(timezone.utc).isoformat()

def _to_jsonable(x):
    """Recursively convert common non-JSON types (numpy/torch/etc.) to JSON-safe."""
    # basic
    if x is None or isinstance(x, (bool, int, float, str)):
        return x

    # dict-like
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}

    # list/tuple/set
    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(v) for v in x]

    # numpy scalars/arrays
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer, np.floating, np.bool_)):
            return x.item()
    except Exception:
        pass

    # torch tensors/devices
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return {
                "__type__": "torch.Tensor",
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "device": str(x.device),
            }
        if isinstance(x, torch.device):
            return str(x)
    except Exception:
        pass

    # pathlib paths
    try:
        from pathlib import Path
        if isinstance(x, Path):
            return str(x)
    except Exception:
        pass

    # fallback: stringify (handles callables, classes, etc.)
    return {"__type__": type(x).__name__, "__repr__": repr(x)}

class RunLogger:
    def __init__(self, out_dir: str, filename: str = "run_manifest.json"):
        self.out_dir = out_dir
        self.path = os.path.join(out_dir, filename)

        self.data = {
            "created_utc": _utc_now(),
            "out_dir": out_dir,
            "env": {
                "python": sys.version,
                "platform": platform.platform(),
            },
            "configs": {},
            "stages": [],   # chronological list of stage records
        }
        self.flush()  # create file immediately

    def set_config(self, key: str, cfg_obj):
        self.data["configs"][key] = _to_jsonable(cfg_obj)
        self.flush()

    def flush(self):
        os.makedirs(self.out_dir, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2, sort_keys=False)
        os.replace(tmp, self.path)  # atomic on POSIX/most OSes

    @contextmanager
    def stage(self, name: str, **meta):
        rec = {
            "name": name,
            "start_utc": _utc_now(),
            "meta": _to_jsonable(meta),
        }
        t0 = time.perf_counter()
        try:
            yield rec  # you can mutate rec inside the with-block to add outputs
            rec["status"] = "ok"
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = {"type": type(e).__name__, "repr": repr(e)}
            raise
        finally:
            rec["end_utc"] = _utc_now()
            rec["duration_s"] = float(time.perf_counter() - t0)
            self.data["stages"].append(rec)
            self.flush()


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
        self.k_pos = float(cfg.get("k_pos"))

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

        rho1 = np.linalg.norm(p1 - self.center)/ self.radius
        rho2 = np.linalg.norm(p2 - self.center) / self.radius

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

        v2n2 = float(np.dot(v2, v2)) / (v_scale**2)
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

        use_security = True


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
                burn_frac_att = (mdot_att_all[0] * self.dt) / (self.m0_att - self.mdry_att + 1e-9)
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
                    g -= self.wallK
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

            

        else:

            if need_def:
                r1 = (
                    self.alpha * delta_d2
                    + self.k_pos * d2
                    # - self.k_rel * rel2
                    # - self.k_vel * v1n2
                    # - k_vrad * (vrad1**2)
                    - self.lD * a1n2
                    - wall1
                    - center_keepout
                )

            if need_att:

                dist = float(np.linalg.norm(p2 - p1))  # meters
                x = dist / (float(self.att_min_sep) + 1e-9)  # 1.0 at the boundary
                close_pen = max(0.0, 1.0 - x) ** 2           # ramps up as dist -> min_sep
                d2_prev = float(self._d2_prev if self._d2_prev is not None else d2)

                progress = d2_prev - d2  # positive inward
                r2 = (
                    + self.k_att_prog * progress
                    - self.k_att_close * close_pen
                    - self.lA  * a2n2
                    - wall2
                )

            if need_def:
                if done:
                    if collision:
                        r1 -= self.collision_penalty
                    if oob1: 
                        r1 -= self.wallK
                    if att_hit_target: 
                        r1 -= self.target_hit_reward_penalty
                    if def_hit_target: 
                        r1 -= self.target_hit_reward_penalty

            if need_att:
                if done:
                    if collision:
                        r2 -= self.collision_penalty
                    if oob2_any: 
                        r2 -= self.wallK
                    if att_hit_target: 
                        r2 += self.target_hit_reward_penalty


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
            fuel_frac_att = (self.m_att[0] - self.mdry_att) / (self.m0_att - self.mdry_att + 1e-9)

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

        self.feature_dim = 4 * self.D + (2 if self.use_fuel else 0)

    def _split_obs(self, obs: torch.Tensor):
        """
        Single-attacker observation parser.

        Returns:
            p1c, p2c, rel, v1, v2, fdef, fatt
        where fdef/fatt are None if fuel is disabled.
        """
        D = self.D
        expected = 5 * D + (2 if self.use_fuel else 0)
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
            fdef = obs[:, 5*D:5*D+1]
            fatt = obs[:, 5*D+1:5*D+2]
        else:
            fdef = None
            fatt = None

        return p1c, p2c, rel, v1, v2, fdef, fatt

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
        p1c, p2c, rel, v1, v2, fdef, fatt = self._split_obs(obs)

        u_def_prior = self._one_step_prior(p1c, v1)
        u_att_prior = self._one_step_prior(p2c, v2)
        u_prior = u_def_prior if who == "def" else u_att_prior

        if self.use_fuel:
            feats = torch.cat([p1c, p2c, v1, v2, fdef, fatt], dim=-1)
        else:
            feats = torch.cat([p1c, p2c, v1, v2], dim=-1)

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

        if self.num_attackers != 1:
            raise NotImplementedError(
                "NoPriorLayer currently supports only num_attackers=1."
            )

        self.feature_dim = 4 * self.D + (2 if self.use_fuel else 0)

    def _split_obs(self, obs: torch.Tensor):
        D = self.D
        expected = 5 * D + (2 if self.use_fuel else 0)
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
            fdef = obs[:, 5*D:5*D+1]
            fatt = obs[:, 5*D+1:5*D+2]
        else:
            fdef = None
            fatt = None

        return p1c, p2c, rel, v1, v2, fdef, fatt

    def forward(self, obs: torch.Tensor, who: str):
        B, D = obs.shape[0], self.D
        device, dtype = obs.device, obs.dtype

        p1c, p2c, rel, v1, v2, fdef, fatt = self._split_obs(obs)

        if self.use_fuel:
            feats = torch.cat([p1c, p2c, v1, v2, fdef, fatt], dim=-1)
        else:
            feats = torch.cat([p1c, p2c, v1, v2], dim=-1)

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
    obs_list = []
    for e in vec.envs:
        p1, v1, pA_list, vA_list = e._unpack(e.state)
        p2 = pA_list[0]
        v2 = vA_list[0]
        c = e.center

        parts = [
            p1 - c,
            p2 - c,
            p2 - p1,
            v1,
            v2,
        ]

        if e.use_fuel:
            fdef = (e.m_def - e.mdry_def) / (e.m0_def - e.mdry_def + 1e-9)
            fatt = (e.m_att[0] - e.mdry_att) / (e.m0_att - e.mdry_att + 1e-9)
            parts.append(np.array([np.clip(fdef, 0.0, 1.0)], dtype=np.float32))
            parts.append(np.array([np.clip(fatt, 0.0, 1.0)], dtype=np.float32))

        obs_list.append(np.concatenate(parts).astype(np.float32))

    return torch.as_tensor(np.stack(obs_list, axis=0), dtype=torch.float32, device=device)


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
    num_envs      = int(cfg.get("num_envs"))
    steps_per_env = int(cfg.get("steps_per_env"))
    total_updates = int(cfg.get("total_updates"))
    log_every     = int(cfg.get("log_every"))

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

    expected_obs_dim = 5 * cfg["D"] + (2 if cfg.get("fuel", {}).get("enable", False) else 0)
    assert obs_dim == expected_obs_dim, f"obs_dim={obs_dim}, expected={expected_obs_dim}"
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
            o2_np, _, _, done, infos = vec.step(
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

    Collect (obs, a_rule) by rolling the env with:
      - defender policy (either random-ish PPO init, or optional frozen ckpt if provided)
      - attacker actions from AttackerRuleController

    Then train attacker network to predict a_rule via MSE on dist.mean.
    """
    assert cfg.get("attacker_mode", "rule") == "rl", "BC pretrain requires attacker_mode='rl'"
    assert int(cfg.get("num_attackers", 1)) == 1, "BC pretrain currently assumes num_attackers=1"

    set_seed(cfg["seed"])
    device = cfg["device"]


    # build_dyn(cfg)             # <-- ADD THIS (or ensure_dyn(cfg))

    def make_env():
        return Env(cfg)

    num_envs = int(cfg.get("num_envs"))
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

def _cpu_state_dict(m: torch.nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in m.state_dict().items()}

def _save_role_checkpoint(ppo, train_role: str, path: str):
    if train_role == "def":
        net = ppo.def_net
    elif train_role == "att":
        net = ppo.att_net
        if net is None:
            raise RuntimeError("Tried to save attacker checkpoint, but ppo.att_net is None.")
    else:
        raise ValueError(f"Unknown train_role={train_role!r}")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(_cpu_state_dict(net), path)

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

    # -------------------------
    # Checkpoint saving config
    # -------------------------
    save_best_ckpt = bool(cfg.get("save_best_ckpt", True))
    save_last_ckpt = bool(cfg.get("save_last_ckpt", True))
    checkpoint_dir = cfg.get("checkpoint_dir", None)
    checkpoint_prefix = cfg.get("checkpoint_prefix", f"{train_role}_teacher")

    tracked_metric_name = "R_def_mean" if train_role == "def" else "R_att_mean"
    best_metric = -float("inf")
    best_update = None
    best_ckpt_path = None
    last_ckpt_path = None

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    def make_env():
        return Env(cfg)

    num_envs = int(cfg.get("num_envs"))
    steps_per_env = int(cfg.get("steps_per_env"))
    total_updates = int(cfg.get("total_updates"))
    log_every = int(cfg.get("log_every"))

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
    verify_freeze = bool(cfg.get("verify_freeze"))
    freeze_tol = float(cfg.get("freeze_tol"))  # set 0.0 for exact; or 1e-12 if you’re paranoid

    snap_def = None
    snap_att = None

    if verify_freeze and cfg.get("freeze_defender"):
        # defender is frozen opponent in attacker-training
        snap_def = snapshot_state_dict(ppo.def_net)

    if verify_freeze and cfg.get("freeze_attacker") and (ppo.att_net is not None):
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

    if cfg.get("fuel", {}).get("enable", False):
        metrics["fuel_used_def_mean"] = []
        metrics["fuel_used_att_mean"] = []
        metrics["fuel_frac_def_mean"] = []
        metrics["fuel_frac_att_mean"] = []

        # optional
        metrics["thrust_def_mean"] = []
        metrics["thrust_att_mean"] = []
        metrics["mdot_def_mean"] = []
        metrics["mdot_att_mean"] = []



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

        if cfg.get("fuel", {}).get("enable", False):
            fuel_used_def_acc = 0.0
            fuel_used_att_acc = 0.0
            fuel_frac_def_acc = 0.0
            fuel_frac_att_acc = 0.0

            thrust_def_acc = 0.0
            thrust_att_acc = 0.0
            mdot_def_acc = 0.0
            mdot_att_acc = 0.0
            fuel_info_count = 0

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

                if inf.get("oob_def", False): term_counts["oob_def"] += 1
                if inf.get("oob_att", False): term_counts["oob_att"] += 1
                if inf.get("hit_target", False): term_counts["hit_target"] += 1
                if inf.get("collision", False): term_counts["collision"] += 1

                if cfg.get("fuel", {}).get("enable", False):
                    if "fuel_used_def" in inf:
                        fuel_used_def_acc += inf["fuel_used_def"]
                        fuel_used_att_acc += inf["fuel_used_att"]
                        fuel_frac_def_acc += inf["fuel_frac_def"]
                        fuel_frac_att_acc += inf["fuel_frac_att"]

                        thrust_def_acc += inf.get("thrust_def", 0.0)
                        thrust_att_acc += inf.get("thrust_att", 0.0)
                        mdot_def_acc += inf.get("mdot_def", 0.0)
                        mdot_att_acc += inf.get("mdot_att", 0.0)

                        fuel_info_count += 1


            global_env_step += num_envs




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

            tracked_metric_value = R_def_mean if train_role == "def" else R_att_mean

            if save_best_ckpt and checkpoint_dir is not None:
                if tracked_metric_value > best_metric:
                    best_metric = float(tracked_metric_value)
                    best_update = int(upd)
                    best_ckpt_path = os.path.join(
                        checkpoint_dir,
                        f"{checkpoint_prefix}__best.pt"
                    )
                    _save_role_checkpoint(ppo, train_role, best_ckpt_path)
                    print(
                        f"[checkpoint] new best {tracked_metric_name}={best_metric:+.3f} "
                        f"at update {best_update} -> {best_ckpt_path}"
                    )

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

            if cfg.get("fuel", {}).get("enable", False):
                if fuel_info_count > 0:
                    fuel_used_def_mean = fuel_used_def_acc / fuel_info_count
                    fuel_used_att_mean = fuel_used_att_acc / fuel_info_count
                    fuel_frac_def_mean = fuel_frac_def_acc / fuel_info_count
                    fuel_frac_att_mean = fuel_frac_att_acc / fuel_info_count

                    thrust_def_mean = thrust_def_acc / fuel_info_count
                    thrust_att_mean = thrust_att_acc / fuel_info_count
                    mdot_def_mean = mdot_def_acc / fuel_info_count
                    mdot_att_mean = mdot_att_acc / fuel_info_count
                else:
                    fuel_used_def_mean = fuel_used_att_mean = 0.0
                    fuel_frac_def_mean = fuel_frac_att_mean = 0.0
                    thrust_def_mean = thrust_att_mean = 0.0
                    mdot_def_mean = mdot_att_mean = 0.0

                metrics["fuel_used_def_mean"].append(fuel_used_def_mean)
                metrics["fuel_used_att_mean"].append(fuel_used_att_mean)
                metrics["fuel_frac_def_mean"].append(fuel_frac_def_mean)
                metrics["fuel_frac_att_mean"].append(fuel_frac_att_mean)

                metrics["thrust_def_mean"].append(thrust_def_mean)
                metrics["thrust_att_mean"].append(thrust_att_mean)
                metrics["mdot_def_mean"].append(mdot_def_mean)
                metrics["mdot_att_mean"].append(mdot_att_mean)


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

            if cfg.get("fuel", {}).get("enable", False):
                print(
                    f"   fuel used: def={fuel_used_def_mean:.6f}, att={fuel_used_att_mean:.6f}   "
                    f"fuel remaining: def={fuel_frac_def_mean:.6f}, att={fuel_frac_att_mean:.6f}"
                )
                print(
                    f"   thrust mean: def={thrust_def_mean:.6e}, att={thrust_att_mean:.6e}   "
                    f"mdot mean: def={mdot_def_mean:.6e}, att={mdot_att_mean:.6e}"
                )

            
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

                if cfg.get("fuel", {}).get("enable", False):
                    writer.add_scalar("fuel/used_def_mean", fuel_used_def_mean, gs)
                    writer.add_scalar("fuel/used_att_mean", fuel_used_att_mean, gs)
                    writer.add_scalar("fuel/remaining_def_mean", fuel_frac_def_mean, gs)
                    writer.add_scalar("fuel/remaining_att_mean", fuel_frac_att_mean, gs)

                    writer.add_scalar("fuel/thrust_def_mean", thrust_def_mean, gs)
                    writer.add_scalar("fuel/thrust_att_mean", thrust_att_mean, gs)
                    writer.add_scalar("fuel/mdot_def_mean", mdot_def_mean, gs)
                    writer.add_scalar("fuel/mdot_att_mean", mdot_att_mean, gs)



    ic_used_path = None
    if cfg.get("record_ic_history", False) and checkpoint_dir is not None:
        def_used, att_used = collect_ic_history_from_vecenv(vec)
        ic_used_path = os.path.join(checkpoint_dir, f"ic_samples_{checkpoint_prefix}.npz")
        np.savez(
            ic_used_path,
            def_pos=def_used,
            att_pos=att_used,
        )
        print(f"[ic] saved actual training IC samples -> {ic_used_path}")

            
            
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

    if save_last_ckpt and checkpoint_dir is not None:
        last_ckpt_path = os.path.join(
            checkpoint_dir,
            f"{checkpoint_prefix}.pt"
        )
        _save_role_checkpoint(ppo, train_role, last_ckpt_path)
        print(f"[checkpoint] saved last checkpoint -> {last_ckpt_path}")

    ckpt_info = {
        "tracked_metric_name": tracked_metric_name,
        "best_metric": best_metric,
        "best_update": best_update,
        "best_ckpt_path": best_ckpt_path,
        "last_ckpt_path": last_ckpt_path,
        "ic_used_path": ic_used_path,

    }

    print("Training finished.")
    return ppo, metrics, ckpt_info


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

    # 6) Fuel utilization:

    fuel_keys = [k for k in [
        "fuel_used_def_mean",
        "fuel_used_att_mean",
        "fuel_frac_def_mean",
        "fuel_frac_att_mean",
    ] if k in metrics and len(np.asarray(metrics[k]).reshape(-1)) == len(x)]

    if fuel_keys:
        fig = plt.figure()
        ax = plt.gca()
        _plot_if(ax, "fuel_used_def_mean", "Defender fuel used frac")
        _plot_if(ax, "fuel_used_att_mean", "Attacker fuel used frac")
        _plot_if(ax, "fuel_frac_def_mean", "Defender fuel remaining frac")
        _plot_if(ax, "fuel_frac_att_mean", "Attacker fuel remaining frac")
        ax.set_xlabel("Update")
        ax.set_ylabel("Fraction")
        ax.set_title(f"{title} — Fuel" if title else "Fuel")
        ax.grid(True)
        ax.legend()
        figs.append(("fuel", fig))

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

    for item in labeled_paths:
        if len(item) == 2:
            label, path = item
            metric_key = metric
        else:
            label, path, metric_key = item

        m = load_npz_metrics(path)
        if metric_key not in m:
            print(f"[plot_compare_phases] skipping {label}: missing key {metric_key!r}")
            continue

        x = _as_1d(m["update"]) if "update" in m else np.arange(len(m[metric_key]))
        y = _smooth_series(m[metric_key], method=smooth, param=smooth_param)
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

def _draw_circle(ax, radius, center_xy=(0.0, 0.0), **kwargs):
    th = np.linspace(0.0, 2.0 * np.pi, 400)
    x = center_xy[0] + radius * np.cos(th)
    y = center_xy[1] + radius * np.sin(th)
    ax.plot(x, y, **kwargs)


def _plot_ic_projection(
    ax,
    def_pos: np.ndarray,
    att_pos: np.ndarray,
    dims=(0, 1),
    title="",
    arena_r=None,
    oi_r=None,
    center=None,
    alpha_def=0.15,
    alpha_att=0.15,
    s=4,
):
    i, j = dims
    center = np.zeros(3, dtype=float) if center is None else np.asarray(center, dtype=float)

    if def_pos.shape[0] > 0:
        ax.scatter(
            def_pos[:, i], def_pos[:, j],
            s=s, alpha=alpha_def, label="Defender starts"
        )

    if att_pos.shape[0] > 0:
        ax.scatter(
            att_pos[:, i], att_pos[:, j],
            s=s, alpha=alpha_att, label="Attacker starts"
        )

    # arena / OI circles only make geometric sense in centered projections
    if arena_r is not None:
        _draw_circle(ax, arena_r, center_xy=(center[i], center[j]), linestyle="--", linewidth=1.0)
    if oi_r is not None and oi_r > 0.0:
        _draw_circle(ax, oi_r, center_xy=(center[i], center[j]), linestyle="-", linewidth=1.0)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_title(title)
    ax.legend()


def plot_ic_samples(
    def_pos: np.ndarray,
    att_pos: np.ndarray,
    cfg: Dict[str, Any],
    title: str = "",
    out_path: str | None = None,
    show: bool = False,
    close: bool = True,
):
    """
    Plot IC samples in XY and, if D=3, also XZ projection.
    """
    D = int(cfg["D"])
    ar = cfg["arena"]
    center = np.array(
        [ar["cx"], ar["cy"], (ar.get("cz", 0.0) if D == 3 else 0.0)],
        dtype=float,
    )[:D]
    arena_r = float(ar["r"])
    oi_r = float(cfg.get("oi", {}).get("r", 0.0))

    if D == 2:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        _plot_ic_projection(
            ax,
            def_pos,
            att_pos,
            dims=(0, 1),
            title=title or "IC samples (XY)",
            arena_r=arena_r,
            oi_r=oi_r,
            center=np.pad(center, (0, 1)),
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        _plot_ic_projection(
            axes[0],
            def_pos,
            att_pos,
            dims=(0, 1),
            title=(title + " — XY") if title else "IC samples — XY",
            arena_r=arena_r,
            oi_r=oi_r,
            center=center,
        )
        _plot_ic_projection(
            axes[1],
            def_pos,
            att_pos,
            dims=(0, 2),
            title=(title + " — XZ") if title else "IC samples — XZ",
            arena_r=arena_r,
            oi_r=oi_r,
            center=center,
        )

    fig.tight_layout()

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    if close:
        plt.close(fig)

    return fig

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

def plot_ic_support_from_cfg(
    cfg: Dict[str, Any],
    n_scenes: int = 20000,
    seed: int = 123,
    title: str = "Feasible initial-condition support",
    out_path: str | None = None,
    show: bool = False,
):
    def_pos, att_pos = sample_ic_support(cfg, n_scenes=n_scenes, seed=seed)
    return plot_ic_samples(
        def_pos,
        att_pos,
        cfg,
        title=title,
        out_path=out_path,
        show=show,
    )


def plot_ic_used_from_npz(
    npz_path: str,
    cfg: Dict[str, Any],
    title: str = "Initial conditions actually used during training",
    out_path: str | None = None,
    show: bool = False,
):
    data = np.load(npz_path)
    def_pos = data["def_pos"]
    att_pos = data["att_pos"]
    return plot_ic_samples(
        def_pos,
        att_pos,
        cfg,
        title=title,
        out_path=out_path,
        show=show,
    )


    # ---------------------------------------------------------
    # Helper: train defender teacher + distill to UKF student
    # ---------------------------------------------------------

def train_with_distill(
    phase_name: str,
    attacker_mode: str,
    train_role: str,
    extra_train_cfg: Dict[str, Any] | None = None,
):
    """
    Unified wrapper for:
      - teacher training (full-state)
      - optional UKF student distillation

    Replaces both:
      - train_defender_with_distill(...)
      - train_attacker_with_distill(...)

    Args:
        phase_name: label used for checkpoints / metrics filenames
        attacker_mode: "rule" or "rl"
        train_role: "def" or "att"
        extra_train_cfg: optional overrides merged into both teacher and student cfgs

    Returns
    -------
    teacher_ckpt : str
        Path to best teacher checkpoint (or last if best missing)
    student_ckpt : str | None
        Path to distilled student checkpoint if distill=True, else None
    """
    if train_role not in ("def", "att"):
        raise ValueError(f"train_role must be 'def' or 'att', got {train_role!r}")

    role_upper = "DEFENDER" if train_role == "def" else "ATTACKER"
    role_lower = "defender" if train_role == "def" else "attacker"

    # =========================================================
    # TEACHER (full-state)
    # =========================================================
    cfg_teacher = config_for_train(
        attacker_mode=attacker_mode,
        train_role=train_role,
    )
    cfg_teacher["use_ukf"] = False  # teacher is always full-state

    if extra_train_cfg is not None:
        cfg_teacher.update(extra_train_cfg)

    # if cfg_teacher["train_ic_mode"] == "random_shell_advantage":
    #     if train_role == "att": 
    #         cfg_teacher["r_att_min"] = 0.0
    #         cfg_teacher["train_ic_mode"] = "random_shell"

    DISTILL = bool(cfg_teacher.get("distill", False))

    build_dyn(cfg_teacher)

    # dynamics_config = cfg_teacher["dyn"]
    # print(dynamics_config["Ad"])
    # print(dynamics_config["Bd"])

    # raise("Debug")

    if cfg_teacher["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"[{phase_name.upper()} TEACHER] device='cuda' but CUDA not available."
        )

    print(f"[{phase_name.upper()} TEACHER] Using device: {cfg_teacher['device']}")
    print(f"[{phase_name.upper()}] train_role={train_role}  distill={DISTILL}")

    cfg_teacher["checkpoint_dir"] = OUT_DIR
    cfg_teacher["checkpoint_prefix"] = phase_name + "_teacher"

    ppo_teacher, metrics_teacher, ckpt_info = train(cfg_teacher)

    teacher_ckpt = ckpt_info["last_ckpt_path"]
    teacher_ckpt_best = ckpt_info["best_ckpt_path"]

    print(f"[{phase_name.upper()} TEACHER] Using {role_lower} checkpoint: {teacher_ckpt}")
    print(
        f"[{phase_name.upper()} TEACHER] "
        f"best {ckpt_info['tracked_metric_name']}={ckpt_info['best_metric']:+.3f} "
        f"at update {ckpt_info['best_update']}"
    )

    metrics_path = os.path.join(OUT_DIR, f"train_metrics_{phase_name}_teacher.npz")
    np.savez(metrics_path, **metrics_teacher)
    print(f"[{phase_name.upper()} TEACHER] Saved metrics to {metrics_path}")

    try:
        del ppo_teacher, metrics_teacher
    except Exception:
        pass

    # =========================================================
    # STUDENT (UKF) via distillation
    # =========================================================
    student_out = None

    if DISTILL:
        cfg_student = config_for_train(
            attacker_mode=attacker_mode,
            train_role=train_role,
        )
        cfg_student["use_ukf"] = True
        cfg_student["seed"] = cfg_teacher["seed"] + 1

        if extra_train_cfg is not None:
            cfg_student.update(extra_train_cfg)

        build_dyn(cfg_student)

        if cfg_student["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"[{phase_name.upper()} STUDENT] device='cuda' but CUDA not available."
            )

        print(f"[{phase_name.upper()} STUDENT] Using device: {cfg_student['device']}")

        student_out = os.path.join(OUT_DIR, f"{phase_name}_ukf_student.pt")

        # NOTE:
        # distill_from_teacher() is currently defender-centric internally.
        # This unified wrapper preserves your current behavior, but if you want
        # truly symmetric attacker distillation, distill_from_teacher() itself
        # should be generalized later.
        student, metrics_student = distill_from_teacher(
            cfg_student,
            teacher_ckpt,
            out_path=student_out,
        )

        print(f"[{phase_name.upper()} STUDENT] Distilled UKF student saved to {student_out}")

        distill_metrics_path = os.path.join(
            OUT_DIR, f"distill_metrics_{phase_name}_student.npz"
        )
        np.savez(distill_metrics_path, **metrics_student)
        print(f"[{phase_name.upper()} STUDENT] Saved distillation metrics to {distill_metrics_path}")

        try:
            del student, metrics_student
        except Exception:
            pass

    meta = {
        "train_role": train_role,
        "teacher_ckpt": teacher_ckpt,
        "teacher_best_metric_name": ckpt_info["tracked_metric_name"],
        "teacher_best_metric": ckpt_info["best_metric"],
        "teacher_best_update": ckpt_info["best_update"],
        "teacher_best_ckpt_path": ckpt_info["best_ckpt_path"],
        "teacher_last_ckpt_path": ckpt_info["last_ckpt_path"],
        "student_ckpt": student_out,
        "distill_enabled": DISTILL,
    }

    return teacher_ckpt, student_out, meta


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



    # =============================================================
# Main: stepped training (Def₀ -> Att₁ -> Def₁) + defender distillation
# =============================================================
if __name__ == "__main__":

    do_phase_0 = False
    do_phase_1 = True
    do_phase_2 = True

    def0_teacher_ckpt = "Training_Policy/def0_teacher.pt"
    att1_teacher_ckpt = "Training_Policy/att1_teacher.pt"
 

    OUT_DIR = "Training_Policy"
    os.makedirs(OUT_DIR, exist_ok=True)

    PLOTS_ROOT = os.path.join(OUT_DIR, "Plots")
    PLOTS_DEF0 = os.path.join(PLOTS_ROOT, "def0")
    PLOTS_ATT1 = os.path.join(PLOTS_ROOT, "att1")
    PLOTS_DEF1 = os.path.join(PLOTS_ROOT, "def1")
    PLOTS_COMP = os.path.join(PLOTS_ROOT, "comparisons")
    for d in [PLOTS_DEF0, PLOTS_ATT1, PLOTS_DEF1, PLOTS_COMP]:
        os.makedirs(d, exist_ok=True)

    runlog = RunLogger(OUT_DIR, filename="run_manifest.json")

    cfg_distillation = config_for_train(
        attacker_mode="rl",   # attacker is RL now
        train_role="att",     # PPO will only update attacker
    )
    DISTILL = cfg_distillation["distill"]


    cfg_training_log = config_for_train(
        attacker_mode="train",
        train_role="rl",
    )

    runlog.set_config("training config used", cfg_training_log)

    # save the exact distillation config you started with

    # runlog.set_config("cfg_distillation", cfg_distillation)

    phase0_extra = {
        "gamma": 0.990,
        "hit_buffer_def": 0.25,
    }

    # =========================================================
    # PHASE 0: Defender₀ vs rule-based attacker (teacher + distill)
    # =========================================================
    if do_phase_0 is True:
        print("\n===== PHASE 0: Train DEFENDER_0 vs RULE attacker =====")
        with runlog.stage(
            "PHASE0_train_def0",
            phase_name="def0",
            attacker_mode="rule",
            extra_train_cfg=phase0_extra,
        ) as st:
            def0_teacher_ckpt, def0_student_ckpt, def0_meta = train_with_distill(
                phase_name="def0",
                attacker_mode="rule",
                train_role="def",
                extra_train_cfg=phase0_extra,
            )
            st["outputs"] = def0_meta

        # def0_teacher_ckpt = "Training_Policy/def0_teacher.pt"

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

        plot_ic_support_from_cfg(
            cfg_training_log,
            n_scenes=30000,
            seed=123,
            title="def0 feasible IC support",
            out_path=os.path.join(PLOTS_DEF0, "def0_ic_support.png"),
            show=False,
        )

        plot_ic_used_from_npz(
            os.path.join(OUT_DIR, "ic_samples_def0_teacher.npz"),
            cfg_training_log,
            title="def0 ICs actually used during training",
            out_path=os.path.join(PLOTS_DEF0, "def0_ic_used.png"),
            show=False,
        )


    # =========================================================
    # PHASE 1: Attacker₁ vs fixed Defender₀ (teacher only, for now)
    # =========================================================
    if do_phase_1 is True:

        print("\n===== PHASE 1: Train ATTACKER_1 vs frozen DEFENDER_0 =====")
        phase1_extra = {
            "def_ckpt_path": def0_teacher_ckpt,
            "freeze_defender": True,
            "train_ic_mode": "random_shell",
            "r_att_min": 0.0,
            # NEW:
            # "opp_mix": {
            #     "modes": ["none", "def0", "weak"],
            #     "probs": [0.1, 0.8, 0.1],     # must sum to 1
            #     "resample": "episode",           # "episode" or "never"
            #     "weak_scale": 0.25,              # 0.0 -> basically none, 1.0 -> full def0
            #     "weak_noise_std": 0.00,          # optional additive Gaussian in action space
            # },
        }

        with runlog.stage(
            "PHASE1_train_att1",
            phase_name="att1",
            attacker_mode="rl",
            extra_train_cfg=phase1_extra,
        ) as st:
            att1_teacher_ckpt, att1_student_ckpt, att1_meta = train_with_distill(
                phase_name="att1",
                attacker_mode="rl",
                train_role="att",
                extra_train_cfg=phase1_extra,
            )
            st["outputs"] = att1_meta

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

        plot_ic_support_from_cfg(
            cfg_training_log,
            n_scenes=30000,
            seed=123,
            title="att1 feasible IC support",
            out_path=os.path.join(PLOTS_ATT1, "att1_ic_support.png"),
            show=False,
        )

        plot_ic_used_from_npz(
            os.path.join(OUT_DIR, "ic_samples_att1_teacher.npz"),
            cfg_training_log,
            title="att1 ICs actually used during training",
            out_path=os.path.join(PLOTS_ATT1, "att1_ic_used.png"),
            show=False,
        )


    # =========================================================
    # PHASE 2: Defender₁ vs frozen Attacker₁ (teacher + distill)
    # =========================================================
    if do_phase_2 is True:

        print("\n===== PHASE 2: Train DEFENDER_1 vs frozen ATTACKER_1 =====")
        phase2_extra = {"att_ckpt_path": att1_teacher_ckpt,
                        "def_ckpt_path": def0_teacher_ckpt,
                        "freeze_attacker": True,
                        }

        with runlog.stage(
            "PHASE2_train_def1",
            phase_name="def1",
            attacker_mode="rl",
            extra_train_cfg=phase2_extra,
        ) as st:
            def1_teacher_ckpt, def1_student_ckpt, def1_meta = train_with_distill(
                phase_name="def1",
                attacker_mode="rl",
                train_role="def",
                extra_train_cfg=phase2_extra,
            )
            st["outputs"] = def1_meta

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

        plot_ic_support_from_cfg(
            cfg_training_log,
            n_scenes=30000,
            seed=123,
            title="def1 feasible IC support",
            out_path=os.path.join(PLOTS_DEF1, "def1_ic_support.png"),
            show=False,
        )

        plot_ic_used_from_npz(
            os.path.join(OUT_DIR, "ic_samples_def1_teacher.npz"),
            cfg_training_log,
            title="def1 ICs actually used during training",
            out_path=os.path.join(PLOTS_DEF1, "def1_ic_used.png"),
            show=False,
        )

    # =========================================================
    # Multi phase Plotting 
    # =========================================================
    if do_phase_0 or do_phase_2:
        plot_compare_phases(
            [
                ("def0_teacher", os.path.join(OUT_DIR, "train_metrics_def0_teacher.npz"), "R_def_mean"),
                ("def1_teacher", os.path.join(OUT_DIR, "train_metrics_def1_teacher.npz"), "R_def_mean"),
            ],
            metric=None,
            ylabel="Mean defender return",
            title="Defender return across phases",
            smooth="ema",
            smooth_param=0.2,
            show=False,
            out_dir=PLOTS_COMP,
            filename="compare__R_def_mean__def0_vs_def1.png",
        )

    if do_phase_0 or do_phase_1 or do_phase_2:
        plot_compare_phases(
            [
                ("def0_teacher", os.path.join(OUT_DIR, "train_metrics_def0_teacher.npz"), "R_def_mean"),
                ("att1_teacher", os.path.join(OUT_DIR, "train_metrics_att1_teacher.npz"), "R_att_mean"),
                ("def1_teacher", os.path.join(OUT_DIR, "train_metrics_def1_teacher.npz"), "R_def_mean"),
            ],
            metric=None,
            ylabel="Mean return",
            title="Return across phases",
            smooth="ema",
            smooth_param=0.2,
            show=False,
            out_dir=PLOTS_COMP,
            filename="compare__R_mean__def0_vs_def1_vs_att1.png",
        )

    if not do_phase_0:
        def0_meta = None
    
    if not do_phase_1:
        att1_meta = None

    if not do_phase_2:
        def1_meta = None

        

    # record a final summary block too
    runlog.set_config("final_outputs", {
        "def0": def0_meta,
        "att1": att1_meta,
        "def1": def1_meta,
        "plots_root": PLOTS_ROOT,
    })

    print("Saved plots under:", PLOTS_ROOT)
    print(" -", PLOTS_DEF0)
    print(" -", PLOTS_ATT1)
    print(" -", PLOTS_DEF1)
    print(" -", PLOTS_COMP)

    print("\n===== ALL PHASES COMPLETE =====")
    if do_phase_0:
        print(f"Defender_0 teacher:  {def0_teacher_ckpt if def0_teacher_ckpt else '(skipped)'}")
        print(f"Defender_0 student:  {def0_student_ckpt if def0_student_ckpt else '(skipped)'}")
    else:
        print("Defender_0 training skipped")


    if do_phase_1:
        print(f"Attacker_1 teacher:  {att1_teacher_ckpt if att1_teacher_ckpt else '(skipped)'}")
        print(f"Attacker_1 student:  {att1_student_ckpt if att1_student_ckpt else '(skipped)'}")
    else:
        print("Attacker_1 training skipped")
    
    if do_phase_2:
        print(f"Defender_1 teacher:  {def1_teacher_ckpt if def1_teacher_ckpt else '(skipped)'}")
        print(f"Defender_1 student:  {def1_student_ckpt if def1_student_ckpt else '(skipped)'}")
    else:
        print("Defender_1 training skipped")
