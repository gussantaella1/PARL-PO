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
- Differentiable one-step ridge prior (DiffNash-style) blended into actor mean
- Minimal, single-process VecEnv for reproducible, fixed-length rollouts
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Callable, List

import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os  # <---- add this


# --- Single source of truth for config & dynamics ---
import config_rl
importlib.reload(config_rl)
from config_rl import config_for_train, config_for_eval, build_dyn


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
        self.cfg = cfg
        self.D = int(cfg["D"])
        self.dt = float(cfg["dt"])
        self.T  = int(cfg["T"])

        ar = cfg["arena"]
        if ar["type"] != "sphere":
            raise ValueError("Only 'sphere' arena is implemented.")
        self.center = np.array([ar["cx"], ar["cy"], (ar["cz"] if self.D == 3 else 0.0)], dtype=np.float32)[:self.D]
        self.radius = float(ar["r"])

        umax = float(cfg["umax"]) ; self.u_lo, self.u_hi = -umax, +umax

        Ad = cfg["dyn"]["Ad"]; Bd = cfg["dyn"]["Bd"]
        if Ad is None or Bd is None:
            raise ValueError("cfg['dyn']['Ad'] and ['Bd'] must be provided (call build_dyn(cfg)).")
        self.Ad = np.asarray(Ad, dtype=np.float32)
        self.Bd = np.asarray(Bd, dtype=np.float32)

        self.nx_agent = 2 * self.D
        self.nx_total = 2 * self.nx_agent
        self.act_dim = self.D

        x0 = np.asarray(cfg["x0"], dtype=float)
        assert x0.shape == (2, 2*self.D)
        self._x0 = x0

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

        # Training-only initial-condition randomization
        # Defaults to "fixed" if keys are absent (e.g., eval config)
        self.train_ic_mode = cfg.get("train_ic_mode", "fixed")
        self.train_ic_vmax = float(cfg.get("train_ic_vmax", 0.05))
        self.train_min_sep = float(cfg.get("train_min_sep", 1.0))


        self.state = None
        self.t = 0
        self._d2_prev = None

    def reset(self) -> np.ndarray:
        self.t = 0
        mode = self.train_ic_mode

        if mode == "fixed":
            # Original behavior: base x0 plus small jitter
            x0 = self._x0.copy()
            jit = self.cfg.get("x0_jitter", None)
            if jit:
                jp = float(jit.get("pos", 0.0))
                jv = float(jit.get("vel", 0.0))
                # defender
                x0[0, 0:self.D]        += np.random.uniform(-jp, jp, size=(self.D,))
                x0[0, self.D:2*self.D] += np.random.uniform(-jv, jv, size=(self.D,))
                # attacker
                x0[1, 0:self.D]        += np.random.uniform(-jp, jp, size=(self.D,))
                x0[1, self.D:2*self.D] += np.random.uniform(-jv, jv, size=(self.D,))

        elif mode == "random_shell":
            # Robust training: sample both agents anywhere within the sphere,
            # with a bias: defender near center, attacker nearer outer shell,
            # and enforce a minimum separation.

            R = self.radius
            v_max = self.train_ic_vmax
            min_sep = self.train_min_sep

            def sample_in_ball(r_min, r_max):
                # uniform direction
                d = np.random.normal(size=(self.D,))
                d /= (np.linalg.norm(d) + 1e-9)
                # radius (uniform in volume)
                u = np.random.rand()
                r = (r_min**3 + (r_max**3 - r_min**3) * u) ** (1.0 / 3.0)
                return self.center + r * d

            # You can tune these fractions; this uses inner half vs outer band
            r_def_min = 0.0
            r_def_max = 0.5 * R
            r_att_min = 0.4 * R
            r_att_max = 0.95 * R

            # sample until separation constraint satisfied
            for _ in range(1000):  # safety cap
                p1 = sample_in_ball(r_def_min, r_def_max)
                p2 = sample_in_ball(r_att_min, r_att_max)
                if np.linalg.norm(p2 - p1) >= min_sep:
                    break

            v1 = np.random.uniform(-v_max, v_max, size=(self.D,))
            v2 = np.random.uniform(-v_max, v_max, size=(self.D,))

            x0 = np.zeros_like(self._x0)
            x0[0, 0:self.D]        = p1
            x0[0, self.D:2*self.D] = v1
            x0[1, 0:self.D]        = p2
            x0[1, self.D:2*self.D] = v2

        else:
            raise ValueError(f"Unknown train_ic_mode='{mode}'")

        self.state = np.concatenate([x0[0], x0[1]])

        # initialize d2_prev as before
        p1, v1, p2, v2 = self._unpack(self.state)
        d2_raw = float(np.dot(p2 - self.center, p2 - self.center))
        self._d2_prev = d2_raw / (self.radius**2)

        return self._obs()


    def step(self, a1_env: np.ndarray, a2_env: np.ndarray) -> Tuple[np.ndarray, float, float, bool, dict]:
        a1 = np.clip(np.asarray(a1_env, float), self.u_lo, self.u_hi)
        a2 = np.clip(np.asarray(a2_env, float), self.u_lo, self.u_hi)

        # propagate
        self.state = self._plant_step(self.state, a1, a2)
        self.t += 1

        p1, v1, p2, v2 = self._unpack(self.state)

        # normalized squared distances to center
        d2_raw = float(np.dot(p2 - self.center, p2 - self.center))
        d1_raw = float(np.dot(p1 - self.center, p1 - self.center))
        d2 = d2_raw / (self.radius**2)
        d1 = d1_raw / (self.radius**2)
        delta_d2 = d2 - (self._d2_prev if self._d2_prev is not None else d2)

        # relative distance (tiny zero-sum shaping)
        rel2 = float(np.dot((p2 - p1), (p2 - p1))) / (self.radius**2)

        # soft walls
        rho1 = np.linalg.norm(p1 - self.center) / self.radius
        rho2 = np.linalg.norm(p2 - self.center) / self.radius
        wall1 = ((max(0.0, rho1 - self.soft_wall))**2) * self.wallK
        wall2 = ((max(0.0, rho2 - self.soft_wall))**2) * self.wallK

        # defender radial velocity
        rhat1 = (p1 - self.center)
        rnorm = np.linalg.norm(rhat1) + 1e-9
        vrad1 = float(np.dot(v1, rhat1 / rnorm)) / self.radius  # 1/s
        k_vrad = self.k_vel * 3.0

        # rewards
        r1 = ( self.alpha * delta_d2
             + self.k_pos * d2
             - self.k_rel * rel2
             - self.k_cent * d1
             - self.k_vel * float(np.dot(v1, v1))
             - k_vrad * (vrad1**2)
             - self.lD * float(np.dot(a1, a1))
             - wall1 )

        r2 = (- self.alpha * delta_d2
             - self.k_pos * d2
             + self.k_rel * rel2
             - self.k_vel * float(np.dot(v2, v2))
             - self.lA * float(np.dot(a2, a2))
             - wall2 )

        # termination
        oob1 = (rho1 >= self.margin)
        oob2 = (rho2 >= self.margin)
        done = (oob1 or oob2) or (self.t >= self.T)
        if oob1: r1 -= self.wallK
        if oob2: r2 -= self.wallK
        if done:
            r1 += self.beta * d2
            r2 -= self.beta * d2
            r1 -= 0.10 * d1

        self._d2_prev = d2

        info = {
            "t": self.t, "d2_norm": d2, "d1_norm": d1,
            "oob_def": bool(oob1), "oob_att": bool(oob2),
        }
        return self._obs(), float(r1), float(r2), bool(done), info

    def _obs(self) -> np.ndarray:
        p1, v1, p2, v2 = self._unpack(self.state)
        obs = np.concatenate([p1 - self.center, p2 - self.center, p2 - p1, v1, v2])
        return obs.astype(np.float32)

    def _plant_step(self, s: np.ndarray, a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
        p1, v1, p2, v2 = self._unpack(s)
        x1 = np.concatenate([p1, v1]) ; x2 = np.concatenate([p2, v2])
        x1n = self.Ad @ x1 + self.Bd @ a1
        x2n = self.Ad @ x2 + self.Bd @ a2
        p1n, v1n = x1n[:self.D], x1n[self.D:]
        p2n, v2n = x2n[:self.D], x2n[self.D:]
        return np.concatenate([p1n, v1n, p2n, v2n])

    def _unpack(self, s: np.ndarray):
        D = self.D
        p1 = s[0:D];       v1 = s[D:2*D]
        p2 = s[2*D:3*D];   v2 = s[3*D:4*D]
        return p1, v1, p2, v2


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

    def step(self, a1_env: np.ndarray, a2_env: np.ndarray):
        obs_next = []
        r1, r2, done, info = [], [], [], []
        for i, e in enumerate(self.envs):
            o, R1, R2, d, inf = e.step(a1_env[i], a2_env[i])
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
        self.obs[t]  = obs
        self.act[t]  = act
        self.logp[t] = logp
        self.val[t]  = val
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

    def act(self,
            p1: np.ndarray, v1: np.ndarray,
            p2: np.ndarray, v2: np.ndarray) -> np.ndarray:
        # Center pull and repulsion
        uc = self.u_center(p2, v2)
        ur = self.u_repulse(p1, p2)

        # Compute separation to optionally *down-weight center* close in
        dist = float(np.linalg.norm(p2 - p1))
        if dist < self.min_sep:
            # Near defender: ignore center attraction
            w_center_eff = 0.0
        else:
            w_center_eff = self.w_center

        u = w_center_eff * uc + self.w_avoid * ur - self.w_damp * v2
        return np.clip(u, -self.umax, +self.umax)



# =============================================================
# DiffNash Layer & Actor-Critic
# =============================================================
class DiffNashLayer(nn.Module):
    """One-step ridge prior (center pull) for each agent; blended in actor mean."""
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.D = int(cfg["D"])
        self.dt = float(cfg["dt"])
        self.ridge = float(cfg.get("prior_ridge", 1e-2))

        self.register_buffer("Ad", torch.as_tensor(np.asarray(cfg["dyn"]["Ad"], np.float32), dtype=torch.float32))
        self.register_buffer("Bd", torch.as_tensor(np.asarray(cfg["dyn"]["Bd"], np.float32), dtype=torch.float32))

        P = np.hstack([np.eye(self.D, dtype=np.float32), np.zeros((self.D, self.D), dtype=np.float32)])
        self.register_buffer("P", torch.tensor(P, dtype=torch.float32))

        ar = cfg["arena"]
        c = np.array([ar["cx"], ar["cy"], (ar["cz"] if self.D==3 else 0.0)], dtype=np.float32)[:self.D]
        self.register_buffer("center", torch.tensor(c, dtype=torch.float32))

    def forward(self, obs: torch.Tensor, who: str):
        B, D = obs.shape[0], self.D
        dtype = obs.dtype
        Ad = self.Ad.to(dtype=dtype)
        Bd = self.Bd.to(dtype=dtype)
        P  = self.P.to(dtype=dtype)
        center = self.center.to(dtype=dtype)

        # obs = [p1c, p2c, rel, v1, v2]
        p1c = obs[:, 0:D]
        p2c = obs[:, D:2*D]
        v1  = obs[:, 3*D:4*D]
        v2  = obs[:, 4*D:5*D]
        p1 = p1c + center
        p2 = p2c + center

        # One-step next-position error relative to center
        x1 = torch.cat([p1, v1], dim=-1)
        x2 = torch.cat([p2, v2], dim=-1)
        E1 = (x1 @ Ad.T) @ P.T - center
        E2 = (x2 @ Ad.T) @ P.T - center
        F  = (Bd.T @ P.T).T

        FtF = F.T @ F
        lamI = self.ridge * torch.eye(D, dtype=dtype, device=obs.device)
        K = torch.linalg.solve(FtF + lamI, F.T)

        u_def_prior = -(K @ E1.T).T
        u_att_prior = -(K @ E2.T).T
        u_prior = u_def_prior if who == "def" else u_att_prior

        feats = torch.cat([p1c, p2c, v1, v2], dim=-1)  # (B, 4D)
        return feats, u_prior


class ActorCriticDiff(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg: Dict[str, Any]):
        super().__init__()
        hidden = 128
        self.layer = DiffNashLayer(cfg)
        self.pi = nn.Sequential(
            nn.Linear(4*cfg["D"], hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu_res = nn.Linear(hidden, act_dim)
        self.logstd = nn.Parameter(torch.full((act_dim,), -1.0))  # std ~ 0.37
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
    def act(self, obs_batch: torch.Tensor, who: str):
        if who == "att" and self.attacker_mode == "rule":
            # obs = [p1c, p2c, rel, v1, v2]
            B = obs_batch.shape[0]
            D = obs_batch.shape[-1] // 5
            p1c = obs_batch[:, 0:D].cpu().numpy()
            p2c = obs_batch[:, D:2*D].cpu().numpy()
            v1  = obs_batch[:, 3*D:4*D].cpu().numpy()
            v2  = obs_batch[:, 4*D:5*D].cpu().numpy()
            center = self.rule_ctrl.center
            acts = []
            for i in range(B):
                p1 = p1c[i] + center
                p2 = p2c[i] + center
                a = self.rule_ctrl.act(p1, v1[i], p2, v2[i])
                acts.append(a.astype(np.float32))
            a_np = np.stack(acts, axis=0)
            a_t  = torch.as_tensor(a_np, dtype=obs_batch.dtype, device=obs_batch.device)
            zero = torch.zeros(B, dtype=obs_batch.dtype, device=obs_batch.device)
            return a_t, zero, zero
        else:
            net = self.def_net if who == "def" else self.att_net
            return net.act(obs_batch, who, self.act_scale)

    def _update_one(self,
                    net: ActorCriticDiff, opt: optim.Optimizer,
                    obs: torch.Tensor, act_env: torch.Tensor,
                    old_logp: torch.Tensor, old_val: torch.Tensor,
                    adv: torch.Tensor, ret: torch.Tensor, who: str):
        B = obs.shape[0]
        idx = np.arange(B)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for st in range(0, B, self.mb_size):
                j = idx[st:st+self.mb_size]
                o = obs[j]; a = act_env[j]; lp_old = old_logp[j]; v_old = old_val[j]; A = adv[j]; R = ret[j]

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

    def update_both(self, buf_def: RolloutBuffer, buf_att: RolloutBuffer):
        advD, retD = compute_gae_from_buffer(buf_def, self.gamma, self.lam)
        advA, retA = compute_gae_from_buffer(buf_att, self.gamma, self.lam)
        oD, aD, lpD, vD, _, _ = buf_def.get()
        oA, aA, lpA, vA, _, _ = buf_att.get()
        self._update_one(self.def_net, self.def_opt, oD, aD, lpD, vD, advD, retD, who="def")
        self._update_one(self.att_net, self.att_opt, oA, aA, lpA, vA, advA, retA, who="att")


# =============================================================
# Training & Evaluation
# =============================================================
def train(cfg: Dict[str, Any]):
    set_seed(cfg["seed"])
    device = cfg["device"]

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
        "d1_mean": [],
        "d2_mean": [],   # <--- NEW
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
        if not rule_att:
            bufA = RolloutBuffer(obs_dim, act_dim, num_envs, steps_per_env, device)

        o = torch.as_tensor(vec.obs, dtype=torch.float32, device=device)
        ep_ret_def = np.zeros(num_envs, dtype=np.float64)
        ep_ret_att = np.zeros(num_envs, dtype=np.float64)

        for _ in range(steps_per_env):
            with torch.no_grad():
                a1, lp1, v1 = ppo.act(o, who="def")
                a2, lp2, v2 = ppo.act(o, who="att")
            o2_np, r1_np, r2_np, d_np, _ = vec.step(a1.cpu().numpy(), a2.cpu().numpy())
            o2 = torch.as_tensor(o2_np, dtype=torch.float32, device=device)
            r1 = torch.as_tensor(r1_np, dtype=torch.float32, device=device)
            r2 = torch.as_tensor(r2_np, dtype=torch.float32, device=device)
            d  = torch.as_tensor(d_np,  dtype=torch.float32, device=device)

            bufD.add(o, a1, lp1, v1, r1, d)
            if not rule_att:
                bufA.add(o, a2, lp2, v2, r2, d)

            ep_ret_def += r1_np
            ep_ret_att += r2_np
            o = o2

        with torch.no_grad():
            next_v_def = ppo.def_net.value(o)
        bufD.finalize(next_v_def)

        if not rule_att:
            with torch.no_grad():
                next_v_att = ppo.att_net.value(o)
            bufA.finalize(next_v_att)
            ppo.update_both(bufD, bufA)
        else:
            ppo.update_defender_only(bufD)

        if upd % log_every == 0:
            R_def_mean = ep_ret_def.mean()
            R_att_mean = ep_ret_att.mean()

            with torch.no_grad():
                flat_obs = bufD.obs.reshape(-1, obs_dim)
                distD = ppo.def_net.dist(flat_obs, who="def")
                muD = distD.mean.abs().mean().item()
                stdD = distD.stddev.mean().item()

                # obs = [p1c, p2c, rel, v1, v2]
                Dcfg = cfg["D"]
                p1c = flat_obs[:, :Dcfg]
                p2c = flat_obs[:, Dcfg:2*Dcfg]

                d1_mean = p1c.pow(2).sum(-1).mean().sqrt().item()
                d2_mean = p2c.pow(2).sum(-1).mean().sqrt().item()

            # grab learning rates (assuming two param groups: policy+logstd and value)
            lr_pi = ppo.def_opt.param_groups[0]["lr"]
            lr_vf = ppo.def_opt.param_groups[-1]["lr"]

            # ---- store in metrics ----
            metrics["update"].append(upd)
            metrics["R_def_mean"].append(R_def_mean)
            metrics["R_att_mean"].append(R_att_mean)
            metrics["muD_abs_mean"].append(muD)
            metrics["stdD_mean"].append(stdD)
            metrics["d1_mean"].append(d1_mean)
            metrics["d2_mean"].append(d2_mean)
            metrics["lr_pi"].append(lr_pi)
            metrics["lr_vf"].append(lr_vf)

            # keep your printout if you like
            print(f"[update {upd:05d}] R_def_mean={R_def_mean:+.3f}  R_att_mean={R_att_mean:+.3f}  (batch={num_envs*steps_per_env})")
            print(f"   [def] |mu|_mean={muD:.3e}  std_mean={stdD:.3e}")
            print(f"   approx <||p1-center||> ≈ {d1_mean:.3f}")
            print(f"   approx <||p2-center||> ≈ {d2_mean:.3f}")



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
# Main
# =============================================================
if __name__ == "__main__":
    # Build training config from the single shared source
    cfg = config_for_train(
        # Example optional overrides:
        # attacker_mode="rule",   # train with fixed attacker per prof guidance (default)
        # umax=0.02,
        # T=120,
    )
    build_dyn(cfg)

    # Device check
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cfg['device']='cuda' but CUDA is not available.")
    print(f"Using device: {cfg['device']}")

    # ---- Output directory for policies + plots ----
    OUT_DIR = "Training_Policy"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Train
    ppo, metrics = train(cfg)

    # Eval config from same source; safer defaults applied inside config_for_eval
    cfg_eval = config_for_eval(
        attacker_mode=cfg.get("attacker_mode", "rule"),
        umax=cfg["umax"],
        T=cfg["T"],
    )
    build_dyn(cfg_eval)

    # Rollout
    trajs = evaluate(ppo, cfg_eval, episodes=2)

    # Metrics
    ar = cfg_eval["arena"]
    D = cfg_eval["D"]
    center = np.array([ar["cx"], ar["cy"], (ar["cz"] if D == 3 else 0.0)], dtype=float)[:D]
    R = float(ar["r"])
    m = rollout_metrics(trajs[0]["states"], center, R)
    print(f"[metrics] d2_T={m['d2_norm'][-1]:.3f}  d1_med={np.median(m['d1_norm']):.3f}  rel2_med={np.median(m['rel2_norm']):.3f}")

    # ---- Plot training curves ----
    updates = np.array(metrics["update"], dtype=float)

    plt.figure()
    plt.plot(updates, metrics["R_def_mean"], label="Defender return")
    plt.plot(updates, metrics["R_att_mean"], label="Attacker return")
    plt.xlabel("Update")
    plt.ylabel("Episode return (mean over envs)")
    plt.title("Training returns")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "train_returns.png"), dpi=200)

    plt.figure()
    plt.plot(updates, metrics["muD_abs_mean"], label="|μ_def| mean")
    plt.plot(updates, metrics["stdD_mean"], label="σ_def mean")
    plt.xlabel("Update")
    plt.ylabel("Policy stats")
    plt.title("Defender policy mean/std")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "policy_stats.png"), dpi=200)

    plt.figure()
    plt.plot(updates, metrics["d1_mean"])
    plt.xlabel("Update")
    plt.ylabel("<||p1 - center||>")
    plt.title("Avg defender distance to center")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "def_center_dist.png"), dpi=200)

    plt.figure()
    plt.plot(updates, metrics["d2_mean"])
    plt.xlabel("Update")
    plt.ylabel("<||p2 - center||>")
    plt.title("Avg attacker distance to center")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "att_center_dist.png"), dpi=200)


    plt.figure()
    plt.plot(updates, metrics["lr_pi"], label="policy LR")
    plt.plot(updates, metrics["lr_vf"], label="value LR")
    plt.xlabel("Update")
    plt.ylabel("Learning rate")
    plt.title("Learning rates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "learning_rates.png"), dpi=200)

    # (Optional) save raw metrics for later analysis
    np.savez(os.path.join(OUT_DIR, "train_metrics.npz"), **metrics)

    # Save policies
    torch.save(ppo.def_net.state_dict(), os.path.join(OUT_DIR, "ppo_def.pt"))
    if ppo.attacker_mode == "rl":
        torch.save(ppo.att_net.state_dict(), os.path.join(OUT_DIR, "ppo_att.pt"))
    print(
        "Saved: "
        + os.path.join(OUT_DIR, "ppo_def.pt")
        + (", " + os.path.join(OUT_DIR, "ppo_att.pt") if ppo.attacker_mode == "rl" else "")
    )
