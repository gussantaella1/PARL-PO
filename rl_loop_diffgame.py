"""
rl_loop_diffgame.py
====================
Two–agent PPO (Defender vs Attacker) with a differentiable game‑theory prior head
("DiffNashLayer"). This version fixes dtype mismatches by making all registered
buffers float32 and casting them to the incoming tensor dtype inside `forward`.

Observation layout must match training:
    obs = [p1-center, p2-center, (p2-p1), v1, v2]

The DiffNashLayer computes a one‑step ridge solution for the attacker that
minimizes next‑step distance to the center under linear HCW dynamics, and a
sign‑flipped version for the defender (pushes the attacker outward). These serve
as action priors. The policy learns a residual on top of the prior.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Callable, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---- import your HCW helpers ----
from dyn_models import hcw_mean_motion, hcw_discrete_mats, as_numpy_const

# =========================
# CONFIG
# =========================
CONFIG: Dict[str, Any] = {
    "seed": 42,
    "device": "cpu",            # "cuda" if available

    # --- dynamics ---
    "D": 3,
    "dt": 0.1,
    "T": 45,
    "dynamics": "hcw",
    "hcw": {"mu": 3.986004418e14, "r0": 6_371_000.0 + 400_000.0},

    # arena (centered)
    "arena": {"type": "sphere", "cx": 0.0, "cy": 0.0, "cz": 0.0, "r": 20.0},
    "arena_terminate_margin": 1.0,

    # action bound (LVLH acceleration [m/s^2])
    "umax": 0.02,

    # initial states: shape (2, 2*D) as [px,py,pz, vx,vy,vz] per agent
    "x0": np.array([
        [ 2.5,  0.0, 0.0,  -0.02, 0.00,  0.000],  # defender
        [-19.9, 0.0, 0.0,   +0.02, 0.00, 0.000],  # attacker
    ], dtype=float),
    "x0_jitter": {"pos": 0.5, "vel": 0.01},

    # PPO rollout settings (fixed length batches)
    "num_envs": 8,
    "steps_per_env": 256,  # batch = 2048 / update
    "total_updates": 300,

    # PPO hyperparams
    "gamma": 0.97,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "policy_lr": 3e-4,
    "value_lr": 1e-3,
    "train_epochs": 10,
    "minibatch_size": 1024,
    "entropy_coef": 0.02,
    "value_coef": 0.5,
    "max_grad_norm": 1.0,
    "log_every": 10,

    # dynamics matrices are filled in __main__
    "dyn": {"Ad": None, "Bd": None},

    # shaping knobs (normalized distance)
    "dense_coef": 0.03,     # α: (d2_next - d2_prev)
    "term_coef": 1.0,       # β: terminal ± d_T^2
    "step_pos_coef": 0.01,  # +k d^2 for def, −k d^2 for att
    "step_vel_coef": 0.0005,
    "effort_def": 0.01,
    "effort_att": 0.01,
    "wall_penalty": 2.0,

    # DiffNash prior ridge regularizer (λ I)
    "prior_ridge": 1e-2,
    # Weight to blend prior and learned residual mean in the policy output
    "prior_blend": 1.0,    # 1.0 = full prior + residual; set 0 to disable
}

# =========================
# Utils
# =========================
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

# =========================
# Environment
# =========================
class Env:
    """
    State per agent: [px,py,pz,vx,vy,vz]; s = concat(s1, s2). Actions are accelerations in R^D.

    Rewards per step (with d^2 normalized by radius^2):
      r_def =  +α Δd^2 + k_pos d^2 − k_vel ||v_D||^2 − λ_D ||a_D||^2 + wall
      r_att =  −α Δd^2 − k_pos d^2 − k_vel ||v_A||^2 − λ_A ||a_A||^2 − wall

    Terminal bonus at done: +β d_T^2 (def), −β d_T^2 (att)
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.D = int(cfg["D"])
        self.dt = float(cfg["dt"])
        self.T = int(cfg["T"])

        ar = cfg["arena"]
        if ar["type"] == "sphere":
            c = np.array([ar["cx"], ar["cy"], (ar["cz"] if self.D == 3 else 0.0)], dtype=np.float32)
            self.radius = float(ar["r"])
        elif ar["type"] == "box":
            cx = 0.5 * (ar["xmin"] + ar["xmax"]) ; cy = 0.5 * (ar["ymin"] + ar["ymax"]) ; cz = 0.5 * (ar["zmin"] + ar["zmax"]) if self.D==3 else 0.0
            c = np.array([cx, cy, cz], dtype=np.float32)
            hx = 0.5 * (ar["xmax"] - ar["xmin"]) ; hy = 0.5 * (ar["ymax"] - ar["ymin"]) ; hz = 0.5 * (ar.get("zmax",0.0) - ar.get("zmin",0.0)) if self.D==3 else 0.0
            self.radius = float(np.sqrt(hx*hx + hy*hy + hz*hz))
        else:
            raise ValueError(f"Unsupported arena type: {ar['type']!r}")
        self.center = c[:self.D]

        umax = float(cfg["umax"]) ; self.u_lo, self.u_hi = -umax, +umax

        Ad = cfg["dyn"]["Ad"] ; Bd = cfg["dyn"]["Bd"]
        if Ad is None or Bd is None:
            raise ValueError("CONFIG['dyn']['Ad'] and ['Bd'] must be provided.")
        self.Ad = np.asarray(Ad, dtype=np.float32)
        self.Bd = np.asarray(Bd, dtype=np.float32)

        self.nx_agent = 2 * self.D
        self.nx_total = 2 * self.nx_agent
        self.act_dim = self.D

        x0 = np.asarray(cfg["x0"], dtype=float)
        assert x0.shape == (2, 2*self.D)
        self._x0 = x0

        self.alpha = float(cfg["dense_coef"]) ; self.beta = float(cfg["term_coef"]) ;
        self.k_pos = float(cfg.get("step_pos_coef", 0.0)) ; self.k_vel = float(cfg.get("step_vel_coef", 0.0))
        self.lD = float(cfg["effort_def"]) ; self.lA = float(cfg["effort_att"]) ; self.wallK = float(cfg["wall_penalty"]) ; self.margin = float(cfg["arena_terminate_margin"]) ;

        self.state = None ; self.t = 0 ; self._d2_prev = None

    def reset(self) -> np.ndarray:
        self.t = 0
        x0 = self._x0.copy()
        jit = self.cfg.get("x0_jitter", None)
        if jit:
            jp = float(jit.get("pos", 0.0)) ; jv = float(jit.get("vel", 0.0))
            x0[0, 0:self.D] += np.random.uniform(-jp, jp, size=(self.D,))
            x0[0, self.D:2*self.D] += np.random.uniform(-jv, jv, size=(self.D,))
            x0[1, 0:self.D] += np.random.uniform(-jp, jp, size=(self.D,))
            x0[1, self.D:2*self.D] += np.random.uniform(-jv, jv, size=(self.D,))
        self.state = np.concatenate([x0[0], x0[1]])
        p1, v1, p2, v2 = self._unpack(self.state)
        d2_raw = float(np.dot(p2 - self.center, p2 - self.center))
        self._d2_prev = d2_raw / (self.radius**2)
        return self._obs()

    def step(self, a1_env: np.ndarray, a2_env: np.ndarray) -> Tuple[np.ndarray, float, float, bool, dict]:
        a1 = np.clip(np.asarray(a1_env, float), self.u_lo, self.u_hi)
        a2 = np.clip(np.asarray(a2_env, float), self.u_lo, self.u_hi)

        self.state = self._plant_step(self.state, a1, a2)
        self.t += 1

        p1, v1, p2, v2 = self._unpack(self.state)
        d2_raw = float(np.dot(p2 - self.center, p2 - self.center))
        d2 = d2_raw / (self.radius**2)
        delta_d2 = d2 - (self._d2_prev if self._d2_prev is not None else d2)

        rho1 = np.linalg.norm(p1 - self.center) / self.radius
        rho2 = np.linalg.norm(p2 - self.center) / self.radius
        wall_soft = 0.0
        for rho in (rho1, rho2):
            if rho > 0.7:
                wall_soft += (rho - 0.7)**2 * self.wallK

        r1 = ( self.alpha * delta_d2 + self.k_pos * d2 - self.k_vel * float(np.dot(v1, v1)) - self.lD * float(np.dot(a1, a1)) + wall_soft)
        r2 = (- self.alpha * delta_d2 - self.k_pos * d2 - self.k_vel * float(np.dot(v2, v2)) - self.lA * float(np.dot(a2, a2)) - wall_soft)

        oob = (np.linalg.norm(p1 - self.center) >= self.margin*self.radius) or (np.linalg.norm(p2 - self.center) >= self.margin*self.radius)
        done = oob or (self.t >= self.T)
        if oob:
            r1 += +self.wallK ; r2 += -self.wallK
        if done:
            r1 += self.beta * d2 ; r2 -= self.beta * d2

        self._d2_prev = d2
        info = {"t": self.t, "d2_norm": d2, "oob": bool(oob)}
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

# =========================
# Simple vectorized env (single-process)
# =========================
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
        obs_next = [] ; r1, r2, done, info = [], [], [], []
        for i, e in enumerate(self.envs):
            o, R1, R2, d, inf = e.step(a1_env[i], a2_env[i])
            if d:
                o = e.reset()
            obs_next.append(o) ; r1.append(R1) ; r2.append(R2) ; done.append(d) ; info.append(inf)
        self.obs = np.stack(obs_next, axis=0)
        return self.obs, np.array(r1), np.array(r2), np.array(done, dtype=np.float32), info

# =========================
# PPO storage
# =========================
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

# =========================
# Differentiable Nash prior layer
# =========================
class DiffNashLayer(nn.Module):
    """Game‑theory prior: one‑step ridge control for attacker (pull to center)
    and sign‑flipped for defender (push attacker out). Returns (features, mean,
    prior_used) so the policy can blend residuals with the prior.
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.D = int(cfg["D"]) ; self.dt = float(cfg["dt"]) ; self.ridge = float(cfg.get("prior_ridge", 1e-2))
        # Register buffers as float32 to avoid float64/float32 matmul errors
        self.register_buffer("Ad", torch.as_tensor(np.asarray(cfg["dyn"]["Ad"], np.float32), dtype=torch.float32))
        self.register_buffer("Bd", torch.as_tensor(np.asarray(cfg["dyn"]["Bd"], np.float32), dtype=torch.float32))
        P = np.hstack([np.eye(self.D, dtype=np.float32), np.zeros((self.D, self.D), dtype=np.float32)])
        self.register_buffer("P", torch.tensor(P, dtype=torch.float32))
        ar = cfg["arena"]
        c = np.array([ar["cx"], ar["cy"], (ar["cz"] if self.D == 3 else 0.0)], dtype=np.float32)[:self.D]
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

        # Attacker kinematics (next position) without control: P @ (Ad @ x2)
        x2 = torch.cat([p2, v2], dim=-1)                   # (B,2D)
        E = (x2 @ Ad.T) @ P.T - center                     # (B,D)
        F = (Bd.T @ P.T).T                                 # (D,D)

        # Ridge minimizer for attacker:  u* = - (F^T F + λ I)^-1 F^T E
        FtF = F.T @ F
        lamI = self.ridge * torch.eye(D, dtype=dtype, device=obs.device)
        K = torch.linalg.solve(FtF + lamI, F.T)            # (D,D)
        u_att_prior = -(K @ E.T).T                         # (B,D)
        # Defender: sign‑flip to push away (heuristic prior)
        u_def_prior = -u_att_prior

        u_prior = u_def_prior if who == "def" else u_att_prior

        # Simple features for residual head
        feats = torch.cat([p1c, p2c, v1, v2], dim=-1)      # (B,4D)
        return feats, u_prior, {"E": E, "F": F}

# =========================
# Actor‑Critic with DiffNash prior (squashed Gaussian)
# =========================
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
        self.prior_blend = float(cfg.get("prior_blend", 1.0))

    def dist(self, obs: torch.Tensor, who: str):
        feats, u_prior, _ = self.layer(obs, who)           # (B,4D), (B,D)
        h = self.pi(feats)
        mu_res = self.mu_res(h)
        mu = mu_res + self.prior_blend * u_prior
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

# =========================
# PPO core
# =========================
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

        self.def_net = ActorCriticDiff(obs_dim, act_dim, cfg).to(device)
        self.att_net = ActorCriticDiff(obs_dim, act_dim, cfg).to(device)

        self.def_opt = optim.Adam([
            {"params": list(self.def_net.pi.parameters()) + list(self.def_net.mu_res.parameters()), "lr": cfg["policy_lr"]},
            {"params": [self.def_net.logstd], "lr": cfg["policy_lr"] * 0.5},
            {"params": self.def_net.vf.parameters(), "lr": cfg["value_lr"]},
        ])
        self.att_opt = optim.Adam([
            {"params": list(self.att_net.pi.parameters()) + list(self.att_net.mu_res.parameters()), "lr": cfg["policy_lr"]},
            {"params": [self.att_net.logstd], "lr": cfg["policy_lr"] * 0.5},
            {"params": self.att_net.vf.parameters(), "lr": cfg["value_lr"]},
        ])

    @torch.no_grad()
    def act(self, obs_batch: torch.Tensor, who: str):
        net = self.def_net if who=="def" else self.att_net
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

    def update(self, buf_def: RolloutBuffer, buf_att: RolloutBuffer):
        advD, retD = compute_gae_from_buffer(buf_def, self.gamma, self.lam)
        advA, retA = compute_gae_from_buffer(buf_att, self.gamma, self.lam)
        oD, aD, lpD, vD, _, _ = buf_def.get()
        oA, aA, lpA, vA, _, _ = buf_att.get()
        self._update_one(self.def_net, self.def_opt, oD, aD, lpD, vD, advD, retD, who="def")
        self._update_one(self.att_net, self.att_opt, oA, aA, lpA, vA, advA, retA, who="att")

# =========================
# Training loop with fixed rollouts
# =========================
def train(cfg: Dict[str, Any]):
    set_seed(cfg["seed"]) ; device = cfg["device"]

    def make_env():
        return Env(cfg)

    num_envs = int(cfg.get("num_envs", 8))
    steps_per_env = int(cfg.get("steps_per_env", 256))
    total_updates = int(cfg.get("total_updates", 300))
    log_every = int(cfg.get("log_every", 10))

    vec = VecEnv(make_env, num_envs)
    obs_dim = vec.obs.shape[1] ; act_dim = int(cfg["D"])
    ppo = PPO(obs_dim, act_dim, cfg, device=device)

    for upd in range(1, total_updates + 1):
        bufD = RolloutBuffer(obs_dim, act_dim, num_envs, steps_per_env, device)
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
            bufA.add(o, a2, lp2, v2, r2, d)

            ep_ret_def += r1_np ; ep_ret_att += r2_np ; o = o2

        with torch.no_grad():
            next_v_def = ppo.def_net.value(o)
            next_v_att = ppo.att_net.value(o)
        bufD.finalize(next_v_def)
        bufA.finalize(next_v_att)

        ppo.update(bufD, bufA)

        if upd % log_every == 0:
            print(f"[update {upd:05d}] R_def_mean={ep_ret_def.mean():+.3f}  R_att_mean={ep_ret_att.mean():+.3f}  (batch={num_envs*steps_per_env})")
            with torch.no_grad():
                distD = ppo.def_net.dist(bufD.obs.reshape(-1, obs_dim), who="def")
                distA = ppo.att_net.dist(bufA.obs.reshape(-1, obs_dim), who="att")
                muD = distD.mean.abs().mean().item() ; stdD = distD.stddev.mean().item()
                muA = distA.mean.abs().mean().item() ; stdA = distA.stddev.mean().item()
                print(f"   [def] |mu|_mean={muD:.3e}  std_mean={stdD:.3e} | [att] |mu|_mean={muA:.3e}  std_mean={stdA:.3e}")

    print("Training finished.")
    return ppo

# =========================
# Rollout for sanity check (single environment)
# =========================
def evaluate(ppo: PPO, cfg: Dict[str, Any], episodes: int = 2):
    env = Env(cfg)
    trajs = []
    for _ in range(episodes):
        obs = env.reset()
        states = [env.state.copy()] ; actions = [] ; infos = [] ; done = False
        while not done:
            o_t = torch.as_tensor(obs[None, :], dtype=torch.float32, device=ppo.device)
            with torch.no_grad():
                a1, _, _ = ppo.act(o_t, who="def")
                a2, _, _ = ppo.act(o_t, who="att")
            a1_np = a1.squeeze(0).cpu().numpy() ; a2_np = a2.squeeze(0).cpu().numpy()
            obs, r1, r2, done, info = env.step(a1_np, a2_np)
            states.append(env.state.copy()) ; actions.append((a1_np.copy(), a2_np.copy())) ; infos.append(info)
        trajs.append({"states": np.stack(states), "actions": actions, "infos": infos})
    return trajs

# =========================
# Main
# =========================
if __name__ == "__main__":
    if CONFIG["dynamics"].lower() != "hcw":
        raise ValueError("This script expects CONFIG['dynamics'] == 'hcw'.")

    hcw_block = CONFIG["hcw"]
    n = hcw_mean_motion(hcw_block)
    Ad_mx, Bd_mx = hcw_discrete_mats(float(n), float(CONFIG["dt"]))
    # Make upstream matrices float32 as well
    CONFIG["dyn"]["Ad"] = as_numpy_const(Ad_mx).astype(np.float32)
    CONFIG["dyn"]["Bd"] = as_numpy_const(Bd_mx).astype(np.float32)

    if CONFIG["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CONFIG['device']='cuda' but CUDA is not available.")
    device = CONFIG["device"]

    print(f"Using device: {device}")
    ppo = train(CONFIG)
    _ = evaluate(ppo, CONFIG, episodes=2)

    torch.save(ppo.def_net.state_dict(), "ppo_def.pt")
    torch.save(ppo.att_net.state_dict(), "ppo_att.pt")
    print("Saved RL policies: ppo_def_diff.pt, ppo_att_diff.pt")
