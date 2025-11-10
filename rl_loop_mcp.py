# rl_loop_teacher.py — MCP-anchored PPO (teacher-ready, residual & NI-gap hooks)
# =============================================================================
# Two-agent PPO (Defender vs. Attacker) for HCW LVLH pursuit–evasion with
# optional game-theoretic regularizers:
#   • MCP/KKT residual penalty   (lambda_mcp)
#   • Exploitability / NI-gap    (lambda_gap)
#   • (Optional) behavior cloning pretrain from PATH teacher (BC)
#
# Notes:
# - This file keeps the original observation contract used in prior runs:
#       obs = [p1 - c, p2 - c, (p2 - p1), v1, v2]   (all R^D blocks)
#   We can reconstruct (p1, v1, p2, v2) given the known center c.
# - The MCP residual here starts minimal (bound-violation + simple stationarity
#   on quadratic effort). You can progressively replace it with a full
#   differentiable KKT/MCP stack per your PATH model once ready.
# - The NI-gap is computed via a cheap, single-step proximal best-response
#   (local, H=1). You can increase steps/horizon later.
# - A BC pretrain hook is provided; pass a teacher dataset path in CONFIG to use.
# =============================================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Callable, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- your dynamics helpers (unchanged imports) ---
from dyn_models import hcw_mean_motion, hcw_discrete_mats, as_numpy_const


# =========================
# CONFIG (sane defaults)
# =========================
CONFIG: Dict[str, Any] = {
    "seed": 42,
    "device": "cpu",            # "cuda" if available

    # --- dynamics ---
    "D": 3,
    "dt": 0.1,
    "T": 120,                    # steps per episode
    "dynamics": "hcw",
    "hcw": {"mu": 3.986004418e14, "r0": 6_371_000.0 + 400_000.0},

    # arena (centered)
    "arena": {"type": "sphere", "cx": 0.0, "cy": 0.0, "cz": 0.0, "r": 20.0},
    "arena_terminate_margin": 1.0,

    # action bound (per-axis clip ±umax) — LVLH acceleration [m/s^2]
    "umax": 1.0,

    # initial states [px,py,pz,vx,vy,vz] per agent
    "x0": np.array([
        [  2.5, 0.0, 0.0,  -0.02, 0.00,  0.000],  # defender
        [ -19.9, 0.0, 0.0,  +0.02, 0.00,  0.000],  # attacker
    ], dtype=float),
    "x0_jitter": {"pos": 0.5, "vel": 0.01},

    # PPO rollout settings (fixed length batches)
    "num_envs": 8,
    "steps_per_env": 256,       # total batch = 8 * 256 = 2048 transitions / update
    "total_updates": 300,

    # PPO hyperparams
    "gamma": 0.995, #0.97,
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
    "dense_coef": 0.03,         # α: weight on (d2_{t+1} - d2_t)
    "term_coef": 1.0,           # β: terminal ±d_T^2
    "step_pos_coef": 0.01,      # +k d^2 for def, -k d^2 for att
    "step_vel_coef": 0.0005,    # penalize speed
    "effort_def": 0.01,         # λ_D: ||a_def||^2
    "effort_att": 0.01,         # λ_A: ||a_att||^2
    "wall_penalty": 2.0,        # soft wall pressure / OOB bonus

    # ---- game-theoretic regularizers ----
    "use_mcp": True,
    "lambda_mcp": 0.1,          # weight for MCP residual penalty
    "use_gap": True,
    "lambda_gap": 0.1,          # weight for NI-gap (exploitability) term
    "gap_br_steps": 1,          # steps in proximal BR
    "gap_br_alpha": 0.1,        # step size in proximal BR

    # ---- optional BC pretrain ----
    # If provided, we will run a quick BC warmup before PPO updates.
    # Teacher file must contain tensors: obs [N,obs_dim], u_def [N,D], u_att [N,D]
    "bc_teacher_path": None,    # e.g., "teacher.pt"
    "bc_epochs": 10,
    "bc_batch": 8192,
}


# =========================
# Utils & numerics
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
# Environment (unchanged dynamics + shaped rewards)
# =========================
class Env:
    """
    State per agent: [px,py,pz,vx,vy,vz] (first D are pos, next D are vel)
    Full state s = concat(s1, s2). Actions are accelerations in R^D.

    Rewards per step (with d^2 normalized by radius^2):
      r_def =  α * (d^2_{t+1} - d^2_t) + k_pos * d_t^2 - k_vel * ||v_D||^2 - λ_D ||a_D||^2 + wall_soft
      r_att = -α * (d^2_{t+1} - d^2_t) - k_pos * d_t^2 - k_vel * ||v_A||^2 - λ_A ||a_A||^2 - wall_soft

    Terminal bonus at done:
      +β * d_T^2  (defender),  -β * d_T^2 (attacker)
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.D = int(cfg["D"])
        self.dt = float(cfg["dt"])
        self.T = int(cfg["T"])

        # center & radius
        ar = cfg["arena"]
        if ar["type"] == "sphere":
            c = np.array([ar["cx"], ar["cy"], (ar["cz"] if self.D == 3 else 0.0)], float)
            self.radius = float(ar["r"])
        elif ar["type"] == "box":
            cx = 0.5 * (ar["xmin"] + ar["xmax"])
            cy = 0.5 * (ar["ymin"] + ar["ymax"])
            cz = 0.5 * (ar["zmin"] + ar["zmax"]) if self.D == 3 else 0.0
            c = np.array([cx, cy, cz], float)
            hx = 0.5 * (ar["xmax"] - ar["xmin"])
            hy = 0.5 * (ar["ymax"] - ar["ymin"])
            hz = 0.5 * (ar["zmax"] - ar["zmin"]) if self.D == 3 else 0.0
            self.radius = float(np.sqrt(hx*hx + hy*hy + hz*hz))
        else:
            raise ValueError(f"Unsupported arena type: {ar['type']!r}")
        self.center = c[:self.D]

        # action bound
        umax = float(cfg["umax"])
        self.u_lo, self.u_hi = -umax, +umax

        # dynamics matrices
        Ad = cfg["dyn"]["Ad"]; Bd = cfg["dyn"]["Bd"]
        if Ad is None or Bd is None:
            raise ValueError("CONFIG['dyn']['Ad'] and ['Bd'] must be provided.")
        self.Ad = np.asarray(Ad, float)
        self.Bd = np.asarray(Bd, float)

        self.nx_agent = 2 * self.D  # [p, v]
        self.nx_total = 2 * self.nx_agent
        self.act_dim = self.D

        # initial state
        x0 = np.asarray(cfg["x0"], float)
        assert x0.shape == (2, 2*self.D), f"x0 must be shape (2, {2*self.D}), got {x0.shape}"
        self._x0 = x0

        # reward knobs
        self.alpha = float(cfg["dense_coef"])
        self.beta  = float(cfg["term_coef"])
        self.k_pos = float(cfg.get("step_pos_coef", 0.0))
        self.k_vel = float(cfg.get("step_vel_coef", 0.0))
        self.lD    = float(cfg["effort_def"])
        self.lA    = float(cfg["effort_att"])
        self.wallK = float(cfg["wall_penalty"])
        self.margin = float(cfg["arena_terminate_margin"])

        self.state = None
        self.t = 0
        self._d2_prev = None

    # ---------- Public API ----------
    def reset(self) -> np.ndarray:
        self.t = 0
        # base
        x0 = self._x0.copy()
        # jitter
        jit = self.cfg.get("x0_jitter", None)
        if jit:
            jp = float(jit.get("pos", 0.0))
            jv = float(jit.get("vel", 0.0))
            x0[0, 0:self.D]          += np.random.uniform(-jp, jp, size=(self.D,))
            x0[0, self.D:2*self.D]   += np.random.uniform(-jv, jv, size=(self.D,))
            x0[1, 0:self.D]          += np.random.uniform(-jp, jp, size=(self.D,))
            x0[1, self.D:2*self.D]   += np.random.uniform(-jv, jv, size=(self.D,))
        self.state = np.concatenate([x0[0], x0[1]])
        # initialize d^2 tracker (normalized to radius^2)
        p1, v1, p2, v2 = self._unpack(self.state)
        d2_raw = float(np.dot(p2 - self.center, p2 - self.center))
        self._d2_prev = d2_raw / (self.radius**2)
        return self._obs()

    def step(self, a1_env: np.ndarray, a2_env: np.ndarray) -> Tuple[np.ndarray, float, float, bool, dict]:
        # env receives clamped actions
        a1 = np.clip(np.asarray(a1_env, float), self.u_lo, self.u_hi)
        a2 = np.clip(np.asarray(a2_env, float), self.u_lo, self.u_hi)

        # propagate
        self.state = self._plant_step(self.state, a1, a2)
        self.t += 1

        # new normalized squared distance for attacker
        p1, v1, p2, v2 = self._unpack(self.state)
        d2_raw = float(np.dot(p2 - self.center, p2 - self.center))
        d2 = d2_raw / (self.radius**2)
        delta_d2 = d2 - (self._d2_prev if self._d2_prev is not None else d2)

        # soft wall pressure (before termination): push away from wall
        rho1 = np.linalg.norm(p1 - self.center) / self.radius
        rho2 = np.linalg.norm(p2 - self.center) / self.radius
        wall_soft = 0.0
        for rho in (rho1, rho2):
            if rho > 0.7:
                wall_soft += (rho - 0.7)**2 * self.wallK

        # per-step shaping + effort
        r1 = ( self.alpha * delta_d2
             + self.k_pos * d2
             - self.k_vel * float(np.dot(v1, v1))
             - self.lD    * float(np.dot(a1, a1))
             + wall_soft)

        r2 = (- self.alpha * delta_d2
             - self.k_pos * d2
             - self.k_vel * float(np.dot(v2, v2))
             - self.lA    * float(np.dot(a2, a2))
             - wall_soft)

        # early termination
        oob = (np.linalg.norm(p1 - self.center) >= self.margin*self.radius) or \
              (np.linalg.norm(p2 - self.center) >= self.margin*self.radius)
        done = oob or (self.t >= self.T)

        if oob:
            r1 += +self.wallK
            r2 += -self.wallK

        # terminal bonus
        if done:
            r1 += self.beta * d2
            r2 -= self.beta * d2

        # update tracker
        self._d2_prev = d2

        info = {"t": self.t, "d2_norm": d2, "oob": bool(oob)}
        return self._obs(), float(r1), float(r2), bool(done), info

    # ---------- Observations (fixed relative) ----------
    def _obs(self) -> np.ndarray:
        p1, v1, p2, v2 = self._unpack(self.state)
        obs = np.concatenate([
            p1 - self.center,            # defender pos rel center
            p2 - self.center,            # attacker pos rel center
            p2 - p1,                     # relative position att - def
            v1, v2,                      # velocities (world/LVLH frame)
        ])
        return obs.astype(np.float32)

    # ---------- Physics ----------
    def _plant_step(self, s: np.ndarray, a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
        p1, v1, p2, v2 = self._unpack(s)
        x1 = np.concatenate([p1, v1])
        x2 = np.concatenate([p2, v2])
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
        # Keep center for obs→state reconstruction
        self.center = self.envs[0].center.copy()
        self.D = self.envs[0].D

    def reset(self):
        o = [e.reset() for e in self.envs]
        self.obs = np.stack(o, axis=0)
        return self.obs

    def step(self, a1_env: np.ndarray, a2_env: np.ndarray):
        # a*_env: (N, act_dim)
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


# =========================
# PPO storage (fixed rollout length)
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
        self.next_val = next_val  # (N,)

    def get(self):
        # Flatten T,N -> B
        B = self.T * self.N
        obs  = self.obs.reshape(B, -1)
        act  = self.act.reshape(B, -1)
        logp = self.logp.reshape(B)
        val  = self.val.reshape(B)
        rew  = self.rew.reshape(B)
        done = self.done.reshape(B)
        return obs, act, logp, val, rew, done


def compute_gae_from_buffer(buf: RolloutBuffer, gamma: float, lam: float):
    # Vectorized GAE over T with masks
    T, N = buf.T, buf.N
    device = buf.device
    adv = torch.zeros(T, N, device=device)
    lastgaelam = torch.zeros(N, device=device)
    next_val = buf.next_val  # (N,)

    for t in reversed(range(T)):
        nonterminal = 1.0 - buf.done[t]         # (N,)
        next_v = next_val if t == T-1 else buf.val[t+1]
        delta = buf.rew[t] + gamma * next_v * nonterminal - buf.val[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam

    ret = adv + buf.val
    A = adv.reshape(T*N)
    R = ret.reshape(T*N)
    # normalize advantages
    A = (A - A.mean()) / (A.std() + 1e-8)
    return A, R


# =========================
# Actor-Critic (squashed Gaussian)
# =========================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.logstd = nn.Parameter(torch.full((act_dim,), -1.0))  # std ~ 0.37
        self.vf = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def dist(self, obs: torch.Tensor):
        h = self.pi(obs)
        mu = self.mu(h)
        std = self.logstd.exp()
        return torch.distributions.Normal(mu, std)

    def value(self, obs: torch.Tensor):
        return self.vf(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, act_scale: float):
        dist = self.dist(obs)
        u_raw = dist.rsample()
        a_env = squash_action(u_raw, act_scale)
        logp  = logprob_squashed(dist, u_raw)
        val   = self.value(obs)
        return a_env, logp, val


# =========================
# Game-theoretic regularizers (minimal viable versions)
# =========================
# ---- obs → (p1,v1,p2,v2) torch reconstruction ----

def obs_to_state(o: torch.Tensor, center: torch.Tensor, D: int) -> Tuple[torch.Tensor, ...]:
    """Given obs=[p1c, p2c, (p2-p1), v1, v2], reconstruct p1,p2 (world) and v1,v2.
    o: (B, 5D), center: (D,) or (B,D)
    Returns: p1, v1, p2, v2  (each (B,D))
    """
    p1c = o[:, 0:D]
    p2c = o[:, D:2*D]
    v1  = o[:, 3*D:4*D]
    v2  = o[:, 4*D:5*D]
    if center.dim() == 1:
        c = center.unsqueeze(0).expand_as(p1c)
    else:
        c = center
    p1 = p1c + c
    p2 = p2c + c
    return p1, v1, p2, v2


# ---- stage costs approximations (single-step, consistent with Env reward signs) ----

def stage_cost_def(o: torch.Tensor, aD: torch.Tensor, aA: torch.Tensor, center: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    D = int(cfg["D"]); r = float(cfg["arena"]["r"]) ; k_pos = float(cfg.get("step_pos_coef", 0.0))
    k_vel = float(cfg.get("step_vel_coef", 0.0))
    lD = float(cfg["effort_def"])
    p1,v1,p2,v2 = obs_to_state(o, torch.as_tensor(center, device=o.device, dtype=o.dtype), D)
    d2 = ((p2 - torch.as_tensor(center, device=o.device, dtype=o.dtype))**2).sum(dim=-1) / (r*r)
    # Def wants large d2 and small effort/vel  => cost := -(k_pos*d2) + k_vel*||v1||^2 + lD*||aD||^2
    return -(k_pos * d2) + k_vel * (v1*v1).sum(-1) + lD * (aD*aD).sum(-1)


def stage_cost_att(o: torch.Tensor, aD: torch.Tensor, aA: torch.Tensor, center: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    D = int(cfg["D"]); r = float(cfg["arena"]["r"]) ; k_pos = float(cfg.get("step_pos_coef", 0.0))
    k_vel = float(cfg.get("step_vel_coef", 0.0))
    lA = float(cfg["effort_att"])
    p1,v1,p2,v2 = obs_to_state(o, torch.as_tensor(center, device=o.device, dtype=o.dtype), D)
    d2 = ((p2 - torch.as_tensor(center, device=o.device, dtype=o.dtype))**2).sum(dim=-1) / (r*r)
    # Att wants small d2 and small effort/vel  => cost := +(k_pos*d2) + k_vel*||v2||^2 + lA*||aA||^2
    return +(k_pos * d2) + k_vel * (v2*v2).sum(-1) + lA * (aA*aA).sum(-1)


# ---- proximal best-response (1–K steps of prox-gradient) ----

def prox_best_response(o: torch.Tensor, u_other: torch.Tensor, center: torch.Tensor,
                       cfg: Dict[str, Any], player: str = "def",
                       steps: int = 1, alpha: float = 0.1) -> torch.Tensor:
    umax = float(cfg["umax"]) ; D = int(cfg["D"]) ; B = o.shape[0]
    u = torch.zeros(B, D, device=o.device, dtype=o.dtype, requires_grad=True)
    for _ in range(steps):
        if player == "def":
            J = stage_cost_def(o, u, u_other, center, cfg)
        else:
            J = stage_cost_att(o, u_other, u, center, cfg)
        # proximal term around current u (identity here — you can set a moving anchor)
        J = J.mean()  # scalar
        g, = torch.autograd.grad(J, u, create_graph=False)
        u = (u - alpha * g).clamp_(-umax, +umax).detach().requires_grad_(True)
    return u.detach()


# ---- MCP residual (minimal: bound violation + simple stationarity proxy) ----

def mcp_residual(o: torch.Tensor, aD: torch.Tensor, aA: torch.Tensor,
                 center: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    """Return per-sample residual vector R; loss uses ||R||^2.
    This minimal version includes:
      (i) stationarity proxy on quadratic effort: grad_u (λ||u||^2) = 2λ u
      (ii) soft bound violations for box constraints u ∈ [-umax, +umax]
    Replace with your full KKT/MCP residuals once your torch-ified FONC is ready.
    """
    lD = float(cfg["effort_def"]) ; lA = float(cfg["effort_att"]) ; umax = float(cfg["umax"]) ;
    # stationarity residuals (should be ~0 at optimum for pure quadratic effort)
    r_stat_D = 2.0 * lD * aD
    r_stat_A = 2.0 * lA * aA
    # bound residuals
    over_D = torch.relu(torch.abs(aD) - umax)
    over_A = torch.relu(torch.abs(aA) - umax)
    R = torch.cat([r_stat_D, r_stat_A, over_D, over_A], dim=-1)
    return R


def mcp_residual_loss(o: torch.Tensor, aD: torch.Tensor, aA: torch.Tensor,
                      center: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    R = mcp_residual(o, aD, aA, center, cfg)  # (B, 4D)
    return (R*R).sum(dim=-1)                  # (B,)


# ---- NI-gap (exploitability) ----

def ni_gap_terms(o: torch.Tensor, aD: torch.Tensor, aA: torch.Tensor,
                 center: torch.Tensor, cfg: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    steps = int(cfg.get("gap_br_steps", 1))
    alpha = float(cfg.get("gap_br_alpha", 0.1))
    # BR for attacker vs current defender
    uA_br = prox_best_response(o, aD, center, cfg, player="att", steps=steps, alpha=alpha)
    # BR for defender vs current attacker
    uD_br = prox_best_response(o, aA, center, cfg, player="def", steps=steps, alpha=alpha)

    Jd = stage_cost_def(o, aD, aA, center, cfg)
    Jd_br = stage_cost_def(o, uD_br, aA, center, cfg)
    Ja = stage_cost_att(o, aD, aA, center, cfg)
    Ja_br = stage_cost_att(o, aD, uA_br, center, cfg)

    # gap := max(0, J(π) - J(BR, opponent)) (since lower cost is better)
    gap_def = (Jd - Jd_br).clamp_min(0.0)
    gap_att = (Ja - Ja_br).clamp_min(0.0)
    return gap_def, gap_att


# =========================
# PPO core (with GT regularizers)
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
        self.act_scale = float(cfg["umax"])  # maps to env clip
        self.v_clip_eps = 0.2

        self.center = None  # filled from VecEnv on train()
        self.D = int(cfg["D"])  # convenience for residual functions

        self.use_mcp = bool(cfg.get("use_mcp", False))
        self.lambda_mcp = float(cfg.get("lambda_mcp", 0.0))
        self.use_gap = bool(cfg.get("use_gap", False))
        self.lambda_gap = float(cfg.get("lambda_gap", 0.0))

        self.def_net = ActorCritic(obs_dim, act_dim).to(device)
        self.att_net = ActorCritic(obs_dim, act_dim).to(device)

        self.def_opt = optim.Adam([
            {"params": list(self.def_net.pi.parameters()) + list(self.def_net.mu.parameters()), "lr": cfg["policy_lr"]},
            {"params": [self.def_net.logstd], "lr": cfg["policy_lr"] * 0.5},
            {"params": self.def_net.vf.parameters(), "lr": cfg["value_lr"]},
        ])
        self.att_opt = optim.Adam([
            {"params": list(self.att_net.pi.parameters()) + list(self.att_net.mu.parameters()), "lr": cfg["policy_lr"]},
            {"params": [self.att_net.logstd], "lr": cfg["policy_lr"] * 0.5},
            {"params": self.att_net.vf.parameters(), "lr": cfg["value_lr"]},
        ])

    @torch.no_grad()
    def act(self, obs_batch: torch.Tensor, who: str):
        net = self.def_net if who=="def" else self.att_net
        return net.act(obs_batch, self.act_scale)

    def _update_one(self,
                    net: ActorCritic, opt: optim.Optimizer,
                    obs: torch.Tensor, act_env: torch.Tensor,
                    old_logp: torch.Tensor, old_val: torch.Tensor,
                    adv: torch.Tensor, ret: torch.Tensor,
                    # extras for GT terms
                    obs_other: torch.Tensor, act_other: torch.Tensor,
                    center: torch.Tensor, cfg: Dict[str, Any], who: str):
        B = obs.shape[0]
        idx = np.arange(B)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for st in range(0, B, self.mb_size):
                j = idx[st:st+self.mb_size]
                o = obs[j]; a = act_env[j]; lp_old = old_logp[j]; v_old = old_val[j]; A = adv[j]; R = ret[j]
                o_opp = obs_other[j]; a_opp = act_other[j]

                dist = net.dist(o)
                # invert squash to raw u for consistent log-prob
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

                # ---- GT regularizers (symmetric, but we add them on each player's update) ----
                if self.use_mcp and self.lambda_mcp > 0.0:
                    if who == "def":
                        aD, aA = a, a_opp
                    else:
                        aD, aA = a_opp, a
                    Lm = mcp_residual_loss(o, aD, aA, center, cfg).mean()
                    loss = loss + self.lambda_mcp * Lm

                if self.use_gap and self.lambda_gap > 0.0:
                    if who == "def":
                        aD, aA = a, a_opp
                        gap_def, _ = ni_gap_terms(o, aD, aA, center, cfg)
                        Lg = gap_def.mean()
                    else:
                        aD, aA = a_opp, a
                        _, gap_att = ni_gap_terms(o, aD, aA, center, cfg)
                        Lg = gap_att.mean()
                    loss = loss + self.lambda_gap * Lg

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), self.max_grad)
                opt.step()

    def update(self, buf_def: RolloutBuffer, buf_att: RolloutBuffer, center: torch.Tensor, cfg: Dict[str, Any]):
        # compute advantages/returns
        advD, retD = compute_gae_from_buffer(buf_def, self.gamma, self.lam)
        advA, retA = compute_gae_from_buffer(buf_att, self.gamma, self.lam)

        # flatten obs/act/logp/val
        oD, aD, lpD, vD, _, _ = buf_def.get()
        oA, aA, lpA, vA, _, _ = buf_att.get()

        # defender update (other = attacker)
        self._update_one(self.def_net, self.def_opt,
                         oD, aD, lpD, vD, advD, retD,
                         obs_other=oA, act_other=aA,
                         center=center, cfg=cfg, who="def")

        # attacker update (other = defender)
        self._update_one(self.att_net, self.att_opt,
                         oA, aA, lpA, vA, advA, retA,
                         obs_other=oD, act_other=aD,
                         center=center, cfg=cfg, who="att")


# =========================
# Behavior Cloning (optional warm start)
# =========================
@torch.no_grad()
def _infer_actions_from_net(net: ActorCritic, obs: torch.Tensor, act_scale: float) -> torch.Tensor:
    dist = net.dist(obs)
    u = torch.tanh(dist.mean) * act_scale
    return u


def bc_pretrain(ppo: PPO, dataset_path: str, epochs: int = 10, batch: int = 8192, act_scale: float = 1.0, device: str = "cpu"):
    import torch.nn.functional as F
    data = torch.load(dataset_path, map_location=device)
    obs = data["obs"].to(device)
    uD  = torch.clamp(data["u_def"].to(device), -act_scale, act_scale)
    uA  = torch.clamp(data["u_att"].to(device), -act_scale, act_scale)

    def _run(net: ActorCritic, opt: optim.Optimizer, u_star: torch.Tensor):
        B = obs.shape[0]
        for _ in range(epochs):
            idx = torch.randperm(B, device=device)
            for st in range(0, B, batch):
                j = idx[st:st+batch]
                dist = net.dist(obs[j])
                u_pred = torch.tanh(dist.mean) * act_scale
                loss = F.mse_loss(u_pred, u_star[j])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()

    _run(ppo.def_net, ppo.def_opt, uD)
    _run(ppo.att_net, ppo.att_opt, uA)


# =========================
# Training loop with fixed rollouts
# =========================
def train(cfg: Dict[str, Any]):
    set_seed(cfg["seed"])
    device = cfg["device"]

    # factory for env instances (shares cfg)
    def make_env():
        return Env(cfg)

    num_envs = int(cfg.get("num_envs", 8))
    steps_per_env = int(cfg.get("steps_per_env", 256))
    total_updates = int(cfg.get("total_updates", 300))
    log_every = int(cfg.get("log_every", 10))

    vec = VecEnv(make_env, num_envs)

    # dims
    obs_dim = vec.obs.shape[1]
    act_dim = int(cfg["D"])  # same as LVLH accel dimension

    ppo = PPO(obs_dim, act_dim, cfg, device=device)
    ppo.center = torch.as_tensor(vec.center, device=device, dtype=torch.float32)

    # optional BC warmstart
    if cfg.get("bc_teacher_path"):
        print(f"[BC] Pretraining from teacher: {cfg['bc_teacher_path']}")
        bc_pretrain(ppo, cfg["bc_teacher_path"], epochs=int(cfg.get("bc_epochs", 10)),
                    batch=int(cfg.get("bc_batch", 8192)), act_scale=float(cfg["umax"]), device=device)

    for upd in range(1, total_updates + 1):
        bufD = RolloutBuffer(obs_dim, act_dim, num_envs, steps_per_env, device)
        bufA = RolloutBuffer(obs_dim, act_dim, num_envs, steps_per_env, device)

        # rollout
        o = torch.as_tensor(vec.obs, dtype=torch.float32, device=device)
        ep_ret_def = np.zeros(num_envs, dtype=np.float64)
        ep_ret_att = np.zeros(num_envs, dtype=np.float64)

        for t in range(steps_per_env):
            with torch.no_grad():
                a1, lp1, v1 = ppo.act(o, who="def")  # (N,D), (N), (N)
                a2, lp2, v2 = ppo.act(o, who="att")

            # step envs (numpy)
            o2_np, r1_np, r2_np, d_np, _ = vec.step(a1.cpu().numpy(), a2.cpu().numpy())
            o2 = torch.as_tensor(o2_np, dtype=torch.float32, device=device)
            r1 = torch.as_tensor(r1_np, dtype=torch.float32, device=device)
            r2 = torch.as_tensor(r2_np, dtype=torch.float32, device=device)
            d  = torch.as_tensor(d_np,  dtype=torch.float32, device=device)

            bufD.add(o, a1, lp1, v1, r1, d)
            bufA.add(o, a2, lp2, v2, r2, d)

            ep_ret_def += r1_np
            ep_ret_att += r2_np
            o = o2

        # bootstrap values on final obs
        with torch.no_grad():
            next_v_def = ppo.def_net.value(o)   # (N,)
            next_v_att = ppo.att_net.value(o)

        bufD.finalize(next_v_def)
        bufA.finalize(next_v_att)

        # update
        ppo.update(bufD, bufA, center=ppo.center, cfg=cfg)

        if upd % log_every == 0:
            print(f"[update {upd:05d}] "
                  f"R_def_mean={ep_ret_def.mean():+.3f}  R_att_mean={ep_ret_att.mean():+.3f}  "
                  f"(batch={num_envs*steps_per_env})")
            # quick on-batch policy stats (means of mu/std)
            with torch.no_grad():
                distD = ppo.def_net.dist(bufD.obs.reshape(-1, obs_dim))
                distA = ppo.att_net.dist(bufA.obs.reshape(-1, obs_dim))
                muD = distD.mean.abs().mean().item()
                stdD = distD.stddev.mean().item()
                muA = distA.mean.abs().mean().item()
                stdA = distA.stddev.mean().item()
                print(f"   [def] |mu|_mean={muD:.3e}  std_mean={stdD:.3e} | [att] |mu|_mean={muA:.3e}  std_mean={stdA:.3e}")

    print("Training finished.")
    return ppo


# =========================
# Rollout for sanity check (single environment)
# =========================
@torch.no_grad()
def evaluate(ppo: PPO, cfg: Dict[str, Any], episodes: int = 2):
    env = Env(cfg)
    device = ppo.def_net.logstd.device
    center = torch.as_tensor(env.center, device=device, dtype=torch.float32)
    trajs = []
    for _ in range(episodes):
        obs = env.reset()
        states = [env.state.copy()]
        actions = []
        infos = []
        done = False
        while not done:
            o_t = torch.as_tensor(obs[None, :], dtype=torch.float32, device=device)
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


# =========================
# Main
# =========================
if __name__ == "__main__":
    # Build Ad/Bd from HCW
    if CONFIG["dynamics"].lower() != "hcw":
        raise ValueError("This script expects CONFIG['dynamics'] == 'hcw'.")

    hcw_block = CONFIG["hcw"]
    n = hcw_mean_motion(hcw_block)  # requires mu & r0 if no explicit n
    Ad_mx, Bd_mx = hcw_discrete_mats(float(n), float(CONFIG["dt"]))
    CONFIG["dyn"]["Ad"] = as_numpy_const(Ad_mx)
    CONFIG["dyn"]["Bd"] = as_numpy_const(Bd_mx)

    # device choice
    if CONFIG["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CONFIG['device']='cuda' but CUDA is not available.")
    device = CONFIG["device"]

    print(f"Using device: {device}")
    ppo = train(CONFIG)
    _ = evaluate(ppo, CONFIG, episodes=2)

    torch.save(ppo.def_net.state_dict(), "ppo_def.pt")
    torch.save(ppo.att_net.state_dict(), "ppo_att.pt")
    print("Saved RL policies: ppo_def.pt, ppo_att.pt")
