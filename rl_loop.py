# rl_loop.py
# Multi-agent PPO (P1 defender, P2 attacker) with terminal objective
# + potential-based progress shaping at each step.
#
# Strict OG-config version:
# - Uses ONLY OG config keys (D, dt, T, dynamics/hcw, arena, umax, x0, episodes, PPO hparams).
# - No fallbacks/defaults. Fails fast if keys are missing.
# - Initial state comes strictly from x0 (no random init).
# - Observations are fixed to "relative": [p1-c, p2-c, (p2-p1), v1, v2].
# - Dynamics require Ad/Bd (constructed from HCW in __main__).
# - Reward normalization is explicit: CONFIG["reward_norm"] in {"arena_r2","by_d0","none"}.
# - Optional effort penalties are explicit: CONFIG["effort_reg_def"], CONFIG["effort_reg_att"].

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dyn_models import hcw_mean_motion, hcw_discrete_mats, as_numpy_const


# =========================
# CONFIG (OG schema)
# =========================
CONFIG: Dict[str, Any] = {
    "seed": 42,
    "device": "cpu",          # "cuda" if available

    # --- OG config keys ---
    "D": 3,
    "dt": 0.1,
    "T": 80,                  # (↑) more control steps per episode
    "dynamics": "hcw",
    "hcw": {"mu": 3.986004418e14, "r0": 6_371_000.0 + 400_000.0},  # or set {"n": 0.0011}

    # arena -> center (STRICT)
    "arena": {"type": "sphere", "cx": 0.0, "cy": 0.0, "cz": 0.0, "r": 30.0},

    # action bound (STRICT, scalar => per-axis clip ±umax)
    "umax": 1e-2,             # (↑) m/s^2 in LVLH to make actions matter

    # initial states (STRICT): shape (2, 2*D) as [px,py,pz, vx,vy,vz] per agent
    "x0": np.array([
        [ 1.0,  0.0, 0.0,  -0.02, 0.00,  0.000],  # agent 1 (def)
        [-29.9, 0.0, 0.0,   0.00, 0.00, -2.000],  # agent 2 (att)
    ], dtype=float),

    # training
    "episodes": 10000,

    # PPO hyperparams
    "gamma": 1.0,             # terminal objective (shaping is potential-based)
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "policy_lr": 5e-4,        # (↑) slightly stronger
    "value_lr": 1.5e-3,       # (↑)
    "train_epochs": 6,
    "minibatch_size": 1024,
    "entropy_coef": 0.02,     # (↑) encourage exploration a bit more
    "value_coef": 0.5,
    "max_grad_norm": 1.0,
    "log_every": 20,

    # reward shaping (STRICT: explicit keys only)
    #   "arena_r2": divides d^2 by arena radius^2 (sphere) or half-diagonal^2 (box)
    #   "by_d0":    divides d^2 by (initial distance)^2 (episode-relative)
    #   "none":     no normalization
    "reward_norm": "arena_r2",
    "effort_reg_def": 0.0,    # λ_def * ||a_def||^2 (applied at terminal step only)
    "effort_reg_att": 0.0,    # λ_att * ||a_att||^2 (applied at terminal step only)

    # progress-shaping coefficient (small)
    "progress_coef": 0.1,     # defender gets +k*Δ(d^2_norm), attacker gets -k*Δ(d^2_norm)

    # dynamics matrices will be filled in __main__
    "dyn": {"Ad": None, "Bd": None},
}


# =====================================================
# Utils
# =====================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


# =====================================================
# Actor-Critic
# =====================================================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.pi_body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        # (↑) larger initial std for more exploration
        self.logstd = nn.Parameter(torch.full((act_dim,), 0.3))  # init σ ≈ exp(0.3) ≈ 1.35

        self.vf = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def policy_dist(self, obs: torch.Tensor):
        h = self.pi_body(obs)
        mu = self.mu_head(h)
        std = self.logstd.exp()
        return torch.distributions.Normal(mu, std)

    def value(self, obs: torch.Tensor):
        return self.vf(obs).squeeze(-1)


# =====================================================
# PPO Storage
# =====================================================
@dataclass
class RolloutBuffer:
    obs: list
    act: list
    logp: list
    rew: list
    val: list
    done: list

    def __init__(self):
        self.obs, self.act, self.logp, self.rew, self.val, self.done = ([] for _ in range(6))

    def add(self, obs, act, logp, rew, val, done):
        self.obs.append(obs)
        self.act.append(act)
        self.logp.append(logp)
        self.rew.append(rew)
        self.val.append(val)
        self.done.append(done)

    def to_tensors(self, device="cpu"):
        def cat(x): return torch.as_tensor(np.array(x), dtype=torch.float32, device=device)
        return {
            "obs":  cat(self.obs),
            "act":  cat(self.act),
            "logp": cat(self.logp),
            "rew":  cat(self.rew),
            "val":  cat(self.val),
            "done": cat(self.done),
        }


def compute_gae(buf_tensors, gamma: float, lam: float, device="cpu"):
    rews = buf_tensors["rew"]
    vals = buf_tensors["val"]
    done = buf_tensors["done"]
    T = rews.shape[0]
    adv = torch.zeros_like(rews, device=device)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - done[t]
        next_value = vals[t+1] if t < T-1 else torch.tensor(0.0, device=device)
        delta = rews[t] + gamma * next_value * next_nonterminal - vals[t]
        lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = adv + vals
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret


# =====================================================
# Environment (strict)
# =====================================================
class Env:
    """
    State per agent: [px,py,pz,vx,vy,vz] (first D are pos, next D are vel)
    Full state s = concat(s1, s2). Actions are accelerations in R^D.
    Terminal reward uses explicit normalization & penalties according to CONFIG.
    Progress shaping provides small dense signal without changing the optimal policy.
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.D = int(cfg["D"])
        self.dt = float(cfg["dt"])
        self.T = int(cfg["T"])

        # --- center from arena (STRICT) ---
        ar = cfg["arena"]
        ar_type = ar["type"]
        if ar_type == "sphere":
            c = np.array([ar["cx"], ar["cy"], (ar["cz"] if self.D == 3 else 0.0)], float)
        elif ar_type == "box":
            cx = 0.5 * (ar["xmin"] + ar["xmax"])
            cy = 0.5 * (ar["ymin"] + ar["ymax"])
            cz = 0.5 * (ar["zmin"] + ar["zmax"]) if self.D == 3 else 0.0
            c = np.array([cx, cy, cz], float)
        else:
            raise ValueError(f"Unsupported arena type: {ar_type!r}")
        self.center = c[:self.D]

        # --- action bound (STRICT) ---
        umax = float(cfg["umax"])
        self.u_lo, self.u_hi = -umax, +umax

        # --- dynamics matrices (STRICT) ---
        Ad = cfg["dyn"]["Ad"]
        Bd = cfg["dyn"]["Bd"]
        if Ad is None or Bd is None:
            raise ValueError("CONFIG['dyn']['Ad'] and ['Bd'] must be provided.")
        self.Ad = np.asarray(Ad, float)
        self.Bd = np.asarray(Bd, float)

        self.nx_agent = 2 * self.D  # [p, v]
        self.nx_total = 2 * self.nx_agent
        self.act_dim = self.D

        # initial state strictly from x0
        x0 = np.asarray(cfg["x0"], float)
        assert x0.shape == (2, 2*self.D), f"x0 must be shape (2, {2*self.D}), got {x0.shape}"
        self._x0_flat = np.concatenate([x0[0], x0[1]])

        # reward config
        rn = cfg["reward_norm"]
        if rn not in ("arena_r2", "by_d0", "none"):
            raise ValueError("CONFIG['reward_norm'] must be one of {'arena_r2','by_d0','none'}")
        self.reward_norm = rn
        self.l_def = float(cfg.get("effort_reg_def", 0.0))
        self.l_att = float(cfg.get("effort_reg_att", 0.0))

        # progress shaping
        self.k_prog = float(cfg["progress_coef"])

        self.state = None
        self.t = 0
        self._d0 = None  # for 'by_d0' normalization
        self._r2 = None  # for 'arena_r2'
        self._p2_prev = None  # for progress shaping

        # precompute arena radius^2 for normalization if needed
        if self.reward_norm == "arena_r2":
            if ar_type == "sphere":
                self._r2 = float(ar["r"]) ** 2
            else:
                # box: use half-diagonal as "radius"
                hx = 0.5 * (ar["xmax"] - ar["xmin"])
                hy = 0.5 * (ar["ymax"] - ar["ymin"])
                hz = 0.5 * (ar["zmax"] - ar["zmin"]) if self.D == 3 else 0.0
                self._r2 = hx*hx + hy*hy + hz*hz
            if self._r2 <= 0.0:
                raise ValueError("arena radius^2 must be positive for 'arena_r2' normalization.")

    # ---------- helpers ----------
    def _unpack(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        D = self.D
        p1 = s[0:D];       v1 = s[D:2*D]
        p2 = s[2*D:3*D];   v2 = s[3*D:4*D]
        return p1, v1, p2, v2

    def _obs(self) -> np.ndarray:
        p1, v1, p2, v2 = self._unpack(self.state)
        obs = np.concatenate([
            p1 - self.center, p2 - self.center,
            p2 - p1,
            v1, v2,
        ])
        return obs.astype(np.float32)

    def _normalize_d2(self, d2: float) -> float:
        if self.reward_norm == "arena_r2":
            return d2 / self._r2
        if self.reward_norm == "by_d0":
            scale = (self._d0 * self._d0) if (self._d0 and self._d0 > 0.0) else 1.0
            return d2 / scale
        return d2

    # ---------- Public API ----------
    def reset(self) -> np.ndarray:
        self.t = 0
        self.state = self._x0_flat.copy()

        # seed d0 if needed
        if self.reward_norm == "by_d0":
            p1, v1, p2, v2 = self._unpack(self.state)
            self._d0 = float(np.linalg.norm(p2 - self.center))
            if self._d0 <= 0.0:
                self._d0 = 1.0  # avoid divide-by-zero; explicit constant
        else:
            self._d0 = None

        # init progress shaping memory
        self._p2_prev = self._unpack(self.state)[2].copy()
        return self._obs()

    def step(self, a1: np.ndarray, a2: np.ndarray):
        # clip to bounds
        a1 = np.clip(np.asarray(a1, float), self.u_lo, self.u_hi)
        a2 = np.clip(np.asarray(a2, float), self.u_lo, self.u_hi)

        # store previous p2 for progress shaping
        p1_prev, v1_prev, p2_prev, v2_prev = self._unpack(self.state)

        # propagate one step with provided Ad/Bd (STRICT)
        x1 = np.concatenate([p1_prev, v1_prev])
        x2 = np.concatenate([p2_prev, v2_prev])
        x1n = self.Ad @ x1 + self.Bd @ a1
        x2n = self.Ad @ x2 + self.Bd @ a2
        p1n, v1n = x1n[:self.D], x1n[self.D:]
        p2n, v2n = x2n[:self.D], x2n[self.D:]
        self.state = np.concatenate([p1n, v1n, p2n, v2n])

        self.t += 1
        done = (self.t >= self.T)

        # progress shaping (dense, small, potential-based on d^2 normalized)
        d2_prev = float(np.dot(p2_prev - self.center, p2_prev - self.center))
        d2_now  = float(np.dot(p2n      - self.center, p2n      - self.center))
        prog_norm = self._normalize_d2(d2_now) - self._normalize_d2(d2_prev)
        r1 = + self.k_prog * prog_norm
        r2 = - self.k_prog * prog_norm

        # terminal-only main objective
        if done:
            d2T_norm = self._normalize_d2(d2_now)
            e1 = self.cfg["effort_reg_def"] * float(np.dot(a1, a1)) if self.cfg["effort_reg_def"] > 0.0 else 0.0
            e2 = self.cfg["effort_reg_att"] * float(np.dot(a2, a2)) if self.cfg["effort_reg_att"] > 0.0 else 0.0
            r1 += +d2T_norm - e1    # defender maximizes attacker distance^2
            r2 += -d2T_norm - e2    # attacker minimizes attacker distance^2

        info = {"t": self.t, "p2": p2n.copy(), "center": self.center.copy(), "d_T": float(np.linalg.norm(p2n - self.center))}
        return self._obs(), r1, r2, done, info


# =====================================================
# PPO (two agents by self-play)
# =====================================================
class PPO:
    def __init__(self, obs_dim: int, act_dim: int, cfg: Dict[str, Any], device="cpu"):
        self.device = device
        self.clip_eps = cfg["clip_eps"]
        self.entropy_coef = cfg["entropy_coef"]
        self.value_coef = cfg["value_coef"]
        self.max_grad_norm = cfg["max_grad_norm"]
        self.train_epochs = cfg["train_epochs"]
        self.minibatch_size = cfg["minibatch_size"]
        self.gamma = cfg["gamma"]
        self.lam = cfg["gae_lambda"]

        self.pi_def = ActorCritic(obs_dim, act_dim).to(device)
        self.pi_att = ActorCritic(obs_dim, act_dim).to(device)

        self.opt_def = optim.Adam([
            {"params": list(self.pi_def.pi_body.parameters()) + list(self.pi_def.mu_head.parameters()),
             "lr": cfg["policy_lr"]},
            {"params": [self.pi_def.logstd], "lr": cfg["policy_lr"] * 0.5},
            {"params": self.pi_def.vf.parameters(), "lr": cfg["value_lr"]},
        ])
        self.opt_att = optim.Adam([
            {"params": list(self.pi_att.pi_body.parameters()) + list(self.pi_att.mu_head.parameters()),
             "lr": cfg["policy_lr"]},
            {"params": [self.pi_att.logstd], "lr": cfg["policy_lr"] * 0.5},
            {"params": self.pi_att.vf.parameters(), "lr": cfg["value_lr"]},
        ])

    def act(self, obs: np.ndarray, who: str):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if who == "def":
            dist = self.pi_def.policy_dist(obs_t)
            val = self.pi_def.value(obs_t)
        else:
            dist = self.pi_att.policy_dist(obs_t)
            val = self.pi_att.value(obs_t)
        a = dist.sample()
        logp = dist.log_prob(a).sum(-1)
        return a.squeeze(0).cpu().numpy(), logp.item(), val.item()

    def update(self, buf: RolloutBuffer, who: str):
        net = self.pi_def if who == "def" else self.pi_att
        opt = self.opt_def if who == "def" else self.opt_att

        B = buf.to_tensors(self.device)
        adv, ret = compute_gae(B, self.gamma, self.lam, self.device)

        obs = B["obs"]; act = B["act"]; old_logp = B["logp"]

        N = obs.shape[0]
        idx = np.arange(N)

        for _ in range(self.train_epochs):
            np.random.shuffle(idx)
            for start in range(0, N, self.minibatch_size):
                j = idx[start:start + self.minibatch_size]
                if len(j) == 0:
                    continue
                o = obs[j]; a = act[j]; lp_old = old_logp[j]; adv_j = adv[j]; ret_j = ret[j]

                dist = net.policy_dist(o)
                lp = dist.log_prob(a).sum(-1)
                ratio = torch.exp(lp - lp_old)
                surr1 = ratio * adv_j
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_j
                pi_loss = -torch.min(surr1, surr2).mean()

                v = net.value(o)
                v_loss = (v - ret_j).pow(2).mean() * self.value_coef

                ent = dist.entropy().sum(-1).mean() * self.entropy_coef

                loss = pi_loss + v_loss - ent

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), self.max_grad_norm)
                opt.step()

        # clear buffer
        buf.__init__()


# =====================================================
# Training loop
# =====================================================
def train(cfg: Dict[str, Any]):
    set_seed(cfg["seed"])
    device = cfg["device"]
    env = Env(cfg)

    obs_example = env.reset()
    obs_dim = obs_example.shape[0]
    act_dim = env.act_dim

    ppo = PPO(obs_dim, act_dim, cfg, device=device)

    ep_ret_def_hist = []
    ep_ret_att_hist = []

    for ep in range(1, cfg["episodes"] + 1):
        obs = env.reset()

        buf_def = RolloutBuffer()
        buf_att = RolloutBuffer()

        done = False
        t = 0
        last_info = None
        while not done:
            a1, logp1, v1 = ppo.act(obs, who="def")
            a2, logp2, v2 = ppo.act(obs, who="att")

            next_obs, r1, r2, done, info = env.step(a1, a2)

            buf_def.add(obs, a1, logp1, r1, v1, float(done))
            buf_att.add(obs, a2, logp2, r2, v2, float(done))

            obs = next_obs
            t += 1
            last_info = info

        R1 = float(np.sum(buf_def.rew))
        R2 = float(np.sum(buf_att.rew))
        ep_ret_def_hist.append(R1)
        ep_ret_att_hist.append(R2)

        ppo.update(buf_def, who="def")
        ppo.update(buf_att, who="att")

        if ep % cfg["log_every"] == 0:
            m1 = np.mean(ep_ret_def_hist[-cfg["log_every"]:])
            m2 = np.mean(ep_ret_att_hist[-cfg["log_every"]:])
            dT = last_info["d_T"] if last_info is not None else np.nan
            print(f"[ep {ep:05d}] steps={t:02d}  R_def={R1:+.3f}  R_att={R2:+.3f}  "
                  f"mean(R_def)={m1:+.3f}  mean(R_att)={m2:+.3f}  d(T)={dT:.3f}")

    print("Training finished.")
    return ppo


# =====================================================
# Rollout for quick sanity
# =====================================================
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
            a1, _, _ = ppo.act(obs, who="def")
            a2, _, _ = ppo.act(obs, who="att")
            obs, r1, r2, done, info = env.step(a1, a2)
            states.append(env.state.copy())
            actions.append((a1.copy(), a2.copy()))
            infos.append(info)
        trajs.append({"states": np.stack(states), "actions": actions, "infos": infos})
    return trajs


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    # Build Ad/Bd STRICTLY from HCW block
    if CONFIG["dynamics"].lower() != "hcw":
        raise ValueError("This strict script expects CONFIG['dynamics'] == 'hcw'.")

    hcw_block = CONFIG["hcw"]
    n = hcw_block["n"] if ("n" in hcw_block) else hcw_mean_motion(hcw_block)  # requires mu & r0 if no n
    Ad_mx, Bd_mx = hcw_discrete_mats(float(n), float(CONFIG["dt"]))
    CONFIG["dyn"]["Ad"] = as_numpy_const(Ad_mx)
    CONFIG["dyn"]["Bd"] = as_numpy_const(Bd_mx)

    # device choice (strict)
    if CONFIG["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CONFIG['device']='cuda' but CUDA is not available.")
    device = CONFIG["device"]

    print(f"Using device: {device}")
    ppo = train(CONFIG)
    _ = evaluate(ppo, CONFIG, episodes=2)

    torch.save(ppo.pi_def.state_dict(), "ppo_def.pt")
    torch.save(ppo.pi_att.state_dict(), "ppo_att.pt")
    print("Saved RL policies: ppo_def.pt, ppo_att.pt")
