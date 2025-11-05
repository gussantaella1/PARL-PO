# rl_infer.py
import os
import numpy as np
import torch
from rl_loop import ActorCritic, Env  # must match the training definitions


def _center_from_arena_strict(cfg: dict) -> np.ndarray:
    """
    Strictly derive center from cfg['arena'].
    Requires:
      - cfg['D'] ∈ {2,3}
      - cfg['arena'] with either:
         * type == 'sphere' and keys: cx, cy, (cz if D==3), or
         * type == 'box'    and keys: xmin,xmax, ymin,ymax, (zmin,zmax if D==3)
    """
    D = int(cfg["D"])
    ar = cfg["arena"]
    ar_type = ar["type"]

    if ar_type == "sphere":
        if D == 3:
            return np.array([ar["cx"], ar["cy"], ar["cz"]], dtype=float)
        else:
            return np.array([ar["cx"], ar["cy"]], dtype=float)

    elif ar_type == "box":
        cx = 0.5 * (ar["xmin"] + ar["xmax"])
        cy = 0.5 * (ar["ymin"] + ar["ymax"])
        if D == 3:
            cz = 0.5 * (ar["zmin"] + ar["zmax"])
            return np.array([cx, cy, cz], dtype=float)
        else:
            return np.array([cx, cy], dtype=float)

    else:
        raise ValueError(f"Unsupported arena type: {ar_type!r}. Use 'sphere' or 'box'.")


class RLPolicy:
    """
    Deterministic evaluation wrapper around trained PPO policies.
    Uses ONLY OG config keys: D, T, dt, arena, umax, x0 (Env enforces these).
    State expected here: [p1(0:D), v1(0:D), p2(0:D), v2(0:D)].
    """
    def __init__(self, cfg: dict, device: str = "cpu",
                 def_ckpt: str | None = None,
                 att_ckpt: str | None = None):
        # strict copy to avoid accidental mutation
        self.cfg = dict(cfg)
        self.D = int(self.cfg["D"])

        # device (no silent CUDA fallback beyond availability)
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"

        # checkpoint paths: allow explicit args, otherwise default filenames
        self.def_ckpt = def_ckpt or "ppo_def.pt"
        self.att_ckpt = att_ckpt or "ppo_att.pt"

        # Build Env to infer obs_dim/act_dim (Env relies ONLY on OG keys)
        self.env = Env(self.cfg)
        obs_dim = self.env.reset().shape[0]
        act_dim = self.env.act_dim

        # Networks must match training architecture
        self.pi_def = ActorCritic(obs_dim, act_dim).to(self.device)
        self.pi_att = ActorCritic(obs_dim, act_dim).to(self.device)

        # Load weights strictly
        if not os.path.exists(self.def_ckpt):
            raise FileNotFoundError(f"Defender checkpoint not found: {self.def_ckpt}")
        if not os.path.exists(self.att_ckpt):
            raise FileNotFoundError(f"Attacker checkpoint not found: {self.att_ckpt}")

        self.pi_def.load_state_dict(torch.load(self.def_ckpt, map_location=self.device))
        self.pi_att.load_state_dict(torch.load(self.att_ckpt, map_location=self.device))
        self.pi_def.eval()
        self.pi_att.eval()

        # scalar action bounds ±umax (OG key)
        umax = float(self.cfg["umax"])
        self.u_lo, self.u_hi = -umax, umax

        # center strictly from arena
        self.center = _center_from_arena_strict(self.cfg)

    # ---------- observation mapping (fixed "relative" as in training) ----------
    def _obs_from_state(self, state: np.ndarray) -> np.ndarray:
        """
        state: [p1, v1, p2, v2], each in R^D
        returns: observation vector exactly as used in training
        """
        D = self.D
        s  = np.asarray(state, dtype=float).ravel()
        p1 = s[0:D];       v1 = s[D:2*D]
        p2 = s[2*D:3*D];   v2 = s[3*D:4*D]
        c  = self.center

        # fixed relative mapping (no cfg knob):
        # [p1-c, p2-c, (p2-p1), v1, v2]
        obs = np.concatenate([p1 - c, p2 - c, p2 - p1, v1, v2])
        return obs.astype(np.float32)

    # ---------- public API ----------
    @torch.no_grad()
    def act_def(self, state: np.ndarray) -> np.ndarray:
        obs = self._obs_from_state(state)
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.pi_def.policy_dist(o)
        a = dist.mean.squeeze(0).cpu().numpy()  # deterministic evaluation
        return np.clip(a, self.u_lo, self.u_hi)

    @torch.no_grad()
    def act_att(self, state: np.ndarray) -> np.ndarray:
        obs = self._obs_from_state(state)
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.pi_att.policy_dist(o)
        a = dist.mean.squeeze(0).cpu().numpy()  # deterministic evaluation
        return np.clip(a, self.u_lo, self.u_hi)

    # ---------- helper ----------
    @staticmethod
    def pack_state(p1, v1, p2, v2) -> np.ndarray:
        """Build [p1, v1, p2, v2] from blocks."""
        return np.concatenate([np.asarray(p1, float),
                               np.asarray(v1, float),
                               np.asarray(p2, float),
                               np.asarray(v2, float)])
