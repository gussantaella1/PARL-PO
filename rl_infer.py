# rl_infer.py — replacement block

import os
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn

# ---------- model ----------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128, logstd_init: float = -1.2):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        self.pi_body = nn.Sequential(
            nn.Linear(self.obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, self.act_dim)
        self.logstd  = nn.Parameter(torch.full((self.act_dim,), float(logstd_init)))

        self.vf = nn.Sequential(
            nn.Linear(self.obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def policy_dist(self, obs: torch.Tensor):
        h   = self.pi_body(obs)
        mu  = self.mu_head(h)
        std = self.logstd.exp()
        return torch.distributions.Normal(mu, std)

    def value(self, obs: torch.Tensor):
        return self.vf(obs).squeeze(-1)


def _remap_state_dict_keys(sd: dict) -> OrderedDict:
    """
    Accepts checkpoints saved with older names and maps to new ones:
      'pi.' -> 'pi_body.',  'mu.' -> 'mu_head.',  'v.' -> 'vf.', 'log_std'->'logstd'
    If already new-style, passes through.
    """
    new_sd = OrderedDict()
    for k, v in sd.items():
        k2 = k
        if k2.startswith("pi."): k2 = k2.replace("pi.", "pi_body.", 1)
        if k2.startswith("mu."): k2 = k2.replace("mu.", "mu_head.", 1)
        if k2.startswith("v." ): k2 = k2.replace("v." , "vf."     , 1)
        if "log_std" in k2:      k2 = k2.replace("log_std", "logstd")
        new_sd[k2] = v
    return new_sd


class RLPolicy:
    """
    Inference wrapper that adapts to checkpoints trained with either:
      - obs_dim = 5*D: [p1-c, p2-c, (p2-p1), v1, v2]
      - obs_dim = 4*D: [p1, v1, p2, v2]
    and outputs accelerations in R^D, clipped to ±umax.
    """
    def __init__(self, cfg, device="cpu", def_ckpt=None, att_ckpt=None):
        self.device = device
        self.D      = int(cfg["D"])
        self.act_dim = self.D
        self.umax    = float(cfg.get("umax", 5e-4))

        # arena center
        ar = cfg.get("arena", {"type": "sphere", "cx": 0.0, "cy": 0.0, "cz": 0.0})
        if ar.get("type", "sphere") == "sphere":
            self.center = np.array([ar.get("cx",0.0), ar.get("cy",0.0),
                                    (ar.get("cz",0.0) if self.D==3 else 0.0)], dtype=np.float32)[:self.D]
        else:
            # fallback: use provided center-ish triple if box; or zeros
            cx = float(ar.get("cx", 0.0)); cy = float(ar.get("cy", 0.0))
            cz = float(ar.get("cz", 0.0)) if self.D==3 else 0.0
            self.center = np.array([cx, cy, cz], dtype=np.float32)[:self.D]

        # ckpt paths
        self.def_ckpt = def_ckpt or cfg.get("ckpt_def", "ppo_def.pt")
        self.att_ckpt = att_ckpt or cfg.get("ckpt_att", "ppo_att.pt")
        if not os.path.exists(self.def_ckpt): raise FileNotFoundError(self.def_ckpt)
        if not os.path.exists(self.att_ckpt): raise FileNotFoundError(self.att_ckpt)

        # load state dicts (cpu map); unwrap if wrapped
        sd_def = torch.load(self.def_ckpt, map_location="cpu")
        sd_att = torch.load(self.att_ckpt, map_location="cpu")
        if isinstance(sd_def, dict) and "state_dict" in sd_def: sd_def = sd_def["state_dict"]
        if isinstance(sd_att, dict) and "state_dict" in sd_att: sd_att = sd_att["state_dict"]
        sd_def = _remap_state_dict_keys(sd_def)
        sd_att = _remap_state_dict_keys(sd_att)

        # ---- infer obs_dim from first layer weight shape in checkpoint ----
        # prefer 'pi_body.0.weight'; if missing, try 'vf.0.weight'
        def _infer_obs_dim(sd: OrderedDict):
            key = None
            for k in ("pi_body.0.weight", "vf.0.weight", "pi.0.weight", "v.0.weight"):
                if k in sd:
                    key = k
                    break
            if key is None:
                raise RuntimeError("Could not infer obs_dim from checkpoint keys.")
            return int(sd[key].shape[1])

        obs_dim_def = _infer_obs_dim(sd_def)
        obs_dim_att = _infer_obs_dim(sd_att)
        if obs_dim_def != obs_dim_att:
            # Extremely unlikely, but handle it anyway
            raise RuntimeError(f"DEF and ATT checkpoints disagree on obs_dim: {obs_dim_def} vs {obs_dim_att}")
        self.obs_dim = obs_dim_def

        # determine expected layout
        self._expects_fiveD = (self.obs_dim == 5 * self.D)
        self._expects_fourD = (self.obs_dim == 4 * self.D)
        if not (self._expects_fiveD or self._expects_fourD):
            raise RuntimeError(
                f"Checkpoint expects obs_dim={self.obs_dim}, "
                f"which is not 4*D={4*self.D} nor 5*D={5*self.D}. "
                f"Update the builder below to match your training obs."
            )

        # build models with matching obs_dim and load weights
        self.pi_def = ActorCritic(self.obs_dim, self.act_dim).to(self.device)
        self.pi_att = ActorCritic(self.obs_dim, self.act_dim).to(self.device)

        miss_def = self.pi_def.load_state_dict(sd_def, strict=False)
        miss_att = self.pi_att.load_state_dict(sd_att, strict=False)
        # Optional logging:
        if miss_def.missing_keys or miss_def.unexpected_keys:
            print("[rl_infer] DEF load: missing:", miss_def.missing_keys, " unexpected:", miss_def.unexpected_keys)
        if miss_att.missing_keys or miss_att.unexpected_keys:
            print("[rl_infer] ATT load: missing:", miss_att.missing_keys, " unexpected:", miss_att.unexpected_keys)

        self.pi_def.eval(); self.pi_att.eval()
        self.use_mean_at_eval = bool(cfg.get("rl_eval_deterministic", True))

    @staticmethod
    def pack_state(p1, v1, p2, v2) -> np.ndarray:
        """Raw concatenation used as input to the obs builder."""
        return np.concatenate([p1, v1, p2, v2], axis=0).astype(np.float32)

    def _build_obs(self, state_raw: np.ndarray) -> np.ndarray:
        """
        Map raw [p1, v1, p2, v2] into the obs vector expected by the checkpoint.
        """
        D = self.D
        p1 = state_raw[0:D]
        v1 = state_raw[D:2*D]
        p2 = state_raw[2*D:3*D]
        v2 = state_raw[3*D:4*D]

        if self._expects_fiveD:
            # [p1-c, p2-c, (p2-p1), v1, v2]
            obs = np.concatenate([p1 - self.center,
                                  p2 - self.center,
                                  (p2 - p1),
                                  v1, v2], axis=0)
        elif self._expects_fourD:
            # [p1, v1, p2, v2]
            obs = np.concatenate([p1, v1, p2, v2], axis=0)
        else:
            # Should never hit due to checks in __init__
            raise RuntimeError("Unsupported obs layout.")
        return obs.astype(np.float32)

    def _act_one(self, net: ActorCritic, state_raw: np.ndarray, deterministic: bool):
        obs_np = self._build_obs(state_raw)
        o = torch.as_tensor(obs_np[None, :], dtype=torch.float32, device=self.device)
        dist = net.policy_dist(o)
        a = dist.mean if (self.use_mean_at_eval or deterministic) else dist.sample()
        a = a.squeeze(0).detach().cpu().numpy()
        return np.clip(a, -self.umax, +self.umax)

    def act_def(self, state_raw: np.ndarray, deterministic: bool = True):
        return self._act_one(self.pi_def, state_raw, deterministic)

    def act_att(self, state_raw: np.ndarray, deterministic: bool = True):
        return self._act_one(self.pi_att, state_raw, deterministic)
