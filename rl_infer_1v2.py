# rl_infer_1v2.py

from __future__ import annotations

import copy
import os
import pickle
from typing import Any, Dict, Tuple

import numpy as np
import torch

from config_rl_1v2 import build_dyn
from core_1v2.controllers import AttackerRuleController
from core_1v2.models import ActorCriticDiff
from core_1v2.utils import permute_obs_for_attacker


def _load_checkpoint_payload(path: str, map_location, label: str):
    try:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError as exc:
        if "Weights only load failed" not in str(exc):
            raise
        print(
            f"[rl_infer_1v2] {label}: legacy checkpoint format detected; "
            "retrying torch.load(..., weights_only=False)."
        )
        return torch.load(path, map_location=map_location, weights_only=False)



class RLPolicy_Multi:
    """
    Inference wrapper for Diff-Nash policies.

    Supports num_attackers >= 1 with the SAME obs layout as Env._obs:
      obs = [p1-center,
             pA0-center, ..., pA{Na-1}-center,
             (pA0-p1), ..., (pA{Na-1}-p1),
             v1,
             vA0, ..., vA{Na-1}]

    dim = (2 + 3*Na) * D

    Attacker policy is shared across attackers:
      act_att_obs(obs, attacker_idx=k) will ego-permute obs so attacker k is slot 0.
    """

    def __init__(self,
                 cfg: Dict[str, Any],
                 device: str = "cpu",
                 def_ckpt: str | None = None,
                 att_ckpt: str | None = None):

        self.cfg = copy.deepcopy(cfg)
        build_dyn(self.cfg)

        self.device = device
        self.D = int(cfg["D"])
        self.Na = int(self.cfg.get("num_attackers", 1))
        self.act_dim = self.D
        self.umax = float(self.cfg.get("umax", 5e-4))
        self.attacker_mode = self.cfg.get("attacker_mode", "rl")
        self.use_mean_at_eval = bool(self.cfg.get("rl_eval_deterministic", True))

        # ---- Arena center ----
        ar = self.cfg.get("arena", {"type": "sphere", "cx": 0.0, "cy": 0.0, "cz": 0.0})
        self.center = np.array(
            [
                ar.get("cx", 0.0),
                ar.get("cy", 0.0),
                (ar.get("cz", 0.0) if self.D == 3 else 0.0),
            ],
            dtype=np.float32,
        )[: self.D]

        def _pick(cfg, keys, default):
            for k in keys:
                v = cfg.get(k, None)
                if v:
                    return v
            return default

        def _strip_ignored_keys(sd: dict) -> dict:
            IGNORE = {"layer.center"}
            for k in list(sd.keys()):
                k_noprefix = k[len("module."):] if k.startswith("module.") else k
                if k_noprefix in IGNORE:
                    sd.pop(k, None)
            return sd

        # ---- obs dim for 1vN ----
        self.obs_dim = (2 + 3 * self.Na) * self.D
        self.obs_dim_def = self.obs_dim
        self.obs_dim_att = self.obs_dim

        # ==================== DEFENDER (RL) ====================
        self.def_ckpt = def_ckpt or _pick(
            self.cfg,
            ["def_ckpt_path", "def_ckpt", "def_policy_path", "defender_ckpt_path", "defender_ckpt", "ckpt_def"],
            "ppo_def.pt",
        )
        if not os.path.exists(self.def_ckpt):
            raise FileNotFoundError(self.def_ckpt)

        self.def_net = ActorCriticDiff(self.obs_dim_def, self.act_dim, self.cfg).to(self.device)
        sd_def = _load_checkpoint_payload(self.def_ckpt, map_location=self.device, label="DEF")
        if isinstance(sd_def, dict) and "state_dict" in sd_def:
            sd_def = sd_def["state_dict"]
        sd_def = _strip_ignored_keys(sd_def)

        miss_def = self.def_net.load_state_dict(sd_def, strict=False)
        if miss_def.missing_keys or miss_def.unexpected_keys:
            print("[rl_infer] DEF load: missing:", miss_def.missing_keys,
                  " unexpected:", miss_def.unexpected_keys)
        self.def_net.eval()

        # ==================== ATTACKER: RL or RULE ====================
        self.att_net = None
        self.rule_ctrl = None

        if self.attacker_mode == "rl":
            self.att_ckpt = att_ckpt or _pick(
                self.cfg,
                ["att_ckpt_path", "att_ckpt", "att_policy_path", "attacker_ckpt_path", "attacker_ckpt", "ckpt_att"],
                "ppo_att.pt",
            )
            if not os.path.exists(self.att_ckpt):
                raise FileNotFoundError(self.att_ckpt)

            self.att_net = ActorCriticDiff(self.obs_dim_att, self.act_dim, self.cfg).to(self.device)
            sd_att = _load_checkpoint_payload(self.att_ckpt, map_location=self.device, label="ATT")
            if isinstance(sd_att, dict) and "state_dict" in sd_att:
                sd_att = sd_att["state_dict"]
            sd_att = _strip_ignored_keys(sd_att)

            miss_att = self.att_net.load_state_dict(sd_att, strict=False)
            if miss_att.missing_keys or miss_att.unexpected_keys:
                print("[rl_infer] ATT load: missing:", miss_att.missing_keys,
                      " unexpected:", miss_att.unexpected_keys)
            self.att_net.eval()
        else:
            self.rule_ctrl = AttackerRuleController(self.cfg)

    # ------------------------------------------------------------------
    def verify_ckpt_compat(self) -> Tuple[int, int]:
        return self.obs_dim_def, self.obs_dim_att

    # ------------------------------------------------------------------
    def _act_one_obs(self, net: ActorCriticDiff, obs: np.ndarray, who: str, deterministic: bool):
        obs_np = np.asarray(obs, dtype=np.float32)
        o = torch.as_tensor(obs_np[None, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if deterministic:
                dist = net.dist(o, who=who)
                a_env = torch.tanh(dist.mean) * self.umax
            else:
                a_env, _, _ = net.act(o, who=who, act_scale=self.umax)
        a = a_env.squeeze(0).detach().cpu().numpy()
        return np.clip(a, -self.umax, +self.umax)

    # ------------------------------------------------------------------
    def _permute_obs_for_attacker(self, obs: np.ndarray, attacker_idx: int) -> np.ndarray:
        if self.Na == 1 or attacker_idx == 0:
            return obs

        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)[None, :], dtype=torch.float32)
        obs_perm = permute_obs_for_attacker(obs_t, attacker_idx, self.D, self.Na)
        return obs_perm.squeeze(0).cpu().numpy()

    # ---- Defender: always RL ----
    def act_def_obs(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return self._act_one_obs(self.def_net, obs, who="def", deterministic=deterministic)

    # ---- Attacker: shared policy; choose attacker_idx ----
    def act_att_obs(self, obs: np.ndarray, deterministic: bool = True, attacker_idx: int = 0) -> np.ndarray:
        obs_np = np.asarray(obs, dtype=np.float32)

        if self.attacker_mode == "rl":
            if self.att_net is None:
                raise RuntimeError("attacker_mode='rl' but attacker net is not initialized.")
            obs_ego = self._permute_obs_for_attacker(obs_np, attacker_idx)
            return self._act_one_obs(self.att_net, obs_ego, who="att", deterministic=deterministic)

        # Rule-based attacker: for Na>1, we need p1,v1 and pAk,vAk for that attacker
        D = self.D
        Na = self.Na
        c = self.center

        # unpack
        p1c = obs_np[0:D]; off = D
        pAc = [obs_np[off + k*D: off + (k+1)*D] for k in range(Na)]
        off += Na*D
        rel = [obs_np[off + k*D: off + (k+1)*D] for k in range(Na)]
        off += Na*D
        v1  = obs_np[off:off + D]
        off += D
        vA  = [obs_np[off + k*D: off + (k+1)*D] for k in range(Na)]

        p1 = p1c + c
        pk = pAc[attacker_idx] + c
        vk = vA[attacker_idx]

        u = self.rule_ctrl.act(p1, v1, pk, vk)
        return np.clip(np.asarray(u, dtype=np.float32), -self.umax, +self.umax)
    
