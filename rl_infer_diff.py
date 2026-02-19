# rl_infer_diff.py

import os
from typing import Dict, Any, Tuple

import numpy as np
import torch

# Import the Diff-Nash network + rule-based attacker from training script
from rl_loop_diffgame import ActorCriticDiff, AttackerRuleController


class RLPolicyDiff:
    """
    Inference wrapper for Diff-Nash policies.

    - Defender is always RL (ActorCriticDiff loaded from ppo_def.pt).
    - Attacker can be:
        * "rl"   -> RL attacker (ppo_att.pt required)
        * "rule" -> rule-based attacker (AttackerRuleController; no ppo_att.pt)

    Exposes:
      - verify_ckpt_compat() -> (obs_dim_def, obs_dim_att)
      - act_def_obs(obs, deterministic=True)
      - act_att_obs(obs, deterministic=True)

    Where obs is expected to be:
        obs = [p1 - center, p2 - center, (p2 - p1), v1, v2]  in R^{5D}.
    """

    def __init__(self,
                 cfg: Dict[str, Any],
                 device: str = "cpu",
                 def_ckpt: str | None = None,
                 att_ckpt: str | None = None):

        self.cfg = cfg
        self.device = device
        self.D = int(cfg["D"])
        self.act_dim = self.D
        self.umax = float(cfg.get("umax", 5e-4))
        self.attacker_mode = cfg.get("attacker_mode", "rl")  # "rl" or "rule"
        self.use_mean_at_eval = bool(cfg.get("rl_eval_deterministic", True))

        # ---- Arena center (for reconstructing p1, p2 from obs) ----
        ar = cfg.get("arena", {"type": "sphere", "cx": 0.0, "cy": 0.0, "cz": 0.0})
        if ar.get("type", "sphere") == "sphere":
            self.center = np.array(
                [
                    ar.get("cx", 0.0),
                    ar.get("cy", 0.0),
                    (ar.get("cz", 0.0) if self.D == 3 else 0.0),
                ],
                dtype=np.float32,
            )[: self.D]
        else:
            cx = float(ar.get("cx", 0.0))
            cy = float(ar.get("cy", 0.0))
            cz = float(ar.get("cz", 0.0)) if self.D == 3 else 0.0
            self.center = np.array([cx, cy, cz], dtype=np.float32)[: self.D]

        def _pick(cfg, keys, default):
            for k in keys:
                v = cfg.get(k, None)
                if v:
                    return v
            return default
        
        def _strip_ignored_keys(sd: dict) -> dict:
            # handle wrapped {"state_dict": ...} already outside
            IGNORE = {
                "layer.center",
                # if you ever see these again, add them too:
                # "layer.Ad", "layer.Bd", "layer.P",
            }

            # also handle possible DataParallel "module." prefix
            for k in list(sd.keys()):
                k_noprefix = k[len("module."):] if k.startswith("module.") else k
                if k_noprefix in IGNORE:
                    sd.pop(k, None)
            return sd



        # ==================== DEFENDER (RL) ====================
        self.def_ckpt = def_ckpt or _pick(
            cfg,
            ["def_ckpt_path", "def_ckpt", "def_policy_path", "defender_ckpt_path", "defender_ckpt", "ckpt_def"],
            "ppo_def.pt",
        )
        
        if not os.path.exists(self.def_ckpt):
            raise FileNotFoundError(self.def_ckpt)

        # Diff-Nash training uses obs_dim = 5 * D
        self.obs_dim_def = 5 * self.D
        self.def_net = ActorCriticDiff(self.obs_dim_def, self.act_dim, cfg).to(self.device)

        sd_def = torch.load(self.def_ckpt, map_location=self.device)
        if isinstance(sd_def, dict) and "state_dict" in sd_def:
            sd_def = sd_def["state_dict"]
        sd_def = _strip_ignored_keys(sd_def)

        miss_def = self.def_net.load_state_dict(sd_def, strict=False)
        if miss_def.missing_keys or miss_def.unexpected_keys:
            print("[rl_infer_diff] DEF load: missing:", miss_def.missing_keys,
                " unexpected:", miss_def.unexpected_keys)

        self.def_net.eval()

        # ==================== ATTACKER: RL or RULE ====================

        self.att_net = None
        self.rule_ctrl = None

        if self.attacker_mode == "rl":
            # RL attacker – ppo_att.pt required

            self.att_ckpt = att_ckpt or _pick(
                cfg,
                ["att_ckpt_path", "att_ckpt", "att_policy_path", "attacker_ckpt_path", "attacker_ckpt", "ckpt_att"],
                "ppo_att.pt",
            )

            if not os.path.exists(self.att_ckpt):
                raise FileNotFoundError(self.att_ckpt)

            self.obs_dim_att = 5 * self.D
            self.att_net = ActorCriticDiff(self.obs_dim_att, self.act_dim, cfg).to(self.device)

            sd_att = torch.load(self.att_ckpt, map_location=self.device)
            if isinstance(sd_att, dict) and "state_dict" in sd_att:
                sd_att = sd_att["state_dict"]
            sd_att = _strip_ignored_keys(sd_att)

            miss_att = self.att_net.load_state_dict(sd_att, strict=False)


            if miss_att.missing_keys or miss_att.unexpected_keys:
                print(
                    "[rl_infer_diff] ATT load: missing:",
                    miss_att.missing_keys,
                    " unexpected:",
                    miss_att.unexpected_keys,
                )
            self.att_net.eval()
        else:
            # Rule-based attacker – no ppo_att required
            self.obs_dim_att = 5 * self.D
            self.rule_ctrl = AttackerRuleController(cfg)

    # ------------------------------------------------------------------
    #  Public helpers
    # ------------------------------------------------------------------
    def verify_ckpt_compat(self) -> Tuple[int, int]:
        """
        For runner sanity-check; the runner expects (5*D, 5*D).
        """
        return self.obs_dim_def, self.obs_dim_att

    # ---- Core one-step for obs-level policies ----
    def _act_one_obs(self, net: ActorCriticDiff, obs: np.ndarray, who: str, deterministic: bool):
        obs_np = np.asarray(obs, dtype=np.float32)
        o = torch.as_tensor(obs_np[None, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            # ActorCriticDiff.act already does Normal + tanh + scaling
            a_env, _, _ = net.act(o, who=who, act_scale=self.umax)
        a = a_env.squeeze(0).detach().cpu().numpy()
        return np.clip(a, -self.umax, +self.umax)

    # ---- Defender: always RL ----
    def act_def_obs(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return self._act_one_obs(self.def_net, obs, who="def", deterministic=deterministic)

    # ---- Attacker: RL or rule-based ----
    def act_att_obs(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_np = np.asarray(obs, dtype=np.float32)

        if self.attacker_mode == "rl":
            if self.att_net is None:
                raise RuntimeError("attacker_mode='rl' but attacker net is not initialized.")
            return self._act_one_obs(self.att_net, obs_np, who="att", deterministic=deterministic)

        # Rule-based attacker: reconstruct (p1, v1, p2, v2) from obs = [p1-c, p2-c, rel, v1, v2]
        D = self.D
        c = self.center

        p1c = obs_np[0:D]
        p2c = obs_np[D : 2 * D]
        # rel = obs_np[2*D : 3*D]  # (p2 - p1), not needed explicitly
        v1 = obs_np[3 * D : 4 * D]
        v2 = obs_np[4 * D : 5 * D]

        p1 = p1c + c
        p2 = p2c + c

        u = self.rule_ctrl.act(p1, v1, p2, v2)
        return np.clip(np.asarray(u, dtype=np.float32), -self.umax, +self.umax)
    

