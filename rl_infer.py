#rl_infer.py

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from config_rl import build_dyn
from rl_loop import ActorCriticDiff, AttackerRuleController


class RLPolicyDiff:
    """
    Inference wrapper for policies trained with the new rl_loop.py.

    Supports:
      - Defender: always RL (ActorCriticDiff)
      - Attacker:
          * "rl"   -> ActorCriticDiff loaded from checkpoint
          * "rule" -> AttackerRuleController (no attacker checkpoint required)

    Expected single-attacker observation format:
      if fuel disabled:
          obs = [p1-center, p2-center, (p2-p1), v1, v2]                in R^{5D}
      if fuel enabled:
          obs = [p1-center, p2-center, (p2-p1), v1, v2, fuel_def, fuel_att]
                                                                     in R^{5D+2}

    Notes
    -----
    - This wrapper currently assumes num_attackers == 1.
    - If deterministic=True, actions are produced from the policy mean.
    - If deterministic=False, actions are sampled from the policy.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        device: str = "cpu",
        def_ckpt: Optional[str] = None,
        att_ckpt: Optional[str] = None,
    ):
        self.cfg = copy.deepcopy(cfg)

        if device == "cuda" and not torch.cuda.is_available():
            print("[rl_infer_diff] CUDA requested but not available; falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)

        # Build discrete dynamics for new DiffLSLayer
        build_dyn(self.cfg)

        self.D = int(self.cfg["D"])
        self.act_dim = self.D
        self.umax = float(self.cfg.get("umax", 5e-4))
        self.num_attackers = int(self.cfg.get("num_attackers", 1))
        if self.num_attackers != 1:
            raise NotImplementedError(
                "RLPolicyDiff currently supports only num_attackers == 1."
            )

        raw_attacker_mode = str(self.cfg.get("attacker_mode", "rl")).lower()
        if raw_attacker_mode in ("rl", "train"):
            self.attacker_mode = "rl"
        elif raw_attacker_mode == "rule":
            self.attacker_mode = "rule"
        else:
            raise ValueError(
                f"Unsupported attacker_mode={raw_attacker_mode!r}; expected 'rl' or 'rule'."
            )

        self.use_mean_at_eval = bool(self.cfg.get("rl_eval_deterministic", True))
        self.use_fuel = bool(self.cfg.get("fuel", {}).get("enable"))
        self.obs_extra = 2 if self.use_fuel else 0

        self.obs_dim_def = 5 * self.D + self.obs_extra
        self.obs_dim_att = 5 * self.D + self.obs_extra

        # ---- Arena center (for reconstructing p1, p2 from obs in rule mode) ----
        ar = self.cfg.get("arena", {"type": "sphere", "cx": 0.0, "cy": 0.0, "cz": 0.0})
        self.center = np.array(
            [
                float(ar.get("cx", 0.0)),
                float(ar.get("cy", 0.0)),
                (float(ar.get("cz", 0.0)) if self.D == 3 else 0.0),
            ],
            dtype=np.float32,
        )[: self.D]

        # ==================== DEFENDER (always RL) ====================
        self.def_ckpt = def_ckpt or self._pick(
            self.cfg,
            [
                "def_ckpt_path",
                "def_ckpt",
                "def_policy_path",
                "defender_ckpt_path",
                "defender_ckpt",
                "ckpt_def",
            ],
            "ppo_def.pt",
        )
        if not os.path.exists(self.def_ckpt):
            raise FileNotFoundError(f"Defender checkpoint not found: {self.def_ckpt}")

        self.def_net = ActorCriticDiff(self.obs_dim_def, self.act_dim, self.cfg).to(self.device)
        self._load_net_checkpoint(self.def_net, self.def_ckpt, name="DEF")
        self.def_net.eval()
        for p in self.def_net.parameters():
            p.requires_grad_(False)

        # ==================== ATTACKER (RL or RULE) ====================
        self.att_net = None
        self.rule_ctrl = None

        if self.attacker_mode == "rl":
            self.att_ckpt = att_ckpt or self._pick(
                self.cfg,
                [
                    "att_ckpt_path",
                    "att_ckpt",
                    "att_policy_path",
                    "attacker_ckpt_path",
                    "attacker_ckpt",
                    "ckpt_att",
                ],
                "ppo_att.pt",
            )
            if not os.path.exists(self.att_ckpt):
                raise FileNotFoundError(f"Attacker checkpoint not found: {self.att_ckpt}")

            self.att_net = ActorCriticDiff(self.obs_dim_att, self.act_dim, self.cfg).to(self.device)
            self._load_net_checkpoint(self.att_net, self.att_ckpt, name="ATT")
            self.att_net.eval()
            for p in self.att_net.parameters():
                p.requires_grad_(False)
        else:
            self.rule_ctrl = AttackerRuleController(self.cfg)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pick(cfg: Dict[str, Any], keys, default):
        for k in keys:
            v = cfg.get(k, None)
            if v:
                return v
        return default

    @staticmethod
    def _strip_ignored_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Strip keys that may appear in older checkpoints or wrappers and
        should not block loading.
        """
        ignore_exact = {
            "layer.center",  # old NoPriorLayer buffer
        }

        out = {}
        for k, v in sd.items():
            k_noprefix = k[len("module."):] if k.startswith("module.") else k
            if k_noprefix in ignore_exact:
                continue
            out[k_noprefix] = v
        return out

    def _extract_state_dict(self, payload: Any) -> Dict[str, torch.Tensor]:
        """
        Accept:
          - raw state_dict
          - {"state_dict": ...}
          - {"model": ...}
          - {"policy": ...}
          - {"net": ...}
          - {"def_net": ...}
          - {"att_net": ...}
          - {"actor_critic": ...}
        """
        if not isinstance(payload, dict):
            raise RuntimeError("Checkpoint payload is not a dict; unsupported format.")

        # Raw state_dict case
        if all(isinstance(k, str) for k in payload.keys()):
            if any(isinstance(v, torch.Tensor) for v in payload.values()):
                return payload

        for key in ["state_dict", "model", "policy", "net", "def_net", "att_net", "actor_critic"]:
            if key in payload and isinstance(payload[key], dict):
                return payload[key]

        # Wrapped PPO save with prefixes like def_net.xxx / att_net.xxx
        if any(k.startswith("def_net.") for k in payload.keys()):
            return {k[len("def_net."):]: v for k, v in payload.items() if k.startswith("def_net.")}
        if any(k.startswith("att_net.") for k in payload.keys()):
            return {k[len("att_net."):]: v for k, v in payload.items() if k.startswith("att_net.")}

        raise RuntimeError("Could not extract state_dict from checkpoint payload.")

    def _load_net_checkpoint(self, model: torch.nn.Module, ckpt_path: str, name: str = "NET") -> None:
        payload = torch.load(ckpt_path, map_location=self.device)
        sd = self._extract_state_dict(payload)
        sd = self._strip_ignored_keys(sd)

        miss = model.load_state_dict(sd, strict=False)
        if miss.missing_keys or miss.unexpected_keys:
            print(
                f"[rl_infer_diff] {name} load: "
                f"missing={miss.missing_keys} unexpected={miss.unexpected_keys}"
            )

    def _validate_obs(self, obs: np.ndarray, who: str) -> np.ndarray:
        obs_np = np.asarray(obs, dtype=np.float32).reshape(-1)
        expected = self.obs_dim_def if who == "def" else self.obs_dim_att
        if obs_np.shape[0] != expected:
            raise ValueError(
                f"Observation size mismatch for {who}: got {obs_np.shape[0]}, expected {expected}. "
                f"{'Fuel is enabled, so expected 5D+2.' if self.use_fuel else 'Fuel is disabled, so expected 5D.'}"
            )
        return obs_np

    def _resolve_deterministic(self, deterministic: Optional[bool]) -> bool:
        if deterministic is None:
            return self.use_mean_at_eval
        return bool(deterministic)

    def _act_one_obs(
        self,
        net: ActorCriticDiff,
        obs: np.ndarray,
        who: str,
        deterministic: Optional[bool],
    ) -> np.ndarray:
        obs_np = self._validate_obs(obs, who=who)
        deterministic = self._resolve_deterministic(deterministic)

        o = torch.as_tensor(obs_np[None, :], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if deterministic:
                dist = net.dist(o, who=who)
                u_raw = dist.mean
                a_env = torch.tanh(u_raw) * self.umax
            else:
                a_env, _, _ = net.act(o, who=who, act_scale=self.umax)

        a = a_env.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return np.clip(a, -self.umax, +self.umax)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def verify_ckpt_compat(self) -> Tuple[int, int]:
        """
        Returns expected obs dims for defender and attacker.
        """
        return self.obs_dim_def, self.obs_dim_att

    def act_def_obs(self, obs: np.ndarray, deterministic: Optional[bool] = None) -> np.ndarray:
        """
        Defender action from observation.
        """
        return self._act_one_obs(self.def_net, obs, who="def", deterministic=deterministic)

    def act_att_obs(self, obs: np.ndarray, deterministic: Optional[bool] = None) -> np.ndarray:
        """
        Attacker action from observation.

        - RL mode: uses attacker network.
        - Rule mode: reconstructs (p1, p2, v1, v2) from obs and applies rule controller.
        """
        obs_np = self._validate_obs(obs, who="att")

        if self.attacker_mode == "rl":
            if self.att_net is None:
                raise RuntimeError("attacker_mode='rl' but attacker net is not initialized.")
            return self._act_one_obs(self.att_net, obs_np, who="att", deterministic=deterministic)

        if self.rule_ctrl is None:
            raise RuntimeError("attacker_mode='rule' but rule controller is not initialized.")

        D = self.D
        c = self.center

        # obs = [p1c, p2c, rel, v1, v2, (optional fuel_def, fuel_att)]
        p1c = obs_np[0:D]
        p2c = obs_np[D : 2 * D]
        v1 = obs_np[3 * D : 4 * D]
        v2 = obs_np[4 * D : 5 * D]

        p1 = p1c + c
        p2 = p2c + c

        u = self.rule_ctrl.act(p1, v1, p2, v2)
        return np.clip(np.asarray(u, dtype=np.float32), -self.umax, +self.umax)