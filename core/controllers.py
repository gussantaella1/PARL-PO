"""
core/controllers.py

Rule-based attacker controllers used for warm starts, baselines, and non-learning opponents.
"""

from typing import Any, Dict, List

import numpy as np
import torch
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
        """Store configuration and initialize the runtime state for this object."""
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
        self._torch_param_cache = {}

        # One-step ridge solution gain for u_center
        self.K = np.linalg.solve(
            FtF + self.lam * np.eye(D, dtype=np.float32),
            self.F.T
        )

    def u_center(self, p2: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Handle u center for this workflow."""
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

    def _torch_params(self, device: torch.device, dtype: torch.dtype):
        """Internal helper for torch params."""
        key = (device.type, device.index, dtype)
        cached = self._torch_param_cache.get(key)
        if cached is None:
            cached = (
                torch.as_tensor(self.center, device=device, dtype=dtype),
                torch.as_tensor(self.Ad, device=device, dtype=dtype),
                torch.as_tensor(self.K, device=device, dtype=dtype),
            )
            self._torch_param_cache[key] = cached
        return cached

    def act_multi_batch_torch(
        self,
        p1: torch.Tensor,
        v1: torch.Tensor,
        pA: torch.Tensor,
        vA: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched torch implementation of the rule controller.

        Parameters
        ----------
        p1, v1 : (B, D)
            Defender positions/velocities.
        pA, vA : (B, Na, D)
            Attacker positions/velocities.

        Returns
        -------
        actions : (B, Na, D)
        """
        center, Ad, K = self._torch_params(p1.device, p1.dtype)

        B, Na, D = pA.shape
        x2 = torch.cat([pA, vA], dim=-1)  # (B, Na, 2D)
        E2 = torch.matmul(x2, Ad.transpose(0, 1))[..., :D] - center
        uc = -torch.matmul(E2, K.transpose(0, 1))

        r = pA - p1[:, None, :]
        dist = torch.linalg.vector_norm(r, dim=-1, keepdim=True).clamp_min(1e-9)
        r_hat = r / dist
        repulse_mag = self.repulse_gain / dist.square()
        ur = torch.where(
            dist < self.min_sep,
            self.umax * r_hat,
            repulse_mag * r_hat,
        )

        w_center_eff = torch.where(
            dist < self.min_sep,
            torch.zeros_like(dist),
            torch.full_like(dist, self.w_center),
        )

        u = w_center_eff * uc + self.w_avoid * ur - self.w_damp * vA
        return torch.clamp(u, -self.umax, +self.umax)


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
    
    
