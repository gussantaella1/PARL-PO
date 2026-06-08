"""
core/ppo.py

PPO update logic for defender and attacker policy phases.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any

from core.models import ActorCriticDiff
from core.controllers import AttackerRuleController
from core.buffers import compute_gae_from_buffer, RolloutBuffer
from core.utils import atanh, squash_action, logprob_squashed

# =============================================================
# PPO Core (defender learns; attacker optionally rule-based)
# =============================================================
class PPO:
    """PPO optimizer wrapper for defender-only and attacker-only training phases."""
    def __init__(self, obs_dim: int, act_dim: int, cfg: Dict[str, Any], device="cpu"):
        """Store configuration and initialize the runtime state for this object."""
        self.cfg       = cfg
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
        self.num_attackers = int(cfg.get("num_attackers", 1))
        self.D = int(cfg["D"])

        self.freeze_defender = bool(cfg.get("freeze_defender", False))
        self.freeze_attacker = bool(cfg.get("freeze_attacker", False))  # optional symmetry


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

        self._rule_obs_pos_scale = 1.0
        self._rule_obs_vel_scale = 1.0

    @torch.no_grad()
    def act(self, obs_batch: torch.Tensor, who: str, deterministic: bool = False):
        """
        Returns:
        a_env:   (B, D) for defender
                (B, D) or (B, Na, D) for attacker depending on cfg["num_attackers"]
        logp:    (B,)
        val:     (B,)
        """
        B = obs_batch.shape[0]
        D = self.D
        Na = self.num_attackers

        # -------- RULE ATTACKER --------
        if who == "att" and self.attacker_mode == "rule":
            # obs layout: [p1c, pA1c..pANc, rel1..relN, v1, vA1..vAN]
            p1c = obs_batch[:, 0:D]
            off = D

            pA_c = obs_batch[:, off:off + Na * D].reshape(B, Na, D)
            off += Na * D

            # Skip relative-position blocks; the rule controller only needs absolute state.
            off += Na * D

            v1 = obs_batch[:, off:off + D]
            off += D
            vA = obs_batch[:, off:off + Na * D].reshape(B, Na, D)

            center = torch.as_tensor(self.rule_ctrl.center, dtype=obs_batch.dtype, device=obs_batch.device)
            pos_scale = float(self._rule_obs_pos_scale)
            vel_scale = float(self._rule_obs_vel_scale)

            p1 = (p1c / pos_scale) + center
            pA = (pA_c / pos_scale) + center
            v1_abs = v1 / vel_scale
            vA_abs = vA / vel_scale

            a_t = self.rule_ctrl.act_multi_batch_torch(p1, v1_abs, pA, vA_abs)

            zero = torch.zeros(B, dtype=obs_batch.dtype, device=obs_batch.device)
            return a_t, zero, zero

        # -------- RL POLICY (defender OR attacker) --------
        net = self.def_net if who == "def" else self.att_net

        if deterministic:
            dist = net.dist(obs_batch, who=who)
            u_raw = dist.mean
            a_env = squash_action(u_raw, self.act_scale)
            logp = torch.zeros(B, dtype=obs_batch.dtype, device=obs_batch.device)
            val  = net.value(obs_batch)
            return a_env, logp, val

        return net.act(obs_batch, who, self.act_scale)


    def _update_one(self,
                    net: ActorCriticDiff, opt: optim.Optimizer,
                    obs: torch.Tensor, act_env: torch.Tensor,
                    old_logp: torch.Tensor, old_val: torch.Tensor,
                    adv: torch.Tensor, ret: torch.Tensor, who: str):
        """Internal helper for update one."""
        B = obs.shape[0]
        for _ in range(self.epochs):
            idx = torch.randperm(B, device=obs.device)  # <-- shuffle indices on same device
            for st in range(0, B, self.mb_size):
                j = idx[st:st+self.mb_size]
                o = obs[j]; a = act_env[j]; lp_old = old_logp[j]; v_old = old_val[j]; A = adv[j]; R = ret[j]

                assert not lp_old.requires_grad
                assert not v_old.requires_grad


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
        """Handle update defender only for this workflow."""
        advD, retD = compute_gae_from_buffer(buf_def, self.gamma, self.lam)
        oD, aD, lpD, vD, _, _ = buf_def.get()
        self._update_one(self.def_net, self.def_opt, oD, aD, lpD, vD, advD, retD, who="def")

    def update_attacker_only(self, buf_att: RolloutBuffer):
        """Handle update attacker only for this workflow."""
        advA, retA = compute_gae_from_buffer(buf_att, self.gamma, self.lam)
        oA, aA, lpA, vA, _, _ = buf_att.get()
        self._update_one(self.att_net, self.att_opt, oA, aA, lpA, vA, advA, retA, who="att")
