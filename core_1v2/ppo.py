from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core_1v2.buffers import RolloutBuffer, compute_gae_from_buffer
from core_1v2.controllers import AttackerRuleController
from core_1v2.models import ActorCriticDiff
from core_1v2.utils import atanh, logprob_squashed, permute_obs_for_attacker, squash_action


class PPO:
    def __init__(self, obs_dim: int, act_dim: int, cfg: Dict[str, Any], device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.clip_eps = cfg["clip_eps"]
        self.ent_coef = cfg["entropy_coef"]
        self.vf_coef = cfg["value_coef"]
        self.max_grad = cfg["max_grad_norm"]
        self.epochs = cfg["train_epochs"]
        self.mb_size = cfg["minibatch_size"]
        self.gamma = cfg["gamma"]
        self.lam = cfg["gae_lambda"]
        self.act_scale = float(cfg["umax"])
        self.v_clip_eps = 0.2

        self.attacker_mode = cfg.get("attacker_mode", "rule")
        self.freeze_defender = bool(cfg.get("freeze_defender", False))
        self.freeze_attacker = bool(cfg.get("freeze_attacker", False))
        self.D = int(cfg["D"])
        self.Na = int(cfg.get("num_attackers", 1))
        self._rule_obs_pos_scale = 1.0
        self._rule_obs_vel_scale = 1.0

        self.def_net = ActorCriticDiff(obs_dim, act_dim, cfg).to(device)
        self.def_opt = optim.Adam(
            [
                {
                    "params": list(self.def_net.pi.parameters()) + list(self.def_net.mu_res.parameters()),
                    "lr": cfg["policy_lr"],
                },
                {"params": [self.def_net.logstd], "lr": cfg["policy_lr"] * 0.5},
                {"params": self.def_net.vf.parameters(), "lr": cfg["value_lr"]},
            ]
        )
        self.def_base_lrs = [g["lr"] for g in self.def_opt.param_groups]

        if self.attacker_mode == "rl":
            self.att_net = ActorCriticDiff(obs_dim, act_dim, cfg).to(device)
            self.att_opt = optim.Adam(
                [
                    {
                        "params": list(self.att_net.pi.parameters()) + list(self.att_net.mu_res.parameters()),
                        "lr": cfg["policy_lr"],
                    },
                    {"params": [self.att_net.logstd], "lr": cfg["policy_lr"] * 0.5},
                    {"params": self.att_net.vf.parameters(), "lr": cfg["value_lr"]},
                ]
            )
            self.att_base_lrs = [g["lr"] for g in self.att_opt.param_groups]
            self.rule_ctrl = None
        else:
            self.att_net = None
            self.att_opt = None
            self.att_base_lrs = None
            self.rule_ctrl = AttackerRuleController(cfg)

    @torch.no_grad()
    def act(self, obs_batch: torch.Tensor, who: str, deterministic: bool = False):
        B = obs_batch.shape[0]
        D = self.D
        Na = self.Na

        if who == "att" and self.attacker_mode == "rule":
            if self.rule_ctrl is None:
                raise RuntimeError("rule_ctrl missing for rule attacker_mode")

            ob = obs_batch.detach().cpu().numpy()
            base_dim = (2 + 3 * Na) * D
            if ob.shape[1] < base_dim:
                raise ValueError(
                    f"Expected obs dim >= {base_dim} for D={D}, Na={Na}, got {ob.shape[1]}"
                )

            p1c = ob[:, 0:D]
            off = D
            pA_c = ob[:, off:off + Na * D].reshape(B, Na, D)
            off += Na * D
            off += Na * D
            v1 = ob[:, off:off + D]
            off += D
            vA = ob[:, off:off + Na * D].reshape(B, Na, D)

            center = self.rule_ctrl.center
            pos_scale = self._rule_obs_pos_scale
            vel_scale = self._rule_obs_vel_scale
            acts = np.zeros((B, Na, D), dtype=np.float32)
            for i in range(B):
                p1 = (p1c[i] / pos_scale) + center
                v1_i = v1[i] / vel_scale
                for k in range(Na):
                    p2 = (pA_c[i, k] / pos_scale) + center
                    v2 = vA[i, k] / vel_scale
                    acts[i, k] = self.rule_ctrl.act(p1, v1_i, p2, v2).astype(np.float32)

            a_t = torch.as_tensor(acts, dtype=obs_batch.dtype, device=obs_batch.device)
            if Na == 1:
                zero = torch.zeros(B, dtype=obs_batch.dtype, device=obs_batch.device)
                return a_t[:, 0, :], zero, zero
            zero2 = torch.zeros((B, Na), dtype=obs_batch.dtype, device=obs_batch.device)
            return a_t, zero2, zero2

        net = self.def_net if who == "def" else self.att_net
        if net is None:
            raise RuntimeError("att_net is None but attacker_mode='rl' required")

        if who == "def" or Na == 1:
            if deterministic:
                dist = net.dist(obs_batch, who=who)
                a_env = squash_action(dist.mean, self.act_scale)
                logp = torch.zeros(B, dtype=obs_batch.dtype, device=obs_batch.device)
                val = net.value(obs_batch)
                return a_env, logp, val
            return net.act(obs_batch, who, self.act_scale)

        a_list = []
        lp_list = []
        v_list = []
        for k in range(Na):
            obs_k = permute_obs_for_attacker(obs_batch, k, D, Na)
            if deterministic:
                dist = net.dist(obs_k, who="att")
                a_env = squash_action(dist.mean, self.act_scale)
                logp = torch.zeros(B, dtype=obs_batch.dtype, device=obs_batch.device)
                val = net.value(obs_k)
            else:
                a_env, logp, val = net.act(obs_k, who="att", act_scale=self.act_scale)
            a_list.append(a_env)
            lp_list.append(logp)
            v_list.append(val)

        return (
            torch.stack(a_list, dim=1),
            torch.stack(lp_list, dim=1),
            torch.stack(v_list, dim=1),
        )

    def _update_one(
        self,
        net: ActorCriticDiff,
        opt: optim.Optimizer,
        obs: torch.Tensor,
        act_env: torch.Tensor,
        old_logp: torch.Tensor,
        old_val: torch.Tensor,
        adv: torch.Tensor,
        ret: torch.Tensor,
        who: str,
    ):
        B = obs.shape[0]
        for _ in range(self.epochs):
            idx = torch.randperm(B, device=obs.device)
            for st in range(0, B, self.mb_size):
                j = idx[st:st + self.mb_size]
                o = obs[j]
                a = act_env[j]
                lp_old = old_logp[j]
                v_old = old_val[j]
                A = adv[j]
                R = ret[j]

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
        advD, retD = compute_gae_from_buffer(buf_def, self.gamma, self.lam)
        oD, aD, lpD, vD, _, _ = buf_def.get()
        self._update_one(self.def_net, self.def_opt, oD, aD, lpD, vD, advD, retD, who="def")

    def update_attacker_only(self, buf_att: RolloutBuffer):
        advA, retA = compute_gae_from_buffer(buf_att, self.gamma, self.lam)
        oA, aA, lpA, vA, _, _ = buf_att.get()
        self._update_one(self.att_net, self.att_opt, oA, aA, lpA, vA, advA, retA, who="att")


__all__ = ["PPO"]
