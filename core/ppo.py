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
        self.act_scale = float(cfg["umax"])
        self.v_clip_eps = 0.2

        self.attacker_mode = cfg.get("attacker_mode", "rule")

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
        D = int(self.act_scale * 0 + obs_batch.shape[-1] // 5)  # not used except sanity
        D = obs_batch.shape[-1] // 5  # true D from obs layout
        Na = int(getattr(self, "num_attackers", 1) or 1)
        # Better: read Na from cfg once and store it:
        # in PPO.__init__: self.num_attackers = int(cfg.get("num_attackers", 1))
        # then use that here. We'll fallback if missing.
        if hasattr(self, "cfg"):
            Na = int(self.cfg.get("num_attackers", Na))
        elif hasattr(self, "num_attackers"):
            Na = int(self.num_attackers)

        # -------- RULE ATTACKER --------
        if who == "att" and self.attacker_mode == "rule":
            # obs layout: [p1c, pA1c..pANc, rel1..relN, v1, vA1..vAN]
            # We must parse based on Na.
            D = int(self.act_scale * 0 + obs_batch.shape[-1])  # noop keep torch happy
            D = int((obs_batch.shape[1]) // (2 + 2*Na + 1 + Na))  # not reliable
            # Instead use cfg D (robust):
            D = int(self.def_net.layer.D) if hasattr(self.def_net, "layer") else int(obs_batch.shape[1] // 5)

            ob = obs_batch.detach().cpu().numpy()
            # parse p1c
            p1c = ob[:, 0:D]
            off = D

            # pA centered (Na blocks)
            pA_c = []
            for k in range(Na):
                pA_c.append(ob[:, off:off + D])
                off += D

            # rel blocks (Na)
            rels = []
            for k in range(Na):
                rels.append(ob[:, off:off + D])
                off += D

            # v1
            v1 = ob[:, off:off + D]
            off += D

            # vA blocks (Na)
            vA = []
            for k in range(Na):
                vA.append(ob[:, off:off + D])
                off += D

            center = self.rule_ctrl.center

            acts = []
            for i in range(B):
                p1 = p1c[i] + center
                # recover absolute attackers:
                pA_list = [pA_c[k][i] + center for k in range(Na)]
                vA_list = [vA[k][i] for k in range(Na)]
                uA = self.rule_ctrl.act_multi(p1, v1[i], pA_list, vA_list)  # (Na, D)
                acts.append(uA)

            a_np = np.stack(acts, axis=0)  # (B, Na, D)
            a_t = torch.as_tensor(a_np, dtype=obs_batch.dtype, device=obs_batch.device)

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
        advD, retD = compute_gae_from_buffer(buf_def, self.gamma, self.lam)
        oD, aD, lpD, vD, _, _ = buf_def.get()
        self._update_one(self.def_net, self.def_opt, oD, aD, lpD, vD, advD, retD, who="def")

    def update_attacker_only(self, buf_att: RolloutBuffer):
        advA, retA = compute_gae_from_buffer(buf_att, self.gamma, self.lam)
        oA, aA, lpA, vA, _, _ = buf_att.get()
        self._update_one(self.att_net, self.att_opt, oA, aA, lpA, vA, advA, retA, who="att")
