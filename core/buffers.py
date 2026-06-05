"""
Rollout storage and generalized advantage estimation used by the PPO training loop.
"""

import torch


# =============================================================
# PPO Storage & Advantage
# =============================================================
class RolloutBuffer:
    """Fixed-shape PPO rollout buffer with tensors for observations, actions, rewards, and advantages."""
    def __init__(self, obs_dim, act_dim, num_envs, horizon, device):
        """Store configuration and initialize the runtime state for this object."""
        self.N, self.T = num_envs, horizon
        self.device = device
        self.obs  = torch.zeros(self.T, self.N, obs_dim, device=device)
        self.act  = torch.zeros(self.T, self.N, act_dim, device=device)
        self.logp = torch.zeros(self.T, self.N, device=device)
        self.val  = torch.zeros(self.T, self.N, device=device)
        self.rew  = torch.zeros(self.T, self.N, device=device)
        self.done = torch.zeros(self.T, self.N, device=device)
        self.next_val = torch.zeros(self.N, device=device)
        self.ptr = 0

    def add(self, obs, act, logp, val, rew, done):
        """Append one transition batch to the rollout buffer."""
        t = self.ptr

        # Make rollout data "dead" (no grad graph can leak into PPO update)
        self.obs[t]  = obs.detach()
        self.act[t]  = act.detach()
        self.logp[t] = logp.detach()
        self.val[t]  = val.detach()

        self.rew[t]  = rew
        self.done[t] = done
        self.ptr += 1


    def finalize(self, next_val):
        """Compute returns and advantages after the rollout is complete."""
        self.next_val = next_val

    def get(self):
        """Yield minibatches from the stored rollout data."""
        B = self.T * self.N
        obs  = self.obs.reshape(B, -1)
        act  = self.act.reshape(B, -1)
        logp = self.logp.reshape(B)
        val  = self.val.reshape(B)
        rew  = self.rew.reshape(B)
        done = self.done.reshape(B)
        return obs, act, logp, val, rew, done


def compute_gae_from_buffer(buf: RolloutBuffer, gamma: float, lam: float):
    """Compute gae from buffer from the provided rollout or config data."""
    T, N = buf.T, buf.N
    device = buf.device
    adv = torch.zeros(T, N, device=device)
    lastgaelam = torch.zeros(N, device=device)
    next_val = buf.next_val
    for t in reversed(range(T)):
        nonterminal = 1.0 - buf.done[t]
        next_v = next_val if t == T-1 else buf.val[t+1]
        delta = buf.rew[t] + gamma * next_v * nonterminal - buf.val[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = adv + buf.val
    A = adv.reshape(T*N)
    R = ret.reshape(T*N)
    A = (A - A.mean()) / (A.std() + 1e-8)
    return A, R

