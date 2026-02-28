# policy_smoke_test.py
import os, sys, numpy as np, torch

# --- import your ActorCritic and RLPolicy (the eval wrapper we wrote) ---
from rl_loop_multi import ActorCritic
import rl_infer  # must contain RLPolicy as in our latest version

def _print_param_norms(model, tag):
    with torch.no_grad():
        tot = 0.0
        for k, v in model.state_dict().items():
            tot += v.float().pow(2).sum().item()
        print(f"[{tag}] param L2-norm = {tot**0.5:.4f}")

def quick_policy_check(cfg, obs_mode="relative_centered", det=True, sample_cnt=4):
    """
    1) Loads ppo_def.pt / ppo_att.pt
    2) Prints parameter norms
    3) Probes actions on simple, canonical inputs
    """
    cfg = dict(cfg)
    cfg["obs_mode"] = obs_mode
    # Disable obsnorm unless you have matching stats on disk
    if not cfg.get("obs_stats") and cfg.get("obsnorm"):
        cfg["obsnorm"] = False

    # Build the eval wrapper (does NOT depend on Env rewards)
    pol = rl_infer.RLPolicy(cfg, device=cfg.get("device","cpu"))
    print(f"\n== RLPolicy built on device={pol.device}, D={pol.D} ==")


    # Param norms (sanity: not ~0, not NaN)
    _print_param_norms(pol.pi_def, "pi_def")
    _print_param_norms(pol.pi_att, "pi_att")

    D = pol.D
    umax = cfg.get("umax", 5e-4)

    def pack(p1, v1, p2, v2):
        return rl_infer.RLPolicy.pack_state(p1, v1, p2, v2)

    # -------- canonical test states (tiny batch) --------
    tests = {
        "zeros": pack(np.zeros(D), np.zeros(D), np.zeros(D), np.zeros(D)),
        "offset_pos": pack(np.array([10,0,0][:D]), np.zeros(D), np.array([0,0,0][:D]), np.zeros(D)),
        "closing_vel": pack(np.zeros(D), np.array([-0.1,0,0][:D]), np.array([5,0,0][:D]), np.array([-0.1,0,0][:D])),
        "randomish": pack(np.array([2, -1, 0][:D]), np.array([0.02, 0.00, -0.01][:D]),
                          np.array([-3, 1, 0][:D]), np.array([0.00, -0.03, 0.01][:D])),
    }

    # probe deterministic mean and a few stochastic samples
    for name, s in tests.items():
        aD = pol.act_def(s, deterministic=det)
        aA = pol.act_att(s, deterministic=det)
        print(f"\n[{name}] deterministic:")
        print(f"  def: {aD}   ||a||={np.linalg.norm(aD):.3e}   (umax={umax})")
        print(f"  att: {aA}   ||a||={np.linalg.norm(aA):.3e}")

        # stochastic samples (if mean ~0, samples should still have variance)
        norms_def, norms_att = [], []
        for _ in range(sample_cnt):
            aDs = pol.act_def(s, deterministic=False)
            aAs = pol.act_att(s, deterministic=False)
            norms_def.append(np.linalg.norm(aDs))
            norms_att.append(np.linalg.norm(aAs))
        print(f"  def(samples x{sample_cnt}) ||a|| mean={np.mean(norms_def):.3e}, std={np.std(norms_def):.3e}")
        print(f"  att(samples x{sample_cnt}) ||a|| mean={np.mean(norms_att):.3e}, std={np.std(norms_att):.3e}")

if __name__ == "__main__":
    # Minimal cfg stub: only keys RLPolicy actually uses
    CONFIG_MIN = {
        "D": 3,
        "umax": 5e-4,
        "device": "cpu",
        "arena": {"type": "sphere", "cx": 0.0, "cy": 0.0, "cz": 0.0, "r": 30.0},
        # Turn off obsnorm unless you have stats:
        "obsnorm": False,
        # If you trained with tanh scaling inside the net and multiplied by umax there, set act_squash=False.
        # If the net outputs unbounded gaussians and you tanh*umax outside, set act_squash=True.
        "act_squash": False,
        # deterministic vs sample
        "rl_eval_deterministic": True,
    }

    # Quick existence check for ckpts
    print("Has ppo_def.pt:", os.path.exists("ppo_def.pt"),
          "| Has ppo_att.pt:", os.path.exists("ppo_att.pt"))

    # Try both common observation mappings, back-to-back:
    print("\n=== Probe with obs_mode='relative_centered' ===")
    quick_policy_check(CONFIG_MIN, obs_mode="relative_centered", det=True)

    print("\n=== Probe with obs_mode='concat_raw' ===")
    quick_policy_check(CONFIG_MIN, obs_mode="concat_raw", det=True)
