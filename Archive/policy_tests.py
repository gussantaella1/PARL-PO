# policy_tests.py
# =========================
# Policy test suite
# =========================
from __future__ import annotations
import numpy as np
import torch

# --- try to get Env from caller, else fall back to rl_loop ---
try:
    Env  # type: ignore[name-defined]
except NameError:  # pragma: no cover
    from rl_loop_multi import Env  # avoids NameError when run standalone

# --- small helpers ---
def explained_variance(y_pred: np.ndarray, y_true: np.ndarray):
    # 1 - Var(y - yhat)/Var(y)
    var_y = np.var(y_true)
    return np.nan if var_y < 1e-12 else 1.0 - np.var(y_true - y_pred) / (var_y + 1e-12)

def pack_obs_from_state(env: Env) -> np.ndarray:
    p1, v1, p2, v2 = env._unpack(env.state)
    return np.concatenate([p1 - env.center, p2 - env.center, (p2 - p1), v1, v2]).astype(np.float32)

def _policy_act_mean(ppo, obs_np: np.ndarray, who: str, act_scale: float) -> np.ndarray:
    o = torch.as_tensor(obs_np[None, :], dtype=torch.float32, device=ppo.device)
    net = ppo.def_net if who == "def" else ppo.att_net
    with torch.no_grad():
        dist = net.dist(o)                  # Normal(μ, σ)
        mu_raw = dist.mean                  # (1, D)
        a = torch.tanh(mu_raw) * act_scale  # squash to env scale
    return a.squeeze(0).cpu().numpy()

# ---------- A) Directionality probe (stateless) ----------
def probe_directionality(cfg, ppo, num_samples=200):
    """
    Samples synthetic states with the attacker around center and checks:
      def: grad(d2_att)·a_def  should be > 0 on average (push attacker outward)
      att: grad(d2_att)·a_att  should be < 0 on average (pull inward)
    """
    D = int(cfg["D"])
    act_scale = float(cfg["umax"])
    center = np.array([
        cfg["arena"]["cx"],
        cfg["arena"]["cy"],
        (cfg["arena"].get("cz", 0.0) if D == 3 else 0.0)
    ], float)[:D]
    radius = float(cfg["arena"]["r"])

    rng = np.random.default_rng(0)
    signs_def, signs_att = [], []
    mags_def, mags_att = [], []

    for _ in range(num_samples):
        # attacker position on [0.2R, 0.8R]
        dir_vec = rng.normal(size=D); dir_vec /= (np.linalg.norm(dir_vec) + 1e-9)
        r = rng.uniform(0.2 * radius, 0.8 * radius)
        p_att = center + r * dir_vec
        v_att = rng.normal(scale=0.05, size=D)

        # defender elsewhere
        p_def = center + rng.normal(scale=0.5 * radius, size=D)
        v_def = rng.normal(scale=0.05, size=D)

        obs = np.concatenate([p_def - center, p_att - center, p_att - p_def, v_def, v_att]).astype(np.float32)

        # grad wrt attacker position of d^2_att = ||p_att - c||^2
        grad = 2.0 * (p_att - center)

        a_def = _policy_act_mean(ppo, obs, who="def", act_scale=act_scale)
        a_att = _policy_act_mean(ppo, obs, who="att", act_scale=act_scale)

        s_def = np.sign(np.dot(grad, a_def[:D]))
        s_att = np.sign(np.dot(grad, a_att[:D]))
        signs_def.append(s_def); signs_att.append(s_att)
        mags_def.append(np.dot(grad, a_def[:D])); mags_att.append(np.dot(grad, a_att[:D]))

    frac_def_pos = np.mean(np.array(signs_def) > 0.0)
    frac_att_neg = np.mean(np.array(signs_att) < 0.0)
    return {
        "def_frac_push_out": float(frac_def_pos),
        "att_frac_pull_in": float(frac_att_neg),
        "def_avg_grad_dot_action": float(np.mean(mags_def)),
        "att_avg_grad_dot_action": float(np.mean(mags_att)),
    }

# ---------- B) Static-opponent rollout ----------
def eval_vs_static(cfg, ppo, active="def", episodes=50, steps=60):
    """
    Measures Δ d2_attacker = d2_T - d2_0 with exactly one active agent
    and the opponent clamped to zero action.
    Expectation: def => Δd2 > 0 ; att => Δd2 < 0 (on average).
    """
    env = Env(cfg)
    act_scale = float(cfg["umax"])
    D = env.D
    center = env.center
    d2_changes, anorms = [], []

    for _ in range(episodes):
        env.reset()
        # baseline attacker distance^2
        _, _, p2, _ = env._unpack(env.state)
        d2_0 = float(np.dot(p2 - center, p2 - center))

        for _k in range(steps):
            obs = pack_obs_from_state(env)
            if active == "def":
                a_def = _policy_act_mean(ppo, obs, "def", act_scale)
                a_att = np.zeros(D, float)
            else:
                a_def = np.zeros(D, float)
                a_att = _policy_act_mean(ppo, obs, "att", act_scale)
            _, _, _, done, _ = env.step(a_def, a_att)
            anorms.append(np.linalg.norm(a_def if active == "def" else a_att))
            if done:
                break

        _, _, p2, _ = env._unpack(env.state)
        d2_T = float(np.dot(p2 - center, p2 - center))
        d2_changes.append(d2_T - d2_0)

    return {
        "mean_final_d2_change": float(np.mean(d2_changes)),
        "mean_action_norm": float(np.mean(anorms) if anorms else 0.0),
    }

# ---------- C) Random-opponent rollout ----------
def eval_vs_random(cfg, ppo, active="def", episodes=50, steps=60, rand_scale=1.0):
    env = Env(cfg)
    D = env.D
    act_scale = float(cfg["umax"])
    rng = np.random.default_rng(1)

    score = 0.0
    for _ in range(episodes):
        env.reset()
        ret = 0.0
        for _k in range(steps):
            obs = pack_obs_from_state(env)
            a_act = _policy_act_mean(ppo, obs, active, act_scale)
            a_opp = rng.normal(scale=rand_scale * act_scale, size=D)
            a_opp = np.clip(a_opp, -act_scale, +act_scale)
            _, r_def, r_att, done, _ = env.step(
                a_act if active == "def" else a_opp,
                a_act if active == "att" else a_opp
            )
            ret += (r_def if active == "def" else r_att)
            if done:
                break
        score += ret
    return {"avg_return": float(score / episodes)}

# ---------- D) Critic explained variance ----------
def critic_explained_variance(cfg, ppo, rollout_episodes=10, steps=60):
    env = Env(cfg)
    gamma = float(cfg["gamma"])
    vals, rets = [], []
    for _ in range(rollout_episodes):
        env.reset()
        ep_rew = []
        with torch.no_grad():
            obs0 = pack_obs_from_state(env)
            o0 = torch.as_tensor(obs0[None, :], dtype=torch.float32, device=ppo.device)
            v_def0 = ppo.def_net.value(o0).item()
            v_att0 = ppo.att_net.value(o0).item()
        done = False
        t = 0
        while (not done) and (t < steps):
            obs = pack_obs_from_state(env)
            a1 = _policy_act_mean(ppo, obs, "def", float(cfg["umax"]))
            a2 = _policy_act_mean(ppo, obs, "att", float(cfg["umax"]))
            _, r1, r2, done, _ = env.step(a1, a2)
            ep_rew.append((r1, r2))
            t += 1
        # Monte Carlo returns from t=0
        G1 = 0.0; G2 = 0.0
        for (r1, r2) in reversed(ep_rew):
            G1 = r1 + gamma * G1
            G2 = r2 + gamma * G2
        vals.append((v_def0, v_att0))
        rets.append((G1, G2))
    vals = np.asarray(vals)  # (E, 2)
    rets = np.asarray(rets)
    ev_def = explained_variance(vals[:, 0], rets[:, 0])
    ev_att = explained_variance(vals[:, 1], rets[:, 1])
    return {"ev_def": float(ev_def), "ev_att": float(ev_att)}

# ---------- E) Action histogram (dead-policy detector) ----------
def action_stats(cfg, ppo, samples=200):
    env = Env(cfg)
    act_scale = float(cfg["umax"])
    norms_def, norms_att = [], []
    for _ in range(samples):
        env.reset()
        obs = pack_obs_from_state(env)
        a1 = _policy_act_mean(ppo, obs, "def", act_scale)
        a2 = _policy_act_mean(ppo, obs, "att", act_scale)
        norms_def.append(np.linalg.norm(a1))
        norms_att.append(np.linalg.norm(a2))
    return {
        "def_mean_norm": float(np.mean(norms_def)),
        "att_mean_norm": float(np.mean(norms_att)),
        "def_frac_near_zero": float(np.mean(np.array(norms_def) < 1e-6)),
        "att_frac_near_zero": float(np.mean(np.array(norms_att) < 1e-6)),
    }

# ---------- Master runner ----------
def run_policy_tests(cfg, ppo):
    print("\n[TEST] Directionality probe …")
    print(probe_directionality(cfg, ppo, num_samples=300))

    print("\n[TEST] vs STATIC opponent (def active) …")
    print(eval_vs_static(cfg, ppo, active="def", episodes=60, steps=min(60, cfg["T"])))

    print("\n[TEST] vs STATIC opponent (att active) …")
    print(eval_vs_static(cfg, ppo, active="att", episodes=60, steps=min(60, cfg["T"])))

    print("\n[TEST] vs RANDOM opponent (def active) …")
    print(eval_vs_random(cfg, ppo, active="def", episodes=50, steps=min(60, cfg["T"]), rand_scale=0.7))

    print("\n[TEST] vs RANDOM opponent (att active) …")
    print(eval_vs_random(cfg, ppo, active="att", episodes=50, steps=min(60, cfg["T"]), rand_scale=0.7))

    print("\n[TEST] Critic explained variance …")
    print(critic_explained_variance(cfg, ppo, rollout_episodes=12, steps=min(60, cfg["T"])))

    print("\n[TEST] Action stats …")
    print(action_stats(cfg, ppo, samples=200))
