import gc
import os
import time
from typing import Any, Dict

import numpy as np
import torch

from config_rl import build_dyn, config_for_train
from core.buffers import RolloutBuffer
from core.distill import distill_from_teacher
from core.env import Env, VecEnv, collect_ic_history_from_vecenv
from core.plotting import make_tb_writer
from core.ppo import PPO
from core.utils import set_seed
from core.freeze_utils import freeze_module_, snapshot_state_dict, assert_frozen_unchanged, assert_deterministic_action

# =============================================================
# Training & Evaluation
# =============================================================


def _cpu_state_dict(m: torch.nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in m.state_dict().items()}

def _save_role_checkpoint(ppo, train_role: str, path: str):
    if train_role == "def":
        net = ppo.def_net
    elif train_role == "att":
        net = ppo.att_net
        if net is None:
            raise RuntimeError("Tried to save attacker checkpoint, but ppo.att_net is None.")
    else:
        raise ValueError(f"Unknown train_role={train_role!r}")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(_cpu_state_dict(net), path)

def train(cfg: Dict[str, Any]):



    set_seed(cfg["seed"])
    device = cfg["device"]

    writer = None
    tb_logdir = None
    global_env_step = 0

    if cfg.get("use_tensorboard", False):
        writer, tb_logdir = make_tb_writer(cfg)

        # Optional: show model graphs once (often noisy / can break with custom ops)
        # writer.add_text("notes", "PPO Diffgame run", 0)

    train_role = cfg.get("train_role", "def")  # <-- NEW

    if train_role == "def":
        reward_mode = "def"
    elif train_role == "att":
        reward_mode = "att"

    # -------------------------
    # Checkpoint saving config
    # -------------------------
    save_best_ckpt = bool(cfg.get("save_best_ckpt", True))
    save_last_ckpt = bool(cfg.get("save_last_ckpt", True))
    checkpoint_dir = cfg.get("checkpoint_dir", None)
    checkpoint_prefix = cfg.get("checkpoint_prefix", f"{train_role}_teacher")

    tracked_metric_name = "R_def_mean" if train_role == "def" else "R_att_mean"
    best_metric = -float("inf")
    best_update = None
    best_ckpt_path = None
    last_ckpt_path = None

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    def make_env():
        return Env(cfg)

    num_envs = int(cfg.get("num_envs"))
    steps_per_env = int(cfg.get("steps_per_env"))
    total_updates = int(cfg.get("total_updates"))
    log_every = int(cfg.get("log_every"))

    vec = VecEnv(make_env, num_envs)
    obs_dim = vec.obs.shape[1]
    act_dim = int(cfg["D"])

    ppo = PPO(obs_dim, act_dim, cfg, device=device)

    # Optional: initialize attacker from a BC checkpoint (before PPO training)
    att_init = cfg.get("att_init_path", None)
    if att_init is not None:
        if ppo.att_net is None:
            raise RuntimeError("att_init_path provided but attacker_mode != 'rl'")
        state = torch.load(att_init, map_location=device)
        ppo.att_net.load_state_dict(state)
        print(f"[train] Loaded attacker init from: {att_init}")


    # Optional: load fixed defender
    def_ckpt = cfg.get("def_ckpt_path", None)
    if def_ckpt is not None:
        state = torch.load(def_ckpt, map_location=device)
        ppo.def_net.load_state_dict(state)
        # If defender should be frozen:
        if cfg.get("freeze_defender", False):
            for p in ppo.def_net.parameters():
                p.requires_grad_(False)

    # Optional: load fixed attacker
    att_ckpt = cfg.get("att_ckpt_path", None)
    if att_ckpt is not None and ppo.att_net is not None:
        state = torch.load(att_ckpt, map_location=device)
        ppo.att_net.load_state_dict(state)
        if cfg.get("freeze_attacker", False):
            for p in ppo.att_net.parameters():
                p.requires_grad_(False)

    # Freeze whichever role we do NOT train in this phase
    if cfg.get("freeze_defender", False):
        freeze_module_(ppo.def_net)

    if cfg.get("freeze_attacker", False) and (ppo.att_net is not None):
        freeze_module_(ppo.att_net)

    # =========================================================
    # NEW: Freeze verification snapshots (opponents must not move)
    # =========================================================
    verify_freeze = bool(cfg.get("verify_freeze"))
    freeze_tol = float(cfg.get("freeze_tol"))  # set 0.0 for exact; or 1e-12 if you’re paranoid

    snap_def = None
    snap_att = None

    if verify_freeze and cfg.get("freeze_defender"):
        # defender is frozen opponent in attacker-training
        snap_def = snapshot_state_dict(ppo.def_net)

    if verify_freeze and cfg.get("freeze_attacker") and (ppo.att_net is not None):
        # attacker is frozen opponent in defender-training
        snap_att = snapshot_state_dict(ppo.att_net)




    # NEW: LR schedule config
    lr_schedule = cfg.get("lr_schedule", "none")
    lr_final_factor = float(cfg.get("lr_final_factor", 0.1))

    # ---- NEW: metrics container ----
    metrics = {
        "update": [],
        "R_def_mean": [],
        "R_att_mean": [],
        "muD_abs_mean": [],
        "stdD_mean": [],
        "d1_mean": [],          # defender true distance
        "d2_mean": [],          # attacker belief distance (what obs sees)
        "d2_true_mean": [],     # attacker true distance
        "meas_innov_mean": [],  # optional
        "ukf_trPpos_mean": [],  # optional
        "lr_pi": [],
        "lr_vf": [],
    }

    if cfg.get("fuel", {}).get("enable", False):
        metrics["fuel_used_def_mean"] = []
        metrics["fuel_used_att_mean"] = []
        metrics["fuel_frac_def_mean"] = []
        metrics["fuel_frac_att_mean"] = []

        # optional
        metrics["thrust_def_mean"] = []
        metrics["thrust_att_mean"] = []
        metrics["mdot_def_mean"] = []
        metrics["mdot_att_mean"] = []



    # Optional anneal of defender center tether
    def_center_base = cfg.get("def_center_coef", 0.0)
    min_anneal = float(cfg.get("def_center_min_anneal", 0.5))

    for upd in range(1, total_updates + 1):
        term_counts = {"oob_def":0, "oob_att":0, "hit_target":0, "collision":0}


        # ---------- optional LR decay (linear) ----------
        if lr_schedule == "linear":
            # frac_lr goes 0 → 1 over training
            frac_lr = (upd - 1) / max(1, total_updates - 1)
            scale = 1.0 - frac_lr * (1.0 - lr_final_factor)

            # defender
            for g, base in zip(ppo.def_opt.param_groups, ppo.def_base_lrs):
                g["lr"] = base * scale

            # attacker RL (if you ever turn it on)
            if ppo.att_opt is not None and getattr(ppo, "att_base_lrs", None) is not None:
                for g, base in zip(ppo.att_opt.param_groups, ppo.att_base_lrs):
                    g["lr"] = base * scale
        # ------------------------------------------------

        # Linear anneal multiplier from 1.0 → min_anneal for k_cent
        center_frac = upd / max(1, total_updates)
        k_cent_mul = 1.0 - (1.0 - min_anneal) * center_frac
        for e in vec.envs:
            e.k_cent = def_center_base * k_cent_mul

        # Buffers
        bufD = RolloutBuffer(obs_dim, act_dim, num_envs, steps_per_env, device)
        rule_att = (cfg.get("attacker_mode", "rule") == "rule")
        if (not rule_att) and (train_role in ("att", "both")):
            bufA = RolloutBuffer(obs_dim, act_dim, num_envs, steps_per_env, device)
        else:
            bufA = None


        o = torch.as_tensor(vec.obs, dtype=torch.float32, device=device)
        ep_ret_def = np.zeros(num_envs, dtype=np.float64)
        ep_ret_att = np.zeros(num_envs, dtype=np.float64)

        # accumulators for metrics over this update
        d1_true_acc = 0.0
        d2_true_acc = 0.0
        d2_belief_acc = 0.0
        meas_innov_acc = 0.0
        trP_acc = 0.0
        info_count = 0

        if cfg.get("fuel", {}).get("enable", False):
            fuel_used_def_acc = 0.0
            fuel_used_att_acc = 0.0
            fuel_frac_def_acc = 0.0
            fuel_frac_att_acc = 0.0

            thrust_def_acc = 0.0
            thrust_att_acc = 0.0
            mdot_def_acc = 0.0
            mdot_att_acc = 0.0
            fuel_info_count = 0

        for _ in range(steps_per_env):
            with torch.no_grad():
                det_def = (train_role != "def")   # defender is opponent unless training defender
                det_att = (train_role != "att")   # attacker is opponent unless training attacker

                a1, lp1, v1 = ppo.act(o, who="def", deterministic=det_def)
                a2, lp2, v2 = ppo.act(o, who="att", deterministic=det_att)


            a1_np = a1.cpu().numpy()

            a2_np = a2.cpu().numpy()
            # normalize shapes to what Env expects:
            # - if attacker RL: (B, D) ok
            # - if rule attacker: (B, Na, D) ok
            # - if rule attacker but produced (B, D): expand to (B,1,D)
            if a2_np.ndim == 2 and cfg.get("attacker_mode", "rule") == "rule" and cfg.get("num_attackers", 1) == 1:
                a2_np = a2_np[:, None, :]

            o2_np, r1_np, r2_np, d_np, infos = vec.step(
                a1_np,
                a2_np,
                reward_mode=reward_mode,
            )


            o2 = torch.as_tensor(o2_np, dtype=torch.float32, device=device)
            r1 = torch.as_tensor(r1_np, dtype=torch.float32, device=device)
            r2 = torch.as_tensor(r2_np, dtype=torch.float32, device=device)
            d  = torch.as_tensor(d_np,  dtype=torch.float32, device=device)

            bufD.add(o.detach(), a1.detach(), lp1.detach(), v1.detach(), r1, d)
            if bufA is not None:
                bufA.add(o.detach(), a2.detach(), lp2.detach(), v2.detach(), r2, d)


            if train_role == "def":
                ep_ret_def += r1_np

            if train_role == "att":
                ep_ret_att += r2_np
            
            o = o2

            # ---- accumulate truth / belief metrics from Env.info ----
            for inf in infos:
                # count every env-step
                info_count += 1

                # always accumulate if present
                if "d1_true_norm" in inf:
                    d1_true_acc += inf["d1_true_norm"]

                if "d2_true_norm" in inf:
                    d2_true_acc += inf["d2_true_norm"]

                # belief distance: fall back safely
                if "d2_belief_norm" in inf:
                    d2_belief_acc += inf["d2_belief_norm"]
                elif "d2_true_norm" in inf:
                    d2_belief_acc += inf["d2_true_norm"]
                elif "d2_norm" in inf:
                    d2_belief_acc += inf["d2_norm"]

                if "meas_innov_sq" in inf:
                    meas_innov_acc += inf["meas_innov_sq"]

                if "ukf_trPpos" in inf:
                    trP_acc += inf["ukf_trPpos"]

                if inf.get("oob_def", False): term_counts["oob_def"] += 1
                if inf.get("oob_att", False): term_counts["oob_att"] += 1
                if inf.get("hit_target", False): term_counts["hit_target"] += 1
                if inf.get("collision", False): term_counts["collision"] += 1

                if cfg.get("fuel", {}).get("enable", False):
                    if "fuel_used_def" in inf:
                        fuel_used_def_acc += inf["fuel_used_def"]
                        fuel_used_att_acc += inf["fuel_used_att"]
                        fuel_frac_def_acc += inf["fuel_frac_def"]
                        fuel_frac_att_acc += inf["fuel_frac_att"]

                        thrust_def_acc += inf.get("thrust_def", 0.0)
                        thrust_att_acc += inf.get("thrust_att", 0.0)
                        mdot_def_acc += inf.get("mdot_def", 0.0)
                        mdot_att_acc += inf.get("mdot_att", 0.0)

                        fuel_info_count += 1


            global_env_step += num_envs




        with torch.no_grad():
            next_v_def = ppo.def_net.value(o)
        bufD.finalize(next_v_def)

        if bufA is not None:
            with torch.no_grad():
                next_v_att = ppo.att_net.value(o)
            bufA.finalize(next_v_att)

        # ---- choose what to update ----
        if train_role == "def":
            ppo.update_defender_only(bufD)

        elif train_role == "att":
            if bufA is None:
                raise RuntimeError("train_role='att' requires attacker_mode='rl'")
            ppo.update_attacker_only(bufA)

        else:
            raise ValueError(f"Unknown train_role={train_role!r}")
        
        #explicit cleanup (place it right here)
        # del bufD
        # try:
        #     del bufA
        # except NameError:
        #     pass

        # gc.collect()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
                
        # =========================================================
        # NEW: Verify frozen opponent did not change
        # =========================================================
        if verify_freeze:
            # If training attacker, defender should be frozen (Phase 1)
            if (train_role == "att") and (snap_def is not None):
                assert_frozen_unchanged(snap_def, ppo.def_net, name="frozen_defender", tol=freeze_tol)

            # If training defender, attacker should be frozen (Phase 2)
            if (train_role == "def") and (snap_att is not None):
                assert_frozen_unchanged(snap_att, ppo.att_net, name="frozen_attacker", tol=freeze_tol)



        if upd % log_every == 0:
            R_def_mean = ep_ret_def.mean()
            R_att_mean = ep_ret_att.mean()

            tracked_metric_value = R_def_mean if train_role == "def" else R_att_mean

            if save_best_ckpt and checkpoint_dir is not None:
                if tracked_metric_value > best_metric:
                    best_metric = float(tracked_metric_value)
                    best_update = int(upd)
                    best_ckpt_path = os.path.join(
                        checkpoint_dir,
                        f"{checkpoint_prefix}__best.pt"
                    )
                    _save_role_checkpoint(ppo, train_role, best_ckpt_path)
                    print(
                        f"[checkpoint] new best {tracked_metric_name}={best_metric:+.3f} "
                        f"at update {best_update} -> {best_ckpt_path}"
                    )

            # means over all steps collected this update
            if info_count > 0:
                R = cfg["arena"]["r"]

                d1_true_mean = np.sqrt(d1_true_acc / info_count) * R
                d2_true_mean = np.sqrt(d2_true_acc / info_count) * R
                d2_belief_mean = np.sqrt(d2_belief_acc / info_count) * R
                meas_innov_mean = meas_innov_acc / info_count
                trP_mean = trP_acc / info_count
            else:
                d1_true_mean = d2_true_mean = d2_belief_mean = 0.0
                meas_innov_mean = trP_mean = 0.0



            with torch.no_grad():
                flat_obs = bufD.obs.reshape(-1, obs_dim)
                distD = ppo.def_net.dist(flat_obs, who="def")
                muD = distD.mean.abs().mean().item()
                stdD = distD.stddev.mean().item()

                if verify_freeze:
                    # Check determinism of the opponent (not the learner)
                    if train_role == "att":
                        # opponent is defender
                        assert_deterministic_action(ppo, flat_obs[:256], who="def", tol=0.0)
                    if train_role == "def" and (ppo.att_net is not None):
                        # opponent is attacker (RL opponent case)
                        assert_deterministic_action(ppo, flat_obs[:256], who="att", tol=0.0)


                # obs = [p1c, p2c, rel, v1, v2]
                Dcfg = cfg["D"]
                p1c = flat_obs[:, :Dcfg]
                p2c = flat_obs[:, Dcfg:2*Dcfg]

                # -------------------------------
                # #5: opponent policy std logging
                # -------------------------------
                dist_opp = (
                    ppo.def_net.dist(flat_obs, who="def")
                    if train_role == "att"  # training attacker => opponent is defender
                    else ppo.att_net.dist(flat_obs, who="att")
                    if (train_role == "def" and ppo.att_net is not None)  # training defender => opponent is attacker RL
                    else None
                )
                if dist_opp is not None:
                    print("opp std mean:", dist_opp.stddev.mean().item())

                # ... your obs-derived d1/d2 means ...
                Dcfg = cfg["D"]
                p1c = flat_obs[:, :Dcfg]
                p2c = flat_obs[:, Dcfg:2*Dcfg]
                d1_obs_mean = p1c.pow(2).sum(-1).mean().sqrt().item()
                d2_obs_mean = p2c.pow(2).sum(-1).mean().sqrt().item()


            # grab learning rates (assuming two param groups: policy+logstd and value)
            lr_pi = ppo.def_opt.param_groups[0]["lr"]
            lr_vf = ppo.def_opt.param_groups[-1]["lr"]

            if cfg.get("fuel", {}).get("enable", False):
                if fuel_info_count > 0:
                    fuel_used_def_mean = fuel_used_def_acc / fuel_info_count
                    fuel_used_att_mean = fuel_used_att_acc / fuel_info_count
                    fuel_frac_def_mean = fuel_frac_def_acc / fuel_info_count
                    fuel_frac_att_mean = fuel_frac_att_acc / fuel_info_count

                    thrust_def_mean = thrust_def_acc / fuel_info_count
                    thrust_att_mean = thrust_att_acc / fuel_info_count
                    mdot_def_mean = mdot_def_acc / fuel_info_count
                    mdot_att_mean = mdot_att_acc / fuel_info_count
                else:
                    fuel_used_def_mean = fuel_used_att_mean = 0.0
                    fuel_frac_def_mean = fuel_frac_att_mean = 0.0
                    thrust_def_mean = thrust_att_mean = 0.0
                    mdot_def_mean = mdot_att_mean = 0.0

                metrics["fuel_used_def_mean"].append(fuel_used_def_mean)
                metrics["fuel_used_att_mean"].append(fuel_used_att_mean)
                metrics["fuel_frac_def_mean"].append(fuel_frac_def_mean)
                metrics["fuel_frac_att_mean"].append(fuel_frac_att_mean)

                metrics["thrust_def_mean"].append(thrust_def_mean)
                metrics["thrust_att_mean"].append(thrust_att_mean)
                metrics["mdot_def_mean"].append(mdot_def_mean)
                metrics["mdot_att_mean"].append(mdot_att_mean)


            # ---- store in metrics ----
            metrics["update"].append(upd)
            metrics["R_def_mean"].append(R_def_mean)
            metrics["R_att_mean"].append(R_att_mean)
            metrics["muD_abs_mean"].append(muD)
            metrics["stdD_mean"].append(stdD)

            metrics["d1_mean"].append(d1_true_mean)
            metrics["d2_mean"].append(d2_belief_mean)
            metrics["d2_true_mean"].append(d2_true_mean)
            metrics["meas_innov_mean"].append(meas_innov_mean)
            metrics["ukf_trPpos_mean"].append(trP_mean)

            metrics["lr_pi"].append(lr_pi)
            metrics["lr_vf"].append(lr_vf)

            print(f"[update {upd:05d}] R_def_mean={R_def_mean:+.3f}  R_att_mean={R_att_mean:+.3f}  (batch={num_envs*steps_per_env})")
            print(f"   [def] |mu|_mean={muD:.3e}  std_mean={stdD:.3e}")
            print(f"   approx true <||p1-center||> ≈ {d1_true_mean:.3f}")
            print(f"   approx true <||p2-center||> ≈ {d2_true_mean:.3f}")
            if cfg.get("use_ukf", False):
                print(f"   approx belief <||p2-center||> ≈ {d2_belief_mean:.3f}")
                print(f"   meas_innov_mean={meas_innov_mean:.3e},  trPpos_mean={trP_mean:.3e}")

            if cfg.get("fuel", {}).get("enable", False):
                print(
                    f"   fuel used: def={fuel_used_def_mean:.6f}, att={fuel_used_att_mean:.6f}   "
                    f"fuel remaining: def={fuel_frac_def_mean:.6f}, att={fuel_frac_att_mean:.6f}"
                )
                print(
                    f"   thrust mean: def={thrust_def_mean:.6e}, att={thrust_att_mean:.6e}   "
                    f"mdot mean: def={mdot_def_mean:.6e}, att={mdot_att_mean:.6e}"
                )

            
            if writer is not None:
                gs = global_env_step  # x-axis = env steps

                # ===== Returns =====
                writer.add_scalar("returns/def_mean", R_def_mean, gs)
                writer.add_scalar("returns/att_mean", R_att_mean, gs)

                # ===== Distances (meters) =====
                writer.add_scalar("dist/def_true_p1_to_center_m", d1_true_mean, gs)
                writer.add_scalar("dist/att_true_p2_to_center_m", d2_true_mean, gs)
                writer.add_scalar("dist/att_belief_p2_to_center_m", d2_belief_mean, gs)

                # ===== Policy stats =====
                writer.add_scalar("policy/def_mu_abs_mean", muD, gs)
                writer.add_scalar("policy/def_std_mean", stdD, gs)

                # ===== Learning rates =====
                writer.add_scalar("lr/def_policy", lr_pi, gs)
                writer.add_scalar("lr/def_value",  lr_vf, gs)

                # ===== UKF stats (if enabled) =====
                if cfg.get("use_ukf", False):
                    writer.add_scalar("ukf/meas_innov_sq_mean", meas_innov_mean, gs)
                    writer.add_scalar("ukf/trP_pos_mean", trP_mean, gs)

                if info_count > 0:
                    term_rates = {k: v / info_count for k, v in term_counts.items()}
                else:
                    term_rates = {k: 0.0 for k in term_counts}


                writer.add_scalar("term_rate/oob_def", term_rates["oob_def"], gs)
                writer.add_scalar("term_rate/oob_att", term_rates["oob_att"], gs)
                writer.add_scalar("term_rate/hit_target", term_rates["hit_target"], gs)
                writer.add_scalar("term_rate/collision", term_rates["collision"], gs)

                writer.add_scalar("act/def_abs_mean", a1.abs().mean().item(), global_env_step)
                writer.add_scalar("act/def_abs_max",  a1.abs().max().item(),  global_env_step)

                if cfg.get("fuel", {}).get("enable", False):
                    writer.add_scalar("fuel/used_def_mean", fuel_used_def_mean, gs)
                    writer.add_scalar("fuel/used_att_mean", fuel_used_att_mean, gs)
                    writer.add_scalar("fuel/remaining_def_mean", fuel_frac_def_mean, gs)
                    writer.add_scalar("fuel/remaining_att_mean", fuel_frac_att_mean, gs)

                    writer.add_scalar("fuel/thrust_def_mean", thrust_def_mean, gs)
                    writer.add_scalar("fuel/thrust_att_mean", thrust_att_mean, gs)
                    writer.add_scalar("fuel/mdot_def_mean", mdot_def_mean, gs)
                    writer.add_scalar("fuel/mdot_att_mean", mdot_att_mean, gs)



    ic_used_path = None
    if cfg.get("record_ic_history", False) and checkpoint_dir is not None:
        def_used, att_used = collect_ic_history_from_vecenv(vec)
        ic_used_path = os.path.join(checkpoint_dir, f"ic_samples_{checkpoint_prefix}.npz")
        np.savez(
            ic_used_path,
            def_pos=def_used,
            att_pos=att_used,
        )
        print(f"[ic] saved actual training IC samples -> {ic_used_path}")

            
            
    # ---- end-of-train cleanup ----
    try:
        del bufD
    except: pass
    try:
        del bufA
    except: pass
    try:
        del vec
    except: pass


    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if writer is not None:
        writer.flush()
        writer.close()

    if save_last_ckpt and checkpoint_dir is not None:
        last_ckpt_path = os.path.join(
            checkpoint_dir,
            f"{checkpoint_prefix}.pt"
        )
        _save_role_checkpoint(ppo, train_role, last_ckpt_path)
        print(f"[checkpoint] saved last checkpoint -> {last_ckpt_path}")

    ckpt_info = {
        "tracked_metric_name": tracked_metric_name,
        "best_metric": best_metric,
        "best_update": best_update,
        "best_ckpt_path": best_ckpt_path,
        "last_ckpt_path": last_ckpt_path,
        "ic_used_path": ic_used_path,

    }

    print("Training finished.")
    return ppo, metrics, ckpt_info


def evaluate(ppo: PPO, cfg: Dict[str, Any], episodes: int = 2):
    env = Env(cfg)
    trajs = []
    for _ in range(episodes):
        obs = env.reset()
        states = [env.state.copy()]
        actions = []
        infos = []

        done = False
        while not done:
            o_t = torch.as_tensor(obs[None, :], dtype=torch.float32, device=ppo.def_net.logstd.device)
            with torch.no_grad():
                a1, _, _ = ppo.act(o_t, who="def")
                a2, _, _ = ppo.act(o_t, who="att")
            a1_np = a1.squeeze(0).cpu().numpy()
            a2_np = a2.squeeze(0).cpu().numpy()
            obs, r1, r2, done, info = env.step(a1_np, a2_np)
            states.append(env.state.copy())
            actions.append((a1_np.copy(), a2_np.copy()))
            infos.append(info)
        trajs.append({"states": np.stack(states), "actions": actions, "infos": infos})
    return trajs

    # ---------------------------------------------------------
    # Helper: train defender teacher + distill to UKF student
    # ---------------------------------------------------------

def train_with_distill(
    phase_name: str,
    attacker_mode: str,
    train_role: str,
    out_dir: str,
    extra_train_cfg: Dict[str, Any] | None = None,
):
    """
    Unified wrapper for:
      - teacher training (full-state)
      - optional UKF student distillation

    Replaces both:
      - train_defender_with_distill(...)
      - train_attacker_with_distill(...)

    Args:
        phase_name: label used for checkpoints / metrics filenames
        attacker_mode: "rule" or "rl"
        train_role: "def" or "att"
        extra_train_cfg: optional overrides merged into both teacher and student cfgs

    Returns
    -------
    teacher_ckpt : str
        Path to best teacher checkpoint (or last if best missing)
    student_ckpt : str | None
        Path to distilled student checkpoint if distill=True, else None
    """
    if train_role not in ("def", "att"):
        raise ValueError(f"train_role must be 'def' or 'att', got {train_role!r}")

    role_upper = "DEFENDER" if train_role == "def" else "ATTACKER"
    role_lower = "defender" if train_role == "def" else "attacker"

    # =========================================================
    # TEACHER (full-state)
    # =========================================================
    cfg_teacher = config_for_train(
        attacker_mode=attacker_mode,
        train_role=train_role,
    )
    cfg_teacher["use_ukf"] = False  # teacher is always full-state

    if extra_train_cfg is not None:
        cfg_teacher.update(extra_train_cfg)

    # if cfg_teacher["train_ic_mode"] == "random_shell_advantage":
    #     if train_role == "att": 
    #         cfg_teacher["r_att_min"] = 0.0
    #         cfg_teacher["train_ic_mode"] = "random_shell"

    DISTILL = bool(cfg_teacher.get("distill", False))

    build_dyn(cfg_teacher)

    # dynamics_config = cfg_teacher["dyn"]
    # print(dynamics_config["Ad"])
    # print(dynamics_config["Bd"])

    # raise("Debug")

    if cfg_teacher["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"[{phase_name.upper()} TEACHER] device='cuda' but CUDA not available."
        )

    print(f"[{phase_name.upper()} TEACHER] Using device: {cfg_teacher['device']}")
    print(f"[{phase_name.upper()}] train_role={train_role}  distill={DISTILL}")

    cfg_teacher["checkpoint_dir"] = out_dir
    cfg_teacher["checkpoint_prefix"] = phase_name + "_teacher"

    ppo_teacher, metrics_teacher, ckpt_info = train(cfg_teacher)

    teacher_ckpt = ckpt_info["last_ckpt_path"]
    teacher_ckpt_best = ckpt_info["best_ckpt_path"]

    print(f"[{phase_name.upper()} TEACHER] Using {role_lower} checkpoint: {teacher_ckpt}")
    print(
        f"[{phase_name.upper()} TEACHER] "
        f"best {ckpt_info['tracked_metric_name']}={ckpt_info['best_metric']:+.3f} "
        f"at update {ckpt_info['best_update']}"
    )

    metrics_path = os.path.join(out_dir, f"train_metrics_{phase_name}_teacher.npz")
    np.savez(metrics_path, **metrics_teacher)
    print(f"[{phase_name.upper()} TEACHER] Saved metrics to {metrics_path}")

    try:
        del ppo_teacher, metrics_teacher
    except Exception:
        pass

    # =========================================================
    # STUDENT (UKF) via distillation
    # =========================================================
    student_out = None

    if DISTILL:
        cfg_student = config_for_train(
            attacker_mode=attacker_mode,
            train_role=train_role,
        )
        cfg_student["use_ukf"] = True
        cfg_student["seed"] = cfg_teacher["seed"] + 1

        if extra_train_cfg is not None:
            cfg_student.update(extra_train_cfg)

        build_dyn(cfg_student)

        if cfg_student["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"[{phase_name.upper()} STUDENT] device='cuda' but CUDA not available."
            )

        print(f"[{phase_name.upper()} STUDENT] Using device: {cfg_student['device']}")

        student_out = os.path.join(out_dir, f"{phase_name}_ukf_student.pt")

        # NOTE:
        # distill_from_teacher() is currently defender-centric internally.
        # This unified wrapper preserves your current behavior, but if you want
        # truly symmetric attacker distillation, distill_from_teacher() itself
        # should be generalized later.
        student, metrics_student = distill_from_teacher(
            cfg_student,
            teacher_ckpt,
            out_path=student_out,
        )

        print(f"[{phase_name.upper()} STUDENT] Distilled UKF student saved to {student_out}")

        distill_metrics_path = os.path.join(
            out_dir, f"distill_metrics_{phase_name}_student.npz"
        )
        np.savez(distill_metrics_path, **metrics_student)
        print(f"[{phase_name.upper()} STUDENT] Saved distillation metrics to {distill_metrics_path}")

        try:
            del student, metrics_student
        except Exception:
            pass

    meta = {
        "train_role": train_role,
        "teacher_ckpt": teacher_ckpt,
        "teacher_best_metric_name": ckpt_info["tracked_metric_name"],
        "teacher_best_metric": ckpt_info["best_metric"],
        "teacher_best_update": ckpt_info["best_update"],
        "teacher_best_ckpt_path": ckpt_info["best_ckpt_path"],
        "teacher_last_ckpt_path": ckpt_info["last_ckpt_path"],
        "student_ckpt": student_out,
        "distill_enabled": DISTILL,
    }

    return teacher_ckpt, student_out, meta




# =============================================================
# End phase cleanup
# =============================================================

def end_phase_cleanup(
    tag: str = "",
    *,
    clear_cuda: bool = True,
    clear_ipc: bool = True,
    clear_mps: bool = True,
    clear_matplotlib: bool = True,
    sleep_s: float = 0.0,
):
    """
    Best-effort memory cleanup after a training phase.

    - CPU RAM: delete references + gc.collect()
    - GPU VRAM (CUDA): empty_cache(), ipc_collect()
    - MPS (Apple): empty_cache()
    - Matplotlib: close figures so they don't accumulate
    """
    print(f"\n[cleanup] {tag} ...")

    # ---- close any lingering matplotlib figures (common silent RAM leak) ----
    if clear_matplotlib:
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass

    # ---- Python heap cleanup ----
    gc.collect()

    # ---- PyTorch device-specific cleanup ----
    if clear_cuda and torch.cuda.is_available():
        # Clears cached blocks held by the CUDA allocator (does not free tensors you still reference).
        torch.cuda.empty_cache()

        if clear_ipc:
            # Helps in some multi-process / DataLoader / vector-env setups.
            torch.cuda.ipc_collect()

        # Optional: if you're debugging fragmentation, you can print stats:
        # print(torch.cuda.memory_summary())

    if clear_mps and hasattr(torch, "mps") and torch.mps.is_available():
        # Apple Silicon
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    if sleep_s > 0:
        time.sleep(sleep_s)

    print(f"[cleanup] {tag} done.")


def rollout_metrics(states: np.ndarray, center: np.ndarray, R: float):
    """
    states: [T+1, 12] for D=3 => [p1(3), v1(3), p2(3), v2(3)]
    center: (D,)
    R: arena radius (m)
    """
    D = center.shape[0]
    p1 = states[:, 0:D];      v1 = states[:, D:2*D]
    p2 = states[:, 2*D:3*D];  v2 = states[:, 3*D:4*D]

    d1 = np.sum((p1 - center)**2, axis=1) / (R*R)
    d2 = np.sum((p2 - center)**2, axis=1) / (R*R)
    rel2 = np.sum((p2 - p1)**2, axis=1) / (R*R)
    d2_delta = np.diff(d2, prepend=d2[:1])

    return {"d1_norm": d1, "d2_norm": d2, "rel2_norm": rel2, "d2_delta": d2_delta}

