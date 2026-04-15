from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from rl_infer import RLPolicyDiff


def _load_json_dict(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise RuntimeError(f"Could not find JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse JSON file: {path}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected top-level JSON object in {path}.")
    return payload


def _extract_manifest_training_cfg(manifest: Dict[str, Any], manifest_path: Path) -> Dict[str, Any]:
    configs = manifest.get("configs") or {}
    if not isinstance(configs, dict):
        raise RuntimeError(f"Manifest {manifest_path} is missing a dict-valued 'configs' section.")

    cfg = configs.get("training config used")
    if isinstance(cfg, dict):
        return copy.deepcopy(cfg)

    dict_cfgs = [(k, v) for k, v in configs.items() if isinstance(v, dict)]
    if len(dict_cfgs) == 1:
        return copy.deepcopy(dict_cfgs[0][1])

    keys = ", ".join(sorted(str(k) for k in configs.keys()))
    raise RuntimeError(
        f"Could not find 'configs[\"training config used\"]' in {manifest_path}. "
        f"Available config keys: {keys}"
    )


def _resolve_run_manifest_path(
    run_manifest: Optional[str],
    run_dir: Optional[str],
    ckpt_path: str,
) -> Path:
    if run_manifest:
        return Path(run_manifest).expanduser().resolve()
    if run_dir:
        return (Path(run_dir).expanduser().resolve() / "run_manifest.json").resolve()

    ckpt = Path(ckpt_path).expanduser().resolve()
    candidate = ckpt.parent / "run_manifest.json"
    if candidate.exists():
        return candidate.resolve()

    raise RuntimeError(
        "Could not find run_manifest.json. Pass --run_manifest or --run_dir, "
        "or keep the checkpoint next to its run manifest."
    )


def _build_dyn_for_cfg(cfg: Dict[str, Any]) -> None:
    if int(cfg.get("num_attackers", 1)) > 1:
        from config_rl_1v2 import build_dyn
    else:
        from config_rl import build_dyn
    build_dyn(cfg)


def _env_cls_for_cfg(cfg: Dict[str, Any]):
    if int(cfg.get("num_attackers", 1)) > 1:
        from core_1v2.env import Env
    else:
        from core.env import Env
    return Env


def _set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _reset_recurrent_state(wrapper: RLPolicyDiff) -> None:
    if getattr(wrapper, "def_is_student", False):
        wrapper.def_hidden = wrapper.def_net.init_hidden(batch_size=1, device=wrapper.device)
        wrapper.def_u_prev = np.zeros((wrapper.act_dim,), dtype=np.float32)
    if getattr(wrapper, "att_is_student", False):
        wrapper.att_hidden = wrapper.att_net.init_hidden(batch_size=1, device=wrapper.device)
        wrapper.att_u_prev = np.zeros((wrapper.act_dim,), dtype=np.float32)


def _observation_groups(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if int(cfg.get("num_attackers", 1)) != 1:
        raise NotImplementedError("This analysis script currently supports only num_attackers=1.")

    D = int(cfg.get("D", 3))
    use_fuel = bool(cfg.get("fuel", {}).get("enable", False))
    axis_names = ["x", "y", "z"][:D]

    groups: List[Tuple[str, str, slice, Sequence[str]]] = [
        ("def_pos_centered", "Defender centered position", slice(0, D), axis_names),
        ("opp_pos_centered", "Opponent centered position", slice(D, 2 * D), axis_names),
        ("relative_pos", "Relative position", slice(2 * D, 3 * D), axis_names),
        ("def_vel", "Defender velocity", slice(3 * D, 4 * D), axis_names),
        ("opp_vel", "Opponent velocity", slice(4 * D, 5 * D), axis_names),
    ]

    out: List[Dict[str, Any]] = []
    for name, label, sl, dims in groups:
        out.append(
            {
                "name": name,
                "label": label,
                "indices": list(range(sl.start, sl.stop)),
                "dim_labels": list(dims),
            }
        )

    if use_fuel:
        out.append(
            {
                "name": "fuel_def",
                "label": "Defender fuel fraction",
                "indices": [5 * D],
                "dim_labels": ["fuel_def"],
            }
        )
        out.append(
            {
                "name": "fuel_att",
                "label": "Attacker fuel fraction",
                "indices": [5 * D + 1],
                "dim_labels": ["fuel_att"],
            }
        )
    return out


def _target_outputs(
    wrapper: RLPolicyDiff,
    policy_role: str,
    obs_batch: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if policy_role == "def":
        model = wrapper.def_net
        is_student = bool(wrapper.def_is_student)
        who = "def"
    else:
        model = wrapper.att_net
        is_student = bool(wrapper.att_is_student)
        who = "att"

    if model is None:
        raise RuntimeError(f"No loaded policy found for role '{policy_role}'.")

    if not is_student:
        dist = model.dist(obs_batch, who=who)
        action = torch.tanh(dist.mean) * wrapper.umax
        value = model.value(obs_batch)
        return action, value

    D = wrapper.D
    rel = obs_batch[:, 2 * D : 3 * D]
    v1 = obs_batch[:, 3 * D : 4 * D]
    v2 = obs_batch[:, 4 * D : 5 * D]
    xhat_rel = torch.cat([rel, v2 - v1], dim=-1)
    sigma = torch.zeros(
        (obs_batch.shape[0], wrapper.student_sigma_dim),
        dtype=obs_batch.dtype,
        device=obs_batch.device,
    )
    u_prev = torch.zeros(
        (obs_batch.shape[0], wrapper.act_dim),
        dtype=obs_batch.dtype,
        device=obs_batch.device,
    )
    hidden = model.init_hidden(batch_size=obs_batch.shape[0], device=obs_batch.device)
    action, _, _ = model.step(xhat_rel, sigma, u_prev, hidden)
    return action, None


def _collect_observations(
    cfg: Dict[str, Any],
    wrapper: RLPolicyDiff,
    *,
    policy_role: str,
    opponent_mode: str,
    episodes: int,
    max_steps: int,
    deterministic: bool,
    seed: int,
) -> np.ndarray:
    Env = _env_cls_for_cfg(cfg)
    env = Env(cfg)
    rng = np.random.default_rng(seed + 17)
    observations: List[np.ndarray] = []

    def _random_action() -> np.ndarray:
        umax = float(cfg.get("umax", 1.0))
        D = int(cfg.get("D", 3))
        return rng.uniform(-umax, umax, size=(D,)).astype(np.float32)

    def _zero_action() -> np.ndarray:
        D = int(cfg.get("D", 3))
        return np.zeros((D,), dtype=np.float32)

    for _ in range(episodes):
        env.reset()
        _reset_recurrent_state(wrapper)

        done = False
        steps = 0
        while (not done) and (steps < max_steps):
            obs_def, obs_att = env.get_obs_pair()
            target_obs = obs_def if policy_role == "def" else obs_att
            observations.append(np.asarray(target_obs, dtype=np.float32).copy())

            if policy_role == "def":
                a_def = wrapper.act_def_obs(obs_def, deterministic=deterministic)
                if opponent_mode in {"loaded", "rule"}:
                    a_att = wrapper.act_att_obs(obs_att, deterministic=deterministic)
                elif opponent_mode == "random":
                    a_att = _random_action()
                elif opponent_mode == "zero":
                    a_att = _zero_action()
                else:
                    raise ValueError(f"Unsupported opponent_mode='{opponent_mode}'.")
            else:
                if opponent_mode == "loaded":
                    a_def = wrapper.act_def_obs(obs_def, deterministic=deterministic)
                elif opponent_mode == "random":
                    a_def = _random_action()
                elif opponent_mode == "zero":
                    a_def = _zero_action()
                else:
                    raise ValueError(
                        f"Unsupported opponent_mode='{opponent_mode}' for attacker analysis."
                    )
                a_att = wrapper.act_att_obs(obs_att, deterministic=deterministic)

            _, _, _, done, _ = env.step(a_def, a_att)
            steps += 1

    if not observations:
        raise RuntimeError("No observations were collected. Try increasing --episodes or --max_steps.")
    return np.stack(observations, axis=0)


def _compute_gradient_scores(
    wrapper: RLPolicyDiff,
    policy_role: str,
    obs_np: np.ndarray,
    *,
    batch_size: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    device = wrapper.device
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    n = obs_t.shape[0]
    obs_dim = obs_t.shape[1]

    action_grad_sum = torch.zeros((obs_dim,), dtype=torch.float64, device=device)
    value_grad_sum: Optional[torch.Tensor] = None
    have_value: Optional[bool] = None

    for start in range(0, n, batch_size):
        batch = obs_t[start : start + batch_size].clone().detach().requires_grad_(True)
        action, value = _target_outputs(wrapper, policy_role, batch)
        if have_value is None:
            have_value = value is not None
            if have_value:
                value_grad_sum = torch.zeros((obs_dim,), dtype=torch.float64, device=device)

        grad_sq = torch.zeros_like(batch)
        for j in range(action.shape[1]):
            retain = (j + 1) < action.shape[1]
            grad_j = torch.autograd.grad(
                action[:, j].sum(),
                batch,
                retain_graph=retain,
                create_graph=False,
            )[0]
            grad_sq = grad_sq + grad_j.pow(2)
        action_grad_sum += grad_sq.sqrt().sum(dim=0).to(torch.float64)

        if value is not None and value_grad_sum is not None:
            value_grad = torch.autograd.grad(
                value.sum(),
                batch,
                retain_graph=False,
                create_graph=False,
            )[0]
            value_grad_sum += value_grad.abs().sum(dim=0).to(torch.float64)

    action_grad = (action_grad_sum / float(n)).detach().cpu().numpy()
    if value_grad_sum is None:
        return action_grad, None
    value_grad = (value_grad_sum / float(n)).detach().cpu().numpy()
    return action_grad, value_grad


def _compute_ablation_scores(
    wrapper: RLPolicyDiff,
    policy_role: str,
    obs_np: np.ndarray,
    groups: Sequence[Dict[str, Any]],
    *,
    batch_size: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    device = wrapper.device
    baseline = obs_np.mean(axis=0, keepdims=True).astype(np.float32)
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    baseline_t = torch.as_tensor(baseline, dtype=torch.float32, device=device)

    action_deltas = {g["name"]: 0.0 for g in groups}
    value_deltas = {g["name"]: 0.0 for g in groups}
    n = obs_t.shape[0]
    have_value: Optional[bool] = None

    for start in range(0, n, batch_size):
        batch = obs_t[start : start + batch_size]
        batch_n = batch.shape[0]
        action_ref, value_ref = _target_outputs(wrapper, policy_role, batch)
        if have_value is None:
            have_value = value_ref is not None

        for group in groups:
            idx = group["indices"]
            batch_abl = batch.clone()
            batch_abl[:, idx] = baseline_t[:, idx]
            action_abl, value_abl = _target_outputs(wrapper, policy_role, batch_abl)

            action_change = torch.linalg.norm(action_abl - action_ref, dim=-1).sum().item()
            action_deltas[group["name"]] += action_change

            if value_ref is not None and value_abl is not None:
                value_change = torch.abs(value_abl - value_ref).sum().item()
                value_deltas[group["name"]] += value_change
            elif have_value:
                raise RuntimeError("Unexpected value head mismatch during ablation analysis.")

    for name in action_deltas:
        action_deltas[name] /= float(n)
    if not have_value:
        value_deltas = {}
    else:
        for name in value_deltas:
            value_deltas[name] /= float(n)

    return action_deltas, value_deltas


def _summarize_groups(
    groups: Sequence[Dict[str, Any]],
    action_grad: np.ndarray,
    value_grad: Optional[np.ndarray],
    action_ablate: Dict[str, float],
    value_ablate: Dict[str, float],
) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for group in groups:
        idx = group["indices"]
        action_grad_group = float(np.mean(action_grad[idx]))
        row: Dict[str, Any] = {
            "name": group["name"],
            "label": group["label"],
            "indices": list(idx),
            "dim_labels": list(group["dim_labels"]),
            "action_grad_mean": action_grad_group,
            "action_ablation_delta": float(action_ablate[group["name"]]),
            "per_dim_action_grad": {
                dim_label: float(action_grad[i]) for dim_label, i in zip(group["dim_labels"], idx)
            },
        }
        if value_grad is not None:
            row["value_grad_mean"] = float(np.mean(value_grad[idx]))
            row["per_dim_value_grad"] = {
                dim_label: float(value_grad[i]) for dim_label, i in zip(group["dim_labels"], idx)
            }
        if group["name"] in value_ablate:
            row["value_ablation_delta"] = float(value_ablate[group["name"]])
        summary.append(row)

    summary.sort(key=lambda item: item["action_ablation_delta"], reverse=True)
    return summary


def _print_summary(
    summary: Sequence[Dict[str, Any]],
    *,
    value_available: bool,
    sample_count: int,
) -> None:
    print(f"Collected {sample_count} observation samples.")
    print("")
    print("Ranked observation groups")
    if value_available:
        header = (
            f"{'group':<22} {'action_grad':>12} {'action_abl':>12} "
            f"{'value_grad':>12} {'value_abl':>12}"
        )
        print(header)
        print("-" * len(header))
        for row in summary:
            print(
                f"{row['name']:<22} "
                f"{row['action_grad_mean']:>12.4e} "
                f"{row['action_ablation_delta']:>12.4e} "
                f"{row.get('value_grad_mean', 0.0):>12.4e} "
                f"{row.get('value_ablation_delta', 0.0):>12.4e}"
            )
    else:
        header = f"{'group':<22} {'action_grad':>12} {'action_abl':>12}"
        print(header)
        print("-" * len(header))
        for row in summary:
            print(
                f"{row['name']:<22} "
                f"{row['action_grad_mean']:>12.4e} "
                f"{row['action_ablation_delta']:>12.4e}"
            )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Rank observation groups by how much they influence a policy's action and value. "
            "This script currently targets the single-attacker 1v1 path."
        )
    )
    ap.add_argument("--ckpt_path", required=True, help="Checkpoint for the policy being analyzed.")
    ap.add_argument("--policy_role", choices=["def", "att"], default="def")
    ap.add_argument("--run_manifest", default=None, help="Path to run_manifest.json.")
    ap.add_argument("--run_dir", default=None, help="Run directory containing run_manifest.json.")
    ap.add_argument(
        "--opponent_mode",
        choices=["loaded", "rule", "random", "zero"],
        default=None,
        help=(
            "How to drive the non-target agent during rollout collection. "
            "Default: defender analysis uses 'rule' when no opponent checkpoint is given, "
            "otherwise 'loaded'; attacker analysis uses 'loaded' when an opponent checkpoint is given, "
            "otherwise 'zero'."
        ),
    )
    ap.add_argument(
        "--opponent_ckpt_path",
        default=None,
        help="Optional checkpoint for the non-target agent when --opponent_mode=loaded.",
    )
    ap.add_argument("--episodes", type=int, default=8, help="Number of rollout episodes to sample.")
    ap.add_argument("--max_steps", type=int, default=150, help="Max steps per sampled rollout.")
    ap.add_argument("--batch_size", type=int, default=128, help="Batch size for attribution passes.")
    ap.add_argument("--device", default="cpu", help="Torch device for the policy, e.g. cpu or cuda.")
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="Use the deterministic policy mean during rollout collection.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_json", default=None, help="Optional path to save the ranked results as JSON.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    ckpt_path = str(Path(args.ckpt_path).expanduser().resolve())
    manifest_path = _resolve_run_manifest_path(args.run_manifest, args.run_dir, ckpt_path)
    manifest = _load_json_dict(manifest_path)
    cfg = _extract_manifest_training_cfg(manifest, manifest_path)

    if int(cfg.get("num_attackers", 1)) != 1:
        raise NotImplementedError("This analysis script currently supports only num_attackers=1.")

    cfg = copy.deepcopy(cfg)
    cfg["rl_eval_deterministic"] = bool(args.deterministic)
    cfg["device"] = args.device
    _build_dyn_for_cfg(cfg)
    _set_global_seed(int(args.seed))

    opponent_mode = args.opponent_mode
    if opponent_mode is None:
        if args.policy_role == "def":
            opponent_mode = "loaded" if args.opponent_ckpt_path else "rule"
        else:
            opponent_mode = "loaded" if args.opponent_ckpt_path else "zero"

    if args.policy_role == "att" and opponent_mode == "rule":
        raise ValueError("opponent_mode='rule' is only supported for defender analysis.")
    if opponent_mode == "loaded" and not args.opponent_ckpt_path and args.policy_role == "att":
        raise ValueError("Attacker analysis with opponent_mode='loaded' requires --opponent_ckpt_path.")

    wrapper_cfg = copy.deepcopy(cfg)
    target_ckpt = ckpt_path
    opponent_ckpt = (
        str(Path(args.opponent_ckpt_path).expanduser().resolve()) if args.opponent_ckpt_path else None
    )

    if args.policy_role == "def":
        if opponent_mode == "rule":
            wrapper_cfg["attacker_mode"] = "rule"
            wrapper = RLPolicyDiff(
                wrapper_cfg,
                device=args.device,
                def_ckpt=target_ckpt,
                att_ckpt=None,
                load_defender=True,
                load_attacker=True,
            )
        elif opponent_mode == "loaded":
            wrapper_cfg["attacker_mode"] = "rl"
            wrapper = RLPolicyDiff(
                wrapper_cfg,
                device=args.device,
                def_ckpt=target_ckpt,
                att_ckpt=opponent_ckpt,
                load_defender=True,
                load_attacker=True,
            )
        else:
            wrapper_cfg["attacker_mode"] = "rl"
            wrapper = RLPolicyDiff(
                wrapper_cfg,
                device=args.device,
                def_ckpt=target_ckpt,
                att_ckpt=None,
                load_defender=True,
                load_attacker=False,
            )
    else:
        wrapper_cfg["attacker_mode"] = "rl"
        wrapper = RLPolicyDiff(
            wrapper_cfg,
            device=args.device,
            def_ckpt=opponent_ckpt,
            att_ckpt=target_ckpt,
            load_defender=(opponent_mode == "loaded"),
            load_attacker=True,
        )

    obs_np = _collect_observations(
        cfg,
        wrapper,
        policy_role=args.policy_role,
        opponent_mode=opponent_mode,
        episodes=int(args.episodes),
        max_steps=int(args.max_steps),
        deterministic=bool(args.deterministic),
        seed=int(args.seed),
    )
    groups = _observation_groups(cfg)
    action_grad, value_grad = _compute_gradient_scores(
        wrapper,
        args.policy_role,
        obs_np,
        batch_size=int(args.batch_size),
    )
    action_ablate, value_ablate = _compute_ablation_scores(
        wrapper,
        args.policy_role,
        obs_np,
        groups,
        batch_size=int(args.batch_size),
    )
    summary = _summarize_groups(groups, action_grad, value_grad, action_ablate, value_ablate)
    _print_summary(summary, value_available=(value_grad is not None), sample_count=int(obs_np.shape[0]))

    if args.output_json:
        payload = {
            "ckpt_path": ckpt_path,
            "policy_role": args.policy_role,
            "run_manifest": str(manifest_path),
            "opponent_mode": opponent_mode,
            "opponent_ckpt_path": opponent_ckpt,
            "episodes": int(args.episodes),
            "max_steps": int(args.max_steps),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "sample_count": int(obs_np.shape[0]),
            "value_available": bool(value_grad is not None),
            "normalize_pos_obs": bool(cfg.get("normalize_pos_obs", False)),
            "summary": summary,
        }
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.write_text(json.dumps(payload, indent=2))
        print("")
        print(f"Saved JSON results to {out_path}")


if __name__ == "__main__":
    main()
