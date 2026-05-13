from __future__ import annotations

import math

import numpy as np


def clamp_intercept_mix(value: float) -> float:
    return float(min(1.0, max(0.0, float(value))))


def annealed_prior_blend(
    progress: float,
    *,
    start_blend: float,
    end_blend: float = 0.0,
    anneal_fraction: float = 1.0,
) -> float:
    """
    Linearly anneal a blend from start_blend to end_blend over the requested
    fraction of training progress, then hold the final value.
    """
    start = clamp_intercept_mix(start_blend)
    end = clamp_intercept_mix(end_blend)
    progress_clamped = clamp_intercept_mix(progress)
    frac = float(anneal_fraction)
    if frac <= 0.0:
        return end
    alpha = min(progress_clamped / frac, 1.0)
    return float((1.0 - alpha) * start + alpha * end)


def rollout_coasting_state_np(Ad: np.ndarray, x: np.ndarray, steps: float) -> np.ndarray:
    """
    Roll a linear discrete state forward under zero control for a possibly
    fractional number of steps using linear interpolation between adjacent
    integer rollouts.
    """
    steps_f = max(0.0, float(steps))
    lo = int(math.floor(steps_f))
    hi = int(math.ceil(steps_f))

    x_lo = np.asarray(x, dtype=np.float32).reshape(-1)
    Ad_np = np.asarray(Ad, dtype=np.float32)
    for _ in range(lo):
        x_lo = Ad_np @ x_lo

    if hi == lo:
        return x_lo.astype(np.float32)

    x_hi = Ad_np @ x_lo
    alpha = steps_f - float(lo)
    return ((1.0 - alpha) * x_lo + alpha * x_hi).astype(np.float32)


def blended_intercept_target_np(
    Ad: np.ndarray,
    p_attacker_centered: np.ndarray,
    v_attacker: np.ndarray,
    *,
    lookahead_steps: float,
    mix: float,
) -> np.ndarray:
    """
    Blend between the attacker's current centered position and a coasting
    prediction of its future centered position.
    """
    p_now = np.asarray(p_attacker_centered, dtype=np.float32).reshape(-1)
    v_now = np.asarray(v_attacker, dtype=np.float32).reshape(-1)
    mix_clamped = clamp_intercept_mix(mix)
    if mix_clamped <= 0.0:
        return p_now

    x_pred = rollout_coasting_state_np(
        Ad,
        np.concatenate([p_now, v_now], axis=0),
        lookahead_steps,
    )
    D = p_now.shape[0]
    p_pred = x_pred[:D]
    if mix_clamped >= 1.0:
        return p_pred
    return ((1.0 - mix_clamped) * p_now + mix_clamped * p_pred).astype(np.float32)


def intercept_direction_prior_np(
    Ad: np.ndarray,
    p_defender_centered: np.ndarray,
    p_attacker_centered: np.ndarray,
    v_attacker: np.ndarray,
    *,
    lookahead_steps: float,
    mix: float,
    gain: float = 1.0,
) -> np.ndarray:
    """
    Direct defender heuristic: point toward a blended intercept target with a
    fixed raw-action gain.
    """
    p_def = np.asarray(p_defender_centered, dtype=np.float32).reshape(-1)
    target = blended_intercept_target_np(
        Ad,
        p_attacker_centered,
        v_attacker,
        lookahead_steps=lookahead_steps,
        mix=mix,
    )
    delta = target - p_def
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-6:
        return np.zeros_like(delta, dtype=np.float32)
    return (float(gain) * delta / norm).astype(np.float32)
