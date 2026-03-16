
import importlib
import os
from typing import Dict, Any

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# from __future__ import annotations


from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import json

from pathlib import Path




# --- Single source of truth for config & dynamics ---
import config_rl
importlib.reload(config_rl)

from core.env import sample_ic_support

# =============================================================
# Plotting scripts
# =============================================================

def load_npz_metrics(path: str) -> dict:
    """
    Load a .npz metrics file saved via: np.savez(metrics_path, **metrics)

    Returns
    -------
    metrics : dict[str, np.ndarray]
        Keys map to 1D numpy arrays (or arrays as saved).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Metrics file not found: {path}")

    data = np.load(path, allow_pickle=True)
    metrics = {}
    for k in data.files:
        metrics[k] = data[k]
    return metrics

def plot_training_metrics(
    metrics: dict,
    title: str = "",
    smooth: str = "none",         # "none" | "ema" | "ma"
    smooth_param: float = 0.2,    # ema alpha OR ma window
    show: bool = False,

    # NEW saving knobs
    out_dir: str | None = None,
    save_prefix: str | None = None,
    dpi: int = 200,
    close: bool = True,
):
    """
    Plot the metrics saved by your train() loop.

    If out_dir is provided, saves figures there (one PNG per figure) and
    does not require showing.

    Returns
    -------
    figs : list[matplotlib.figure.Figure]
    saved_paths : list[str]  (empty if out_dir is None)
    """
    def _as_1d(x):
        x = np.asarray(x)
        if x.ndim == 0:
            return x.reshape(1).astype(float)
        if x.ndim > 1:
            x = x.reshape(-1)
        return x.astype(float)

    def _smooth_series(y, method="none", param=0.2):
        y = _as_1d(y)
        if method is None or method == "none":
            return y
        if method == "ema":
            alpha = float(param)
            if not (0.0 < alpha <= 1.0):
                raise ValueError("EMA alpha must be in (0, 1].")
            out = np.empty_like(y)
            out[0] = y[0]
            for i in range(1, len(y)):
                out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]
            return out
        if method == "ma":
            win = int(param)
            if win <= 1:
                return y
            pad = win - 1
            ypad = np.pad(y, (pad, 0), mode="edge")
            kernel = np.ones(win, dtype=float) / win
            return np.convolve(ypad, kernel, mode="valid")
        raise ValueError(f"Unknown smoothing method: {method!r}")

    # x-axis
    if "update" in metrics:
        x = _as_1d(metrics["update"])
    else:
        for k in ["R_def_mean", "R_att_mean", "d1_mean", "d2_mean", "lr_pi"]:
            if k in metrics:
                x = np.arange(len(metrics[k]), dtype=float)
                break
        else:
            raise ValueError("No recognizable metrics keys found to infer x-axis.")

    def _plot_if(ax, key, label=None):
        if key not in metrics:
            return False
        y = _smooth_series(metrics[key], method=smooth, param=smooth_param)
        ax.plot(x, y, label=(label or key))
        return True

    figs = []
    saved_paths = []

    # pick a reasonable prefix for filenames
    if save_prefix is None:
        save_prefix = title.strip() if title.strip() else "run"

    # 1) Returns
    if ("R_def_mean" in metrics) or ("R_att_mean" in metrics):
        fig = plt.figure()
        ax = plt.gca()
        any_plotted = False
        any_plotted |= _plot_if(ax, "R_def_mean", "R_def_mean (per update)")
        any_plotted |= _plot_if(ax, "R_att_mean", "R_att_mean (per update)")
        if any_plotted:
            ax.set_xlabel("Update")
            ax.set_ylabel("Mean return (sum over rollout)")
            ax.set_title(f"{title} — Returns" if title else "Returns")
            ax.grid(True)
            ax.legend()
            figs.append(("returns", fig))

    # 2) Distances
    dist_keys = [k for k in ["d1_mean", "d2_mean", "d2_true_mean"] if k in metrics]
    if dist_keys:
        fig = plt.figure()
        ax = plt.gca()
        _plot_if(ax, "d1_mean", "Defender true ⟨||p1-center||⟩ (m)")
        _plot_if(ax, "d2_true_mean", "Attacker true ⟨||p2-center||⟩ (m)")
        _plot_if(ax, "d2_mean", "Attacker belief ⟨||p2-center||⟩ (m)")
        ax.set_xlabel("Update")
        ax.set_ylabel("Distance to center (m)")
        ax.set_title(f"{title} — Distances" if title else "Distances")
        ax.grid(True)
        ax.legend()
        figs.append(("distances", fig))

    # 3) Learning rates
    if ("lr_pi" in metrics) or ("lr_vf" in metrics):
        fig = plt.figure()
        ax = plt.gca()
        _plot_if(ax, "lr_pi", "lr_pi")
        _plot_if(ax, "lr_vf", "lr_vf")
        ax.set_xlabel("Update")
        ax.set_ylabel("Learning rate")
        ax.set_title(f"{title} — Learning rates" if title else "Learning rates")
        ax.grid(True)
        ax.legend()
        figs.append(("lrs", fig))

    # 4) Policy stats
    if ("muD_abs_mean" in metrics) or ("stdD_mean" in metrics):
        fig = plt.figure()
        ax = plt.gca()
        _plot_if(ax, "muD_abs_mean", "|mu| mean (def)")
        _plot_if(ax, "stdD_mean", "std mean (def)")
        ax.set_xlabel("Update")
        ax.set_ylabel("Value")
        ax.set_title(f"{title} — Policy stats" if title else "Policy stats")
        ax.grid(True)
        ax.legend()
        figs.append(("policy_stats", fig))

    # 5) UKF stats
    if ("meas_innov_mean" in metrics) or ("ukf_trPpos_mean" in metrics):
        fig = plt.figure()
        ax = plt.gca()
        _plot_if(ax, "meas_innov_mean", "meas_innov_mean (E[||innov||^2])")
        _plot_if(ax, "ukf_trPpos_mean", "ukf_trPpos_mean (trace(P_pos))")
        ax.set_xlabel("Update")
        ax.set_ylabel("Value")
        ax.set_title(f"{title} — UKF stats" if title else "UKF stats")
        ax.grid(True)
        ax.legend()
        figs.append(("ukf_stats", fig))

    # 6) Fuel utilization:

    fuel_keys = [k for k in [
        "fuel_used_def_mean",
        "fuel_used_att_mean",
        "fuel_frac_def_mean",
        "fuel_frac_att_mean",
    ] if k in metrics and len(np.asarray(metrics[k]).reshape(-1)) == len(x)]

    if fuel_keys:
        fig = plt.figure()
        ax = plt.gca()
        _plot_if(ax, "fuel_used_def_mean", "Defender fuel used frac")
        _plot_if(ax, "fuel_used_att_mean", "Attacker fuel used frac")
        _plot_if(ax, "fuel_frac_def_mean", "Defender fuel remaining frac")
        _plot_if(ax, "fuel_frac_att_mean", "Attacker fuel remaining frac")
        ax.set_xlabel("Update")
        ax.set_ylabel("Fraction")
        ax.set_title(f"{title} — Fuel" if title else "Fuel")
        ax.grid(True)
        ax.legend()
        figs.append(("fuel", fig))

    if not figs:
        raise ValueError("None of the expected keys were present; nothing to plot.")

    # ---- save if requested ----
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        for tag, fig in figs:
            fname = f"{save_prefix}__{tag}.png"
            fpath = os.path.join(out_dir, fname)
            fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
            saved_paths.append(fpath)

    # ---- show / close behavior ----
    if show:
        plt.show()

    if close:
        for _, fig in figs:
            plt.close(fig)

    # return plain fig list for compatibility + saved paths
    return [fig for _, fig in figs], saved_paths



def plot_compare_phases(
    labeled_paths,
    metric: str,
    ylabel: str = None,
    title: str = None,
    smooth: str = "none",
    smooth_param: float = 0.2,
    show: bool = False,

    # NEW saving knobs
    out_dir: str | None = None,
    filename: str | None = None,
    dpi: int = 200,
    close: bool = True,
):
    """
    Compare a single metric across multiple runs and optionally save it.
    """
    fig = plt.figure()
    ax = plt.gca()

    def _as_1d(x):
        x = np.asarray(x)
        if x.ndim == 0:
            return x.reshape(1).astype(float)
        if x.ndim > 1:
            x = x.reshape(-1)
        return x.astype(float)

    def _smooth_series(y, method="none", param=0.2):
        y = _as_1d(y)
        if method is None or method == "none":
            return y
        if method == "ema":
            alpha = float(param)
            out = np.empty_like(y)
            out[0] = y[0]
            for i in range(1, len(y)):
                out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]
            return out
        if method == "ma":
            win = int(param)
            if win <= 1:
                return y
            pad = win - 1
            ypad = np.pad(y, (pad, 0), mode="edge")
            kernel = np.ones(win, dtype=float) / win
            return np.convolve(ypad, kernel, mode="valid")
        raise ValueError(f"Unknown smoothing method: {method!r}")

    for item in labeled_paths:
        if len(item) == 2:
            label, path = item
            metric_key = metric
        else:
            label, path, metric_key = item

        m = load_npz_metrics(path)
        if metric_key not in m:
            print(f"[plot_compare_phases] skipping {label}: missing key {metric_key!r}")
            continue

        x = _as_1d(m["update"]) if "update" in m else np.arange(len(m[metric_key]))
        y = _smooth_series(m[metric_key], method=smooth, param=smooth_param)
        ax.plot(x, y, label=label)

    ax.set_xlabel("Update")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or f"Compare: {metric}")
    ax.grid(True)
    ax.legend()

    saved_path = None
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        if filename is None:
            safe_metric = metric.replace("/", "_")
            filename = f"compare__{safe_metric}.png"
        saved_path = os.path.join(out_dir, filename)
        fig.savefig(saved_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    if close:
        plt.close(fig)

    return fig, saved_path

def _draw_circle(ax, radius, center_xy=(0.0, 0.0), **kwargs):
    th = np.linspace(0.0, 2.0 * np.pi, 400)
    x = center_xy[0] + radius * np.cos(th)
    y = center_xy[1] + radius * np.sin(th)
    ax.plot(x, y, **kwargs)


def _plot_ic_projection(
    ax,
    def_pos: np.ndarray,
    att_pos: np.ndarray,
    dims=(0, 1),
    title="",
    arena_r=None,
    oi_r=None,
    center=None,
    alpha_def=0.15,
    alpha_att=0.15,
    s=4,
):
    i, j = dims
    center = np.zeros(3, dtype=float) if center is None else np.asarray(center, dtype=float)

    if def_pos.shape[0] > 0:
        ax.scatter(
            def_pos[:, i], def_pos[:, j],
            s=s, alpha=alpha_def, label="Defender starts"
        )

    if att_pos.shape[0] > 0:
        ax.scatter(
            att_pos[:, i], att_pos[:, j],
            s=s, alpha=alpha_att, label="Attacker starts"
        )

    # arena / OI circles only make geometric sense in centered projections
    if arena_r is not None:
        _draw_circle(ax, arena_r, center_xy=(center[i], center[j]), linestyle="--", linewidth=1.0)
    if oi_r is not None and oi_r > 0.0:
        _draw_circle(ax, oi_r, center_xy=(center[i], center[j]), linestyle="-", linewidth=1.0)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_title(title)
    ax.legend()


def plot_ic_samples(
    def_pos: np.ndarray,
    att_pos: np.ndarray,
    cfg: Dict[str, Any],
    title: str = "",
    out_path: str | None = None,
    show: bool = False,
    close: bool = True,
):
    """
    Plot IC samples in XY and, if D=3, also XZ projection.
    """
    D = int(cfg["D"])
    ar = cfg["arena"]
    center = np.array(
        [ar["cx"], ar["cy"], (ar.get("cz", 0.0) if D == 3 else 0.0)],
        dtype=float,
    )[:D]
    arena_r = float(ar["r"])
    oi_r = float(cfg.get("oi", {}).get("r", 0.0))

    if D == 2:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        _plot_ic_projection(
            ax,
            def_pos,
            att_pos,
            dims=(0, 1),
            title=title or "IC samples (XY)",
            arena_r=arena_r,
            oi_r=oi_r,
            center=np.pad(center, (0, 1)),
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        _plot_ic_projection(
            axes[0],
            def_pos,
            att_pos,
            dims=(0, 1),
            title=(title + " — XY") if title else "IC samples — XY",
            arena_r=arena_r,
            oi_r=oi_r,
            center=center,
        )
        _plot_ic_projection(
            axes[1],
            def_pos,
            att_pos,
            dims=(0, 2),
            title=(title + " — XZ") if title else "IC samples — XZ",
            arena_r=arena_r,
            oi_r=oi_r,
            center=center,
        )

    fig.tight_layout()

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    if close:
        plt.close(fig)

    return fig

def make_tb_writer(cfg: dict, run_name: str | None = None):
    """
    Creates a TensorBoard SummaryWriter under:
      runs/<run_name or timestamped name>/

    Also writes config.json for reproducibility.
    """
    root = cfg.get("tb_logdir", "runs")
    if run_name is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = cfg.get("tb_run_name", f"diffgame_{stamp}")

    logdir = os.path.join(root, run_name)
    os.makedirs(logdir, exist_ok=True)

    # Save config snapshot next to TB logs (nice for later)
    try:
        with open(os.path.join(logdir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2, default=str)
    except Exception as e:
        print("[tb] could not write config.json:", e)

    writer = SummaryWriter(log_dir=logdir)
    print(f"[tb] logging to: {logdir}")
    return writer, logdir



def plot_ic_support_from_cfg(
    cfg: Dict[str, Any],
    n_scenes: int = 20000,
    seed: int = 123,
    title: str = "Feasible initial-condition support",
    out_path: str | None = None,
    show: bool = False,
):
    def_pos, att_pos = sample_ic_support(cfg, n_scenes=n_scenes, seed=seed)
    return plot_ic_samples(
        def_pos,
        att_pos,
        cfg,
        title=title,
        out_path=out_path,
        show=show,
    )


def plot_ic_used_from_npz(
    npz_path: str,
    cfg: Dict[str, Any],
    title: str = "Initial conditions actually used during training",
    out_path: str | None = None,
    show: bool = False,
):
    data = np.load(npz_path)
    def_pos = data["def_pos"]
    att_pos = data["att_pos"]
    return plot_ic_samples(
        def_pos,
        att_pos,
        cfg,
        title=title,
        out_path=out_path,
        show=show,
    )

