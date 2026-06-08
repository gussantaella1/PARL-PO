"""
tensorboard_plotters.py

TensorBoard scalar readers and plotting helpers for comparing training curves.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def find_event_dirs(root):
    """
    Return directories that contain at least one events.out.tfevents* file.
    Searches recursively.
    """
    # TensorBoard usually nests event files under run/seed folders, so the recursive
    # search lets one experiment directory become the only required input.
    event_files = glob.glob(os.path.join(root, "**", "events.out.tfevents.*"), recursive=True)
    dirs = sorted(set(os.path.dirname(p) for p in event_files))
    return dirs

def list_scalar_tags(event_dir):
    """List scalar tags present in a TensorBoard event directory."""
    ea = event_accumulator.EventAccumulator(event_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    return ea.Tags().get("scalars", [])

def read_scalar(event_dir, tag):
    """Read one scalar time series from TensorBoard event files."""
    ea = event_accumulator.EventAccumulator(event_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    ev = ea.Scalars(tag)
    if len(ev) == 0:
        return None, None
    steps = np.array([e.step for e in ev], dtype=np.int64)
    vals  = np.array([e.value for e in ev], dtype=np.float64)

    # Ensure strictly increasing in case of weird logging duplicates
    order = np.argsort(steps)
    steps, vals = steps[order], vals[order]
    uniq_steps, uniq_idx = np.unique(steps, return_index=True)
    return uniq_steps, vals[uniq_idx]

def ema(y, alpha=0.9):
    """Compute an exponential moving average for a scalar sequence."""
    y = np.asarray(y, dtype=np.float64)
    if len(y) == 0:
        return y
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha*out[i-1] + (1 - alpha)*y[i]
    return out

def composite_curve(run_dirs, tag, grid_size=300, smooth_alpha=None):
    """
    Build a composite curve across runs:
      - Reads (step, value) per run
      - Builds a common x-grid (steps) over the overlap region
      - Interpolates each run onto that grid
      - Returns grid, mean, std, n_runs_used
    """
    series = []
    for d in run_dirs:
        s, v = read_scalar(d, tag)
        if s is None:
            continue
        if smooth_alpha is not None:
            # Smooth before interpolation so each run keeps its own time history.
            v = ema(v, smooth_alpha)
        series.append((s, v))

    if len(series) == 0:
        raise ValueError(f"No runs contained scalar tag '{tag}'")

    # Use overlap region so we aren't averaging where only 1 run has data.
    min_last = min(s[-1] for s, _ in series)
    max_first = max(s[0] for s, _ in series)

    if min_last <= max_first:
        # fallback: use union region (less “fair”), but at least you get a plot
        global_min = min(s[0] for s, _ in series)
        global_max = max(s[-1] for s, _ in series)
        x_grid = np.linspace(global_min, global_max, grid_size).astype(np.float64)
        use_overlap = False
    else:
        x_grid = np.linspace(max_first, min_last, grid_size).astype(np.float64)
        use_overlap = True

    Y = []
    used = 0
    for s, v in series:
        # Only interpolate within the run’s data range
        left, right = float(s[0]), float(s[-1])
        mask = (x_grid >= left) & (x_grid <= right)
        if mask.sum() < max(10, grid_size // 10):
            continue
        y_interp = np.full_like(x_grid, np.nan, dtype=np.float64)
        y_interp[mask] = np.interp(x_grid[mask], s.astype(np.float64), v)
        Y.append(y_interp)
        used += 1

    # Shape is (n_runs, grid_size); NaNs mark regions where a run did not contribute.
    Y = np.stack(Y, axis=0)
    mean = np.nanmean(Y, axis=0)
    std  = np.nanstd(Y, axis=0)

    # If we used union region, some parts may have fewer runs contributing.
    # Compute counts per x for optional CI.
    counts = np.sum(~np.isnan(Y), axis=0)

    return x_grid, mean, std, counts, used, use_overlap

def plot_composite(
    exp_logdir,
    tag,
    outpath="composite.png",
    ci95=False,
    smooth_alpha=0.9,
    grid_size=300,
    title=None
):
    """Plot composite using the current metrics or rollout data."""
    run_dirs = find_event_dirs(exp_logdir)
    if len(run_dirs) == 0:
        raise FileNotFoundError(f"No TensorBoard event files found under: {exp_logdir}")

    x, mean, std, counts, used, use_overlap = composite_curve(
        run_dirs, tag, grid_size=grid_size, smooth_alpha=smooth_alpha
    )

    plt.figure()
    plt.plot(x, mean, label=f"mean (n={used})")

    if ci95:
        # 95% CI is computed per x-position because union-mode grids can have
        # different numbers of runs contributing at different steps.
        n_eff = np.maximum(counts, 1)
        ci = 1.96 * (std / np.sqrt(n_eff))
        plt.fill_between(x, mean - ci, mean + ci, alpha=0.2, label="95% CI")
    else:
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, label="±1 std")

    plt.xlabel("step")
    plt.ylabel(tag)
    if title is None:
        title = f"Composite: {os.path.basename(os.path.normpath(exp_logdir))}"
        title += " (overlap)" if use_overlap else " (union)"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    print(f"Saved: {outpath}")

if __name__ == "__main__":
    # Example usage:
    #   exp_logdir = "runs/exp_name"  # contains seed subfolders, each with event files
    exp_logdir = "runs/exp_name"
    tag = "returns/def_mean"
    plot_composite(exp_logdir, tag, outpath="returns_def_mean.png", ci95=True, smooth_alpha=0.9)
