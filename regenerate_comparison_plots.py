#!/usr/bin/env python3
"""
Rebuild staged training comparison plots from an existing run directory.

This is a plots-only helper: it reads saved ``train_metrics_*.npz`` files and
recreates the comparison PNGs under ``<run_dir>/Plots/comparisons`` without
re-running any training.
"""

import argparse
from pathlib import Path

from core.plotting import plot_compare_phases


PHASE_SPECS = [
    ("def0_teacher", "train_metrics_def0_teacher.npz", "R_def_mean", "def"),
    ("att1_teacher", "train_metrics_att1_teacher.npz", "R_att_mean", "att"),
    ("def1_teacher", "train_metrics_def1_teacher.npz", "R_def_mean", "def"),
    ("att2_teacher", "train_metrics_att2_teacher.npz", "R_att_mean", "att"),
    ("def2_teacher", "train_metrics_def2_teacher.npz", "R_def_mean", "def"),
]


def _existing_phase_metrics(run_dir: Path):
    """Internal helper for existing phase metrics."""
    defender = []
    attacker = []
    combined = []

    for label, filename, metric_key, role in PHASE_SPECS:
        metric_path = run_dir / filename
        if not metric_path.is_file():
            continue
        item = (label, str(metric_path), metric_key)
        combined.append(item)
        if role == "def":
            defender.append(item)
        else:
            attacker.append(item)

    return defender, attacker, combined


def _plot_group(group, ylabel: str, title: str, out_dir: Path, filename: str, smooth: str, smooth_param: float):
    """Plot group using the current metrics or rollout data."""
    if len(group) < 2:
        return None
    _, saved_path = plot_compare_phases(
        group,
        metric=None,
        ylabel=ylabel,
        title=title,
        smooth=smooth,
        smooth_param=smooth_param,
        show=False,
        out_dir=str(out_dir),
        filename=filename,
    )
    return saved_path


def main():
    """Parse command-line arguments and run this script."""
    ap = argparse.ArgumentParser(
        description="Regenerate staged training comparison plots from a completed run folder.",
    )
    ap.add_argument(
        "--run_dir",
        required=True,
        help="Existing training run directory containing train_metrics_*.npz files.",
    )
    ap.add_argument(
        "--smooth",
        choices=("none", "ema", "ma"),
        default="ema",
        help="Smoothing method to apply before plotting.",
    )
    ap.add_argument(
        "--smooth_param",
        type=float,
        default=0.2,
        help="EMA alpha or moving-average window, matching core.plotting semantics.",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    comparisons_dir = run_dir / "Plots" / "comparisons"
    defender, attacker, combined = _existing_phase_metrics(run_dir)

    saved_paths = []
    maybe_path = _plot_group(
        defender,
        ylabel="Mean defender return",
        title="Defender return across phases",
        out_dir=comparisons_dir,
        filename="compare__R_def_mean__def0_vs_def1_vs_def2.png",
        smooth=args.smooth,
        smooth_param=args.smooth_param,
    )
    if maybe_path is not None:
        saved_paths.append(maybe_path)

    maybe_path = _plot_group(
        attacker,
        ylabel="Mean attacker return",
        title="Attacker return across phases",
        out_dir=comparisons_dir,
        filename="compare__R_att_mean__att1_vs_att2.png",
        smooth=args.smooth,
        smooth_param=args.smooth_param,
    )
    if maybe_path is not None:
        saved_paths.append(maybe_path)

    maybe_path = _plot_group(
        combined,
        ylabel="Mean return",
        title="Return across phases",
        out_dir=comparisons_dir,
        filename="compare__R_mean__def0_vs_att1_vs_def1_vs_att2_vs_def2.png",
        smooth=args.smooth,
        smooth_param=args.smooth_param,
    )
    if maybe_path is not None:
        saved_paths.append(maybe_path)

    if not saved_paths:
        raise RuntimeError(
            "No comparison plots were generated. The run directory needs at least two matching "
            "train_metrics_*.npz files for one of the staged comparison groups."
        )

    print("Saved comparison plots:")
    for path in saved_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()


"""
python regenerate_comparison_plots.py --run_dir Training_Policy
"""

"""
Training_Policy_Large_Arena_Staged_Static_Reward_Third_Attempt
"""