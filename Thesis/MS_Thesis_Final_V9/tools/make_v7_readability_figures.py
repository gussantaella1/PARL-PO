#!/usr/bin/env python3
"""Generate V7-only readability figures without overwriting V6/V7 originals."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple


V7_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = V7_ROOT.parents[1]
SOURCE_FIG_ROOT = V7_ROOT / "Figures" / "Merged_Thesis"
OUT_FIG_ROOT = V7_ROOT / "Figures" / "Merged_Thesis_V7"

sys.path.insert(0, str(REPO_ROOT / "LaTeX_Tables"))

import generate_outcome_distribution_histograms as hist  # noqa: E402


def _prepare_matplotlib_cache() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "research_repo_matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def _case_slug(eval_dir: str) -> str:
    if eval_dir == "MC_eval_20m":
        return "hcw_20m"
    if eval_dir == "MC_eval_100m":
        return "hcw_100m"
    if eval_dir == "MC_eval_20m_elliptic_ltv":
        return "elliptic_20m"
    if eval_dir == "MC_eval_100m_elliptic_ltv":
        return "elliptic_100m"
    return hist.label_slug(eval_dir).replace("-", "_")


def _style_axis(ax) -> None:
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12, labelleft=True)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.9, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _annotate_bars(ax, bars, values, errors, distributions, fontsize: float = 11.0) -> None:
    for bar, value, error, dist in zip(bars, values, errors, distributions):
        if dist.total <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(value + error + 1.0, 98.5),
            f"{value:.1f}%\n+/-{error:.1f}%",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=0,
        )


def _add_phase_delta_v_table_readable(ax, stats_by_phase: List[hist.MetricStats]) -> None:
    cells = [
        [r"$\mathbf{D\ \Delta v\!:}$"]
        + [hist.format_phase_delta_v_values(stats, "def") for stats in stats_by_phase],
        [r"$\mathbf{A\ \Delta v\!:}$"]
        + [hist.format_phase_delta_v_values(stats, "att") for stats in stats_by_phase],
    ]
    table = ax.table(
        cellText=cells,
        cellLoc="center",
        loc="bottom",
        bbox=[-0.06, -0.292, 1.06, 0.17],
        colWidths=[0.065, 0.25, 0.25, 0.25, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.2)
    for cell in table.get_celld().values():
        cell.PAD = 0.01
        cell.set_edgecolor("#d0d0d0")
        cell.set_linewidth(0.5)
        cell.set_facecolor("#fbfbfb")


def _add_step_time_table_readable(ax, stats: hist.MetricStats) -> None:
    step_text = hist.format_step_stats(stats)
    total_text = hist.format_total_n(stats)
    if not step_text:
        return
    table = ax.table(
        cellText=[[step_text, total_text]],
        cellLoc="center",
        loc="bottom",
        bbox=[0.0, -0.365, 1.0, 0.075],
        colWidths=[0.5, 0.5],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.0)
    for cell in table.get_celld().values():
        cell.PAD = 0.01
        cell.set_edgecolor("#d0d0d0")
        cell.set_linewidth(0.5)
        cell.set_facecolor("#fbfbfb")


def plot_aggregate_one_column(use_kf: bool) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    suffix = "_KF_ON" if use_kf else ""
    state_slug = "kf_on" if use_kf else "kf_off"
    state_label = "KF on" if use_kf else "KF off"
    output_path = OUT_FIG_ROOT / f"outcome_distribution_{state_slug}_one_column.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(11.0, 15.6), sharey=True)
    bar_width = 0.42
    offsets = np.array([-bar_width, 0.0, bar_width])
    x = np.arange(len(hist.POLICIES)) * 2.2

    for ax, (case_label, eval_dir) in zip(axes, hist.TEST_CASES):
        distributions = [
            hist.aggregate_distribution(REPO_ROOT / f"{policy_dir}{suffix}", eval_dir)
            for policy_dir, _ in hist.POLICIES
        ]
        hist.add_group_separators(ax, x)
        for idx, (group_label, _) in enumerate(hist.GROUPS):
            values = [dist.percent(group_label) for dist in distributions]
            errors = [dist.percent_se(group_label) for dist in distributions]
            bars = ax.bar(
                x + offsets[idx],
                values,
                yerr=errors,
                width=bar_width,
                color=hist.COLORS[group_label],
                label=group_label,
                zorder=2,
                error_kw={"ecolor": "#252525", "elinewidth": 0.9, "capsize": 3.0, "capthick": 0.9},
            )
            _annotate_bars(ax, bars, values, errors, distributions, fontsize=10.8)

        hist.set_case_title(ax, case_label, y=1.035, pad=7)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in hist.POLICIES], fontsize=11.2)
        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")
        ax.set_ylabel("Outcome share (%)", fontsize=12.5)
        _style_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.957), ncol=3, frameon=False, fontsize=12.0)
    fig.suptitle(
        f"Monte Carlo aggregate outcome distributions across four learning phases ({state_label})",
        fontsize=17.0,
        y=0.992,
    )
    # The bottom subplot uses multiline tick labels, so keep extra canvas below
    # the fourth panel to avoid clipping after LaTeX rescales the exported PNG.
    fig.subplots_adjust(left=0.095, right=0.985, top=0.90, bottom=0.075, hspace=0.46)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output_path


def plot_policy_phase_one_column(policy_dir: str, policy_label: str, use_kf: bool) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    suffix = "_KF_ON" if use_kf else ""
    state_slug = "kf_on" if use_kf else "kf_off"
    state_label = "KF on" if use_kf else "KF off"
    output_dir = OUT_FIG_ROOT / policy_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"outcome_distribution_by_phase_{state_slug}_one_column.png"
    run_dir = REPO_ROOT / f"{policy_dir}{suffix}"

    fig, axes = plt.subplots(4, 1, figsize=(12.4, 18.8), sharey=True)
    bar_width = 0.68
    offsets = np.array([-bar_width, 0.0, bar_width])
    x = np.arange(len(hist.MATCHUPS)) * 2.7

    for ax, (case_label, eval_dir) in zip(axes, hist.TEST_CASES):
        distributions = [
            hist.matchup_distribution(run_dir, eval_dir, matchup)
            for matchup, _ in hist.MATCHUPS
        ]
        case_stats = hist.aggregate_metric_stats(
            run_dir,
            eval_dir,
            (matchup for matchup, _ in hist.MATCHUPS),
        )
        phase_stats = [
            hist.aggregate_metric_stats(run_dir, eval_dir, (matchup,))
            for matchup, _ in hist.MATCHUPS
        ]

        hist.add_group_separators(ax, x)
        for idx, (group_label, _) in enumerate(hist.GROUPS):
            values = [dist.percent(group_label) for dist in distributions]
            errors = [dist.percent_se(group_label) for dist in distributions]
            bars = ax.bar(
                x + offsets[idx],
                values,
                yerr=errors,
                width=bar_width,
                color=hist.COLORS[group_label],
                label=group_label,
                zorder=2,
                error_kw={"ecolor": "#252525", "elinewidth": 0.9, "capsize": 3.0, "capthick": 0.9},
            )
            _annotate_bars(ax, bars, values, errors, distributions, fontsize=10.4)

        hist.set_case_title(ax, case_label, y=1.08)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in hist.MATCHUPS], fontsize=10.7)
        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")
        ax.set_ylabel("Outcome share (%)", fontsize=12.3)
        _style_axis(ax)
        _add_phase_delta_v_table_readable(ax, phase_stats)
        _add_step_time_table_readable(ax, case_stats)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.947), ncol=3, frameon=False, fontsize=12.0)
    fig.suptitle(
        f"Monte Carlo outcome distributions across each learning phase\n({state_label}) {policy_label}",
        fontsize=17.0,
        y=0.994,
    )
    # Keep extra canvas below the final panel so the embedded summary tables
    # remain visibly separated from the image edge after LaTeX rescales them.
    fig.subplots_adjust(left=0.075, right=0.99, top=0.875, bottom=0.04, hspace=0.70)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return output_path


def generate_histograms() -> None:
    _prepare_matplotlib_cache()
    for use_kf in (False, True):
        print(f"Wrote: {plot_aggregate_one_column(use_kf)}")
        for policy_dir, policy_label in hist.POLICIES:
            print(
                "Wrote: "
                + str(
                    plot_policy_phase_one_column(
                        policy_dir,
                        policy_label.replace("\n", ", "),
                        use_kf,
                    )
                )
            )


def crop_rollout_panels() -> None:
    from PIL import Image

    crop_boxes: List[Tuple[str, Tuple[int, int, int, int]]] = [
        ("distance_to_center", (0, 80, 1545, 1340)),
        ("relative_distance", (0, 1400, 1545, 2734)),
        ("thrust_agent_1", (1560, 80, 3090, 1340)),
        ("thrust_agent_2", (1560, 1400, 3090, 2734)),
        ("velocity_agent_1", (3160, 80, 4645, 1340)),
        ("velocity_agent_2", (3160, 1400, 4645, 2734)),
        ("delta_v", (4685, 640, 6034, 2230)),
    ]

    for rollout_dir in (
        V7_ROOT / "Images" / "Defender_Win_Rollout",
        V7_ROOT / "Images" / "Attacker_Win_Rollout",
    ):
        grid_path = rollout_dir / "grid.png"
        out_dir = rollout_dir / "Panels"
        out_dir.mkdir(parents=True, exist_ok=True)
        image = Image.open(grid_path)
        for name, box in crop_boxes:
            image.crop(box).save(out_dir / f"{name}.png")
            print(f"Wrote: {out_dir / f'{name}.png'}")


def crop_rollout_snapshots() -> None:
    from PIL import Image, ImageChops

    for rollout_dir in (
        V7_ROOT / "Images" / "Defender_Win_Rollout",
        V7_ROOT / "Images" / "Attacker_Win_Rollout",
    ):
        out_dir = rollout_dir / "Snapshots_V7"
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(4):
            candidates = sorted(rollout_dir.glob(f"{idx}.*"))
            if not candidates:
                continue

            source_path = candidates[0]
            image = Image.open(source_path).convert("RGB")
            background = Image.new("RGB", image.size, (255, 255, 255))
            diff = ImageChops.difference(image, background)
            mask = diff.convert("L").point(lambda p: 255 if p > 8 else 0)
            bbox = mask.getbbox()
            if bbox:
                pad = 16
                left = max(0, bbox[0] - pad)
                top = max(0, bbox[1] - pad)
                right = min(image.width, bbox[2] + pad)
                bottom = min(image.height, bbox[3] + pad)
                image = image.crop((left, top, right, bottom))

            output_path = out_dir / f"{idx}.png"
            image.save(output_path)
            print(f"Wrote: {output_path}")


def main() -> None:
    generate_histograms()
    crop_rollout_panels()
    crop_rollout_snapshots()


if __name__ == "__main__":
    main()
