#!/usr/bin/env python3
"""Generate thesis outcome-distribution histograms from MC results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


POLICIES = (
    (
        "Training_Policy_0.1u_1vmax_0.05_icVmax",
        r"$\mathbf{u}_{max}=\mathbf{0.1}\ \mathbf{m/s^2}$"
        "\n"
        r"$\mathbf{v}_{max}=\mathbf{1}\ \mathbf{m/s}$"
        "\n"
        r"$\mathbf{v}_{IC}=\mathbf{0.05}\ \mathbf{m/s}$",
    ),
    (
        "Training_Policy_0.5u_1.5vmax_0.25_icVmax",
        r"$\mathbf{u}_{max}=\mathbf{0.5}\ \mathbf{m/s^2}$"
        "\n"
        r"$\mathbf{v}_{max}=\mathbf{1.5}\ \mathbf{m/s}$"
        "\n"
        r"$\mathbf{v}_{IC}=\mathbf{0.25}\ \mathbf{m/s}$",
    ),
    (
        "Training_Policy_2.0u_1vmax_1.0_icVmax",
        r"$\mathbf{u}_{max}=\mathbf{2.0}\ \mathbf{m/s^2}$"
        "\n"
        r"$\mathbf{v}_{max}=\mathbf{1}\ \mathbf{m/s}$"
        "\n"
        r"$\mathbf{v}_{IC}=\mathbf{1.0}\ \mathbf{m/s}$",
    ),
)

TEST_CASES = (
    ("HCW - 20 m radius", "MC_eval_20m"),
    ("HCW - 100 m radius", "MC_eval_100m"),
    ("Elliptic - 20 m radius", "MC_eval_20m_elliptic_ltv"),
    ("Elliptic - 100 m radius", "MC_eval_100m_elliptic_ltv"),
)

MATCHUPS = (
    ("def0_vs_att1", r"$\mathbf{\pi}_{def,0}$ vs $\mathbf{\pi}_{att,1}$"),
    ("def1_vs_att1", r"$\mathbf{\pi}_{def,1}$ vs $\mathbf{\pi}_{att,1}$"),
    ("def1_vs_att2", r"$\mathbf{\pi}_{def,1}$ vs $\mathbf{\pi}_{att,2}$"),
    ("def2_vs_att2", r"$\mathbf{\pi}_{def,2}$ vs $\mathbf{\pi}_{att,2}$"),
)

GROUPS = (
    ("Defender wins", ("defender_capture", "attacker_crashed_wall")),
    (
        "Attacker wins",
        ("attacker_hit_oi", "defender_hit_oi", "defender_crashed_wall"),
    ),
    ("Ties", ("timeout_no_capture",)),
)

COLORS = {
    "Defender wins": "#2f7d59",
    "Attacker wins": "#b8463a",
    "Ties": "#6f6f77",
}


@dataclass(frozen=True)
class Distribution:
    counts: Dict[str, int]
    total: int
    present_matchups: int

    def percent(self, label: str) -> float:
        if self.total <= 0:
            return 0.0
        return 100.0 * float(self.counts.get(label, 0)) / float(self.total)

    def percent_se(self, label: str) -> float:
        if self.total <= 0:
            return 0.0
        p = float(self.counts.get(label, 0)) / float(self.total)
        return 100.0 * (p * (1.0 - p) / float(self.total)) ** 0.5


@dataclass(frozen=True)
class MetricStats:
    def_dv: Tuple[float, float, float]
    att_dv: Tuple[float, float, float]
    step_ms: Tuple[float, float, float]
    n: int

    @property
    def present(self) -> bool:
        return self.n > 0


def load_counts(path: Path) -> Dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = ((data.get("outcome_breakdown") or {}).get("counts") or {})
    return {str(k): int(v) for k, v in raw.items()}


def _finite_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _quartiles(values: List[float]) -> Tuple[float, float, float]:
    import numpy as np

    if not values:
        return (float("nan"), float("nan"), float("nan"))
    arr = np.asarray(values, dtype=float)
    q25, q50, q75 = np.nanpercentile(arr, [25.0, 50.0, 75.0])
    return float(q50), float(q25), float(q75)


def load_metric_samples(path: Path) -> Tuple[List[float], List[float], List[float]]:
    def_dv: List[float] = []
    att_dv: List[float] = []
    step_ms: List[float] = []
    trials_path = path.parent / "trials.csv"
    if not trials_path.is_file():
        return def_dv, att_dv, step_ms

    with trials_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = _finite_float(row.get("trial_delta_v_def_total"))
            if val is not None:
                def_dv.append(val)
            val = _finite_float(row.get("trial_delta_v_att_total"))
            if val is not None:
                att_dv.append(val)
            val = _finite_float(row.get("trial_rollout_total_sec_per_step"))
            if val is not None:
                step_ms.append(1000.0 * val)
    return def_dv, att_dv, step_ms


def aggregate_metric_stats(run_dir: Path, eval_dir: str, matchups: Iterable[str]) -> MetricStats:
    def_dv: List[float] = []
    att_dv: List[float] = []
    step_ms: List[float] = []
    for matchup in matchups:
        path = run_dir / eval_dir / matchup / "results.json"
        if not path.is_file():
            continue
        d_vals, a_vals, s_vals = load_metric_samples(path)
        def_dv.extend(d_vals)
        att_dv.extend(a_vals)
        step_ms.extend(s_vals)
    return MetricStats(
        def_dv=_quartiles(def_dv),
        att_dv=_quartiles(att_dv),
        step_ms=_quartiles(step_ms),
        n=min(len(def_dv), len(att_dv), len(step_ms)),
    )


def format_metric_stats(stats: MetricStats) -> str:
    if not stats.present:
        return ""

    def _fmt_triplet(vals: Tuple[float, float, float], decimals: int) -> str:
        med, q1, q3 = vals
        return f"{med:.{decimals}f} [{q1:.{decimals}f}, {q3:.{decimals}f}]"

    return (
        r"$\Delta v_D$ "
        + _fmt_triplet(stats.def_dv, 2)
        + " m/s\n"
        + r"$\Delta v_A$ "
        + _fmt_triplet(stats.att_dv, 2)
        + " m/s\n"
        + "Step "
        + _fmt_triplet(stats.step_ms, 4)
        + " ms"
    )


def format_step_stats(stats: MetricStats) -> str:
    if not stats.present:
        return ""
    med, q1, q3 = stats.step_ms
    return rf"$\mathbf{{Step\ time\!:}}$ {med:.4f} [{q1:.4f}, {q3:.4f}] ms"


def format_total_n(stats: MetricStats) -> str:
    if not stats.present:
        return ""
    per_matchup_n = stats.n // len(MATCHUPS) if len(MATCHUPS) > 0 else stats.n
    return rf"$\mathbf{{Total\ n\ (per\ matchup)\!:}}$ {per_matchup_n}"


def format_phase_delta_v_cell(stats: MetricStats) -> str:
    if not stats.present:
        return ""
    d_med, d_q1, d_q3 = stats.def_dv
    a_med, a_q1, a_q3 = stats.att_dv
    return (
        rf"$\mathbf{{D\ \Delta v:}}$ {d_med:.2f} [{d_q1:.2f}, {d_q3:.2f}] m/s"
        "\n"
        rf"$\mathbf{{A\ \Delta v:}}$ {a_med:.2f} [{a_q1:.2f}, {a_q3:.2f}] m/s"
    )


def format_phase_delta_v_values(stats: MetricStats, agent: str) -> str:
    if not stats.present:
        return ""
    values = stats.def_dv if agent == "def" else stats.att_dv
    med, q1, q3 = values
    return f"{med:.2f} [{q1:.2f}, {q3:.2f}] m/s"


def annotate_metric_stats(ax, stats: MetricStats) -> None:
    text = format_metric_stats(stats)
    if not text:
        return
    ax.text(
        0.985,
        0.965,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        linespacing=1.15,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#b8b8b8",
            "linewidth": 0.6,
            "alpha": 0.88,
        },
    )


def add_step_time_table(ax, stats: MetricStats) -> None:
    step_text = format_step_stats(stats)
    total_text = format_total_n(stats)
    if not step_text:
        return
    table = ax.table(
        cellText=[[step_text, total_text]],
        cellLoc="center",
        loc="bottom",
        bbox=[0.0, -0.465, 1.0, 0.095],
        colWidths=[0.5, 0.5],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    for cell in table.get_celld().values():
        cell.PAD = 0.01
        cell.set_edgecolor("#d0d0d0")
        cell.set_linewidth(0.5)
        cell.set_facecolor("#fbfbfb")


def set_case_title(ax, case_label: str, *, y: float = 1.18, pad: float = 0) -> None:
    ax.set_title(case_label, fontsize=13.2, fontweight="bold", y=y, pad=pad)


def add_group_separators(ax, x_positions: Iterable[float]) -> None:
    x_vals = list(x_positions)
    for left, right in zip(x_vals, x_vals[1:]):
        ax.axvline(
            0.5 * (left + right),
            color="#404040",
            linewidth=0.8,
            linestyle="--",
            zorder=0,
        )


def add_phase_delta_v_table(ax, stats_by_phase: List[MetricStats]) -> None:
    cells = [
        [r"$\mathbf{D\ \Delta v\!:}$"]
        + [format_phase_delta_v_values(stats, "def") for stats in stats_by_phase],
        [r"$\mathbf{A\ \Delta v\!:}$"]
        + [format_phase_delta_v_values(stats, "att") for stats in stats_by_phase],
    ]
    table = ax.table(
        cellText=cells,
        cellLoc="center",
        loc="bottom",
        bbox=[-0.065, -0.365, 1.065, 0.205],
        colWidths=[0.065, 0.25, 0.25, 0.25, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    for cell in table.get_celld().values():
        cell.PAD = 0.015
        cell.set_edgecolor("#d0d0d0")
        cell.set_linewidth(0.5)
        cell.set_facecolor("#fbfbfb")


def aggregate_distribution(run_dir: Path, eval_dir: str) -> Distribution:
    grouped = {label: 0 for label, _ in GROUPS}
    total = 0
    present = 0
    for matchup, _ in MATCHUPS:
        path = run_dir / eval_dir / matchup / "results.json"
        if not path.is_file():
            continue
        counts = load_counts(path)
        present += 1
        for label, keys in GROUPS:
            grouped[label] += sum(counts.get(key, 0) for key in keys)
        total += sum(counts.values())
    return Distribution(grouped, total, present)


def matchup_distribution(run_dir: Path, eval_dir: str, matchup: str) -> Distribution:
    grouped = {label: 0 for label, _ in GROUPS}
    path = run_dir / eval_dir / matchup / "results.json"
    if not path.is_file():
        return Distribution(grouped, 0, 0)
    counts = load_counts(path)
    for label, keys in GROUPS:
        grouped[label] = sum(counts.get(key, 0) for key in keys)
    return Distribution(grouped, sum(counts.values()), 1)


def plot_histogram(repo_root: Path, out_dir: Path, use_kf: bool) -> Path:
    cache_dir = Path(tempfile.gettempdir()) / "research_repo_matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib.pyplot as plt
    import numpy as np

    suffix = "_KF_ON" if use_kf else ""
    state_label = "KF on" if use_kf else "KF off"
    output_path = out_dir / f"outcome_distribution_{'kf_on' if use_kf else 'kf_off'}.png"

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), sharey=True)
    axes = axes.ravel()
    bar_width = 0.475
    offsets = np.array([-bar_width, 0.0, bar_width])
    x = np.arange(len(POLICIES)) * 2.15

    for ax, (case_label, eval_dir) in zip(axes, TEST_CASES):
        distributions: List[Distribution] = []
        for policy_dir, _ in POLICIES:
            run_dir = repo_root / f"{policy_dir}{suffix}"
            distributions.append(aggregate_distribution(run_dir, eval_dir))

        add_group_separators(ax, x)

        for idx, (group_label, _) in enumerate(GROUPS):
            values = [dist.percent(group_label) for dist in distributions]
            errors = [dist.percent_se(group_label) for dist in distributions]
            bars = ax.bar(
                x + offsets[idx],
                values,
                yerr=errors,
                width=bar_width,
                color=COLORS[group_label],
                label=group_label,
                zorder=2,
                error_kw={
                    "ecolor": "#252525",
                    "elinewidth": 0.8,
                    "capsize": 2.5,
                    "capthick": 0.8,
                },
            )
            for bar, value, error, dist in zip(bars, values, errors, distributions):
                if dist.total <= 0:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    min(value + error + 1.0, 98.5),
                    f"{value:.1f}%\n±{error:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9.4,
                    rotation=0,
                )

        for xpos, dist in zip(x, distributions):
            if dist.present_matchups == 0:
                ax.text(
                    xpos,
                    50.0,
                    "missing",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#555555",
                    rotation=90,
                )
            elif dist.present_matchups < len(MATCHUPS):
                ax.text(
                    xpos,
                    96.0,
                    f"{dist.present_matchups}/{len(MATCHUPS)}",
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    color="#555555",
                )

        set_case_title(ax, case_label, y=1.035, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in POLICIES], fontsize=9.3)
        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Outcome share (%)", fontsize=11)
    axes[2].set_ylabel("Outcome share (%)", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        f"Monte Carlo aggregate outcome distributions across four learning phases ({state_label})",
        fontsize=16.2,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.06,
        right=0.99,
        top=0.86,
        bottom=0.12,
        hspace=0.62,
        wspace=0.08,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_policy_phase_histogram(
    repo_root: Path,
    out_dir: Path,
    policy_dir: str,
    policy_label: str,
    use_kf: bool,
) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    suffix = "_KF_ON" if use_kf else ""
    state_slug = "kf_on" if use_kf else "kf_off"
    state_label = "KF on" if use_kf else "KF off"
    policy_out_dir = out_dir / policy_dir
    policy_out_dir.mkdir(parents=True, exist_ok=True)
    output_path = policy_out_dir / f"outcome_distribution_by_phase_{state_slug}.png"
    run_dir = repo_root / f"{policy_dir}{suffix}"

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2), sharey=True)
    axes = axes.ravel()
    bar_width = 0.82
    offsets = np.array([-bar_width, 0.0, bar_width])
    x = np.arange(len(MATCHUPS)) * 2.75

    for ax, (case_label, eval_dir) in zip(axes, TEST_CASES):
        distributions = [
            matchup_distribution(run_dir, eval_dir, matchup)
            for matchup, _ in MATCHUPS
        ]
        case_stats = aggregate_metric_stats(
            run_dir,
            eval_dir,
            (matchup for matchup, _ in MATCHUPS),
        )
        phase_stats = [
            aggregate_metric_stats(run_dir, eval_dir, (matchup,))
            for matchup, _ in MATCHUPS
        ]

        add_group_separators(ax, x)

        for idx, (group_label, _) in enumerate(GROUPS):
            values = [dist.percent(group_label) for dist in distributions]
            errors = [dist.percent_se(group_label) for dist in distributions]
            bars = ax.bar(
                x + offsets[idx],
                values,
                yerr=errors,
                width=bar_width,
                color=COLORS[group_label],
                label=group_label,
                zorder=2,
                error_kw={
                    "ecolor": "#252525",
                    "elinewidth": 0.8,
                    "capsize": 2.5,
                    "capthick": 0.8,
                },
            )
            for bar, value, error, dist in zip(bars, values, errors, distributions):
                if dist.total <= 0:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    min(value + error + 1.0, 98.5),
                    f"{value:.1f}%\n±{error:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9.4,
                    rotation=0,
                )

        for xpos, dist in zip(x, distributions):
            if dist.total <= 0:
                ax.text(
                    xpos,
                    50.0,
                    "missing",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#555555",
                    rotation=90,
                )

        set_case_title(ax, case_label)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in MATCHUPS], fontsize=9.3)
        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        add_phase_delta_v_table(ax, phase_stats)
        add_step_time_table(ax, case_stats)

    axes[0].set_ylabel("Outcome share (%)", fontsize=11)
    axes[2].set_ylabel("Outcome share (%)", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"Monte Carlo outcome distributions across each learning phase\n({state_label}) {policy_label}",
        fontsize=15.4,
        y=0.98,
    )
    fig.subplots_adjust(
        left=0.045,
        right=0.995,
        top=0.82,
        bottom=0.19,
        hspace=0.90,
        wspace=0.08,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def label_slug(text: str) -> str:
    slug = []
    last_dash = False
    for ch in text.lower():
        if ch.isalnum():
            slug.append(ch)
            last_dash = False
        elif not last_dash:
            slug.append("-")
            last_dash = True
    return "".join(slug).strip("-")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_repo_root = script_dir.parent
    parser = argparse.ArgumentParser(
        description="Generate grouped defender/attacker/tie outcome histograms.",
    )
    parser.add_argument("--repo-root", type=Path, default=default_repo_root)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=script_dir / "Merged_Thesis",
        help="Directory where PNGs will be written.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for use_kf in (False, True):
        print(f"Wrote: {plot_histogram(repo_root, out_dir, use_kf)}")
        for policy_dir, policy_label in POLICIES:
            print(
                "Wrote: "
                + str(
                    plot_policy_phase_histogram(
                        repo_root,
                        out_dir,
                        policy_dir,
                        policy_label.replace("\n", ", "),
                        use_kf,
                    )
                )
            )


if __name__ == "__main__":
    main()
