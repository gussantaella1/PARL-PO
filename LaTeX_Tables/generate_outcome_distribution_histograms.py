#!/usr/bin/env python3
"""Generate thesis outcome-distribution histograms from MC results."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


POLICIES = (
    (
        "Training_Policy_0.1u_1vmax_0.05_icVmax",
        "$u_{max}=0.1$\n$v_{max}=1$, $v_{IC}=0.05$",
    ),
    (
        "Training_Policy_0.5u_1.5vmax_1.0_icVmax",
        "$u_{max}=0.5$\n$v_{max}=1.5$, $v_{IC}=1.0$",
    ),
    (
        "Training_Policy_2.0u_1vmax_1.0_icVmax",
        "$u_{max}=2.0$\n$v_{max}=1$, $v_{IC}=1.0$",
    ),
)

TEST_CASES = (
    ("HCW 20 m", "MC_eval_20m"),
    ("HCW 100 m", "MC_eval_100m"),
    ("Elliptic LTV 20 m", "MC_eval_20m_elliptic_ltv"),
    ("Elliptic LTV 100 m", "MC_eval_100m_elliptic_ltv"),
)

MATCHUPS = (
    ("def0_vs_att1", "$\\pi_{def,0}$ vs $\\pi_{att,1}$"),
    ("def1_vs_att1", "$\\pi_{def,1}$ vs $\\pi_{att,1}$"),
    ("def1_vs_att2", "$\\pi_{def,1}$ vs $\\pi_{att,2}$"),
    ("def2_vs_att2", "$\\pi_{def,2}$ vs $\\pi_{att,2}$"),
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


def load_counts(path: Path) -> Dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = ((data.get("outcome_breakdown") or {}).get("counts") or {})
    return {str(k): int(v) for k, v in raw.items()}


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

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.8), sharey=True)
    axes = axes.ravel()
    bar_width = 0.22
    offsets = np.array([-bar_width, 0.0, bar_width])
    x = np.arange(len(POLICIES))

    for ax, (case_label, eval_dir) in zip(axes, TEST_CASES):
        distributions: List[Distribution] = []
        for policy_dir, _ in POLICIES:
            distributions.append(
                aggregate_distribution(repo_root / f"{policy_dir}{suffix}", eval_dir)
            )

        for idx, (group_label, _) in enumerate(GROUPS):
            values = [dist.percent(group_label) for dist in distributions]
            bars = ax.bar(
                x + offsets[idx],
                values,
                width=bar_width,
                color=COLORS[group_label],
                label=group_label,
            )
            for bar, value, dist in zip(bars, values, distributions):
                if dist.total <= 0:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + 1.0,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
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
                    fontsize=9,
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
                    fontsize=8,
                    color="#555555",
                )

        ax.set_title(case_label, fontsize=12, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in POLICIES], fontsize=8)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Outcome share (%)")
    axes[2].set_ylabel("Outcome share (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"Monte Carlo net outcome distributions across four learning phases ({state_label})",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.95))
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

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.8), sharey=True)
    axes = axes.ravel()
    bar_width = 0.22
    offsets = np.array([-bar_width, 0.0, bar_width])
    x = np.arange(len(MATCHUPS))

    for ax, (case_label, eval_dir) in zip(axes, TEST_CASES):
        distributions = [
            matchup_distribution(run_dir, eval_dir, matchup)
            for matchup, _ in MATCHUPS
        ]

        for idx, (group_label, _) in enumerate(GROUPS):
            values = [dist.percent(group_label) for dist in distributions]
            bars = ax.bar(
                x + offsets[idx],
                values,
                width=bar_width,
                color=COLORS[group_label],
                label=group_label,
            )
            for bar, value, dist in zip(bars, values, distributions):
                if dist.total <= 0:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + 1.0,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
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
                    fontsize=9,
                    color="#555555",
                    rotation=90,
                )

        ax.set_title(case_label, fontsize=12, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in MATCHUPS], fontsize=8)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Outcome share (%)")
    axes[2].set_ylabel("Outcome share (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"Monte Carlo outcome distributions across learning phases ({state_label})\n{policy_label}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.92))
    fig.savefig(output_path, dpi=220)
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
