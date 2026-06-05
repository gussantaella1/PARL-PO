#!/usr/bin/env python3
"""
Generate compact LaTeX tables for Monte Carlo outcome summaries.

The script scans top-level Training_Policy* folders, reads each
MC_eval_{20,50,100}m[/_elliptic_ltv]/<matchup>/results.json file, and emits
booktabs-ready tables. Each table compares the staged policy matchups with one
compact cell per phase:

    count; percent% \\raisebox{0.25ex}{\\tiny [lo%, hi%]}

The bottom rows add the total trial count and mean rollout compute time per
simulated step in milliseconds, with a 95% normal error bound.

Missing result files are rendered as N/A. By default, only top-level
Training_Policy* folders are included; pass --include-failed-runs to also scan
Failed_Runs/Training_Policy*.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


RADII_M = (20, 50, 100)
DYNAMICS = (
    ("hcw", "HCW"),
    ("elliptic_ltv", "Elliptic LTV"),
)
KF_STATES = (
    (False, "KF off"),
    (True, "KF on"),
)

PHASES = (
    ("def0/att1", "def0_vs_att1", r"$\pi_{\mathrm{def},0}$ vs $\pi_{\mathrm{att},1}$"),
    ("def1/att1", "def1_vs_att1", r"$\pi_{\mathrm{def},1}$ vs $\pi_{\mathrm{att},1}$"),
    ("def1/att2", "def1_vs_att2", r"$\pi_{\mathrm{def},1}$ vs $\pi_{\mathrm{att},2}$"),
    ("def2/att2", "def2_vs_att2", r"$\pi_{\mathrm{def},2}$ vs $\pi_{\mathrm{att},2}$"),
)

OUTCOMES = (
    ("defender_capture", "Def. capture"),
    ("attacker_hit_oi", "Att. hit OI"),
    ("defender_hit_oi", "Def. hit OI"),
    ("defender_crashed_wall", "Def. wall"),
    ("attacker_crashed_wall", "Att. wall"),
    ("timeout_no_capture", "Timeout"),
)

BOLD_OUTCOMES = {"defender_capture", "attacker_hit_oi"}
TABLE_FONT_SIZE = r"\footnotesize"


@dataclass(frozen=True)
class TableSpec:
    repo_root: Path
    run_dir: Path
    radius_m: int
    dynamics: str
    dynamics_label: str
    use_kf: bool
    kf_label: str

    @property
    def key(self) -> Tuple[str, int, str, bool]:
        return (self.display_run_dir, self.radius_m, self.dynamics, self.use_kf)

    @property
    def display_run_dir(self) -> str:
        try:
            return self.run_dir.relative_to(self.repo_root).as_posix()
        except ValueError:
            return self.run_dir.as_posix()


def canonical_dyn(value: Any) -> str:
    text = str(value or "hcw").strip().lower()
    if text in {"", "hcw"}:
        return "hcw"
    if text in {"elliptic_ltv", "elliptic-ltv", "elliptic ltv"}:
        return "elliptic_ltv"
    return text


def eval_dir_name(radius_m: int, dynamics: str) -> str:
    suffix = "" if dynamics == "hcw" else f"_{dynamics}"
    return f"MC_eval_{radius_m}m{suffix}"


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def label_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return slug or "table"


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def discover_run_dirs(repo_root: Path, include_failed_runs: bool) -> List[Path]:
    run_dirs = sorted(
        path for path in repo_root.glob("Training_Policy*") if path.is_dir()
    )
    if include_failed_runs:
        failed_root = repo_root / "Failed_Runs"
        run_dirs.extend(
            sorted(path for path in failed_root.glob("Training_Policy*") if path.is_dir())
        )
    return run_dirs


def requested_specs(repo_root: Path, run_dirs: Iterable[Path]) -> List[TableSpec]:
    specs: List[TableSpec] = []
    for run_dir in run_dirs:
        for radius_m in RADII_M:
            for dynamics, dynamics_label in DYNAMICS:
                for use_kf, kf_label in KF_STATES:
                    specs.append(
                        TableSpec(
                            repo_root=repo_root,
                            run_dir=run_dir,
                            radius_m=radius_m,
                            dynamics=dynamics,
                            dynamics_label=dynamics_label,
                            use_kf=use_kf,
                            kf_label=kf_label,
                        )
                    )
    return specs


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def fallback_counts_from_trials(results_path: Path) -> Optional[Dict[str, Any]]:
    trials_path = results_path.with_name("trials.csv")
    if not trials_path.is_file():
        return None
    counts: Counter[str] = Counter()
    n = 0
    with trials_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n += 1
            outcome = (row.get("outcome_label") or "timeout_no_capture").strip()
            counts[outcome] += 1
    return {
        "num_trials": n,
        "outcome_breakdown": {
            "counts": dict(counts),
            "proportion_stats": {},
        },
    }


def result_matches_spec(data: Dict[str, Any], spec: TableSpec) -> bool:
    cfg = data.get("rollout_cfg", {}) or {}
    actual_kf = bool(cfg.get("use_kf", False))
    actual_dyn = canonical_dyn(cfg.get("dynamics", spec.dynamics))
    try:
        actual_radius = int(round(float(cfg.get("arena_radius", spec.radius_m))))
    except (TypeError, ValueError):
        actual_radius = spec.radius_m
    return (
        actual_kf == spec.use_kf
        and actual_dyn == spec.dynamics
        and actual_radius == spec.radius_m
    )


def load_result(spec: TableSpec, matchup: str) -> Optional[Dict[str, Any]]:
    path = spec.run_dir / eval_dir_name(spec.radius_m, spec.dynamics) / matchup / "results.json"
    data = read_json(path)
    if data is None:
        return None
    if not result_matches_spec(data, spec):
        return None
    if "outcome_breakdown" not in data:
        fallback = fallback_counts_from_trials(path)
        if fallback is None:
            return None
        fallback["rollout_cfg"] = data.get("rollout_cfg", {})
        return fallback
    return data


def outcome_cell(data: Optional[Dict[str, Any]], outcome: str) -> str:
    if data is None:
        return "N/A"

    breakdown = data.get("outcome_breakdown", {}) or {}
    counts = breakdown.get("counts", {}) or {}
    stats_by_outcome = breakdown.get("proportion_stats", {}) or {}

    count = counts.get(outcome)
    if count is None:
        return "N/A"

    try:
        count_i = int(count)
    except (TypeError, ValueError):
        return "N/A"

    stats = stats_by_outcome.get(outcome, {}) or {}
    n = stats.get("n", data.get("num_trials"))
    try:
        n_i = int(n)
    except (TypeError, ValueError):
        return "N/A"
    if n_i <= 0:
        return "N/A"

    rate = stats.get("rate", count_i / n_i)
    ci = stats.get("ci_wilson", {}) or {}
    lo = ci.get("lo")
    hi = ci.get("hi")
    if lo is None or hi is None:
        lo, hi = wilson_ci(count_i, n_i)

    return (
        f"{count_i}; "
        f"{100.0 * float(rate):.1f}\\% "
        r"\raisebox{0.25ex}{\tiny ["
        f"{100.0 * float(lo):.1f}\\%, "
        f"{100.0 * float(hi):.1f}\\%]"
        r"}"
    )


def total_count_cell(data: Optional[Dict[str, Any]]) -> str:
    if data is None:
        return "N/A"
    n = data.get("num_trials")
    if n is None:
        stats_by_outcome = (
            data.get("outcome_breakdown", {})
            .get("proportion_stats", {})
        )
        for stats in stats_by_outcome.values():
            if isinstance(stats, dict) and stats.get("n") is not None:
                n = stats["n"]
                break
    try:
        return str(int(n))
    except (TypeError, ValueError):
        return "N/A"


def step_time_cell(data: Optional[Dict[str, Any]]) -> str:
    if data is None:
        return "N/A"

    summary = (
        data.get("metrics_trial", {})
        .get("rollout_total_sec_per_step", {})
    )
    if not isinstance(summary, dict):
        return "N/A"

    try:
        mean_ms = 1000.0 * float(summary["mean"])
    except (KeyError, TypeError, ValueError):
        return "N/A"

    try:
        stderr_ms = 1000.0 * float(summary.get("stderr", 0.0))
    except (TypeError, ValueError):
        stderr_ms = 0.0

    half_width = 1.959963984540054 * stderr_ms
    lo = max(0.0, mean_ms - half_width)
    hi = mean_ms + half_width
    return f"{mean_ms:.2f} ms [{lo:.2f} ms, {hi:.2f} ms]"


def step_time_summary(data: Optional[Dict[str, Any]]) -> Optional[Tuple[int, float, float]]:
    if data is None:
        return None
    summary = (
        data.get("metrics_trial", {})
        .get("rollout_total_sec_per_step", {})
    )
    if not isinstance(summary, dict):
        return None
    try:
        n = int(summary["n"])
        mean = float(summary["mean"])
    except (KeyError, TypeError, ValueError):
        return None
    if n <= 0:
        return None

    std_sample = summary.get("std_sample")
    if std_sample is None:
        stderr = summary.get("stderr")
        try:
            std_sample = float(stderr) * math.sqrt(n)
        except (TypeError, ValueError):
            std_sample = 0.0
    try:
        std = float(std_sample)
    except (TypeError, ValueError):
        std = 0.0
    return n, mean, std


def combined_step_time_cell(phase_results: Dict[str, Optional[Dict[str, Any]]]) -> str:
    summaries = [
        step_time_summary(phase_results[matchup])
        for _, matchup, _ in PHASES
    ]
    summaries = [item for item in summaries if item is not None]
    if not summaries:
        return "N/A"

    total_n = sum(n for n, _, _ in summaries)
    if total_n <= 0:
        return "N/A"
    mean = sum(n * m for n, m, _ in summaries) / total_n

    if total_n > 1:
        ss_within = sum(max(n - 1, 0) * (s ** 2) for n, _, s in summaries)
        ss_between = sum(n * ((m - mean) ** 2) for n, m, _ in summaries)
        var = (ss_within + ss_between) / (total_n - 1)
        stderr = math.sqrt(max(var, 0.0)) / math.sqrt(total_n)
    else:
        stderr = 0.0

    mean_ms = 1000.0 * mean
    half_width = 1.959963984540054 * 1000.0 * stderr
    lo = max(0.0, mean_ms - half_width)
    hi = mean_ms + half_width
    return f"{mean_ms:.2f} ms [{lo:.2f} ms, {hi:.2f} ms]"


def bold_cell(cell: str) -> str:
    if cell == "N/A":
        return cell
    return r"\textbf{" + cell + "}"


def total_count_row(phase_results: Dict[str, Optional[Dict[str, Any]]]) -> str:
    cells = [
        total_count_cell(phase_results[matchup])
        for _, matchup, _ in PHASES
    ]
    present = [cell for cell in cells if cell != "N/A"]
    if not present:
        return r"Total $n$ & \multicolumn{4}{l@{}}{N/A} \\"
    if len(set(present)) == 1 and all(cell in {"N/A", present[0]} for cell in cells):
        if all(cell != "N/A" for cell in cells):
            return r"Total $n$ & \multicolumn{4}{l@{}}{" + present[0] + r"} \\"
    return r"Total $n$ & \multicolumn{4}{l@{}}{" + ", ".join(cells) + r"} \\"


def spec_has_any_data(spec: TableSpec) -> bool:
    return any(load_result(spec, matchup) is not None for _, matchup, _ in PHASES)


def render_table(spec: TableSpec) -> str:
    phase_results = {
        matchup: load_result(spec, matchup)
        for _, matchup, _ in PHASES
    }
    caption = (
        "Monte Carlo outcomes for "
        + r"\texttt{"
        + latex_escape(spec.display_run_dir)
        + r"}, "
        + f"{spec.radius_m} m, {spec.dynamics_label}, {spec.kf_label}. "
        + r"Cells are count; \% [95\% Wilson CI]."
    )
    label = "tab:mc-" + label_slug(
        f"{spec.display_run_dir}-{spec.radius_m}m-{spec.dynamics}-"
        f"{'kf-on' if spec.use_kf else 'kf-off'}"
    )
    cols = " & ".join(phase_label for _, _, phase_label in PHASES)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        TABLE_FONT_SIZE,
        r"\setlength{\tabcolsep}{2pt}",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        "Outcome & " + cols + r" \\",
        r"\midrule",
    ]

    for outcome, display in OUTCOMES:
        cells = [
            outcome_cell(phase_results[matchup], outcome)
            for _, matchup, _ in PHASES
        ]
        row_label = latex_escape(display)
        if outcome in BOLD_OUTCOMES:
            row_label = r"\textbf{" + row_label + "}"
            cells = [bold_cell(cell) for cell in cells]
        lines.append(row_label + " & " + " & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\midrule",
            total_count_row(phase_results),
            r"Step time & \multicolumn{4}{l@{}}{"
            + combined_step_time_cell(phase_results)
            + r"} \\",
        ]
    )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def table_filename(spec: TableSpec) -> str:
    return (
        label_slug(spec.display_run_dir)
        + f"__r{spec.radius_m}m__{spec.dynamics}__"
        + ("kf_on" if spec.use_kf else "kf_off")
        + ".tex"
    )


def write_table_set(
    specs: List[TableSpec],
    out_dir: Path,
    aggregate_name: str,
    subdir_name: str,
) -> None:
    table_dir = out_dir / subdir_name
    table_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in table_dir.glob("*.tex"):
        stale_path.unlink()

    aggregate_parts = [
        "% Auto-generated by generate_mc_latex_tables.py.",
        "% Requires: \\usepackage{booktabs} and \\usepackage{graphicx}.",
        "% Phase columns are policy matchups.",
        "% Cell format: count; percent\\% \\raisebox{0.25ex}{\\tiny [lo\\%, hi\\%]}.",
        "% Bottom rows: total n and pooled rollout total ms/step [95% normal CI].",
        "% Tables use \\resizebox{\\textwidth}{!}{...} to avoid page overflow.",
        "",
    ]

    for spec in specs:
        rendered = render_table(spec)
        filename = table_filename(spec)
        (table_dir / filename).write_text(rendered + "\n", encoding="utf-8")
        aggregate_parts.append(rendered)

    (out_dir / aggregate_name).write_text("\n".join(aggregate_parts), encoding="utf-8")


def write_index(specs: List[TableSpec], out_dir: Path) -> None:
    path = out_dir / "mc_outcome_table_index.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_dir",
                "radius_m",
                "dynamics",
                "kf",
                "present_phase_count",
                "missing_phases",
                "table_filename",
            ]
        )
        for spec in specs:
            missing = []
            present = 0
            for phase_key, matchup, _ in PHASES:
                if load_result(spec, matchup) is None:
                    missing.append(phase_key)
                else:
                    present += 1
            writer.writerow(
                [
                    spec.display_run_dir,
                    spec.radius_m,
                    spec.dynamics,
                    "on" if spec.use_kf else "off",
                    present,
                    ";".join(missing),
                    table_filename(spec),
                ]
            )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_repo_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Generate compact LaTeX tables from Monte Carlo results.json files.",
    )
    parser.add_argument("--repo-root", type=Path, default=default_repo_root)
    parser.add_argument("--out-dir", type=Path, default=script_dir)
    parser.add_argument(
        "--include-failed-runs",
        action="store_true",
        help="Also scan Failed_Runs/Training_Policy* directories.",
    )
    parser.add_argument(
        "--skip-complete-grid",
        action="store_true",
        help="Do not write the complete-grid file with all-N/A missing combinations.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_run_dirs(repo_root, args.include_failed_runs)
    all_specs = requested_specs(repo_root, run_dirs)
    present_specs = [spec for spec in all_specs if spec_has_any_data(spec)]

    write_table_set(
        present_specs,
        out_dir,
        aggregate_name="mc_outcome_tables_present.tex",
        subdir_name="tables_present",
    )
    if not args.skip_complete_grid:
        write_table_set(
            all_specs,
            out_dir,
            aggregate_name="mc_outcome_tables_complete_grid.tex",
            subdir_name="tables_complete_grid",
        )
    write_index(all_specs, out_dir)

    print(f"Scanned {len(run_dirs)} run directories")
    print(f"Present-data tables: {len(present_specs)}")
    print(f"Wrote: {out_dir / 'mc_outcome_tables_present.tex'}")
    if not args.skip_complete_grid:
        print(f"Wrote: {out_dir / 'mc_outcome_tables_complete_grid.tex'}")
    print(f"Wrote: {out_dir / 'mc_outcome_table_index.csv'}")


if __name__ == "__main__":
    main()
