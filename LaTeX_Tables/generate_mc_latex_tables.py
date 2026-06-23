#!/usr/bin/env python3
"""
Generate compact LaTeX tables for Monte Carlo outcome summaries.

The script scans top-level Training_Policy* folders, reads each
MC_eval_{20,50,100}m[/_elliptic_ltv]/<matchup>/results.json file, and emits
booktabs-ready tables. Each table compares the staged policy matchups with one
compact cell per phase:

    count; $percent \\pm se\\%$

The bottom rows add the total trial count and rollout compute time per
simulated step in milliseconds as median [Q1, Q3].

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
import shutil
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

OUTCOME_GROUPS = (
    (
        "Defender wins",
        (
            ("defender_capture", "Def. capture"),
            ("attacker_crashed_wall", "Att. wall"),
        ),
    ),
    (
        "Attacker wins",
        (
            ("attacker_hit_oi", "Att. hit OI"),
            ("defender_hit_oi", "Def. hit OI"),
            ("defender_crashed_wall", "Def. wall"),
        ),
    ),
    (
        "Ties",
        (
            ("timeout_no_capture", "Timeout"),
        ),
    ),
)

BOLD_OUTCOMES = {"defender_capture", "attacker_hit_oi"}
TABLE_FONT_SIZE = r"\scriptsize"
TABLE_PLACEMENT = "H"
TABLE_RESIZE_WIDTH = r"\dimexpr\textwidth+0.5in\relax"
TABLE_ARRAYSTRETCH = "0.94"
THESIS_TABLE_RESIZE_WIDTH = r"\dimexpr\textwidth+0.75in\relax"
THESIS_TABLE_ARRAYSTRETCH = "0.94"
THESIS_KF_BLOCK_VSPACE = "0.35em"
TABLE_NEEDSPACE = r"0.82\textheight"
THESIS_TABLE_NEEDSPACE = r"0.92\textheight"
TRAINING_CURVE_NEEDSPACE = r"0.45\textheight"
TABLE_COLSEP = "1pt"
OUTCOME_GROUP_ADDLINESPACE = "0.18em"
DELTA_V_UNIT = r"$\mathrm{m}/\mathrm{s}$"
NA_CELL = r"\makebox[8em][c]{\tiny N/A}"
TABLE_STATS_NOTE = (
    r" Cells: count; \% $\pm$ SE. "
    r"$\Delta v$ and time: median [Q1, Q3]."
)
MERGED_THESIS_FILENAME = "merged_thesis_tables.tex"
MERGED_THESIS_INPUTS_FILENAME = "merged_thesis_inputs.tex"
TRAINING_CURVE_FILENAME = "compare__R_mean__def0_vs_att1_vs_def1_vs_att2_vs_def2.png"
KF_ON_SUFFIX = "_KF_ON"
THESIS_SELECTIONS = (
    ("HCW", "hcw", (20, 100)),
    ("Elliptic LTV", "elliptic_ltv", (20, 100)),
)


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


def binary_stderr_from_count(count: int, n: int) -> float:
    if n <= 0:
        return math.nan
    p = max(0.0, min(1.0, float(count) / float(n)))
    return math.sqrt(max(0.0, p * (1.0 - p) / float(n)))


def proportion_stderr_from_stats(stats: Dict[str, Any], count: int, n: int) -> float:
    try:
        stderr = float(stats.get("stderr"))
    except (TypeError, ValueError):
        stderr = math.nan
    if math.isfinite(stderr):
        return max(0.0, stderr)
    return binary_stderr_from_count(count, n)


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
        return NA_CELL

    breakdown = data.get("outcome_breakdown", {}) or {}
    counts = breakdown.get("counts", {}) or {}
    stats_by_outcome = breakdown.get("proportion_stats", {}) or {}

    count = counts.get(outcome)
    if count is None:
        return NA_CELL

    try:
        count_i = int(count)
    except (TypeError, ValueError):
        return NA_CELL

    stats = stats_by_outcome.get(outcome, {}) or {}
    n = stats.get("n", data.get("num_trials"))
    try:
        n_i = int(n)
    except (TypeError, ValueError):
        return NA_CELL
    if n_i <= 0:
        return NA_CELL

    rate = stats.get("rate", count_i / n_i)
    stderr = proportion_stderr_from_stats(stats, count_i, n_i)

    return (
        f"{count_i}; "
        r"$"
        f"{100.0 * float(rate):.1f}"
        r" \pm "
        f"{100.0 * float(stderr):.1f}\\%"
        r"$"
    )


def total_count_cell(data: Optional[Dict[str, Any]]) -> str:
    if data is None:
        return NA_CELL
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
        return NA_CELL


def step_time_cell(data: Optional[Dict[str, Any]]) -> str:
    if data is None:
        return NA_CELL

    summary = (
        data.get("metrics_trial", {})
        .get("rollout_total_sec_per_step", {})
    )
    if not isinstance(summary, dict):
        return NA_CELL

    try:
        q25_ms = 1000.0 * float(summary["q25"])
        q50_ms = 1000.0 * float(summary["q50"])
        q75_ms = 1000.0 * float(summary["q75"])
    except (KeyError, TypeError, ValueError):
        return NA_CELL

    return step_time_iqr_cell(q25_ms, q50_ms, q75_ms)


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


def step_time_iqr_summary(data: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float, float]]:
    if data is None:
        return None
    summary = (
        data.get("metrics_trial", {})
        .get("rollout_total_sec_per_step", {})
    )
    if not isinstance(summary, dict):
        return None
    try:
        return (
            1000.0 * float(summary["q25"]),
            1000.0 * float(summary["q50"]),
            1000.0 * float(summary["q75"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def step_time_iqr_cell(q25_ms: float, q50_ms: float, q75_ms: float) -> str:
    return (
        r"$"
        f"{q50_ms:.4f}"
        r"\,["
        f"{q25_ms:.4f}, {q75_ms:.4f}"
        r"]$ ms"
    )


def combined_step_time_cell(phase_results: Dict[str, Optional[Dict[str, Any]]]) -> str:
    iqr_summaries = [
        step_time_iqr_summary(phase_results[matchup])
        for _, matchup, _ in PHASES
    ]
    iqr_summaries = [item for item in iqr_summaries if item is not None]
    if iqr_summaries:
        q25_ms = sum(item[0] for item in iqr_summaries) / len(iqr_summaries)
        q50_ms = sum(item[1] for item in iqr_summaries) / len(iqr_summaries)
        q75_ms = sum(item[2] for item in iqr_summaries) / len(iqr_summaries)
        return step_time_iqr_cell(q25_ms, q50_ms, q75_ms)

    summaries = [step_time_summary(phase_results[matchup]) for _, matchup, _ in PHASES]
    summaries = [item for item in summaries if item is not None]
    if not summaries:
        return NA_CELL

    total_n = sum(n for n, _, _ in summaries)
    if total_n <= 0:
        return NA_CELL
    mean = sum(n * m for n, m, _ in summaries) / total_n

    if total_n > 1:
        ss_within = sum(max(n - 1, 0) * (s ** 2) for n, _, s in summaries)
        ss_between = sum(n * ((m - mean) ** 2) for n, m, _ in summaries)
        var = (ss_within + ss_between) / (total_n - 1)
        stderr = math.sqrt(max(var, 0.0)) / math.sqrt(total_n)
    else:
        stderr = 0.0

    return step_time_iqr_cell(1000.0 * mean, 1000.0 * mean, 1000.0 * mean)


def quartile_metric_cell(data: Optional[Dict[str, Any]], metric_key: str, unit: str) -> str:
    if data is None:
        return NA_CELL

    summary = (
        data.get("metrics_trial", {})
        .get(metric_key, {})
    )
    if not isinstance(summary, dict):
        return NA_CELL

    try:
        q25 = float(summary["q25"])
        q50 = float(summary["q50"])
        q75 = float(summary["q75"])
    except (KeyError, TypeError, ValueError):
        return NA_CELL

    return (
        r"$"
        f"{q50:.2f}"
        r"\,["
        f"{q25:.2f}, {q75:.2f}"
        r"]$ "
        + unit
    )


def delta_v_row(
    phase_results: Dict[str, Optional[Dict[str, Any]]],
    metric_key: str,
    label: str,
) -> str:
    cells = [
        quartile_metric_cell(phase_results[matchup], metric_key, DELTA_V_UNIT)
        for _, matchup, _ in PHASES
    ]
    return label + " & " + " & ".join(cells) + r" \\"


def bold_cell(cell: str) -> str:
    if cell == NA_CELL:
        return cell
    return r"{\bfseries\boldmath " + cell + "}"


def is_na_cell(cell: str) -> bool:
    return cell == NA_CELL


def total_count_row(phase_results: Dict[str, Optional[Dict[str, Any]]]) -> str:
    cells = [
        total_count_cell(phase_results[matchup])
        for _, matchup, _ in PHASES
    ]
    present = [cell for cell in cells if not is_na_cell(cell)]
    if not present:
        return r"Total $n$ & \multicolumn{4}{l@{}}{" + NA_CELL + r"} \\"
    if len(set(present)) == 1 and all(is_na_cell(cell) or cell == present[0] for cell in cells):
        if all(not is_na_cell(cell) for cell in cells):
            return r"Total $n$ & \multicolumn{4}{l@{}}{" + present[0] + r"} \\"
    return r"Total $n$ & \multicolumn{4}{l@{}}{" + ", ".join(cells) + r"} \\"


def spec_has_any_data(spec: TableSpec) -> bool:
    return any(load_result(spec, matchup) is not None for _, matchup, _ in PHASES)


def training_identifier(params: Dict[str, str]) -> str:
    return (
        r"$u_{\max} = "
        + latex_escape(params["umax"])
        + r"$, $v_{\max} = "
        + latex_escape(params["vmax"])
        + r"$, $v_{\max}^{\mathrm{IC}} = "
        + latex_escape(params["ic_vmax"])
        + r"$"
    )


def phase_results_for_spec(spec: TableSpec) -> Dict[str, Optional[Dict[str, Any]]]:
    return {
        matchup: load_result(spec, matchup)
        for _, matchup, _ in PHASES
    }


def phase_column_headers() -> str:
    return " & ".join(
        r"{\boldmath\textbf{" + phase_label + r"}}"
        for _, _, phase_label in PHASES
    )


def render_outcome_tabular(spec: TableSpec) -> str:
    phase_results = phase_results_for_spec(spec)
    cols = phase_column_headers()

    lines = [
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"\textbf{Outcome} & " + cols + r" \\",
        r"\midrule",
    ]

    for group_label, outcomes in OUTCOME_GROUPS:
        lines.extend(
            [
                rf"\addlinespace[{OUTCOME_GROUP_ADDLINESPACE}]",
                r"\multicolumn{5}{@{}l@{}}{\underline{\textit{"
                + latex_escape(group_label)
                + r":}}} \\",
            ]
        )
        for outcome, display in outcomes:
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
            r"\multicolumn{5}{@{}l@{}}{\underline{\textit{Delta-v statistics:}}} \\",
            delta_v_row(phase_results, "delta_v_def_total", r"Def. total $\Delta v$"),
            delta_v_row(phase_results, "delta_v_att_total", r"Att. total $\Delta v$"),
            r"\midrule",
            total_count_row(phase_results),
            r"Step time & \multicolumn{4}{l@{}}{"
            + combined_step_time_cell(phase_results)
            + r"} \\",
            r"\bottomrule",
            r"\end{tabular}%",
        ]
    )
    return "\n".join(lines)


def render_table(spec: TableSpec, policy_caption: Optional[str] = None) -> str:
    policy_desc = (
        policy_caption
        if policy_caption is not None
        else r"\texttt{" + latex_escape(spec.display_run_dir) + r"}"
    )
    caption = (
        "Monte Carlo outcomes for "
        + policy_desc
        + ", "
        + f"{spec.radius_m} m, {spec.dynamics_label}, {spec.kf_label}."
        + TABLE_STATS_NOTE
    )
    label = "tab:mc-" + label_slug(
        f"{spec.display_run_dir}-{spec.radius_m}m-{spec.dynamics}-"
        f"{'kf-on' if spec.use_kf else 'kf-off'}"
    )

    lines = [
        rf"\Needspace{{{TABLE_NEEDSPACE}}}",
        rf"\begin{{table}}[{TABLE_PLACEMENT}]",
        r"\centering",
        TABLE_FONT_SIZE,
        rf"\setlength{{\tabcolsep}}{{{TABLE_COLSEP}}}",
        rf"\renewcommand{{\arraystretch}}{{{TABLE_ARRAYSTRETCH}}}",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\makebox[\textwidth][c]{%",
        r"\resizebox{" + TABLE_RESIZE_WIDTH + r"}{!}{%",
        render_outcome_tabular(spec),
        r"}%",
        r"}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def radius_heading(radius_m: int) -> str:
    return f"{radius_m} meter radius arena"


def table_filename(spec: TableSpec) -> str:
    return (
        label_slug(spec.display_run_dir)
        + f"__r{spec.radius_m}m__{spec.dynamics}__"
        + ("kf_on" if spec.use_kf else "kf_off")
        + ".tex"
    )


def grouped_table_path(spec: TableSpec) -> Path:
    return Path(spec.display_run_dir) / (
        f"r{spec.radius_m}m__{spec.dynamics}__"
        + ("kf_on" if spec.use_kf else "kf_off")
        + ".tex"
    )


def display_path(repo_root: Path, path: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def strip_kf_on_suffix(name: str) -> str:
    if name.endswith(KF_ON_SUFFIX):
        return name[: -len(KF_ON_SUFFIX)]
    return name


def base_run_dir_for(run_dir: Path) -> Path:
    return run_dir.with_name(strip_kf_on_suffix(run_dir.name))


def kf_on_run_dir_for(base_run_dir: Path) -> Path:
    return base_run_dir.with_name(base_run_dir.name + KF_ON_SUFFIX)


def parse_training_params(run_dir: Path) -> Dict[str, str]:
    name = run_dir.name
    match = re.search(
        r"Training_Policy_(?P<umax>[0-9.]+)u_(?P<vmax>[0-9.]+)vmax_(?P<ic_vmax>[0-9.]+)_icVmax",
        name,
        flags=re.IGNORECASE,
    )
    if match is None:
        return {"umax": "N/A", "vmax": "N/A", "ic_vmax": "N/A"}
    return match.groupdict()


def render_kf_table_block(label: str, spec: TableSpec) -> str:
    lines = [
        r"\textbf{" + latex_escape(label) + r"}\\[0.25em]",
        r"{\setlength{\tabcolsep}{" + TABLE_COLSEP + r"}",
        rf"\renewcommand{{\arraystretch}}{{{THESIS_TABLE_ARRAYSTRETCH}}}",
        r"\makebox[\textwidth][c]{%",
        r"\resizebox{" + THESIS_TABLE_RESIZE_WIDTH + r"}{!}{%",
        render_outcome_tabular(spec),
        r"}%",
        r"}",
        r"}",
    ]
    return "\n".join(lines)


def render_paired_kf_table(
    display_run_dir: str,
    radius_m: int,
    dynamics: str,
    dynamics_label: str,
    kf_off_spec: TableSpec,
    kf_on_spec: TableSpec,
    policy_caption: str,
) -> str:
    caption = (
        f"Monte Carlo outcomes: {dynamics_label}, "
        + radius_heading(radius_m)
        + ", "
        + policy_caption
        + r"."
        + TABLE_STATS_NOTE
    )
    label = "tab:mc-" + label_slug(
        f"{display_run_dir}-{radius_m}m-{dynamics}-kf-off-vs-on"
    )
    lines = [
        rf"\Needspace{{{THESIS_TABLE_NEEDSPACE}}}",
        rf"\begin{{table}}[{TABLE_PLACEMENT}]",
        r"\centering",
        TABLE_FONT_SIZE,
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        render_kf_table_block("KF off", kf_off_spec),
        rf"\vspace{{{THESIS_KF_BLOCK_VSPACE}}}",
        "",
        render_kf_table_block("KF on", kf_on_spec),
        r"\end{table}",
        r"\clearpage",
        "",
    ]
    return "\n".join(lines)


def copy_training_curve(run_dir: Path, merged_dir: Path, filename: str) -> Optional[Path]:
    src = run_dir / "Plots" / "comparisons" / TRAINING_CURVE_FILENAME
    if not src.is_file():
        return None
    dst = merged_dir / filename
    shutil.copy2(src, dst)
    return dst


def latex_graphics_path(repo_root: Path, path: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def render_training_curve_panel(label: str, image_path: Optional[Path], repo_root: Path) -> str:
    lines = [
        r"\begin{minipage}[t]{0.48\textwidth}",
        r"\centering",
        r"\textbf{" + latex_escape(label) + r"}\\[0.25em]",
    ]
    if image_path is None:
        lines.append(
            r"\fbox{\parbox[c][1.55in][c]{0.92\linewidth}{\centering N/A}}"
        )
    else:
        graphics_path = latex_graphics_path(repo_root, image_path)
        lines.append(
            r"\includegraphics[width=\linewidth]{\detokenize{"
            + graphics_path
            + r"}}"
        )
    lines.append(r"\end{minipage}")
    return "\n".join(lines)


def render_training_curves_figure(
    repo_root: Path,
    merged_dir: Path,
    display_run_dir: str,
    base_run_dir: Path,
    kf_on_run_dir: Path,
    policy_caption: str,
) -> str:
    kf_off_curve = copy_training_curve(
        base_run_dir,
        merged_dir,
        "combined_training_curve_kf_off.png",
    )
    kf_on_curve = copy_training_curve(
        kf_on_run_dir,
        merged_dir,
        "combined_training_curve_kf_on.png",
    )
    label = "fig:mc-" + label_slug(display_run_dir) + "-training-curves-kf"
    lines = [
        rf"\Needspace{{{TRAINING_CURVE_NEEDSPACE}}}",
        rf"\begin{{figure}}[{TABLE_PLACEMENT}]",
        r"\centering",
        render_training_curve_panel("KF off", kf_off_curve, repo_root),
        r"\hfill",
        render_training_curve_panel("KF on", kf_on_curve, repo_root),
        r"\caption{Combined training return curves for "
        + policy_caption
        + r".}",
        r"\label{" + label + r"}",
        r"\end{figure}",
        "",
    ]
    return "\n".join(lines)


def write_merged_thesis_tables(
    all_specs: List[TableSpec],
    present_specs: List[TableSpec],
    out_dir: Path,
) -> None:
    if not present_specs:
        return

    repo_root = all_specs[0].repo_root if all_specs else present_specs[0].repo_root
    merged_root = out_dir / "Merged_Thesis"
    if merged_root.exists():
        shutil.rmtree(merged_root)
    merged_root.mkdir(parents=True, exist_ok=True)

    present_base_specs: List[TableSpec] = []
    seen_base_run_dirs = set()
    for spec in present_specs:
        base_run_dir = base_run_dir_for(spec.run_dir)
        if base_run_dir in seen_base_run_dirs:
            continue
        seen_base_run_dirs.add(base_run_dir)
        present_base_specs.append(spec)

    specs_by_source = {
        (spec.run_dir, spec.radius_m, spec.dynamics, spec.use_kf): spec
        for spec in all_specs
    }
    input_paths: List[Path] = []
    for run_spec in present_base_specs:
        base_run_dir = base_run_dir_for(run_spec.run_dir)
        kf_on_run_dir = kf_on_run_dir_for(base_run_dir)
        display_run_dir = display_path(repo_root, base_run_dir)
        policy_caption = training_identifier(parse_training_params(base_run_dir))
        merged_dir = merged_root / Path(display_run_dir)
        merged_dir.mkdir(parents=True, exist_ok=True)

        parts = [
            "% Auto-generated by generate_mc_latex_tables.py.",
            "% Requires: \\usepackage{booktabs}, \\usepackage{graphicx}, \\usepackage{float}, and \\usepackage{needspace}.",
            "% Thesis selection: training curves, HCW 20/100 m KF off/on, Elliptic LTV 20/100 m KF off/on.",
            "% KF-off tables are read from the base run folder; KF-on tables are read from the matching _KF_ON folder.",
            "",
            r"\section{Training case: " + policy_caption + r"}",
            "",
            render_training_curves_figure(
                repo_root,
                merged_dir,
                display_run_dir,
                base_run_dir,
                kf_on_run_dir,
                policy_caption,
            ),
        ]

        for dynamics_label, dynamics, radii_m in THESIS_SELECTIONS:
            parts.extend(["", r"\subsection{" + dynamics_label + r"}", ""])
            for radius_m in radii_m:
                kf_off_spec = specs_by_source.get(
                    (base_run_dir, radius_m, dynamics, False)
                )
                if kf_off_spec is None:
                    kf_off_spec = TableSpec(
                        repo_root=repo_root,
                        run_dir=base_run_dir,
                        radius_m=radius_m,
                        dynamics=dynamics,
                        dynamics_label=dynamics_label,
                        use_kf=False,
                        kf_label="KF off",
                    )

                kf_on_spec = specs_by_source.get(
                    (kf_on_run_dir, radius_m, dynamics, True)
                )
                if kf_on_spec is None:
                    kf_on_spec = TableSpec(
                        repo_root=repo_root,
                        run_dir=kf_on_run_dir,
                        radius_m=radius_m,
                        dynamics=dynamics,
                        dynamics_label=dynamics_label,
                        use_kf=True,
                        kf_label="KF on",
                    )

                parts.extend(
                    [
                        r"\subsubsection{" + radius_heading(radius_m) + r"}",
                        "",
                        render_paired_kf_table(
                            display_run_dir,
                            radius_m,
                            dynamics,
                            dynamics_label,
                            kf_off_spec,
                            kf_on_spec,
                            policy_caption,
                        ),
                    ]
                )

        merged_file = merged_dir / MERGED_THESIS_FILENAME
        merged_file.write_text("\n".join(parts), encoding="utf-8")
        input_paths.append(merged_file)

    input_lines = [
        "% Auto-generated by generate_mc_latex_tables.py.",
        "% Include this file from a repo-root LaTeX document with:",
        "% \\input{LaTeX_Tables/Merged_Thesis/merged_thesis_inputs.tex}",
        "% Comment out individual \\input lines below while iterating.",
        "",
    ]
    for path in input_paths:
        input_lines.append(r"\input{" + latex_graphics_path(repo_root, path) + r"}")
    input_lines.append("")
    (merged_root / MERGED_THESIS_INPUTS_FILENAME).write_text(
        "\n".join(input_lines),
        encoding="utf-8",
    )


def write_table_set(
    specs: List[TableSpec],
    out_dir: Path,
    aggregate_name: str,
    subdir_name: str,
    group_by_run_dir: bool = False,
) -> None:
    table_dir = out_dir / subdir_name
    if table_dir.exists():
        shutil.rmtree(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    aggregate_parts = [
        "% Auto-generated by generate_mc_latex_tables.py.",
        "% Requires: \\usepackage{booktabs}, \\usepackage{graphicx}, \\usepackage{float}, and \\usepackage{needspace}.",
        "% Phase columns are policy matchups.",
        "% Outcome rows are grouped into defender wins, attacker wins, and ties.",
        "% Outcome cells: count; $percent \\pm SE\\%$.",
        "% Delta-v rows: median [Q1, Q3].",
        "% Step time row: median [Q1, Q3] ms/step.",
        f"% Tables use [{TABLE_PLACEMENT}] placement with \\Needspace guards to keep declaration-order placement predictable.",
        f"% Table row spacing uses \\arraystretch={TABLE_ARRAYSTRETCH}.",
        "% Tables are centered in a \\makebox and resized to \\textwidth+0.5in.",
        "% This allows roughly 0.25in overhang into each margin for larger text.",
        f"% Column padding uses \\tabcolsep={TABLE_COLSEP} to reduce scaling shrinkage.",
        "",
    ]

    current_run_dir: Optional[str] = None
    for spec in specs:
        rendered = render_table(spec)
        rel_path = grouped_table_path(spec) if group_by_run_dir else Path(table_filename(spec))
        target_path = table_dir / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(rendered + "\n", encoding="utf-8")
        if group_by_run_dir and spec.display_run_dir != current_run_dir:
            current_run_dir = spec.display_run_dir
            aggregate_parts.extend(
                [
                    "",
                    "% " + current_run_dir,
                    "",
                ]
            )
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
                "present_table_path",
                "complete_grid_table_filename",
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
                    grouped_table_path(spec).as_posix() if present > 0 else "",
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
        group_by_run_dir=True,
    )
    write_merged_thesis_tables(all_specs, present_specs, out_dir)
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
    print(f"Wrote: {out_dir / 'Merged_Thesis'}")
    if not args.skip_complete_grid:
        print(f"Wrote: {out_dir / 'mc_outcome_tables_complete_grid.tex'}")
    print(f"Wrote: {out_dir / 'mc_outcome_table_index.csv'}")


if __name__ == "__main__":
    main()
