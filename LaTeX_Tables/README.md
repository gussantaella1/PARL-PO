# Monte Carlo LaTeX Tables

Regenerate the tables from the repository root:

```bash
python "LaTeX_Tables/generate_mc_latex_tables.py"
```

Outputs:

- `mc_outcome_tables_present.tex`: compact tables for policy/radius/dynamics/KF combinations with at least one result file.
- `mc_outcome_tables_complete_grid.tex`: all top-level `Training_Policy*` folders crossed with 20/50/100 m, HCW/elliptic LTV, and KF on/off; unavailable phase cells are `N/A`.
- `tables_present/`: present-data tables grouped under the exact
  `Training_Policy*` folder name they came from.
- `Merged_Thesis/<Training_Policy*>/merged_thesis_tables.tex`:
  thesis-oriented selection for that base run, with side-by-side combined
  training curves for KF off/on and stacked KF-off/KF-on result tables for HCW
  and elliptic LTV at the 20 m and 100 m arena radii. KF-off results and curves
  are read from the base run folder, and KF-on results and curves are read from
  the matching folder with `_KF_ON` appended to the same training folder name.
  The visible headings and captions use parameter-based identifiers instead of
  literal folder names, with training cases as `\section{...}`, dynamics as
  `\subsection{...}`, and arena radii as `\subsubsection{...}`.
- `Merged_Thesis/merged_thesis_inputs.tex`: a small include manifest with one
  `\input{...}` line per merged training-case file. From a repo-root document
  such as `testing.tex`, include all merged tables with:
  `\input{LaTeX_Tables/Merged_Thesis/merged_thesis_inputs.tex}`. For faster
  iteration, comment out individual `\input` lines in that manifest or call one
  training-case file directly.
- `tables_complete_grid/`: one `.tex` file per complete-grid table.
- `mc_outcome_table_index.csv`: coverage index showing which phase/matchup
  cells were missing and where present-table files were written.

Generated table captions define the displayed statistics: outcome cells report
count; percentage `\pm` standard error, delta-v rows report median `[Q1, Q3]`,
and step time reports median `[Q1, Q3]`.

Each outcome cell is kept on one compact line:

```text
count; $percent \pm se\%$
```

The `\pm` value is the standard error of the binary outcome proportion,
reported in percentage points and derived from the outcome count and total
trial count.

Outcome rows are grouped into defender wins, attacker wins, and ties. Ties
currently contain only timeout / no capture.

Each result table also includes a delta-v statistics mini-section before the
bottom summary rows. The defender and attacker rows report median
trajectory-total delta-v with interquartile range:

```text
median [Q1, Q3] m/s
```

The bottom rows report total trial count and rollout total time per simulated
step in milliseconds as median with interquartile range. Combined four-phase
tables average the per-phase quartiles:

```text
median [Q1, Q3] ms
```

Tables use `\scriptsize` text with tightened row spacing, centered in
`\makebox[\textwidth][c]{...}` and wrapped in
`\resizebox{\dimexpr\textwidth+0.5in\relax}{!}{...}`. This lets each table
borrow roughly 0.25 inches from the left and right margins so the scaled text
is larger while still staying centered. Column padding is tightened with
`\setlength{\tabcolsep}{1pt}` to recover more of the larger text size without
increasing the margin overhang.

The stacked thesis result blocks use a slightly wider resize target
(`\textwidth+0.75in`) with tightened row spacing so the full-width KF-off and
KF-on tables stay compact.

The generated LaTeX assumes `booktabs`, `graphicx`, `float`, and `needspace`:

```latex
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage{needspace}
```

To include failed runs too:

```bash
python "LaTeX_Tables/generate_mc_latex_tables.py" --include-failed-runs
```
