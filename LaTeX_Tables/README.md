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
- `tables_complete_grid/`: one `.tex` file per complete-grid table.
- `mc_outcome_table_index.csv`: coverage index showing which phase/matchup
  cells were missing and where present-table files were written.

Each outcome cell is kept on one compact line:

```text
count; percent% \raisebox{0.25ex}{\tiny [lo%, hi%]}
```

Outcome rows are grouped into defender wins, attacker wins, and ties. Ties
currently contain only timeout / no capture.

The bottom rows report total trial count and mean rollout total time per
simulated step in milliseconds, pooled across the four phase/matchup columns:

```text
mean ms [95% normal CI]
```

Tables are centered in `\makebox[\textwidth][c]{...}` and wrapped in
`\resizebox{\dimexpr\textwidth+0.5in\relax}{!}{...}`. This lets each table
borrow roughly 0.25 inches from the left and right margins so the scaled text
is larger while still staying centered. Column padding is tightened with
`\setlength{\tabcolsep}{1pt}` to recover more of the larger text size without
increasing the margin overhang.

The generated LaTeX assumes `booktabs`, `graphicx`, and `float`:

```latex
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
```

To include failed runs too:

```bash
python "LaTeX_Tables/generate_mc_latex_tables.py" --include-failed-runs
```
