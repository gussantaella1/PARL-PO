# Monte Carlo LaTeX Tables

Regenerate the tables from the repository root:

```bash
python "LaTeX_Tables/generate_mc_latex_tables.py"
```

Outputs:

- `mc_outcome_tables_present.tex`: compact tables for policy/radius/dynamics/KF combinations with at least one result file.
- `mc_outcome_tables_complete_grid.tex`: all top-level `Training_Policy*` folders crossed with 20/50/100 m, HCW/elliptic LTV, and KF on/off; unavailable phase cells are `N/A`.
- `tables_present/`: one `.tex` file per present-data table.
- `tables_complete_grid/`: one `.tex` file per complete-grid table.
- `mc_outcome_table_index.csv`: coverage index showing which phase/matchup cells were missing.

Each outcome cell is kept on one compact line:

```text
count; percent% \raisebox{0.25ex}{\tiny [lo%, hi%]}
```

The bottom rows report total trial count and mean rollout total time per
simulated step in milliseconds, pooled across the four phase/matchup columns:

```text
mean ms [95% normal CI]
```

Tables are wrapped in `\resizebox{\textwidth}{!}{...}` and use
`\footnotesize` text so they fit inside the page width.

The generated LaTeX assumes `booktabs` and `graphicx`:

```latex
\usepackage{booktabs}
\usepackage{graphicx}
```

To include failed runs too:

```bash
python "LaTeX_Tables/generate_mc_latex_tables.py" --include-failed-runs
```
