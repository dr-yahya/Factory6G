# Script Outputs (Tracked)

This folder holds **regenerable artifacts** produced by visualization and
plotting scripts. It is version-controlled: plots and the raw files behind them
are pushed with the code, so a figure shown in a discussion can be linked
instead of attached. Bulk binaries (`*.h5`, `*.pkl`, `*.parquet`, `*.zip`,
`*.mp4`, `*.npz`) stay local — see `.gitignore`.

Use this zone for ad-hoc PNGs and diagrams that are not timestamped simulation
runs and not yet promoted into `reports/`. Because it is tracked, prune stale
figures instead of letting them accumulate; regenerating one is a script call.

## Typical contents

- `visualizations/` — from `visualize.py`
- `system_design/` — from `scripts/tools/visualize_system_design.py`
- `comparison_jidd_vs_ls_rayleigh.png` — from `scripts/plot_comparison.py`

Promote figures cited in progress reports or the thesis into
`reports/weekly/<date>/assets/` or `reports/evidence/<topic>/`.
