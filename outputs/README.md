# Local Script Outputs

This folder holds **regenerable artifacts** produced by visualization and
plotting scripts. It is gitignored under the lean-git policy.

Use this zone for ad-hoc PNGs and diagrams that are not timestamped simulation
runs and not yet promoted into `reports/`.

## Typical contents

- `visualizations/` — from `visualize.py`
- `system_design/` — from `scripts/tools/visualize_system_design.py`
- `comparison_jidd_vs_ls_rayleigh.png` — from `scripts/plot_comparison.py`

Promote figures cited in progress reports or the thesis into
`reports/weekly/<date>/assets/` or `reports/evidence/<topic>/`.
