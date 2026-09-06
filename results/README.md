# Simulation Results (Tracked)

This folder stores **full runs** produced by `main.py`. Run directories are
**version-controlled and pushed**: the raw run files (`simulation.log`,
`summary_v2.json` / `.csv`, `stage_results_v2.json` / `.csv`, `checkpoint.json`)
and the plots (`*.png`) go to GitHub with the code, so every reported number can
be traced to the run that produced it without asking for a local copy.

Bulk binaries a run can regenerate (`*.h5`, `*.npz`, `*.pkl`, `*.parquet`,
`*.zip`, `*.mat`) stay local — see `.gitignore`. Trained model artifacts belong
in `models/`, training datasets in `data/`.

Curated summaries, plots, and tables cited in progress reports or the thesis
still live under `reports/`; that promotion is now about curation, not about
getting artifacts into git.

## Run directory layout

Each run creates a timestamped folder:

```text
results/YYYYMMDD_HHMMSS_<methods>_<channel>_<modulation>_<factory-size>/
├── simulation.log
├── summary_v2.json
├── summary_v2.csv
├── estimators/              # when the estimator stage ran
└── resource_managers/       # when the resource-manager stage ran
```

Channel-specific runs may nest stage folders, for example
`rayleigh/resource_managers/` or `tr38901/estimators/`.

## Stage outputs

Each stage directory contains:

- `stage_results_v2.json` / `stage_results_v2.csv`
- `ber_vs_ebno.png`, `ber_raw_vs_ebno.png`
- `latency_vs_ebno.png`, `throughput_vs_ebno.png`, `power_vs_ebno.png`
- `runtime_by_method.png`
- `checkpoint.json` when a long run supports resume

## Key metrics

- `ber`: `bit_errors / total_bits`
- `ber_upper_confidence`: conservative upper bound when few errors are observed
- `point_status`: `resolved` vs `upper_bound_only`
- `throughput_bits_per_batch`: successfully delivered bits per batch
- `latency_ms`: average per-batch latency (not simulation wall time)

## Running and resuming

```bash
docker compose run --rm simulation --config config.json
```

Resume an interrupted resource-manager run (in-container path):

```bash
docker compose run --rm simulation --config config.json \
  --resource-managers static,round_robin,max_throughput,pf,wmmse,queue_aware,drl \
  --channel tr38901 --modulation low --factory-size s \
  --resume /app/results/<run_dir>
```

## Committing a run

A finished run is committed as-is:

```bash
git add results/<run_dir>
git commit -m "results: <what the run shows>"
git push -u origin <branch>
```

Keep commits to one run (or one comparison family) each, and say in the message
what the run is evidence for. Two things to check before committing:

- **Size.** `du -sh results/<run_dir>` — a run of logs, JSON/CSV and PNGs is
  normally well under a few MB. GitHub warns above 50 MB per file and rejects
  above 100 MB; if a run is that large, commit the summaries and plots and leave
  the bulk artifact local.
- **Completeness.** Partial or superseded runs are noise in history. Delete them
  (`scripts/tools/cleanup_results_from_manifest.py`) rather than committing them.

## Promotion

Before deleting old runs, copy any cited artifacts into `reports/weekly/<date>/`
or `reports/evidence/` if they are referenced outside this folder. Removing a
run directory from `results/` is a normal commit now, so the deletion is
recorded rather than silently local.
