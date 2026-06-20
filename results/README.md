# Simulation Results (Local Workspace)

This folder stores **full runs** produced by `main.py`. Under the lean-git
policy, run directories are **local only** — they are not version-controlled.

Curated summaries, plots, and tables cited in progress reports or the thesis
live under `reports/`.

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

## Promotion

Before deleting old runs, copy any cited artifacts into `reports/weekly/<date>/`
or `reports/evidence/` if they are referenced outside this folder.
