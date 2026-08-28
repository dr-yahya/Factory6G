# Factory6G

Factory6G is a 6G and Beyond-5G smart-factory simulation project focused on
physical-layer reliability and AI/ML-assisted resource management. The current
runtime uses Sionna/TensorFlow components to run Monte Carlo Eb/No sweeps for
two main benchmark families:

- channel estimator comparison
- resource manager comparison

The project is Docker-first. Run scripts, simulations, training utilities, and
tests inside the Docker container so the TensorFlow, Sionna, Dr.Jit, and
plotting environment stays consistent.

## Project Map

| Path | Purpose |
|---|---|
| `main.py` | CLI entrypoint for simulation runs. It loads `config.json`, creates a timestamped run directory, configures the environment, and dispatches the simulation flow. |
| `config.json` | Main simulation configuration: Monte Carlo policy, enabled estimators/resource managers, system numerology, factory geometry, and output settings. |
| `ARCHITECTURE.md` | Main architecture reference for runtime flow, stage data flow, interfaces, and output schema. Start there when reading the implementation deeply. |
| `src/sim/` | Simulation orchestration, config loading, run context creation, stage execution, checkpointing, output writing, and plotting. |
| `src/models/` | PHY model composition plus resource-manager implementations, learned CNN/DRL policy wrappers, and scheduling directives. |
| `src/components/` | Signal-processing building blocks: antenna arrays, channel model, transmitter, receiver, and custom channel estimators. |
| `scripts/` | Dataset generation, model training, reporting, visualization, and maintenance utilities. Use Docker with a repo bind mount for these scripts. |
| `config/` | Auxiliary configuration files, including factory size profiles. |
| `data/` | Generated training datasets and dataset documentation. See `data/README.md`. |
| `models/` | Trained channel-estimator and resource-manager artifacts. |
| `results/` | Local full simulation runs (gitignored). See `results/README.md`. |
| `reports/` | Progress reports and curated evidence summaries. |
| `reports/evidence/` | Cross-cutting promoted plots, tables, and stage summaries. |
| `tests/` | Unit and integration tests for config, CLI flow, estimators, resource managers, plotting, datasets, and DRL policy loading. |
| `system_design/` | Static system-design images. |
| `dr_athirah_simulation/` | Reference MATLAB-origin PHY/MAC/APP layer material and JIDD-SCMA assets. |

## Docker Setup

Build the CPU simulation image:

```bash
docker compose build simulation
```

Run CPU simulations with the `simulation` service:

```bash
docker compose run --rm simulation --config config.json
```

Run GPU simulations with the `simulation-gpu` service when the host has NVIDIA
Docker support and `config.json` is set for GPU execution:

```bash
docker compose run --rm simulation-gpu --config config.json
```

Use `--build` when source code, config defaults, model files, or dependencies
may have changed and you want Docker Compose to rebuild before the run:

```bash
docker compose run --rm --build simulation --config config.json
```

Use a repo bind mount when running tests or scripts that are not copied into the
image, or when you need the container to see the exact checkout without a
rebuild:

```bash
docker compose run --rm --entrypoint python -v "$PWD:/app" simulation main.py --help
```

## CLI Behavior

The CLI is implemented in `main.py`.

- `--estimators ls,dft,adaptive` runs only the estimator stage unless
  `--resource-managers` is also passed.
- `--resource-managers static,pf,drl` runs only the resource-manager stage
  unless `--estimators` is also passed.
- With no method override flags, the run uses estimators from `config.json` and
  defaults the resource-manager stage to `max_throughput`.
- `--channel` accepts `rayleigh`, `rician`, `tr38901`, or `awgn`.
- `--modulation` accepts `low` for QPSK, `mid` for 16-QAM, and `high` for
  64-QAM.
- `--factory-size` accepts `s`, `m`, `l`, and `apple`.
- `--stage jidd_scma` runs the JIDD-SCMA flow. For standard estimator or
  resource-manager experiments, prefer selecting methods with `--estimators`
  and/or `--resource-managers`.

## Reproduce Current Result Families

Full reproduction runs can take hours. Do not use these as quick validation
checks unless you intentionally want to regenerate research results.

### Baseline Resource Managers - Rayleigh

```bash
docker compose run --rm --build simulation --config config.json --resource-managers static,round_robin,max_throughput,pf,wmmse,queue_aware,drl --channel rayleigh --modulation low --factory-size s
```

### Baseline Resource Managers - TR 38.901 UMi

```bash
docker compose run --rm --build simulation --config config.json --resource-managers static,round_robin,max_throughput,pf,wmmse,queue_aware,drl --channel tr38901 --modulation low --factory-size s
```

### BER-First Learned Resource Manager - TR 38.901 UMi

```bash
docker compose run --rm --build simulation --config config.json --resource-managers reliability_drl --channel tr38901 --modulation low --factory-size s
```

### BER-First Learned Resource Manager - Rayleigh

```bash
docker compose run --rm --build simulation --config config.json --resource-managers reliability_drl --channel rayleigh --modulation low --factory-size s
```

### Resume An Interrupted Resource-Manager Run

Use the channel that matches the original run directory. For UMi runs, pass
`--channel tr38901`; the scenario label in output names is usually `umi`.

```bash
docker compose run -d --rm simulation --config config.json --resource-managers static,round_robin,max_throughput,pf,wmmse,queue_aware,drl --channel <rayleigh|tr38901> --modulation low --factory-size s --resume /app/results/<run_dir>
```

The resume path must be the in-container path under `/app/results/`. A resumed
resource-manager stage should log a checkpoint message such as:

```text
[checkpoint] Resuming resource_managers from batch ...
```

## Outputs And Result Interpretation

Every run creates a timestamped directory under `results/`.

The run root contains:

- `simulation.log`: console output and runtime progress.
- `summary_v2.json`: structured summary of stage paths and run metadata.
- `summary_v2.csv`: tabular summary for quick inspection.

Each stage directory, such as `estimators/` or `resource_managers/`, contains:

- `stage_results_v2.json`
- `stage_results_v2.csv`
- `ber_vs_ebno.png`
- `ber_raw_vs_ebno.png`
- `latency_vs_ebno.png`
- `throughput_vs_ebno.png`
- `power_vs_ebno.png`
- `runtime_by_method.png`

Key metrics:

- `ber`: measured bit error rate, computed as `bit_errors / total_bits`.
- `ber_upper_confidence`: conservative upper confidence bound for BER. This is
  important when few or zero bit errors are observed.
- `point_status`: evidence classification for each Eb/No point. `resolved`
  points have enough observed errors; `upper_bound_only` points should be read
  through the confidence bound.
- `throughput_bits_per_batch`: successfully delivered bits per simulated batch.
- `latency_ms`: average per-batch latency in milliseconds.
- `runtime_sec` and `runtime_totals_sec`: simulation runtime, not link-layer
  latency.

For publication-style reliability comparisons, prefer the confidence-aware BER
plot and inspect `ber_upper_confidence` alongside raw `ber`.

## Dataset, Training, And Reports

Dataset and training workflows are documented in `data/README.md`. Keep those
commands Dockerized. Use a repo bind mount for script-based workflows because
the `scripts/` tree is intended to be executed from the checkout:

```bash
docker compose run --rm --entrypoint python -v "$PWD:/app" simulation scripts/tools/generate_rm_ber_report.py
```

Generated summaries and research-facing tables live under `reports/`. Some
older result documentation may mention legacy result filenames; the current
runtime writes `stage_results_v2.*` and `summary_v2.*`.

## Progress Report Log

Each dated weekly folder includes the stakeholder message draft and, when
available, a concise evidence deck. The table links one primary readable
progress report per date. These weekly packages are the evidence trail for
remote/Friday progress updates used in the Semester 5 Meeting Log (alongside
Microsoft Teams supervisory meetings on 10 Feb, 26 Feb, 16 Mar, and 2 Apr 2026).

| Date | Progress report | Focus |
|---|---|---|
| 2026-01-15 | [Factory6G weekly messages](reports/weekly/2026-01-15/factory6g_weekly_messages.md) | Sem-5 plan and Chapter 2 literature-review structure; transition toward learned estimators and AI resource management. |
| 2026-03-27 | [Factory6G weekly messages](reports/weekly/2026-03-27/factory6g_weekly_messages.md) | March PHY benchmark: Adaptive/PSO estimator evidence, TR 38.901 BER floor, modulation/factory-size impact, and corrected JIDD-SCMA result. |
| 2026-04-20 | [Factory6G weekly messages](reports/weekly/2026-04-20/factory6g_weekly_messages.md) | April neural-estimator update: retrained Neural vs LS Rayleigh evidence, corrected DFT comparison, TR 38.901 estimator tier, and remaining large-factory/JIDD findings. |
| 2026-04-27 | [Factory6G weekly messages](reports/weekly/2026-04-27/factory6g_weekly_messages.md) | AI resource-manager recovery, Docker workflow validation, and Rayleigh QPSK benchmark evidence. |
| 2026-05-01 | [Factory6G weekly messages](reports/weekly/2026-05-01/factory6g_weekly_messages.md) | BER-first DRL resource-manager workflow, Rayleigh/UMi benchmark runs, and Docker validation evidence. |
| 2026-05-15 | [Factory6G weekly messages](reports/weekly/2026-05-15/factory6g_weekly_messages.md) | Special Absence start (11 May 2026); remote Friday weekly cadence and absence milestone plan. |
| 2026-05-23 | [Factory6G weekly messages](reports/weekly/2026-05-23/factory6g_weekly_messages.md) | Cross-channel resource-manager BER screening across Rayleigh, Rician, and UMI/TR38901, with [evidence deck](reports/weekly/2026-05-23/factory6g_weekly_evidence.pptx), [BER comparison](reports/weekly/2026-05-23/resource_manager_ber_comparison.md), and [synthetic channel curves](reports/weekly/2026-05-23/resource_manager_channel_comparison_synthetic.md); `ber_drl` matched the Rayleigh best baseline and ranked second on Rician and UMI/TR38901 in the shortened run. |
| 2026-05-29 | [Factory6G weekly messages](reports/weekly/2026-05-29/factory6g_weekly_messages.md) | Remote thesis drafting (Chapters 3–6) and Chapter 4 evidence-family organization under Special Absence. |
| 2026-06-05 | [Factory6G weekly messages](reports/weekly/2026-06-05/factory6g_weekly_messages.md) | Softbound drafting status; remaining large-factory BER and longer UMi RM/BER-DRL validation gaps. |
| 2026-06-12 | [Factory6G weekly messages](reports/weekly/2026-06-12/factory6g_weekly_messages.md) | Softbound thesis package progress; Chapters 1–5 substantially in place. |
| 2026-06-19 | [Factory6G weekly messages](reports/weekly/2026-06-19/factory6g_weekly_messages.md) | Curated evidence promotion and lean-git zones for tracked thesis claims. |
| 2026-06-26 | [Factory6G weekly messages](reports/weekly/2026-06-26/factory6g_weekly_messages.md) | Sem-5 softbound package ready for supervisor review and signature. |

## Quick Validation

Validate the CLI syntax inside Docker:

```bash
docker compose run --rm --entrypoint python -v "$PWD:/app" simulation main.py --help
```

Run a lightweight test subset inside Docker:

```bash
docker compose run --rm --entrypoint python -v "$PWD:/app" simulation -m pytest tests/test_config_loader.py tests/test_main_cli.py -q
```

Run the DRL policy loader regression inside Docker:

```bash
docker compose run --rm --entrypoint python -v "$PWD:/app" simulation -m pytest tests/test_drl_policy_pipeline.py -q
```

Do not run full reproduction simulations as routine validation. They are meant
for regenerating research artifacts and may take several hours depending on the
channel model, enabled methods, hardware, and Monte Carlo stopping behavior.

## Reading Path For New Contributors

1. Read `ARCHITECTURE.md` for the fixed simulation flow and data contracts.
2. Inspect `main.py` to understand CLI overrides and run-directory creation.
3. Follow `src/sim/flow.py` into `src/sim/stages/estimators.py` and
   `src/sim/stages/resource_managers.py`.
4. Read `src/models/model.py` for the PHY model composition.
5. Read `src/models/resource_manager.py` for the scheduling and power-control
   interface used by all resource managers.
6. Use `data/README.md` only when you need dataset generation or model training
   workflows.
