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
| `pyproject.toml` | Packaging metadata. `pip install -e .` exposes the `factory6g` package plus the `factory6g-run` / `factory6g-train` / `factory6g-visualize` console entrypoints. |
| `src/factory6g/cli/run.py` | CLI entrypoint for simulation runs (`python -m factory6g.cli.run`). Loads `config/config.json`, creates a timestamped run directory, configures the environment, and dispatches the simulation flow. |
| `src/factory6g/cli/train.py` | NeuralChannelEstimator training entrypoint (`python -m factory6g.cli.train`). |
| `src/factory6g/cli/visualize.py` | Factory ray-tracing visualization entrypoint (`python -m factory6g.cli.visualize`). |
| `config/config.json` | Main simulation configuration: Monte Carlo policy, enabled estimators/resource managers, system numerology, factory geometry, and output settings. |
| `config/factory_size_profiles.json` | Factory size profile definitions (S/M/L) for dataset generation. |
| `docs/ARCHITECTURE.md` | Main architecture reference for runtime flow, stage data flow, interfaces, and output schema. Start there when reading the implementation deeply. |
| `docs/CONTEXT.md` | Repo-wide working context and repository zone policy. |
| `docs/THESIS_RESULTS.md` | Maps each thesis claim to the config and command that produces its evidence, the metrics that carry it, and the gaps that remain. Start here when generating results. |
| `docs/SIMULATION_REVIEW.md` | Standing review of the simulation code against the research objectives: correctness issues that affect reported results, AI/ML and factory-realism gaps, and a prioritized work order. |
| `docs/assets/system_design/` | Static system-design images. |
| `src/factory6g/sim/` | Simulation orchestration, config loading, run context creation, stage execution, checkpointing, output writing, and plotting. |
| `src/factory6g/models/` | PHY model composition plus resource-manager implementations, learned CNN/DRL policy wrappers, and scheduling directives. |
| `src/factory6g/components/` | Signal-processing building blocks: antenna arrays, channel model, transmitter, receiver, and custom channel estimators. |
| `src/factory6g/training/` | NeuralChannelEstimator training plus the reinforcement-learning resource-manager trainer (`rl_resource_manager.py`). |
| `src/factory6g/athirah/` | Python port of the MATLAB polar-coded SCMA-OFDM (JIDD-SCMA) reference. |
| `scripts/` | Dataset generation, model training, reporting, visualization, and maintenance utilities. Use Docker with a repo bind mount for these scripts. |
| `data/` | Generated training datasets and dataset documentation. See `data/README.md`. |
| `models/` | Trained channel-estimator and resource-manager artifacts. |
| `results/` | Full simulation runs, tracked and pushed (raw run files and plots). See `results/README.md`. |
| `reports/` | Progress reports and curated evidence summaries. |
| `reports/evidence/` | Cross-cutting promoted plots, tables, and stage summaries. |
| `tests/` | Unit and integration tests for config, CLI flow, estimators, resource managers, plotting, datasets, and DRL policy loading. |
| `reference/dr_athirah_simulation/` | Reference MATLAB-origin PHY/MAC/APP layer material and JIDD-SCMA assets. |
| `thesis/` | LaTeX thesis sources, figures, and notes (gitignored; local only). |
| `archive/` | Superseded drafting workflows kept for reference (gitignored). |

## Docker Setup

The code is a proper Python package (`factory6g`, see `pyproject.toml`). The
Docker image runs `pip install -e .` during build, so image-based runs work out
of the box. For bind-mounted runs against a live checkout, run `pip install -e .`
inside the container first (shown in the examples below).

Build the CPU simulation image:

```bash
docker compose build simulation
```

Run CPU simulations with the `simulation` service:

```bash
docker compose run --rm simulation --config config/config.json
```

Run GPU simulations with the `simulation-gpu` service when the host has NVIDIA
Docker support and `config/config.json` is set for GPU execution:

```bash
docker compose run --rm simulation-gpu --config config/config.json
```

Use `--build` when source code, config defaults, model files, or dependencies
may have changed and you want Docker Compose to rebuild before the run:

```bash
docker compose run --rm --build simulation --config config/config.json
```

Use a repo bind mount when running tests or scripts that are not copied into the
image, or when you need the container to see the exact checkout without a
rebuild:

```bash
docker compose run --rm --entrypoint python -v "$PWD:/app" simulation -m factory6g.cli.run --help
```

## CLI Behavior

The CLI is implemented in `src/factory6g/cli/run.py` (run it with
`python -m factory6g.cli.run` or the installed `factory6g-run` console script;
the `simulation` Docker service uses it as its entrypoint).

- `--estimators ls,dft,adaptive` runs only the estimator stage unless
  `--resource-managers` is also passed.
- `--resource-managers static,pf,drl` runs only the resource-manager stage
  unless `--estimators` is also passed.
- With no method override flags, the run uses estimators from `config/config.json` and
  defaults the resource-manager stage to `max_throughput`.
- `--channel` accepts `rayleigh`, `rician`, `tr38901`, `inf`, or `awgn`.
  `inf` selects the TR 38.901 Indoor Factory model, where the hall geometry in
  `factory_scenario` drives LOS probability and path loss.
- `--modulation` accepts `low` for QPSK, `mid` for 16-QAM, and `high` for
  64-QAM.
- `--factory-size` accepts `s`, `m`, `l`, and `apple`. Each preset sets both the
  user count and the hall geometry; with `--channel inf` the geometry genuinely
  changes propagation, with the other channel models only the user count matters.
- `--stage jidd_scma` runs the JIDD-SCMA flow. For standard estimator or
  resource-manager experiments, prefer selecting methods with `--estimators`
  and/or `--resource-managers`.

## Reproduce Current Result Families

Full reproduction runs can take hours. Do not use these as quick validation
checks unless you intentionally want to regenerate research results.

### Baseline Resource Managers - Rayleigh

```bash
docker compose run --rm --build simulation --config config/config.json --resource-managers static,round_robin,max_throughput,pf,wmmse,queue_aware,drl --channel rayleigh --modulation low --factory-size s
```

### Baseline Resource Managers - TR 38.901 UMi

```bash
docker compose run --rm --build simulation --config config/config.json --resource-managers static,round_robin,max_throughput,pf,wmmse,queue_aware,drl --channel tr38901 --modulation low --factory-size s
```

### BER-First Learned Resource Manager - TR 38.901 UMi

```bash
docker compose run --rm --build simulation --config config/config.json --resource-managers reliability_drl --channel tr38901 --modulation low --factory-size s
```

### BER-First Learned Resource Manager - Rayleigh

```bash
docker compose run --rm --build simulation --config config/config.json --resource-managers reliability_drl --channel rayleigh --modulation low --factory-size s
```

### Resume An Interrupted Resource-Manager Run

Use the channel that matches the original run directory. For UMi runs, pass
`--channel tr38901`; the scenario label in output names is usually `umi`.

```bash
docker compose run -d --rm simulation --config config/config.json --resource-managers static,round_robin,max_throughput,pf,wmmse,queue_aware,drl --channel <rayleigh|tr38901> --modulation low --factory-size s --resume /app/results/<run_dir>
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
- `bler_vs_ebno.png`
- `nmse_vs_ebno.png`
- `latency_p999_vs_ebno.png`
- `ber_vs_ebno.png`
- `ber_raw_vs_ebno.png`
- `latency_vs_ebno.png`
- `throughput_vs_ebno.png`
- `power_vs_ebno.png`
- `runtime_by_method.png`

Key metrics:

- `bler`: block error rate, the headline reliability metric. Codewords are close
  to independent, so this is the quantity whose confidence bound
  (`bler_upper_confidence`, exact Clopper-Pearson) is statistically defensible.
  Prefer it over `ber` for reliability claims: bits within a codeword are
  strongly correlated, so a bit-level interval is far too narrow.
- `worst_user_bler`: the worst-served user. In factory URLLC the weakest device
  sets system reliability, and an aggregate hides it.
- `latency_ms`, `latency_p99_ms`, `latency_p999_ms`: link latency as slot time
  times the HARQ rounds actually used. Tail percentiles are the URLLC-relevant
  figures; the mean is not. This is not host runtime — that is `runtime_sec`.
- `nmse_db`: channel-estimate accuracy, `E|h - h_hat|^2 / E|h|^2` in dB. The
  primary metric for the estimator benchmark, since it isolates estimator
  quality from the equalizer and decoder.
- `jains_index`, `num_scheduled_users`, `radiated_power_w`: fairness, served
  load and transmit power. `num_scheduled_users` matters when comparing methods:
  `static` runs at full load while the schedulers run at `num_active_users`, so
  compare against `static_subset` for an equal-load control.
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

For publication-style reliability comparisons, use `bler_vs_ebno.png` and read
`bler_upper_confidence` alongside it.

### Statistical comparison between methods

Every method sees the identical channel realisation, noise draw and source bits
at a given (batch, Eb/No) — common random numbers. `stage_results_v2.json`
therefore carries `paired_comparisons`: per-batch BLER differences against
`resource_managers.paired_reference` with a paired bootstrap confidence interval
and a `significant` flag. These intervals are far tighter than comparing the two
marginal curves, and they let a result be written as "improves BLER by X
(95% CI [a, b], n = N batches)" rather than "the curve is lower".

### Provenance

Each stage payload records `run_provenance` (git commit, branch, dirty flag and
library versions) and, for the resource-manager stage, `manager_provenance` —
whether each learned policy actually loaded, its resolved path and a checkpoint
SHA-256. Policy loading is strict by default: a checkpoint that fails to load
raises rather than silently substituting the heuristic actor under a learned
method's name. Set `resource_managers.strict_policy_loading: false` to opt into
the fallback deliberately.

### Evidence ceiling

Every run prints what its Monte Carlo budget can actually resolve, for example:

```text
Evidence ceiling per Eb/No point: 2,457,600 bits / 1,600 codewords.
Smallest resolvable BER ~1.22e-05, BLER ~1.87e-02.
```

Reliability below that ceiling cannot be measured, only bounded. Use
`factory6g.sim.evidence.extrapolate_bler` to project the deep tail with a
prediction interval, and report it as an extrapolation.

## Training The Learned Resource Managers

Two trainers, and the difference matters for how results are described.

`scripts/tools/train_drl_resource_manager.py` is **supervised imitation** of an
oracle's scheduling labels. Its checkpoints record
`checkpoint_type: "offline_behaviour_cloning"`. It is bounded above by the
candidate search that produced its labels — it cannot beat the heuristics it
learned from.

`scripts/tools/train_rl_resource_manager.py` is **reinforcement learning**:
REINFORCE with a learned value baseline, acting in the simulator and rewarded by
what the physical layer actually delivered. The problem is formulated as a
contextual bandit (channel realisations are drawn independently, so only the
fairness debt carries across slots).

```bash
docker compose run --rm --entrypoint bash -v "$PWD:/app" simulation -lc \
  "pip install -e . -q && python scripts/tools/train_rl_resource_manager.py \
     --config config/config.json --iterations 300 \
     --initial-checkpoint models/drl_resource_manager_policy \
     --output-dir models/rl_resource_manager_policy"
```

Warm-starting RL from a behaviour-cloning checkpoint usually beats either alone.
The trainer evaluates a max-SNR baseline through the identical path before and
after training, and stores both in the checkpoint metadata, so a claimed gain is
always reported against a measured baseline.

Reward weights (`--reward-reliability`, `--reward-throughput`,
`--reward-energy`, `--reward-fairness`) scalarise the objective; each term is
also reported separately so the trade-off can be shown rather than only the
scalar.

## Dataset, Training, And Reports

Dataset and training workflows are documented in `data/README.md`. Keep those
commands Dockerized. Use a repo bind mount for script-based workflows because
the `scripts/` tree is intended to be executed from the checkout; run an
editable install first so the bind-mounted `factory6g` package resolves:

```bash
docker compose run --rm --entrypoint bash -v "$PWD:/app" simulation -lc "pip install -e . -q && python scripts/tools/generate_rm_ber_report.py"
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

Validate the CLI syntax inside Docker (the bind-mounted checkout needs an
editable install so the `factory6g` package resolves):

```bash
docker compose run --rm --entrypoint bash -v "$PWD:/app" simulation -lc "pip install -e . -q && python -m factory6g.cli.run --help"
```

Run a lightweight test subset inside Docker:

```bash
docker compose run --rm --entrypoint bash -v "$PWD:/app" simulation -lc "pip install -e . -q && python -m pytest tests/test_config_loader.py tests/test_main_cli.py -q"
```

Run the DRL policy loader regression inside Docker:

```bash
docker compose run --rm --entrypoint bash -v "$PWD:/app" simulation -lc "pip install -e . -q && python -m pytest tests/test_drl_policy_pipeline.py -q"
```

Do not run full reproduction simulations as routine validation. They are meant
for regenerating research artifacts and may take several hours depending on the
channel model, enabled methods, hardware, and Monte Carlo stopping behavior.

## Reading Path For New Contributors

1. Read `docs/ARCHITECTURE.md` for the fixed simulation flow and data contracts.
2. Inspect `src/factory6g/cli/run.py` to understand CLI overrides and
   run-directory creation.
3. Follow `src/factory6g/sim/flow.py` into `src/factory6g/sim/stages/estimators.py`
   and `src/factory6g/sim/stages/resource_managers.py`.
4. Read `src/factory6g/models/model.py` for the PHY model composition.
5. Read `src/factory6g/models/resource_manager.py` for the scheduling and
   power-control interface used by all resource managers.
6. Use `data/README.md` only when you need dataset generation or model training
   workflows.
