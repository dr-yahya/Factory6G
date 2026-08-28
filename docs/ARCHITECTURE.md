# Factory6G Architecture And Flow

## Purpose

This document is the current source of truth for how the Factory6G codebase is organized and how a simulation run moves through the system.

It covers:

- repository structure
- runtime execution flow
- stage-level data flow
- core classes and interfaces
- configuration schema responsibilities
- output artifacts
- test coverage map
- documentation cleanup plan

## System Summary

Factory6G is a fixed-flow simulation harness for two benchmark stages:

1. channel estimator comparison
2. resource manager comparison

The runtime is built around a shared PHY model using Sionna/TensorFlow components for:

- OFDM resource-grid construction
- TR 38.901 or Rayleigh channel generation
- channel estimation
- equalization
- demapping
- LDPC decoding

The execution model is Monte Carlo over an Eb/No sweep.

## Repository Map

### Top Level

- `src/factory6g/cli/run.py`
  - CLI entrypoint
  - environment setup
  - config loading
  - run-directory creation
  - seed control
  - dispatch into the fixed simulation flow
- `config/config.json`
  - single runtime config file used by the CLI
- `docs/ARCHITECTURE.md`
  - this document
- `requirements.txt`
  - Python dependency list

### `src/factory6g/sim/`

Simulation orchestration and result emission.

- `config.py`
  - config schema
  - validation
  - normalized `Factory6GConfig`
- `env.py`
  - pre-import TensorFlow/Sionna environment setup
  - CPU/GPU visibility
  - matplotlib cache setup
- `run_context.py`
  - run-id generation
  - output directory naming
- `flow.py`
  - canonical fixed-order execution: `estimators -> resource_managers`
- `simulation.py`
  - backward-compatible shim for older callers
- `types.py`
  - shared runtime dataclasses such as `BatchContext`
- `output.py`
  - stage JSON/CSV writing
  - summary JSON/CSV writing
  - plot generation
- `results.py`
  - older result helper functions retained in the repo
- `stages/common.py`
  - shared Monte Carlo bookkeeping and stopping logic
- `stages/estimators.py`
  - estimator benchmark stage
- `stages/resource_managers.py`
  - resource-manager benchmark stage

### `src/factory6g/models/`

Runtime model and scheduling interfaces.

- `model.py`
  - end-to-end PHY wrapper
  - owns transmitter, channel, receiver, and estimator selection
  - exposes reusable `prepare_batch_context()` and `run_batch()`
- `resource_manager.py`
  - `ResourceDirectives`
  - resource manager base class
  - heuristic RM implementations
  - RM factory
- `cnn_resource_manager.py`
  - learned supervised RM wrapper
- `drl_policy.py`
  - DRL policy wrapper and inference support

### `src/factory6g/components/`

Signal-processing building blocks.

- `antenna.py`
  - BS/UT antenna-array construction
- `channel.py`
  - channel-model wrapper and noise sampling
- `transmitter.py`
  - bit source, LDPC encoding, QAM mapping, resource-grid mapping
- `receiver.py`
  - channel estimation, equalization, demapping, LDPC decoding
- `estimators/`
  - custom estimator implementations:
    - `dft_estimator.py`
    - `lmmse_estimator.py`
    - `adaptive_estimator.py`
    - `pso_estimator.py`

### Support Directories

- `scripts/`
  - dataset generation
  - model training
  - standalone inference
  - visualization helpers
- `data/`
  - generated datasets and their documentation
- `models/`
  - trained estimator / RM artifacts
- `results/`
  - simulation outputs
- `tests/`
  - unit and integration coverage
- `config/`
  - auxiliary config files such as factory size profiles

## Runtime Call Graph

The main runtime path is:

```text
python -m factory6g.cli.run --config config/config.json
  -> load_config()
  -> create_run_context()
  -> configure_env()
  -> set seeds
  -> run_simulation_flow()
      -> run_estimator_stage()
      -> write_stage_outputs("estimators")
      -> run_resource_manager_stage()
      -> write_stage_outputs("resource_managers")
      -> write_summary_outputs()
```

## Detailed Execution Flow

### 1. CLI Startup

`src/factory6g/cli/run.py` does the following:

1. parse `--config`
2. load and validate JSON into `Factory6GConfig`
3. create a timestamped run directory
4. open `simulation.log`
5. wrap stdout/stderr so noisy backend lines are filtered
6. configure TensorFlow/Sionna environment
7. set deterministic seeds for Python, NumPy, TensorFlow, and Sionna
8. call the fixed simulation flow

Important behavior:

- stage order is fixed
- the flow is not user-selectable from config
- config load errors fail fast before heavy runtime imports matter

### 2. Flow Orchestration

`src/factory6g/sim/flow.py` owns the canonical run sequence.

Responsibilities:

- resolve `run_id` and `run_dir`
- print run banner and Eb/No sweep
- execute the two stages in order
- persist stage outputs after each stage
- build a single summary payload across stages

### 3. Estimator Stage

`src/factory6g/sim/stages/estimators.py` benchmarks channel estimators only.

Current flow:

1. build one shared context-generation `Model`
2. build one per-method `Model`
3. for each Monte Carlo batch:
   - for each Eb/No point:
     - prepare one shared `BatchContext`
     - reuse that exact context across all estimator methods
4. accumulate per-method/per-point statistics
5. apply shared stopping policy
6. emit final metric arrays

Why shared batch contexts matter:

- estimator methods now see the same channel realization and noise draw
- method-to-method BER differences are less contaminated by Monte Carlo variance

### 4. Resource Manager Stage

`src/factory6g/sim/stages/resource_managers.py` benchmarks schedulers and power allocators.

Current flow:

1. build one shared `Model` using the adaptive estimator for feedback and decoding
2. build one resource-manager instance per method
3. for each Monte Carlo batch:
   - for each Eb/No point:
     - prepare one `BatchContext` with channel feedback enabled
     - let each RM consume the same feedback
     - convert RM output into `ResourceDirectives`
     - decode with the shared PHY model under those directives
4. accumulate per-method/per-point statistics
5. apply shared stopping policy
6. emit final metric arrays

Key difference from estimator stage:

- estimator stage varies the channel estimator while holding scheduling fixed
- resource-manager stage varies scheduling/power directives while holding the PHY model shared

## Core Runtime Data Structures

### `Factory6GConfig`

Normalized validated config object returned by `load_config()`.

Key sections:

- `simulation`
- `monte_carlo`
- `estimators`
- `resource_managers`
- `system`
- `transceiver`
- `factory_scenario`
- `ray_tracing`

### `BatchContext`

Carries the reusable physical realization for one batch at one Eb/No.

Fields:

- `batch_size`
- `ebno_db`
- `noise_variance`
- `h_freq`
- `probe_noise`
- `data_noise`
- `source_bits`
- `feedback`

This is the object that lets different methods share the same Monte Carlo realization.

### `ResourceManagerFeedback`

Feedback surface passed to RM methods when they require channel knowledge.

Fields:

- `h_hat`
- `err_var`

### `ResourceDirectives`

Runtime scheduling and power-control directives.

Fields:

- `active_ut_mask`
- `per_ut_power`
- `pilot_reuse_factor`

These directives are the RM stage’s control surface into the PHY model.

## PHY Model Composition

`src/factory6g/models/model.py` is the runtime composition root for the physical layer.

It owns:

- `AntennaConfig`
- `Transmitter`
- `ChannelModel`
- `Receiver`
- channel-estimator selection

### `prepare_batch_context()`

Creates a reusable physical batch by:

1. sampling topology if needed
2. sampling channel frequency response
3. converting Eb/No to noise variance
4. drawing probe and data noise
5. generating source bits
6. optionally generating channel feedback

### `run_batch()`

Consumes a `BatchContext` by:

1. building a transmit resource grid from source bits
2. applying optional RM directives
3. pushing the grid through the channel
4. estimating the channel unless perfect CSI is enabled
5. equalizing and decoding
6. returning decoded bits and diagnostics

## Signal Processing Pipeline

### Transmitter

`src/factory6g/components/transmitter.py`

Pipeline:

1. sample information bits
2. encode using 5G LDPC
3. map to QAM symbols
4. map onto the OFDM resource grid
5. optionally apply UT mask and per-UT power scaling

### Channel

`src/factory6g/components/channel.py`

Responsibilities:

- construct TR 38.901 `UMi/UMa/RMa` or Rayleigh channel
- generate OFDM frequency response
- apply channel to the transmit grid
- sample complex Gaussian noise

### Receiver

`src/factory6g/components/receiver.py`

Pipeline:

1. estimate the channel
2. run LMMSE equalization
3. demap to soft bits
4. clip extreme LLRs
5. LDPC decode

### Custom Estimators

`src/factory6g/components/estimators/`

- `dft_estimator.py`
  - delay-domain truncation style estimator
- `lmmse_estimator.py`
  - structured smoothing estimator
- `adaptive_estimator.py`
  - branch-selection hybrid
- `pso_estimator.py`
  - PSO search over structured DFT/LMMSE blend parameters

## Monte Carlo Policy

Shared Monte Carlo logic lives in `src/factory6g/sim/stages/common.py`.

Tracked metrics per point:

- `ber`
- `ber_upper_confidence`
- `latency_ms`
- `throughput_bits_per_batch`
- `energy_joules_per_batch`
- `avg_power_w`
- `runtime_sec`
- `bit_errors`
- `total_bits`
- `block_errors`
- `total_blocks`
- `num_batches`
- `stop_reason`
- `point_status`

### Stop Policy

Current supported policies:

- `sweep`
  - used for Eb/No sweeps
  - ignores `target_ber`
  - stops on evidence gates plus `target_block_errors`
- `threshold`
  - keeps BER-threshold stopping semantics

### Point Status

Current point classification:

- `resolved`
  - at least `30` observed bit errors
- `upper_bound_only`
  - fewer than `30` observed bit errors

This status is used by the plotter so publication plots do not pretend that zero-error points are exact BER measurements.

## Output Architecture

Current outputs are written by `src/factory6g/sim/output.py`.

### Per Stage

Each stage emits:

- `stage_results_v2.json`
- `stage_results_v2.csv`
- `ber_vs_ebno.png`
- `ber_raw_vs_ebno.png`
- `latency_vs_ebno.png`
- `throughput_vs_ebno.png`
- `power_vs_ebno.png`
- `runtime_by_method.png`

### Per Run

The run root emits:

- `summary_v2.json`
- `summary_v2.csv`
- `simulation.log`

### Plot Semantics

Main BER plot:

- resolved points use raw BER
- low-evidence points use `ber_upper_confidence`
- `pso` can be omitted from the headline plot when it behaves like a flat experimental outlier

Raw BER plot:

- uses stored raw BER values only
- zero BER points are omitted on the log axis instead of clipped to a fake floor

## Configuration Responsibilities

### `simulation`

- runtime logging
- output directory
- plotting on/off
- CPU/GPU choice
- random seed

### `monte_carlo`

- batch size
- minimum/maximum batches
- target block-error count
- optional BER target
- stop policy
- confidence level
- minimum total bits
- Eb/No sweep range

### `estimators`

- enabled estimator list
- per-estimator kwargs

### `resource_managers`

- enabled RM list
- model paths for learned RMs
- active-user count
- per-RM kwargs

### `system`

- OFDM numerology
- antenna counts
- coderate
- scenario
- channel model selection
- mobility controls

### `transceiver`

- antenna patterns
- polarization
- geometry parameters

### `factory_scenario`

- room geometry
- machine geometry
- material definitions

### `ray_tracing`

- ray depth
- source sampling density
- path caps

## Resource Manager Inventory

The RM layer contains:

- static
- round robin
- max throughput
- proportional fair
- WMMSE
- queue-aware Lyapunov policy
- learned CNN policy
- learned DRL policy

They all converge to the same interface:

```text
feedback + config + ebno_db -> ResourceDirectives
```

## Tests Map

The `tests/` directory currently covers:

- config parsing and schema validation
- CLI flow and file generation
- logging stream behavior
- transmitter directive application
- resource-manager method behavior
- RM fairness and masking logic
- adaptive estimator behavior
- batch-context reuse in estimator stage
- Monte Carlo stop-policy behavior
- BER plotting behavior
- dataset-generation and DRL-pipeline paths

This is a mixed unit/integration test suite. Some tests require a working Sionna runtime.

## Known Constraints And Current Truths

1. Stage order is fixed.
2. The runtime is organized around one shared `config/config.json`.
3. A single config file cannot currently express different Eb/No sweep ranges for different stages without new schema support.
4. The runtime depends on a correct Sionna/TensorFlow/Dr.Jit environment.
5. The current plotter is confidence-aware by design; it is intentionally not smoothing BER curves.
6. Some older documentation in the repo still describes legacy result formats and needs alignment with the current `stage_results_v2` / `summary_v2` flow.

## Code Flow Diagram

```text
Config JSON
  -> Factory6GConfig
      -> run_simulation_flow
          -> estimator stage
              -> shared Model.prepare_batch_context
              -> per-estimator Model.run_batch
              -> aggregate metrics
          -> stage output writer
          -> resource-manager stage
              -> shared Model.prepare_batch_context(include_feedback=True)
              -> RM.get_runtime_directives
              -> shared Model.run_batch(directives=...)
              -> aggregate metrics
          -> stage output writer
          -> summary writer
```

## Documentation Plan

### Immediate

1. Keep this file as the architecture source of truth.
2. Align `results/README.md` with the current `stage_results_v2` and `summary_v2` output layout.
3. Add a root `README.md` that links:
   - setup
   - `config/config.json`
   - `docs/ARCHITECTURE.md`
   - data/model/result docs

### Next

1. Add module-level architecture notes to:
   - `src/factory6g/models/model.py`
   - `src/factory6g/sim/flow.py`
   - `src/factory6g/sim/stages/estimators.py`
   - `src/factory6g/sim/stages/resource_managers.py`
2. Add a config reference table with defaults and constraints extracted from `config.py`.
3. Add an output-schema reference for:
   - `stage_results_v2.json`
   - `summary_v2.json`

### Follow-Up

1. Mark legacy docs explicitly as legacy or update them.
2. Add a developer troubleshooting section for:
   - Sionna/Dr.Jit runtime issues
   - CPU vs GPU behavior
   - plot interpretation for low-evidence BER tails
3. Add a short “how to extend” guide for:
   - new estimator methods
   - new resource managers
   - new output metrics
