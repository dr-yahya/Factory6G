# Results Documentation

This folder stores simulation outputs produced by `python main.py`.

## Folder Structure

Each run is saved in a timestamped subfolder:

- `YYYYMMDD_HHMMSS_estimators`
- `YYYYMMDD_HHMMSS_resource_managers`

In this repository, the current runs are:

- `20260217_150702_estimators` (February 17, 2026): channel estimator comparison
- `20260217_150833_resource_managers` (February 17, 2026): resource manager comparison

## Files In Each Run Folder

- `simulation_results_*.json`
  - Full raw output: config used, per-method metrics, runtime, and metadata.
- `simulation_results_*.csv`
  - Flattened table version of the same metrics for quick spreadsheet analysis.
- `*_ber_comparison.png`
  - BER vs Eb/No curve for each method (log scale).
- `*_ber_confidence_bound.png`
  - One-sided BER upper confidence bound vs Eb/No (log scale).
- `*_latency_comparison.png`
  - Average latency vs Eb/No in milliseconds.
- `*_throughput_comparison.png`
  - Average successful bits per batch vs Eb/No.
- `*_ber_latency_tradeoff.png`
  - Scatter plot of latency (x) vs BER (y, log scale). Lower-left is better.
- `*_runtime_bar.png`
  - Total wall-clock runtime per method for the full run.

## Metric Definitions

For each method and each Eb/No point:

- `bit_errors`: total bit mismatches across simulated batches
- `total_bits`: total transmitted bits across simulated batches
- `ber`: `bit_errors / total_bits`
- `ber_upper_confidence`: one-sided upper confidence bound on BER
  - Wilson-style bound when errors > 0
  - Zero-error bound `-ln(alpha)/N` when errors = 0
- `throughput`: average successful bits per batch
  - Computed as `(total_bits - bit_errors)` accumulated, then averaged by number of batches
- `latency`: average per-batch latency in seconds
  - Plots convert this to milliseconds
- `method_runtime_sec` (JSON top-level): total runtime in seconds spent by each method over the whole run

## How To Interpret The Plots

- BER plots: lower is better.
- BER confidence bound: lower is better; this is a conservative reliability view.
- Throughput: higher is better.
- Latency: lower is better.
- BER-latency tradeoff: better methods appear toward lower-left.
- Runtime bar: lower means faster simulation execution, not necessarily better link quality.

## Interpretation Of Current Runs

### 1) `20260217_150702_estimators`

- `perfect` is the ideal reference (best BER/throughput), as expected.
- Among practical estimators, `dft` is strongest overall in this run (low average BER, high throughput).
- `pso` is clearly worst here (highest BER and much higher runtime).
- All methods show identical average latency (`~0.1212 ms`), so this run is mainly BER/throughput driven.

### 2) `20260217_150833_resource_managers`

- `pf` has the best average BER and highest average throughput in this run.
- `cnn` is close to `pf` and outperforms `static`/`round_robin` on average.
- `round_robin` is weakest on average BER/throughput in this run.
- Runtime is fastest for `static`/`round_robin`; `cnn`, `pf`, and `max_throughput` take longer.

## Important Caveats

- These runs use small sample counts (`batch_size=1`, `total_batches=2`), so curves can be noisy/non-monotonic across Eb/No.
- `target_ber` is `null` in these runs, so no confidence-driven early stopping was used.
- `simulation_results_unknown_est.*` naming appears because estimator run naming uses `system_config.estimator_type`; when not set, it falls back to `unknown_est`.

## Re-running

To generate new results in this folder:

```bash
python main.py
```

Use `config.json` to control:

- run mode (`single` vs `both`)
- method lists (`estimators`, `resource_managers`)
- Eb/No sweep
- batch counts and confidence settings
- output directory

