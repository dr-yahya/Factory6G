# Results & Analysis

This directory contains documentation of simulation results and analysis.

## Files

1. **[01_channel_estimation_comparison.md](01_channel_estimation_comparison.md)** - Channel estimation method comparison
   - Comparison of LS, LS_LIN, LS_NN, and Neural estimators
   - Performance metrics (BER, BLER)
   - Numerical summaries and plots

2. **[02_failure_analysis.md](02_failure_analysis.md)** - Simulation failure analysis
   - Common failure modes
   - Debugging strategies
   - Solutions and workarounds

## Related

- Simulation results are stored in `results/` directory
- To run baseline vs AI-estimator scenarios, use:
  - Baseline: `python main.py`
  - AI estimator: `python main.py --scenario-profile 6g_ai_estimator --neural-weights artifacts/neural_channel_estimator.weights.h5`
- Baseline results may be generated via `scripts/create_baseline_results.py`

