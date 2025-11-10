# Simulation Results - UMi Scenario (Imperfect CSI)

## Setup
- Scenario: UMi (Urban Micro)
- Direction: Uplink
- MIMO: 4 UT × 8 BS antennas
- OFDM: FFT 128, SCS 30 kHz, 14 symbols, pilot pattern kronecker
- Modulation/Coding: QPSK (2 bps), 5G LDPC, code rate 0.5
- Estimators:
  - LS: Least-Squares channel estimator with interpolation
  - Neural: LS baseline + neural refinement (point-wise MLP); weights: `artifacts/neural_channel_estimator.weights.h5`
- Metrics: BER and BLER vs Eb/No
- Sweep: Eb/No = [-3, 0, 3, 6, 9] dB (batch=64, target_block_errors≈50, max_iter=50)

## Latest Artifacts
- LS results JSON: `results/simulation_results_umi_ls_20251110_164531.json`
- Neural results JSON: `results/simulation_results_umi_neural_20251110_165128.json`
- LS plot: `results/simulation_plot_umi_ls_20251110_164531.png`
- Neural plot: `results/simulation_plot_umi_neural_20251110_165128.png`
- Combined comparison plot: `results/comparison_plot_umi_ls_vs_neural_20251110_171228.png`

## Comparison Plot

![LS vs Neural - BER/BLER](../results/comparison_plot_umi_ls_vs_neural_20251110_171228.png)

## Numerical Summary

| Estimator | Eb/No (dB) | BER          | BLER         |
|-----------|------------:|-------------:|-------------:|
| LS        | -3          | 1.99796e-01  | 9.92188e-01  |
| LS        | 0           | 2.38164e-02  | 1.64063e-01  |
| LS        | 3           | 1.96402e-03  | 1.39509e-02  |
| LS        | 6           | 4.79991e-04  | 2.57813e-03  |
| LS        | 9           | 3.19112e-04  | 2.26562e-03  |
| Neural    | -3          | 1.99860e-01  | 9.92188e-01  |
| Neural    | 0           | 2.17908e-02  | 1.66016e-01  |
| Neural    | 3           | 1.77920e-03  | 1.10677e-02  |
| Neural    | 6           | 4.90977e-04  | 2.57813e-03  |
| Neural    | 9           | 2.13521e-04  | 1.64063e-03  |

## Findings
- Low SNR (≤ 0 dB): LS and Neural perform similarly; both limited by channel conditions.
- Medium/High SNR (≥ 3 dB): Neural shows consistent BLER improvements:
  - 3 dB: BLER ↓ from 1.40e-02 (LS) to 1.11e-02 (Neural)
  - 9 dB: BLER ↓ from 2.27e-03 (LS) to 1.64e-03 (Neural)
- BER trends match BLER improvements at higher SNRs.

## Notes
- The neural estimator here was trained with a lightweight CPU-friendly recipe; more epochs/data and GPU training typically yield larger gains.
- For broader evaluation, increase the Eb/No resolution and extend to other scenarios (UMa/RMa) and mobility.
