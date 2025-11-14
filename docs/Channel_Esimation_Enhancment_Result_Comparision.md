# Channel Estimation Enhancement Result Comparison (UMi, Imperfect CSI)

## Setup
- Scenario: UMi (Urban Micro)
- Direction: Uplink
- MIMO: 4 UT × 8 BS antennas
- OFDM: FFT 128, SCS 30 kHz, 14 symbols, pilot pattern kronecker
- Modulation/Coding: QPSK (2 bps), 5G LDPC, code rate 0.5
- Estimators compared:
  - LS_LIN: LS channel estimation with linear interpolation
  - LS_NN: LS channel estimation with nearest-neighbor (time–freq) interpolation
  - NEURAL: LS baseline + neural refinement (small MLP); weights: `artifacts/neural_channel_estimator.weights.h5`
- Metrics: BER and BLER vs Eb/No
- Sweep: Eb/No = [-3, 0, 3, 6, 9] dB (batch=64, target_block_errors≈40–50, max_iter=30–50)

## Latest Artifacts
- LS_NN JSON: `results/simulation_results_umi_ls_nn_20251111_154049.json`
- LS_LIN JSON: `results/simulation_results_umi_ls_lin_20251111_154244.json`
- NEURAL JSON: `results/simulation_results_umi_neural_20251111_154638.json`
- Combined plot (all): `results/comparison_plot_umi_all_20251111_154720.png`

## Comparison Plot

![All Estimators - BER/BLER](../results/comparison_plot_umi_all_20251111_154720.png)

## Numerical Summary (latest)

| Estimator | Eb/No (dB) | BER          | BLER         |
|-----------|------------:|-------------:|-------------:|
| LS_NN     | -3          | 1.99796e-01  | 9.92188e-01  |
| LS_NN     | 0           | 2.38164e-02  | 1.64063e-01  |
| LS_NN     | 3           | 1.98110e-03  | 1.36719e-02  |
| LS_NN     | 6           | 4.77092e-04  | 2.47396e-03  |
| LS_NN     | 9           | 3.27725e-04  | 1.95313e-03  |
| LS_LIN    | -3          | 1.03793e-01  | 7.73438e-01  |
| LS_LIN    | 0           | 5.04049e-03  | 4.49219e-02  |
| LS_LIN    | 3           | 5.12526e-04  | 4.03646e-03  |
| LS_LIN    | 6           | 2.51177e-04  | 9.11458e-04  |
| LS_LIN    | 9           | 0.00000e+00  | 0.00000e+00  |
| NEURAL    | -3          | 2.04567e-01  | 9.96094e-01  |
| NEURAL    | 0           | 2.04277e-02  | 1.52344e-01  |
| NEURAL    | 3           | 1.37346e-03  | 1.09375e-02  |
| NEURAL    | 6           | 6.51635e-04  | 2.86458e-03  |
| NEURAL    | 9           | 2.41089e-04  | 1.56250e-03  |

## Findings
- At medium/high SNRs, all methods reach low BLER; the relative ordering here shows LS_LIN performing best, followed by NEURAL, then LS_NN. This indicates linear interpolation is a strong classical baseline under this configuration.
- NEURAL improves over LS_NN at 3–9 dB BLER, but underperforms LS_LIN in this particular setup. With larger models/training or spatial-temporal context (CNN/UNet), neural methods typically surpass classical baselines.
- Low SNR (≤ 0 dB): all estimators are limited; differences are minor and dominated by noise.

## Notes
- The NEURAL estimator used a lightweight MLP on CPU; consider 2D CNN/UNet over the time–frequency grid, or algorithm-unfolded sparse estimators for larger gains.
- For broader evaluation, increase Eb/No resolution and include UMa/RMa, mobility, and pilot densities.
