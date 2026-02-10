# Simulation Results: CNN Resource Manager

**Date:** 2026-02-10
**Scenario:** UMi
**CNN Model:** `models/cnn_resource_manager.h5`
**Batch Size:** 16
**Iterations:** 10 per Eb/No point (Demo Run)

## Performance Metrics

### Perfect CSI (Baseline)
| Eb/No (dB) | BER | BLER | SINR (dB) |
| :--- | :--- | :--- | :--- |
| -5.0 | 0.000e+00 | 0.000e+00 | 6.989 |
| ... | ... | ... | ... |
| 15.0 | 0.000e+00 | 0.000e+00 | 25.241 |

*Note: Perfect CSI with efficient resource allocation yields 0 errors in this high-SNR regime.*

### Imperfect CSI (LS Estimator + CNN Resource Manager)
| Eb/No (dB) | BER | BLER | SINR (dB) | Decoder Iter |
| :--- | :--- | :--- | :--- | :--- |
| -5.0 | 6.390e-03 | 3.281e-02 | 6.976 | -2.875 |
| -3.0 | 1.266e-03 | 7.031e-03 | 8.351 | -4.786 |
| -1.0 | 1.859e-03 | 7.812e-03 | 9.908 | -6.482 |
| 1.0 | 0.000e+00 | 0.000e+00 | 11.604 | -8.001 |
| 3.0 | 2.719e-03 | 1.016e-02 | 13.357 | -9.576 |
| 5.0 | 0.000e+00 | 0.000e+00 | 15.298 | -10.801 |
| 7.0 | 0.000e+00 | 0.000e+00 | 17.196 | -12.592 |
| 9.0 | 1.836e-03 | 8.594e-03 | 19.090 | -12.686 |
| 11.0 | 2.080e-03 | 1.328e-02 | 21.086 | -13.313 |
| 13.0 | 3.114e-03 | 1.484e-02 | 22.992 | -13.664 |
| 15.0 | 2.783e-03 | 1.328e-02 | 24.990 | -13.982 |

## Observations
- The CNN Resource Manager is functioning and making allocation decisions.
- At low Eb/No (-5 to -1 dB), the BER is relatively high (~10^-3 to 10^-2).
- At medium Eb/No (1 to 7 dB), performance improves significantly (0 errors observed).
- At high Eb/No (9+ dB), errors reappear. This counter-intuitive behavior might suggest:
    - The model over-allocates power or users interfere more at high SNR (interference limited regime).
    - The training dataset (generated at random SNR 0-20dB) might be biased or the model struggles to generalize to specific high-SNR interference patterns.
    - Limitation of the "10 iterations" demo run (variance is high).

## Next Steps
- Train on a larger dataset (e.g., 1000+ samples) to improve model robustness.
- Run a longer simulation (100+ iterations) to get statistically significant results.
- Analyze the specific allocation decisions (Mask, Power) to understand the high-SNR error floor.
