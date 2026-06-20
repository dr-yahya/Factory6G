# Factory6G Weekly Update Drafts - 2026-04-20

## WhatsApp

Assalamu alaikum Dr.,

Completed this week:
1. Updated the Factory6G simulation evidence with April results for the retrained neural channel estimator.
2. Evaluated Neural vs LS on Rayleigh across QPSK, 16-QAM, and 64-QAM after retraining on Rayleigh flat-fading data.
3. Confirmed that the retrained neural estimator now outperforms LS on Rayleigh instead of collapsing to the LS baseline.
4. Added the corrected DFT small-factory result, making the TR 38.901 estimator comparison fairer than the previous medium-factory DFT run.
5. Reconfirmed that Adaptive, PSO, and DFT form the strongest classical estimator tier on TR 38.901 UMi.
6. Preserved the JIDD-SCMA evidence showing zero BER above `9 dB` after the numerical fix.

Current result:
- Neural beats LS on Rayleigh by up to `160x` at QPSK `0 dB`.
- Neural reaches zero BER about `2 dB` earlier than LS across QPSK, 16-QAM, and 64-QAM.
- Neural adds effectively no per-batch latency overhead compared with LS.
- On TR 38.901 UMi, LS still has a BER floor around `3 x 10^-4`, while Adaptive, PSO, and DFT reduce this toward `3-5 x 10^-5`.
- PSO remains impractical because it has very high runtime without a clear reliability advantage over Adaptive or DFT.
- Large factory operation remains unresolved because BER stays around `0.21` across SNR.

Next plan:
1. Train and test the neural estimator on TR 38.901 UMi to check whether deep learning can reduce the realistic-channel BER floor.
2. Compare Neural, Adaptive, and DFT under the same channel, factory size, and modulation settings.
3. Expand neural training across Rayleigh, Rician, and TR 38.901 using curriculum-style data generation.
4. Continue investigating the large-factory reliability issue with scheduling, more antennas, or distributed MIMO.
5. Test Adaptive or Neural estimation with higher-order modulation for high-throughput factory links.

## Email

Subject: Factory6G Weekly Progress Evidence - April Neural Estimator Update

Dear Dr.,

Please find attached the weekly progress evidence slides for the Factory6G research work.

This week's completed work focused on updating the simulation evidence with the retrained neural channel estimator. The new result corrects the earlier conclusion that the neural estimator did not improve over LS. After retraining on Rayleigh flat-fading data, the neural estimator significantly outperforms LS on Rayleigh.

For QPSK at `0 dB`, Neural reduces BER by up to `160x`. Across QPSK, 16-QAM, and 64-QAM, it reaches zero BER about `2 dB` earlier than LS while adding effectively no per-batch latency overhead.

The updated report also includes a fairer DFT comparison on the small-factory setup. On TR 38.901 UMi, Adaptive, PSO, and DFT remain the strongest classical estimators, with BER floors around `3-5 x 10^-5`, while LS remains around `3 x 10^-4`. PSO is still not practical because it has very high runtime without a clear reliability advantage over Adaptive or DFT.

The large-factory result remains unresolved. BER stays around `0.21`, which indicates that the current configuration needs architectural changes such as more BS antennas, user scheduling, or distributed MIMO. The next step is to train Neural on TR 38.901 and compare it directly against Adaptive and DFT on the same scenarios.

Best regards,
Yahya
