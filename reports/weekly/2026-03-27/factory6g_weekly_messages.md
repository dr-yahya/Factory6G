# Factory6G Weekly Update Drafts - 2026-03-27

## WhatsApp

Assalamu alaikum Dr.,

Completed this week:
1. Completed the March Factory6G physical-layer benchmark across channel estimators, modulation schemes, channel models, factory sizes, and the JIDD-SCMA joint detection-decoding system.
2. Compared LS, PSO, Adaptive, ISTA, Neural, and DFT estimator behavior using BER, latency, runtime, and BER-floor evidence.
3. Confirmed Adaptive and PSO as the strongest OFDM estimator options on TR 38.901 UMi, with BER floors around `3-4 x 10^-5`.
4. Validated the corrected JIDD-SCMA run after fixing the high-SNR numerical issue; the corrected run reaches zero BER above `9 dB`.
5. Generated proof plots for estimator BER, modulation impact, channel-model impact, factory-size scaling, JIDD-SCMA behavior, and runtime comparison.

Current result:
- Adaptive is the practical estimator choice because it reaches near-PSO BER while avoiding PSO's heavy runtime cost.
- TR 38.901 UMi remains the hardest realistic channel, with LS showing a BER floor around `3 x 10^-4`.
- Rayleigh and Rician reach zero BER at high SNR, which shows LDPC coding is not the main bottleneck when estimation is accurate.
- 64-QAM is not reliable with LS in the current setup because its BER floor is much worse than QPSK.
- Large factory conditions are unresolved: BER remains around `0.21`, so the system likely needs more antennas, scheduling, or distributed MIMO.

Next plan:
1. Retrain or redesign the neural estimator because the March results show it behaves the same as LS.
2. Re-run DFT under the same small-factory setup to make the comparison fair.
3. Test Adaptive with 16-QAM and 64-QAM to see whether higher throughput becomes practical.
4. Investigate the large-factory reliability issue with scheduling, more antennas, or distributed MIMO.
5. Run fresh resource-manager benchmarks using the current Factory6G configuration.

## Email

Subject: Factory6G Weekly Progress Evidence - March PHY Benchmark

Dear Dr.,

Please find attached the weekly progress evidence slides for the Factory6G research work.

This week's completed work focused on the March physical-layer benchmark. I evaluated the Factory6G simulation platform across channel estimators, modulation schemes, channel models, factory sizes, and the JIDD-SCMA joint detection-decoding system.

The main result is that Adaptive and PSO provide the best OFDM channel-estimation reliability on TR 38.901 UMi, with BER floors around `3-4 x 10^-5`. Adaptive is the stronger practical choice because it achieves similar BER to PSO while avoiding PSO's very high runtime cost.

The report also confirms that TR 38.901 UMi is much harder than Rayleigh or Rician because it creates a persistent BER floor with LS estimation. Higher-order modulation increases the problem, especially 64-QAM. Large factory conditions are currently unreliable, with BER around `0.21`, so this scenario needs architectural changes rather than parameter tuning only.

The JIDD-SCMA results were also reviewed. The first run had a high-SNR numerical issue that caused BER to rebound toward random-guessing behavior. The corrected run removes that issue and reaches zero BER above `9 dB`, showing the reliability benefit of joint iterative detection and decoding.

Best regards,
Yahya
