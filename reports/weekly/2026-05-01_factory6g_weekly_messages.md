# Factory6G Weekly Update Drafts - 2026-05-01

## WhatsApp

Assalamu alaikum Dr.,

Completed this week:
1. Built the BER-first DRL resource-manager workflow for the Factory6G smart-factory simulation.
2. Added BER-focused training support using oracle reliability labels, BER upper-confidence scoring, and a separate `ber_drl` resource-manager checkpoint.
3. Completed `ber_drl` benchmark runs for both Rayleigh and UMI/TR38901 channels and generated proof graphs for BER, throughput, runtime, latency, and power.
4. Validated the workflow inside Docker: 14 tests passed for dataset generation, DRL policy loading/projection, and BER report generation.

Current result:
- Rayleigh: `ber_drl` completed with mean BER `1.149e-07`, but it does not beat the best baseline yet because baseline `drl` reached BER `0`.
- UMI/TR38901: `ber_drl` completed with mean BER `1.687e-04`; it is close to the baseline DRL result but does not beat `max_throughput` yet.

Next plan:
1. Expand the BER-first training dataset beyond the current small training run.
2. Tune the BER-first policy and channel-aware projection to reduce the gap against the strongest baselines.
3. Re-run Rayleigh and UMI/TR38901 comparison benchmarks and update the thesis evidence section.

## Email

Subject: Factory6G Weekly Progress Evidence - BER-First AI Resource Management

Dear Dr.,

Please find attached the weekly progress evidence slides for the Factory6G research work.

This week's completed work focused on the BER-first AI resource-management workflow for 6G smart-factory simulations:
1. I implemented and validated BER-focused training support for the DRL-style resource manager.
2. I created a separate `ber_drl` checkpoint path so BER-first experiments can be compared independently from the existing baseline DRL resource manager.
3. I completed Rayleigh and UMI/TR38901 `ber_drl` benchmark runs and generated proof graphs for BER, throughput, runtime, latency, and power.
4. I validated the related workflow inside Docker, with 14 tests passing across dataset generation, policy loading/projection, and BER report generation.

The current result is useful but not yet a final improvement claim. In Rayleigh, `ber_drl` achieved mean BER `1.149e-07`, while the strongest baseline achieved BER `0`. In UMI/TR38901, `ber_drl` achieved mean BER `1.687e-04`, which is close to the baseline DRL result but still behind `max_throughput` at `9.807e-05`.

The next step is to expand the BER-first training data, tune the policy/projection settings, and re-run the cross-channel benchmarks to target a clearer reliability improvement.

Best regards,
Yahya
