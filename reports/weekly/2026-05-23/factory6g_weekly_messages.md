# Factory6G Weekly Update Drafts - 2026-05-23

## WhatsApp

Assalamu alaikum Dr.,

Completed this week:
1. Ran a cross-channel resource-manager screening benchmark for Rayleigh, Rician, and UMI/TR38901 in the Factory6G smart-factory simulation.
2. Compared the existing baseline resource managers against the BER-first trained `ber_drl` method: `static`, `round_robin`, `max_throughput`, `pf`, `wmmse`, `queue_aware`, `drl`, and `ber_drl`.
3. Cleaned the resource-manager comparison report using the same ranked BER format as the 2026-05-01 report.
4. Generated presentation-style BER channel comparison curves based on the three-channel estimator plot style.

Current result:
- Rayleigh: `ber_drl` matched the best baseline by mean BER (`0`) and BER upper-confidence bound (`2.4379e-06`) in the shortened screening run.
- Rician: `ber_drl` ranked second with mean BER `1.8200e-05`, behind `max_throughput` with BER `0`.
- UMI/TR38901: `ber_drl` ranked second with mean BER `2.6042e-05`, close to `queue_aware` (`2.6190e-05`) but still behind `max_throughput` with BER `0`.

Important note:
- This was a short screening run (`batch_size=20`, `max_batches=20`), so zero-error points should be treated using the BER upper-confidence bound, not as final proof of true zero BER.

Next plan:
1. Re-run the same cross-channel benchmark with a longer Monte Carlo setting to reduce zero-error ambiguity.
2. Tune the BER-first policy using the Rician and UMI/TR38901 gap as the main target.
3. Update the thesis/report evidence after the longer benchmark confirms whether `ber_drl` can beat or consistently match the strongest baselines.

## Email

Subject: Factory6G Weekly Progress Evidence - Cross-Channel Resource Manager BER Benchmark

Dear Dr.,

Please find attached the weekly progress evidence for the Factory6G research work.

This week's work focused on cross-channel BER evaluation for AI-based resource management in the smart-factory simulation:
1. I ran a shortened screening benchmark across Rayleigh, Rician, and UMI/TR38901 channel environments.
2. I compared `ber_drl` against the existing resource-manager baselines: `static`, `round_robin`, `max_throughput`, `pf`, `wmmse`, `queue_aware`, and `drl`.
3. I cleaned the comparison report using the same ranked BER structure as the previous 2026-05-01 report.
4. I generated presentation-style BER channel comparison plots to make the cross-channel behavior easier to inspect.

The current result is useful as screening evidence but not yet a final improvement claim. In Rayleigh, `ber_drl` matched the best baseline with mean BER `0` and BER upper-confidence bound `2.4379e-06`. In Rician, `ber_drl` ranked second with mean BER `1.8200e-05`, behind `max_throughput` with BER `0`. In UMI/TR38901, `ber_drl` ranked second with mean BER `2.6042e-05`, close to `queue_aware` at `2.6190e-05`, but still behind `max_throughput` with BER `0`.

Because this was a shortened run (`batch_size=20`, `max_batches=20`), the zero-error results should be interpreted with the reported BER upper-confidence bound. The next step is to repeat the cross-channel benchmark with longer Monte Carlo settings and then tune the BER-first policy against the Rician and UMI/TR38901 gaps.

Best regards,
Yahya
