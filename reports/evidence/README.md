# Curated Research Evidence

Cross-cutting plots, tables, and stage summaries promoted from local full runs
under `results/`. Each subfolder is an **evidence bundle** with a
`manifest.json` describing the source run and promoted files.

Week-specific stakeholder packages live under `reports/weekly/<date>/`.

## Bundles

| Bundle | Purpose |
|---|---|
| `estimator-benchmarks/` | PHY estimator, modulation, factory-size, and neural-vs-LS plots |
| `rm-ber-first-apr-2026/` | April 2026 resource-manager baseline runs (Rayleigh + UMI) |
| `rm-cross-channel-may-2026/` | May 2026 cross-channel RM screening stage summaries |
| `llr_clip_floor/` | The TR 38.901 error floor is estimation error, not LLR clipping |
| `estimator-floor-tr38901/` | NMSE does not predict coded BER, unconfounded by error-variance declaration |
| `estimators-inf-factory/` | **Superseded** — InF run made before the channel was frequency-selective |
| `estimators-inf-minislot/` | Estimators on a factory channel the carrier can resolve (mini-slot FR3) |
| `adaptive-branch-calibration/` | A clairvoyant DFT/LMMSE branch selector is worth 0.2% — the ceiling on that idea |
| `adaptive-window-estimator/` | Sizing the truncation window to the channel: up to 3.3 dB on a hall, 11.5 dB narrowband |
