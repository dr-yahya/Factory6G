# Channel Estimation In A Factory Hall, At A Bandwidth That Can See It

**Date:** 2026-09-05
**Channel:** TR 38.901 InF-DH, 15×15×5 m hall, 5 machines, RMS delay spread 23.7 ns
**Numerology:** 13 GHz FR3, 120 kHz SCS, **512 subcarriers × 4-symbol mini-slot = 61.4 MHz**
**Selectivity ratio:** 7.27 (signal bandwidth ÷ coherence bandwidth)
**Evidence:** 40 batches × 20 per Eb/No point = 4,915,200 information bits per point
**Config:** `config/thesis/estimators_inf_s.json`

This supersedes `reports/evidence/estimators-inf-factory/`, which ran the same
comparison at 3.8 MHz where the hall is frequency-flat and no estimator could
distinguish itself.

## Why this numerology

An indoor hall has a short delay spread, so its coherence bandwidth is *wide* —
5–8 MHz here. A carrier narrower than that sees a flat channel. The 4-symbol
mini-slot is what makes a wide carrier reachable: it produces a codeword 3.5×
shorter than a 14-symbol slot, so 512 subcarriers give k = 1024, comfortably
inside the 5G LDPC maximum of 8448, where 14 symbols at the same width would
need k = 12288 and fail outright.

The mini-slot TTI is therefore not only the low-latency mechanism — it is the
enabler for the bandwidth at which factory channel estimation is a real problem.

## Eb/N0 required to reach a target BLER

| Target BLER | LS | DFT | LMMSE | Adaptive | Perfect CSI |
|---|---|---|---|---|---|
| 1e-1 | 12.01 dB | **8.54 dB** | 9.93 dB | 9.37 dB | 7.38 dB |
| 1e-2 | 17.22 dB | **14.38 dB** | 16.30 dB | 14.63 dB | 13.24 dB |

**Gain over LS:**

| Target BLER | DFT | LMMSE | Adaptive | Perfect CSI |
|---|---|---|---|---|
| 1e-1 | **+3.47 dB** | +2.08 dB | +2.64 dB | +4.63 dB |
| 1e-2 | **+2.84 dB** | +0.92 dB | +2.59 dB | +3.98 dB |

DFT truncation is the best practical estimator on this channel, capturing about
three quarters of the gain available to perfect CSI. The adaptive hybrid is
second; LMMSE is third.

## NMSE agrees with BLER here

| Eb/N0 | LS | DFT | LMMSE | Adaptive |
|---|---|---|---|---|
| 10 | -12.79 | **-20.95** | -16.84 | -17.94 |
| 14 | -16.72 | **-24.87** | -19.42 | -23.40 |
| 20 | -22.34 | **-30.45** | -23.77 | -30.25 |

DFT has both the best NMSE and the best BLER, so the two metrics rank the
estimators identically. That is a third data point on the earlier UMi anomaly and
it fits the delay-window explanation: the hall's 23.7 ns spread sits inside the
truncation window even at 61 MHz, so DFT discards almost no signal and its
denoising is close to free. On UMi the spread overruns the window, wrecking DFT's
NMSE while leaving its coded BER intact.

## The adaptive hybrid is choosing the wrong branch

Read the low-SNR rows of the BLER table: from 0 to 6 dB, adaptive's BLER is
*identical* to LMMSE's to four significant figures (0.5922, 0.4778, 0.3681,
0.2647). It is sitting entirely in the LMMSE branch. But DFT beats LMMSE from
2 dB upward, so the branch policy is choosing the weaker estimator across most of
the range where it is active.

The thresholds (`quality_low = 3.0`, `quality_high = 12.0`) were set from MSE
reasoning — the assumption that LMMSE is better at low SNR. This channel says the
crossover actually sits near 0–2 dB.

If adaptive matched DFT throughout it would gain **+3.47 / +2.84 dB** instead of
**+2.64 / +2.59 dB** — roughly 0.8 dB at BLER 1e-1. That is the cost of a policy
that is right by luck rather than by calibration, and
`scripts/experiments/calibrate_adaptive_branch.py` fits it against measured BLER
instead.

## Reproducing

```bash
python -m factory6g.cli.run --config config/thesis/estimators_inf_s.json
```

The committed config uses 100 batches; this run used 40 for turnaround. Raw
payload, CSV, figures and the exact config are stored beside this note.

**Caveat:** TensorFlow 2.21 / Sionna 1.2.1 in a local virtualenv, not the pinned
2.15 Docker image. Regenerate in the container before quoting.
