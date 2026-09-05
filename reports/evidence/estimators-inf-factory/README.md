# Channel Estimation In A TR 38.901 Indoor Factory Hall

> ## SUPERSEDED — see `reports/evidence/estimators-inf-minislot/`
>
> These numbers were produced with an InF channel that was **frequency-flat**.
> The TR 38.901 large-scale model (path loss, LOS probability, shadow fading) was
> layered onto single-tap block fading, and the hall volume/surface parameters
> that set the delay spread were never used. Every estimator therefore faced a
> channel with no frequency selectivity, which is why the waterfalls are clean
> and NMSE and BLER agree so neatly.
>
> The channel now implements the InF delay profile (TR 38.901 Table 7.5-6:
> `mu_lgDS = log10(26*(V/S) + 14) - 9.35`), giving 23.7 ns in the small hall and
> 39.4 ns in the large one.
>
> **However**, that measurement also shows a 3.84 MHz carrier cannot resolve
> those spreads — the coherence bandwidth is 5-8 MHz, so the channel really is
> flat at this numerology. The conclusions below are not wrong for *this*
> bandwidth; they are simply uninformative about frequency-domain estimation.
> See the bandwidth analysis in `docs/THESIS_RESULTS.md`.

**Date:** 2026-09-05
**Channel:** TR 38.901 InF-DH (dense clutter, high BS), 15x15x5 m hall, 5 machines
**System:** 3.5 GHz FR1, QPSK, rate 1/2, 4 users, 8 BS antennas, fft 128
**Evidence:** 40 batches x 20 per Eb/No point = 4,915,200 information bits per point
**Config:** `config/thesis/estimators_inf_s.json`

This is the first estimator comparison run on an actual factory channel. Every
previous result family used TR 38.901 UMi — outdoor urban microcell — because the
hall geometry never reached the propagation model and `--factory-size` changed
only the user count.

## Eb/No required to reach a target BLER

| Target BLER | LS | DFT | LMMSE | Adaptive | Perfect CSI |
|---|---|---|---|---|---|
| 1e-1 | 9.25 dB | 7.75 dB | 7.38 dB | **7.23 dB** | 5.00 dB |
| 1e-2 | 15.07 dB | 13.45 dB | 14.07 dB | **13.35 dB** | 10.53 dB |
| 5e-3 | 16.37 dB | 14.58 dB | 15.55 dB | **14.54 dB** | 12.00 dB |

**Gain over LS:**

| Target BLER | DFT | LMMSE | Adaptive | Perfect CSI |
|---|---|---|---|---|
| 1e-1 | +1.50 dB | +1.87 dB | **+2.02 dB** | +4.25 dB |
| 1e-2 | +1.62 dB | +1.00 dB | **+1.72 dB** | +4.54 dB |
| 5e-3 | +1.79 dB | +0.82 dB | **+1.83 dB** | +4.37 dB |

The adaptive hybrid is the best practical estimator at every operating point,
worth **1.7-2.0 dB over LS**. Perfect CSI is a further 2.2-2.8 dB away, so
roughly 40-45% of the total available estimation gain is captured.

## The waterfalls are clean — no floor

Unlike UMi, every estimator on the factory channel produces a monotone waterfall
down to zero observed errors at 20 dB. The irreducible floor documented in
`reports/evidence/llr_clip_floor/` is a UMi phenomenon, not a factory one.

## NMSE and BLER agree here

| Eb/No | LS | DFT | LMMSE | Adaptive |
|---|---|---|---|---|
| 10 | -14.73 | -18.25 | -18.09 | **-18.78** |
| 14 | -18.72 | -22.24 | -20.84 | **-22.40** |
| 20 | -24.72 | **-28.24** | -25.62 | -28.25 |

This matters, because on UMi they did not: there LMMSE had the best NMSE at every
point and a BER two orders of magnitude worse
(`reports/evidence/estimator-floor-tr38901/`). On InF the two metrics rank the
estimators identically, and the estimator with the best NMSE also has the best
BLER.

**The two datasets together explain each other.** DFT truncation keeps 20 delay
taps. InF-DH is an indoor hall with a short delay spread, so almost all channel
energy falls inside that window and truncation is nearly lossless — hence DFT's
-28 dB NMSE and its strong BLER. UMi has a delay spread that overruns the window,
so truncation discards real signal, wrecking DFT's NMSE while leaving its coded
BER intact because the resulting error is a smooth bias rather than
per-subcarrier noise.

That is the mechanism behind the UMi inversion, and it is now supported by two
independent channels rather than asserted. It also carries a design lesson worth
stating in the thesis: **the delay-truncation window must be matched to the
deployment's delay spread**, and an estimator selected on UMi statistics is not
automatically the right one for a factory.

## Why the adaptive hybrid wins here

Adaptive tracks DFT closely at high SNR and beats it at 6-10 dB, where its LMMSE
branch contributes. On this channel LMMSE is genuinely competitive at low SNR
(+1.87 dB over LS at BLER 1e-1, better than DFT's +1.50), so the branch policy is
choosing correctly — which is not the case on UMi, where the same policy leans on
LMMSE at low SNR and loses to DFT.

The adaptive contribution is therefore best defended on the factory channel,
where its branch decision is doing real work.

## Reproducing

```bash
python -m factory6g.cli.run --config config/thesis/estimators_inf_s.json
```

The committed config uses 100 batches; this run used 40 for turnaround. Raw
`stage_results_v2.json`, the CSV and the figures are stored beside this note.

**Caveat:** TensorFlow 2.21 / Sionna 1.2.1 in a local virtualenv, not the pinned
2.15 Docker image. Regenerate in the container before quoting.
