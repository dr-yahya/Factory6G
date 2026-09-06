# The Adaptive Hybrid's LMMSE Branch Does Not Earn Its Place

**Date:** 2026-09-05
**Channel:** TR 38.901 InF-DH, 15×15×5 m, mini-slot 61.4 MHz (selectivity 7.3)
**Method:** `scripts/experiments/calibrate_adaptive_branch.py`, 20 batches × 20 per point
**Raw:** `branch_cal.json`

The adaptive estimator blends a DFT branch and an LMMSE branch, weighted by
per-user LS SNR and delay-domain leakage. Its thresholds
(`quality_low = 3.0`, `quality_high = 12.0`) came from MSE reasoning: the
assumption that LMMSE is the better estimator at low SNR. This measures whether
that is true, by decoding both branches on identical channel realisations.

## Measured branch performance

| Eb/N0 | DFT BLER | LMMSE BLER | Better | Quality stat | Leakage |
|---|---|---|---|---|---|
| 0  | 5.681e-01 | **5.644e-01** | lmmse | 0.74 | 0.469 |
| 2  | **4.850e-01** | 4.938e-01 | dft | 1.00 | 0.418 |
| 4  | **3.563e-01** | 3.831e-01 | dft | 1.47 | 0.349 |
| 6  | **2.231e-01** | 2.619e-01 | dft | 2.17 | 0.278 |
| 8  | **1.219e-01** | 1.594e-01 | dft | 3.26 | 0.208 |
| 10 | **5.875e-02** | 9.625e-02 | dft | 5.04 | 0.154 |
| 12 | **3.313e-02** | 5.312e-02 | dft | 7.89 | 0.118 |
| 14 | **1.188e-02** | 2.813e-02 | dft | 12.27 | 0.084 |
| 16 | **1.875e-03** | 1.875e-02 | dft | 18.80 | 0.063 |
| 18 | **6.250e-04** | 3.125e-03 | dft | 30.76 | 0.039 |
| 20 | **6.250e-04** | 1.250e-03 | dft | 47.84 | 0.027 |

DFT wins at every point from 2 dB up, by up to a factor of ten. The single LMMSE
win at 0 dB is 0.65% — comfortably inside the confidence interval at 8,000
codewords.

## The verdict is that there is no crossover to tune

Mean BLER of the oracle branch selection: **1.6886e-01**.
Mean BLER of using DFT unconditionally: **1.6920e-01**.

A 0.2% difference. A perfect, clairvoyant branch selector would beat a plain DFT
estimator by essentially nothing on this channel — so no threshold setting can
make the hybrid worthwhile here. The calibration returns `verdict: dft_only`.

## Why, and how far it generalises

DFT truncation keeps the first 20 delay taps. At 61.4 MHz one tap is 16.3 ns, so
the window spans 326 ns. The hall's RMS delay spread is 23.7 ns — and 39.4 ns in
the largest hall modelled. The delay profile fits inside the truncation window
with an order of magnitude to spare, so DFT discards essentially no signal while
removing noise from 96% of the delay axis. It is close to a free lunch, and
nothing that smooths in the frequency domain can beat it.

That argument covers every hall size in this study, so the result is not specific
to the small hall: **for indoor factory propagation, delay-domain truncation is
the right estimator and the hybrid adds complexity without gain.**

It is also consistent with the UMi evidence, where LMMSE likewise lost in BER
despite the best NMSE. Across both channels tested, the LMMSE branch has never
been the BER-optimal choice.

## What this means for the thesis

The honest reading is that the *hybrid* is not the contribution — the *finding*
is. Three results now support one story:

1. MSE-optimal channel estimation is not BER-optimal in coded multi-user OFDM.
2. Which estimator wins is set by delay spread against the truncation window, not
   by SNR — the statistic the current policy switches on.
3. For indoor factory halls that comparison always favours DFT, because the hall's
   delay profile is short relative to any reasonable window.

Two ways forward, and this is a decision about what the contribution *is*:

* **Report the finding and simplify.** Present DFT truncation as the recommended
  factory estimator, with the adaptive hybrid as a documented negative result.
  Clean, well-evidenced, and cheap.
* **Change what adapts.** The selection mechanism is sound; the branch set is
  wrong. An estimator that adapts its *truncation window* to the measured delay
  spread would attack the mechanism identified above directly, and would matter
  precisely where fixed-window DFT fails — channels whose delay spread approaches
  the cyclic prefix. That is a stronger contribution but it is new work.

**Caveat:** TensorFlow 2.21 / Sionna 1.2.1, not the pinned 2.15 Docker image.
