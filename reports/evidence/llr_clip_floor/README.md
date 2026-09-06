# What Causes The TR 38.901 BER Floor

**Date:** 2026-09-05
**Verdict:** the floor is **channel-estimation error**, not LLR clipping.
The LLR-clip hypothesis raised in `docs/SIMULATION_REVIEW.md` §1.1 is **refuted**.

## Why this was tested

The receiver clipped demapper LLRs hard at ±20, justified by a diagnostic noting
that 27.5% of LLRs exceeded 50 in magnitude. The review argued that saturating
legitimately large high-SNR LLRs would starve belief propagation of the
reliability information it needs, producing an artificial error floor — and
named this the most likely origin of the "TR 38.901 BER floor" recorded in the
March and May 2026 weekly reports.

That was a plausible mechanism. It is not what is happening.

## Method

`scripts/experiments/llr_clip_floor.py`. For every (Eb/No point, batch), one
channel realisation, noise draw and source-bit set is decoded by receivers
differing *only* in their LLR clip — common random numbers, so the comparison is
exactly paired and free of Monte Carlo variance. A separate arm replaces LS
estimation with perfect CSI to isolate estimation error.

7.4–9.8 million information bits per Eb/No point (60–80 batches × 20 × 4 users).

## Result 1 — Rayleigh is the control: no floor exists

| Eb/No (dB) | clip 20 | clip 200 | no clip |
|---|---|---|---|
| 0 | 8.466e-03 | 8.466e-03 | 8.466e-03 |
| 2 | 6.550e-04 | 6.550e-04 | 6.550e-04 |
| 4 | 9.186e-05 | 9.186e-05 | 9.186e-05 |
| 6 … 20 | 0 errors | 0 errors | 0 errors |

9,830,400 bits per point, so "0 errors" is BER < 3.0e-7 at 95% confidence.
Identical to every digit across all three clip settings, and no floor at all.

## Result 2 — TR 38.901 floors, and the clip is irrelevant to it

| Eb/No (dB) | LS, clip 20 | LS, clip 200 | LS, no clip | **Perfect CSI** |
|---|---|---|---|---|
| 6  | 3.159e-04 | 3.159e-04 | 3.159e-04 | **0 errors** |
| 8  | 2.230e-04 | 2.230e-04 | 2.230e-04 | **0 errors** |
| 10 | 2.081e-04 | 2.081e-04 | 2.081e-04 | **0 errors** |
| 12 | 2.364e-04 | 2.364e-04 | 2.364e-04 | **0 errors** |
| 14 | 2.664e-04 | 2.664e-04 | 2.664e-04 | **0 errors** |
| 16 | 1.298e-04 | 1.298e-04 | 1.298e-04 | **0 errors** |
| 18 | 2.620e-04 | 2.620e-04 | 2.620e-04 | **0 errors** |
| 20 | 6.700e-05 | 6.700e-05 | 6.700e-05 | **0 errors** |

7,372,800 bits per point; "0 errors" is BER < 4.1e-7.

Two things are unambiguous:

1. **The floor is real.** With LS estimation the BER is flat at 1–3e-4 across
   14 dB of Eb/No — it stops responding to SNR entirely.
2. **The clip has no part in it.** All three clip settings agree bit-for-bit at
   every point. Widening the clip tenfold, or removing it, changes nothing.
3. **Perfect CSI removes the floor completely.** Zero errors at every point,
   three orders of magnitude below the LS floor.

## Interpretation

The floor is an irreducible channel-estimation error, and the mechanism is
structural rather than noise-driven. TR 38.901 UMi is frequency-selective; the
configuration uses two pilot symbols with nearest-neighbour interpolation
(`LSChannelEstimator(interpolation_type="nn")`). Interpolation error is then set
by the ratio of pilot spacing to coherence bandwidth, which does not shrink as
Eb/No rises. Raising SNR cleans up the pilot observations but cannot fix an
interpolator that is blind between pilots — so the BER stops improving.

Rayleigh block fading is flat across frequency, so nearest-neighbour
interpolation is exact and no floor appears. That is precisely why the control
arm is clean.

## What this means for the thesis

This is a better outcome than the hypothesis being correct.

* **The floor is a result, not a bug.** It is the honest performance of LS
  estimation in a frequency-selective channel, and it belongs in the thesis as
  such rather than being quietly attributed to a receiver artifact.
* **It is exactly the gap the estimator work exists to close.** DFT truncation,
  LMMSE smoothing and the adaptive hybrid all attack interpolation error
  directly. The perfect-CSI arm gives the floor's lower bound, so each
  estimator's remaining distance from it is a clean, quantitative measure of
  what it contributes.
* **Report NMSE alongside BER.** The floor is an estimation-accuracy phenomenon,
  and NMSE measures it directly without the LDPC decoder in the way.
* **The LLR clip change still stands** — clipping legitimately large LLRs was
  poor practice and the default is now 200 — but it should be described as
  hygiene, not as a fix for the floor. No published curve changes because of it.

## Reproducing

```bash
python scripts/experiments/llr_clip_floor.py --channel rayleigh --batches 80 \
    --output llr_clip_rayleigh.json
python scripts/experiments/llr_clip_floor.py --channel tr38901 --estimator ls \
    --ebno-min 6 --ebno-max 20 --batches 60 --output llr_clip_tr38901_ls.json
python scripts/experiments/llr_clip_floor.py --channel tr38901 --estimator perfect \
    --ebno-min 6 --ebno-max 20 --batches 60 --output llr_clip_tr38901_perfect.json
python scripts/experiments/report_llr_clip_floor.py llr_clip_tr38901_ls.json
```

Raw payloads for all three runs are stored beside this note.

**Environment caveat:** these runs used TensorFlow 2.21 / Sionna 1.2.1 in a local
virtualenv, not the project's Docker image (TensorFlow 2.15). The conclusion is
robust — a three-orders-of-magnitude gap between LS and perfect CSI is not a
library-version artifact — but the numbers should be regenerated in the
container before they are quoted verbatim.
