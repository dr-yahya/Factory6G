# Sizing the DFT truncation window to the channel

**Date:** 2026-09-06
**Branch:** `claude/simulation-review-fixes`
**Code:** `src/factory6g/components/estimators/adaptive_window_estimator.py`
**Environment caveat:** produced under TensorFlow 2.21 / Sionna 1.2.1 in a
virtualenv, not the pinned TensorFlow 2.15 Docker image. Regenerate in the
container before quoting any number in the thesis.

## The question

The adaptive hybrid estimator switches between DFT truncation and LMMSE
smoothing. A clairvoyant version of that switch was measured
(`../adaptive-branch-calibration/`) and beats plain DFT by 0.2% on a factory
channel — so the branch choice carries almost nothing, and adapting it is not a
contribution. What the earlier work did show is that the delay spread against
the truncation window separates the two channels completely. So the window is
the quantity worth adapting.

## The rule

Each retained delay tap admits a tap's worth of estimation noise; each discarded
tap loses whatever signal it held. For a window of `L` taps,

    J(L) = sum_{k < L} nu_k  +  sum_{k >= L} s_k

is the estimate's mean squared error. `J` is evaluated at every admissible
length and the minimiser kept, per user, per slot. The window is capped at the
cyclic prefix, because energy arriving later is inter-symbol interference and is
not recoverable — so the method can only ever tighten below the fixed window.

## What the difficulty actually was

Four implementations were measured before this one worked, and the reason is
worth recording because it is not obvious.

**A detection threshold is the wrong rule.** Keeping taps that rise above the
noise floor won 2.3 dB at 0 dB Eb/No and lost 1.9 dB at 20 dB, because its
window *narrowed* as SNR rose. The MSE trade-off moves the other way: a quieter
channel supports a longer window.

**The estimation noise is not white along the delay axis.** Each user's pilots
occupy every `D`-th subcarrier, `D` being the co-scheduled stream count, and the
nearest-neighbour hold across the gaps is a rectangular filter in frequency —
so the interpolated noise follows the Dirichlet kernel in delay,

    g(k) = |sin(pi k D / N) / (D sin(pi k / N))|^2,

flat near zero delay with its first null at `k = N/D`. Measured, the in-window
noise runs **1.35×** the level past the cyclic prefix with four users and
**12.6×** with sixteen; the kernel predicts **1.32** and **13**. A constant
floor read from outside the window therefore undercharges every retained tap, by
a factor that moves with the user count — so no single correction factor repairs
it, and the window opens to the cap at every SNR.

**The kernel alone is not enough either.** Two errors reach the profile: the
interpolated pilot noise, which follows `g` and shrinks with SNR, and the hold's
own bias, which does not shrink and is far flatter. Attributing all of it to `g`
collapses the window once the second term takes over — up to 4 dB lost at 20 dB
on the large hall. The fit carries both: `nu_k = A g(k) + B`, recovered per user
by non-negative least squares against the profile beyond the cyclic prefix.

**The window only tightens on a clear margin.** Where the cost curve is shallow
near the full window, its ordering is inside the fit's own error. Requiring a 5%
predicted improvement before deviating removes the residual high-SNR losses
(0.24 dB at 20 dB on the large hall) and leaves the low-SNR gains untouched.

## Result

NMSE in dB against the fixed 20-tap DFT window, four batches of 20 per point,
with an exhaustive search over window length as the oracle.

| Eb/No | InF small | InF medium | InF large | Narrowband control | UMi |
|---|---|---|---|---|---|
| 0 dB | **+3.26** | +3.02 | +1.95 | **+11.48** | +1.95 |
| 4 dB | +2.46 | +2.01 | +1.32 | +11.41 | +0.92 |
| 8 dB | +1.83 | +1.61 | +0.74 | +11.16 | −0.30 |
| 12 dB | +1.11 | +0.81 | +0.29 | +10.59 | −0.61 |
| 16 dB | +0.70 | +0.30 | +0.02 | +9.96 | −0.93 |
| 20 dB | +0.38 | 0.00 | 0.00 | +8.78 | −1.08 |

On every factory row the result is within 0.1 dB of the oracle — the rule is
not merely better than the fixed window, it is close to the best any window
could do.

### The narrowband control is where the gain is largest

This inverts what the control was expected to show. At 3.8 MHz the factory
channel is flat, so the true profile is a single tap and the fixed 20-tap window
is twenty times too wide; the adaptive rule selects 1.0 taps and recovers
**11.5 dB**. The fixed window is calibrated to the cyclic prefix, and the
mismatch is worst exactly where the channel is simplest. The control's finding
is therefore not "estimators converge when the channel is flat" but "a
CP-length window is furthest from optimal when the channel is flat".

### Where the method does not apply

The fit reads the noise off the taps beyond the cyclic prefix, so it assumes the
channel has no usable energy there. Measured energy past the prefix is **0.000%**
in both the small and the large hall on the mini-slot numerology — it holds by
construction, since that is the design assumption of OFDM. It does **not** hold
for TR 38.901 UMi at this numerology, where **3.4%** of the channel energy
overruns the prefix, contaminates the fit, and makes the window over-tighten:
UMi loses up to 1.08 dB at high SNR. UMi is the comparison arm rather than a
factory claim, but the precondition is real and belongs with the result.

An explicit overrun detector was tried — fit on the outer part of the signal-free
region, test the inner part against the prediction — and rejected: it fires
spuriously on the large hall (ratio 2.4 at 8 dB against a true overrun of zero),
because the flat term is not flat enough for the extrapolation to hold. The
precondition is stated rather than detected.

## The gain survives the decoder

NMSE does not predict coded BER in this project (`../estimator-floor-tr38901/`),
so the accuracy result above settles nothing on its own. It was a live
possibility that the whole gain would vanish at BLER: it is concentrated at low
SNR, where BLER is already near a half and a better channel estimate has least
room to change a decode outcome.

It did not vanish. 100 batches of 20, 8000 codewords per point, InF small hall,
paired per batch against fixed DFT.

| Eb/No | BLER dft | BLER adaptive | delta (95% CI) | gap to perfect CSI closed |
|---|---|---|---|---|
| 0 dB | 5.879e-1 | 5.560e-1 | −3.19e-2 [−3.60e-2, −2.79e-2] | **43%** |
| 2 dB | 4.622e-1 | 4.330e-1 | −2.92e-2 [−3.25e-2, −2.60e-2] | 38% |
| 4 dB | 3.380e-1 | 3.078e-1 | −3.02e-2 [−3.44e-2, −2.66e-2] | **40%** |
| 6 dB | 2.260e-1 | 2.070e-1 | −1.90e-2 [−2.20e-2, −1.60e-2] | 32% |
| 8 dB | 1.270e-1 | 1.123e-1 | −1.48e-2 [−1.74e-2, −1.21e-2] | 35% |
| 10 dB | 6.71e-2 | 5.89e-2 | −8.25e-3 [−1.05e-2, −6.13e-3] | 32% |
| 12 dB | 2.79e-2 | 2.45e-2 | −3.38e-3 [−4.87e-3, −2.00e-3] | 28% |
| 14 dB | 1.28e-2 | 1.19e-2 | −8.75e-4 [−1.63e-3, −2.50e-4] | 14% |
| 16-20 dB | | | interval includes zero | — |

Every point from 0 to 14 dB is significant. Three things make this more than a
lower curve.

**The denominator is the perfect-CSI bound.** `../llr_clip_floor/` established
that the TR 38.901 floor is estimation error, so perfect CSI is the reachable
limit. The adaptive window closes **28-43%** of the distance from fixed DFT to
that limit across the whole significant range.

**Worst-user BLER improves with it** — 1.370e-1 to 1.195e-1 at 8 dB, a 13%
reduction. That is the URLLC metric, where the weakest device sets the system
guarantee, and a mean-BLER gain that came at the expense of the worst user would
be worth nothing for a factory claim.

**It is not the over-confidence artifact.** This project has already caught one
estimator winning on BER by declaring an optimistic error variance to the
equalizer. Here the winner is the *better-calibrated* estimator: the adaptive
window declares 0.42-0.77 of its true error against fixed DFT's 0.28-0.76, at
every point. The comparison runs the right way.

The 8 dB row is the whole argument in one line: LS 2.695e-1, LMMSE 1.676e-1,
DFT/LMMSE hybrid 1.633e-1, fixed DFT 1.270e-1, **adaptive window 1.123e-1**,
perfect CSI 8.488e-2. Both LMMSE and the hybrid are worse than plain DFT; only
the window improves on it. The branch was the wrong thing to adapt.

### The narrowband control, at BLER

Same run configuration, 8000 codewords per point, on the 3.8 MHz FR1 control
where the factory channel is flat.

| Eb/No | BLER dft | BLER adaptive | perfect CSI | gap closed | relative |
|---|---|---|---|---|---|
| 0 dB | 5.427e-1 | 3.997e-1 | 3.830e-1 | 89.5% | −26% |
| 4 dB | 3.055e-1 | 1.750e-1 | 1.643e-1 | 92.4% | −43% |
| 8 dB | 1.148e-1 | 5.187e-2 | 4.675e-2 | 92.5% | −55% |
| 12 dB | 2.513e-2 | 9.250e-3 | 7.875e-3 | 92.0% | −63% |
| 16 dB | 4.500e-3 | 1.500e-3 | 1.125e-3 | 88.9% | −67% |
| 18 dB | 1.250e-3 | 3.750e-4 | 1.250e-4 | 77.8% | −70% |

Significant at every point from 0 to 18 dB, and around **90% of the gap to
perfect CSI closed** throughout -- against 28-43% on the small hall. At 8 dB the
ordering is LS 1.739e-1, fixed DFT 1.148e-1, LMMSE 1.050e-1, hybrid 1.016e-1,
**adaptive window 5.187e-2**, perfect CSI 4.675e-2: the window sits within 11%
of perfect CSI while every other estimator is at more than double it.

The calibration is the cleanest in the project. The adaptive window declares
1.04-2.06 of its true error -- conservative, converging on 1.00 as SNR rises --
against fixed DFT's flat 0.35. The estimator that tells the equalizer the truth
wins by more than half a decade of BLER.

Two things follow for the thesis. This configuration was filed as a control
expected to show estimators converging on a flat channel; it shows the reverse,
and belongs in the chapter as a result. And unlike the factory halls, where the
gain fades into the noise above 14 dB, here it runs all the way into the URLLC
tail at 18 dB -- which is the regime the reliability claim is actually written
about.

## Caveats

* **The deep tail is not resolved.** Above 14 dB the intervals include zero
  because 8000 codewords give three block errors at 20 dB, not because the
  effect is absent. Importance sampling for the URLLC tail remains unimplemented
  and is the honest limit on any 1e-5 claim.
* **Only the small hall has run at BLER so far.** Narrowband, medium, large and
  UMi are queued.
* **The NMSE table above predates the hall-geometry fix.** It was measured with
  `inf_hall_surface_m2: 900` against the 750 the room implies, so it ran at a
  20.7 ns delay spread and selectivity 6.4 rather than 23.6 ns and 7.3. The
  effect is visible and small: the small hall's 0 dB gain reads +3.26 dB there
  against +2.74 dB on the corrected channel, because a shorter delay spread
  leaves the fixed window more slack to give up. The ordering is unaffected --
  the mechanism does not depend on the exact spread -- but the magnitudes are
  each a few tenths of a dB optimistic and the table is due a rerun.
* **Declared error variance is imperfect for both estimators.** On the small
  hall the adaptive window declares 0.45–0.69 of its true error and fixed DFT
  0.28–0.61; on the large hall at high SNR both over-declare by four to eight
  times. The two are close enough at each point that the comparison is not
  confounded, but neither is calibrated, and that should be said when the
  calibration table is presented.

## Reproducing

```bash
# NMSE against the exhaustive oracle
python scripts/experiments/adaptive_window_sweep.py config/thesis/estimators_inf_s.json

# BLER with the paired interval, from a full run
python -m factory6g.cli.run --config config/thesis/estimators_inf_s.json
python scripts/experiments/report_adaptive_window_bler.py results/<run_id>
```

`bler_inf_small.txt` and `stage_results_inf_small.json` in this bundle are the
run the table above is taken from. The stage results have their per-batch
`paired_samples` stripped for size; the summary retains everything the table
needs.
