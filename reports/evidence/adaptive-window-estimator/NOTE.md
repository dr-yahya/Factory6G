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

## Caveats

* **This is NMSE, not BLER.** Earlier work in this project established that NMSE
  does not predict coded BER (`../estimator-floor-tr38901/`). The estimator's
  thesis claim needs a BLER-level run across the factory sweep; that is next and
  is not yet done.
* **Declared error variance is imperfect for both estimators.** On the small
  hall the adaptive window declares 0.45–0.69 of its true error and fixed DFT
  0.28–0.61; on the large hall at high SNR both over-declare by four to eight
  times. The two are close enough at each point that the comparison is not
  confounded, but neither is calibrated, and that should be said when the
  calibration table is presented.

## Reproducing

```bash
python scripts/experiments/adaptive_window_sweep.py config/thesis/estimators_inf_s.json
```
