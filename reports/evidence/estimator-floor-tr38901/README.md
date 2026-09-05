# Closing The TR 38.901 Estimation Floor

**Date:** 2026-09-05
**Follow-up to:** `reports/evidence/llr_clip_floor/` — which established that the
TR 38.901 BER floor is channel-estimation error, and that perfect CSI removes it
entirely.

That result turns the floor into a measurable target: perfect CSI is the lower
bound, LS is the starting point, and each estimator can be scored by how much of
the gap it closes. This note does that, in BER and — for the first time — in
NMSE.

## Setup

TR 38.901 UMi, QPSK, rate 1/2, 4 users, 8 BS antennas, fft 128.
40 batches x 20 per Eb/No point = 4,915,200 information bits per point.
Shared batch contexts, so every estimator sees the identical channel and noise.

## BER

| Eb/No (dB) | LS | DFT | LMMSE | Adaptive | Perfect |
|---|---|---|---|---|---|
| 6  | 2.364e-04 | **0 err** | 6.612e-05 | 6.795e-05 | 0 err |
| 8  | 3.298e-04 | 1.597e-04 | 2.830e-04 | 1.971e-04 | 0 err |
| 10 | 2.651e-04 | 4.069e-07 | 1.180e-04 | **0 err** | 0 err |
| 12 | 2.846e-04 | 2.035e-07 | 8.138e-07 | 2.035e-07 | 0 err |
| 14 | 2.879e-04 | 3.866e-06 | 7.792e-05 | 3.662e-06 | 0 err |
| 16 | 2.079e-04 | 1.017e-06 | 9.583e-05 | 8.138e-07 | 0 err |
| 18 | 5.570e-04 | 6.104e-07 | 8.179e-05 | **0 err** | 0 err |
| 20 | 9.318e-05 | 8.138e-07 | 7.426e-05 | **0 err** | 0 err |

"0 err" is BER < 6.1e-7 at 95% confidence.

## NMSE (dB)

| Eb/No (dB) | LS | DFT | LMMSE | Adaptive |
|---|---|---|---|---|
| 6  | -10.23 | -11.42 | -13.96 | **-14.03** |
| 8  | -11.94 | -12.27 | -15.24 | **-15.62** |
| 10 | -13.40 | -12.97 | **-16.28** | -15.85 |
| 12 | -14.90 | -13.51 | **-17.62** | -15.25 |
| 14 | -16.09 | -13.87 | **-18.68** | -15.55 |
| 16 | -17.19 | -14.08 | **-19.71** | -15.75 |
| 18 | -17.86 | -14.23 | **-20.21** | -15.91 |
| 20 | -18.62 | -14.39 | **-20.75** | -16.01 |

Perfect CSI has no estimation error, so no NMSE.

## What the data says

**The floor is closable, and the estimator work closes most of it.** LS sits at
2-3e-4 across the entire sweep. DFT and the adaptive hybrid reach 1e-6 or below
from 10 dB — roughly two and a half orders of magnitude of the gap to
perfect CSI. This is the clearest quantitative case yet for the estimator
contribution.

**Adaptive is the best performer overall.** It matches DFT in the tail, reaches
zero observed errors at 10, 18 and 20 dB, and has the best NMSE of any estimator
at low SNR (6-8 dB), where its LMMSE branch is active. That is the behaviour the
hybrid is designed to produce.

**LMMSE has the best NMSE and a much worse BER than DFT.** This is the finding
that needs following up before any of it is written up.

At 20 dB, LMMSE's NMSE is -20.75 dB against DFT's -14.39 dB — over 6 dB better —
yet its BER is 7.4e-5 against DFT's 8.1e-7, about ninety times worse. LS shows
the same inversion in weaker form: better NMSE than DFT from 10 dB up, far worse
BER throughout.

A lower mean-squared error producing worse decisions means MSE is not capturing
what matters here. The plausible mechanisms, in the order worth testing:

1. **Error structure, not error size.** DFT truncation discards delay taps beyond
   the cyclic prefix, which introduces a smooth, correlated bias but removes
   white noise across every subcarrier. LMMSE leaves a smaller but noisier
   residual. LDPC decoding over a whole codeword tolerates smooth bias far
   better than per-subcarrier noise, so the estimator with the larger MSE can
   still hand the decoder better LLRs.
2. **Mis-declared `err_var`.** Each estimator reports an error variance that the
   LMMSE equalizer folds into the effective noise. DFT scales it by
   `tap_count / fft_size`, LMMSE by the mean shrinkage. If either misstates its
   true error, the equalizer weights the estimate wrongly and the BER moves for
   reasons unrelated to estimation accuracy. This is the cheapest to check:
   compare each estimator's declared `err_var` against its measured squared
   error.
3. **LMMSE's correlation model is mismatched.** `r_freq = 0.98` assumes a
   specific frequency correlation. If the true TR 38.901 UMi profile is less
   correlated, the smoother over-smooths — which can lower MSE while flattening
   genuine channel structure the equalizer relies on.

Until this is resolved, **do not present NMSE and BER as interchangeable
measures of estimator quality in the thesis.** Report both, and treat the
divergence as a finding: it is evidence that estimator design for coded
multi-user OFDM cannot be driven by MSE alone.

## Mechanism 2 partly tested: every estimator understates its error

The cheapest check has been run. Each estimator's declared `err_var` was
compared against its measured squared error on the same channel realisations:

| Eb/No | Estimator | Declared err_var | Measured MSE | Declared / measured |
|---|---|---|---|---|
| 10 | LS | 3.372e-02 | 4.172e-02 | 0.81 |
| 10 | DFT | 5.269e-03 | 4.812e-02 | **0.11** |
| 10 | LMMSE | 8.819e-03 | 2.032e-02 | 0.43 |
| 10 | Adaptive | 6.988e-03 | 2.396e-02 | 0.29 |
| 20 | LS | 3.372e-03 | 2.200e-02 | 0.15 |
| 20 | DFT | 5.269e-04 | 3.659e-02 | **0.014** |
| 20 | LMMSE | 2.185e-03 | 1.339e-02 | 0.16 |
| 20 | Adaptive | 8.536e-04 | 2.550e-02 | 0.033 |

**Every estimator understates its own error, and by very different factors.** At
20 dB, DFT declares 1/71st of its actual error while LMMSE declares 1/6th. Since
the LMMSE equalizer folds `err_var` into the effective noise, a smaller declared
value produces a smaller `no_eff` and therefore larger-magnitude LLRs. DFT is
handed roughly a twelvefold LLR advantage over LMMSE for reasons that have
nothing to do with how accurate its channel estimate is.

The BER comparison between estimators is therefore **confounded**: it measures
estimation accuracy convolved with each estimator's optimism about itself. Note
also that DFT's measured MSE (3.66e-2 at 20 dB) is *worse* than plain LS
(2.20e-2), independently confirming the NMSE ordering.

Re-decoding with `err_var` replaced by each estimator's measured squared error —
so all four are equally honest — moved LS from 6.9e-4 to 1.8e-4 at 20 dB, a
fourfold improvement from the correction alone. The other three showed zero
errors both ways, but that arm used only 384 codewords (590k bits, so a floor of
BER < 5e-6) and cannot discriminate at the 1e-5 level where the question lives.

**So: the confound is real and measurably affects BER, but this sample does not
settle whether it explains the whole NMSE/BER inversion.** Repeating the honest-
`err_var` comparison at the full 4.9M bits per point is the next step, and it is
now a well-posed question rather than a hypothesis.

A further caution for the thesis: at these error rates the BER is dominated by
rare unfavourable channel realisations. LMMSE measured 7.4e-5 over 800
realisations in the main run and zero errors over 96 here. Small samples will
mislead, which is exactly what the paired-bootstrap machinery in
`sim/stages/common.py` exists to guard against — use it for these comparisons.

## Suggested next experiments

1. ~~Compare declared `err_var` against measured squared error.~~ **Done** — see
   above. Every estimator understates, DFT by 71x. Repeat the honest-`err_var`
   re-decode at full sample size to finish the job.
2. Sweep LMMSE's `r_freq` against the measured coherence bandwidth of the
   TR 38.901 UMi profile in use (mechanism 3).
3. Decompose each estimator's error into bias and variance across frequency
   (mechanism 1).
4. Repeat on the InF channels, where the delay profile — and therefore the DFT
   truncation window — differs from UMi.

## Caveat

TensorFlow 2.21 / Sionna 1.2.1 in a local virtualenv, not the project's Docker
image (TensorFlow 2.15). Regenerate in the container before quoting numbers.
