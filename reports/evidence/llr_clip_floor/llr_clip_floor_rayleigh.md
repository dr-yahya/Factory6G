# LLR Clipping And The High-SNR BER Floor

Channel `rayleigh`, estimator `ls`, 80 batches x 20 per Eb/No point (6,400 codewords per point).

## Why this experiment exists

The receiver clipped demapper LLRs hard at +/-20, justified by a diagnostic noting that 27.5% of LLRs exceeded 50 in magnitude. At high Eb/No a large share of LLRs are legitimately large; saturating them discards the reliability information belief propagation needs to correct the remaining weak bits, precisely where the waterfall should be steepest.

Every clip setting below decodes the *same* channel realisations, noise draws and source bits (common random numbers), so the differences are not contaminated by Monte Carlo variance and the paired intervals are exact.

## Measured BER

| Eb/No (dB) | clip 20 | clip 200 | clip none |
|---|---|---|---|
| 0 | 8.466e-03 | 8.466e-03 | 8.466e-03 |
| 2 | 6.550e-04 | 6.550e-04 | 6.550e-04 |
| 4 | 9.186e-05 | 9.186e-05 | 9.186e-05 |
| 6 | < 3.05e-07 | < 3.05e-07 | < 3.05e-07 |
| 8 | < 3.05e-07 | < 3.05e-07 | < 3.05e-07 |
| 10 | < 3.05e-07 | < 3.05e-07 | < 3.05e-07 |
| 12 | < 3.05e-07 | < 3.05e-07 | < 3.05e-07 |
| 14 | < 3.05e-07 | < 3.05e-07 | < 3.05e-07 |
| 16 | < 3.05e-07 | < 3.05e-07 | < 3.05e-07 |
| 18 | < 3.05e-07 | < 3.05e-07 | < 3.05e-07 |
| 20 | < 3.05e-07 | < 3.05e-07 | < 3.05e-07 |

## Tail behaviour

| Clip | Tail slope (decades/dB) | Floored? | Basis |
|---|---|---|---|
| 20 | — | no | no errors observed in the tail (error-free to the evidence limit) |
| 200 | — | no | no errors observed in the tail (error-free to the evidence limit) |
| none | — | no | no errors observed in the tail (error-free to the evidence limit) |

A working link's BER falls steadily with Eb/No, so a tail slope near zero means the curve has stopped responding to SNR -- an error floor.

## Paired difference against the historical +/-20 clip

| Eb/No (dB) | Clip | Mean BER delta | 95% CI | Significant |
|---|---|---|---|---|
| 0 | 200 | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 2 | 200 | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 4 | 200 | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 6 | 200 | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 8 | 200 | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 10 | 200 | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 12 | 200 | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 14 | 200 | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 16 | 200 | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 18 | 200 | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 20 | 200 | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 0 | none | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 2 | none | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 4 | none | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 6 | none | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 8 | none | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 10 | none | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 12 | none | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 14 | none | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 16 | none | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 18 | none | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 20 | none | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |

A negative delta means the wider clip produced *fewer* errors on the same channel realisation. An interval excluding zero is a statistically significant difference.
