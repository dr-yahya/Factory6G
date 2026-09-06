# LLR Clipping And The High-SNR BER Floor

Channel `tr38901`, estimator `ls`, 60 batches x 20 per Eb/No point (4,800 codewords per point).

## Why this experiment exists

The receiver clipped demapper LLRs hard at +/-20, justified by a diagnostic noting that 27.5% of LLRs exceeded 50 in magnitude. At high Eb/No a large share of LLRs are legitimately large; saturating them discards the reliability information belief propagation needs to correct the remaining weak bits, precisely where the waterfall should be steepest.

Every clip setting below decodes the *same* channel realisations, noise draws and source bits (common random numbers), so the differences are not contaminated by Monte Carlo variance and the paired intervals are exact.

## Measured BER

| Eb/No (dB) | clip 20 | clip 200 | clip none |
|---|---|---|---|
| 6 | 3.159e-04 | 3.159e-04 | 3.159e-04 |
| 8 | 2.230e-04 | 2.230e-04 | 2.230e-04 |
| 10 | 2.081e-04 | 2.081e-04 | 2.081e-04 |
| 12 | 2.364e-04 | 2.364e-04 | 2.364e-04 |
| 14 | 2.664e-04 | 2.664e-04 | 2.664e-04 |
| 16 | 1.298e-04 | 1.298e-04 | 1.298e-04 |
| 18 | 2.620e-04 | 2.620e-04 | 2.620e-04 |
| 20 | 6.700e-05 | 6.700e-05 | 6.700e-05 |

## Tail behaviour

| Clip | Tail slope (decades/dB) | Floored? | Basis |
|---|---|---|---|
| 20 | -0.075 | no | fitted over the upper half of the sweep |
| 200 | -0.075 | no | fitted over the upper half of the sweep |
| none | -0.075 | no | fitted over the upper half of the sweep |

A working link's BER falls steadily with Eb/No, so a tail slope near zero means the curve has stopped responding to SNR -- an error floor.

## Paired difference against the historical +/-20 clip

| Eb/No (dB) | Clip | Mean BER delta | 95% CI | Significant |
|---|---|---|---|---|
| 6 | 200 | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 8 | 200 | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 10 | 200 | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 12 | 200 | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 14 | 200 | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 16 | 200 | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 18 | 200 | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 20 | 200 | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 6 | none | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 8 | none | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 10 | none | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 12 | none | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 14 | none | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 16 | none | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 18 | none | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |
| 20 | none | +0.000e+00 | [+0.000e+00, +0.000e+00] | no |

A negative delta means the wider clip produced *fewer* errors on the same channel realisation. An interval excluding zero is a statistically significant difference.
