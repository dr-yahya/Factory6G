# Resource Manager BER Comparison

Generated at: 2026-05-01T07:43:07.496910+00:00

This report ranks resource managers by mean BER. BER upper confidence is used as the second reliability key, followed by throughput, latency, runtime, and power for engineering interpretation.

## Acceptance Check

- Rayleigh: `ber_drl` does not beat best baseline `drl` (ber_drl BER=1.1490e-07, baseline BER=0).
- UMI/TR38901: `ber_drl` does not beat best baseline `max_throughput` (ber_drl BER=1.6874e-04, baseline BER=9.8066e-05).

## Ranked Results

### Rayleigh

#### Baseline Table

| rank | method | type | BER | BER upper | throughput/batch | latency ms | runtime sec | avg power W |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | drl | baseline | 0 | 7.6185e-09 | 1.9661e+05 | 635.467 | 1.5653e+04 | 2.9341e-04 |
| 2 | queue_aware | baseline | 0 | 7.6185e-09 | 1.9661e+05 | 635.5 | 1.5435e+04 | 2.9339e-04 |
| 3 | max_throughput | baseline | 0 | 7.6185e-09 | 1.9661e+05 | 635.795 | 1.5435e+04 | 2.9326e-04 |
| 4 | pf | baseline | 0 | 7.6185e-09 | 1.9661e+05 | 635.922 | 1.5437e+04 | 2.9320e-04 |
| 5 | round_robin | baseline | 0 | 7.6185e-09 | 1.9661e+05 | 636.399 | 1.5347e+04 | 2.9298e-04 |
| 6 | wmmse | baseline | 0 | 7.6185e-09 | 1.9661e+05 | 636.649 | 1.5579e+04 | 2.9286e-04 |
| 8 | static | baseline | 1.0321e-05 | 1.0474e-05 | 3.9321e+05 | 643.637 | 1.4194e+04 | 2.9002e-04 |

#### Trained-Model Table

| rank | method | type | BER | BER upper | throughput/batch | latency ms | runtime sec | avg power W |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 7 | ber_drl | trained | 1.1490e-07 | 1.3063e-07 | 1.9661e+05 | 506.994 | 1.2589e+04 | 3.6776e-04 |

BER plot: `reports/weekly/2026-05-01/assets/rm_baseline_rayleigh_ber_vs_ebno.png`

BER plot: `reports/weekly/2026-05-01/assets/rm_ber_drl_rayleigh_ber_vs_ebno.png`

### UMI/TR38901

#### Baseline Table

| rank | method | type | BER | BER upper | throughput/batch | latency ms | runtime sec | avg power W |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | max_throughput | baseline | 9.8066e-05 | 9.9150e-05 | 1.9659e+05 | 686.431 | 1.3819e+04 | 2.7429e-04 |
| 2 | drl | baseline | 1.5803e-04 | 1.5960e-04 | 1.9658e+05 | 695.444 | 1.3386e+04 | 2.7068e-04 |
| 3 | wmmse | baseline | 1.6510e-04 | 1.6677e-04 | 1.9658e+05 | 698.374 | 1.3222e+04 | 2.6958e-04 |
| 5 | queue_aware | baseline | 1.7731e-04 | 1.7913e-04 | 1.9657e+05 | 703.082 | 1.2950e+04 | 2.6798e-04 |
| 6 | pf | baseline | 1.7941e-04 | 1.8124e-04 | 1.9657e+05 | 703.724 | 1.2904e+04 | 2.6790e-04 |
| 7 | round_robin | baseline | 1.7998e-04 | 1.8182e-04 | 1.9657e+05 | 703.112 | 1.2804e+04 | 2.6813e-04 |
| 8 | static | baseline | 5.4832e-04 | 5.5410e-04 | 3.9300e+05 | 761.261 | 9236.27 | 2.4770e-04 |

#### Trained-Model Table

| rank | method | type | BER | BER upper | throughput/batch | latency ms | runtime sec | avg power W |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 4 | ber_drl | trained | 1.6874e-04 | 1.7042e-04 | 1.9657e+05 | 508.849 | 1.0089e+04 | 3.6647e-04 |

BER plot: `reports/weekly/2026-05-01/assets/rm_baseline_umi_ber_vs_ebno.png`

BER plot: `reports/weekly/2026-05-01/assets/rm_ber_drl_umi_ber_vs_ebno.png`

## Research Anchors

The hybrid BER-first policy follows the wireless resource-management literature: use optimization or heuristic policies to create strong labels, then train a neural policy for low-latency inference and reliability-aware adaptation.

- WMMSE baseline: [Shi et al., 2011](https://doi.org/10.1109/TSP.2011.2147784)
- DNN approximation of wireless optimization: [Sun et al., 2017/2018](https://arxiv.org/abs/1705.09412)
- Deep learning for physical-layer reliability/BER framing: [O'Shea and Hoydis, 2017](https://arxiv.org/abs/1702.00832)
- DRL scheduling with buffer/state features: [Bansbach et al., 2021](https://arxiv.org/abs/2108.12198)
- URLLC reliability/error-probability resource allocation: [Sun et al., 2019](https://doi.org/10.1109/TWC.2018.2880907)
