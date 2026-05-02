# Resource Manager BER Comparison

Generated at: 2026-04-28T10:09:48.371266+00:00

This report ranks resource managers by mean BER. BER upper confidence is used as the second reliability key, followed by throughput, latency, runtime, and power for engineering interpretation.

## Acceptance Check

- Rayleigh: no `ber_drl` benchmark row found yet; no trained-model improvement claim is made.
- UMI/TR38901: no `ber_drl` benchmark row found yet; no trained-model improvement claim is made.

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
| 7 | static | baseline | 1.0321e-05 | 1.0474e-05 | 3.9321e+05 | 643.637 | 1.4194e+04 | 2.9002e-04 |

#### Trained-Model Table

No trained-model benchmark row found for this channel.

BER plot: `results/20260420_040402_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_rayleigh_qpsk_s/resource_managers/ber_vs_ebno.png`

### UMI/TR38901

#### Baseline Table

| rank | method | type | BER | BER upper | throughput/batch | latency ms | runtime sec | avg power W |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | max_throughput | baseline | 9.8066e-05 | 9.9150e-05 | 1.9659e+05 | 686.431 | 1.3819e+04 | 2.7429e-04 |
| 2 | drl | baseline | 1.5803e-04 | 1.5960e-04 | 1.9658e+05 | 695.444 | 1.3386e+04 | 2.7068e-04 |
| 3 | wmmse | baseline | 1.6510e-04 | 1.6677e-04 | 1.9658e+05 | 698.374 | 1.3222e+04 | 2.6958e-04 |
| 4 | queue_aware | baseline | 1.7731e-04 | 1.7913e-04 | 1.9657e+05 | 703.082 | 1.2950e+04 | 2.6798e-04 |
| 5 | pf | baseline | 1.7941e-04 | 1.8124e-04 | 1.9657e+05 | 703.724 | 1.2904e+04 | 2.6790e-04 |
| 6 | round_robin | baseline | 1.7998e-04 | 1.8182e-04 | 1.9657e+05 | 703.112 | 1.2804e+04 | 2.6813e-04 |
| 7 | static | baseline | 5.4832e-04 | 5.5410e-04 | 3.9300e+05 | 761.261 | 9236.27 | 2.4770e-04 |

#### Trained-Model Table

No trained-model benchmark row found for this channel.

BER plot: `results/20260420_043640_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_umi_qpsk_s/resource_managers/ber_vs_ebno.png`

## Research Anchors

The hybrid BER-first policy follows the wireless resource-management literature: use optimization or heuristic policies to create strong labels, then train a neural policy for low-latency inference and reliability-aware adaptation.

- WMMSE baseline: [Shi et al., 2011](https://doi.org/10.1109/TSP.2011.2147784)
- DNN approximation of wireless optimization: [Sun et al., 2017/2018](https://arxiv.org/abs/1705.09412)
- Deep learning for physical-layer reliability/BER framing: [O'Shea and Hoydis, 2017](https://arxiv.org/abs/1702.00832)
- DRL scheduling with buffer/state features: [Bansbach et al., 2021](https://arxiv.org/abs/2108.12198)
- URLLC reliability/error-probability resource allocation: [Sun et al., 2019](https://doi.org/10.1109/TWC.2018.2880907)
