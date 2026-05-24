# Resource Manager BER Comparison

Generated at: 2026-05-24T06:45:06.404402+00:00

This report ranks resource managers by mean BER. BER upper confidence is used as the second reliability key, followed by throughput, latency, runtime, and power for engineering interpretation.

## Acceptance Check

- Rayleigh: `ber_drl` matches best baseline `queue_aware` (ber_drl BER=0, baseline BER=0).
- Rician: `ber_drl` does not beat best baseline `max_throughput` (ber_drl BER=1.8200e-05, baseline BER=0).
- UMI/TR38901: `ber_drl` does not beat best baseline `max_throughput` (ber_drl BER=2.6042e-05, baseline BER=0).

## Ranked Results

### Rayleigh

#### Baseline Table

| rank | method | type | BER | BER upper | throughput/batch | latency ms | runtime sec | avg power W |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | queue_aware | baseline | 0 | 2.4379e-06 | 6.1440e+04 | 277.206 | 65.8869 | 6.7263e-04 |
| 2 | drl | baseline | 0 | 2.4379e-06 | 6.1440e+04 | 277.239 | 66.8741 | 6.7255e-04 |
| 3 | pf | baseline | 0 | 2.4379e-06 | 6.1440e+04 | 277.348 | 65.8809 | 6.7229e-04 |
| 5 | round_robin | baseline | 0 | 2.4379e-06 | 6.1440e+04 | 278.066 | 65.7909 | 6.7055e-04 |
| 6 | max_throughput | baseline | 0 | 2.4379e-06 | 6.1440e+04 | 278.249 | 66.1102 | 6.7014e-04 |
| 7 | wmmse | baseline | 0 | 2.4379e-06 | 6.1440e+04 | 279.768 | 66.8028 | 6.6647e-04 |
| 8 | static | baseline | 8.3970e-06 | 1.0473e-05 | 1.2288e+05 | 279.746 | 66.0563 | 6.6669e-04 |

#### Trained-Model Table

| rank | method | type | BER | BER upper | throughput/batch | latency ms | runtime sec | avg power W |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 4 | ber_drl | trained | 0 | 2.4379e-06 | 6.1440e+04 | 277.563 | 66.9467 | 6.7176e-04 |

BER plot: `results/20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s/rayleigh/resource_managers/ber_vs_ebno.png`

### Rician

#### Baseline Table

| rank | method | type | BER | BER upper | throughput/batch | latency ms | runtime sec | avg power W |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | max_throughput | baseline | 0 | 2.4379e-06 | 6.1440e+04 | 276.114 | 65.6089 | 6.7530e-04 |
| 3 | wmmse | baseline | 2.5006e-05 | 2.9561e-05 | 6.1438e+04 | 278.331 | 66.5694 | 6.6991e-04 |
| 4 | drl | baseline | 3.8397e-05 | 4.3559e-05 | 6.1438e+04 | 275.569 | 66.4974 | 6.7663e-04 |
| 5 | round_robin | baseline | 4.7275e-05 | 5.2668e-05 | 6.1437e+04 | 275.994 | 65.2826 | 6.7558e-04 |
| 6 | pf | baseline | 4.8236e-05 | 5.3661e-05 | 6.1437e+04 | 275.884 | 65.5379 | 6.7585e-04 |
| 7 | queue_aware | baseline | 5.1270e-05 | 5.6790e-05 | 6.1437e+04 | 275.014 | 65.4156 | 6.7799e-04 |
| 8 | static | baseline | 5.8901e-04 | 6.0137e-04 | 1.2281e+05 | 277.493 | 65.5541 | 6.7203e-04 |

#### Trained-Model Table

| rank | method | type | BER | BER upper | throughput/batch | latency ms | runtime sec | avg power W |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 2 | ber_drl | trained | 1.8200e-05 | 2.2427e-05 | 6.1439e+04 | 275.604 | 66.5267 | 6.7654e-04 |

BER plot: `results/20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s/rician/resource_managers/ber_vs_ebno.png`

### UMI/TR38901

#### Baseline Table

| rank | method | type | BER | BER upper | throughput/batch | latency ms | runtime sec | avg power W |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | max_throughput | baseline | 0 | 2.4379e-06 | 6.1440e+04 | 276.073 | 65.5901 | 6.7556e-04 |
| 3 | queue_aware | baseline | 2.6190e-05 | 3.1273e-05 | 6.1438e+04 | 275.102 | 65.413 | 6.7790e-04 |
| 4 | wmmse | baseline | 2.7817e-05 | 3.2495e-05 | 6.1438e+04 | 279.163 | 66.6757 | 6.6809e-04 |
| 5 | round_robin | baseline | 2.8853e-05 | 3.3607e-05 | 6.1438e+04 | 276.009 | 65.2934 | 6.7572e-04 |
| 6 | drl | baseline | 5.8890e-05 | 6.6504e-05 | 6.1436e+04 | 275.424 | 66.4446 | 6.7712e-04 |
| 7 | pf | baseline | 7.1245e-05 | 7.8704e-05 | 6.1436e+04 | 276.714 | 65.7322 | 6.7402e-04 |
| 8 | static | baseline | 2.9615e-04 | 3.0520e-04 | 1.2284e+05 | 276.084 | 65.2061 | 6.7551e-04 |

#### Trained-Model Table

| rank | method | type | BER | BER upper | throughput/batch | latency ms | runtime sec | avg power W |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 2 | ber_drl | trained | 2.6042e-05 | 3.1064e-05 | 6.1438e+04 | 275.903 | 66.5623 | 6.7601e-04 |

BER plot: `results/20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s/tr38901/resource_managers/ber_vs_ebno.png`

## Research Anchors

The hybrid BER-first policy follows the wireless resource-management literature: use optimization or heuristic policies to create strong labels, then train a neural policy for low-latency inference and reliability-aware adaptation.

- WMMSE baseline: [Shi et al., 2011](https://doi.org/10.1109/TSP.2011.2147784)
- DNN approximation of wireless optimization: [Sun et al., 2017/2018](https://arxiv.org/abs/1705.09412)
- Deep learning for physical-layer reliability/BER framing: [O'Shea and Hoydis, 2017](https://arxiv.org/abs/1702.00832)
- DRL scheduling with buffer/state features: [Bansbach et al., 2021](https://arxiv.org/abs/2108.12198)
- URLLC reliability/error-probability resource allocation: [Sun et al., 2019](https://doi.org/10.1109/TWC.2018.2880907)
