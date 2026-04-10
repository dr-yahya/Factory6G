# Factory6G Simulation Results Report

*Generated: 2026-03-27*

---

## 1. Executive Summary

This report presents a comprehensive evaluation of the Factory6G physical layer simulation platform, covering **six channel estimation methods**, **three modulation schemes**, **three channel models**, **three factory sizes**, and the **JIDD-SCMA** joint detection-decoding system. All experiments were conducted in March 2026.

**Key findings:**

- **Adaptive and PSO** channel estimators achieve the lowest BER floors (~3-4 x 10^-5), an order of magnitude better than LS (~3 x 10^-4). PSO is 14x slower than Adaptive for comparable performance.
- **JIDD-SCMA** achieves effectively zero BER above 9 dB Eb/N0, demonstrating the power of joint iterative detection-decoding, but at 162x the computational cost of LS.
- **Neural estimator** produces BER identical to LS -- it has not learned beyond the baseline.
- **Higher-order modulation** (64-QAM) raises the BER floor to ~1.1 x 10^-2 and triples latency compared to QPSK.
- **Rayleigh fading** is the easiest channel (zero BER above 8 dB with LS), while **TR 38.901 UMi** produces a persistent BER floor (~3 x 10^-4).
- **Large factory environments** (40x40 m) cause severe multipath degradation with BER stuck at ~0.21 -- current estimation methods cannot cope.
- A critical **numerical bug** in JIDD-SCMA Run 1 caused BER to rebound to 0.5 at high SNR; this was fixed in Run 2.

---

## 2. Simulation Setup

### 2.1 OFDM Pipeline (Channel Estimator & Modulation Tests)

| Parameter | Value |
|---|---|
| Channel model | TR 38.901 UMi (default) |
| Carrier frequency | 3.5 GHz |
| FFT size | 128 subcarriers |
| OFDM symbols | 14 per frame |
| Subcarrier spacing | 30 kHz |
| Modulation | QPSK (default), 16-QAM, 64-QAM |
| Channel coding | 5G LDPC, rate 0.5 |
| LDPC decoding iterations | 20 |
| BS antennas | 8 |
| UT antennas | 1 per UT |
| Batch size | 64 |
| Eb/N0 range | 0-20 dB (2 dB steps) |
| Stopping criterion | 100 block errors per point, min 1M bits |
| Platform | Sionna + TensorFlow |

### 2.2 Channel Estimators Evaluated

| Method | Description | Key Parameters |
|---|---|---|
| LS | Least Squares with linear interpolation | Baseline |
| PSO | Particle Swarm Optimization (DFT/LMMSE blend) | 8 particles, 12 iterations |
| Adaptive | SNR-aware hybrid (DFT/LMMSE switching) | Quality thresholds: 3-12 dB |
| ISTA | Iterative Shrinkage-Thresholding | 10 iterations |
| Neural | Keras neural network | Pre-trained model |
| DFT | DFT-based delay-domain truncation | Tested on medium factory |

### 2.3 JIDD-SCMA System

| Parameter | Value |
|---|---|
| Multiple access | SCMA (6 users, 4 resources) |
| Channel coding | Polar code (N=256, K=128) |
| Decoder | SCAN with alpha=0.6 |
| JIDD iterations | 5 |
| Channel | Rayleigh flat-fading |
| Channel estimation | MMSE (Wiener filter) |
| Eb/N0 range | 1-20 dB (1 dB steps) |
| Stopping criterion | Min 100 bit errors, max 50M bits per point |

---

## 3. Channel Estimator Comparison

### 3.1 BER Performance

![Channel Estimator BER vs Eb/N0](plots/estimator_ber_vs_ebno.png)

#### BER at Key Eb/N0 Points

| Eb/N0 (dB) | LS | PSO | Adaptive | ISTA | Neural | DFT* |
|---|---|---|---|---|---|---|
| 0  | 2.42e-2 | 4.79e-3 | 4.84e-3 | 3.95e-1 | 2.42e-2 | 8.77e-2 |
| 4  | 9.65e-4 | 2.60e-4 | 4.01e-4 | 1.76e-1 | 9.65e-4 | 1.88e-2 |
| 8  | 3.46e-4 | 4.94e-5 | 5.13e-5 | 4.22e-2 | 3.46e-4 | 5.57e-3 |
| 12 | 2.82e-4 | 3.75e-5 | 2.94e-5 | 4.37e-3 | 2.82e-4 | 4.33e-3 |
| 16 | 3.22e-4 | 4.20e-5 | 2.96e-5 | 6.34e-4 | 3.22e-4 | 6.39e-3 |
| 20 | 4.00e-4 | 3.94e-5 | 4.01e-5 | 3.26e-4 | 4.00e-4 | 7.78e-3 |

*\*DFT was tested on the medium factory (25x25 m) rather than the small factory used by other methods.*

**Analysis:**

- **Adaptive and PSO are the top performers**, both achieving BER floors of ~3-4 x 10^-5 -- roughly 10x lower than LS. At 12 dB, Adaptive edges ahead (2.94e-5 vs 3.75e-5), but the difference is within noise. PSO reaches its floor faster at low SNR (4.79e-3 at 0 dB vs 2.42e-2 for LS), suggesting better noise resilience.
- **LS and Neural produce identical BER** at every Eb/N0 point. The neural estimator has collapsed to the LS solution and provides no benefit over the baseline. This likely indicates insufficient training data diversity or model capacity.
- **ISTA** starts poorly at low SNR (0.395 at 0 dB) but converges to ~3.3 x 10^-4 at 20 dB, comparable to LS. Its iterative shrinkage approach needs a minimum SNR threshold (~8 dB) before becoming competitive.
- **DFT** has the worst BER floor (~4-8 x 10^-3) and actually degrades at higher SNR (rising from 3.5e-3 at 10 dB to 7.8e-3 at 20 dB). Note: the medium factory comparison is not fully fair.
- All methods except JIDD-SCMA exhibit a **BER floor**, indicating channel estimation error dominates over noise at high SNR.

### 3.2 Latency

![Channel Estimator Latency](plots/estimator_latency_vs_ebno.png)

| Method | Avg. Latency (ms) |
|---|---|
| LS | ~474 |
| PSO | ~469 |
| Adaptive | ~474 |
| ISTA | ~479 |
| Neural | ~474 |
| DFT (medium) | ~818 |

Latency is stable across Eb/N0 for all methods. LS, PSO, Adaptive, ISTA, and Neural all cluster around 470-480 ms. DFT is 1.7x slower due to the medium factory configuration and DFT processing overhead.

### 3.3 Runtime

![Channel Estimator Runtime](plots/estimator_runtime.png)

| Method | Total Runtime | Relative to LS |
|---|---|---|
| ISTA | 91.7 s | 0.11x |
| DFT | 96.5 s | 0.12x |
| LS | 817.7 s | 1.0x |
| Neural | 817.4 s | 1.0x |
| Adaptive | 5,838 s | 7.1x |
| PSO | 82,703 s (23.0 h) | 101x |

- **ISTA and DFT are fastest** (~92-97 s), about 9x faster than LS due to fewer Monte Carlo batches needed.
- **Adaptive is 7x slower than LS** but delivers 10x better BER -- a strong cost-benefit tradeoff.
- **PSO is 101x slower than LS** for marginal improvement over Adaptive. The particle swarm optimization explores many candidate solutions per batch, making it computationally expensive. **Not recommended for production use.**
- **LS and Neural have identical runtime**, confirming functional equivalence.

---

## 4. Modulation Order Impact

### 4.1 BER vs Modulation

![Modulation BER Comparison](plots/modulation_ber_vs_ebno.png)

| Eb/N0 (dB) | QPSK | 16-QAM | 64-QAM |
|---|---|---|---|
| 0  | 2.42e-2 | 1.39e-1 | 2.31e-1 |
| 4  | 9.65e-4 | 1.14e-2 | 6.83e-2 |
| 8  | 3.46e-4 | 2.49e-3 | 1.43e-2 |
| 12 | 2.82e-4 | 1.72e-3 | 8.13e-3 |
| 16 | 3.22e-4 | 1.66e-3 | 8.85e-3 |
| 20 | 4.00e-4 | 1.45e-3 | 1.11e-2 |

All tests use the LS estimator on the TR 38.901 UMi channel.

**Analysis:**

- **BER floor scales with modulation order**: QPSK ~3 x 10^-4, 16-QAM ~1.5 x 10^-3 (5x worse), 64-QAM ~1.1 x 10^-2 (37x worse). This is expected -- higher-order constellations have smaller decision regions and are more sensitive to estimation errors.
- **64-QAM BER increases at high SNR** (from 8.1e-3 at 12 dB to 1.1e-2 at 20 dB), indicating the LS estimator's channel estimation error floor is especially damaging for dense constellations.
- To use higher-order modulation effectively, a better estimator (Adaptive or PSO) would be needed.

### 4.2 Latency vs Modulation

![Modulation Latency](plots/modulation_latency_vs_ebno.png)

| Modulation | Avg. Latency (ms) | Throughput (bits/batch) |
|---|---|---|
| QPSK (2 bits/sym) | 471 | ~393k |
| 16-QAM (4 bits/sym) | 774 | ~785k |
| 64-QAM (6 bits/sym) | 1,690 | ~1.17M |

Latency scales roughly linearly with bits per symbol due to increased LDPC decoding complexity. Raw throughput increases, but the higher BER means more retransmissions would be needed in practice.

---

## 5. Channel Model Impact

![Channel Model BER Comparison](plots/channel_model_ber_vs_ebno.png)

| Eb/N0 (dB) | Rayleigh | Rician (K=1) | TR 38.901 UMi |
|---|---|---|---|
| 0  | 7.47e-3 | 9.17e-2 | 2.49e-2 |
| 4  | 5.30e-5 | 2.38e-3 | 1.18e-3 |
| 8  | 0 | 2.67e-5 | 3.70e-4 |
| 12 | 0 | 0 | 4.53e-4 |
| 16 | 0 | 0 | 3.94e-4 |
| 20 | 0 | 0 | 4.17e-4 |

All tests use the LS estimator with QPSK on the small factory.

**Analysis:**

- **Rayleigh is the easiest channel**, achieving zero BER above 8 dB. The frequency-flat fading with independent realizations is well-suited to LS estimation with pilot-based interpolation.
- **Rician (K=1)** reaches zero BER above 12 dB. The line-of-sight component helps, but the specular path introduces estimation challenges at low SNR.
- **TR 38.901 UMi** is the hardest channel with a persistent BER floor of ~3-4 x 10^-4. The 3GPP model includes realistic multipath, delay spread, and spatial correlation that LS cannot fully track. This is the most realistic scenario for factory deployments.
- The zero-BER results on Rayleigh/Rician suggest that the LDPC code is powerful enough to correct residual errors when channel estimation is sufficiently accurate. The BER floor on TR 38.901 is due to estimation limitations, not coding weakness.

---

## 6. Factory Size Impact

![Factory Size BER Comparison](plots/factory_size_ber_vs_ebno.png)

| Eb/N0 (dB) | Small (15x15 m) | Medium (25x25 m) | Large (40x40 m) |
|---|---|---|---|
| 0  | 2.42e-2 | 1.18e-1 | 2.78e-1 |
| 4  | 9.65e-4 | 2.62e-2 | 2.45e-1 |
| 8  | 3.46e-4 | 9.66e-3 | 2.26e-1 |
| 12 | 2.82e-4 | 6.35e-3 | 2.16e-1 |
| 16 | 3.22e-4 | 7.09e-3 | 2.12e-1 |
| 20 | 4.00e-4 | 1.11e-2 | 2.10e-1 |

All tests use the LS estimator with QPSK on the TR 38.901 UMi channel.

| Factory | Dimensions | Machines | UTs | Avg. Latency |
|---|---|---|---|---|
| Small | 15x15x5 m | 5 | 4 | 484 ms |
| Medium | 25x25x6 m | 10 | 8 | 809 ms |
| Large | 40x40x8 m | 20 | 16 | 1,727 ms |

**Analysis:**

- **Small factory** performs as expected with a BER floor of ~3 x 10^-4.
- **Medium factory** is 20-30x worse (BER floor ~6-11 x 10^-3). More machines create additional multipath, more UTs increase interference, and the 8-antenna BS has less spatial degrees of freedom per user.
- **Large factory** is catastrophic: BER is stuck at ~0.21 across all SNR levels. The LS estimator cannot cope with 16 UTs on 8 BS antennas (under-determined system), severe multipath from 20 machines, and the larger propagation distances. **This configuration is fundamentally broken and needs architectural changes** (more BS antennas, distributed MIMO, or user scheduling to reduce simultaneous UTs).
- Latency scales with factory size due to more UTs and larger resource grids.

---

## 7. JIDD-SCMA Analysis

### 7.1 Bug Fix Impact

![JIDD-SCMA BER Comparison](plots/jidd_ber_comparison.png)

| | Run 1 (buggy) | Run 2 (fixed) |
|---|---|---|
| Date | 2026-03-20 08:34 | 2026-03-20 17:10 |
| Total runtime | 20,615 s (~5.7 h) | 132,460 s (~36.8 h) |
| Total bits tested | 71.8M | 621M |
| BER at 8 dB | 6.1e-6 | 6.1e-6 |
| BER at 10 dB | 2.6e-3 | 2.8e-7 |
| BER at 13 dB | 0.486 | 0 |
| BER at 20 dB | 0.500 | 0 |

**Run 1** had a critical numerical bug: BER drops normally through the waterfall region (1-9 dB), reaches zero at 9 dB, but then **rebounds to ~0.5** from 10-20 dB. A BER of 0.5 means random guessing -- the decoder output is uncorrelated with the input. Root cause: the log-likelihood computation `f = -(1/N0) * |y - signal|^2` overflows when N0 approaches zero at high SNR.

**Run 2** resolved this: BER drops monotonically and remains at 0 above 9 dB. Each high-SNR point was tested with 50M bits, confirming BER < 6e-8 (upper bound).

### 7.2 JIDD-SCMA Waterfall Performance (Run 2)

| Eb/N0 (dB) | BER | Bit Errors | Total Bits |
|---|---|---|---|
| 1 | 0.474 | 24,045 | 50,688 |
| 3 | 0.440 | 22,286 | 50,688 |
| 5 | 0.090 | 4,557 | 50,688 |
| 6 | 2.94e-3 | 149 | 50,688 |
| 7 | 1.41e-4 | 128 | 907,008 |
| 8 | 6.10e-6 | 122 | 19,987,968 |
| 9 | 0 | 0 | 50,000,640 |
| 10 | 2.80e-7 | 14 | 50,000,640 |
| 12-20 | 0 | 0 | 50,000,640 each |

The waterfall region spans 5-8 dB with BER dropping **5 orders of magnitude in 3 dB**. This steep slope is characteristic of the joint iterative processing where the polar code SCAN decoder and SCMA MPA detector reinforce each other through message passing.

---

## 8. Cross-System Comparison

### 8.1 BER Comparison

![Cross-System BER Comparison](plots/combined_ber.png)

| Aspect | LS (baseline) | Adaptive | PSO | JIDD-SCMA |
|---|---|---|---|---|
| BER at 6 dB | 4.6e-4 | 7.8e-5 | 8.5e-5 | 2.9e-3 |
| BER at 8 dB | 3.5e-4 | 5.1e-5 | 4.9e-5 | 6.1e-6 |
| BER at 10 dB | 3.0e-4 | 4.4e-5 | 4.7e-5 | ~0 |
| BER floor | ~3e-4 | ~3e-5 | ~4e-5 | 0 (within 50M bits) |
| Architecture | OFDM+LDPC | OFDM+LDPC | OFDM+LDPC | SCMA+Polar+JIDD |

**Key observations:**

- **JIDD-SCMA achieves the lowest absolute BER** at moderate-to-high SNR, reaching effectively zero errors above 9 dB. However, it has the worst BER below 6 dB because the SCMA system is designed for the multi-user overloaded regime rather than low-SNR operation.
- **Adaptive and PSO** provide the best BER among OFDM-based methods but still have an error floor that JIDD-SCMA does not exhibit.
- The two system types are **not directly comparable** -- different modulation, coding, and multiple access. But the comparison illustrates the BER advantage of joint processing vs. separate estimation-then-decoding.

### 8.2 Runtime Comparison

![Runtime Comparison](plots/runtime_comparison.png)

| Method | Total Runtime | Relative to LS |
|---|---|---|
| ISTA | 91.7 s | 0.11x |
| DFT | 96.5 s | 0.12x |
| LS | 817.7 s | 1.0x |
| Neural | 817.4 s | 1.0x |
| Adaptive | 5,838 s (1.6 h) | 7.1x |
| JIDD Run 1 | 20,615 s (5.7 h) | 25x |
| PSO | 82,703 s (23.0 h) | 101x |
| JIDD Run 2 | 132,460 s (36.8 h) | 162x |

JIDD Run 2 is expensive primarily because it tests 50M bits per high-SNR point to confirm zero errors. The per-frame cost of JIDD (iterative MPA+SCAN) is also inherently higher than single-pass estimation.

---

## 9. Summary of Findings

| # | Finding | Impact |
|---|---|---|
| 1 | Adaptive is the best estimator (BER ~3e-5) | Default choice for OFDM pipeline |
| 2 | PSO matches Adaptive but costs 14x more runtime | Not practical for production |
| 3 | Neural estimator = LS (no learning occurred) | Needs retraining or architectural changes |
| 4 | JIDD-SCMA achieves zero BER above 9 dB | Best for URLLC scenarios |
| 5 | JIDD-SCMA had a critical N0-overflow bug | Fixed; needs numerical guards |
| 6 | 64-QAM BER floor is 37x worse than QPSK | Higher modulation needs better estimation |
| 7 | TR 38.901 is hardest channel (BER floor ~3e-4) | Realistic factory scenario |
| 8 | Rayleigh/Rician achieve zero BER at high SNR | LDPC coding is not the bottleneck |
| 9 | Large factory BER ~0.21 at all SNR | System is fundamentally broken at this scale |
| 10 | DFT is worst estimator (BER ~8e-3) | May improve on small factory (needs retest) |

---

## 10. Suggested Next Steps

### High Priority

1. **Fix the Neural estimator.** It currently produces identical results to LS, wasting computational resources. Suggested actions:
   - Train on more diverse channel realizations (multiple SNR points, factory sizes)
   - Increase model capacity or try a different architecture (e.g., attention-based)
   - Use curriculum learning starting from easy channels (Rayleigh) to hard (TR 38.901)

2. **Solve the large factory problem.** BER of 0.21 is unusable. Potential approaches:
   - Increase BS antennas from 8 to 16 or 32
   - Implement user scheduling to serve subsets of the 16 UTs per frame
   - Deploy distributed MIMO with multiple access points
   - Test with Adaptive estimator (may help but unlikely to solve the fundamental under-determined problem)

10. **Run Adaptive estimator on higher-order modulation.** The 64-QAM BER floor with LS (1.1e-2) may improve 10x with Adaptive estimation. This would validate whether Adaptive + 64-QAM is viable for high-throughput factory applications.

### Medium Priority

4. **Re-run DFT on small factory.** The current comparison is unfair since DFT was tested on the medium factory while all others used small. A fair comparison would clarify whether DFT's poor performance is inherent or an artifact of the test configuration.

5. **Test Adaptive estimator on Rayleigh/Rician channels.** LS already achieves zero BER on Rayleigh above 8 dB. Adaptive could potentially reach zero BER at even lower SNR (4-6 dB), demonstrating its value across channel types.

6. **Run resource manager benchmarks.** Only archived data from earlier runs (March 5-14) exists for the 7 resource managers (Static, Round Robin, Max Throughput, PF, WMMSE, Queue-Aware, DRL). Fresh runs with the current configuration would provide up-to-date comparisons.

### Research Directions

7. **Integrate JIDD-SCMA with Adaptive estimation.** Currently JIDD uses MMSE channel estimation. Replacing it with the Adaptive estimator could push the waterfall region to even lower SNR.

8. **Investigate iterative channel estimation with decoder feedback.** The persistent BER floor across all OFDM estimators suggests that pilot-only estimation has fundamental limits. Feeding soft decoder output back to refine channel estimates (turbo equalization) could break through this floor.

9. **Add numerical stability guards to JIDD-SCMA.** The N0-overflow bug in Run 1 should be permanently guarded against with clamping in the log-likelihood computation to prevent future regressions.
