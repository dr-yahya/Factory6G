# 6G Simulation Metrics - Quick Reference

## Currently Implemented Metrics ✅

### Error Rate Metrics
- ✅ **BER** (Bit Error Rate) - Per-stream & Overall
- ✅ **BLER** (Block Error Rate) - Per-stream & Overall

### Channel Quality Metrics
- ✅ **SINR** (Signal-to-Interference-plus-Noise Ratio) - Per-stream & Overall (dB)
- ✅ **NMSE** (Normalized Mean Squared Error) - Overall (dB)

### Throughput Metrics
- ✅ **Throughput** (bits) - Per-stream & Overall
- ✅ **Spectral Efficiency** (bits/s/Hz) - Overall

### Signal Quality Metrics
- ✅ **EVM** (Error Vector Magnitude) - Overall (%)

### Decoder Metrics
- ✅ **Decoder Iterations** (average) - Per-stream & Overall

### Fairness Metrics
- ✅ **Jain's Fairness Index** - Overall

---

## Critical Missing Metrics ⚠️ (HIGH PRIORITY)

### 1. **End-to-End Latency** ⭐⭐⭐
- **6G Target**: < 0.1 ms (air interface), < 1 ms (end-to-end)
- **Components**: Encoding + transmission + propagation + decoding
- **Why Critical**: Core 6G URLLC requirement
- **Implementation**: Track timestamps at each stage

### 2. **Energy per Bit** ⭐⭐⭐
- **6G Target**: 1 pJ/bit
- **Formula**: `Total_Energy / Successful_Bits`
- **Why Critical**: 6G energy efficiency requirement
- **Implementation**: Estimate power consumption per component

### 3. **Outage Probability** ⭐⭐⭐
- **Definition**: `P(SINR < threshold)`
- **6G Target**: < 10⁻⁶ for reliability
- **Why Critical**: Reliability assessment for URLLC
- **Implementation**: Count blocks with SINR below threshold

### 4. **Channel Capacity** ⭐⭐
- **Formula**: `C = log₂(1 + SINR)` (Shannon capacity)
- **Why Important**: Theoretical performance limit
- **Implementation**: Compute from SINR values

### 5. **SNR** (Signal-to-Noise Ratio) ⭐⭐
- **Why Important**: Baseline performance (without interference)
- **Implementation**: Similar to SINR but exclude interference

---

## Recommended Additional Metrics (MEDIUM PRIORITY)

### Spatial/MIMO Metrics
- **Spatial Multiplexing Gain**: Effective number of independent streams
- **Beamforming Gain**: Array gain from beamforming (dB)
- **Channel Hardening**: Reduction in channel variation
- **Interference Suppression**: Interference reduction from MIMO (dB)

### Fairness Metrics
- **Per-UT Throughput**: Throughput per user terminal
- **Min-Max Throughput Ratio**: Fairness indicator

### Advanced Error Metrics
- **FER** (Frame Error Rate): Frame-level errors
- **PER** (Packet Error Rate): Packet-level errors

### Energy Metrics
- **Power Consumption**: Total power (Watts)
- **Energy Efficiency**: Bits per Joule

---

## Metric Priority Matrix

| Metric | Priority | Status | 6G Target | Implementation Effort |
|--------|----------|--------|-----------|----------------------|
| BER | ⭐⭐⭐ | ✅ Done | < 10⁻⁹ | - |
| BLER | ⭐⭐⭐ | ✅ Done | < 10⁻⁹ | - |
| SINR | ⭐⭐⭐ | ✅ Done | > 0 dB | - |
| Throughput | ⭐⭐⭐ | ✅ Done | 1 Tbps peak | - |
| Spectral Efficiency | ⭐⭐⭐ | ✅ Done | 100 bits/s/Hz | - |
| **End-to-End Latency** | ⭐⭐⭐ | ❌ Missing | < 1 ms | Medium |
| **Energy per Bit** | ⭐⭐⭐ | ❌ Missing | 1 pJ/bit | Medium |
| **Outage Probability** | ⭐⭐⭐ | ❌ Missing | < 10⁻⁶ | Low |
| **Channel Capacity** | ⭐⭐ | ❌ Missing | - | Low |
| **SNR** | ⭐⭐ | ❌ Missing | - | Low |
| NMSE | ⭐⭐ | ✅ Done | < -20 dB | - |
| EVM | ⭐⭐ | ✅ Done | < 5% | - |
| Decoder Iterations | ⭐⭐ | ✅ Done | < 10 | - |
| Fairness (Jain) | ⭐⭐ | ✅ Done | > 0.9 | - |
| Spatial Multiplexing Gain | ⭐ | ❌ Missing | - | Medium |
| Beamforming Gain | ⭐ | ❌ Missing | - | Medium |

---

## Recommended Implementation Plan

### Phase 1: Critical Metrics (Week 1-2)
1. **Outage Probability** - Easy to implement, high value
2. **Channel Capacity** - Easy to implement (from SINR)
3. **SNR** - Easy to implement (similar to SINR)

### Phase 2: High-Value Metrics (Week 3-4)
4. **End-to-End Latency** - Medium effort, critical for 6G
5. **Energy per Bit** - Medium effort, critical for 6G
6. **Per-UT Throughput** - Low effort, better fairness analysis

### Phase 3: Advanced Metrics (Week 5+)
7. **Spatial Multiplexing Gain** - Medium effort
8. **Beamforming Gain** - Medium effort
9. **Channel Hardening** - Medium effort

---

## Quick Decision Guide

**For 6G Smart Factory Simulations, you MUST compute:**
1. ✅ BER, BLER (reliability)
2. ✅ SINR, NMSE (channel quality)
3. ✅ Throughput, Spectral Efficiency (data rate)
4. ❌ **End-to-End Latency** (URLLC requirement)
5. ❌ **Energy per Bit** (energy efficiency)
6. ❌ **Outage Probability** (reliability)

**Nice to have:**
- Channel Capacity (theoretical limit)
- Spatial metrics (MIMO assessment)
- Per-UT metrics (fairness analysis)

---

## References

- Full documentation: `docs/6G_SIMULATION_METRICS.md`
- IMT-2030 Requirements: ITU-R M.2083
- 3GPP Standards: TS 38.211, TR 38.901

