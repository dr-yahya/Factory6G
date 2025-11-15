# 6G Physical Layer Metrics - Quick Reference

> **Focus**: Physical Layer (PHY) metrics only. Higher-layer metrics excluded.

---

## Currently Implemented Metrics ✅

### Error Rate Metrics
- ✅ **BER** (Bit Error Rate) - Per-stream & Overall
- ✅ **BLER** (Block Error Rate) - Per-stream & Overall

### Signal Quality Metrics
- ✅ **SINR** (Signal-to-Interference-plus-Noise Ratio) - Per-stream & Overall (dB)
- ✅ **EVM** (Error Vector Magnitude) - Overall (%)

### Channel Estimation Metrics
- ✅ **NMSE** (Normalized Mean Squared Error) - Overall (dB)

### Throughput Metrics
- ✅ **Throughput** (bits) - Per-stream & Overall
- ✅ **Spectral Efficiency** (bits/s/Hz) - Overall

### Decoder Metrics
- ✅ **Decoder Iterations** (average) - Per-stream & Overall

### Fairness Metrics
- ✅ **Jain's Fairness Index** - Overall

---

## Critical Missing Metrics ⚠️ (HIGH PRIORITY)

### 1. **Air Interface Latency** ⭐⭐⭐
- **6G Target**: < 0.1 ms (air interface)
- **Components**: Encoding + transmission + propagation + decoding
- **Why Critical**: Core 6G URLLC requirement at physical layer
- **Implementation**: Track timestamps at each PHY stage

### 2. **Energy per Bit (PHY)** ⭐⭐⭐
- **6G Target**: 1 pJ/bit
- **Formula**: `Total_PHY_Energy / Successful_Bits`
- **Components**: Encoding + RF + Decoding energy
- **Why Critical**: 6G energy efficiency requirement at physical layer
- **Implementation**: Estimate power consumption per PHY component

### 3. **Outage Probability** ⭐⭐⭐
- **Definition**: `P(SINR < threshold)`
- **6G Target**: < 10⁻⁶ for reliability
- **Why Critical**: Physical layer reliability assessment
- **Implementation**: Count blocks with SINR below threshold

### 4. **Channel Capacity** ⭐⭐
- **Formula**: `C = log₂(1 + SINR)` (Shannon capacity)
- **Why Important**: Theoretical physical layer performance limit
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
- **Per-UT SINR**: SINR per user terminal
- **Min-Max Throughput Ratio**: Fairness indicator

### Signal Quality Metrics
- **Symbol Error Rate (SER)**: Symbol errors before decoding
- **Constellation Error**: Distance from ideal constellation points
- **Pre-Decoder BER**: Raw channel BER before decoding

### Resource Grid Metrics
- **Resource Element Utilization**: Percentage of used REs
- **Pilot Overhead**: Percentage of REs used for pilots
- **Effective Subcarriers**: Number of active subcarriers

---

## Metric Priority Matrix

| Metric | Priority | Status | 6G Target | Implementation Effort |
|--------|----------|--------|-----------|----------------------|
| BER | ⭐⭐⭐ | ✅ Done | < 10⁻⁹ | - |
| BLER | ⭐⭐⭐ | ✅ Done | < 10⁻⁹ | - |
| SINR | ⭐⭐⭐ | ✅ Done | > 0 dB | - |
| Throughput | ⭐⭐⭐ | ✅ Done | 1 Tbps peak | - |
| Spectral Efficiency | ⭐⭐⭐ | ✅ Done | 100 bits/s/Hz | - |
| **Air Interface Latency** | ⭐⭐⭐ | ❌ Missing | < 0.1 ms | Medium |
| **Energy per Bit (PHY)** | ⭐⭐⭐ | ❌ Missing | 1 pJ/bit | Medium |
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

## Physical Layer Scope

### ✅ Included (Physical Layer):
- Error rates (BER, BLER, SER)
- Signal quality (SINR, SNR, EVM)
- Channel estimation (NMSE)
- Throughput at PHY
- Spectral efficiency at PHY
- Latency at PHY (encoding, transmission, decoding)
- Energy at PHY (encoding, RF, decoding)
- MIMO/spatial metrics
- Resource grid metrics
- Decoder performance

### ❌ Excluded (Higher Layers):
- Packet Error Rate (PER) - Network layer
- Frame Error Rate (FER) - Data link layer
- End-to-end latency (includes network/transport)
- Application-layer throughput
- Network-layer fairness
- Positioning accuracy (unless PHY sensing)
- Connection density (network layer)

---

## Recommended Implementation Plan

### Phase 1: Critical Metrics (Week 1-2)
1. **Outage Probability** - Easy, high value
2. **Channel Capacity** - Easy (from SINR)
3. **SNR** - Easy (similar to SINR)

### Phase 2: High-Value Metrics (Week 3-4)
4. **Air Interface Latency** - Medium effort, critical
5. **Energy per Bit (PHY)** - Medium effort, critical
6. **Per-UT Throughput** - Low effort, better fairness

### Phase 3: Advanced Metrics (Week 5+)
7. **Spatial Multiplexing Gain** - Medium effort
8. **Beamforming Gain** - Medium effort
9. **Channel Hardening** - Medium effort

---

## Quick Decision Guide

**For 6G Physical Layer Simulations, you MUST compute:**
1. ✅ BER, BLER (reliability)
2. ✅ SINR, NMSE (channel quality)
3. ✅ Throughput, Spectral Efficiency (data rate)
4. ❌ **Air Interface Latency** (URLLC requirement)
5. ❌ **Energy per Bit (PHY)** (energy efficiency)
6. ❌ **Outage Probability** (reliability)

**Nice to have:**
- Channel Capacity (theoretical limit)
- Spatial metrics (MIMO assessment)
- Per-UT metrics (fairness analysis)

---

## References

- Full documentation: `docs/6G_PHYSICAL_LAYER_METRICS.md`
- IMT-2030 Requirements: ITU-R M.2083
- 3GPP Physical Layer: TS 38.211, TS 38.212, TR 38.901

