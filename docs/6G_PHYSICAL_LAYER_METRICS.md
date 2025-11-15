# 6G Physical Layer Simulation Metrics

This document defines the comprehensive list of **physical layer metrics** to compute for 6G smart factory simulations, based on IMT-2030 requirements and 3GPP physical layer standards.

> **Note**: This list focuses exclusively on **physical layer (PHY)** metrics. Higher-layer metrics (network, transport, application) are excluded.

---

## Metric Categories

### 1. Error Rate Metrics (Reliability) ⭐⭐⭐
**Priority: CRITICAL** - Core physical layer reliability indicators

#### Currently Implemented:
- **BER (Bit Error Rate)**: `P(b̂ ≠ b)` - Probability of bit errors after decoding
  - **Per-stream**: Matrix `[num_ebno, num_streams]`
  - **Overall**: Vector `[num_ebno]`
  - **6G Target**: < 10⁻⁹ for ultra-reliable communications
  - **Unit**: Dimensionless (0-1)
  - **PHY Relevance**: Direct measure of physical layer reliability

- **BLER (Block Error Rate)**: `P(block has errors)` - Probability of block errors after decoding
  - **Per-stream**: Matrix `[num_ebno, num_streams]`
  - **Overall**: Vector `[num_ebno]`
  - **6G Target**: < 10⁻⁹ for mission-critical applications
  - **Unit**: Dimensionless (0-1)
  - **PHY Relevance**: Block-level reliability at physical layer

#### Recommended Additions:
- **Pre-Decoder BER**: Bit error rate before LDPC decoding (raw channel BER)
  - **Use Case**: Channel quality assessment independent of coding
  - **PHY Relevance**: Raw physical channel performance

- **Outage Probability**: `P(SINR < threshold)` - Probability of link outage
  - **6G Target**: < 10⁻⁶ for reliability
  - **Threshold**: Typically -5 to 0 dB for 6G
  - **PHY Relevance**: Physical link reliability assessment
  - **Unit**: Dimensionless (0-1)

---

### 2. Signal Quality Metrics ⭐⭐⭐
**Priority: CRITICAL** - Essential for physical layer performance

#### Currently Implemented:
- **SINR (Signal-to-Interference-plus-Noise Ratio)**: 
  - **Per-stream**: Matrix `[num_ebno, num_streams]` (dB)
  - **Overall**: Vector `[num_ebno]` (dB)
  - **6G Relevance**: Key for massive MIMO and interference management
  - **Unit**: dB
  - **PHY Relevance**: Fundamental physical layer quality metric

- **EVM (Error Vector Magnitude)**: Modulation quality
  - **Overall**: Vector `[num_ebno]` (%)
  - **6G Relevance**: Important for higher-order modulations (1024QAM, 4096QAM)
  - **Unit**: % (lower is better)
  - **PHY Relevance**: Physical layer modulation accuracy

#### Recommended Additions:
- **SNR (Signal-to-Noise Ratio)**: Pure signal quality (without interference)
  - **Per-stream**: Matrix `[num_ebno, num_streams]` (dB)
  - **Overall**: Vector `[num_ebno]` (dB)
  - **PHY Relevance**: Baseline performance metric (no interference)
  - **Unit**: dB

- **Constellation Error**: Distance from ideal constellation points
  - **Per-stream**: Matrix `[num_ebno, num_streams]` (linear or dB)
  - **PHY Relevance**: Modulation symbol accuracy
  - **Unit**: Linear or dB

- **Symbol Error Rate (SER)**: Probability of symbol errors before decoding
  - **Per-stream**: Matrix `[num_ebno, num_streams]`
  - **PHY Relevance**: Raw symbol-level performance
  - **Unit**: Dimensionless (0-1)

---

### 3. Channel Estimation Quality Metrics ⭐⭐⭐
**Priority: CRITICAL** - Essential for imperfect CSI scenarios

#### Currently Implemented:
- **NMSE (Normalized Mean Squared Error)**: Channel estimation quality
  - **Overall**: Vector `[num_ebno]` (dB)
  - **6G Relevance**: Critical for imperfect CSI scenarios
  - **Unit**: dB (lower is better)
  - **PHY Relevance**: Physical channel estimation accuracy

#### Recommended Additions:
- **Channel Estimation Error Variance**: Variance of channel estimation error
  - **Overall**: Vector `[num_ebno]` (dB)
  - **PHY Relevance**: Channel estimator performance
  - **Unit**: dB

- **Pilot SNR**: Signal-to-noise ratio at pilot symbols
  - **Overall**: Vector `[num_ebno]` (dB)
  - **PHY Relevance**: Pilot symbol quality for channel estimation
  - **Unit**: dB

---

### 4. Throughput and Data Rate Metrics ⭐⭐⭐
**Priority: CRITICAL** - Core 6G performance indicators at physical layer

#### Currently Implemented:
- **Throughput (bits)**: Successful bits transmitted at physical layer
  - **Per-stream**: Matrix `[num_ebno, num_streams]` (bits)
  - **Overall**: Vector `[num_ebno]` (bits)
  - **6G Target**: Peak 1 Tbps, user-experienced 1 Gbps
  - **Unit**: bits
  - **PHY Relevance**: Physical layer data rate

- **Spectral Efficiency**: `Throughput / (Bandwidth × Time)`
  - **Overall**: Vector `[num_ebno]` (bits/s/Hz)
  - **6G Target**: Peak 100 bits/s/Hz
  - **Unit**: bits/s/Hz or bits/resource-element
  - **PHY Relevance**: Physical layer efficiency

#### Recommended Additions:
- **Peak Data Rate**: Maximum achievable physical layer data rate
  - **6G Target**: 1 Tbps
  - **Unit**: bits/s
  - **PHY Relevance**: Physical layer peak performance

- **User Experienced Data Rate**: 5th percentile user data rate at physical layer
  - **6G Target**: 1 Gbps
  - **Unit**: bits/s
  - **PHY Relevance**: Physical layer user experience

- **Channel Capacity**: `C = log₂(1 + SINR)` (Shannon capacity)
  - **Per-stream**: Matrix `[num_ebno, num_streams]` (bits/s/Hz)
  - **Overall**: Vector `[num_ebno]` (bits/s/Hz)
  - **PHY Relevance**: Theoretical physical layer performance limit
  - **Unit**: bits/s/Hz

---

### 5. Physical Layer Latency Metrics ⭐⭐⭐
**Priority: CRITICAL** - 6G target: < 0.1 ms air interface latency

#### Currently NOT Implemented - **HIGH PRIORITY**:
- **Air Interface Latency**: Physical layer transmission delay
  - **Components**: 
    - Encoding time (LDPC)
    - OFDM symbol transmission time
    - Propagation delay
    - Decoding time (LDPC)
  - **6G Target**: < 0.1 ms (air interface)
  - **PHY Relevance**: Physical layer latency
  - **Unit**: seconds or milliseconds

- **Encoding Latency**: Time for LDPC encoding
  - **PHY Relevance**: Physical layer processing delay
  - **Unit**: seconds or milliseconds

- **Decoding Latency**: Time for LDPC decoder convergence
  - **Related**: `decoder_iter_avg` (already tracked)
  - **Formula**: `decoder_iter_avg × iteration_time`
  - **PHY Relevance**: Physical layer processing delay
  - **Unit**: seconds or milliseconds

- **OFDM Symbol Transmission Time**: Time to transmit one OFDM symbol
  - **Formula**: `T_sym = 1/Δf + T_CP`
  - **PHY Relevance**: Physical layer frame structure delay
  - **Unit**: seconds or milliseconds

- **Frame Transmission Time**: Time to transmit one frame
  - **Formula**: `num_ofdm_symbols × T_sym`
  - **PHY Relevance**: Physical layer frame delay
  - **Unit**: seconds or milliseconds

---

### 6. Decoder Performance Metrics ⭐⭐
**Priority: HIGH** - LDPC decoder efficiency at physical layer

#### Currently Implemented:
- **Decoder Iterations (Average)**: Average LDPC decoder iterations
  - **Per-stream**: Matrix `[num_ebno, num_streams]`
  - **Overall**: Vector `[num_ebno]`
  - **6G Relevance**: Affects latency and energy consumption
  - **Unit**: Iterations (dimensionless)
  - **PHY Relevance**: Physical layer decoder performance

#### Recommended Additions:
- **Decoder Convergence Rate**: Percentage of blocks that converge
  - **Unit**: % (0-100)
  - **PHY Relevance**: Physical layer decoder reliability

- **Decoder Convergence Time**: Average time per iteration
  - **Unit**: seconds or milliseconds
  - **PHY Relevance**: Physical layer processing speed

---

### 7. Energy Efficiency Metrics (Physical Layer) ⭐⭐
**Priority: HIGH** - 6G target: 1 pJ/bit at physical layer

#### Currently NOT Implemented - **HIGH PRIORITY**:
- **Energy per Bit**: Total physical layer energy consumed per successfully transmitted bit
  - **6G Target**: 1 pJ/bit
  - **Formula**: `Total_PHY_Energy / Successful_Bits`
  - **Components**: 
    - Encoding energy
    - Transmission energy (RF)
    - Reception energy (RF)
    - Decoding energy
  - **PHY Relevance**: Physical layer energy efficiency
  - **Unit**: Joules/bit or pJ/bit

- **Power Consumption (PHY)**: Total physical layer power consumed
  - **Components**: 
    - Baseband processing power
    - RF power (transmit + receive)
    - ADC/DAC power
  - **PHY Relevance**: Physical layer power efficiency
  - **Unit**: Watts

- **Energy Efficiency**: Bits per Joule at physical layer
  - **Formula**: `Successful_Bits / Total_PHY_Energy`
  - **PHY Relevance**: Physical layer energy efficiency
  - **Unit**: bits/Joule

---

### 8. Massive MIMO and Spatial Metrics ⭐⭐
**Priority: HIGH** - 6G XL-MIMO specific (32-4096 antennas)

#### Currently NOT Implemented - **MEDIUM PRIORITY**:
- **Spatial Multiplexing Gain**: Effective number of independent streams
  - **Formula**: `rank(H)` or effective degrees of freedom
  - **Unit**: Dimensionless
  - **6G Relevance**: Massive MIMO (32-4096 antennas)
  - **PHY Relevance**: Physical layer spatial multiplexing capability

- **Beamforming Gain**: Array gain from beamforming
  - **Formula**: `10×log₁₀(num_antennas)` (theoretical maximum)
  - **Unit**: dB
  - **6G Relevance**: XL-MIMO beamforming
  - **PHY Relevance**: Physical layer array gain

- **Interference Suppression**: Interference reduction from MIMO
  - **Unit**: dB
  - **PHY Relevance**: Physical layer interference mitigation

- **Channel Hardening**: Reduction in channel variation
  - **Formula**: `std(SINR) / mean(SINR)` (coefficient of variation)
  - **Unit**: Dimensionless (lower = more hardening)
  - **6G Relevance**: Massive MIMO benefit
  - **PHY Relevance**: Physical layer channel stability

- **Spatial Correlation**: Correlation between antenna elements
  - **Unit**: Dimensionless (0-1)
  - **PHY Relevance**: Physical layer antenna array performance

---

### 9. Resource Grid and OFDM Metrics ⭐
**Priority: MEDIUM** - Physical layer resource utilization

#### Currently NOT Implemented - **MEDIUM PRIORITY**:
- **Resource Element (RE) Utilization**: Percentage of used resource elements
  - **Formula**: `(Data_REs + Pilot_REs) / Total_REs`
  - **Unit**: % (0-100)
  - **PHY Relevance**: Physical layer resource efficiency

- **Pilot Overhead**: Percentage of resource elements used for pilots
  - **Formula**: `Pilot_REs / Total_REs`
  - **Unit**: % (0-100)
  - **PHY Relevance**: Physical layer pilot efficiency

- **Effective Subcarriers**: Number of active (non-nulled) subcarriers
  - **Unit**: Count
  - **PHY Relevance**: Physical layer bandwidth utilization

---

### 10. Fairness Metrics (Physical Layer Resource Allocation) ⭐⭐
**Priority: HIGH** - Multi-user fairness at physical layer

#### Currently Implemented:
- **Jain's Fairness Index**: `(Σxᵢ)² / (n × Σxᵢ²)`
  - **Overall**: Vector `[num_ebno]`
  - **Range**: 0-1 (1 = perfect fairness)
  - **6G Relevance**: Critical for massive connectivity (10⁶ devices/km²)
  - **Unit**: Dimensionless
  - **PHY Relevance**: Physical layer resource allocation fairness

#### Recommended Additions:
- **Per-UT Throughput**: Throughput per user terminal at physical layer
  - **Per-UT**: Matrix `[num_ebno, num_ut]` (bits)
  - **PHY Relevance**: Physical layer user fairness

- **Per-UT SINR**: SINR per user terminal
  - **Per-UT**: Matrix `[num_ebno, num_ut]` (dB)
  - **PHY Relevance**: Physical layer user channel quality

- **Min-Max Throughput Ratio**: Ratio of minimum to maximum UT throughput
  - **Range**: 0-1 (1 = perfect fairness)
  - **PHY Relevance**: Physical layer fairness indicator
  - **Unit**: Dimensionless

---

## Metric Priority Summary

### Tier 1: CRITICAL (Must Compute) ⭐⭐⭐
1. ✅ **BER** - Already implemented
2. ✅ **BLER** - Already implemented
3. ✅ **SINR** - Already implemented
4. ✅ **Throughput** - Already implemented
5. ✅ **Spectral Efficiency** - Already implemented
6. ✅ **NMSE** - Already implemented
7. ❌ **Air Interface Latency** - **NOT IMPLEMENTED** ⚠️
8. ❌ **Energy per Bit (PHY)** - **NOT IMPLEMENTED** ⚠️
9. ❌ **Outage Probability** - **NOT IMPLEMENTED** ⚠️

### Tier 2: HIGH PRIORITY (Should Compute) ⭐⭐
10. ✅ **EVM** - Already implemented
11. ✅ **Decoder Iterations** - Already implemented
12. ✅ **Fairness (Jain's Index)** - Already implemented
13. ❌ **Channel Capacity** - **NOT IMPLEMENTED** ⚠️
14. ❌ **SNR** - **NOT IMPLEMENTED** ⚠️
15. ❌ **Per-UT Throughput** - **NOT IMPLEMENTED** ⚠️

### Tier 3: MEDIUM PRIORITY (Nice to Have) ⭐
16. ❌ **Spatial Multiplexing Gain** - **NOT IMPLEMENTED**
17. ❌ **Beamforming Gain** - **NOT IMPLEMENTED**
18. ❌ **Channel Hardening** - **NOT IMPLEMENTED**
19. ❌ **Resource Element Utilization** - **NOT IMPLEMENTED**

---

## Recommended Implementation Order

### Phase 1: Critical Missing Metrics (Week 1-2)
1. **Outage Probability** - Easy to implement, high value
2. **Channel Capacity** - Easy to implement (from SINR)
3. **SNR** - Easy to implement (similar to SINR)

### Phase 2: High-Value Additions (Week 3-4)
4. **Air Interface Latency** - Medium effort, critical for 6G URLLC
5. **Energy per Bit (PHY)** - Medium effort, critical for 6G energy efficiency
6. **Per-UT Throughput** - Low effort, better fairness analysis

### Phase 3: Advanced Metrics (Week 5+)
7. **Spatial Multiplexing Gain** - Medium effort
8. **Beamforming Gain** - Medium effort
9. **Channel Hardening** - Medium effort

---

## Metric Storage Format

### Per-Stream Metrics (Matrices)
- Shape: `[num_ebno_points, num_streams]`
- Example: `[8, 16]` for 8 Eb/No points and 16 streams
- File: `{metric}_per_stream_{csi}_run{idx}.npy`

### Overall Metrics (Vectors)
- Shape: `[num_ebno_points]`
- Example: `[8]` for 8 Eb/No points
- File: `{metric}_overall_{csi}_run{idx}.npy`

### Per-UT Metrics (Matrices)
- Shape: `[num_ebno_points, num_ut]`
- Example: `[8, 8]` for 8 Eb/No points and 8 UTs
- File: `{metric}_per_ut_{csi}_run{idx}.npy`

---

## Physical Layer Scope Definition

### Included (Physical Layer):
- ✅ Error rates (BER, BLER, SER)
- ✅ Signal quality (SINR, SNR, EVM)
- ✅ Channel estimation (NMSE, estimation error)
- ✅ Throughput at PHY
- ✅ Spectral efficiency at PHY
- ✅ Latency at PHY (encoding, transmission, decoding)
- ✅ Energy at PHY (encoding, RF, decoding)
- ✅ MIMO/spatial metrics
- ✅ Resource grid metrics
- ✅ Decoder performance

### Excluded (Higher Layers):
- ❌ Packet Error Rate (PER) - Network layer
- ❌ Frame Error Rate (FER) - Data link layer
- ❌ End-to-end latency (includes network/transport layers)
- ❌ Application-layer throughput
- ❌ Network-layer fairness
- ❌ Positioning accuracy (unless integrated sensing at PHY)
- ❌ Connection density (network layer concept)

---

## References

1. **ITU-R M.2083**: IMT-2030 (6G) vision and requirements
2. **3GPP TS 38.211**: Physical channels and modulation
3. **3GPP TS 38.212**: Multiplexing and channel coding
4. **3GPP TR 38.901**: Channel model for frequencies 0.5-100 GHz
5. **3GPP TR 38.802**: Study on new radio access technology physical layer aspects

---

## Notes

- All metrics are computed at the **physical layer** only
- Metrics should be computed for both **Perfect CSI** and **Imperfect CSI** conditions
- Metrics should be computed across the **Eb/No range** (typically -5 to 15 dB)
- **Per-stream** metrics enable stream-level analysis
- **Overall** metrics provide system-level performance summary
- **Per-UT** metrics enable user-level fairness analysis

