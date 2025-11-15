# 6G Simulation Metrics - Comprehensive List

This document defines the comprehensive list of metrics to compute for 6G smart factory physical layer simulations, based on IMT-2030 requirements and industry standards.

## Metric Categories

### 1. Error Rate Metrics (Reliability) ⭐⭐⭐
**Priority: CRITICAL** - Core reliability indicators for 6G (target: 99.9999% reliability)

#### Currently Implemented:
- **BER (Bit Error Rate)**: `P(b̂ ≠ b)` - Probability of bit errors
  - **Per-stream**: Matrix `[num_ebno, num_streams]`
  - **Overall**: Vector `[num_ebno]`
  - **6G Target**: < 10⁻⁹ for ultra-reliable communications
  - **Unit**: Dimensionless (0-1)

- **BLER (Block Error Rate)**: `P(block has errors)` - Probability of block errors
  - **Per-stream**: Matrix `[num_ebno, num_streams]`
  - **Overall**: Vector `[num_ebno]`
  - **6G Target**: < 10⁻⁹ for mission-critical applications
  - **Unit**: Dimensionless (0-1)

#### Recommended Additions:
- **FER (Frame Error Rate)**: Frame-level error rate
  - **Use Case**: Higher-layer reliability assessment
  - **6G Relevance**: Important for ultra-reliable low-latency communications (URLLC)

- **PER (Packet Error Rate)**: Packet-level error rate
  - **6G Target**: < 10⁻⁹
  - **Use Case**: Network layer performance

- **Outage Probability**: `P(SINR < threshold)`
  - **6G Relevance**: Critical for reliability assessment
  - **Threshold**: Typically -5 to 0 dB for 6G

---

### 2. Channel Quality Metrics ⭐⭐⭐
**Priority: CRITICAL** - Essential for link adaptation and resource allocation

#### Currently Implemented:
- **SINR (Signal-to-Interference-plus-Noise Ratio)**: 
  - **Per-stream**: Matrix `[num_ebno, num_streams]` (dB)
  - **Overall**: Vector `[num_ebno]` (dB)
  - **6G Relevance**: Key for massive MIMO and interference management
  - **Unit**: dB

- **NMSE (Normalized Mean Squared Error)**: Channel estimation quality
  - **Overall**: Vector `[num_ebno]` (dB)
  - **6G Relevance**: Critical for imperfect CSI scenarios
  - **Unit**: dB (lower is better)

#### Recommended Additions:
- **SNR (Signal-to-Noise Ratio)**: Pure signal quality (without interference)
  - **Per-stream**: Matrix `[num_ebno, num_streams]` (dB)
  - **Use Case**: Baseline performance assessment

- **CQI (Channel Quality Indicator)**: Standardized channel quality metric
  - **Range**: 0-15 (3GPP/6G standard)
  - **Use Case**: Link adaptation decisions

- **Channel Capacity**: `C = log₂(1 + SINR)` (Shannon capacity)
  - **Per-stream**: Matrix `[num_ebno, num_streams]` (bits/s/Hz)
  - **Overall**: Vector `[num_ebno]` (bits/s/Hz)
  - **6G Relevance**: Theoretical performance limit

- **Channel Correlation**: Spatial correlation between antennas
  - **Use Case**: Massive MIMO performance assessment
  - **6G Relevance**: Important for XL-MIMO systems

---

### 3. Throughput and Data Rate Metrics ⭐⭐⭐
**Priority: CRITICAL** - Core 6G performance indicators

#### Currently Implemented:
- **Throughput (bits)**: Successful bits transmitted
  - **Per-stream**: Matrix `[num_ebno, num_streams]` (bits)
  - **Overall**: Vector `[num_ebno]` (bits)
  - **6G Target**: Peak 1 Tbps, user-experienced 1 Gbps
  - **Unit**: bits

- **Spectral Efficiency**: `Throughput / (Bandwidth × Time)`
  - **Overall**: Vector `[num_ebno]` (bits/s/Hz)
  - **6G Target**: Peak 100 bits/s/Hz
  - **Unit**: bits/s/Hz or bits/resource-element

#### Recommended Additions:
- **Peak Data Rate**: Maximum achievable data rate
  - **6G Target**: 1 Tbps
  - **Unit**: bits/s

- **User Experienced Data Rate**: 5th percentile user data rate
  - **6G Target**: 1 Gbps
  - **Unit**: bits/s

- **Goodput**: Application-layer throughput (excluding retransmissions)
  - **Unit**: bits/s
  - **Use Case**: Real application performance

- **Area Throughput Density**: Throughput per unit area
  - **6G Target**: 1 Gbps/m²
  - **Unit**: bits/s/m²
  - **Use Case**: Smart factory capacity planning

---

### 4. Latency Metrics ⭐⭐⭐
**Priority: CRITICAL** - 6G target: < 1 ms end-to-end latency

#### Currently NOT Implemented - **HIGH PRIORITY**:
- **End-to-End Latency**: Total transmission delay
  - **6G Target**: < 0.1 ms (air interface), < 1 ms (end-to-end)
  - **Components**: Encoding + transmission + propagation + decoding
  - **Unit**: seconds or milliseconds

- **Processing Latency**: Time for encoding/decoding
  - **Component**: LDPC encoding + decoding time
  - **Unit**: seconds or milliseconds

- **Decoding Latency**: Time for LDPC decoder convergence
  - **Related**: `decoder_iter_avg` (already tracked)
  - **Unit**: seconds or milliseconds

- **Frame Transmission Time**: Time to transmit one frame
  - **Formula**: `num_ofdm_symbols × symbol_duration`
  - **Unit**: seconds or milliseconds

#### Recommended Additions:
- **Jitter**: Variation in latency
  - **6G Target**: < 1 μs
  - **Unit**: seconds or milliseconds
  - **Use Case**: Real-time applications

---

### 5. Modulation and Signal Quality Metrics ⭐⭐
**Priority: HIGH** - Signal quality assessment

#### Currently Implemented:
- **EVM (Error Vector Magnitude)**: Modulation quality
  - **Overall**: Vector `[num_ebno]` (%)
  - **6G Relevance**: Important for higher-order modulations (1024QAM, 4096QAM)
  - **Unit**: % (lower is better)

#### Recommended Additions:
- **Constellation Error**: Distance from ideal constellation points
  - **Per-stream**: Matrix `[num_ebno, num_streams]`
  - **Unit**: Linear or dB

- **Modulation Order**: Effective modulation order achieved
  - **Use Case**: Adaptive modulation assessment
  - **6G Relevance**: 6G supports up to 4096QAM

---

### 6. Decoder Performance Metrics ⭐⭐
**Priority: HIGH** - LDPC decoder efficiency

#### Currently Implemented:
- **Decoder Iterations (Average)**: Average LDPC decoder iterations
  - **Per-stream**: Matrix `[num_ebno, num_streams]`
  - **Overall**: Vector `[num_ebno]`
  - **6G Relevance**: Affects latency and energy consumption
  - **Unit**: Iterations (dimensionless)

#### Recommended Additions:
- **Decoder Convergence Rate**: Percentage of blocks that converge
  - **Unit**: % (0-100)
  - **Use Case**: Decoder reliability assessment

- **Decoder Energy**: Energy consumed per decoded bit
  - **6G Target**: 1 pJ/bit
  - **Unit**: Joules/bit or pJ/bit
  - **Use Case**: Energy efficiency assessment

---

### 7. Fairness and Resource Allocation Metrics ⭐⭐
**Priority: HIGH** - Multi-user fairness

#### Currently Implemented:
- **Jain's Fairness Index**: `(Σxᵢ)² / (n × Σxᵢ²)`
  - **Overall**: Vector `[num_ebno]`
  - **Range**: 0-1 (1 = perfect fairness)
  - **6G Relevance**: Critical for massive connectivity (10⁶ devices/km²)
  - **Unit**: Dimensionless

#### Recommended Additions:
- **Per-UT Throughput**: Throughput per user terminal
  - **Per-UT**: Matrix `[num_ebno, num_ut]` (bits)
  - **Use Case**: User fairness assessment

- **Min-Max Throughput Ratio**: Ratio of minimum to maximum UT throughput
  - **Range**: 0-1 (1 = perfect fairness)
  - **Unit**: Dimensionless

- **Resource Utilization**: Percentage of allocated resources
  - **Unit**: % (0-100)
  - **Use Case**: Resource efficiency

---

### 8. Energy Efficiency Metrics ⭐⭐
**Priority: HIGH** - 6G target: 1 pJ/bit

#### Currently NOT Implemented - **HIGH PRIORITY**:
- **Energy per Bit**: Total energy consumed per successfully transmitted bit
  - **6G Target**: 1 pJ/bit
  - **Formula**: `Total_Energy / Successful_Bits`
  - **Unit**: Joules/bit or pJ/bit

- **Power Consumption**: Total power consumed
  - **Components**: Transmitter + Receiver + Processing
  - **Unit**: Watts

- **Energy Efficiency**: Bits per Joule
  - **Formula**: `Successful_Bits / Total_Energy`
  - **Unit**: bits/Joule
  - **6G Relevance**: Critical for IoT devices

---

### 9. Massive MIMO and Spatial Metrics ⭐⭐
**Priority: HIGH** - 6G XL-MIMO specific

#### Currently NOT Implemented - **MEDIUM PRIORITY**:
- **Spatial Multiplexing Gain**: Effective number of independent streams
  - **Formula**: `rank(H)` or effective degrees of freedom
  - **Unit**: Dimensionless
  - **6G Relevance**: Massive MIMO (32-4096 antennas)

- **Beamforming Gain**: Array gain from beamforming
  - **Formula**: `10×log₁₀(num_antennas)` (theoretical)
  - **Unit**: dB
  - **6G Relevance**: XL-MIMO beamforming

- **Interference Suppression**: Interference reduction from MIMO
  - **Unit**: dB
  - **Use Case**: Multi-user interference assessment

- **Channel Hardening**: Reduction in channel variation
  - **Formula**: `std(SINR) / mean(SINR)`
  - **Unit**: Dimensionless (lower = more hardening)
  - **6G Relevance**: Massive MIMO benefit

---

### 10. Connection Density Metrics ⭐
**Priority: MEDIUM** - 6G target: 10⁶ devices/km²

#### Currently NOT Implemented - **MEDIUM PRIORITY**:
- **Active Connections**: Number of simultaneously active UTs
  - **Current**: `num_ut` (fixed in config)
  - **Future**: Dynamic scheduling assessment
  - **Unit**: Count

- **Connection Success Rate**: Percentage of successful connections
  - **Unit**: % (0-100)
  - **6G Relevance**: Massive connectivity

---

### 11. Positioning and Sensing Metrics ⭐
**Priority: LOW** - 6G integrated sensing

#### Currently NOT Implemented - **LOW PRIORITY**:
- **Positioning Accuracy**: Location estimation error
  - **6G Target**: < 1 cm (3D)
  - **Unit**: meters or centimeters
  - **Note**: Requires additional sensing capabilities

---

## Metric Priority Summary

### Tier 1: CRITICAL (Must Compute) ⭐⭐⭐
1. **BER** - Already implemented
2. **BLER** - Already implemented
3. **SINR** - Already implemented
4. **Throughput** - Already implemented
5. **Spectral Efficiency** - Already implemented
6. **End-to-End Latency** - **NOT IMPLEMENTED** ⚠️
7. **NMSE** - Already implemented

### Tier 2: HIGH PRIORITY (Should Compute) ⭐⭐
8. **EVM** - Already implemented
9. **Decoder Iterations** - Already implemented
10. **Fairness (Jain's Index)** - Already implemented
11. **Energy per Bit** - **NOT IMPLEMENTED** ⚠️
12. **Channel Capacity** - **NOT IMPLEMENTED** ⚠️
13. **Outage Probability** - **NOT IMPLEMENTED** ⚠️

### Tier 3: MEDIUM PRIORITY (Nice to Have) ⭐
14. **Spatial Multiplexing Gain** - **NOT IMPLEMENTED**
15. **Beamforming Gain** - **NOT IMPLEMENTED**
16. **Channel Hardening** - **NOT IMPLEMENTED**
17. **Per-UT Throughput** - **NOT IMPLEMENTED**

### Tier 4: LOW PRIORITY (Future Work)
18. **Positioning Accuracy** - **NOT IMPLEMENTED**
19. **Jitter** - **NOT IMPLEMENTED**
20. **Connection Density** - **NOT IMPLEMENTED**

---

## Recommended Implementation Order

### Phase 1: Critical Missing Metrics
1. **End-to-End Latency** - Most critical for 6G URLLC
2. **Energy per Bit** - Critical for 6G energy efficiency
3. **Outage Probability** - Critical for reliability assessment

### Phase 2: High-Value Additions
4. **Channel Capacity** - Theoretical performance limit
5. **Per-UT Throughput** - Better fairness assessment
6. **SNR** - Baseline performance metric

### Phase 3: Advanced Metrics
7. **Spatial Multiplexing Gain** - Massive MIMO assessment
8. **Beamforming Gain** - XL-MIMO assessment
9. **Channel Hardening** - Massive MIMO benefit

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

## References

1. **ITU-R M.2150**: IMT-2020 (5G) requirements
2. **ITU-R M.2083**: IMT-2030 (6G) vision and requirements
3. **3GPP TR 38.901**: Channel model for frequencies 0.5-100 GHz
4. **3GPP TS 38.211**: Physical channels and modulation
5. **6G Flagship White Papers**: 6G use cases and requirements

---

## Notes

- All metrics should be computed for both **Perfect CSI** and **Imperfect CSI** conditions
- Metrics should be computed across the **Eb/No range** (typically -5 to 15 dB)
- **Per-stream** metrics enable stream-level analysis and fairness assessment
- **Overall** metrics provide system-level performance summary
- **Per-UT** metrics enable user-level fairness and resource allocation analysis

