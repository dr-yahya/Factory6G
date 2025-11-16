# Implemented Physical Layer Metrics

## Summary

All critical physical layer metrics for 6G simulation have been implemented with full support for matrices, plots, and visualization.

---

## ✅ Implemented Metrics

### 1. **SNR (Signal-to-Noise Ratio)** ⭐⭐
- **Type**: Per-stream (matrix) & Overall (vector)
- **Unit**: dB
- **Description**: Pure signal quality without interference
- **Formula**: `SNR = 10×log₁₀(Signal_Power / Noise_Power)`
- **Files**:
  - Matrices: `snr_db_per_stream_{csi}_run{idx}.npy`
  - Vectors: `snr_db_overall_{csi}_run{idx}.npy`
  - Plots: `snr_db_per_stream_{csi}_run{idx}.png`, `snr_db_overall_{csi}_run{idx}.png`, `snr_db_comparison.png`

### 2. **Channel Capacity** ⭐⭐
- **Type**: Per-stream (matrix) & Overall (vector)
- **Unit**: bits/s/Hz
- **Description**: Theoretical maximum data rate (Shannon capacity)
- **Formula**: `C = log₂(1 + SINR)`
- **6G Relevance**: Performance limit for physical layer
- **Files**:
  - Matrices: `channel_capacity_per_stream_{csi}_run{idx}.npy`
  - Vectors: `channel_capacity_overall_{csi}_run{idx}.npy`
  - Plots: `channel_capacity_per_stream_{csi}_run{idx}.png`, `channel_capacity_overall_{csi}_run{idx}.png`, `channel_capacity_comparison.png`

### 3. **Outage Probability** ⭐⭐⭐
- **Type**: Per-stream (matrix) & Overall (vector)
- **Unit**: Probability (0-1)
- **Description**: Probability that SINR falls below threshold
- **Formula**: `P(SINR < threshold)`
- **Threshold**: -5 dB (configurable)
- **6G Target**: < 10⁻⁶ for ultra-reliable communications
- **Files**:
  - Matrices: `outage_probability_per_stream_{csi}_run{idx}.npy`
  - Vectors: `outage_probability_overall_{csi}_run{idx}.npy`
  - Plots: `outage_probability_per_stream_{csi}_run{idx}.png`, `outage_probability_overall_{csi}_run{idx}.png`, `outage_probability_comparison.png`
  - **Note**: Plots include 6G target line at 1e-6

### 4. **Air Interface Latency** ⭐⭐⭐
- **Type**: Overall (vector)
- **Unit**: milliseconds (ms)
- **Description**: Physical layer transmission delay
- **Components**:
  - Encoding time (LDPC)
  - OFDM symbol transmission time
  - Decoding time (LDPC)
- **6G Target**: < 0.1 ms (air interface)
- **Files**:
  - Vectors: `air_interface_latency_ms_overall_{csi}_run{idx}.npy`
  - Plots: `air_interface_latency_ms_overall_{csi}_run{idx}.png`, `air_interface_latency_ms_comparison.png`
  - **Note**: Plots include 6G target line at 0.1 ms

### 5. **Energy per Bit (PHY)** ⭐⭐⭐
- **Type**: Overall (vector)
- **Unit**: picojoules per bit (pJ)
- **Description**: Physical layer energy consumption per successfully transmitted bit
- **Formula**: `Energy_per_Bit = Total_PHY_Energy / Successful_Bits`
- **Components**:
  - Encoding energy (baseband processing)
  - RF transmission energy
  - RF reception energy
  - Decoding energy (LDPC)
- **6G Target**: 1 pJ/bit
- **Files**:
  - Vectors: `energy_per_bit_pj_overall_{csi}_run{idx}.npy`
  - Plots: `energy_per_bit_pj_overall_{csi}_run{idx}.png`, `energy_per_bit_pj_comparison.png`
  - **Note**: Plots include 6G target line at 1.0 pJ/bit

---

## Implementation Details

### MetricsAccumulator Updates
- Added tracking for:
  - `latency_sum`, `latency_count`: Air interface latency
  - `energy_sum`: Total physical layer energy
  - `outage_count`: Outage events (SINR < threshold)
  - `snr_sum`, `snr_count`: SNR accumulation
- Computes:
  - SNR (per-stream and overall)
  - Channel Capacity (from SINR)
  - Outage Probability (from SINR)
  - Air Interface Latency (average)
  - Energy per Bit (from total energy and successful bits)

### Model.run_batch() Updates
- Measures latency using `time.time()` timestamps
- Estimates energy consumption:
  - Encoding: ~10 mW per Mbps
  - RF TX: ~200 mW
  - RF RX: ~100 mW
  - Decoding: ~50 mW per Mbps (scaled by iterations)
- Provides `noise_power` for SNR calculation
- Returns `latency_sec` and `energy_joules` per batch

### Plotting Enhancements
- All plots include proper units in y-axis labels
- 6G target lines added for critical metrics:
  - Latency: 0.1 ms target
  - Energy: 1 pJ/bit target
  - Outage: 1e-6 target
- Improved plot formatting (font sizes, legends, grid)
- Comparison plots show both Perfect and Imperfect CSI

---

## File Structure

```
results/
├── baseline_6g_simulation/          # Baseline example results
│   ├── matrices/                    # NumPy arrays (.npy)
│   │   ├── snr_db_per_stream_*.npy
│   │   ├── channel_capacity_*.npy
│   │   ├── outage_probability_*.npy
│   │   ├── air_interface_latency_ms_*.npy
│   │   └── energy_per_bit_pj_*.npy
│   ├── plots/                       # Visualization plots (.png)
│   │   ├── snr_db_*.png
│   │   ├── channel_capacity_*.png
│   │   ├── outage_probability_*.png
│   │   ├── air_interface_latency_ms_*.png
│   │   └── energy_per_bit_pj_*.png
│   └── simulation_results.json
└── run_YYYYMMDD_HHMMSS/            # Actual simulation results
    ├── matrices/
    ├── plots/
    └── simulation_results.json
```

---

## Usage Example

```python
import numpy as np

# Load SNR matrix
snr_matrix = np.load('results/run_20251115_013722/matrices/snr_db_per_stream_imperfect_run0.npy')
print(f"SNR matrix shape: {snr_matrix.shape}")  # [num_ebno, num_streams]

# Load channel capacity vector
capacity = np.load('results/run_20251115_013722/matrices/channel_capacity_overall_perfect_run0.npy')
print(f"Channel capacity: {capacity} bits/s/Hz")

# Load latency
latency = np.load('results/run_20251115_013722/matrices/air_interface_latency_ms_overall_imperfect_run0.npy')
print(f"Air interface latency: {latency} ms")

# Check against 6G targets
print(f"Meets 6G latency target (< 0.1 ms): {np.all(latency < 0.1)}")
```

---

## Testing

All metrics have been tested and verified:
- ✅ MetricsAccumulator initialization
- ✅ Baseline results generation
- ✅ Matrix and plot generation
- ✅ Comparison plots
- ✅ 6G target lines in plots

---

## Next Steps

The implementation is complete and ready for use. When running actual simulations:

1. Baseline run: `python main.py`
2. AI estimator run: `python main.py --scenario-profile 6g_ai_estimator --neural-weights artifacts/neural_channel_estimator.weights.h5`
3. Results will be saved in `results/`
4. All metrics (including new ones) will be computed, saved, and plotted automatically

---

## References

- Full metrics documentation: `docs/metrics/03_physical_layer.md`
- Quick reference: `docs/metrics/05_physical_layer_summary.md`
- Baseline results: `results/baseline_6g_simulation/`

