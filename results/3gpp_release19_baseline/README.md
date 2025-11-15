# 3GPP Release 19 Baseline Results

This directory contains baseline/example results for 6G smart factory simulations based on **3GPP Release 19** (2024).

## 3GPP Release Information

- **Release**: Release 19
- **Year**: 2024
- **Description**: 5G-Advanced (Phase 2) - Bridge/Transition to 6G with early 6G research and requirement studies
- **Note**: Note: Release 19 is not part of 6G standards. It serves as a bridge to 6G. Release 20 will contain formal 6G Studies, and Release 21 will start normative 6G work (IMT-2030).

## 6G Standards Timeline (IMT-2030)

**Important**: Release 19 is **not part of 6G standards**. It serves as a bridge/transition to 6G.

- **Release 19** (2024): Bridge/Transition to 6G - Early 6G research and requirement studies (5G-Advanced Phase 2)
- **Release 20** (Expected 2026): Formal 6G Studies (IMT-2030)
- **Release 21** (Expected 2027+): Official start of normative 6G work - First 6G specifications

**Relevance**: Release 19 provides the transition baseline with early 6G research insights, making it relevant for 6G simulation studies while acknowledging it is still part of 5G-Advanced.

## Directory Structure

```
3gpp_release19_baseline/
├── matrices/          # NumPy arrays (.npy files) for each metric (3GPP Release 19)
├── plots/            # Visualization plots (.png files) for each metric (3GPP Release 19)
└── simulation_results.json  # Complete simulation results in JSON format
```

## Metrics

### Per-Stream Metrics (Matrices)
These metrics are computed per stream and saved as 2D matrices `[num_ebno_points, num_streams]`:
- **ber**: Bit Error Rate per stream
- **bler**: Block Error Rate per stream
- **throughput_bits**: Throughput in bits per stream
- **decoder_iter_avg**: Average LDPC decoder iterations per stream
- **sinr_db**: Signal-to-Interference-plus-Noise Ratio in dB per stream

### Overall Metrics (Vectors)
These metrics are computed overall and saved as 1D vectors `[num_ebno_points]`:
- **ber**: Overall Bit Error Rate
- **bler**: Overall Block Error Rate
- **nmse_db**: Normalized Mean Squared Error in dB (channel estimation quality)
- **evm_percent**: Error Vector Magnitude in percent (modulation quality)
- **sinr_db**: Overall Signal-to-Interference-plus-Noise Ratio in dB
- **decoder_iter_avg**: Average LDPC decoder iterations
- **throughput_bits**: Total throughput in bits
- **spectral_efficiency**: Bits per resource element
- **fairness_jain**: Jain's fairness index (0-1, higher is better)

## File Naming Convention

### Matrices
- Per-stream: `3gpp_release19_{metric}_per_stream_{csi}_run{idx}.npy`
  - Example: `3gpp_release19_ber_per_stream_imperfect_run0.npy`
- Overall: `3gpp_release19_{metric}_overall_{csi}_run{idx}.npy`
  - Example: `3gpp_release19_ber_overall_perfect_run0.npy`

### Plots
- Comparison: `3gpp_release19_{metric}_comparison.png` (both CSI conditions, 3GPP Release 19 baseline)
  - Example: `3gpp_release19_ber_comparison.png`

Where:
- `{metric}`: Metric name (ber, bler, sinr_db, etc.)
- `{csi}`: CSI condition (imperfect or perfect)
- `{idx}`: Run index (typically 0)

## Loading Results

### Load NumPy Arrays
```python
import numpy as np

# Load per-stream matrix
ber_matrix = np.load('matrices/3gpp_release19_ber_per_stream_imperfect_run0.npy')
print(f"Shape: {ber_matrix.shape}")  # [num_ebno, num_streams]

# Load overall vector
ber_vector = np.load('matrices/3gpp_release19_ber_overall_perfect_run0.npy')
print(f"Shape: {ber_vector.shape}")  # [num_ebno]
```

### Load JSON Results
```python
import json

with open('simulation_results.json', 'r') as f:
    results = json.load(f)

# Access metrics
for run in results['runs']:
    csi_str = "Perfect" if run['perfect_csi'] else "Imperfect"
    print(f"{csi_str} CSI:")
    for metric in run['metrics']:
        ebno = metric['ebno_db']
        ber = metric['overall']['ber']
        print(f"  Eb/No={ebno:.1f} dB: BER={ber:.3e}")
```

## Simulation Parameters

- **Scenario**: UMi (Urban Microcell)
- **Estimator**: LS (Least Squares channel estimation)
- **Eb/No Range**: -5 to 9 dB (step: 2 dB)
- **FFT Size**: 512
- **BS Antennas**: 32
- **User Terminals**: 8
- **UT Antennas**: 2
- **OFDM Symbols**: 14
- **Modulation**: QPSK (2 bits/symbol)
- **Code Rate**: 0.5

## Notes

- This is baseline/example data for demonstration purposes based on **3GPP Release 19**
- Actual simulation results will replace these when simulations are run
- All plots are saved at 300 DPI for publication quality
- Matrices use NumPy format for efficient storage and loading
- All metrics, plots, and files are labeled with 3GPP Release 19 for traceability

## References

- 3GPP Release 19 (2024): 5G-Advanced (Phase 2) - Bridge/Transition to 6G with early 6G research and requirement studies
- 6G (IMT-2030) Timeline: Release 20 (Formal Studies), Release 21 (Normative Work)
- For more information on 3GPP releases, visit: https://www.3gpp.org/specifications/releases
- IMT-2030 Framework: https://www.itu.int/en/ITU-R/study-groups/rsg5/rwp5d/imt-2030/Pages/default.aspx
