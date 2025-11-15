#!/usr/bin/env python3
"""
Create baseline results structure with example metrics and plots for 6G simulation.

This script generates a baseline results directory structure with placeholder
matrices and plots to demonstrate the expected output format.
Based on 3GPP Release 19 (initiated 2024) - 5G-Advanced with early 6G research.
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 3GPP Release information
TGPP_RELEASE = "Release 19"
TGPP_RELEASE_YEAR = "2024"
TGPP_RELEASE_DESCRIPTION = "5G-Advanced (Phase 2) - Bridge/Transition to 6G with early 6G research and requirement studies"
TGPP_RELEASE_NOTE = "Note: Release 19 is not part of 6G standards. It serves as a bridge to 6G. Release 20 will contain formal 6G Studies, and Release 21 will start normative 6G work (IMT-2030)."

def create_baseline_results():
    """Create baseline results structure with example data based on 3GPP Release 19."""
    
    # Create baseline directory with 3GPP Release 19 naming
    baseline_dir = project_root / "results" / "3gpp_release19_baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    matrices_dir = baseline_dir / "matrices"
    plots_dir = baseline_dir / "plots"
    matrices_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print(f"Creating 3GPP {TGPP_RELEASE} Baseline Results")
    print(f"3GPP {TGPP_RELEASE} ({TGPP_RELEASE_YEAR}): {TGPP_RELEASE_DESCRIPTION}")
    print(f"{TGPP_RELEASE_NOTE}")
    print("=" * 80)
    print(f"Baseline directory: {baseline_dir}")
    print()
    
    # Simulation parameters (from max_params_config.json)
    ebno_db_range = np.arange(-5.0, 11.0, 2.0)  # -5, -3, -1, 1, 3, 5, 7, 9 dB
    num_ebno = len(ebno_db_range)
    num_streams = 16  # 8 UTs * 2 antennas = 16 streams
    
    # Create example data for both CSI conditions
    csi_conditions = [
        ("imperfect", False),
        ("perfect", True)
    ]
    
    # Metrics to process
    per_stream_metrics = ["ber", "bler", "throughput_bits", "decoder_iter_avg", "sinr_db",
                          "snr_db", "channel_capacity", "outage_probability"]
    overall_metrics = ["ber", "bler", "nmse_db", "evm_percent", "sinr_db", "snr_db",
                       "decoder_iter_avg", "throughput_bits", "spectral_efficiency", "fairness_jain",
                       "channel_capacity", "outage_probability", "air_interface_latency_ms", "energy_per_bit_pj"]
    
    print("Generating example matrices...")
    print()
    
    # Generate data for each CSI condition (save matrices only, no individual plots)
    for csi_str, perfect_csi in csi_conditions:
        print(f"Processing {csi_str.upper()} CSI condition...")
        
        # Generate per-stream metrics (matrices: [num_ebno, num_streams])
        for metric_name in per_stream_metrics:
            # Create realistic example data
            matrix = np.zeros((num_ebno, num_streams))
            
            if metric_name == "ber":
                # BER decreases exponentially with Eb/No
                base_ber = 1e-1 if not perfect_csi else 1e-2
                for i, ebno in enumerate(ebno_db_range):
                    # Add some variation across streams
                    stream_variation = 1 + 0.1 * np.random.randn(num_streams)
                    matrix[i, :] = base_ber * 10**(-ebno/10) * stream_variation
                    matrix[i, :] = np.clip(matrix[i, :], 1e-6, 1.0)
            elif metric_name == "bler":
                # BLER similar to BER but typically higher
                base_bler = 2e-1 if not perfect_csi else 5e-2
                for i, ebno in enumerate(ebno_db_range):
                    stream_variation = 1 + 0.15 * np.random.randn(num_streams)
                    matrix[i, :] = base_bler * 10**(-ebno/8) * stream_variation
                    matrix[i, :] = np.clip(matrix[i, :], 1e-5, 1.0)
            elif metric_name == "throughput_bits":
                # Throughput increases with Eb/No
                base_throughput = 1000 if not perfect_csi else 1500
                for i, ebno in enumerate(ebno_db_range):
                    stream_variation = 1 + 0.05 * np.random.randn(num_streams)
                    matrix[i, :] = base_throughput * (1 + ebno/10) * stream_variation
                    matrix[i, :] = np.maximum(matrix[i, :], 0)
            elif metric_name == "decoder_iter_avg":
                # Decoder iterations decrease with Eb/No (better channel = fewer iterations)
                base_iter = 8.0 if not perfect_csi else 4.0
                for i, ebno in enumerate(ebno_db_range):
                    stream_variation = 1 + 0.1 * np.random.randn(num_streams)
                    matrix[i, :] = base_iter * (1 - ebno/20) * stream_variation
                    matrix[i, :] = np.clip(matrix[i, :], 1.0, 20.0)
            elif metric_name == "sinr_db":
                # SINR increases with Eb/No
                base_sinr = -5.0 if not perfect_csi else 5.0
                for i, ebno in enumerate(ebno_db_range):
                    stream_variation = 1 + 0.05 * np.random.randn(num_streams)
                    matrix[i, :] = base_sinr + ebno * stream_variation
                    matrix[i, :] = np.clip(matrix[i, :], -20, 30)
            elif metric_name == "snr_db":
                # SNR is similar to SINR but typically higher (no interference)
                base_snr = 0.0 if not perfect_csi else 8.0
                for i, ebno in enumerate(ebno_db_range):
                    stream_variation = 1 + 0.05 * np.random.randn(num_streams)
                    matrix[i, :] = base_snr + ebno * stream_variation
                    matrix[i, :] = np.clip(matrix[i, :], -15, 35)
            elif metric_name == "channel_capacity":
                # Channel capacity: C = log2(1 + SINR)
                # Use SINR to compute capacity
                base_sinr_linear = 10**(((-5.0 if not perfect_csi else 5.0) + ebno_db_range) / 10)
                for i, ebno in enumerate(ebno_db_range):
                    sinr_linear = base_sinr_linear[i] * (1 + 0.05 * np.random.randn(num_streams))
                    sinr_linear = np.clip(sinr_linear, 0.01, 1000)
                    matrix[i, :] = np.log2(1 + sinr_linear)
            elif metric_name == "outage_probability":
                # Outage probability decreases with Eb/No
                base_outage = 1e-2 if not perfect_csi else 1e-4
                for i, ebno in enumerate(ebno_db_range):
                    stream_variation = 1 + 0.1 * np.random.randn(num_streams)
                    matrix[i, :] = base_outage * 10**(-ebno/5) * stream_variation
                    matrix[i, :] = np.clip(matrix[i, :], 1e-8, 1.0)
            
            # Save matrix with 3GPP Release 19 prefix
            matrix_file = matrices_dir / f"3gpp_release19_{metric_name}_per_stream_{csi_str}_run0.npy"
            np.save(matrix_file, matrix)
            print(f"  ✓ Saved matrix: {matrix_file.name} (shape: {matrix.shape})")
        
        # Generate overall metrics (vectors: [num_ebno])
        for metric_name in overall_metrics:
            # Create realistic example data
            if metric_name == "ber":
                base_ber = 1e-1 if not perfect_csi else 1e-2
                vector = base_ber * 10**(-ebno_db_range/10)
                vector = np.clip(vector, 1e-6, 1.0)
            elif metric_name == "bler":
                base_bler = 2e-1 if not perfect_csi else 5e-2
                vector = base_bler * 10**(-ebno_db_range/8)
                vector = np.clip(vector, 1e-5, 1.0)
            elif metric_name == "nmse_db":
                # NMSE decreases (improves) with Eb/No
                base_nmse = -10.0 if not perfect_csi else -20.0
                vector = base_nmse + ebno_db_range * 0.5
                vector = np.clip(vector, -40, 0)
            elif metric_name == "evm_percent":
                # EVM decreases with Eb/No
                base_evm = 15.0 if not perfect_csi else 5.0
                vector = base_evm * 10**(-ebno_db_range/20)
                vector = np.clip(vector, 0.1, 50.0)
            elif metric_name == "sinr_db":
                base_sinr = -5.0 if not perfect_csi else 5.0
                vector = base_sinr + ebno_db_range
                vector = np.clip(vector, -20, 30)
            elif metric_name == "snr_db":
                base_snr = 0.0 if not perfect_csi else 8.0
                vector = base_snr + ebno_db_range
                vector = np.clip(vector, -15, 35)
            elif metric_name == "channel_capacity":
                # Channel capacity: C = log2(1 + SINR)
                base_sinr_linear = 10**(((-5.0 if not perfect_csi else 5.0) + ebno_db_range) / 10)
                vector = np.log2(1 + np.clip(base_sinr_linear, 0.01, 1000))
            elif metric_name == "outage_probability":
                base_outage = 1e-2 if not perfect_csi else 1e-4
                vector = base_outage * 10**(-ebno_db_range/5)
                vector = np.clip(vector, 1e-8, 1.0)
            elif metric_name == "air_interface_latency_ms":
                # Latency decreases slightly with Eb/No (better channel = faster decoding)
                base_latency = 0.15 if not perfect_csi else 0.08  # ms
                vector = base_latency * (1 - ebno_db_range/30)
                vector = np.clip(vector, 0.05, 0.5)
            elif metric_name == "energy_per_bit_pj":
                # Energy per bit decreases with Eb/No (better channel = fewer retransmissions)
                base_energy = 2.0 if not perfect_csi else 0.8  # pJ/bit
                vector = base_energy * (1 - ebno_db_range/25)
                vector = np.clip(vector, 0.1, 5.0)
            elif metric_name == "decoder_iter_avg":
                base_iter = 8.0 if not perfect_csi else 4.0
                vector = base_iter * (1 - ebno_db_range/20)
                vector = np.clip(vector, 1.0, 20.0)
            elif metric_name == "throughput_bits":
                base_throughput = 10000 if not perfect_csi else 15000
                vector = base_throughput * (1 + ebno_db_range/10)
                vector = np.maximum(vector, 0)
            elif metric_name == "spectral_efficiency":
                # Bits per resource element
                base_se = 0.5 if not perfect_csi else 0.8
                vector = base_se * (1 + ebno_db_range/15)
                vector = np.clip(vector, 0, 2.0)
            elif metric_name == "fairness_jain":
                # Fairness index (0-1, higher is better)
                base_fairness = 0.85 if not perfect_csi else 0.95
                vector = base_fairness + 0.05 * np.sin(ebno_db_range/5)
                vector = np.clip(vector, 0.5, 1.0)
            
            # Save vector with 3GPP Release 19 prefix
            vector_file = matrices_dir / f"3gpp_release19_{metric_name}_overall_{csi_str}_run0.npy"
            np.save(vector_file, vector)
            print(f"  ✓ Saved vector: {vector_file.name} (shape: {vector.shape})")
    
    # Create one comparison plot per metric (both CSI conditions on same plot)
    print()
    print("Creating comparison plots (one per metric)...")
    for metric_name in overall_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for csi_str, perfect_csi in csi_conditions:
            # Load or regenerate the vector
            if metric_name == "ber":
                base_ber = 1e-1 if not perfect_csi else 1e-2
                metric_values = base_ber * 10**(-ebno_db_range/10)
                metric_values = np.clip(metric_values, 1e-6, 1.0)
            elif metric_name == "bler":
                base_bler = 2e-1 if not perfect_csi else 5e-2
                metric_values = base_bler * 10**(-ebno_db_range/8)
                metric_values = np.clip(metric_values, 1e-5, 1.0)
            elif metric_name == "nmse_db":
                base_nmse = -10.0 if not perfect_csi else -20.0
                metric_values = base_nmse + ebno_db_range * 0.5
                metric_values = np.clip(metric_values, -40, 0)
            elif metric_name == "evm_percent":
                base_evm = 15.0 if not perfect_csi else 5.0
                metric_values = base_evm * 10**(-ebno_db_range/20)
                metric_values = np.clip(metric_values, 0.1, 50.0)
            elif metric_name == "sinr_db":
                base_sinr = -5.0 if not perfect_csi else 5.0
                metric_values = base_sinr + ebno_db_range
                metric_values = np.clip(metric_values, -20, 30)
            elif metric_name == "snr_db":
                base_snr = 0.0 if not perfect_csi else 8.0
                metric_values = base_snr + ebno_db_range
                metric_values = np.clip(metric_values, -15, 35)
            elif metric_name == "channel_capacity":
                base_sinr_linear = 10**(((-5.0 if not perfect_csi else 5.0) + ebno_db_range) / 10)
                metric_values = np.log2(1 + np.clip(base_sinr_linear, 0.01, 1000))
            elif metric_name == "outage_probability":
                base_outage = 1e-2 if not perfect_csi else 1e-4
                metric_values = base_outage * 10**(-ebno_db_range/5)
                metric_values = np.clip(metric_values, 1e-8, 1.0)
            elif metric_name == "air_interface_latency_ms":
                base_latency = 0.15 if not perfect_csi else 0.08
                metric_values = base_latency * (1 - ebno_db_range/30)
                metric_values = np.clip(metric_values, 0.05, 0.5)
            elif metric_name == "energy_per_bit_pj":
                base_energy = 2.0 if not perfect_csi else 0.8
                metric_values = base_energy * (1 - ebno_db_range/25)
                metric_values = np.clip(metric_values, 0.1, 5.0)
            elif metric_name == "decoder_iter_avg":
                base_iter = 8.0 if not perfect_csi else 4.0
                metric_values = base_iter * (1 - ebno_db_range/20)
                metric_values = np.clip(metric_values, 1.0, 20.0)
            elif metric_name == "throughput_bits":
                base_throughput = 10000 if not perfect_csi else 15000
                metric_values = base_throughput * (1 + ebno_db_range/10)
                metric_values = np.maximum(metric_values, 0)
            elif metric_name == "spectral_efficiency":
                base_se = 0.5 if not perfect_csi else 0.8
                metric_values = base_se * (1 + ebno_db_range/15)
                metric_values = np.clip(metric_values, 0, 2.0)
            elif metric_name == "fairness_jain":
                base_fairness = 0.85 if not perfect_csi else 0.95
                metric_values = base_fairness + 0.05 * np.sin(ebno_db_range/5)
                metric_values = np.clip(metric_values, 0.5, 1.0)
            
            csi_label = "Perfect" if perfect_csi else "Imperfect"
            ax.plot(ebno_db_range, metric_values, marker='o', linewidth=2, 
                   markersize=8, label=f"{csi_label} CSI")
        
        ax.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=12)
        ylabel = metric_name.upper().replace('_', ' ')
        # Add units
        if metric_name == "sinr_db" or metric_name == "snr_db":
            ylabel += " (dB)"
        elif metric_name == "channel_capacity":
            ylabel += " (bits/s/Hz)"
        elif metric_name == "outage_probability":
            ylabel += " (probability)"
        elif metric_name == "air_interface_latency_ms":
            ylabel += " (ms)"
        elif metric_name == "energy_per_bit_pj":
            ylabel += " (pJ)"
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{ylabel} Comparison - 3GPP {TGPP_RELEASE} Baseline", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        if metric_name in ["ber", "bler", "outage_probability"]:
            ax.set_yscale('log')
        # Add 6G target lines
        if metric_name == "air_interface_latency_ms":
            ax.axhline(y=0.1, color='r', linestyle='--', linewidth=2, label='6G Target (< 0.1 ms)')
            ax.legend(fontsize=10)
        elif metric_name == "energy_per_bit_pj":
            ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='6G Target (1 pJ/bit)')
            ax.legend(fontsize=10)
        elif metric_name == "outage_probability":
            ax.axhline(y=1e-6, color='r', linestyle='--', linewidth=2, label='6G Target (< 1e-6)')
            ax.legend(fontsize=10)
        plt.tight_layout()
        plot_file = plots_dir / f"3gpp_release19_{metric_name}_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved comparison plot: {plot_file.name}")
    
    # Create example simulation_results.json
    example_results = {
        "baseline_type": "3GPP Release 19 (Bridge to 6G)",
        "3gpp_release": TGPP_RELEASE,
        "3gpp_release_year": TGPP_RELEASE_YEAR,
        "3gpp_release_description": TGPP_RELEASE_DESCRIPTION,
        "3gpp_release_note": TGPP_RELEASE_NOTE,
        "6g_standards_timeline": {
            "release_19": "Bridge/Transition to 6G - Early 6G research and requirement studies (5G-Advanced Phase 2)",
            "release_20": "Formal 6G Studies (IMT-2030)",
            "release_21": "Official start of normative 6G work - First 6G specifications"
        },
        "profile": "6G_Compliant",
        "scenario": "umi",
        "estimator": "ls",
        "ebno_db": ebno_db_range.tolist(),
        "config": {
            "fft_size": 512,
            "num_bs_ant": 32,
            "num_ut": 8,
            "num_ut_ant": 2,
            "num_ofdm_symbols": 14,
            "num_bits_per_symbol": 2,
            "coderate": 0.5
        },
        "runs": [
            {
                "perfect_csi": False,
                "metrics": [
                    {
                        "ebno_db": float(ebno),
                        "overall": {
                            "ber": float(1e-1 * 10**(-ebno/10)),
                            "bler": float(2e-1 * 10**(-ebno/8)),
                            "nmse_db": float(-10.0 + ebno * 0.5),
                            "evm_percent": float(15.0 * 10**(-ebno/20)),
                            "sinr_db": float(-5.0 + ebno),
                            "snr_db": float(0.0 + ebno),
                            "decoder_iter_avg": float(8.0 * (1 - ebno/20)),
                            "throughput_bits": int(10000 * (1 + ebno/10)),
                            "spectral_efficiency": float(0.5 * (1 + ebno/15)),
                            "fairness_jain": float(0.85 + 0.05 * np.sin(ebno/5)),
                            "channel_capacity": float(np.log2(1 + 10**((-5.0 + ebno)/10))),
                            "outage_probability": float(1e-2 * 10**(-ebno/5)),
                            "air_interface_latency_ms": float(0.15 * (1 - ebno/30)),
                            "energy_per_bit_pj": float(2.0 * (1 - ebno/25))
                        }
                    } for ebno in ebno_db_range
                ]
            },
            {
                "perfect_csi": True,
                "metrics": [
                    {
                        "ebno_db": float(ebno),
                        "overall": {
                            "ber": float(1e-2 * 10**(-ebno/10)),
                            "bler": float(5e-2 * 10**(-ebno/8)),
                            "nmse_db": float(-20.0 + ebno * 0.5),
                            "evm_percent": float(5.0 * 10**(-ebno/20)),
                            "sinr_db": float(5.0 + ebno),
                            "snr_db": float(8.0 + ebno),
                            "decoder_iter_avg": float(4.0 * (1 - ebno/20)),
                            "throughput_bits": int(15000 * (1 + ebno/10)),
                            "spectral_efficiency": float(0.8 * (1 + ebno/15)),
                            "fairness_jain": float(0.95 + 0.05 * np.sin(ebno/5)),
                            "channel_capacity": float(np.log2(1 + 10**((5.0 + ebno)/10))),
                            "outage_probability": float(1e-4 * 10**(-ebno/5)),
                            "air_interface_latency_ms": float(0.08 * (1 - ebno/30)),
                            "energy_per_bit_pj": float(0.8 * (1 - ebno/25))
                        }
                    } for ebno in ebno_db_range
                ]
            }
        ],
        "duration": 3600.0
    }
    
    results_file = baseline_dir / "simulation_results.json"
    with open(results_file, 'w') as f:
        json.dump(example_results, f, indent=2)
    print(f"  ✓ Saved example results JSON: {results_file.name}")
    
    # Create README
    readme_content = f"""# 3GPP {TGPP_RELEASE} Baseline Results

This directory contains baseline/example results for 6G smart factory simulations based on **3GPP {TGPP_RELEASE}** ({TGPP_RELEASE_YEAR}).

## 3GPP Release Information

- **Release**: {TGPP_RELEASE}
- **Year**: {TGPP_RELEASE_YEAR}
- **Description**: {TGPP_RELEASE_DESCRIPTION}
- **Note**: {TGPP_RELEASE_NOTE}

## 6G Standards Timeline (IMT-2030)

**Important**: {TGPP_RELEASE} is **not part of 6G standards**. It serves as a bridge/transition to 6G.

- **Release 19** ({TGPP_RELEASE_YEAR}): Bridge/Transition to 6G - Early 6G research and requirement studies (5G-Advanced Phase 2)
- **Release 20** (Expected 2026): Formal 6G Studies (IMT-2030)
- **Release 21** (Expected 2027+): Official start of normative 6G work - First 6G specifications

**Relevance**: {TGPP_RELEASE} provides the transition baseline with early 6G research insights, making it relevant for 6G simulation studies while acknowledging it is still part of 5G-Advanced.

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
- Per-stream: `3gpp_release19_{{metric}}_per_stream_{{csi}}_run{{idx}}.npy`
  - Example: `3gpp_release19_ber_per_stream_imperfect_run0.npy`
- Overall: `3gpp_release19_{{metric}}_overall_{{csi}}_run{{idx}}.npy`
  - Example: `3gpp_release19_ber_overall_perfect_run0.npy`

### Plots
- Comparison: `3gpp_release19_{{metric}}_comparison.png` (both CSI conditions, 3GPP Release 19 baseline)
  - Example: `3gpp_release19_ber_comparison.png`

Where:
- `{{metric}}`: Metric name (ber, bler, sinr_db, etc.)
- `{{csi}}`: CSI condition (imperfect or perfect)
- `{{idx}}`: Run index (typically 0)

## Loading Results

### Load NumPy Arrays
```python
import numpy as np

# Load per-stream matrix
ber_matrix = np.load('matrices/3gpp_release19_ber_per_stream_imperfect_run0.npy')
print(f"Shape: {{ber_matrix.shape}}")  # [num_ebno, num_streams]

# Load overall vector
ber_vector = np.load('matrices/3gpp_release19_ber_overall_perfect_run0.npy')
print(f"Shape: {{ber_vector.shape}}")  # [num_ebno]
```

### Load JSON Results
```python
import json

with open('simulation_results.json', 'r') as f:
    results = json.load(f)

# Access metrics
for run in results['runs']:
    csi_str = "Perfect" if run['perfect_csi'] else "Imperfect"
    print(f"{{csi_str}} CSI:")
    for metric in run['metrics']:
        ebno = metric['ebno_db']
        ber = metric['overall']['ber']
        print(f"  Eb/No={{ebno:.1f}} dB: BER={{ber:.3e}}")
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

- This is baseline/example data for demonstration purposes based on **3GPP {TGPP_RELEASE}**
- Actual simulation results will replace these when simulations are run
- All plots are saved at 300 DPI for publication quality
- Matrices use NumPy format for efficient storage and loading
- All metrics, plots, and files are labeled with 3GPP Release 19 for traceability

## References

- 3GPP {TGPP_RELEASE} ({TGPP_RELEASE_YEAR}): {TGPP_RELEASE_DESCRIPTION}
- 6G (IMT-2030) Timeline: Release 20 (Formal Studies), Release 21 (Normative Work)
- For more information on 3GPP releases, visit: https://www.3gpp.org/specifications/releases
- IMT-2030 Framework: https://www.itu.int/en/ITU-R/study-groups/rsg5/rwp5d/imt-2030/Pages/default.aspx
"""
    
    readme_file = baseline_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    print(f"  ✓ Saved README: {readme_file.name}")
    
    print()
    print("=" * 80)
    print("Baseline results created successfully!")
    print(f"Location: {baseline_dir}")
    print("=" * 80)

if __name__ == "__main__":
    create_baseline_results()

