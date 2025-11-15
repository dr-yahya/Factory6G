# Channel Estimation Methods Comparison

This script (`compare_estimators.py`) runs simulations for all available channel estimation methods, collects comprehensive metrics, and generates detailed comparison visualizations.

## Available Estimators

- **ls_nn**: LS with nearest neighbor interpolation
- **ls_lin**: LS with linear interpolation  
- **neural**: Neural network-based estimator (requires trained weights)
- **ls_smooth**: Smoothed LS estimator with 2D smoothing
- **ls_temporal**: Temporal EMA estimator

## Usage

### Basic Usage (All Estimators)

```bash
python scripts/compare_estimators.py --scenario umi
```

### Custom Eb/No Range

```bash
python scripts/compare_estimators.py \
    --scenario umi \
    --ebno-min -3 \
    --ebno-max 9 \
    --ebno-step 3
```

### Select Specific Estimators

```bash
python scripts/compare_estimators.py \
    --scenario umi \
    --estimators ls_nn ls_lin neural
```

### Full Options

```bash
python scripts/compare_estimators.py \
    --scenario umi \
    --estimators ls_nn ls_lin neural ls_smooth ls_temporal \
    --ebno-min -3 \
    --ebno-max 9 \
    --ebno-step 3 \
    --batch-size 128 \
    --max-iter 1000 \
    --target-block-errors 1000 \
    --target-bler 1e-3 \
    --output-dir results \
    --gpu 0 \
    --seed 42
```

## Output

The script generates:

1. **JSON Results File**: `results/estimator_comparison_{scenario}_{timestamp}.json`
   - Contains all metrics for all estimators
   - Structured format for further analysis

2. **Visualization Files**:
   - `comparison_ber_bler_{scenario}_{timestamp}.png/pdf` - BER and BLER comparison
   - `comparison_nmse_sinr_{scenario}_{timestamp}.png/pdf` - NMSE and SINR comparison
   - `comparison_decoder_evm_{scenario}_{timestamp}.png/pdf` - Decoder iterations and EVM
   - `comparison_dashboard_{scenario}_{timestamp}.png/pdf` - Comprehensive dashboard with all metrics

## Metrics Collected

- **BER** (Bit Error Rate)
- **BLER** (Block Error Rate)
- **NMSE** (Normalized Mean Squared Error) in dB
- **EVM** (Error Vector Magnitude) in %
- **SINR** (Signal-to-Interference-plus-Noise Ratio) in dB
- **Decoder Iterations** (Average LDPC decoder iterations)
- **Throughput** (Successful bits transmitted)
- **Spectral Efficiency**
- **Fairness** (Jain's fairness index)

## Notes

- The neural estimator requires trained weights at `artifacts/neural_channel_estimator.weights.h5`
- If neural weights are not found, the neural estimator will be skipped
- All simulations use imperfect CSI (channel estimation) for fair comparison
- Results are saved in both PNG (high resolution) and PDF formats

