# Quick Start Guide

## Running the Main Simulation

The `main.py` script provides a complete command-line interface for running system simulations.

### Basic Usage

```bash
# Run default simulation
python main.py
```

This will:
- Run simulations for UMi scenario
- Test both perfect and imperfect CSI
- Use Eb/No range from -5 to 15 dB (step 2)
- Generate BER/BLER plots
- Save results to `results/` directory

### Command-Line Options

#### Scenario Selection
```bash
# Use different channel scenarios
python main.py --scenario umi    # Urban Micro (default)
python main.py --scenario uma    # Urban Macro
python main.py --scenario rma    # Rural Macro
```

#### CSI Conditions
```bash
# Run only perfect CSI
python main.py --perfect-csi-only

# Run only imperfect CSI
python main.py --imperfect-csi-only

# Run both (default)
python main.py
```

#### Eb/No Range
```bash
# Custom Eb/No range
python main.py --ebno-min -5 --ebno-max 20 --ebno-step 1

# Quick test with fewer points
python main.py --ebno-min 0 --ebno-max 10 --ebno-step 5
```

#### Simulation Parameters
```bash
# Adjust batch size
python main.py --batch-size 256

# Change stopping criteria
python main.py --target-block-errors 500 --target-bler 1e-4

# Maximum iterations
python main.py --max-iter 500

# Select channel estimator (ls or neural)
python main.py --estimator neural --neural-weights artifacts/neural_channel_estimator.weights.h5
```

#### Output Control
```bash
# Skip plots
python main.py --no-plot

# Skip saving results
python main.py --no-save

# Custom output directory
python main.py --output-dir my_results
```

#### GPU Configuration
```bash
# Use specific GPU
python main.py --gpu 0

# Use CPU (if no GPU available)
# Set CUDA_VISIBLE_DEVICES="" before running
```

#### Reproducibility
```bash
# Set random seed
python main.py --seed 42
```

### Example Workflows

#### Quick Test Run
```bash
# Fast simulation with fewer points
python main.py \
    --ebno-min 0 \
    --ebno-max 10 \
    --ebno-step 5 \
    --batch-size 64 \
    --max-iter 100
```

#### Production Run
```bash
# Full simulation with all options
python main.py \
    --scenario umi \
    --ebno-min -5 \
    --ebno-max 20 \
    --ebno-step 1 \
    --batch-size 256 \
    --max-iter 2000 \
    --target-block-errors 1000 \
    --seed 42 \
    --output-dir results/production
```

#### Compare Scenarios
```bash
# Run UMi
python main.py --scenario umi --output-dir results/umi

# Run UMa
python main.py --scenario uma --output-dir results/uma

# Run RMa
python main.py --scenario rma --output-dir results/rma
```

### Output Files

The script generates:

1. **JSON Results** (`results/simulation_results_<scenario>_<estimator>_<timestamp>.json`)
   - Complete simulation data
   - BER and BLER for each Eb/No point
   - Configuration parameters
   - Simulation duration

2. **Plots** (`results/simulation_plot_<scenario>_<estimator>_<timestamp>.png/pdf`)
   - BER vs Eb/No plot
   - BLER vs Eb/No plot
   - Both perfect and imperfect CSI curves

### Reading Results

```python
import json

# Load results
with open('results/simulation_results_umi_20240101_120000.json', 'r') as f:
    results = json.load(f)

# Access data
ebno_db = results['ebno_db']
ber_perfect = results['ber'][0]  # First entry is perfect CSI
bler_imperfect = results['bler'][1]  # Second entry is imperfect CSI
```

### Troubleshooting

#### GPU Issues
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Force CPU usage
CUDA_VISIBLE_DEVICES="" python main.py
```

#### Memory Issues
```bash
# Reduce batch size
python main.py --batch-size 64

# Reduce Eb/No range
python main.py --ebno-min 0 --ebno-max 10
```

#### Import Errors
```bash
# Ensure you're in the project root
cd /path/to/Factory6G

# Check Python path
python -c "import sys; print(sys.path)"

# Install dependencies
pip install -r requirements.txt
```

### Integration with Notebooks

You can also use the simulation functions in your own scripts:

```python
from main import run_simulation, setup_gpu
import numpy as np

# Setup
setup_gpu(0)

# Run simulation
results = run_simulation(
    scenario="umi",
    perfect_csi_list=[True, False],
    ebno_db_range=np.arange(-5, 15, 2),
    batch_size=128
)

# Access results
print(f"BER (perfect CSI): {results['ber'][0]}")
print(f"BLER (imperfect CSI): {results['bler'][1]}")
```

### Training the Neural Channel Estimator

Before running simulations with the neural estimator, train and save weights:

```bash
# Lightweight CPU-friendly training run
python scripts/train_neural_estimator.py --num-batches 30 --batch-size 8 --epochs 2

# Custom output path (must end with .weights.h5)
python scripts/train_neural_estimator.py --output artifacts/custom_neural_estimator.weights.h5
```

### Comparing LS and Neural Estimators

```bash
python main.py \
    --imperfect-csi-only \
    --ebno-min 0 --ebno-max 5 --ebno-step 5 \
    --batch-size 32 --max-iter 20 --target-block-errors 20 \
    --estimator ls neural \
    --neural-weights artifacts/neural_channel_estimator.weights.h5 \
    --no-plot
```

See also: `docs/results/01_channel_estimation_comparison.md` for the latest comparison plots and numerical summary.

