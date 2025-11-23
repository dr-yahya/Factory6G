# Quick Start Guide

Get up and running with Factory6G in minutes.

## Prerequisites

- Python 3.9+
- 8GB+ RAM
- Linux/WSL2 (recommended) or macOS

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Factory6G.git
cd Factory6G

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Run Your First Simulation

### Baseline Simulation (LS Linear Estimator)

```bash
python main.py --scenario-profile 6g_smart_factory_sionna_baseline
```

**Expected runtime:** ~6-8 hours  
**Output:** `results/6g_smart_factory_sionna_baseline/`

### PSO Enhanced Simulation

```bash
python main.py --scenario-profile 6g_pso_enhanced
```

**Expected runtime:** ~6-8 hours  
**Output:** `results/6g_pso_enhanced/`

## Generate Comparison Plot

```bash
python scripts/plot_comparison_stabilized.py
```

**Output:** `results/baseline_comparison/comparison_stabilized_*.png`

## View Results

Results are saved in JSON format with accompanying plots:

```
results/
├── 6g_smart_factory_sionna_baseline/
│   ├── simulation_results_*.json
│   └── simulation_plot_*.png
├── 6g_pso_enhanced/
│   ├── simulation_results_*.json
│   └── simulation_plot_*.png
└── baseline_comparison/
    └── comparison_stabilized_*.png
```

## Quick Configuration

Edit scenario parameters in:
- `src/sim/scenarios/6g_smart_factory_sionna_baseline.py`
- `src/sim/scenarios/6g_pso_enhanced.py`

Key parameters:
```python
batch_size=16          # Samples per iteration
max_iter=500           # Total iterations
target_block_errors=1000  # Convergence criterion
ebno_min=-4.0          # Minimum Eb/No (dB)
ebno_max=6.0           # Maximum Eb/No (dB)
```

## Next Steps

- Read [Project Overview](03_Project_Overview.md) for architecture details
- See [Channel Estimation](05_Channel_Estimation.md) for PSO theory
- Check [Performance Results](07_Performance_Results.md) for benchmarks

## Troubleshooting

**Out of memory?**
- Reduce `batch_size` to 8 or 4
- Reduce `max_iter` to 200

**Simulation too slow?**
- Reduce `max_iter` to 100 for quick tests
- Use `--cpu` flag to disable GPU warnings

**Import errors?**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again

## Support

- GitHub Issues: [Factory6G Issues](https://github.com/yourusername/Factory6G/issues)
- Documentation: [docs/README.md](README.md)
