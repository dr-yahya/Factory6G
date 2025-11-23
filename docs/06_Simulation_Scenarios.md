# Simulation Scenarios

## Available Scenarios

Factory6G provides two main scenarios for comparison:

### 1. 6G Sionna Baseline (`6g_smart_factory_sionna_baseline`)

**Description:** Baseline 6G smart factory simulation using LS Linear channel estimation.

**Key Parameters:**
```python
name = "6g_smart_factory_sionna_baseline"
estimators = ["ls_lin"]
batch_size = 16
max_iter = 500
target_block_errors = 1000
ebno_min = -4.0
ebno_max = 6.0
ebno_step = 1.0
channel_model_type = "rayleigh"
num_bs_ant = 4
num_ut = 4
num_ut_ant = 1
```

**Use Case:** Baseline performance reference

**Expected Runtime:** ~6-8 hours

### 2. 6G PSO Enhanced (`6g_pso_enhanced`)

**Description:** Enhanced 6G smart factory simulation using PSO-based channel estimation.

**Key Parameters:**
```python
name = "6g_pso_enhanced"
estimators = ["pso"]
batch_size = 16
max_iter = 500
target_block_errors = 1000
ebno_min = -4.0
ebno_max = 6.0
ebno_step = 1.0
channel_model_type = "rayleigh"
num_bs_ant = 4
num_ut = 4
num_ut_ant = 1

# PSO-specific parameters
estimator_kwargs = {
    "pso": {
        "degree": 3,
        "swarm_size": 30,
        "iters": 40,
        "inertia_start": 0.7,
        "inertia_end": 0.4,
        "c1": 1.5,
        "c2": 1.5,
        "seed": 42
    }
}
```

**Use Case:** Enhanced performance with PSO

**Expected Runtime:** ~6-8 hours

## Running Scenarios

### Command Line

```bash
# Run baseline
python main.py --scenario-profile 6g_smart_factory_sionna_baseline

# Run PSO enhanced
python main.py --scenario-profile 6g_pso_enhanced

# List available scenarios
python main.py --list-scenarios
```

### Programmatic

```python
from src.sim.scenarios import SCENARIO_PRESETS

# Get scenario
baseline_spec = SCENARIO_PRESETS["6g_smart_factory_sionna_baseline"]
pso_spec = SCENARIO_PRESETS["6g_pso_enhanced"]

# Access parameters
print(f"Batch size: {baseline_spec.batch_size}")
print(f"Estimators: {pso_spec.estimators}")
```

## Creating Custom Scenarios

### Step 1: Create Scenario File

Create `src/sim/scenarios/my_scenario.py`:

```python
from .spec import ScenarioSpec

SCENARIO = ScenarioSpec(
    name="my_scenario",
    description="My custom scenario",
    estimators=["ls_lin"],  # or ["pso"]
    batch_size=16,
    max_iter=500,
    target_block_errors=1000,
    ebno_min=-4.0,
    ebno_max=6.0,
    ebno_step=1.0,
    channel_model_type="rayleigh",
    num_bs_ant=4,
    num_ut=4,
    num_ut_ant=1,
    perfect_csi=[False],
    # Add custom parameters as needed
)
```

### Step 2: Register Scenario

Edit `src/sim/scenarios/__init__.py`:

```python
SCENARIO_PRESETS = {
    "6g_smart_factory_sionna_baseline": _import_scenario("6g_smart_factory_sionna_baseline"),
    "6g_pso_enhanced": _import_scenario("6g_pso_enhanced"),
    "my_scenario": _import_scenario("my_scenario"),  # Add this line
}
```

### Step 3: Run Custom Scenario

```bash
python main.py --scenario-profile my_scenario
```

## Scenario Parameters Reference

### Simulation Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `batch_size` | int | Samples per iteration | 16 |
| `max_iter` | int | Maximum iterations | 500 |
| `target_block_errors` | int | Convergence criterion | 1000 |
| `target_bler` | float | Early stopping BLER | 1e-5 |

### SNR Range

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `ebno_min` | float | Minimum Eb/No (dB) | -4.0 |
| `ebno_max` | float | Maximum Eb/No (dB) | 6.0 |
| `ebno_step` | float | Eb/No step (dB) | 1.0 |

### Channel Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `channel_scenario` | str | 3GPP scenario | "umi" |
| `channel_model_type` | str | Channel model | "rayleigh" |
| `min_ut_velocity` | float | Min UE velocity (m/s) | 0.0 |
| `max_ut_velocity` | float | Max UE velocity (m/s) | 0.0 |

### Antenna Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `num_bs_ant` | int | BS antennas | 4 |
| `num_ut` | int | User terminals | 4 |
| `num_ut_ant` | int | UE antennas | 1 |

### Estimator Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `estimators` | list | Estimator types | ["ls_lin"] |
| `estimator_kwargs` | dict | Estimator parameters | {} |
| `perfect_csi` | list | CSI conditions | [False] |

## Parameter Tuning Guidelines

### For Speed
```python
batch_size = 4           # Reduce
max_iter = 100           # Reduce
target_block_errors = 200  # Reduce
```

### For Accuracy
```python
batch_size = 32          # Increase
max_iter = 1000          # Increase
target_block_errors = 2000  # Increase
```

### For Stability
```python
batch_size = 16          # Balanced
max_iter = 500           # Balanced
target_block_errors = 1000  # Balanced
```

## Output Structure

Results are saved in `results/{scenario_name}/`:

```
results/6g_pso_enhanced/
├── simulation_results_*.json    # Detailed metrics
├── simulation_plot_*.png        # BER/BLER plots
└── simulation_plot_*.pdf        # Vector plots
```

## Next Steps

- [Performance Results](07_Performance_Results.md) - Benchmark results
- [Configuration Guide](10_Configuration_Guide.md) - Advanced tuning
- [API Reference](09_API_Reference.md) - Programmatic access
