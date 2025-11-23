# API Reference

Quick reference for Factory6G programmatic interface.

## Core Classes

### Model

End-to-end system model.

```python
from src.models.model import Model

model = Model(
    scenario="umi",
    perfect_csi=False,
    config=None,  # Optional SystemConfig
    estimator_type="pso",
    estimator_kwargs={"degree": 3, "swarm_size": 30, "iters": 40}
)

# Run batch
results = model.run_batch(batch_size=16, ebno_db=0.0, include_details=True)

# Access results
ber = results['bits']
bler = results['bits_hat']
```

### SystemConfig

System configuration.

```python
from src.components.config import SystemConfig

config = SystemConfig(
    scenario="umi",
    carrier_frequency=3.5e9,
    fft_size=512,
    subcarrier_spacing=30e3,
    num_bs_ant=4,
    num_ut=4,
    channel_model_type="rayleigh"
)
```

### ScenarioSpec

Scenario specification.

```python
from src.sim.scenarios.spec import ScenarioSpec

spec = ScenarioSpec(
    name="my_scenario",
    estimators=["pso"],
    batch_size=16,
    max_iter=500,
    ebno_min=-4.0,
    ebno_max=6.0
)
```

## Estimators

### PSO Channel Estimator

```python
from src.components.estimators import PSOChannelEstimator

estimator = PSOChannelEstimator(
    config=config,
    resource_grid=rg,
    degree=3,
    swarm_size=30,
    iters=40,
    seed=42
)

h_hat, err_var = estimator(y, noise_variance)
```

## Simulation

### Run Simulation

```python
from src.sim.runner import run_simulation

results = run_simulation(
    scenario="umi",
    perfect_csi_list=[False],
    ebno_db_range=np.arange(-4.0, 7.0, 1.0),
    batch_size=16,
    max_mc_iter=500,
    estimator_type="pso",
    save_results=True,
    plot_results=True
)
```

### Metrics

```python
from src.sim.metrics import MetricsAccumulator

accumulator = MetricsAccumulator(config)
accumulator.update(batch_results)
metrics = accumulator.finalize()

# Access metrics
ber = metrics['overall']['ber']
bler = metrics['overall']['bler']
nmse = metrics['overall']['nmse_db']
```

## Plotting

```python
from src.sim.plotting import plot_simulation_results

plot_simulation_results(results, output_dir="results")
```

## Command Line

```bash
# List scenarios
python main.py --list-scenarios

# Run scenario
python main.py --scenario-profile 6g_pso_enhanced

# Custom parameters
python main.py --scenario umi --estimator pso --batch-size 16
```

For detailed documentation, see source code docstrings.
