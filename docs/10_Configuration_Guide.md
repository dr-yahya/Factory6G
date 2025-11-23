# Configuration Guide

## Parameter Tuning

### Simulation Parameters

**batch_size**
- **Range:** 4-32
- **Impact:** Statistics vs memory
- **Recommendation:** 16 for stability

**max_iter**
- **Range:** 100-1000
- **Impact:** Confidence vs time
- **Recommendation:** 500 for publication quality

**target_block_errors**
- **Range:** 200-2000
- **Impact:** Smoothness vs time
- **Recommendation:** 1000 for smooth curves

### PSO Parameters

**degree**
- **Range:** 2-5
- **Impact:** Smoothing vs flexibility
- **Recommendation:** 3 for most channels

**swarm_size**
- **Range:** 10-50
- **Impact:** Exploration vs speed
- **Recommendation:** 30 for production

**iters**
- **Range:** 20-60
- **Impact:** Convergence vs speed
- **Recommendation:** 40 for stability

## Performance Presets

### Fast (Testing)
```python
batch_size = 4
max_iter = 100
target_block_errors = 200
swarm_size = 20
iters = 30
```
**Runtime:** ~1-2 hours

### Balanced (Development)
```python
batch_size = 16
max_iter = 500
target_block_errors = 1000
swarm_size = 30
iters = 40
```
**Runtime:** ~6-8 hours

### High Quality (Publication)
```python
batch_size = 32
max_iter = 1000
target_block_errors = 2000
swarm_size = 40
iters = 50
```
**Runtime:** ~24-48 hours

## Troubleshooting

**Out of memory?**
- Reduce batch_size to 4
- Reduce max_iter to 200

**Simulation too slow?**
- Reduce max_iter to 100
- Reduce swarm_size to 20
- Reduce iters to 30

**Unstable results?**
- Increase batch_size to 32
- Increase max_iter to 1000
- Increase target_block_errors to 2000

## Best Practices

1. **Start small** - Use fast preset for testing
2. **Validate** - Check results make sense
3. **Scale up** - Use balanced/high quality for final results
4. **Document** - Record all parameters used
5. **Reproduce** - Use fixed seeds (seed=42)

## Next Steps

- [Simulation Scenarios](06_Simulation_Scenarios.md)
- [Performance Results](07_Performance_Results.md)
