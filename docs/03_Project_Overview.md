# Project Overview

Factory6G is a 6G physical layer simulation framework for smart factory environments, featuring PSO-enhanced channel estimation.

## Vision

Enable reliable, ultra-low latency wireless communication for Industry 4.0 applications through advanced channel estimation techniques.

## Key Features

- ✅ **PSO-Enhanced Channel Estimation** - 48.2% BER improvement over baseline
- ✅ **Sionna Integration** - Built on NVIDIA's state-of-the-art framework
- ✅ **Production Ready** - Validated through extensive simulations
- ✅ **Modular Architecture** - Easy to extend and customize
- ✅ **Comprehensive Metrics** - BER, BLER, SINR, NMSE, latency, energy

## Architecture Overview

```
Factory6G/
├── src/
│   ├── components/        # Physical layer components
│   │   ├── antenna.py     # Antenna configuration
│   │   ├── channel.py     # Channel models
│   │   ├── transmitter.py # LDPC encoding, QAM mapping
│   │   ├── receiver.py    # Equalization, decoding
│   │   └── estimators/    # Channel estimators (LS, PSO)
│   ├── models/            # System models
│   │   └── model.py       # End-to-end system
│   └── sim/               # Simulation framework
│       ├── scenarios/     # Predefined scenarios
│       ├── runner.py      # Simulation execution
│       └── plotting.py    # Visualization
├── scripts/               # Utility scripts
└── results/               # Simulation outputs
```

## Technology Stack

- **Framework:** NVIDIA Sionna 0.16+
- **Deep Learning:** TensorFlow 2.13+
- **Numerics:** NumPy 1.24+
- **Visualization:** Matplotlib 3.7+
- **Language:** Python 3.9+

## Use Cases

### Smart Factory Communication
- Machine-to-machine (M2M) communication
- Sensor data aggregation
- Real-time control systems
- Predictive maintenance

### Research Applications
- Channel estimation algorithms
- Physical layer optimization
- 6G system design
- Performance benchmarking

## Design Philosophy

1. **Modularity** - Components are independent and reusable
2. **Extensibility** - Easy to add new estimators or scenarios
3. **Reproducibility** - Fixed seeds and documented parameters
4. **Performance** - Optimized for both accuracy and speed
5. **Usability** - Clear APIs and comprehensive documentation

## Project Structure

See [System Architecture](04_System_Architecture.md) for detailed component descriptions.

## Roadmap

- [x] PSO channel estimator implementation
- [x] Baseline LS Linear comparison
- [x] Comprehensive performance evaluation
- [ ] Real-time implementation (FPGA/ASIC)
- [ ] Multi-user MIMO support
- [ ] Adaptive PSO parameters
- [ ] Neural network hybrid estimators

## Contributing

We welcome contributions! See CONTRIBUTING.md for guidelines.

## License

MIT License - see LICENSE file for details.
