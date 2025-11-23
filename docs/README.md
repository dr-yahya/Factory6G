# Factory6G Documentation

Welcome to the Factory6G project documentation. This project implements a 6G physical layer simulation for smart factory environments using the Sionna framework, with a focus on PSO-enhanced channel estimation.

## 📚 Documentation Structure

### Getting Started
- **[Quick Start Guide](01_Quick_Start.md)** - Get up and running in minutes
- **[Installation](02_Installation.md)** - Detailed installation instructions
- **[Project Overview](03_Project_Overview.md)** - Architecture and design philosophy

### Core Documentation
- **[System Architecture](04_System_Architecture.md)** - Component design and interactions
- **[Channel Estimation](05_Channel_Estimation.md)** - PSO-based channel estimation deep dive
- **[Simulation Scenarios](06_Simulation_Scenarios.md)** - Available scenarios and configuration

### Results and Analysis
- **[Performance Results](07_Performance_Results.md)** - Benchmark results and comparisons
- **[PSO vs Baseline Comparison](08_PSO_Baseline_Comparison.md)** - Detailed performance analysis

### Reference
- **[API Reference](09_API_Reference.md)** - Code documentation
- **[Configuration Guide](10_Configuration_Guide.md)** - Parameter tuning and optimization

## 🎯 Quick Links

**For Researchers:**
- [PSO Channel Estimation Theory](05_Channel_Estimation.md#theory)
- [Performance Results](07_Performance_Results.md)
- [Comparison Analysis](08_PSO_Baseline_Comparison.md)

**For Developers:**
- [System Architecture](04_System_Architecture.md)
- [API Reference](09_API_Reference.md)
- [Configuration Guide](10_Configuration_Guide.md)

**For Users:**
- [Quick Start](01_Quick_Start.md)
- [Installation](02_Installation.md)
- [Simulation Scenarios](06_Simulation_Scenarios.md)

## 📊 Key Results

- **48.2% average BER reduction** with PSO vs baseline LS Linear
- **12.7 dB average NMSE improvement** in channel estimation
- **Production-ready** implementation with Sionna framework
- **Scientifically validated** through matched-parameter comparison

## 👥 Project Team & Affiliation

**Author:**
**Yahya Khamayseh**
*PhD Student, Sunway University*

**Supervisor:**
**Prof. Ir. Rosdiadee Nordin**

**Affiliation:**
Faculty of Engineering and Technology
Sunway University | [www.sunway.edu.my](https://www.sunway.edu.my)

## 🙏 Acknowledgements

We acknowledge the support of the **Ministry of Higher Education, Malaysia**, through the **Fundamental Research Grant Scheme (FRGS)**.
*Ref: FRGS/1/2022/ICT09/SYUC/03/1*

## ⚠️ Disclaimer & Framework

This project is built upon and makes extensive use of **NVIDIA Sionna™**, an open-source library for link-level simulations.

While this project leverages Sionna for physical layer components (such as OFDM implementations and channel modeling), the PSO-enhanced channel estimation algorithms and specific Smart Factory scenarios described herein are custom implementations developed by the Factory6G authors. This project is not affiliated with or endorsed by NVIDIA.

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@software{factory6g2025,
  title={Factory6G: PSO-Enhanced Channel Estimation for 6G Smart Factory},
  author={Khamayseh, Yahya and Nordin, Rosdiadee},
  year={2025},
  note={Supported by FRGS/1/2022/ICT09/SYUC/03/1},
  url={[https://github.com/yourusername/Factory6G](https://github.com/yourusername/Factory6G)}
}
````

## 🤝 Contributing

See [CONTRIBUTING.md](https://www.google.com/search?q=../CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](https://www.google.com/search?q=../LICENSE) for details.

## 📧 Contact

For questions, collaboration, or inquiries regarding the implementation:

**Yahya Khamayseh**
Email: [23102254@imail.sunway.edu.my](mailto:23102254@imail.sunway.edu.my)
