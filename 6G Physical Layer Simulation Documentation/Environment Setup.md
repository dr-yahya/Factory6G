# Step 1: Define Smart Factory Environment and 6G Requirements

## Objective
Establish the simulation environment and define 6G-specific performance requirements for a smart factory setting.

## Actions
1. **Model the Smart Factory Environment**:
   - Represent the smart factory as an indoor environment with high device density, including sensors, robots, automated guided vehicles (AGVs), and IoT devices.
   - Assume a factory floor size of approximately 100m x 100m to simulate a realistic indoor hotspot scenario.
   - Account for obstructions (e.g., metal machinery, walls) and reflective surfaces (e.g., metal panels) that cause multipath and non-line-of-sight (NLOS) conditions.
   - Include dynamic elements, such as moving AGVs with velocities between 0.5–2 m/s.

2. **Define 6G Performance Requirements**:
   - **Ultra-Low Latency**: Target end-to-end latency < 1 ms for ultra-reliable low-latency communications (URLLC) use cases, such as real-time robot control.
   - **Ultra-High Reliability**: Target block error rate (BLER) < 10^-5 to ensure robust communication for critical applications.
   - **Massive Connectivity**: Support 1000+ devices per km² (equivalent to 100 devices in a 100m x 100m factory) to accommodate sensors, actuators, and monitoring devices.
   - **High Throughput**: Support enhanced mobile broadband (eMBB) for applications like high-definition video monitoring, requiring data rates up to 1 Gbps.

3. **Select Frequency Band**:
   - Use the 28 GHz mmWave band for high bandwidth and compatibility with 5G/6G standards.
   - Optionally, explore sub-THz bands (e.g., 100 GHz) for future-proofing, as they offer even higher bandwidth but face challenges with path loss and penetration.
   - Bandwidth: Set to 400 MHz (scalable to 1 GHz) to support high data rates.

4. **Simulation Parameters**:
   - **Carrier Frequency**: 28 GHz (mmWave) as the primary choice, with provisions for testing at 100 GHz (sub-THz).
   - **Bandwidth**: 400 MHz to support high-throughput applications.
   - **Scenario**: Indoor factory, modeled as an indoor hotspot (InH) environment, extending the 3GPP 38.901 UMi model.
   - **Device Density**: 100 user terminals (UTs) in a 100m x 100m area, representing sensors, robots, and AGVs.
   - **Mobility**: UT velocities ranging from 0 m/s (static sensors) to 2 m/s (moving AGVs).

## Code Snippet for Environment Setup
Below is a Python code snippet to initialize the simulation environment using Sionna, based on the provided Multiuser MIMO OFDM code, with parameters tailored for a smart factory.

```python
import sionna.phy
import tensorflow as tf
import numpy as np

# Set random seed for reproducibility
sionna.phy.config.seed = 42

# Simulation parameters
scenario = "inh"  # Indoor hotspot, to be customized from UMi
carrier_frequency = 28e9  # 28 GHz mmWave
bandwidth = 400e6  # 400 MHz
num_ut = 100  # 100 devices in 100m x 100m factory
batch_size = 128  # For Monte Carlo simulations
ut_velocity_min = 0.0  # Static sensors
ut_velocity_max = 2.0  # Moving AGVs

# Define the UT antenna array (single antenna for simplicity)
ut_array = sionna.phy.channel.tr38901.Antenna(
    polarization="single",
    polarization_type="V",
    antenna_pattern="omni",
    carrier_frequency=carrier_frequency
)

# Define the BS antenna array (massive MIMO setup to be expanded in later steps)
bs_array = sionna.phy.channel.tr38901.AntennaArray(
    num_rows=1,
    num_cols=4,  # Placeholder, will increase for massive MIMO
    polarization="dual",
    polarization_type="VH",
    antenna_pattern="38.901",
    carrier_frequency=carrier_frequency
)

# Configure the channel model (customized indoor hotspot)
channel_model = sionna.phy.channel.tr38901.UMi(  # Placeholder, to be replaced with InH
    carrier_frequency=carrier_frequency,
    o2i_model="low",
    ut_array=ut_array,
    bs_array=bs_array,
    direction="uplink",
    enable_pathloss=True,  # Enable for realistic factory conditions
    enable_shadow_fading=True  # Enable for obstructions
)

# Generate topology for smart factory
topology = sionna.phy.channel.gen_single_sector_topology(
    batch_size=batch_size,
    num_ut=num_ut,
    scenario=scenario,
    min_ut_velocity=ut_velocity_min,
    max_ut_velocity=ut_velocity_max,
    area_size=[100, 100]  # 100m x 100m factory floor
)

# Set the topology
channel_model.set_topology(*topology)

# Visualize the topology
channel_model.show_topology()
```

## Notes on Code
- **Scenario**: The `scenario` is set to `"inh"` (indoor hotspot), but since Sionna’s default 3GPP 38.901 implementation may not include InH, we start with UMi and will customize it in Step 2 to reflect factory-specific conditions.
- **Pathloss and Shadow Fading**: Enabled to model realistic signal attenuation due to machinery and walls.
- **Topology**: The `gen_single_sector_topology` function is used with a 100m x 100m area and 100 UTs to simulate high device density.
- **Mobility**: Velocities are set to 0–2 m/s to account for static sensors and moving AGVs.

## Next Steps
This setup will be extended in subsequent steps to include a custom indoor channel model, massive MIMO, advanced OFDM numerology, and 6G-specific coding/modulation schemes.

## References
1. **3GPP TR 38.901**: "Study on channel model for frequencies from 0.5 to 100 GHz." Provides the UMi and InH models for indoor environments. Available at: https://www.3gpp.org/ftp/Specs/archive/38_series/38.901/
2. **ITU-R M.2160**: "Framework and overall objectives of the future development of IMT for 2030 and beyond." Defines 6G requirements, including latency, reliability, and connectivity. Available at: https://www.itu.int/rec/R-REC-M.2160
3. **Sionna Documentation**: Details on channel models, antenna configurations, and topology generation. Available at: https://nvlabs.github.io/sionna/phy/api/channel.wireless.html
4. **Z. Zhang et al., "6G Wireless Networks: Vision, Requirements, Architecture, and Key Technologies,"** IEEE Vehicular Technology Magazine, 2019. Discusses 6G requirements and mmWave/sub-THz bands. DOI: 10.1109/MVT.2019.2921394
5. **M. Katz et al., "6G and the Factory of the Future,"** 6G Flagship White Paper, 2020. Outlines 6G use cases in smart factories. Available at: https://www.6gflagship.com/
6. **T. S. Rappaport et al., "Millimeter Wave Mobile Communications for 5G Cellular: It Will Work!,"** IEEE Access, 2013. Provides insights into mmWave propagation, applicable to 6G factory settings. DOI: 10.1109/ACCESS.2013.2260813

These references provide the theoretical and practical foundation for defining the smart factory environment and 6G requirements.