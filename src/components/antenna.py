"""
Antenna array configuration for 6G smart factory systems.

This module implements antenna array configurations for base stations and user
terminals, following 3GPP TR 38.901 specifications for MIMO systems.

Theory:
    Antenna arrays are essential for MIMO systems to achieve spatial diversity,
    beamforming, and spatial multiplexing gains. Key concepts:
    
    1. Antenna Array Geometry:
       - Linear arrays: Antennas arranged in a line
       - Planar arrays: Antennas arranged in a 2D grid (rows × columns)
       - The spacing between antennas affects the beam pattern and spatial
         resolution. Typical spacing: λ/2 (half wavelength) to avoid grating lobes
       
    2. Polarization:
       - Single polarization: All antennas have the same polarization (e.g., vertical)
       - Dual polarization: Antennas have two orthogonal polarizations (e.g., ±45°)
       - Polarization diversity: Exploits the fact that orthogonally polarized
         waves experience independent fading, providing diversity gain
       - For N physical antennas with dual polarization, we get 2N antenna ports
       
    3. Antenna Patterns:
       - Omni-directional: Equal gain in all directions (isotropic)
       - Directional: Higher gain in specific directions (beamforming)
       - 3GPP 38.901 pattern: Standardized antenna pattern with:
         * Horizontal and vertical beamwidths
         * Front-to-back ratio
         * Cross-polarization discrimination (XPD)
         
    4. Beamforming:
       - Phase and amplitude control across array elements
       - Creates constructive interference in desired directions
       - Suppresses interference from other directions
       - Beamforming gain: G_bf = 10*log10(N_ant) dB (theoretical maximum)
       
    5. Array Factor:
       The array factor determines the radiation pattern:
       AF(θ, φ) = Σ w_n * exp(j*k*r_n)
       where:
       - w_n: complex weight for element n
       - k: wave vector
       - r_n: position vector of element n
       
    For MIMO systems:
    - More antennas = higher spatial resolution = better interference suppression
    - Massive MIMO (100+ antennas) provides significant gains through:
      * Channel hardening: Channel becomes nearly deterministic
      * Favorable propagation: Inter-user interference vanishes asymptotically
      * Array gain: Improved SNR through coherent combining

References:
    - 3GPP TR 38.901: Antenna array models and patterns
    - Balanis, "Antenna Theory: Analysis and Design", 4th Edition
    - Marzetta et al., "Fundamentals of Massive MIMO"
"""

from sionna.phy.channel.tr38901 import AntennaArray
from .config import SystemConfig


class AntennaConfig:
    """
    Manages antenna array configurations for base stations and user terminals.
    
    This class creates and manages antenna arrays according to 3GPP TR 38.901
    specifications, supporting both single and dual polarization configurations.
    The antenna arrays are used by the channel model to compute channel responses
    with realistic antenna patterns and polarization effects.
    
    Theory:
        The antenna configuration determines:
        1. Array geometry (rows, columns)
        2. Polarization (single or dual)
        3. Antenna pattern (omni-directional or directional)
        4. Carrier frequency dependence
        
        These parameters affect the channel matrix H, which determines:
        - Path gain and fading
        - Spatial correlation
        - Polarization mismatch losses
        - Beamforming capabilities
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize antenna arrays for base station and user terminals.
        
        Creates separate antenna arrays for user terminals (UTs) and base
        station (BS) according to the system configuration. The arrays are
        configured with appropriate patterns and polarization for their roles.
        
        Args:
            config: System configuration parameters containing carrier frequency,
                number of BS antennas, and other relevant settings.
        """
        self.config = config
        self.ut_array = self._create_ut_array()
        self.bs_array = self._create_bs_array()
    
    def _create_ut_array(self) -> AntennaArray:
        """
        Create user terminal antenna array.
        
        User terminals typically use simple antenna configurations:
        - Single antenna (1×1 array) for cost and size constraints
        - Omni-directional pattern for coverage in all directions
        - Single vertical polarization (V)
        
        Theory:
            Mobile devices have limited space and power, so they typically use
            single antennas or small arrays. The omni-directional pattern ensures
            coverage regardless of device orientation, though this sacrifices
            directivity gain compared to directional antennas.
            
            For multi-antenna UTs, the array would be:
            - Small form factor (e.g., 2×1 or 2×2)
            - Dual polarization for diversity gain
            - Still relatively omni-directional due to device mobility
            
        Returns:
            AntennaArray configured for user terminals with:
            - 1 row × 1 column (single antenna)
            - Single vertical polarization
            - Omni-directional pattern
        """
        return AntennaArray(
            num_rows=1,
            num_cols=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.config.carrier_frequency
        )
    
    def _create_bs_array(self) -> AntennaArray:
        """
        Create base station antenna array.
        
        Base stations use more sophisticated antenna configurations:
        - Multiple antennas arranged in a planar array
        - Dual polarization (±45° cross-polarization) for polarization diversity
        - 3GPP 38.901 standardized directional pattern
        - Total antenna ports = num_cols × 2 (dual polarization)
        
        Theory:
            Base stations can accommodate larger, more complex antenna arrays:
            
            1. Dual Polarization:
               - Each physical antenna has two orthogonal polarizations
               - Provides polarization diversity gain (typically 3-6 dB)
               - Reduces correlation between antenna elements
               - For N physical antennas, we get 2N antenna ports
               
            2. 3GPP 38.901 Pattern:
               - Standardized antenna element pattern
               - Horizontal half-power beamwidth (HPBW): ~65°
               - Vertical HPBW: ~65°
               - Front-to-back ratio: ~30 dB
               - Cross-polarization discrimination: ~25 dB
               
            3. Array Geometry:
               - Linear array: num_rows=1, num_cols=N
               - Planar array: num_rows=M, num_cols=N
               - Element spacing: λ/2 (default in Sionna)
               - Total elements: num_rows × num_cols
               
            4. Beamforming:
               - With N antennas, can create beams with ~N× gain
               - Beamwidth inversely proportional to array size
               - Larger arrays = narrower beams = higher gain
               
        Returns:
            AntennaArray configured for base station with:
            - 1 row × (num_bs_ant/2) columns (assuming dual polarization)
            - Dual cross-polarization (±45°)
            - 3GPP 38.901 directional pattern
        """
        return AntennaArray(
            num_rows=1,
            num_cols=int(self.config.num_bs_ant / 2),  # Dual polarization
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=self.config.carrier_frequency
        )
    
    def get_ut_array(self) -> AntennaArray:
        """
        Get user terminal antenna array.
        
        Returns:
            AntennaArray object for user terminals, used by the channel model
            to compute channel responses with realistic antenna characteristics.
        """
        return self.ut_array
    
    def get_bs_array(self) -> AntennaArray:
        """
        Get base station antenna array.
        
        Returns:
            AntennaArray object for base station, used by the channel model
            to compute channel responses with realistic antenna characteristics
            including directional patterns and dual polarization.
        """
        return self.bs_array

