"""
Main 6G smart factory physical layer model using component-based architecture
"""

import tensorflow as tf
import numpy as np
from sionna.phy import Block
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.utils import ebnodb2no

from ..components.config import SystemConfig
from ..components.antenna import AntennaConfig
from ..components.transmitter import Transmitter
from ..components.channel import ChannelModel
from ..components.receiver import Receiver
from ..components.estimators import NeuralChannelEstimator


class Model(Block):
    """
    Complete 6G smart factory physical layer model.
    
    This model composes transmitter, channel, and receiver components
    to simulate OFDM MIMO transmissions over 3GPP TR 38.901 channel models.
    """
    
    def __init__(
        self,
        scenario: str = "umi",
        perfect_csi: bool = False,
        config: SystemConfig | None = None,
        estimator_type: str = "ls",
        estimator_weights: str | None = None,
        estimator_kwargs: dict | None = None,
    ):
        """
        Initialize the complete system model.
        
        Args:
            scenario: Channel scenario ("umi", "uma", "rma")
            perfect_csi: Whether to use perfect channel state information
            config: Optional custom system configuration. If None, uses defaults.
        """
        super().__init__()
        
        # Initialize configuration
        if config is None:
            self.config = SystemConfig(scenario=scenario)
        else:
            self.config = config
            self.config.scenario = scenario
        
        self.perfect_csi = perfect_csi
        self.estimator_type = estimator_type
        
        # Setup resource grid
        rx_tx_association = self.config.get_rx_tx_association()
        self._rg = ResourceGrid(
            num_ofdm_symbols=self.config.num_ofdm_symbols,
            fft_size=self.config.fft_size,
            subcarrier_spacing=self.config.subcarrier_spacing,
            num_tx=self.config.num_tx,
            num_streams_per_tx=self.config.num_streams_per_tx,
            cyclic_prefix_length=self.config.cyclic_prefix_length,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=self.config.pilot_ofdm_symbol_indices
        )
        
        # Setup stream management
        self._sm = StreamManagement(rx_tx_association, self.config.num_streams_per_tx)
        
        # Initialize components
        self._antenna_config = AntennaConfig(self.config)
        self._transmitter = Transmitter(self.config, self._rg)
        self._channel = ChannelModel(self.config, self._antenna_config, self._rg)

        # Prepare optional channel estimator
        channel_estimator = None
        if not perfect_csi:
            estimator_kwargs = estimator_kwargs or {}
            if estimator_type.lower() == "ls":
                channel_estimator = None
            elif estimator_type.lower() == "neural":
                channel_estimator = NeuralChannelEstimator(
                    self.config,
                    self._rg,
                    weights_path=estimator_weights,
                    **estimator_kwargs,
                )
            else:
                raise ValueError(
                    f"Unsupported estimator_type '{estimator_type}'. "
                    "Supported: 'ls', 'neural'."
                )

        # Receiver needs encoder reference for LDPC decoder
        encoder = self._transmitter._encoder
        self._receiver = Receiver(
            self.config,
            self._rg,
            self._sm,
            encoder,
            perfect_csi=perfect_csi,
            channel_estimator=channel_estimator,
        )
    
    def new_topology(self, batch_size: int):
        """
        Generate and set new topology for the channel.
        
        Args:
            batch_size: Batch size for topology generation
        """
        self._channel.set_topology(batch_size)
    
    @tf.function
    def call(self, batch_size: int, ebno_db: float) -> tuple:
        """
        Simulate transmission through the complete system.
        
        Args:
            batch_size: Batch size for simulation
            ebno_db: Energy per bit to noise ratio in dB
            
        Returns:
            Tuple of (transmitted bits, received bits)
        """
        # Generate new topology
        self.new_topology(batch_size)
        
        # Calculate noise variance
        no = ebnodb2no(
            ebno_db,
            self.config.num_bits_per_symbol,
            self.config.coderate,
            self._rg
        )
        
        # Transmitter: Generate bits and map to resource grid
        x_rg, b = self._transmitter.call(batch_size)
        
        # Channel: Apply channel and noise
        y, h = self._channel(x_rg, no)
        
        # Receiver: Estimate channel, equalize, demap, and decode
        if self.perfect_csi:
            b_hat = self._receiver.process_with_perfect_csi(y, h, no)
        else:
            h_hat, err_var = self._receiver.estimate_channel(y, no)
            b_hat = self._receiver(y, h_hat, err_var, no)
        
        return b, b_hat
    
    def get_config(self) -> SystemConfig:
        """Get system configuration"""
        return self.config
    
    def get_transmitter(self) -> Transmitter:
        """Get transmitter component"""
        return self._transmitter
    
    def get_channel(self) -> ChannelModel:
        """Get channel model component"""
        return self._channel
    
    def get_receiver(self) -> Receiver:
        """Get receiver component"""
        return self._receiver
