from abc import ABC, abstractmethod
from sionna.channel import OFDMChannel, AWGN, RayleighBlockFading, KroneckerModel
from sionna.utils import complex_normal
import tensorflow as tf

class ChannelInterface(ABC):
    @abstractmethod
    def apply_channel(self, symbols, noise_variance):
        pass

    @abstractmethod
    def get_channel_response(self):
        pass

class ChannelModel(ChannelInterface):
    def __init__(self, num_tx, num_rx, num_tx_ant, num_rx_ant, resource_grid, correlation_distance):
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.num_tx_ant = num_tx_ant
        self.num_rx_ant = num_rx_ant
        self.resource_grid = resource_grid
        self.correlation_distance = correlation_distance

        self.rayleigh_channel = RayleighBlockFading(
            num_rx=num_rx, num_rx_ant=num_rx_ant, num_tx=num_tx, num_tx_ant=num_tx_ant, dtype=tf.complex64
        )
        self.spatial_corr = self._create_spatial_correlation()
        self.ofdm_channel = OFDMChannel(
            channel_model=self.rayleigh_channel,
            resource_grid=resource_grid,
            add_awgn=False,
            normalize_channel=False,
            return_channel=True,
            dtype=tf.complex64
        )
        self.awgn = AWGN(dtype=tf.complex64)

    def _create_spatial_correlation(self):
        def manual_exp_corr_mat(size, rho):
            indices = tf.range(size, dtype=tf.float32)
            i, j = tf.meshgrid(indices, indices)
            distances = tf.abs(i - j)
            corr_matrix = tf.pow(rho, distances)
            return tf.cast(corr_matrix, tf.complex64)

        r_tx = manual_exp_corr_mat(self.num_tx_ant, self.correlation_distance)
        r_rx = manual_exp_corr_mat(self.num_rx_ant, self.correlation_distance)
        return KroneckerModel(r_tx=r_tx, r_rx=r_rx)

    def apply_channel(self, symbols, noise_variance):
        channel_output, h_freq = self.ofdm_channel(symbols)
        noise = complex_normal(tf.shape(channel_output), var=noise_variance, dtype=tf.complex64)
        noisy_output = channel_output + noise
        return noisy_output, h_freq

    def get_channel_response(self):
        return self.rayleigh_channel(), self.spatial_corr
    
import tensorflow as tf
import sionna as sn

# Create resource grid with subcarrier spacing
rg = sn.ofdm.ResourceGrid(
    num_ofdm_symbols=14,
    fft_size=76,
    subcarrier_spacing=60e3  # 60 kHz
)

# Create channel instance
channel = ChannelModel(
    num_tx=1,
    num_rx=1,
    num_tx_ant=2,
    num_rx_ant=2,
    resource_grid=rg,
    correlation_distance=0.5
)

# Define symbols with correct shape: [batch_size, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
batch_size = 1
num_tx = 1
num_tx_ant = 2
num_ofdm_symbols = 14
num_subcarriers = 76

# Create complex symbols (real and imaginary parts)
real_part = tf.ones([batch_size, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers])
imag_part = tf.zeros([batch_size, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers])
symbols = tf.complex(real_part, imag_part)  # Shape: [1, 1, 2, 14, 76]

# Define noise variance
noise_var = 0.1

# Apply channel
output, h_freq = channel.apply_channel(symbols, noise_var)

# Print shapes for debugging
print("Symbols shape:", symbols.shape)
print("Output shape:", output.shape)
print("H_freq shape:", h_freq.shape)