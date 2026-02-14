
import os
import argparse
import numpy as np
import tensorflow as tf
import h5py
from tensorflow.keras import layers, models, optimizers

# Sionna imports
from sionna.phy.channel import OFDMChannel
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, PilotPattern, RemoveNulledSubcarriers
from sionna.phy.utils import flatten_last_dims, expand_to_rank

# We need to reconstruct the channel from the HDF5 paths
from sionna.phy.channel import cir_to_ofdm_channel, cir_to_time_channel

def load_dataset(filename):
    # This generator yields (H_noisy_LS, H_perfect) pairs
    # It reads the paths from HDF5 and computes the channel on the fly.
    
    with h5py.File(filename, 'r') as f:
        # Load all paths (if memory allows)
        paths_a = f['paths_a'][:] # [num_samples, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, 1]
        paths_tau = f['paths_tau'][:] # [num_samples, num_rx, num_tx, num_paths]
        freq = f.attrs['frequency']
        
    num_samples = paths_a.shape[0]
    
    # System Params (Should match config, but we hardcode for training stability or pass as args)
    # FFT Size: 512
    # Subcarrier Spacing: 30kHz or 15kHz? Config said 120kHz
    
    fft_size = 512
    subcarrier_spacing = 120e3
    num_ofdm_symbols = 14
    
    # Grid Re-creation
    # Need to match the Pilot Pattern used in simulation or expected.
    # Config: pilot_ofdm_symbol_indices: [2, 11]
    # pilot_pattern = PilotPattern(

        # # If we train per-link, we can treat each TX-RX pair independently.
        # # Let's train a model for a single link (SISO) or fixed MIMO size.
        # # The dataset is massive MIMO (64 BS ant).
        # # We can train on (BS_Ant, OFDM_Sym, Subcarrier) images?
        # # Or train a small CNN that runs per antenna?
        # # Usually Channel Estimation is done per-antenna in simple baselines,
        # # or joint if exploiting correlation.
        
        # # Let's assume we train on the full grid [Num_RX_Ant, Num_OFDM, FFT_Size] ?
        # # That's too big.
        # # Let's train on 2D grids [Num_OFDM, FFT_Size] treating antennas as batch/channels?
        
        # # Strategy: Input is [Batch, Num_OFDM, FFT_Size, 2] (Real/Imag)
        # # This is the LS estimate grid.
        # # Target is Perfect Channel [Batch, Num_OFDM, FFT_Size, 2].
        
        # # We need to simulate the pilot contamination/noise here.
        # num_tx = 1, # We interpret the input as "per stream" estimates
        # num_streams_per_tx = 1,
        # pilot_ofdm_symbol_indices = [2, 11],
        # pilot_subcarrier_indices = None, # Full band?
        # dc_null = True
    # )
    
    # Frequencies
    frequencies = tf.range(fft_size, dtype=tf.float32) * subcarrier_spacing
    frequencies = frequencies - (fft_size/2 * subcarrier_spacing) # Centered?
    
    # Generator
    def generator():
        for i in range(num_samples):
            # 1. Get Path Parameters
            # a: [1, 1, 1, 64, paths, 1]
            a = paths_a[i] # complex
            tau = paths_tau[i] # float
            
            # Squeeze dim 0 (RX) and 2 (TX) -> [1, 64, paths, 1]
            # Let's pick a random RX antenna and TX antenna to train on "Single Link" properties
            # Or yield all of them?
            
            # Selecting 1 random link to keep it simple and light
            rx_ant_idx = np.random.randint(0, a.shape[1])
            tx_ant_idx = np.random.randint(0, a.shape[3]) 
            
            a_link = a[0, rx_ant_idx, tx_ant_idx, :, 0] # [paths]
            tau_link = tau[0, :] # [paths]
            
            # 2. Compute Perfect Channel H(f, t)
            # Shapes for Sionna function:
            # a: [batch, num_rx, num_tx, num_paths, num_time_steps]
            # tau: [batch, num_rx, num_tx, num_paths]
            
            # Reshape for broadcasting
            # We want H[num_ofdm, fft_size]
            
            # Freq Response H[k] = sum(a * exp(-j2pi * tau * f_k))
            # Assume time-invariant for specific sample (Doppler=0 from dataset?)
            
            # f_grid: [fft_size]
            # tau: [paths]
            # phase: [paths, fft_size]
            
            # Shift tau? No, tau is delay.
            
            phase = -1j * 2 * np.pi * np.outer(tau_link, frequencies) # [paths, fft]
            h_freq = np.sum(a_link[:, None] * np.exp(phase), axis=0) # [fft]
            
            # Expand to OFDM symbols (assume constant over time for now as velocity=0)
            h_perfect = np.tile(h_freq[None, :], (num_ofdm_symbols, 1)) # [14, 512]
            
            # 3. Simulate LS Estimate (Add Noise + Masking)
            # LS Estimate exists only at Pilot Locations.
            # We can use interpolation or just Zero-Filled input?
            # Creating a "Noisy Grid" input.
            
            # Simple Noise
            noise_power_db = np.random.uniform(-40, -10) # High SNR to Low SNR
            noise_std = np.sqrt(10**(noise_power_db/10) / 2) # /2 for complex
            noise = (np.random.normal(0, noise_std, h_perfect.shape) + 
                     1j * np.random.normal(0, noise_std, h_perfect.shape))
            
            y_noisy = h_perfect + noise
            
            # Mask non-pilot symbols?
            # If we are training an estimator, we input the LS estimate at pilots,
            # and maybe zeros elsewhere? Or simple interpolation?
            
            # Let's assume input is full noisy grid (as if Data was also Pilots? No)
            # Correct: Input should be sparse (only pilots non-zero) or coarse interpolated.
            # Let's use simple Pilot Mask.
            
            mask = np.zeros_like(h_perfect, dtype=np.float32)
            mask[[2, 11], :] = 1.0 # Pilot symbols
            
            # Input: Noisy Estimates at Pilots, Zeros elsewhere
            h_ls_input = y_noisy * mask
            
            # Stack Real/Imag for CNN [14, 512, 2]
            inputs = np.stack([h_ls_input.real, h_ls_input.imag], axis=-1)
            targets = np.stack([h_perfect.real, h_perfect.imag], axis=-1)
            
            yield inputs, targets
            
    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(num_ofdm_symbols, fft_size, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(num_ofdm_symbols, fft_size, 2), dtype=tf.float32)
        )
    )

def create_estimator_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Simple SRCNN-like / U-Net architecture
    # Treating Time-Freq grid as image
    
    x = layers.Conv2D(64, (9, 9), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, (5, 5), padding='same', activation='relu')(x)
    x = layers.Conv2D(2, (5, 5), padding='same', activation='linear')(x) # Output Real/Imag
    
    # Residual Connection?
    # No, input is sparse/zeros. We want full reconstruction.
    
    model = models.Model(inputs=inputs, outputs=x)
    return model

def train_estimator(args):
    dataset = load_dataset(args.data)
    dataset = dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Validate split? (Skip for prototype, use all for train)
    
    model = create_estimator_model((14, 512, 2))
    
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    history = model.fit(dataset, epochs=args.epochs, steps_per_epoch=args.steps)
    
    model.save(args.output)
    print(f"Estimator saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/factory_dataset_refactored.h5", help="Path to HDF5")
    parser.add_argument("--output", type=str, default="models/channel_estimator.h5", help="Output model path")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    
    args = parser.parse_args()
    train_estimator(args)
