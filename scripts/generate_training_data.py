
import os
import argparse
import json
import h5py
import numpy as np
import tensorflow as tf
import pandas as pd
from itertools import product
from tqdm import tqdm

def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def compute_capacity(h_hat, noise_power_db=-100):
    """
    Compute sum-rate capacity for a multi-user MIMO channel.
    h_hat: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_subcarriers]
    Adjust shapes as needed based on dataset.
    """
    # Simplified capacity calculation: Sum log2(1 + SNR)
    # We assume simple beamforming or ZF.
    # For training the RM, we often want to maximize a utility function.
    
    # Placeholder: Use simple channel power for now as a proxy if full ZF is too heavy
    # capacity = np.sum(np.log2(1 + snr))
    pass

def generate_training_data(args):
    # Load Config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")
    config = load_config(config_path)
    
    # Load HDF5 Dataset (Single User Samples)
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    print(f"Loading source dataset: {args.input}")
    with h5py.File(args.input, "r") as f:
        # Load all data into memory (assuming it fits, otherwise use indices)
        # paths_a: [num_samples, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_steps]
        # We need to reconstruct the frequency domain channel from paths.
        # This is expensive. 
        # Alternatively, if the original script saved CIR or path parameters, we can use sionna functions to reconstruct.
        # But `generate_factory_dataset.py` saves `paths_a` and `paths_tau`.
        
        paths_a_ds = f["paths_a"][:]
        paths_tau_ds = f["paths_tau"][:]
        
        # Shapes:
        # a: [num_samples, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
        # tau: [num_samples, num_rx, num_tx, num_paths]
        
    num_source_samples = paths_a_ds.shape[0]
    print(f"Loaded {num_source_samples} single-user samples.")
    
    # System Param
    sys_params = config.get("system_params", {})
    fft_size = sys_params.get("fft_size", 512)
    subcarrier_spacing = sys_params.get("subcarrier_spacing", 15e3)
    num_ut = args.num_ut # Number of users to simulate in one scene
    
    # Output Lists
    data_records = []
    
    # Frequency array for CIR -> H(f)
    # frequencies = np.fft.fftfreq(fft_size, d=1/subcarrier_spacing) # Wrong, baseband
    # Simple OFDM: H[k] = sum(a * exp(-j * 2pi * k * subcarrier_spacing * tau))
    
    # Pre-compute frequency grid
    # k indices: -fft_size/2 to fft_size/2 or 0 to fft_size
    # Sionna usually centers around carrier.
    # Let's use a simplified channel characteristic: Sum of path powers.
    # If we want detailed fading, we must compute the exponential.
    
    print(f"Generating {args.num_scenes} multi-user scenes...")
    
    for scene_idx in tqdm(range(args.num_scenes)):
        # 1. Select N random users
        indices = np.random.choice(num_source_samples, num_ut, replace=False)
        
        scene_a = paths_a_ds[indices]     # [num_ut, ...]
        scene_tau = paths_tau_ds[indices] # [num_ut, ...]
        
        # 2. Compute Features (Channel Energy/Power Profile per user)
        # We want a feature vector for the CNN.
        # Simple feature: Average Channel Magnitude per Subcarrier (or just average power)
        # For full Freq selection, we need H(f).
        
        # Let's approximate H(f)'s energy without full reconstruction if possible, 
        # or just reconstructed on a coarse grid.
        # H(f) ~ sum(a_i * exp(-j 2pi tau_i f))
        # Power(f) ~ |Sum|^2.
        
        # For the Resource Manager, we need "Quality" of the user.
        # Let's simply compute the Total Channel Power for now.
        # Power = sum(|a|^2)
        
        # shape of scene_a: [num_ut, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, 1]
        # Squeeze last dim
        # Sum over num_rx (1), num_rx_ant, num_tx_ant, num_paths
        # We assume 1 RX (BS), 1 TX (UT) for reciprocity or vice versa.
        # Actually in `generate_factory_dataset.py`:
        # tx = Transmitter("BS"...)
        # rx = Receiver("RX"...) (The moving part)
        # paths = solver(scene) takes paths from TX to RX.
        # So "RX" is the variable one.
        # So `a` is [1, 1, 1, 64, paths, 1] roughly.
        # (1 RX unit, 1 RX ant, 1 TX unit, 64 TX ant) if downlink?
        # Let's check config: "direction": "uplink"
        # Wait, the script has `scene.add(tx)` as BS and `scene.add(rx)` as Mobile.
        # If it's Downlink (BS->UT), then yes.
        # If Uplink, we should swap. But the geometric path loss is reciprocal.
        
        # Let's just take Total Power = Sum(|a|^2)
        # This ignores fading dips but gives good proxy for "Geometry Geometry".
        # [num_ut]
        user_powers = np.sum(np.abs(scene_a)**2, axis=(1,2,3,4,5))
        
        # H_eff for capacity calculation?
        # That requires orthogonality check.
        # Ideally we compute H vectors and check correlation.
        
        # For this prototype:
        # We define "Optimal Resource Allocation" as:
        # Maximize Sum Rate.
        # Rate_i = log2(1 + SNR_i) if orthogonal?
        # If we schedule multiple, interference?
        # We assume standard MU-MIMO ZF:
        # If users are well-separated, we supports M users.
        # If correlated, we lose rank.
        
        # Simplified Logic for "Ground Truth":
        # 1. Users with High Power are good.
        # 2. Users with Low Correlation are good.
        
        # Since I cannot easily run full ZF in this script without full H construction
        # I will use a simplified "Total Power" metric for training the mask.
        # Target: Select top K users?
        # Or just "Active if Power > Threshold"?
        
        # Let's make it slightly non-trivial:
        # Active Mask = Top 4 users by Power.
        
        sorted_indices = np.argsort(user_powers)[::-1]
        best_mask = np.zeros(num_ut, dtype=int)
        
        # "Optimal" Strategy: Schedule top 50% users?
        # Or users with SNR > 0dB?
        # Let's say we want to schedule at most 4 users.
        users_to_schedule = min(4, num_ut)
        best_mask[sorted_indices[:users_to_schedule]] = 1
        
        # Compute Power Allocation (Waterfilling?)
        # For now: Equal power to active users.
        best_power = best_mask.astype(float) # 1.0 or 0.0
        
        # Construct Feature Vector
        # We need something frequency selective if we want the CNN to do Freq scheduling?
        # The CNN input is [num_ut, fft_size].
        # So we should expand the power across frequency?
        # Or just repeat flat fading?
        # To make it realistic, let's reconstruct coarse H(f).
        
        # Feature: [num_ut, fft_size]
        # Approximate: Power[u, k] = |sum a_p * exp(-j 2pi tau_p f_k)|^2
        # Frequencies f_k:
        f_k = np.linspace(-subcarrier_spacing*fft_size/2, subcarrier_spacing*fft_size/2, fft_size)
        
        # Reconstruct H for each user
        channel_energies = []
        for u in range(num_ut):
            # a: [paths] (squeezing antennas for now, just SISO equiv for feature)
            # We take mean over antennas for the "Energy Profile"
            a_u = scene_a[u].flatten() # This mixes paths and antennas, not quite right for phase
            tau_u = scene_tau[u].flatten()
            
            # Reduce to "effective paths" or just take dominant ones?
            # Correct way: Coherent sum per antenna, then magnitude, then mean over antennas.
            # a: [1, 1, 1, 64, paths, 1]
            # tau: [1, 1, paths]
            
            # Just take the first RX ant, first TX ant. SISO approximation for Feature.
            # a_siso = scene_a[u, 0, 0, 0, 0, :, 0]
            # tau_siso = scene_tau[u, 0, 0, :]
            
            # Loop over all antenna pairs takes too long.
            # Let's use the first antenna pair as representative of "Fading Profile".
            a_siso = scene_a[u].reshape(-1, scene_a.shape[-2]) # flatten ants, keep paths
            # Actually scene_a is [1, 1, 1, 64, 100, 1]
            # Let's take antenna 0 at TX.
            a_sub = scene_a[u, 0, 0, 0, :, 0] # [paths]
            tau_sub = scene_tau[u, 0, :]      # [paths]
            
            # H[k] = sum(a * exp(...))
            # [fft_size]
            # exponent: -1j * 2 * pi * f_k * tau
            # f_k: [fft], tau: [paths] -> outer product
            phase = -1j * 2 * np.pi * np.outer(tau_sub, f_k) # [paths, fft]
            h_f = np.sum(a_sub[:, None] * np.exp(phase), axis=0) # [fft]
            
            energy_f = np.abs(h_f)**2
            channel_energies.append(energy_f.tolist())
            
        record = {
            "sample_index": scene_idx,
            "channel_energy": channel_energies, # List of lists
            "active_ut_mask": best_mask.tolist(),
            "per_ut_power": best_power.tolist()
        }
        data_records.append(record)

    # Save to Parquet
    df = pd.DataFrame(data_records)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_parquet(args.output)
    print(f"Saved {len(df)} training records to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Training Data for Resource Manager")
    parser.add_argument("--input", type=str, default="data/factory_dataset_refactored.h5", help="Input HDF5 file from generate_factory_dataset.py")
    parser.add_argument("--output", type=str, default="data/rm_training_data.parquet", help="Output Parquet file")
    parser.add_argument("--num_scenes", type=int, default=100, help="Number of multi-user scenes to generate")
    parser.add_argument("--num_ut", type=int, default=8, help="Number of users per scene")
    
    args = parser.parse_args()
    generate_training_data(args)
