
import os
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# Adjust path to import src modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure environment before importing TensorFlow-heavy modules
# Force CPU to avoid Metal plugin issues on specific TF operations
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
# Try to disable GPU visibility explicitly
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

from src.models.model import Model
from src.components.config import SystemConfig


def generate_dataset(
    output_path: str,
    samples: int,
    batch_size: int,
    scenario: str,
    min_ebno: float,
    max_ebno: float,
    seed: int,
    tries: int = 1
):
    # Set random seed
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    # Configure system
    config = SystemConfig(scenario=scenario)
    
    # Initialize model
    # We use LS estimation to get realistic noisy channel estimates
    model = Model(
        scenario=scenario,
        perfect_csi=False,
        config=config,
        estimator_type="ls" 
    )
    
    # Storage for data
    data_records = []
    
    print(f"Generating {samples} channel realizations with {tries} allocations each...")
    print(f"Scenario: {scenario}, Eb/No range: [{min_ebno}, {max_ebno}] dB")
    
    for i in tqdm(range(samples)):
        # Random Eb/No for this channel realization
        ebno = np.random.uniform(min_ebno, max_ebno)
        num_ut = model.config.num_ut
        
        # 1. Baseline / Probing Step
        # All users active, full power to get good channel estimate
        model.config.active_ut_mask = [1] * num_ut
        model.config.per_ut_power = [1.0] * num_ut
        
        # Generate NEW topology
        results_baseline = model.run_batch(1, ebno, include_details=True, regenerate_topology=True)
        
        # Extract features (The OBSERVATION for the decision maker)
        # We use the baseline estimate as the input for ALL subsequent trials on this topology
        h_hat_baseline = tf.cast(results_baseline["channel_hat"], tf.complex64)
        
        # Compute channel energy profile [num_ut, fft_size]
        # h_hat shape: [batch, num_rx, num_rx_ant, num_tx, num_streams, num_ofdm, fft_size]
        # (1, 1, 32, 8, 2, 14, 512)
        
        power = tf.abs(h_hat_baseline)**2
        # Sum over Rx receivers (axis 1) -> [1, 32, 8, 2, 14, 512]
        power = tf.reduce_sum(power, axis=1)
        # Sum over Rx antennas (axis 1) -> [1, 8, 2, 14, 512]
        power = tf.reduce_sum(power, axis=1)
        # Sum over Streams (axis 2) -> [1, 8, 14, 512]
        power = tf.reduce_sum(power, axis=2)
        # Avg over Time (axis 2) -> [1, 8, 512]
        channel_energy = tf.reduce_mean(power, axis=2)
        
        channel_energy_np = channel_energy[0].numpy() # [num_ut, fft_size]
        if i == 0:
            print(f"DEBUG: channel_energy_np shape: {channel_energy_np.shape}")
        
        # Function to process result and save
        
        # Function to process result and save
        def save_result(res, mask, power, try_idx):
            bits = res["bits"]
            bits_hat = res["bits_hat"]
            errors = np.not_equal(bits, bits_hat).astype(int)
            ber_per_ut = np.mean(errors, axis=(2, 3))[0] # [num_ut]
            
            record = {
                "scenario": scenario,
                "ebno_db": ebno,
                "sample_index": i,
                "try_index": try_idx,
                "channel_energy": channel_energy_np.tolist(),
                "active_ut_mask": mask,
                "per_ut_power": power,
                "ber": ber_per_ut.tolist(),
                "avg_ber": float(np.mean(ber_per_ut)),
                "latency": float(res["latency_sec"]),
                "energy": float(res["energy_joules"])
            }
            data_records.append(record)
            
        # Save baseline result (Try 0)
        save_result(results_baseline, model.config.active_ut_mask, model.config.per_ut_power, 0)
        
        # 2. Random Tries
        for t in range(1, tries):
            # Random Allocation
            # Randomly select number of active users (1 to num_ut)
            num_active = np.random.randint(1, num_ut + 1)
            active_indices = np.random.choice(num_ut, num_active, replace=False)
            mask = np.zeros(num_ut, dtype=int)
            mask[active_indices] = 1
            
            power = np.zeros(num_ut, dtype=float)
            power[active_indices] = np.random.uniform(0.1, 1.0, size=num_active)
            
            model.config.active_ut_mask = mask.tolist()
            model.config.per_ut_power = power.tolist()
            
            # Reuse topology!
            res_try = model.run_batch(1, ebno, include_details=True, regenerate_topology=False)
            
            save_result(res_try, mask.tolist(), power.tolist(), t)
            
    # Save to Parquet
    df = pd.DataFrame(data_records)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_parquet(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset for 6G Resource Allocation")
    parser.add_argument("--output", type=str, default="data/dataset.parquet", help="Output Parquet file path")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--scenario", type=str, default="umi", help="Channel scenario")
    parser.add_argument("--min-ebno", type=float, default=0.0, help="Min Eb/No (dB)")
    parser.add_argument("--max-ebno", type=float, default=20.0, help="Max Eb/No (dB)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tries", type=int, default=5, help="Allocations per channel realization")
    
    args = parser.parse_args()
    
    generate_dataset(
        args.output,
        args.samples,
        args.batch_size,
        args.scenario,
        args.min_ebno,
        args.max_ebno,
        args.seed,
        args.tries
    )

