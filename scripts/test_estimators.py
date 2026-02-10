
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import tensorflow as tf
import numpy as np
import pytest
from src.models.model import Model
from src.components.config import SystemConfig

def run_test():
    print("Testing Channel Estimation Baselines...")
    
    # Fast configuration
    # Small number of symbols/antennas for quick test
    config = SystemConfig(
        scenario="umi",
        num_ofdm_symbols=14,
        num_bs_ant=4,
        num_ut=2,
        num_ut_ant=1,
        fft_size=128,
        channel_model_type="rayleigh"
    )
    
    estimators = ["perfect", "ls", "dft", "lmmse", "pso"]
    
    batch_size = 4
    ebno_db = 10.0 # High SNR to see estimation quality clearly
    
    results = {}
    
    for est_type in estimators:
        print(f"\n--- Testing {est_type.upper()} Estimator ---")
        try:
            is_perfect = (est_type == "perfect")
            model = Model(
                config=config,
                estimator_type="ls" if is_perfect else est_type, # Dummy estimator for perfect
                perfect_csi=is_perfect
            )
            
            # Run one batch
            result = model.run_batch(batch_size, ebno_db, include_details=True)
            
            # Calculate MSE of channel estimate if available
            # h: [batch, rx, tx, streams, sym, fft]
            h = result["channel"]
            
            if "channel_hat" in result:
                h_hat = result["channel_hat"]
            else:
                # Perfect CSI usually returns h as hat, or we can use h
                h_hat = h
            
            print(f"h shape: {h.shape}")
            print(f"h_hat shape: {h_hat.shape}")
            
            # Print sample
            # Batch 0, Rx 0, RxAnt 0, Tx 0, TxAnt 0, Sym 2 (Pilot), Subcarrier 0
            # Indices: [0, 0, 0, 0, 0, 2, 0]
            # Note: h has rx=1, rx_ant=4, tx=2, tx_ant=1.
            # h_hat has matching dimensions.
            
            val_h = h[0, 0, 0, 0, 0, 2, 0]
            val_hat = h_hat[0, 0, 0, 0, 0, 2, 0]
            # val_h and val_hat are numpy scalars already
            print(f"Sample h: {val_h}")
            val_h = h[0, 0, 0, 0, 0, 2, 0]
            if "channel_hat" in result:
                val_hat = h_hat[0, 0, 0, 0, 0, 2, 0]
                print(f"Sample h_hat: {val_hat}")
                print(f"Ratio: {val_h / (val_hat + 1e-9)}")
            
            print(f"Sample h: {val_h}")
            
            # Bits stats
            b = result["bits"]
            b_hat = result["bits_hat"]
            print(f"b mean: {np.mean(b)}")
            print(f"b_hat mean: {np.mean(b_hat)}")
            
            # Check if b and b_hat are identical/zero
            print(f"b sum: {np.sum(b)}")
            print(f"b_hat sum: {np.sum(b_hat)}")
            
            mse = tf.reduce_mean(tf.abs(h - h_hat)**2)
            ber = tf.reduce_mean(tf.cast(result["bits"] != result["bits_hat"], tf.float32))
            
            print(f"MSE: {mse:.6f}")
            print(f"BER: {ber:.6f}")
            results[est_type] = {"mse": float(mse), "ber": float(ber)}
            
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[est_type] = "FAILED"

    print("\n\n--- Summary ---")
    print(f"{'Estimator':<10} | {'MSE':<10} | {'BER':<10}")
    print("-" * 36)
    for est, res in results.items():
        if isinstance(res, dict):
            print(f"{est:<10} | {res['mse']:<10.6f} | {res['ber']:<10.6f}")
        else:
            print(f"{est:<10} | {res:<10} | {res:<10}")

if __name__ == "__main__":
    run_test()
