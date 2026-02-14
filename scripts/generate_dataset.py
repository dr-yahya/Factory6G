
import os
import argparse
import json
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm

# Adjust path to import src modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure environment before importing TensorFlow-heavy modules
# Force CPU to avoid Metal plugin issues on specific TF operations
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

from src.models.model import Model


def _load_system_config(config_path: str = "config.json") -> dict:
    """Load system_params from config file to align dataset with runtime simulation."""
    if not os.path.exists(config_path):
        return {"scenario": "umi"}
    with open(config_path, "r", encoding="utf-8") as f:
        raw = f.read()
    # Minimal inline comment stripping to support JSONC config.
    cleaned = []
    in_string = False
    escaped = False
    i = 0
    while i < len(raw):
        ch = raw[i]
        nxt = raw[i + 1] if i + 1 < len(raw) else ""
        if in_string:
            cleaned.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "\"":
                in_string = False
            i += 1
            continue
        if ch == "\"":
            in_string = True
            cleaned.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            i += 2
            while i < len(raw) and raw[i] != "\n":
                i += 1
            continue
        cleaned.append(ch)
        i += 1
    try:
        cfg = json.loads("".join(cleaned))
    except Exception:
        return {"scenario": "umi"}
    return cfg.get("system_params", {"scenario": "umi"})


def _preprocess_channel_for_cnn(h_hat: tf.Tensor) -> np.ndarray:
    """
    Match CNNResourceManager.preprocess_channel() exactly.
    Input shape: [batch, num_rx, num_rx_ant, num_tx, num_streams, num_ofdm, fft_size]
    Output shape: [num_ut, fft_size] for batch=1
    """
    h_hat = tf.cast(h_hat, tf.complex64)
    power = tf.abs(h_hat) ** 2
    power = tf.reduce_mean(power, axis=1)  # mean over num_rx
    power = tf.reduce_mean(power, axis=1)  # mean over num_rx_ant
    power = tf.reduce_mean(power, axis=2)  # mean over num_streams
    channel_energy = tf.reduce_mean(power, axis=2)  # mean over num_ofdm
    return channel_energy[0].numpy().astype(np.float32)


def _sample_random_allocation(num_ut: int) -> tuple[list[int], list[float]]:
    """Generate one random feasible scheduling + power candidate."""
    num_active = np.random.randint(1, num_ut + 1)
    active_indices = np.random.choice(num_ut, num_active, replace=False)

    mask = np.zeros(num_ut, dtype=np.int32)
    mask[active_indices] = 1

    power = np.zeros(num_ut, dtype=np.float32)
    raw = np.random.uniform(0.1, 1.0, size=num_active).astype(np.float32)
    power[active_indices] = raw
    return mask.tolist(), power.tolist()


def _score_candidate(
    res: dict,
    latency_weight: float,
) -> tuple[float, float, float, float]:
    """
    Score one candidate result.
    Returns:
      utility, avg_ber, throughput_eff, latency_ms
    """
    bits = res["bits"]
    bits_hat = res["bits_hat"]
    total_bits = float(bits.size)
    bit_errors = float(np.not_equal(bits, bits_hat).sum())
    avg_ber = bit_errors / total_bits if total_bits > 0 else 1.0
    throughput_eff = 1.0 - avg_ber
    latency_ms = float(res["latency_sec"]) * 1e3
    utility = throughput_eff - latency_weight * latency_ms
    return utility, avg_ber, throughput_eff, latency_ms


def generate_dataset(
    output_path: str,
    samples: int,
    batch_size: int,
    scenario: str,
    min_ebno: float,
    max_ebno: float,
    seed: int,
    tries: int = 16,
    latency_weight: float = 0.002,
):
    if batch_size != 1:
        raise ValueError("For supervised RM dataset generation, batch_size must be 1.")

    # Set random seed
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    system_config = _load_system_config()
    system_config["scenario"] = scenario
    
    # Initialize model
    # We use LS estimation to get realistic noisy channel estimates
    model = Model(
        scenario=scenario,
        perfect_csi=False,
        config=system_config,
        estimator_type="ls" 
    )
    
    data_records = []
    
    print(f"Generating {samples} channel realizations with {tries} candidate allocations each...")
    print(f"Scenario: {scenario}, Eb/No range: [{min_ebno}, {max_ebno}] dB")
    print("Feature extraction: runtime-aligned with CNNResourceManager.preprocess_channel()")
    
    for i in tqdm(range(samples)):
        ebno = np.random.uniform(min_ebno, max_ebno)
        num_ut = int(model.config.get("num_ut", 8))
        
        # Baseline probe to generate topology + feature input for this sample.
        model.config["active_ut_mask"] = [1] * num_ut
        model.config["per_ut_power"] = [1.0] * num_ut
        results_baseline = model.run_batch(1, ebno, include_details=True, regenerate_topology=True)

        channel_energy_np = _preprocess_channel_for_cnn(results_baseline["channel_hat"])
        if i == 0:
            print(f"DEBUG: channel_energy_np shape: {channel_energy_np.shape}")

        best_mask = model.config["active_ut_mask"][:]
        best_power = model.config["per_ut_power"][:]
        best_utility, best_avg_ber, best_thr_eff, best_latency_ms = _score_candidate(
            results_baseline,
            latency_weight=latency_weight,
        )

        # Search for better labels on the same topology (oracle-style candidate selection).
        for _ in range(max(0, tries - 1)):
            cand_mask, cand_power = _sample_random_allocation(num_ut)
            model.config["active_ut_mask"] = cand_mask
            model.config["per_ut_power"] = cand_power
            res_try = model.run_batch(1, ebno, include_details=True, regenerate_topology=False)
            util, avg_ber, thr_eff, lat_ms = _score_candidate(res_try, latency_weight=latency_weight)

            if util > best_utility:
                best_utility = util
                best_avg_ber = avg_ber
                best_thr_eff = thr_eff
                best_latency_ms = lat_ms
                best_mask = cand_mask
                best_power = cand_power

        data_records.append({
            "scenario": scenario,
            "ebno_db": float(ebno),
            "sample_index": int(i),
            "channel_energy": channel_energy_np.tolist(),
            "active_ut_mask": best_mask,
            "per_ut_power": best_power,
            "oracle_utility": float(best_utility),
            "oracle_avg_ber": float(best_avg_ber),
            "oracle_throughput_eff": float(best_thr_eff),
            "oracle_latency_ms": float(best_latency_ms),
            "oracle_candidates": int(tries),
        })

    df = pd.DataFrame(data_records)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    df.to_parquet(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Sionna-aligned dataset for 6G resource management CNN")
    parser.add_argument("--output", type=str, default="data/dataset.parquet", help="Output Parquet file path")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=1, help="Must be 1 for per-sample supervised dataset")
    parser.add_argument("--scenario", type=str, default="umi", help="Channel scenario")
    parser.add_argument("--min-ebno", type=float, default=0.0, help="Min Eb/No (dB)")
    parser.add_argument("--max-ebno", type=float, default=20.0, help="Max Eb/No (dB)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tries", type=int, default=16, help="Allocation candidates evaluated per channel realization")
    parser.add_argument("--latency-weight", type=float, default=0.002, help="Latency penalty weight in oracle utility")
    
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
