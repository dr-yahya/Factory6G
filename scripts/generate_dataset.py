from __future__ import annotations

import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.sim.env import configure_env

configure_env(force_cpu=True, gpu_num=0)

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.models.model import Model
from src.models.resource_manager import ResourceDirectives
from src.sim.config import load_config


def _preprocess_channel_for_cnn(h_hat: tf.Tensor) -> np.ndarray:
    h_hat = tf.cast(h_hat, tf.complex64)
    power = tf.abs(h_hat) ** 2
    power = tf.reduce_mean(power, axis=1)
    power = tf.reduce_mean(power, axis=1)
    power = tf.reduce_mean(power, axis=2)
    channel_energy = tf.reduce_mean(power, axis=2)
    return channel_energy[0].numpy().astype(np.float32)


def _sample_random_allocation(num_ut: int) -> tuple[list[int], list[float]]:
    num_active = np.random.randint(1, num_ut + 1)
    active_indices = np.random.choice(num_ut, num_active, replace=False)
    mask = np.zeros(num_ut, dtype=np.int32)
    mask[active_indices] = 1
    power = np.zeros(num_ut, dtype=np.float32)
    power[active_indices] = np.random.uniform(0.1, 1.0, size=num_active).astype(np.float32)
    return mask.tolist(), power.tolist()


def _score_candidate(res: dict, latency_weight: float) -> tuple[float, float, float, float]:
    bits = res["bits"]
    bits_hat = res["bits_hat"]
    total_bits = float(bits.size)
    bit_errors = float(np.not_equal(bits, bits_hat).sum())
    avg_ber = bit_errors / total_bits if total_bits > 0 else 1.0
    throughput_eff = 1.0 - avg_ber
    latency_ms = float(res["latency_sec"]) * 1e3
    utility = throughput_eff - latency_weight * latency_ms
    return utility, avg_ber, throughput_eff, latency_ms


def _load_runtime_config(config_path: str, scenario: str) -> dict:
    app_config = load_config(config_path)
    runtime_config = app_config.system_runtime_config
    runtime_config["scenario"] = scenario.lower()
    return runtime_config


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
    config_path: str = "config.json",
):
    if batch_size != 1:
        raise ValueError("For supervised RM dataset generation, batch_size must be 1.")

    tf.random.set_seed(seed)
    np.random.seed(seed)

    system_config = _load_runtime_config(config_path, scenario)
    model = Model(config=system_config, perfect_csi=False, estimator_type="ls")
    data_records: list[dict] = []

    print(f"Generating {samples} channel realizations with {tries} candidate allocations each...")
    print(f"Scenario: {scenario}, Eb/No range: [{min_ebno}, {max_ebno}] dB")
    print("Feature extraction: runtime-aligned with CNNResourceManager.preprocess_channel()")

    for sample_index in tqdm(range(samples)):
        ebno = float(np.random.uniform(min_ebno, max_ebno))
        context = model.prepare_batch_context(batch_size=1, ebno_db=ebno, include_feedback=True)
        if context.feedback is None:
            raise RuntimeError("Expected precomputed feedback for dataset generation.")

        channel_energy_np = _preprocess_channel_for_cnn(context.feedback.h_hat)
        default_directives = model.default_directives()
        baseline_results = model.run_batch(context, directives=default_directives, include_details=True)
        best_mask = list(default_directives.active_ut_mask or [])
        best_power = list(default_directives.per_ut_power or [])
        best_utility, best_avg_ber, best_thr_eff, best_latency_ms = _score_candidate(
            baseline_results,
            latency_weight=latency_weight,
        )

        num_ut = int(model.get_config().get("num_ut", 8))
        for _ in range(max(0, tries - 1)):
            cand_mask, cand_power = _sample_random_allocation(num_ut)
            directives = ResourceDirectives(
                active_ut_mask=cand_mask,
                per_ut_power=cand_power,
                pilot_reuse_factor=1,
            )
            res_try = model.run_batch(context, directives=directives, include_details=True)
            util, avg_ber, thr_eff, lat_ms = _score_candidate(res_try, latency_weight=latency_weight)
            if util > best_utility:
                best_utility = util
                best_avg_ber = avg_ber
                best_thr_eff = thr_eff
                best_latency_ms = lat_ms
                best_mask = cand_mask
                best_power = cand_power

        data_records.append(
            {
                "scenario": scenario,
                "ebno_db": ebno,
                "sample_index": int(sample_index),
                "channel_energy": channel_energy_np.tolist(),
                "active_ut_mask": best_mask,
                "per_ut_power": best_power,
                "oracle_utility": float(best_utility),
                "oracle_avg_ber": float(best_avg_ber),
                "oracle_throughput_eff": float(best_thr_eff),
                "oracle_latency_ms": float(best_latency_ms),
                "oracle_candidates": int(tries),
            }
        )

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to write the RM training parquet dataset.") from exc

    df = pd.DataFrame(data_records)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Sionna-aligned dataset for 6G resource management CNN"
    )
    parser.add_argument("--config", type=str, default="config.json", help="Path to the config JSON file")
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
        output_path=args.output,
        samples=args.samples,
        batch_size=args.batch_size,
        scenario=args.scenario,
        min_ebno=args.min_ebno,
        max_ebno=args.max_ebno,
        seed=args.seed,
        tries=args.tries,
        latency_weight=args.latency_weight,
        config_path=args.config,
    )
