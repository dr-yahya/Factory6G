"""
Metrics accumulation and computation for 6G simulations.

This module provides the MetricsAccumulator class for collecting and
computing simulation metrics during Monte Carlo runs.
"""

import numpy as np
from typing import Optional


class MetricsAccumulator:
    """Accumulates per-batch metrics to compute BER/BLER and diagnostics."""

    def __init__(self, config: "SystemConfig"):
        """
        Initialize metrics accumulator.
        
        Args:
            config: SystemConfig instance with system parameters
        """
        self.config = config
        shape = (config.num_tx, config.num_streams_per_tx)
        self.bit_errors = np.zeros(shape, dtype=np.int64)
        self.bits_total = np.zeros(shape, dtype=np.int64)
        self.block_errors = np.zeros(shape, dtype=np.int64)
        self.blocks_total = np.zeros(shape, dtype=np.int64)
        self.success_bits = np.zeros(shape, dtype=np.int64)
        self.decoder_iter_sum = np.zeros(shape, dtype=np.float64)
        self.decoder_iter_count = np.zeros(shape, dtype=np.int64)
        self.nmse_num = 0.0
        self.nmse_den = 0.0
        self.evm_num = 0.0
        self.evm_den = 0.0
        self.sinr_sum = np.zeros(shape, dtype=np.float64)
        self.sinr_count = np.zeros(shape, dtype=np.int64)
        
        # New metrics: Physical layer latency, energy, outage, capacity, SNR
        self.latency_sum = 0.0  # Total latency in seconds
        self.latency_count = 0  # Number of latency measurements
        self.energy_sum = 0.0  # Total energy consumed in Joules
        self.outage_count = np.zeros(shape, dtype=np.int64)  # Outage events (SINR < threshold)
        self.outage_threshold_db = -5.0  # Outage threshold in dB
        self.snr_sum = np.zeros(shape, dtype=np.float64)  # SNR (without interference)
        self.snr_count = np.zeros(shape, dtype=np.int64)

    def update(self, batch: dict):
        """
        Update accumulator with results from a batch.
        
        Args:
            batch: Dictionary containing batch results with keys:
                - bits: transmitted bits
                - bits_hat: received bits
                - decoder_iterations: decoder iteration counts
                - channel: true channel
                - channel_hat: estimated channel
                - qam: transmitted QAM symbols
                - qam_hat: received QAM symbols
                - no_eff: effective noise power
                - noise_power: noise power (optional)
                - latency_sec: latency in seconds (optional)
                - energy_joules: energy in Joules (optional)
        """
        bits = batch["bits"]
        bits_hat = batch["bits_hat"]
        diff = np.not_equal(bits, bits_hat)
        bit_errors = diff.sum(axis=-1).sum(axis=0)
        self.bit_errors += bit_errors

        num_blocks_batch = bits.shape[0]
        bits_per_block = bits.shape[-1]
        self.bits_total += bits_per_block * num_blocks_batch
        self.blocks_total += num_blocks_batch

        block_error_mask = diff.any(axis=-1)
        self.block_errors += block_error_mask.astype(np.int64).sum(axis=0)
        success_counts = (~block_error_mask).sum(axis=0)
        self.success_bits += success_counts * bits_per_block

        decoder_iter = batch["decoder_iterations"]
        self.decoder_iter_sum += decoder_iter.sum(axis=0)
        self.decoder_iter_count += num_blocks_batch

        h = batch["channel"]
        h_hat = batch["channel_hat"]
        diff_h = h - h_hat
        self.nmse_num += np.sum(np.abs(diff_h) ** 2)
        self.nmse_den += np.sum(np.abs(h) ** 2) + 1e-12

        x = batch["qam"]
        x_hat = batch["qam_hat"]
        self.evm_num += np.sum(np.abs(x_hat - x) ** 2)
        self.evm_den += np.sum(np.abs(x) ** 2) + 1e-12

        no_eff = batch["no_eff"]
        signal_power = np.abs(x) ** 2
        sinr_linear = np.divide(
            signal_power,
            no_eff,
            out=np.zeros_like(signal_power, dtype=np.float64),
            where=no_eff > 0,
        )
        sinr_linear = sinr_linear.reshape(-1, self.config.num_tx, self.config.num_streams_per_tx)
        self.sinr_sum += sinr_linear.sum(axis=0)
        self.sinr_count += sinr_linear.shape[0]
        
        # Compute SNR (signal power / noise power, without interference)
        noise_power = batch.get("noise_power", None)
        if noise_power is None:
            noise_power = no_eff
        else:
            if np.isscalar(noise_power) or (isinstance(noise_power, np.ndarray) and noise_power.size == 1):
                noise_power = np.full_like(signal_power, float(noise_power))
            elif isinstance(noise_power, np.ndarray):
                if noise_power.shape != signal_power.shape:
                    try:
                        noise_power = np.broadcast_to(noise_power, signal_power.shape)
                    except:
                        noise_power = np.full_like(signal_power, float(np.mean(noise_power)))
        
        snr_linear = np.divide(
            signal_power,
            noise_power,
            out=np.zeros_like(signal_power, dtype=np.float64),
            where=noise_power > 0,
        )
        snr_linear = snr_linear.reshape(-1, self.config.num_tx, self.config.num_streams_per_tx)
        self.snr_sum += snr_linear.sum(axis=0)
        self.snr_count += snr_linear.shape[0]
        
        # Compute outage probability: P(SINR < threshold)
        sinr_db = np.where(sinr_linear > 0, 10 * np.log10(sinr_linear), -np.inf)
        outage_mask = sinr_db < self.outage_threshold_db
        self.outage_count += outage_mask.astype(np.int64).sum(axis=0)
        
        # Track latency if provided
        if "latency_sec" in batch:
            self.latency_sum += batch["latency_sec"]
            self.latency_count += 1
        
        # Track energy if provided
        if "energy_joules" in batch:
            self.energy_sum += batch["energy_joules"]

    def total_block_errors(self) -> int:
        """Return total number of block errors."""
        return int(self.block_errors.sum())

    def total_blocks(self) -> int:
        """Return total number of blocks."""
        return int(self.blocks_total.sum())

    def current_overall_bler(self) -> Optional[float]:
        """Return current overall BLER or None if no blocks processed."""
        total_blocks = self.total_blocks()
        if total_blocks == 0:
            return None
        return self.total_block_errors() / total_blocks

    def finalize(self) -> dict:
        """
        Finalize metrics computation and return results dictionary.
        
        Returns:
            Dictionary with 'per_stream', 'overall', and 'counts' keys
        """
        total_bits = self.bits_total.sum()
        total_blocks = self.blocks_total.sum()

        ber_per_stream = np.divide(
            self.bit_errors,
            self.bits_total,
            out=np.zeros_like(self.bit_errors, dtype=np.float64),
            where=self.bits_total > 0,
        )
        bler_per_stream = np.divide(
            self.block_errors,
            self.blocks_total,
            out=np.zeros_like(self.block_errors, dtype=np.float64),
            where=self.blocks_total > 0,
        )

        overall_ber = float(self.bit_errors.sum() / total_bits) if total_bits > 0 else None
        overall_bler = float(self.block_errors.sum() / total_blocks) if total_blocks > 0 else None

        nmse = self.nmse_num / self.nmse_den if self.nmse_den > 0 else None
        nmse_db = float(10 * np.log10(nmse)) if nmse and nmse > 0 else None

        evm_ratio = self.evm_num / self.evm_den if self.evm_den > 0 else None
        evm_rms = float(np.sqrt(evm_ratio)) if evm_ratio is not None else None
        evm_percent = float(evm_rms * 100) if evm_rms is not None else None

        sinr_linear = np.divide(
            self.sinr_sum,
            self.sinr_count,
            out=np.zeros_like(self.sinr_sum),
            where=self.sinr_count > 0,
        )
        with np.errstate(divide="ignore"):
            sinr_db = np.where(sinr_linear > 0, 10 * np.log10(sinr_linear), -np.inf)

        decoder_iter_avg_per_stream = np.divide(
            self.decoder_iter_sum,
            self.decoder_iter_count,
            out=np.zeros_like(self.decoder_iter_sum),
            where=self.decoder_iter_count > 0,
        )
        decoder_iter_avg = float(self.decoder_iter_sum.sum() / self.decoder_iter_count.sum()) if self.decoder_iter_count.sum() > 0 else None

        throughput_bits_per_stream = self.success_bits
        throughput_per_ut = throughput_bits_per_stream.sum(axis=1)
        fairness = None
        if np.any(throughput_per_ut > 0):
            fairness = float(
                (throughput_per_ut.sum() ** 2) /
                (len(throughput_per_ut) * np.sum(throughput_per_ut ** 2))
            )

        total_re = (
            total_blocks
            * self.config.num_ofdm_symbols
            * self.config.fft_size
        )
        spectral_eff = (
            float(throughput_bits_per_stream.sum() / total_re)
            if total_re > 0
            else None
        )
        
        # Compute SNR (Signal-to-Noise Ratio, without interference)
        snr_linear = np.divide(
            self.snr_sum,
            self.snr_count,
            out=np.zeros_like(self.snr_sum),
            where=self.snr_count > 0,
        )
        with np.errstate(divide="ignore"):
            snr_db = np.where(snr_linear > 0, 10 * np.log10(snr_linear), -np.inf)
        
        # Compute Channel Capacity: C = log2(1 + SINR)
        capacity_per_stream = np.where(
            sinr_linear > 0,
            np.log2(1 + sinr_linear),
            np.zeros_like(sinr_linear)
        )
        
        # Compute Outage Probability: P(SINR < threshold)
        outage_prob_per_stream = np.divide(
            self.outage_count,
            self.blocks_total,
            out=np.zeros_like(self.outage_count, dtype=np.float64),
            where=self.blocks_total > 0,
        )
        overall_outage_prob = (
            float(self.outage_count.sum() / total_blocks)
            if total_blocks > 0
            else None
        )
        
        # Compute Air Interface Latency (average)
        avg_latency_ms = (
            float(self.latency_sum / self.latency_count * 1000)
            if self.latency_count > 0
            else None
        )
        
        # Compute Energy per Bit (PHY): Energy / Successful Bits
        energy_per_bit_pj = None
        if self.energy_sum > 0 and throughput_bits_per_stream.sum() > 0:
            energy_per_bit_joules = self.energy_sum / throughput_bits_per_stream.sum()
            energy_per_bit_pj = float(energy_per_bit_joules * 1e12)  # Convert to pJ
        
        # Overall SNR
        overall_snr_db = (
            float(np.mean(snr_db[np.isfinite(snr_db)]))
            if np.any(np.isfinite(snr_db))
            else None
        )
        
        # Overall Channel Capacity
        overall_capacity = (
            float(np.mean(capacity_per_stream[capacity_per_stream > 0]))
            if np.any(capacity_per_stream > 0)
            else None
        )

        return {
            "per_stream": {
                "ber": ber_per_stream.tolist(),
                "bler": bler_per_stream.tolist(),
                "throughput_bits": throughput_bits_per_stream.tolist(),
                "decoder_iter_avg": decoder_iter_avg_per_stream.tolist(),
                "sinr_linear": sinr_linear.tolist(),
                "sinr_db": sinr_db.tolist(),
                "snr_linear": snr_linear.tolist(),
                "snr_db": snr_db.tolist(),
                "channel_capacity": capacity_per_stream.tolist(),
                "outage_probability": outage_prob_per_stream.tolist(),
            },
            "overall": {
                "ber": overall_ber,
                "bler": overall_bler,
                "nmse": nmse,
                "nmse_db": nmse_db,
                "evm_rms": evm_rms,
                "evm_percent": evm_percent,
                "sinr_db": float(np.mean(sinr_db[np.isfinite(sinr_db)])) if np.any(np.isfinite(sinr_db)) else None,
                "snr_db": overall_snr_db,
                "decoder_iter_avg": decoder_iter_avg,
                "throughput_bits": int(throughput_bits_per_stream.sum()),
                "spectral_efficiency": spectral_eff,
                "fairness_jain": fairness,
                "channel_capacity": overall_capacity,
                "outage_probability": overall_outage_prob,
                "air_interface_latency_ms": avg_latency_ms,
                "energy_per_bit_pj": energy_per_bit_pj,
            },
            "counts": {
                "bit_errors": int(self.bit_errors.sum()),
                "total_bits": int(total_bits),
                "block_errors": int(self.block_errors.sum()),
                "total_blocks": int(total_blocks),
                "outage_events": int(self.outage_count.sum()),
            },
        }

