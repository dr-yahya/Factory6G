from __future__ import annotations

import time

import numpy as np
import tensorflow as tf
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import LSChannelEstimator, ResourceGrid
from sionna.phy.utils import ebnodb2no

from factory6g.sim.types import BatchContext, ResourceManagerFeedback

from ..components.antenna import AntennaConfig
from ..components.channel import ChannelModel
from ..components.receiver import Receiver, apply_stream_mask
from ..components.transmitter import Transmitter
from ..components.estimators import PSOChannelEstimator
from .resource_manager import ResourceDirectives


class Model:
    """End-to-end PHY model with explicit batch contexts for fair Monte Carlo reuse."""

    def __init__(
        self,
        config: dict[str, object],
        perfect_csi: bool = False,
        estimator_type: str = "ls",
        estimator_kwargs: dict | None = None,
    ) -> None:
        self.config = config.copy()
        self.perfect_csi = perfect_csi
        self.estimator_type = estimator_type
        self.estimator_kwargs = estimator_kwargs or {}

        num_ofdm_symbols = int(self.config.get("num_ofdm_symbols", 14))
        fft_size = int(self.config.get("fft_size", 512))
        subcarrier_spacing = float(self.config.get("subcarrier_spacing", 30e3))
        num_tx = int(self.config.get("num_ut", 8))
        num_streams_per_tx = int(self.config.get("num_ut_ant", 1))
        cyclic_prefix_length = int(self.config.get("cyclic_prefix_length", 20))
        pilot_ofdm_symbol_indices = self.config.get("pilot_ofdm_symbol_indices", [2, 11])

        rx_tx_association = np.zeros([1, num_tx], dtype=np.int32)
        rx_tx_association[0, :] = 1

        self._rg = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=subcarrier_spacing,
            num_tx=num_tx,
            num_streams_per_tx=num_streams_per_tx,
            cyclic_prefix_length=cyclic_prefix_length,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
        )

        self._sm = StreamManagement(rx_tx_association, num_streams_per_tx)

        self.graph_mode = bool(self.config.get("graph_mode", False))
        self._decode_fn = None

        self._antenna_config = AntennaConfig(self.config)
        self._transmitter = Transmitter(self.config, self._rg)
        self._channel = ChannelModel(self.config, self._antenna_config, self._rg)
        self._receiver = Receiver(
            self.config,
            self._rg,
            self._sm,
            self._transmitter._encoder,
            perfect_csi=perfect_csi,
            channel_estimator=self._build_channel_estimator(),
        )

    def _build_channel_estimator(self):
        if self.perfect_csi:
            return None
        estimator_type = self.estimator_type.lower()
        if estimator_type in {"ls", "ls_nn", "ls-nn"}:
            return LSChannelEstimator(self._rg, interpolation_type="nn")
        if estimator_type in {"ls_lin", "ls-lin", "ls_linear"}:
            return LSChannelEstimator(self._rg, interpolation_type="lin")
        if estimator_type in {"pso", "dso"}:
            return PSOChannelEstimator(self.config, self._rg, **self.estimator_kwargs)
        if estimator_type in {"dft", "dft-based"}:
            from ..components.estimators import DFTChannelEstimator

            return DFTChannelEstimator(self._rg, config=self.config, **self.estimator_kwargs)
        if estimator_type in {"lmmse", "approx_lmmse"}:
            from ..components.estimators import LMMSEChannelEstimator

            return LMMSEChannelEstimator(self._rg, config=self.config, **self.estimator_kwargs)
        if estimator_type in {"adaptive_window", "awin", "adaptive_dft"}:
            from ..components.estimators import AdaptiveWindowChannelEstimator

            return AdaptiveWindowChannelEstimator(self._rg, config=self.config, **self.estimator_kwargs)
        if estimator_type in {"adaptive", "adaptive_hybrid"}:
            from ..components.estimators import AdaptiveHybridChannelEstimator

            return AdaptiveHybridChannelEstimator(self._rg, config=self.config, **self.estimator_kwargs)
        if estimator_type in {"ista", "ista_sparse"}:
            from ..components.estimators import ISTAChannelEstimator

            return ISTAChannelEstimator(self._rg, config=self.config, **self.estimator_kwargs)
        if estimator_type in {"neural", "neural_net", "nn"}:
            from ..components.estimators import NeuralChannelEstimator

            return NeuralChannelEstimator(self._rg, config=self.config, **self.estimator_kwargs)
        if estimator_type == "perfect":
            return None
        raise ValueError(f"Unsupported estimator_type '{self.estimator_type}'.")

    def default_directives(self) -> ResourceDirectives:
        num_ut = int(self.config.get("num_ut", 8))
        return ResourceDirectives(
            active_ut_mask=[1] * num_ut,
            per_ut_power=[1.0] * num_ut,
            pilot_reuse_factor=1,
        )

    def prepare_batch_context(
        self,
        batch_size: int,
        ebno_db: float,
        include_feedback: bool,
    ) -> BatchContext:
        self._channel.set_topology(batch_size)
        h_freq = self._channel.sample_frequency_response(batch_size)

        noise_variance = tf.cast(
            ebnodb2no(
                ebno_db,
                self.config.get("num_bits_per_symbol", 2),
                self.config.get("coderate", 0.5),
                self._rg,
            ),
            tf.float32,
        )
        y_shape = self._channel.received_shape_from_response(h_freq)
        probe_noise = self._channel.sample_noise(y_shape, noise_variance)
        data_noise = self._channel.sample_noise(y_shape, noise_variance)
        source_bits = self._transmitter.sample_information_bits(batch_size)

        feedback = None
        if include_feedback:
            probe_directives = self.default_directives()
            # CSI is measured on an earlier slot than the one it schedules. With
            # static users and zero delay the two coincide and an instantaneous
            # max-SNR rule is near optimal -- which is precisely why a learned
            # scheduler has nothing to add. Ageing the feedback restores the
            # problem that a policy exploiting temporal statistics can solve.
            h_feedback = self._age_channel(h_freq)
            x_probe, _, _ = self._transmitter.call(batch_size, directives=probe_directives)
            y_probe = self._channel.apply_frequency_response(x_probe, h_feedback) + probe_noise
            if self.perfect_csi:
                feedback = ResourceManagerFeedback(
                    h_hat=h_feedback,
                    err_var=tf.zeros(tf.shape(h_feedback), dtype=noise_variance.dtype),
                )
            else:
                h_hat, err_var = self._receiver.estimate_channel(y_probe, noise_variance)
                feedback = ResourceManagerFeedback(h_hat=h_hat, err_var=err_var)

        return BatchContext(
            batch_size=batch_size,
            ebno_db=float(ebno_db),
            noise_variance=noise_variance,
            h_freq=h_freq,
            probe_noise=probe_noise,
            data_noise=data_noise,
            source_bits=source_bits,
            feedback=feedback,
        )

    def csi_correlation(self) -> float:
        """Jakes correlation between the CSI feedback and the scheduled slot.

        rho = J0(2*pi*f_d*tau) with f_d = v*f_c/c the maximum Doppler shift and
        tau the feedback delay. rho = 1 means perfectly fresh CSI.
        """
        delay_slots = int(self.config.get("csi_feedback_delay_slots", 0))
        if delay_slots <= 0:
            return 1.0
        max_velocity = float(self.config.get("max_ut_velocity", 0.0))
        if max_velocity <= 0.0:
            return 1.0
        carrier_frequency = float(self.config.get("carrier_frequency", 3.5e9))
        doppler_hz = max_velocity * carrier_frequency / 299_792_458.0
        delay_sec = delay_slots * self._estimate_air_interface_latency()
        from scipy.special import j0

        return float(np.clip(j0(2.0 * np.pi * doppler_hz * delay_sec), -1.0, 1.0))

    def _age_channel(self, h_freq: tf.Tensor) -> tf.Tensor:
        """Return the channel as it was `csi_feedback_delay_slots` earlier.

        Uses the standard first-order Jakes ageing model:

            h_old = rho * h + sqrt(1 - rho^2) * h_independent

        which reproduces the correct autocorrelation without having to simulate
        the intervening slots.
        """
        rho = self.csi_correlation()
        if rho >= 1.0 - 1e-9:
            return h_freq
        shape = tf.shape(h_freq)
        real = tf.random.normal(shape, stddev=np.sqrt(0.5), dtype=tf.float32)
        imag = tf.random.normal(shape, stddev=np.sqrt(0.5), dtype=tf.float32)
        independent = tf.complex(real, imag)
        return tf.cast(rho, tf.complex64) * h_freq + tf.cast(
            np.sqrt(max(1.0 - rho**2, 0.0)), tf.complex64
        ) * independent

    def _scheduled_block_mask(
        self,
        directives: ResourceDirectives,
        block_shape: tuple[int, ...],
    ) -> np.ndarray:
        """Boolean mask over [batch, num_tx, num_streams] of codewords actually sent.

        A user counts as scheduled only if it is both in ``active_ut_mask`` and
        allocated non-zero power.
        """
        num_tx = block_shape[1]
        active = np.ones(num_tx, dtype=bool)
        if directives.active_ut_mask is not None:
            active &= np.asarray(directives.active_ut_mask, dtype=np.float64)[:num_tx] > 0.0
        if directives.per_ut_power is not None:
            active &= np.asarray(directives.per_ut_power, dtype=np.float64)[:num_tx] > 0.0
        return np.broadcast_to(active.reshape(1, num_tx, 1), block_shape).copy()

    def _compiled_decode(self):
        """tf.function-compiled decode pipeline, built lazily and cached.

        Nothing in the simulation path used to be graph-compiled, and the
        adaptive estimator called `.numpy()` mid-forward-pass, which forces the
        whole run into eager mode. With the estimator made traceable, the tensor
        part of the decode path can be compiled -- which matters because the
        Monte Carlo depth the URLLC reliability targets require is out of reach
        at eager speed.

        Off by default: enable with `system.graph_mode: true` once a given
        estimator has been verified to trace.
        """
        if self._decode_fn is None:
            self._decode_fn = tf.function(
                self._decode_once_impl, reduce_retracing=True
            )
        return self._decode_fn

    def _decode_once(
        self,
        x_rg,
        h_freq,
        noise,
        noise_variance,
        active_ut_mask,
    ) -> dict:
        """One transmission attempt: apply channel, estimate, equalize, decode."""
        if self.graph_mode:
            h_hat, x_hat, no_eff, bits_hat, decoder_iter, declared_err_var = (
                self._compiled_decode()(x_rg, h_freq, noise, noise_variance, active_ut_mask)
            )
            return {
                "h_hat": h_hat,
                "x_hat": x_hat,
                "no_eff": no_eff,
                "bits_hat": bits_hat,
                "decoder_iterations": decoder_iter,
                "declared_err_var": declared_err_var,
            }
        return self._decode_once_impl_dict(x_rg, h_freq, noise, noise_variance, active_ut_mask)

    def _decode_once_impl(
        self,
        x_rg,
        h_freq,
        noise,
        noise_variance,
        active_ut_mask,
    ):
        """Traceable core, returning a flat tuple so it can be tf.function'd."""
        result = self._decode_once_impl_dict(
            x_rg, h_freq, noise, noise_variance, active_ut_mask
        )
        return (
            result["h_hat"],
            result["x_hat"],
            result["no_eff"],
            result["bits_hat"],
            result["decoder_iterations"],
            result["declared_err_var"],
        )

    def _decode_once_impl_dict(
        self,
        x_rg,
        h_freq,
        noise,
        noise_variance,
        active_ut_mask,
    ) -> dict:
        y = self._channel.apply_frequency_response(x_rg, h_freq) + noise

        if self.perfect_csi:
            h_hat = h_freq
            err_var = 0.0
        else:
            h_hat, err_var = self._receiver.estimate_channel(y, noise_variance)

        # The scheduler's decision is known at the receiver in any real system.
        # Restrict the effective channel to the scheduled streams so the equalizer
        # does not null interference from users that never transmitted.
        h_hat, err_var = apply_stream_mask(h_hat, err_var, active_ut_mask)

        x_hat, no_eff = self._receiver.equalize(y, h_hat, err_var, noise_variance)
        llr = self._receiver.demap(x_hat, no_eff)
        bits_hat, decoder_iter = self._receiver.decode(llr)
        return {
            "h_hat": h_hat,
            "x_hat": x_hat,
            "no_eff": no_eff,
            "bits_hat": bits_hat,
            "decoder_iterations": decoder_iter,
            # Mean declared estimation-error variance, so the pipeline can check
            # it against the error the estimator actually made.
            "declared_err_var": tf.reduce_mean(tf.cast(err_var, tf.float32)),
        }

    def run_batch(
        self,
        batch_context: BatchContext,
        directives: ResourceDirectives | None = None,
        include_details: bool = True,
        harq_max_rounds: int | None = None,
    ) -> dict:
        """Transport one batch, optionally with HARQ retransmissions.

        HARQ here is Type-I (no soft combining): a codeword that fails is simply
        retransmitted with an independent noise realisation, up to
        ``harq_max_rounds`` attempts. That is deliberately the conservative
        scheme -- it gives a real, defensible latency distribution without
        claiming combining gain the simulator does not model. Set
        ``harq_max_rounds`` to 1 (the default) to disable retransmission.

        Returns per-codeword ``delivery_round`` (1-based; 0 means never
        delivered) so callers can build a latency distribution rather than only
        a mean.
        """
        active_directives = directives or self.default_directives()
        max_rounds = int(
            harq_max_rounds
            if harq_max_rounds is not None
            else self.config.get("harq_max_rounds", 1)
        )
        max_rounds = max(1, max_rounds)

        x_rg, bits, qam_symbols = self._transmitter.call(
            batch_context.batch_size,
            directives=active_directives,
            bits=batch_context.source_bits,
        )

        wall_clock_start = time.perf_counter()
        bits_np = bits.numpy()
        # delivery_round[b, tx, stream]: 1-based round that first decoded the
        # codeword, or 0 if it never did.
        delivery_round = np.zeros(bits_np.shape[:-1], dtype=np.int32)
        bits_hat_np = None
        # Only codewords from scheduled users are in flight. Muted users neither
        # retransmit (so they cost no energy) nor contribute to latency.
        scheduled = self._scheduled_block_mask(active_directives, bits_np.shape[:-1])
        pending = scheduled.copy()
        rounds_executed = 0
        retransmitted_fraction = 0.0
        last = None

        for round_index in range(1, max_rounds + 1):
            if not pending.any():
                break
            noise = (
                batch_context.data_noise
                if round_index == 1
                else self._channel.sample_noise(
                    self._channel.received_shape_from_response(batch_context.h_freq),
                    batch_context.noise_variance,
                )
            )
            last = self._decode_once(
                x_rg,
                batch_context.h_freq,
                noise,
                batch_context.noise_variance,
                active_directives.active_ut_mask,
            )
            round_bits_hat = last["bits_hat"].numpy()
            if bits_hat_np is None:
                bits_hat_np = round_bits_hat.copy()
            else:
                # Keep the first successful decode for each codeword.
                bits_hat_np[pending] = round_bits_hat[pending]

            block_ok = ~np.any(np.not_equal(bits_np, bits_hat_np), axis=-1)
            newly_delivered = pending & block_ok
            delivery_round[newly_delivered] = round_index

            retransmitted_fraction += float(pending.mean())
            rounds_executed = round_index
            pending = pending & ~block_ok

        processing_latency_sec = time.perf_counter() - wall_clock_start

        if not include_details:
            return {"bits": bits_np, "bits_hat": bits_hat_np, "delivery_round": delivery_round}

        slot_duration_sec = self._estimate_air_interface_latency()
        # Scheduled codewords that never decoded are charged the full HARQ budget;
        # unscheduled ones carry no latency at all.
        rounds_used = np.where(delivery_round > 0, delivery_round, max_rounds)
        latency_per_block_sec = np.where(
            scheduled, rounds_used.astype(np.float64) * slot_duration_sec, np.nan
        )
        mean_latency_sec = (
            float(np.nanmean(latency_per_block_sec)) if scheduled.any() else 0.0
        )

        decoder_iter = last["decoder_iterations"]
        energy_one_slot = self._estimate_energy(active_directives, decoder_iter)
        total_energy_joules = energy_one_slot * retransmitted_fraction

        return {
            "bits": bits_np,
            "bits_hat": bits_hat_np,
            "delivery_round": delivery_round,
            "scheduled_block_mask": scheduled,
            "harq_rounds_executed": rounds_executed,
            "decoder_iterations": decoder_iter.numpy(),
            "channel": batch_context.h_freq.numpy(),
            "channel_hat": last["h_hat"].numpy() if hasattr(last["h_hat"], "numpy") else last["h_hat"],
            "qam": qam_symbols.numpy(),
            "qam_hat": last["x_hat"].numpy(),
            "no_eff": last["no_eff"].numpy(),
            "noise_power": self._noise_power_value(batch_context.noise_variance),
            "declared_err_var": float(last["declared_err_var"]),
            "slot_duration_sec": slot_duration_sec,
            "latency_sec": mean_latency_sec,
            "latency_per_block_sec": latency_per_block_sec,
            "processing_latency_sec": processing_latency_sec,
            "energy_joules": total_energy_joules,
            "radiated_power_w": self._radiated_power(active_directives),
        }

    def _estimate_air_interface_latency(self) -> float:
        """Duration of one transmission time interval, in seconds."""
        from factory6g.sim.stages.common import slot_duration_seconds

        return slot_duration_seconds(self.config)

    def _radiated_power(self, directives: ResourceDirectives) -> float:
        """Total radiated power in watts implied by the scheduling directives.

        Each scheduled user transmits at ``per_ut_power`` (a linear fraction of the
        per-UT maximum) times ``ut_max_tx_power_w``. Muted users radiate nothing.
        """
        num_ut = int(self.config.get("num_ut", 8))
        ut_max_tx_power_w = float(self.config.get("ut_max_tx_power_w", 0.2))

        mask = directives.active_ut_mask
        power = directives.per_ut_power
        active = np.ones(num_ut) if mask is None else np.asarray(mask, dtype=np.float64)[:num_ut]
        levels = np.ones(num_ut) if power is None else np.asarray(power, dtype=np.float64)[:num_ut]
        return float(np.sum(np.clip(active, 0.0, 1.0) * np.clip(levels, 0.0, None)) * ut_max_tx_power_w)

    def _estimate_energy(
        self,
        directives: ResourceDirectives,
        decoder_iter: tf.Tensor,
    ) -> float:
        """Energy in joules consumed transporting one batch slot.

        Replaces an earlier model whose terms cancelled to a constant and which
        never saw the scheduling directives at all, so power control could not
        show up in any reported number.

        The model is:

            E = P_radiated * T_slot / eta_pa      (transmit)
              + P_circuit_ut * N_active * T_slot  (UT baseband + RF front end)
              + P_circuit_bs * T_slot             (BS front end)
              + e_decode * N_active * iters       (per-codeword decoding)

        which is linear in the power the scheduler actually allocated and in the
        number of users it actually scheduled. Coefficients come from config so a
        study can substitute measured hardware figures.
        """
        num_ut = int(self.config.get("num_ut", 8))
        slot_duration_sec = self._estimate_air_interface_latency()

        mask = directives.active_ut_mask
        active = np.ones(num_ut) if mask is None else np.asarray(mask, dtype=np.float64)[:num_ut]
        num_active = float(np.sum(np.clip(active, 0.0, 1.0)))

        pa_efficiency = max(float(self.config.get("pa_efficiency", 0.35)), 1e-6)
        circuit_power_ut_w = float(self.config.get("circuit_power_ut_w", 0.1))
        circuit_power_bs_w = float(self.config.get("circuit_power_bs_w", 1.0))
        decode_energy_per_iter_j = float(self.config.get("decode_energy_per_iter_j", 1e-6))

        tx_energy = self._radiated_power(directives) * slot_duration_sec / pa_efficiency
        ut_circuit_energy = circuit_power_ut_w * num_active * slot_duration_sec
        bs_circuit_energy = circuit_power_bs_w * slot_duration_sec
        avg_iterations = float(tf.reduce_mean(decoder_iter))
        decode_energy = decode_energy_per_iter_j * num_active * avg_iterations
        return tx_energy + ut_circuit_energy + bs_circuit_energy + decode_energy

    @staticmethod
    def _noise_power_value(noise_variance: tf.Tensor) -> float:
        if hasattr(noise_variance, "numpy"):
            value = noise_variance.numpy()
        else:
            value = noise_variance
        if np.isscalar(value):
            return float(value)
        arr = np.array(value)
        return float(arr.item()) if arr.size == 1 else float(np.mean(arr))

    def get_config(self) -> dict:
        return self.config.copy()

    def get_transmitter(self) -> Transmitter:
        return self._transmitter

    def get_channel(self) -> ChannelModel:
        return self._channel

    def get_receiver(self) -> Receiver:
        return self._receiver
