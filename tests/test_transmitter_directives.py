from __future__ import annotations

import numpy as np
from sionna.phy.ofdm import ResourceGrid

from factory6g.components.transmitter import Transmitter
from factory6g.models.resource_manager import ResourceDirectives

from .conftest import make_tiny_config


def _build_transmitter():
    config = make_tiny_config("results")["system"] | make_tiny_config("results")["transceiver"]
    rg = ResourceGrid(
        num_ofdm_symbols=config["num_ofdm_symbols"],
        fft_size=config["fft_size"],
        subcarrier_spacing=config["subcarrier_spacing"],
        num_tx=config["num_ut"],
        num_streams_per_tx=config["num_ut_ant"],
        cyclic_prefix_length=config["cyclic_prefix_length"],
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=config["pilot_ofdm_symbol_indices"],
    )
    return config, Transmitter(config, rg)


def test_muted_users_produce_zero_resource_grid_energy():
    _, transmitter = _build_transmitter()
    bits = transmitter.sample_information_bits(batch_size=1)
    directives = ResourceDirectives(active_ut_mask=[1, 0], per_ut_power=[1.0, 1.0], pilot_reuse_factor=1)
    x_rg, _, _ = transmitter.call(batch_size=1, directives=directives, bits=bits)
    muted_energy = np.abs(x_rg.numpy()[:, 1]).sum()
    active_energy = np.abs(x_rg.numpy()[:, 0]).sum()
    assert active_energy > 0.0
    assert muted_energy == 0.0


def test_power_scaling_applies_square_root_at_grid_level():
    _, transmitter = _build_transmitter()
    bits = transmitter.sample_information_bits(batch_size=1)
    baseline_x_rg, _, _ = transmitter.call(batch_size=1, directives=None, bits=bits)
    scaled_x_rg, _, _ = transmitter.call(
        batch_size=1,
        directives=ResourceDirectives(active_ut_mask=[1, 1], per_ut_power=[0.25, 1.0], pilot_reuse_factor=1),
        bits=bits,
    )
    baseline_energy = np.linalg.norm(baseline_x_rg.numpy()[:, 0])
    scaled_energy = np.linalg.norm(scaled_x_rg.numpy()[:, 0])
    assert np.isclose(scaled_energy / baseline_energy, 0.5, atol=1e-5)


def test_no_directives_matches_all_active_unit_power():
    _, transmitter = _build_transmitter()
    bits = transmitter.sample_information_bits(batch_size=1)
    baseline_x_rg, _, _ = transmitter.call(batch_size=1, directives=None, bits=bits)
    default_x_rg, _, _ = transmitter.call(
        batch_size=1,
        directives=ResourceDirectives(active_ut_mask=[1, 1], per_ut_power=[1.0, 1.0], pilot_reuse_factor=1),
        bits=bits,
    )
    assert np.allclose(baseline_x_rg.numpy(), default_x_rg.numpy())
