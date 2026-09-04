"""Tests for the reinforcement-learning resource-manager trainer."""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
pytest.importorskip("sionna")

from factory6g.models.drl_policy import (  # noqa: E402
    compute_policy_normalization,
    create_policy_model,
)
from factory6g.models.model import Model  # noqa: E402
from factory6g.training.rl_resource_manager import (  # noqa: E402
    RewardWeights,
    RLTrainingConfig,
    ResourceManagerEnv,
    compute_reward,
    evaluate_policy,
    gumbel_top_k,
    train_rl_policy,
)

_CONFIG = {
    "num_ut": 4,
    "num_bs_ant": 8,
    "num_ut_ant": 1,
    "fft_size": 32,
    "num_ofdm_symbols": 14,
    "subcarrier_spacing": 30e3,
    "cyclic_prefix_length": 20,
    "channel_model_type": "rayleigh",
    "num_bits_per_symbol": 2,
    "coderate": 0.5,
    "num_decoding_iter": 6,
    "direction": "uplink",
    "carrier_frequency": 3.5e9,
    "tx_pattern": "tr38901",
    "tx_polarization": "cross",
    "rx_pattern": "iso",
    "rx_polarization": "V",
    "antenna_spacing": 0.5,
    "seed": 0,
}


def _env(num_ut: int = 4, num_active: int = 2) -> ResourceManagerEnv:
    model = Model(config={**_CONFIG, "num_ut": num_ut}, estimator_type="ls")
    return ResourceManagerEnv(
        model,
        num_ut=num_ut,
        num_active=num_active,
        batch_size=2,
        ebno_db_range=(10.0, 14.0),
        fairness_alpha=0.15,
        rng=np.random.default_rng(0),
    )


class TestGumbelTopK:
    def test_selects_exactly_k_distinct_items(self):
        rng = np.random.default_rng(0)
        selected, _ = gumbel_top_k(np.array([1.0, 2.0, 3.0, 4.0]), 2, rng)
        assert selected.shape == (2,)
        assert len(set(selected.tolist())) == 2

    def test_k_is_clamped_to_the_number_of_items(self):
        rng = np.random.default_rng(0)
        selected, _ = gumbel_top_k(np.array([1.0, 2.0]), 10, rng)
        assert selected.shape == (2,)

    def test_log_probability_is_negative_and_finite(self):
        rng = np.random.default_rng(0)
        _, log_prob = gumbel_top_k(np.array([0.5, 1.5, -0.5, 2.0]), 2, rng)
        assert np.isfinite(log_prob)
        assert log_prob < 0.0

    def test_higher_logits_are_selected_more_often(self):
        rng = np.random.default_rng(3)
        logits = np.array([-3.0, -3.0, 3.0, 3.0])
        counts = np.zeros(4)
        for _ in range(400):
            selected, _ = gumbel_top_k(logits, 2, rng)
            counts[selected] += 1
        assert counts[2:].sum() > counts[:2].sum() * 5

    def test_single_item_selection_has_softmax_log_probability(self):
        rng = np.random.default_rng(0)
        logits = np.array([0.0, np.log(3.0)])
        # Sampling one item is plain softmax sampling.
        picks = [int(gumbel_top_k(logits, 1, rng)[0][0]) for _ in range(2000)]
        assert np.mean(picks) == pytest.approx(0.75, abs=0.05)


class TestReward:
    _SCALES = dict(throughput_scale=1000.0, energy_scale=0.01)

    def _reward(self, **overrides) -> float:
        outcome = {
            "bler": 1e-2,
            "delivered_bits": 500.0,
            "energy_joules": 0.005,
            "jains_index": 0.9,
        }
        outcome.update(overrides)
        return compute_reward(outcome, weights=RewardWeights(), **self._SCALES)[0]

    def test_lower_bler_earns_more_reward(self):
        assert self._reward(bler=1e-5) > self._reward(bler=1e-2) > self._reward(bler=0.5)

    def test_reliability_term_is_logarithmic(self):
        """URLLC differences live between 1e-3 and 1e-5; a linear term hides them."""
        _, a = compute_reward(
            {"bler": 1e-3, "delivered_bits": 0.0, "energy_joules": 0.0, "jains_index": 0.0},
            weights=RewardWeights(),
            **self._SCALES,
        )
        _, b = compute_reward(
            {"bler": 1e-5, "delivered_bits": 0.0, "energy_joules": 0.0, "jains_index": 0.0},
            weights=RewardWeights(),
            **self._SCALES,
        )
        assert b["reliability"] - a["reliability"] == pytest.approx(2.0 / 6.0, abs=1e-6)

    def test_more_throughput_earns_more_reward(self):
        assert self._reward(delivered_bits=900.0) > self._reward(delivered_bits=100.0)

    def test_more_energy_earns_less_reward(self):
        assert self._reward(energy_joules=0.001) > self._reward(energy_joules=0.02)

    def test_fairer_allocation_earns_more_reward(self):
        assert self._reward(jains_index=1.0) > self._reward(jains_index=0.3)

    def test_breakdown_is_reported_per_objective(self):
        _, terms = compute_reward(
            {"bler": 0.1, "delivered_bits": 1.0, "energy_joules": 1.0, "jains_index": 1.0},
            weights=RewardWeights(),
            **self._SCALES,
        )
        assert set(terms) == {"reliability", "throughput", "energy", "fairness"}

    def test_zero_bler_is_clipped_rather_than_infinite(self):
        assert np.isfinite(self._reward(bler=0.0))


class TestEnvironment:
    def test_step_reports_the_measured_outcome(self):
        env = _env()
        observation, _ = env.observe()
        mask = np.array([1, 1, 0, 0])
        outcome = env.step(observation, mask, mask.astype(float))

        assert 0.0 <= outcome["bler"] <= 1.0
        assert outcome["delivered_bits"] >= 0.0
        assert outcome["energy_joules"] > 0.0
        assert 0.0 <= outcome["jains_index"] <= 1.0
        # Only the scheduled users transmit.
        assert outcome["num_scheduled"] == 2 * env.batch_size

    def test_observation_exposes_the_policy_state_pieces(self):
        env = _env()
        observation, ebno_db = env.observe()
        assert observation["channel_energy"].shape[0] == env.num_ut
        assert observation["fairness_debt"].shape == (env.num_ut,)
        assert ebno_db in env.ebno_db_range

    def test_fairness_debt_responds_to_who_gets_served(self):
        env = _env()
        env.reset_fairness()
        for _ in range(6):
            observation, _ = env.observe()
            mask = np.array([1, 1, 0, 0])
            env.step(observation, mask, mask.astype(float))
        debt = env.fairness_debt
        # Users 2 and 3 were never served, so they carry the most debt.
        assert debt[2] > debt[0]
        assert debt[3] > debt[1]

    def test_scheduling_fewer_users_costs_less_energy(self):
        env = _env()
        observation, _ = env.observe()
        one = env.step(observation, np.array([1, 0, 0, 0]), np.array([1.0, 0, 0, 0]))
        env.reset_fairness()
        many = env.step(observation, np.array([1, 1, 1, 1]), np.ones(4))
        assert many["energy_joules"] > one["energy_joules"]


class TestTrainingLoop:
    def test_produces_a_history_for_every_iteration(self):
        env = _env()
        normalization = compute_policy_normalization(
            np.stack([env.observe()[0]["channel_energy"] for _ in range(4)]),
            np.array([10.0] * 4, dtype=np.float32),
            fairness_debt=np.linspace(0.0, 1.0, 4, dtype=np.float32),
        )
        policy = create_policy_model(
            input_shape=(env.num_ut, _CONFIG["fft_size"] + 2),
            output_dim=env.num_ut,
            hidden_dim=32,
            dropout_rate=0.0,
        )
        history = train_rl_policy(
            env,
            policy,
            normalization,
            RLTrainingConfig(
                num_iterations=3,
                episodes_per_iteration=2,
                batch_size=2,
                ebno_db_range=(10.0, 14.0),
                seed=0,
            ),
            metadata={"fairness_feature": "live"},
        )
        assert set(history) >= {"reward", "bler", "policy_loss", "value_loss", "entropy"}
        assert all(len(values) == 3 for values in history.values())
        assert all(np.isfinite(history["reward"]))

    def test_updates_the_policy_weights(self):
        env = _env()
        normalization = compute_policy_normalization(
            np.stack([env.observe()[0]["channel_energy"] for _ in range(4)]),
            np.array([10.0] * 4, dtype=np.float32),
            fairness_debt=np.linspace(0.0, 1.0, 4, dtype=np.float32),
        )
        policy = create_policy_model(
            input_shape=(env.num_ut, _CONFIG["fft_size"] + 2),
            output_dim=env.num_ut,
            hidden_dim=32,
            dropout_rate=0.0,
        )
        before = [w.numpy().copy() for w in policy.trainable_variables]
        train_rl_policy(
            env,
            policy,
            normalization,
            RLTrainingConfig(
                num_iterations=3,
                episodes_per_iteration=2,
                batch_size=2,
                learning_rate=1e-2,
                ebno_db_range=(10.0, 14.0),
                seed=0,
            ),
            metadata={"fairness_feature": "live"},
        )
        after = [w.numpy() for w in policy.trainable_variables]
        assert any(not np.allclose(a, b) for a, b in zip(before, after))


def test_evaluate_policy_measures_any_callable_through_the_same_path():
    env = _env()

    def always_first_two(observation):
        mask = np.zeros(env.num_ut, dtype=np.int64)
        mask[:2] = 1
        return mask, mask.astype(float)

    metrics = evaluate_policy(env, always_first_two, num_episodes=3)
    assert set(metrics) >= {"bler", "delivered_bits", "energy_joules", "jains_index"}
    assert all(np.isfinite(v) for v in metrics.values())
