"""Policy-gradient training for the resource-manager policy.

Why this exists
---------------
`scripts/tools/train_drl_resource_manager.py` is supervised imitation: it fits
the policy to an oracle's mask/power labels with cross-entropy and MSE. There is
no environment interaction, no reward, no return, no temporal-difference target
and no exploration -- so calling its output "DRL" is a misnomer, and, more
importantly, the result is bounded above by the candidate search that produced
the labels. It cannot discover a policy better than the heuristic that taught it.

This module is the actual reinforcement-learning loop: the policy acts in the
simulator, is rewarded by what the physical layer actually delivered, and is
improved by REINFORCE with a learned value baseline.

Formulation
-----------
The scheduling problem as posed here is a **contextual bandit** (a one-step MDP):
each slot's channel realisation is drawn independently, so an action taken now
does not change the next state. The only state that carries across slots is the
fairness debt, which is included in the observation. A contextual bandit is the
honest description -- claiming a multi-step MDP would require modelling queue
dynamics or channel correlation across slots, which the harness does not yet do.

    state   per-user channel energy, fairness debt, Eb/No
    action  which `num_active` users to schedule, and at what power
    reward  see `RewardWeights` -- reliability, throughput, energy, fairness

Scheduling is sampled with Gumbel top-k, which is exactly the Plackett-Luce
top-k distribution, so the log-probability of the sampled subset has a closed
form and the score-function estimator is unbiased. Powers are sampled from a
Gaussian policy around the power head's output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

_EPS = 1e-9


@dataclass(frozen=True)
class RewardWeights:
    """Scalarisation of the multi-objective scheduling problem.

    Defaults are URLLC-oriented: reliability dominates, throughput is a
    secondary gain, and energy and fairness are mild regularisers. Every term is
    normalised to O(1) so the weights are directly comparable, and every one is
    reported separately by `evaluate_policy` so a study can show the trade-off
    rather than only the scalar.
    """

    reliability: float = 1.0
    throughput: float = 0.3
    energy: float = 0.05
    fairness: float = 0.1


@dataclass
class RLTrainingConfig:
    num_iterations: int = 200
    episodes_per_iteration: int = 8
    batch_size: int = 4
    learning_rate: float = 3e-4
    entropy_bonus: float = 0.01
    value_loss_weight: float = 0.5
    power_log_std: float = -1.0
    ebno_db_range: tuple[float, ...] = (0.0, 4.0, 8.0, 12.0, 16.0)
    fairness_alpha: float = 0.15
    grad_clip_norm: float = 1.0
    seed: int = 0
    reward_weights: RewardWeights = field(default_factory=RewardWeights)


def gumbel_top_k(
    logits: np.ndarray, k: int, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Sample k distinct items without replacement, with the exact log-probability.

    Gumbel top-k is distributionally identical to sequential sampling without
    replacement from a softmax (the Plackett-Luce top-k), so the log-probability
    of the ordered draw is available in closed form:

        log P = sum_i [ logit_{s_i} - logsumexp(logits over items still available) ]

    Returns the selected indices and that log-probability.
    """
    k = max(1, min(int(k), logits.shape[0]))
    perturbed = logits + rng.gumbel(size=logits.shape)
    order = np.argsort(perturbed)[::-1][:k]

    log_prob = 0.0
    remaining = list(range(logits.shape[0]))
    for index in order:
        available = np.array([logits[i] for i in remaining], dtype=np.float64)
        shifted = available - available.max()
        log_prob += float(logits[index] - (available.max() + np.log(np.sum(np.exp(shifted)))))
        remaining.remove(int(index))
    return order.astype(np.int64), log_prob


def _score_logits(schedule_scores):
    """Sigmoid head outputs -> the logits the sampler ranks on."""
    import tensorflow as tf

    clipped = tf.clip_by_value(schedule_scores, 1e-6, 1.0 - 1e-6)
    return tf.math.log(clipped) - tf.math.log(1.0 - clipped)


def plackett_luce_log_prob(schedule_scores, order):
    """Differentiable log-probability of the top-k draw the sampler produced.

    REINFORCE is only unbiased when the score function it differentiates is the
    log-probability of the distribution that actually produced the action. The
    sampler here is Gumbel top-k, which is sampling without replacement from a
    softmax, so the log-probability of the ordered draw ``s_1..s_k`` is

        log P = sum_i [ logit_{s_i} - logsumexp(logits still available at i) ].

    An independent-Bernoulli likelihood over the selected subset -- which is
    what this used to differentiate -- is the score of a different distribution
    entirely. It pushes down the scores of every unselected user, when the
    constraint that exactly k are chosen means only the *relative* ordering can
    move, so the resulting gradient is biased.

    Args:
        schedule_scores: [batch, num_ut] sigmoid outputs of the schedule head.
        order: [batch, k] indices in the order they were drawn.

    Returns:
        [batch] log-probabilities, differentiable in ``schedule_scores``.
    """
    import tensorflow as tf

    logits = _score_logits(schedule_scores)  # [B, N]
    num_ut = tf.shape(logits)[-1]

    chosen = tf.one_hot(order, depth=num_ut, dtype=logits.dtype)  # [B, k, N]
    # Items removed from the pool *before* step i: an exclusive cumulative sum
    # along the draw axis.
    withdrawn = tf.cumsum(chosen, axis=1, exclusive=True)  # [B, k, N]
    available = 1.0 - withdrawn

    # logsumexp over the pool still available at each step.
    masked = tf.where(available > 0.0, logits[:, tf.newaxis, :], logits.dtype.min)
    normalizer = tf.reduce_logsumexp(masked, axis=-1)  # [B, k]
    picked = tf.reduce_sum(chosen * logits[:, tf.newaxis, :], axis=-1)  # [B, k]
    return tf.reduce_sum(picked - normalizer, axis=-1)


def compute_reward(
    outcome: dict[str, Any],
    *,
    weights: RewardWeights,
    throughput_scale: float,
    energy_scale: float,
) -> tuple[float, dict[str, float]]:
    """Scalar reward plus its per-objective breakdown.

    Reliability uses log10(BLER) rather than BLER itself: in the URLLC regime the
    interesting differences are between 1e-3 and 1e-5, and a linear term makes
    them indistinguishable from zero.
    """
    bler = float(np.clip(outcome["bler"], 1e-6, 1.0))
    reliability = -np.log10(bler) / 6.0  # 0 at BLER 1, 1 at BLER 1e-6
    throughput = float(outcome["delivered_bits"]) / max(throughput_scale, _EPS)
    energy = float(outcome["energy_joules"]) / max(energy_scale, _EPS)
    fairness = float(outcome["jains_index"])

    terms = {
        "reliability": reliability,
        "throughput": throughput,
        "energy": -energy,
        "fairness": fairness,
    }
    reward = (
        weights.reliability * terms["reliability"]
        + weights.throughput * terms["throughput"]
        + weights.energy * terms["energy"]
        + weights.fairness * terms["fairness"]
    )
    return float(reward), terms


class ResourceManagerEnv:
    """One-step scheduling environment backed by the real PHY simulator."""

    def __init__(
        self,
        model,
        *,
        num_ut: int,
        num_active: int,
        batch_size: int,
        ebno_db_range: tuple[float, ...],
        fairness_alpha: float,
        harq_max_rounds: int = 1,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.model = model
        self.num_ut = int(num_ut)
        self.num_active = int(num_active)
        self.batch_size = int(batch_size)
        self.ebno_db_range = tuple(float(v) for v in ebno_db_range)
        self.fairness_alpha = float(fairness_alpha)
        self.harq_max_rounds = int(harq_max_rounds)
        self.rng = rng or np.random.default_rng(0)
        self._avg_rates = np.full(self.num_ut, 1e-3, dtype=np.float64)

    def reset_fairness(self) -> None:
        self._avg_rates = np.full(self.num_ut, 1e-3, dtype=np.float64)

    @property
    def fairness_debt(self) -> np.ndarray:
        debt = 1.0 / np.maximum(self._avg_rates, 1e-3)
        return debt / max(float(np.max(debt, initial=0.0)), _EPS)

    def observe(self) -> tuple[dict[str, Any], float]:
        """Draw a fresh channel realisation and return the observation."""
        from factory6g.models.drl_policy import channel_energy_from_h_hat

        ebno_db = float(self.rng.choice(self.ebno_db_range))
        context = self.model.prepare_batch_context(
            batch_size=self.batch_size, ebno_db=ebno_db, include_feedback=True
        )
        channel_energy = channel_energy_from_h_hat(context.feedback.h_hat)[0]
        return {
            "context": context,
            "channel_energy": channel_energy,
            "fairness_debt": self.fairness_debt.copy(),
            "ebno_db": ebno_db,
        }, ebno_db

    def step(self, observation: dict[str, Any], mask: np.ndarray, power: np.ndarray) -> dict[str, Any]:
        """Apply an action and measure what the physical layer delivered."""
        from factory6g.models.resource_manager import ResourceDirectives
        from factory6g.sim.stages.common import jains_fairness_index

        directives = ResourceDirectives(
            active_ut_mask=[int(v) for v in mask],
            per_ut_power=[float(v) for v in power],
            pilot_reuse_factor=1,
        )
        result = self.model.run_batch(
            observation["context"],
            directives=directives,
            include_details=True,
            harq_max_rounds=self.harq_max_rounds,
        )

        bits, bits_hat = result["bits"], result["bits_hat"]
        scheduled = result["scheduled_block_mask"]
        block_error = np.any(np.not_equal(bits, bits_hat), axis=-1)
        num_scheduled = int(scheduled.sum())
        bler = float(block_error[scheduled].mean()) if num_scheduled else 1.0

        bits_per_block = bits.shape[-1]
        delivered_bits = float((~block_error & scheduled).sum() * bits_per_block)

        # Update the fairness memory with what each user actually received.
        per_user_delivered = np.zeros(self.num_ut, dtype=np.float64)
        for user in range(min(self.num_ut, block_error.shape[1])):
            good = (~block_error[:, user, ...] & scheduled[:, user, ...]).sum()
            per_user_delivered[user] = float(good * bits_per_block)
        self._avg_rates = (
            1.0 - self.fairness_alpha
        ) * self._avg_rates + self.fairness_alpha * np.maximum(per_user_delivered, 1e-3)

        return {
            "bler": bler,
            "delivered_bits": delivered_bits,
            "energy_joules": float(result["energy_joules"]),
            "jains_index": jains_fairness_index(self._avg_rates),
            "latency_sec": float(result["latency_sec"]),
            "num_scheduled": num_scheduled,
        }


def train_rl_policy(
    env: ResourceManagerEnv,
    model,
    normalization,
    config: RLTrainingConfig,
    *,
    metadata: dict[str, Any] | None = None,
    progress: Callable[[int, dict[str, float]], None] | None = None,
) -> dict[str, list[float]]:
    """REINFORCE with a learned value baseline.

    The value head predicts the expected reward for the observed state; the
    advantage (reward minus that baseline) weights the score function. This is
    the standard variance-reduced policy-gradient estimator, and it is what makes
    this reinforcement learning rather than imitation.
    """
    import tensorflow as tf

    from factory6g.models.drl_policy import build_policy_state, normalize_policy_state

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    rng = np.random.default_rng(config.seed)
    power_std = float(np.exp(config.power_log_std))
    fairness_regime = (metadata or {}).get("fairness_feature", "live")

    throughput_scale = float(model_throughput_scale(env))
    energy_scale = max(_reference_energy(env), _EPS)

    history: dict[str, list[float]] = {
        "reward": [],
        "bler": [],
        "delivered_bits": [],
        "energy_joules": [],
        "jains_index": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
    }

    for iteration in range(config.num_iterations):
        states, actions, orders, powers, rewards, breakdowns = [], [], [], [], [], []
        # Kept only as a consistency check against the differentiable score
        # below; a mismatch means the two have drifted apart again.
        behaviour_log_probs: list[float] = []
        outcomes: list[dict[str, Any]] = []

        for _ in range(config.episodes_per_iteration):
            observation, _ = env.observe()
            fairness_input = (
                observation["fairness_debt"]
                if fairness_regime == "live"
                else np.ones_like(observation["fairness_debt"])
            )
            state = normalize_policy_state(
                build_policy_state(
                    observation["channel_energy"],
                    observation["ebno_db"],
                    fairness_debt=fairness_input,
                ),
                normalization,
            )
            # Dropout stays off here *and* in the gradient pass below. REINFORCE
            # differentiates the log-probability of the policy that actually
            # acted; scoring under a different dropout mask scores a policy that
            # never chose anything.
            schedule_scores, power_scores, _ = model(state[None, ...], training=False)
            schedule_np = np.asarray(schedule_scores)[0].astype(np.float64)
            power_np = np.asarray(power_scores)[0].astype(np.float64)

            # Sigmoid outputs -> logits for the Plackett-Luce sampler.
            logits = np.log(np.clip(schedule_np, 1e-6, 1 - 1e-6)) - np.log(
                np.clip(1.0 - schedule_np, 1e-6, 1 - 1e-6)
            )
            selected, schedule_log_prob = gumbel_top_k(logits, env.num_active, rng)

            mask = np.zeros(env.num_ut, dtype=np.int64)
            mask[selected] = 1
            noise = rng.normal(0.0, power_std, size=env.num_ut)
            # The Gaussian draw is the action; the clip into the feasible power
            # range belongs to the environment. Scoring the clipped value would
            # be scoring a density that has point masses at the bounds and does
            # not match what was sampled, so the raw draw is what the gradient
            # sees and the clipped one is what the radio gets.
            raw_power = power_np + noise
            applied_power = np.clip(raw_power, 0.05, 1.0) * mask

            outcome = env.step(observation, mask, applied_power)
            reward, terms = compute_reward(
                outcome,
                weights=config.reward_weights,
                throughput_scale=throughput_scale,
                energy_scale=energy_scale,
            )

            states.append(state)
            actions.append(mask)
            orders.append(selected)
            powers.append(raw_power)
            rewards.append(reward)
            behaviour_log_probs.append(schedule_log_prob)
            breakdowns.append(terms)
            outcomes.append(outcome)

        state_batch = tf.constant(np.stack(states), dtype=tf.float32)
        action_batch = tf.constant(np.stack(actions), dtype=tf.float32)
        order_batch = tf.constant(np.stack(orders), dtype=tf.int32)
        power_batch = tf.constant(np.stack(powers), dtype=tf.float32)
        reward_batch = tf.constant(np.asarray(rewards), dtype=tf.float32)

        with tf.GradientTape() as tape:
            # training=False to match the behaviour pass: see the note there.
            schedule_out, power_out, value_out = model(state_batch, training=False)
            baseline = tf.squeeze(value_out, axis=-1)
            advantage = tf.stop_gradient(reward_batch - baseline)

            schedule_log_likelihood = plackett_luce_log_prob(schedule_out, order_batch)
            # Gaussian log-density of the raw power draw. The mask keeps
            # unscheduled users out of it: their power was never applied, so it
            # carries no credit for the reward.
            power_log_likelihood = -0.5 * tf.reduce_sum(
                action_batch * tf.square((power_batch - power_out) / power_std), axis=-1
            )
            policy_loss = -tf.reduce_mean(
                advantage * (schedule_log_likelihood + power_log_likelihood)
            )

            # Entropy of the distribution the sampler actually draws from --
            # the softmax over selection logits -- not of an independent
            # Bernoulli over users, which is a different family.
            selection_logits = _score_logits(schedule_out)
            log_softmax = selection_logits - tf.reduce_logsumexp(
                selection_logits, axis=-1, keepdims=True
            )
            entropy = -tf.reduce_mean(
                tf.reduce_sum(tf.exp(log_softmax) * log_softmax, axis=-1)
            )
            value_loss = tf.reduce_mean(tf.square(reward_batch - baseline))
            loss = (
                policy_loss
                + config.value_loss_weight * value_loss
                - config.entropy_bonus * entropy
            )

        gradients = tape.gradient(loss, model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(
            [g if g is not None else tf.zeros_like(v) for g, v in zip(gradients, model.trainable_variables)],
            config.grad_clip_norm,
        )
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # The differentiable score must agree with the sampler's own
        # log-probability, or REINFORCE is differentiating the wrong
        # distribution again. This is cheap and catches that immediately.
        score_drift = float(
            np.max(np.abs(np.asarray(behaviour_log_probs) - schedule_log_likelihood.numpy()))
        )
        if score_drift > 1e-3:
            raise RuntimeError(
                "policy score function has drifted from the sampler: max "
                f"log-probability difference {score_drift:.3e}"
            )

        record = {
            "reward": float(np.mean(rewards)),
            "bler": float(np.mean([o["bler"] for o in outcomes])),
            "delivered_bits": float(np.mean([o["delivered_bits"] for o in outcomes])),
            "energy_joules": float(np.mean([o["energy_joules"] for o in outcomes])),
            "jains_index": float(np.mean([o["jains_index"] for o in outcomes])),
            "policy_loss": float(policy_loss.numpy()),
            "value_loss": float(value_loss.numpy()),
            "entropy": float(entropy.numpy()),
        }
        for key, value in record.items():
            history[key].append(value)
        if progress is not None:
            progress(iteration, record)

    return history


def model_throughput_scale(env: ResourceManagerEnv) -> float:
    """Information bits a fully loaded slot carries, for reward normalisation."""
    num_info_bits = env.model.get_transmitter().num_info_bits
    return float(num_info_bits * env.num_ut * env.batch_size)


def _reference_energy(env: ResourceManagerEnv) -> float:
    """Energy of a full-power, fully loaded slot, for reward normalisation."""
    import tensorflow as tf

    from factory6g.models.resource_manager import ResourceDirectives

    directives = ResourceDirectives(
        active_ut_mask=[1] * env.num_ut, per_ut_power=[1.0] * env.num_ut
    )
    return float(env.model._estimate_energy(directives, tf.constant([[10.0]])))


def evaluate_policy(
    env: ResourceManagerEnv,
    policy_fn: Callable[[dict[str, Any]], tuple[np.ndarray, np.ndarray]],
    *,
    num_episodes: int = 32,
) -> dict[str, float]:
    """Greedy evaluation of any policy, reported per objective.

    Takes a callable rather than a model so heuristics and learned policies can
    be measured through exactly the same path.
    """
    records: list[dict[str, float]] = []
    env.reset_fairness()
    for _ in range(num_episodes):
        observation, _ = env.observe()
        mask, power = policy_fn(observation)
        records.append(env.step(observation, mask, power))
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in records[0]
    }
