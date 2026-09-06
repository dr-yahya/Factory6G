"""Train the resource-manager policy with reinforcement learning.

Unlike `train_drl_resource_manager.py`, which imitates an oracle's labels, this
script has the policy act in the simulator and learn from the reward its actions
actually earned. It can therefore exceed the heuristics, which behaviour cloning
structurally cannot.

Example:

    python scripts/tools/train_rl_resource_manager.py \
        --config config/config.json \
        --iterations 300 \
        --output-dir models/rl_resource_manager_policy
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import tensorflow as tf

from factory6g.models.drl_policy import (
    FAIRNESS_FEATURE_LIVE,
    PolicyNormalization,
    channel_energy_from_h_hat,
    compute_policy_normalization,
    create_policy_model,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from factory6g.models.model import Model
from factory6g.sim.config import load_config
from factory6g.training.rl_resource_manager import (
    RLTrainingConfig,
    ResourceManagerEnv,
    RewardWeights,
    evaluate_policy,
    train_rl_policy,
)


def _bootstrap_normalization(env: ResourceManagerEnv, num_samples: int) -> PolicyNormalization:
    """Estimate input statistics from a handful of real environment draws."""
    energies, ebnos = [], []
    for _ in range(num_samples):
        observation, ebno_db = env.observe()
        energies.append(observation["channel_energy"])
        ebnos.append(ebno_db)
    return compute_policy_normalization(
        np.stack(energies),
        np.asarray(ebnos, dtype=np.float32),
        # Fairness debt is normalised in [0, 1] by construction.
        fairness_debt=np.linspace(0.0, 1.0, num_samples, dtype=np.float32),
    )


def _greedy_policy(model, normalization, env, fairness_regime: str):
    """Deterministic top-k policy from the trained network, for evaluation."""
    from factory6g.models.drl_policy import build_policy_state, normalize_policy_state

    def policy_fn(observation):
        fairness = (
            observation["fairness_debt"]
            if fairness_regime == FAIRNESS_FEATURE_LIVE
            else np.ones_like(observation["fairness_debt"])
        )
        state = normalize_policy_state(
            build_policy_state(
                observation["channel_energy"], observation["ebno_db"], fairness_debt=fairness
            ),
            normalization,
        )
        schedule, power, _ = model(state[None, ...], training=False)
        schedule = np.asarray(schedule)[0]
        power = np.asarray(power)[0]
        selected = np.argsort(schedule)[::-1][: env.num_active]
        mask = np.zeros(env.num_ut, dtype=np.int64)
        mask[selected] = 1
        return mask, np.clip(power, 0.05, 1.0) * mask

    return policy_fn


def _max_snr_policy(env):
    """Greedy max-channel-energy baseline, evaluated through the same path."""

    def policy_fn(observation):
        energy = observation["channel_energy"].mean(axis=-1)
        selected = np.argsort(energy)[::-1][: env.num_active]
        mask = np.zeros(env.num_ut, dtype=np.int64)
        mask[selected] = 1
        return mask, mask.astype(np.float64)

    return policy_fn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.json")
    parser.add_argument("--output-dir", default="models/rl_resource_manager_policy")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--episodes-per-iteration", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--entropy-bonus", type=float, default=0.01)
    parser.add_argument("--power-log-std", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--norm-samples", type=int, default=16)
    parser.add_argument(
        "--initial-checkpoint",
        default=None,
        help=(
            "Behaviour-cloning checkpoint to warm-start from. Imitating the "
            "oracle first and then improving with RL usually beats either alone."
        ),
    )
    parser.add_argument("--reward-reliability", type=float, default=1.0)
    parser.add_argument("--reward-throughput", type=float, default=0.3)
    parser.add_argument("--reward-energy", type=float, default=0.05)
    parser.add_argument("--reward-fairness", type=float, default=0.1)
    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    app_config = load_config(args.config)
    system_config = app_config.system_runtime_config
    num_ut = int(system_config["num_ut"])
    num_active = int(app_config.resource_managers.num_active_users)

    phy_model = Model(config=system_config, estimator_type="adaptive", perfect_csi=False)
    env = ResourceManagerEnv(
        phy_model,
        num_ut=num_ut,
        num_active=num_active,
        batch_size=args.batch_size,
        ebno_db_range=tuple(app_config.monte_carlo.ebno_db_range),
        fairness_alpha=0.15,
        harq_max_rounds=int(system_config.get("harq_max_rounds", 1)),
        rng=np.random.default_rng(args.seed),
    )

    print(f"Bootstrapping input normalization from {args.norm_samples} environment draws...")
    normalization = _bootstrap_normalization(env, args.norm_samples)

    if args.initial_checkpoint:
        checkpoint = load_policy_checkpoint(args.initial_checkpoint)
        policy_model = checkpoint.model
        if checkpoint.normalization is not None:
            normalization = checkpoint.normalization
        print(f"Warm-starting from {args.initial_checkpoint}")
    else:
        state_dim = int(system_config["fft_size"]) + 2
        policy_model = create_policy_model(
            input_shape=(num_ut, state_dim),
            output_dim=num_ut,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout,
            encoder="deepsets",
        )

    training_config = RLTrainingConfig(
        num_iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        entropy_bonus=args.entropy_bonus,
        power_log_std=args.power_log_std,
        ebno_db_range=tuple(app_config.monte_carlo.ebno_db_range),
        seed=args.seed,
        reward_weights=RewardWeights(
            reliability=args.reward_reliability,
            throughput=args.reward_throughput,
            energy=args.reward_energy,
            fairness=args.reward_fairness,
        ),
    )

    baseline_before = evaluate_policy(
        env, _max_snr_policy(env), num_episodes=args.eval_episodes
    )
    print(f"max-SNR baseline: {json.dumps({k: round(v, 5) for k, v in baseline_before.items()})}")

    def progress(iteration: int, record: dict[str, float]) -> None:
        if iteration % 10 == 0 or iteration == args.iterations - 1:
            print(
                f"  iter {iteration:4d} | reward {record['reward']:+.4f}"
                f" | BLER {record['bler']:.4f}"
                f" | Jain {record['jains_index']:.3f}"
                f" | entropy {record['entropy']:.3f}"
            )

    print(f"Training for {args.iterations} iterations...")
    history = train_rl_policy(
        env,
        policy_model,
        normalization,
        training_config,
        metadata={"fairness_feature": FAIRNESS_FEATURE_LIVE},
        progress=progress,
    )

    learned = evaluate_policy(
        env,
        _greedy_policy(policy_model, normalization, env, FAIRNESS_FEATURE_LIVE),
        num_episodes=args.eval_episodes,
    )
    print(f"learned policy : {json.dumps({k: round(v, 5) for k, v in learned.items()})}")

    metadata = {
        "config_path": args.config,
        "num_ut": num_ut,
        "num_active": num_active,
        "fft_size": int(system_config["fft_size"]),
        "state_dim": int(system_config["fft_size"]) + 2,
        "checkpoint_type": "reinforcement_learning",
        "training_method": "reinforce_with_value_baseline",
        "problem_formulation": "contextual_bandit",
        "fairness_feature": FAIRNESS_FEATURE_LIVE,
        "encoder": "deepsets",
        "policy_outputs": ["schedule_output", "power_output", "value_output"],
        "reward_weights": {
            "reliability": args.reward_reliability,
            "throughput": args.reward_throughput,
            "energy": args.reward_energy,
            "fairness": args.reward_fairness,
        },
        "evaluation": {"max_snr_baseline": baseline_before, "learned_policy": learned},
        "training_args": vars(args),
    }
    checkpoint_dir = save_policy_checkpoint(
        args.output_dir, policy_model, normalization, metadata, history=history
    )
    print(f"Training complete. Policy checkpoint saved to {checkpoint_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
