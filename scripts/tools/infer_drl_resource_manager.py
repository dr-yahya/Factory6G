from __future__ import annotations

import argparse
import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from factory6g.models.drl_policy import (
    build_policy_state,
    load_policy_checkpoint,
    load_rm_policy_dataset,
    predict_policy_outputs,
    project_policy_to_directives,
)


def _load_channel_energy(
    *,
    data_path: str | None,
    sample_index: int,
    channel_energy_json: str | None,
    ebno_db: float | None,
) -> tuple[np.ndarray, float]:
    if data_path:
        dataset = load_rm_policy_dataset(data_path)
        if sample_index < 0 or sample_index >= dataset["channel_energy"].shape[0]:
            raise IndexError(
                f"sample_index={sample_index} is out of bounds for dataset size {dataset['channel_energy'].shape[0]}."
            )
        sample_channel = dataset["channel_energy"][sample_index]
        sample_ebno = float(dataset["ebno_db"][sample_index] if ebno_db is None else ebno_db)
        return sample_channel, sample_ebno

    if channel_energy_json is None or ebno_db is None:
        raise ValueError("Provide either --data or both --channel-energy-json and --ebno-db.")

    sample_channel = np.asarray(json.loads(channel_energy_json), dtype=np.float32)
    if sample_channel.ndim != 2:
        raise ValueError(
            f"Expected --channel-energy-json to decode to [num_ut, fft_size], got shape {sample_channel.shape}."
        )
    return sample_channel, float(ebno_db)


def infer_drl_resource_manager(args: argparse.Namespace) -> None:
    checkpoint = load_policy_checkpoint(args.checkpoint)
    channel_energy, ebno_db = _load_channel_energy(
        data_path=args.data,
        sample_index=args.sample_index,
        channel_energy_json=args.channel_energy_json,
        ebno_db=args.ebno_db,
    )

    fairness = None
    if args.fairness_debt:
        fairness = np.asarray([float(item) for item in args.fairness_debt.split(",")], dtype=np.float32)

    state = build_policy_state(channel_energy, ebno_db, fairness_debt=fairness)
    outputs = predict_policy_outputs(checkpoint, state)
    mask, power = project_policy_to_directives(
        np.asarray(outputs["schedule_output"], dtype=np.float32),
        np.asarray(outputs["power_output"], dtype=np.float32),
        num_active=args.num_active,
        max_power=args.max_power,
        min_active_power=args.min_active_power,
    )

    payload = {
        "checkpoint_format": checkpoint.metadata.get("format"),
        "ebno_db": ebno_db,
        "num_active": args.num_active,
        "schedule_output": np.asarray(outputs["schedule_output"], dtype=np.float32).tolist(),
        "power_output": np.asarray(outputs["power_output"], dtype=np.float32).tolist(),
        "value_output": outputs["value_output"],
        "active_ut_mask": mask,
        "per_ut_power": power,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a Factory6G DRL policy checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Policy checkpoint directory or model file")
    parser.add_argument("--data", type=str, default=None, help="Optional parquet dataset to source a sample from")
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset row index when --data is provided")
    parser.add_argument(
        "--channel-energy-json",
        type=str,
        default=None,
        help="JSON-encoded [num_ut, fft_size] channel energy matrix when not using --data",
    )
    parser.add_argument("--ebno-db", type=float, default=None, help="Eb/No for JSON-input inference mode")
    parser.add_argument(
        "--fairness-debt",
        type=str,
        default=None,
        help="Optional comma-separated fairness debt vector used in policy state construction",
    )
    parser.add_argument("--num-active", type=int, default=2, help="How many users to schedule")
    parser.add_argument("--max-power", type=float, default=1.0, help="Maximum per-user power scale")
    parser.add_argument("--min-active-power", type=float, default=0.2, help="Minimum power for selected users")
    infer_drl_resource_manager(parser.parse_args())
