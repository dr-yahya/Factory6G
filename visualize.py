#!/usr/bin/env python3
"""Factory environment ray-tracing visualization entry point.

Mirrors the main.py simulation CLI but renders visual outputs of the factory
environment (3D scene, coverage map, floor plan) instead of running the PHY
simulation.

Usage:
    python visualize.py                                  # all sizes
    python visualize.py --factory-size apple
    python visualize.py --factory-size s,m,l,apple
    python visualize.py --factory-size apple --num-rx 50 --seed 7

Docker Compose:
    docker compose run visualize
    docker compose run visualize --factory-size apple
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Suppress TF / mitsuba noise before any imports
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLBACKEND", "Agg")


_FACTORY_SIZE_DISPLAY = {
    "s": "Small (Electronics Workcell)",
    "m": "Medium (Automotive Assembly)",
    "l": "Large (Logistics Hall)",
    "apple": "Apple Factory (Consumer Electronics Assembly)",
}
_VALID_SIZES = set(_FACTORY_SIZE_DISPLAY.keys())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Factory6G environments with Sionna RT ray tracing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="config.json",
        help="Path to simulation config JSON.",
    )
    parser.add_argument(
        "--factory-size", metavar="SIZES", default=",".join(sorted(_VALID_SIZES)),
        help=f"Comma-separated factory sizes: {', '.join(sorted(_VALID_SIZES))}.",
    )
    parser.add_argument(
        "--output-dir", default="results/visualizations",
        help="Root directory for PNG outputs.",
    )
    parser.add_argument(
        "--num-rx", type=int, default=30,
        help="Number of sample UE positions shown in layout plots.",
    )
    parser.add_argument(
        "--samples-per-src", type=int, default=1000,
        help="Ray samples per source for path solving (lower = faster).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for machine placement and UE positions.",
    )
    return parser.parse_args()


def _load_flat_config(config_path: str) -> dict:
    """Load config.json and return a single merged flat dict of all sections."""
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)
    flat: dict = {}
    for section in raw.values():
        if isinstance(section, dict):
            flat.update(section)
    # Preserve nested sub-dicts that are needed (e.g. materials)
    for key, val in raw.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                if isinstance(sub_val, dict):
                    flat.setdefault(sub_key, sub_val)
    return flat


def main() -> int:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    # Parse and validate factory sizes
    sizes = [s.strip().lower() for s in args.factory_size.split(",") if s.strip()]
    invalid = [s for s in sizes if s not in _VALID_SIZES]
    if invalid:
        logging.error("Unknown factory size(s): %s. Valid: %s", invalid, sorted(_VALID_SIZES))
        return 1

    # Load config
    try:
        flat_config = _load_flat_config(args.config)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logging.error("Failed to load config '%s': %s", args.config, exc)
        return 1

    # Lazy imports (TF / Sionna) after env vars are set
    from src.sim.flow import _FACTORY_SIZE_PRESETS
    from src.visualization.factory_visualizer import (
        build_scene,
        render_all,
        sample_rx_positions,
        solve_paths,
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("  Factory6G — Environment Visualization")
    print("=" * 70)
    print(f"  Sizes     : {', '.join(_FACTORY_SIZE_DISPLAY.get(s, s) for s in sizes)}")
    print(f"  Output    : {args.output_dir}")
    print(f"  Samples   : {args.samples_per_src} rays/src  |  {args.num_rx} UE positions")
    print("-" * 70)

    all_outputs: dict[str, dict[str, str]] = {}

    for size_label in sizes:
        display = _FACTORY_SIZE_DISPLAY[size_label]
        preset = _FACTORY_SIZE_PRESETS[size_label]

        print(f"\n{'=' * 70}")
        print(f"  {display}")
        print(f"{'=' * 70}")

        # Merge preset geometry overrides into flat config
        size_config = dict(flat_config)
        size_config.update({
            "room_dimensions": preset["room_dimensions"],
            "num_machines": preset["num_machines"],
            "machine_size_range": preset["machine_size_range"],
        })

        out_dir = Path(args.output_dir) / f"{timestamp}_{size_label}"

        try:
            logging.info("Building scene …")
            scene, room_dims, tx_pos, rx_pos, machine_layout = build_scene(
                size_config, seed=args.seed
            )

            logging.info(
                "Room: %.0f × %.0f × %.0f m | %d machines | TX at z=%.1f m",
                room_dims[0], room_dims[1], room_dims[2],
                len(machine_layout), tx_pos[2],
            )

            logging.info("Solving ray paths …")
            paths = solve_paths(
                scene, size_config, samples_per_src=args.samples_per_src
            )

            rx_positions = sample_rx_positions(
                room_dims, size_config, num_rx=args.num_rx, seed=args.seed
            )

            outputs = render_all(
                scene=scene,
                paths=paths,
                room_dims=room_dims,
                machine_layout=machine_layout,
                tx_position=tx_pos,
                rx_positions=rx_positions,
                flat_config=size_config,
                output_dir=out_dir,
                label=display,
            )

            all_outputs[size_label] = outputs
            print(f"\n  Outputs saved to: {out_dir}")
            for name, path in outputs.items():
                print(f"    {name:20s} → {path}")

        except Exception as exc:
            logging.error("Failed to visualize '%s': %s", size_label, exc)
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print(f"  Done. {len(all_outputs)}/{len(sizes)} environment(s) rendered.")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
