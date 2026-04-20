from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _profile_field(profile_data: Any, field: str) -> Any:
    if isinstance(profile_data, dict):
        return profile_data[field]
    return getattr(profile_data, field)


def _read_csv_rows(csv_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(csv_path).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def _plot_layout_2d(
    output_path: Path,
    *,
    scale: str,
    scenario_label: str,
    archetype: str,
    room_dims: list[float],
    machine_layout: list[dict[str, float]],
    tx_position: list[float],
    rx_positions: np.ndarray,
) -> None:
    room_len, room_width = float(room_dims[0]), float(room_dims[1])
    fig, ax = plt.subplots(figsize=(10, 8))
    room_rect = plt.Rectangle(
        (-room_len / 2, -room_width / 2),
        room_len,
        room_width,
        fill=False,
        linestyle="--",
        linewidth=2.0,
        color="black",
    )
    ax.add_patch(room_rect)
    for machine in machine_layout:
        sx = float(machine["sx"])
        sy = float(machine["sy"])
        px = float(machine["x"])
        py = float(machine["y"])
        machine_rect = plt.Rectangle(
            (px - sx / 2, py - sy / 2),
            sx,
            sy,
            fill=True,
            alpha=0.35,
            color="#808080",
        )
        ax.add_patch(machine_rect)

    ax.scatter(rx_positions[:, 0], rx_positions[:, 1], s=18, alpha=0.6, label="RX sample positions")
    ax.scatter([tx_position[0]], [tx_position[1]], marker="^", s=180, color="red", label="Base station")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"{scale} Factory Layout (2D)\n{scenario_label} - {archetype}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_layout_3d(
    output_path: Path,
    *,
    scale: str,
    scenario_label: str,
    archetype: str,
    room_dims: list[float],
    machine_layout: list[dict[str, float]],
    tx_position: list[float],
    rx_positions: np.ndarray,
) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    room_len, room_width, room_height = [float(val) for val in room_dims]
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(rx_positions[:, 0], rx_positions[:, 1], rx_positions[:, 2], s=10, alpha=0.45, label="RX")
    ax.scatter([tx_position[0]], [tx_position[1]], [tx_position[2]], marker="^", s=220, color="red", label="BS")
    ax.plot(
        [tx_position[0], tx_position[0]],
        [tx_position[1], tx_position[1]],
        [0, tx_position[2]],
        color="gray",
        linewidth=2,
    )

    for machine in machine_layout:
        ax.bar3d(
            float(machine["x"]) - float(machine["sx"]) / 2,
            float(machine["y"]) - float(machine["sy"]) / 2,
            0.0,
            float(machine["sx"]),
            float(machine["sy"]),
            float(machine["sz"]),
            color="#808080",
            alpha=0.2,
            shade=True,
        )

    ax.set_xlim([-room_len / 2, room_len / 2])
    ax.set_ylim([-room_width / 2, room_width / 2])
    ax.set_zlim([0, room_height])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=25, azim=38)
    ax.set_title(f"{scale} Factory Layout (3D)\n{scenario_label} - {archetype}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_kpi_panel(
    output_path: Path,
    *,
    scale: str,
    scenario_label: str,
    archetype: str,
    rows: list[dict[str, Any]],
) -> None:
    valid_count = np.asarray([float(row["valid_path_count"]) for row in rows], dtype=np.float32)
    dominant_mag = np.asarray([float(row["dominant_path_mag"]) for row in rows], dtype=np.float32)
    dominant_delay = np.asarray([float(row["dominant_path_delay"]) for row in rows], dtype=np.float32)
    bs_rx_distance = np.asarray([float(row["bs_rx_distance_m"]) for row in rows], dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    axes[0].hist(valid_count, bins=min(20, max(5, len(np.unique(valid_count)))), color="#4c78a8")
    axes[0].set_title("Valid Path Count")
    axes[0].set_xlabel("Count")
    axes[0].set_ylabel("Samples")

    axes[1].hist(dominant_mag, bins=20, color="#f58518")
    axes[1].set_title("Dominant Path Magnitude")
    axes[1].set_xlabel("|a|")
    axes[1].set_ylabel("Samples")

    valid_delay = dominant_delay[dominant_delay >= 0]
    if valid_delay.size > 0:
        axes[2].hist(valid_delay, bins=20, color="#54a24b")
    axes[2].set_title("Dominant Path Delay")
    axes[2].set_xlabel("Delay (s)")
    axes[2].set_ylabel("Samples")

    axes[3].hist(bs_rx_distance, bins=20, color="#b279a2")
    axes[3].set_title("BS-RX Distance")
    axes[3].set_xlabel("Distance (m)")
    axes[3].set_ylabel("Samples")

    fig.suptitle(f"{scale} KPI Panel - {scenario_label} ({archetype})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_profile_visuals(profile_data: Any, visuals_dir: str | Path) -> dict[str, str]:
    profile = _profile_field(profile_data, "profile")
    scale = _profile_field(profile, "factory_scale")
    scenario_label = _profile_field(profile, "scenario_label")
    archetype = _profile_field(profile, "real_world_archetype")
    room_dims = list(_profile_field(profile_data, "room_dimensions"))
    machine_layout = list(_profile_field(profile_data, "machine_layout"))
    tx_position = list(_profile_field(profile_data, "tx_position"))
    rx_positions = np.asarray(_profile_field(profile_data, "rx_positions"), dtype=np.float32)
    rows = list(_profile_field(profile_data, "csv_rows"))

    out_dir = Path(visuals_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layout_2d = out_dir / f"layout_2d_{scale}.png"
    layout_3d = out_dir / f"layout_3d_{scale}.png"
    kpi_panel = out_dir / f"kpi_panel_{scale}.png"

    _plot_layout_2d(
        layout_2d,
        scale=scale,
        scenario_label=scenario_label,
        archetype=archetype,
        room_dims=room_dims,
        machine_layout=machine_layout,
        tx_position=tx_position,
        rx_positions=rx_positions,
    )
    _plot_layout_3d(
        layout_3d,
        scale=scale,
        scenario_label=scenario_label,
        archetype=archetype,
        room_dims=room_dims,
        machine_layout=machine_layout,
        tx_position=tx_position,
        rx_positions=rx_positions,
    )
    _plot_kpi_panel(
        kpi_panel,
        scale=scale,
        scenario_label=scenario_label,
        archetype=archetype,
        rows=rows,
    )
    return {
        "layout_2d": str(layout_2d),
        "layout_3d": str(layout_3d),
        "kpi_panel": str(kpi_panel),
    }


def generate_visuals_from_manifest(
    manifest_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, dict[str, str]]:
    manifest_file = Path(manifest_path)
    with manifest_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    profiles = payload.get("profiles", [])
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("Manifest does not contain any profile entries.")
    base_output = Path(output_dir) if output_dir else manifest_file.parent / "visuals"
    base_output.mkdir(parents=True, exist_ok=True)

    generated_paths: dict[str, dict[str, str]] = {}
    for profile_entry in profiles:
        scale = str(profile_entry["factory_scale"])
        rows = _read_csv_rows(profile_entry["csv_path"])
        rx_positions = np.asarray(
            [
                [
                    float(row["pos_x"]),
                    float(row["pos_y"]),
                    float(row["pos_z"]),
                ]
                for row in rows
            ],
            dtype=np.float32,
        )
        synthetic_profile = {
            "profile": {
                "factory_scale": scale,
                "scenario_label": profile_entry["scenario_label"],
                "real_world_archetype": profile_entry["real_world_archetype"],
            },
            "room_dimensions": profile_entry["room_dimensions"],
            "machine_layout": profile_entry.get("machine_layout", []),
            "tx_position": profile_entry["tx_position"],
            "rx_positions": rx_positions,
            "csv_rows": rows,
        }
        generated_paths[scale] = generate_profile_visuals(synthetic_profile, base_output)
    return generated_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate factory profile visuals from a dataset manifest.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to *_manifest.json file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to <manifest_dir>/visuals.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    out_paths = generate_visuals_from_manifest(args.manifest, output_dir=args.output_dir)
    for scale, paths in out_paths.items():
        print(f"[OK] {scale} visuals: {paths}")


if __name__ == "__main__":
    main()
