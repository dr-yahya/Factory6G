"""Factory environment visualization using Sionna RT ray tracing.

Builds a factory scene from the simulation config (room geometry, machines,
materials) and produces four output images:

  1. 3D scene render with traced ray paths  (Sionna RT renderer)
  2. Signal coverage map heatmap            (Sionna RT coverage_map)
  3. 2D top-down floor plan                 (matplotlib)
  4. 3D perspective layout                  (matplotlib)
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scene XML building (adapted from scripts/tools/generate_factory_dataset.py)
# ---------------------------------------------------------------------------

def _build_scene_xml(
    factory_params: dict[str, Any],
    transceiver_params: dict[str, Any],
    *,
    box_ply_path: Path,
    rng: np.random.Generator,
) -> tuple[Path, list[float], list[dict[str, float]]]:
    """Write a Mitsuba/Sionna RT scene XML file from factory geometry params.

    Returns:
        (scene_xml_path, room_dims, machine_layout)
    """
    xml_lines = ['<scene version="2.1.0">']
    xml_lines.append(
        '    <bsdf type="itu-radio-material" id="mat_default">'
        '<string name="type" value="concrete"/></bsdf>'
    )

    room = factory_params.get("room_dimensions", [20.0, 20.0, 6.0])
    room_dims = [float(room[0]), float(room[1]), float(room[2])]
    room_len, room_width, room_height = room_dims
    wall_thickness = float(transceiver_params.get("wall_thickness", 0.2))
    room_padding = float(transceiver_params.get("room_padding", 2.0))
    machine_ranges = factory_params.get("machine_size_range", [[1.0, 3.0], [1.0, 3.0], [1.0, 2.5]])
    num_machines = int(factory_params.get("num_machines", 5))
    machine_layout: list[dict[str, float]] = []

    def add_shape(shape_id: str, size_xyz: list[float], pos_xyz: list[float]) -> None:
        sx = float(size_xyz[0]) / 10.0
        sy = float(size_xyz[1]) / 10.0
        sz = float(size_xyz[2]) / 5.0
        xml_lines.append(f'    <shape type="ply" id="{shape_id}">')
        xml_lines.append(f'        <string name="filename" value="{box_ply_path}"/>')
        xml_lines.append('        <ref id="mat_default" name="bsdf"/>')
        xml_lines.append('        <transform name="to_world">')
        xml_lines.append(f'            <scale x="{sx}" y="{sy}" z="{sz}"/>')
        xml_lines.append(
            f'            <translate x="{float(pos_xyz[0])}" '
            f'y="{float(pos_xyz[1])}" z="{float(pos_xyz[2])}"/>'
        )
        xml_lines.append("        </transform>")
        xml_lines.append("    </shape>")

    add_shape("floor",    [room_len, room_width, wall_thickness], [0.0, 0.0, -wall_thickness])
    add_shape("ceiling",  [room_len, room_width, wall_thickness], [0.0, 0.0, room_height])
    add_shape("wall_left",  [wall_thickness, room_width, room_height],
              [-room_len / 2 - wall_thickness / 2, 0.0, room_height / 2])
    add_shape("wall_right", [wall_thickness, room_width, room_height],
              [room_len / 2 + wall_thickness / 2, 0.0, room_height / 2])
    add_shape("wall_front", [room_len, wall_thickness, room_height],
              [0.0, -room_width / 2 - wall_thickness / 2, room_height / 2])
    add_shape("wall_back",  [room_len, wall_thickness, room_height],
              [0.0, room_width / 2 + wall_thickness / 2, room_height / 2])

    for i in range(num_machines):
        sx = float(rng.uniform(machine_ranges[0][0], machine_ranges[0][1]))
        sy = float(rng.uniform(machine_ranges[1][0], machine_ranges[1][1]))
        sz = float(rng.uniform(machine_ranges[2][0], machine_ranges[2][1]))
        px = float(rng.uniform(-room_len / 2 + room_padding, room_len / 2 - room_padding))
        py = float(rng.uniform(-room_width / 2 + room_padding, room_width / 2 - room_padding))
        add_shape(f"machine_{i}", [sx, sy, sz], [px, py, sz / 2])
        machine_layout.append({"x": px, "y": py, "z": 0.0, "sx": sx, "sy": sy, "sz": sz})

    xml_lines.append("</scene>")
    tmp = Path(tempfile.mktemp(suffix=".xml"))
    tmp.write_text("\n".join(xml_lines), encoding="utf-8")
    return tmp, room_dims, machine_layout


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

def build_scene(
    flat_config: dict[str, Any],
    *,
    seed: int = 42,
) -> tuple[Any, list[float], list[float], list[float], list[dict[str, float]]]:
    """Build and return a fully configured Sionna RT scene.

    Args:
        flat_config: Merged flat config dict (all JSON sections merged).
        seed: Random seed for machine placement.

    Returns:
        (scene, room_dims, tx_position, rx_position, machine_layout)
    """
    from sionna.rt import (
        PathSolver,  # noqa: F401 — imported here to trigger TF init
        PlanarArray,
        RadioMaterial,
        Receiver,
        Transmitter,
        load_scene,
    )
    from sionna.rt.scene import box as box_xml_path

    rng = np.random.default_rng(seed)
    box_ply = Path(box_xml_path).parent / "meshes" / "box.ply"
    if not box_ply.exists():
        raise FileNotFoundError(f"Sionna RT box.ply not found at {box_ply}")

    scene_xml, room_dims, machine_layout = _build_scene_xml(
        flat_config, flat_config, box_ply_path=box_ply, rng=rng
    )
    try:
        scene = load_scene(str(scene_xml))

        # --- materials ---
        materials_cfg = flat_config.get("materials", {})
        metal_cfg = materials_cfg.get("metal", {})
        concrete_cfg = materials_cfg.get("concrete", {})
        metal = RadioMaterial(
            str(metal_cfg.get("name", "factory_metal")),
            relative_permittivity=float(metal_cfg.get("relative_permittivity", 1.0)),
            conductivity=float(metal_cfg.get("conductivity", 1e7)),
        )
        concrete = RadioMaterial(
            str(concrete_cfg.get("name", "factory_concrete")),
            relative_permittivity=float(concrete_cfg.get("relative_permittivity", 7.0)),
            conductivity=float(concrete_cfg.get("conductivity", 0.1)),
        )
        if metal.name not in scene.radio_materials:
            scene.add(metal)
        if concrete.name not in scene.radio_materials:
            scene.add(concrete)
        for name, obj in scene.objects.items():
            obj.radio_material = metal.name if name.startswith("machine") else concrete.name

        # --- frequency ---
        scene.frequency = float(flat_config.get("carrier_frequency", 3.5e9))
        scene.synthetic_array = True

        # --- TX array ---
        antenna_spacing = float(flat_config.get("antenna_spacing", 0.5))
        num_bs_ant = max(1, int(flat_config.get("num_bs_ant", 8)))
        tx_rows = int(np.sqrt(num_bs_ant))
        while tx_rows > 1 and num_bs_ant % tx_rows != 0:
            tx_rows -= 1
        tx_cols = max(1, num_bs_ant // tx_rows)
        scene.tx_array = PlanarArray(
            num_rows=tx_rows, num_cols=tx_cols,
            vertical_spacing=antenna_spacing, horizontal_spacing=antenna_spacing,
            pattern=str(flat_config.get("tx_pattern", "tr38901")),
            polarization=str(flat_config.get("tx_polarization", "cross")),
        )
        room_height = float(room_dims[2])
        tx_height_offset = float(flat_config.get("tx_height_offset", 1.0))
        tx_position = [0.0, 0.0, room_height - tx_height_offset]
        scene.add(Transmitter("BS", position=tx_position))

        # --- RX array ---
        scene.rx_array = PlanarArray(
            num_rows=1, num_cols=1,
            vertical_spacing=antenna_spacing, horizontal_spacing=antenna_spacing,
            pattern=str(flat_config.get("rx_pattern", "iso")),
            polarization=str(flat_config.get("rx_polarization", "V")),
        )
        rx_height = float(flat_config.get("rx_height", 1.0))
        rx_position = [0.0, 0.0, rx_height]
        scene.add(Receiver("RX", position=rx_position))

    finally:
        try:
            scene_xml.unlink(missing_ok=True)
        except Exception:
            pass

    return scene, room_dims, tx_position, rx_position, machine_layout


# ---------------------------------------------------------------------------
# Path solving
# ---------------------------------------------------------------------------

def solve_paths(
    scene: Any,
    flat_config: dict[str, Any],
    *,
    samples_per_src: int = 1000,
) -> Any:
    """Run PathSolver with a reduced sample count suitable for visualization."""
    from sionna.rt import PathSolver

    solver = PathSolver()
    max_depth = int(flat_config.get("max_depth", 5))
    logger.info("Solving ray paths (max_depth=%d, samples=%d) …", max_depth, samples_per_src)
    return solver(scene, max_depth=max_depth, samples_per_src=samples_per_src)


# ---------------------------------------------------------------------------
# Individual render functions
# ---------------------------------------------------------------------------

def render_3d_scene(scene: Any, paths: Any, output_path: Path) -> None:
    """Save a Sionna RT 3D render of the scene with traced ray paths."""
    logger.info("Rendering 3D scene → %s", output_path)
    scene.render(
        paths=paths,
        show_paths=True,
        show_devices=True,
        resolution=[1280, 720],
        num_samples=512,
        filename=str(output_path),
    )


def render_coverage_map(scene: Any, flat_config: dict[str, Any], output_path: Path) -> None:
    """Compute and save a signal coverage heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger.info("Computing coverage map → %s", output_path)
    max_depth = int(flat_config.get("max_depth", 3))
    cm = scene.coverage_map(
        max_depth=max_depth,
        num_samples=2 ** 16,
        los=True,
        reflection=True,
        diffraction=False,
    )
    try:
        fig = cm.show(show=False)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        # Fallback: get the tensor and plot manually
        cm_db = 10.0 * np.log10(np.maximum(cm.as_tensor().numpy().squeeze(), 1e-12))
        fig, ax = plt.subplots(figsize=(10, 7))
        img = ax.imshow(cm_db, origin="lower", cmap="viridis", aspect="auto")
        plt.colorbar(img, ax=ax, label="Path gain (dB)")
        ax.set_title("Signal Coverage Map")
        ax.set_xlabel("X cells")
        ax.set_ylabel("Y cells")
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)


def render_layout_2d(
    room_dims: list[float],
    machine_layout: list[dict[str, float]],
    tx_position: list[float],
    rx_positions: list[list[float]],
    output_path: Path,
    label: str = "",
) -> None:
    """Save a 2D top-down floor plan with machines, TX, and sample UE positions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    room_len, room_width, _ = room_dims
    fig, ax = plt.subplots(figsize=(12, 8))

    # Room outline
    ax.add_patch(mpatches.Rectangle(
        (-room_len / 2, -room_width / 2), room_len, room_width,
        linewidth=2, edgecolor="black", facecolor="#f8f8f8", linestyle="--",
    ))
    # Machines
    for m in machine_layout:
        ax.add_patch(mpatches.Rectangle(
            (m["x"] - m["sx"] / 2, m["y"] - m["sy"] / 2), m["sx"], m["sy"],
            linewidth=1, edgecolor="#555555", facecolor="#bbbbbb", alpha=0.65,
        ))
    # Sample UE positions
    if rx_positions:
        rx_arr = np.array(rx_positions)
        ax.scatter(rx_arr[:, 0], rx_arr[:, 1], s=18, color="#1f77b4",
                   alpha=0.75, label=f"Sample UE positions ({len(rx_positions)})", zorder=5)
    # Base station
    ax.scatter([tx_position[0]], [tx_position[1]], s=250, marker="^",
               color="red", zorder=10, label="Base station (BS)")

    ax.set_xlim(-room_len / 2 - 2, room_len / 2 + 2)
    ax.set_ylim(-room_width / 2 - 2, room_width / 2 + 2)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Factory Floor Plan — {label}\n"
                 f"{room_len:.0f} × {room_width:.0f} m  |  {len(machine_layout)} machines")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("2D layout saved → %s", output_path)


def render_layout_3d(
    room_dims: list[float],
    machine_layout: list[dict[str, float]],
    tx_position: list[float],
    rx_positions: list[list[float]],
    output_path: Path,
    label: str = "",
) -> None:
    """Save a 3D perspective view of the factory layout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    room_len, room_width, room_height = room_dims
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Machines as 3D boxes
    for m in machine_layout:
        ax.bar3d(
            m["x"] - m["sx"] / 2, m["y"] - m["sy"] / 2, 0.0,
            m["sx"], m["sy"], m["sz"],
            color="#aaaaaa", alpha=0.5, shade=True,
        )

    # Sample UE positions
    if rx_positions:
        rx_arr = np.array(rx_positions)
        ax.scatter(rx_arr[:, 0], rx_arr[:, 1], rx_arr[:, 2],
                   s=15, color="#1f77b4", alpha=0.65, label="Sample UE positions")

    # Base station + vertical pole
    ax.scatter([tx_position[0]], [tx_position[1]], [tx_position[2]],
               s=250, marker="^", color="red", zorder=10, label="Base station (BS)")
    ax.plot([tx_position[0]] * 2, [tx_position[1]] * 2, [0, tx_position[2]],
            "r--", linewidth=1.2)

    ax.set_xlim(-room_len / 2, room_len / 2)
    ax.set_ylim(-room_width / 2, room_width / 2)
    ax.set_zlim(0, room_height)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Factory 3D View — {label}\n"
                 f"{room_len:.0f} × {room_width:.0f} × {room_height:.0f} m")
    ax.view_init(elev=25, azim=38)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("3D layout saved → %s", output_path)


# ---------------------------------------------------------------------------
# Convenience: sample RX positions for layout plots
# ---------------------------------------------------------------------------

def sample_rx_positions(
    room_dims: list[float],
    flat_config: dict[str, Any],
    num_rx: int = 30,
    *,
    seed: int = 42,
) -> list[list[float]]:
    """Return a list of random UE positions inside the room boundaries."""
    rng = np.random.default_rng(seed + 1)
    room_len, room_width, _ = room_dims
    boundary_padding = float(flat_config.get("rx_boundary_padding", 1.0))
    rx_height = float(flat_config.get("rx_height", 1.0))
    positions = []
    for _ in range(num_rx):
        px = float(rng.uniform(-room_len / 2 + boundary_padding, room_len / 2 - boundary_padding))
        py = float(rng.uniform(-room_width / 2 + boundary_padding, room_width / 2 - boundary_padding))
        positions.append([px, py, rx_height])
    return positions


# ---------------------------------------------------------------------------
# Master render function
# ---------------------------------------------------------------------------

def render_all(
    scene: Any,
    paths: Any,
    room_dims: list[float],
    machine_layout: list[dict[str, float]],
    tx_position: list[float],
    rx_positions: list[list[float]],
    flat_config: dict[str, Any],
    output_dir: Path,
    label: str,
) -> dict[str, str]:
    """Run all four visualizations and save PNGs to output_dir.

    Returns:
        Dict mapping render name → saved PNG path (only for successful renders).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    # 1. Sionna RT 3D render with ray paths
    p = output_dir / "3d_render.png"
    try:
        render_3d_scene(scene, paths, p)
        outputs["3d_render"] = str(p)
    except Exception as exc:
        logger.warning("3D render failed (%s) — skipping.", exc)

    # 2. Coverage map
    p = output_dir / "coverage_map.png"
    try:
        render_coverage_map(scene, flat_config, p)
        outputs["coverage_map"] = str(p)
    except Exception as exc:
        logger.warning("Coverage map failed (%s) — skipping.", exc)

    # 3. 2D floor plan (always works — pure matplotlib)
    p = output_dir / "layout_2d.png"
    render_layout_2d(room_dims, machine_layout, tx_position, rx_positions, p, label)
    outputs["layout_2d"] = str(p)

    # 4. 3D perspective (always works — pure matplotlib)
    p = output_dir / "layout_3d.png"
    render_layout_3d(room_dims, machine_layout, tx_position, rx_positions, p, label)
    outputs["layout_3d"] = str(p)

    return outputs
