#!/usr/bin/env python3
"""
Visualize Smart Factory Topology in 3D.

This script generates a random topology and visualizes it in a 3D isometric view,
mimicking a factory floor layout with a Base Station on a tower and User Terminals
(robots/AGVs) on the ground.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sionna.phy.channel import gen_single_sector_topology as gen_topology

def visualize_topology_2d(output_file="topology_mobility.png"):
    """Generate and plot topology with mobility (2D)."""
    
    # Configuration
    num_ut = 10
    min_velocity = 0.0
    max_velocity = 5.0  # m/s
    
    print(f"Generating 2D topology for {num_ut} UTs...")
    
    # Generate topology
    topology = gen_topology(
        batch_size=1,
        num_ut=num_ut,
        scenario="umi",
        min_ut_velocity=min_velocity,
        max_ut_velocity=max_velocity
    )
    
    # Unpack topology
    ut_loc = topology[0].numpy()[0] # [num_ut, 3]
    bs_loc = topology[1].numpy()[0] # [1, 3]
    ut_vel = topology[4].numpy()[0] # [num_ut, 3]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot BS
    ax.scatter(bs_loc[0, 0], bs_loc[0, 1], c='red', marker='^', s=300, label='Base Station', zorder=10, edgecolors='k')
    
    # Plot UTs
    vel_mag = np.linalg.norm(ut_vel, axis=1)
    sc = ax.scatter(ut_loc[:, 0], ut_loc[:, 1], c=vel_mag, cmap='viridis', s=150, label='User Terminal', zorder=5, edgecolors='k')
    cbar = plt.colorbar(sc, label='Velocity (m/s)')
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Velocity (m/s)', fontsize=14)
    
    # Add velocity vectors
    for i in range(num_ut):
        if vel_mag[i] > 0.1: # Only plot vector if moving
            ax.arrow(ut_loc[i, 0], ut_loc[i, 1], ut_vel[i, 0], ut_vel[i, 1], 
                     head_width=2, head_length=3, fc='k', ec='k', alpha=0.6)
            
    ax.set_xlabel("X (m)", fontsize=16)
    ax.set_ylabel("Y (m)", fontsize=16)
    ax.set_title("Smart Factory Topology with Mobility\n(Arrows indicate velocity vector)", fontsize=20)
    ax.legend(loc='upper right', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_aspect('equal')
    
    # Save
    output_path = Path(output_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"2D Topology plot saved to: {output_path.absolute()}")


def visualize_topology_3d(output_file="topology_3d_factory.png"):
    """Generate and plot topology in 3D."""
    
    # Configuration
    num_ut = 12
    min_velocity = 0.0
    max_velocity = 5.0
    
    print(f"Generating topology for {num_ut} UTs...")
    
    # Generate topology
    topology = gen_topology(
        batch_size=1,
        num_ut=num_ut,
        scenario="umi",
        min_ut_velocity=min_velocity,
        max_ut_velocity=max_velocity
    )
    
    # Unpack topology
    # [ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, bs_velocities]
    ut_loc = topology[0].numpy()[0] # [num_ut, 3]
    bs_loc = topology[1].numpy()[0] # [1, 3]
    ut_vel = topology[4].numpy()[0] # [num_ut, 3]
    
    # Force BS to be higher for visualization (e.g., 10m) if generated low
    bs_loc[0, 2] = max(bs_loc[0, 2], 10.0)
    
    # Force UTs to be on ground (z=0) for "factory floor" look
    ut_loc[:, 2] = 0.0
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Plot Base Station
    ax.scatter(bs_loc[0, 0], bs_loc[0, 1], bs_loc[0, 2], 
              c='orange', marker='^', s=300, label='Base Station', zorder=10, edgecolors='k')
    
    # Draw "tower" line for BS
    ax.plot([bs_loc[0, 0], bs_loc[0, 0]], [bs_loc[0, 1], bs_loc[0, 1]], [0, bs_loc[0, 2]], 
            color='gray', linewidth=3, linestyle='-')
    
    # 2. Plot User Terminals
    vel_mag = np.linalg.norm(ut_vel, axis=1)
    is_mobile = vel_mag > 0.1
    
    # Static UTs (Robotic Arms)
    static_mask = ~is_mobile
    if np.any(static_mask):
        ax.scatter(ut_loc[static_mask, 0], ut_loc[static_mask, 1], ut_loc[static_mask, 2], 
                  c='blue', marker='s', s=150, label='Static UT (Robot Arm)', zorder=5, edgecolors='k')
        
    # Mobile UTs (AGVs/Forklifts)
    if np.any(is_mobile):
        ax.scatter(ut_loc[is_mobile, 0], ut_loc[is_mobile, 1], ut_loc[is_mobile, 2], 
                  c='green', marker='o', s=150, label='Mobile UT (AGV)', zorder=5, edgecolors='k')
        
        # Add velocity arrows for mobile UTs
        # Quiver in 3D
        ax.quiver(ut_loc[is_mobile, 0], ut_loc[is_mobile, 1], ut_loc[is_mobile, 2],
                 ut_vel[is_mobile, 0], ut_vel[is_mobile, 1], ut_vel[is_mobile, 2],
                 length=5.0, normalize=True, color='green', alpha=0.6, arrow_length_ratio=0.3)

    # 3. Draw connection lines (BS to UTs)
    for i in range(num_ut):
        ax.plot([bs_loc[0, 0], ut_loc[i, 0]], 
                [bs_loc[0, 1], ut_loc[i, 1]], 
                [bs_loc[0, 2], ut_loc[i, 2]], 
                color='cornflowerblue', linestyle='--', linewidth=1, alpha=0.5)

    # 4. Draw "Factory Floor"
    # Get limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # Draw a grid on z=0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
    zz = np.zeros_like(xx)
    ax.plot_wireframe(xx, yy, zz, color='gray', alpha=0.1)

    # Styling
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Height (m)")
    ax.set_title("6G Smart Factory Topology\n(Blue: Static Robots, Green: Mobile AGVs)", fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    
    # Set view angle for isometric-like look
    ax.view_init(elev=30, azim=45)
    
    # Save
    output_path = Path(output_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"3D Topology plot saved to: {output_path.absolute()}")

if __name__ == "__main__":
    # Configure environment
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    visualize_topology_2d()
    # visualize_topology_3d()
