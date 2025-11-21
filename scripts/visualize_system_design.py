#!/usr/bin/env python3
"""
Visualize System Design Topology (PHY/MAC Coordination).

Generates a block diagram showing the interaction between:
- MAC Layer (Resource Manager)
- PHY Layer (Transmitter, Channel, Receiver)
- Coordination Signals (Scheduling, Feedback)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def draw_system_design():
    output_dir = Path("results/system_design")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Colors
    COLOR_MAC = '#FFD700' # Gold
    COLOR_PHY_TX = '#87CEEB' # SkyBlue
    COLOR_PHY_CH = '#D3D3D3' # LightGray
    COLOR_PHY_RX = '#90EE90' # LightGreen
    COLOR_ARROW = 'black'
    
    # Helper to draw box
    def draw_box(x, y, w, h, color, label, sublabels=[]):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                                      linewidth=2, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        
        # Main Label
        ax.text(x + w/2, y + h - 0.4, label, ha='center', va='top', 
                fontsize=14, fontweight='bold')
        
        # Sublabels
        # Distribute vertically
        start_y = y + h - 1.0
        step = 0.5
        for i, sl in enumerate(sublabels):
            ax.text(x + w/2, start_y - i*step, sl, ha='center', va='top', fontsize=11)
            
    # Helper to draw arrow
    def draw_arrow(x1, y1, x2, y2, label=None, curve=0.0):
        style = f"Simple, tail_width=0.5, head_width=4, head_length=8"
        kw = dict(arrowstyle=style, color=COLOR_ARROW)
        if curve:
            connectionstyle = f"arc3,rad={curve}"
            arrow = patches.FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=connectionstyle, **kw)
        else:
            arrow = patches.FancyArrowPatch((x1, y1), (x2, y2), **kw)
        ax.add_patch(arrow)
        
        if label:
            # Position label near middle
            mx, my = (x1+x2)/2, (y1+y2)/2
            if curve:
                # Adjust label position for curved arrows
                if x1 < x2: # MAC to Tx
                    mx -= 1.0
                    my += 0.5
                else: # Rx to MAC
                    mx += 1.0
                    my += 0.5
            ax.text(mx, my + 0.3, label, ha='center', va='bottom', fontsize=11, fontweight='bold', 
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # ---------------------------------------------------------
    # 1. MAC Layer (Top)
    # ---------------------------------------------------------
    # Increased height to 3.5, lowered y to 8.0
    draw_box(6, 8.0, 8, 3.5, COLOR_MAC, "MAC Layer (Resource Manager)", 
             ["User Scheduling", "Power Control", "Pilot Allocation", "Link Adaptation (MCS)"])
    
    # ---------------------------------------------------------
    # 2. PHY Layer (Bottom)
    # ---------------------------------------------------------
    
    # Increased height to 3.5, lowered y to 2.0
    # Transmitter
    draw_box(1, 2.0, 4, 3.5, COLOR_PHY_TX, "PHY Transmitter", 
             ["LDPC Encoder", "QAM Mapper", "Resource Grid Mapper", "MIMO Precoding"])
    
    # Channel
    draw_box(8, 2.0, 4, 3.5, COLOR_PHY_CH, "Wireless Channel", 
             ["3GPP TR 38.901", "Rayleigh Fading", "Pathloss & Shadowing", "AWGN"])
    
    # Receiver
    draw_box(15, 2.0, 4, 3.5, COLOR_PHY_RX, "PHY Receiver", 
             ["Channel Estimation", "MIMO Equalization", "Demapping (LLR)", "LDPC Decoder"])
    
    # ---------------------------------------------------------
    # 3. Connections
    # ---------------------------------------------------------
    
    # PHY Chain (Tx -> Ch -> Rx)
    # y center is 2.0 + 3.5/2 = 3.75
    draw_arrow(5.2, 3.75, 7.8, 3.75, "Transmitted Signal (x)")
    draw_arrow(12.2, 3.75, 14.8, 3.75, "Received Signal (y)")
    
    # MAC -> PHY (Control)
    # From MAC Bottom (y=8.0) to Tx Top (y=5.5)
    # MAC x center is 10. Tx x center is 3.
    draw_arrow(6.0, 9.0, 3.0, 5.7, "Resource Allocation\nMCS, Power", curve=0.2)
    
    # PHY -> MAC (Feedback)
    # From Rx Top (y=5.5) to MAC Right/Bottom
    # Rx x center is 17. MAC x center is 10.
    draw_arrow(17.0, 5.7, 14.0, 9.0, "CSI Feedback (CQI/PMI)\nACK/NACK", curve=0.2)
    
    # ---------------------------------------------------------
    # Title
    # ---------------------------------------------------------
    ax.text(10, 0.5, "System Design Topology: PHY/MAC Coordination", 
            ha='center', va='center', fontsize=20, fontweight='bold')
    
    # Save
    plot_path = output_dir / "phy_mac_topology.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"System topology diagram saved to: {plot_path.absolute()}")

if __name__ == "__main__":
    draw_system_design()
