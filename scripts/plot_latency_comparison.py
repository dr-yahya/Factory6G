#!/usr/bin/env python3
"""
Create latency comparison plot for 6G Sionna Baseline vs PSO Enhanced simulations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load latest results
baseline_file = "results/6g_smart_factory_sionna_baseline/simulation_results_umi_ls_lin_6g_smart_factory_sionna_baseline_20251122_071742.json"
pso_file = "results/6g_pso_enhanced/simulation_results_umi_pso_6g_pso_enhanced_20251123_081704.json"

with open(baseline_file, 'r') as f:
    baseline_results = json.load(f)

with open(pso_file, 'r') as f:
    pso_results = json.load(f)

# Extract data
ebno_db = baseline_results['ebno_db']

# Baseline data
baseline_metrics = baseline_results['runs'][0]['metrics']
baseline_latency = [m['overall']['air_interface_latency_ms'] for m in baseline_metrics]

# PSO data
pso_metrics = pso_results['runs'][0]['metrics']
pso_latency = [m['overall']['air_interface_latency_ms'] for m in pso_metrics]

# Calculate improvement
latency_improvement = [(b - p) / b * 100 for b, p in zip(baseline_latency, pso_latency)]
avg_improvement = np.mean(latency_improvement)

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('6G Smart Factory: Latency Comparison (Baseline vs PSO Enhanced)', 
             fontsize=16, fontweight='bold')

# Plot 1: Latency vs Eb/No
ax1.plot(ebno_db, baseline_latency, 'o-', label='Baseline (LS Linear)', 
         linewidth=2.5, markersize=10, color='#1f77b4')
ax1.plot(ebno_db, pso_latency, 's-', label='PSO Enhanced', 
         linewidth=2.5, markersize=10, color='#ff7f0e')
ax1.set_xlabel('Eb/No (dB)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Air Interface Latency (ms)', fontsize=12, fontweight='bold')
ax1.set_title('Latency Performance', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11, loc='upper right')

# Plot 2: Latency Improvement
colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in latency_improvement]
bars = ax2.bar(ebno_db, latency_improvement, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.axhline(y=avg_improvement, color='red', linestyle='--', linewidth=2, 
            label=f'Average: {avg_improvement:.1f}%')
ax2.set_xlabel('Eb/No (dB)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Latency Reduction (%)', fontsize=12, fontweight='bold')
ax2.set_title('PSO Latency Improvement', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
ax2.legend(fontsize=11, loc='upper right')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, latency_improvement)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}%',
             ha='center', va='bottom' if val > 0 else 'top',
             fontsize=9, fontweight='bold')

# Add statistics text box
stats_text = (
    f'Latency Statistics:\n'
    f'• Avg Baseline: {np.mean(baseline_latency):.3f} ms\n'
    f'• Avg PSO: {np.mean(pso_latency):.3f} ms\n'
    f'• Avg Reduction: {avg_improvement:.1f}%\n'
    f'• Max Reduction: {max(latency_improvement):.1f}%\n'
    f'• Min Reduction: {min(latency_improvement):.1f}%'
)
fig.text(0.99, 0.01, stats_text, 
         fontsize=9, ha='right', va='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.08, 1, 0.96])

# Save plot
output_dir = Path("results/baseline_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = "20251123"
png_file = output_dir / f"latency_comparison_{timestamp}.png"
pdf_file = output_dir / f"latency_comparison_{timestamp}.pdf"

plt.savefig(png_file, dpi=300, bbox_inches='tight')
plt.savefig(pdf_file, bbox_inches='tight')

print(f"✓ Latency comparison plot saved to:")
print(f"  - {png_file}")
print(f"  - {pdf_file}")
print(f"\nLatency Comparison Summary:")
print(f"  Average Baseline Latency: {np.mean(baseline_latency):.3f} ms")
print(f"  Average PSO Latency: {np.mean(pso_latency):.3f} ms")
print(f"  Average Latency Reduction: {avg_improvement:.1f}%")

# Display plot
plt.show()
