#!/usr/bin/env python3
"""
Create comparison plots for 6G Sionna Baseline vs PSO Enhanced simulations.
Uses the latest stabilized results with matched parameters.
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
baseline_ber = [m['overall']['ber'] for m in baseline_metrics]
baseline_bler = [m['overall']['bler'] for m in baseline_metrics]
baseline_sinr = [m['overall']['sinr_db'] for m in baseline_metrics]
baseline_nmse = [m['overall']['nmse_db'] for m in baseline_metrics]

# PSO data
pso_metrics = pso_results['runs'][0]['metrics']
pso_ber = [m['overall']['ber'] for m in pso_metrics]
pso_bler = [m['overall']['bler'] for m in pso_metrics]
pso_sinr = [m['overall']['sinr_db'] for m in pso_metrics]
pso_nmse = [m['overall']['nmse_db'] for m in pso_metrics]

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('6G Smart Factory: Sionna Baseline vs PSO Enhanced (Stabilized Results)', 
             fontsize=16, fontweight='bold', y=0.995)

# Plot 1: BER
ax1 = axes[0, 0]
ax1.semilogy(ebno_db, baseline_ber, 'o-', label='Baseline (LS Linear)', 
             linewidth=2, markersize=8, color='#1f77b4')
ax1.semilogy(ebno_db, pso_ber, 's-', label='PSO Enhanced', 
             linewidth=2, markersize=8, color='#ff7f0e')
ax1.set_xlabel('Eb/No (dB)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Bit Error Rate (BER)', fontsize=11, fontweight='bold')
ax1.set_title('BER Performance', fontsize=12, fontweight='bold')
ax1.grid(True, which='both', alpha=0.3, linestyle='--')
ax1.legend(fontsize=10, loc='upper right')
ax1.set_ylim([1e-4, 1])

# Plot 2: BLER
ax2 = axes[0, 1]
ax2.semilogy(ebno_db, baseline_bler, 'o-', label='Baseline (LS Linear)', 
             linewidth=2, markersize=8, color='#1f77b4')
ax2.semilogy(ebno_db, pso_bler, 's-', label='PSO Enhanced', 
             linewidth=2, markersize=8, color='#ff7f0e')
ax2.set_xlabel('Eb/No (dB)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Block Error Rate (BLER)', fontsize=11, fontweight='bold')
ax2.set_title('BLER Performance', fontsize=12, fontweight='bold')
ax2.grid(True, which='both', alpha=0.3, linestyle='--')
ax2.legend(fontsize=10, loc='upper right')
ax2.set_ylim([1e-4, 1])

# Plot 3: SINR
ax3 = axes[1, 0]
ax3.plot(ebno_db, baseline_sinr, 'o-', label='Baseline (LS Linear)', 
         linewidth=2, markersize=8, color='#1f77b4')
ax3.plot(ebno_db, pso_sinr, 's-', label='PSO Enhanced', 
         linewidth=2, markersize=8, color='#ff7f0e')
ax3.set_xlabel('Eb/No (dB)', fontsize=11, fontweight='bold')
ax3.set_ylabel('SINR (dB)', fontsize=11, fontweight='bold')
ax3.set_title('Signal-to-Interference-plus-Noise Ratio', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(fontsize=10, loc='lower right')

# Plot 4: NMSE
ax4 = axes[1, 1]
ax4.plot(ebno_db, baseline_nmse, 'o-', label='Baseline (LS Linear)', 
         linewidth=2, markersize=8, color='#1f77b4')
ax4.plot(ebno_db, pso_nmse, 's-', label='PSO Enhanced', 
         linewidth=2, markersize=8, color='#ff7f0e')
ax4.set_xlabel('Eb/No (dB)', fontsize=11, fontweight='bold')
ax4.set_ylabel('NMSE (dB)', fontsize=11, fontweight='bold')
ax4.set_title('Normalized Mean Square Error (Channel Estimation)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.legend(fontsize=10, loc='lower right')

# Calculate average improvements
avg_ber_improvement = np.mean([(b - p) / b * 100 for b, p in zip(baseline_ber, pso_ber) if b > 0])
avg_nmse_improvement = np.mean([b - p for b, p in zip(baseline_nmse, pso_nmse)])

# Add text box with key improvements
improvement_text = (
    'Key Improvements (PSO vs Baseline):\n'
    f'• Avg BER: {avg_ber_improvement:.1f}% reduction\n'
    f'• Avg NMSE: {avg_nmse_improvement:.1f} dB improvement\n'
    '• Stabilized results with matched parameters\n'
    '  (batch_size=16, max_iter=500, target_errors=1000)'
)
fig.text(0.99, 0.01, improvement_text, 
         fontsize=8, ha='right', va='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Add simulation info
info_text = (
    f'Baseline: {baseline_results["estimator"].upper()} | '
    f'Duration: {baseline_results["duration"]:.1f}s\n'
    f'PSO Enhanced: {pso_results["estimator"].upper()} | '
    f'Duration: {pso_results["duration"]:.1f}s'
)
fig.text(0.01, 0.01, info_text, 
         fontsize=8, ha='left', va='bottom',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout(rect=[0, 0.03, 1, 0.99])

# Save plot
output_dir = Path("results/baseline_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = "20251123"
png_file = output_dir / f"comparison_stabilized_{timestamp}.png"
pdf_file = output_dir / f"comparison_stabilized_{timestamp}.pdf"

plt.savefig(png_file, dpi=300, bbox_inches='tight')
plt.savefig(pdf_file, bbox_inches='tight')

print(f"✓ Stabilized comparison plot saved to:")
print(f"  - {png_file}")
print(f"  - {pdf_file}")
print(f"\nAverage BER improvement: {avg_ber_improvement:.1f}%")
print(f"Average NMSE improvement: {avg_nmse_improvement:.1f} dB")

# Display plot
plt.show()
