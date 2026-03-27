#!/usr/bin/env python3
"""Generate comprehensive comparison plots for the Factory6G simulation report."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")
PLOTS_DIR = os.path.join(BASE, "reports", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Common plot styling
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']


def load_stage(path):
    """Load a stage_results_v2.json file."""
    with open(path) as f:
        return json.load(f)


def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {name}")


# ── Data sources ──────────────────────────────────────────────────────────────

# Estimators (individual runs on small factory, QPSK, TR38901 UMi)
EST_SOURCES = {
    'LS':       os.path.join(RESULTS, "20260319_103327_ls_umi_qpsk_s/estimators/stage_results_v2.json"),
    'ISTA':     os.path.join(RESULTS, "20260319_102908_ista_umi_qpsk_s/estimators/stage_results_v2.json"),
    'Neural':   os.path.join(RESULTS, "20260319_110248_neural_umi_qpsk_s/estimators/stage_results_v2.json"),
    'DFT':      os.path.join(RESULTS, "20260319_083829_dft_umi_qpsk_m/estimators/stage_results_v2.json"),
    'Adaptive': os.path.join(RESULTS, "20260318_094228_adaptive_umi_qpsk/estimators/stage_results_v2.json"),
    'PSO':      os.path.join(RESULTS, "20260317_083505_ls_pso_umi_qpsk/estimators/stage_results_v2.json"),
}

# Multi-modulation (LS estimator, TR38901 UMi, small factory)
MOD_BASE = os.path.join(RESULTS, "20260318_172603_ls_umi_qpsk_16qam_64qam")
MOD_SOURCES = {
    'QPSK':    os.path.join(MOD_BASE, "low/estimators/stage_results_v2.json"),
    '16-QAM':  os.path.join(MOD_BASE, "mid/estimators/stage_results_v2.json"),
    '64-QAM':  os.path.join(MOD_BASE, "high/estimators/stage_results_v2.json"),
}

# Channel models (LS estimator, QPSK, small factory)
CH_BASE = os.path.join(RESULTS, "20260318_200508_ls_rayleigh_rician_umi_qpsk")
CH_SOURCES = {
    'Rayleigh': os.path.join(CH_BASE, "rayleigh/estimators/stage_results_v2.json"),
    'Rician':   os.path.join(CH_BASE, "rician/estimators/stage_results_v2.json"),
    'TR 38.901 UMi': os.path.join(CH_BASE, "tr38901/estimators/stage_results_v2.json"),
}

# Factory sizes (LS estimator, QPSK, TR38901 UMi)
SZ_BASE = os.path.join(RESULTS, "20260319_072409_ls_umi_qpsk_s_m_l")
SZ_SOURCES = {
    'Small (15x15 m)':  os.path.join(SZ_BASE, "s/estimators/stage_results_v2.json"),
    'Medium (25x25 m)': os.path.join(SZ_BASE, "m/estimators/stage_results_v2.json"),
    'Large (40x40 m)':  os.path.join(SZ_BASE, "l/estimators/stage_results_v2.json"),
}

# JIDD-SCMA
JIDD_SOURCES = {
    'JIDD Run 1 (buggy)': os.path.join(RESULTS, "20260320_083455_jidd_scma/jidd_scma/stage_results_v2.json"),
    'JIDD Run 2 (fixed)': os.path.join(RESULTS, "20260320_171006_jidd_scma/jidd_scma/stage_results_v2.json"),
}


def replace_zero_ber(ber_list):
    """Replace 0 with None for log-scale plotting."""
    return [b if b > 0 else None for b in ber_list]


def plot_ber(ax, ebno, ber, label, color, marker, linestyle='-'):
    """Plot BER handling zero values on log scale."""
    ber_clean = replace_zero_ber(ber)
    eb_plot = [e for e, b in zip(ebno, ber_clean) if b is not None]
    b_plot = [b for b in ber_clean if b is not None]
    ax.semilogy(eb_plot, b_plot, marker=marker, color=color, label=label,
                linestyle=linestyle, markersize=6, linewidth=1.8)


# ── 1. Estimator BER comparison ──────────────────────────────────────────────
print("1. Estimator BER comparison...")
fig, ax = plt.subplots()
for i, (name, path) in enumerate(EST_SOURCES.items()):
    data = load_stage(path)
    ebno = data['ebno_db_range']
    # For PSO run, the file has both ls and pso; pick pso
    if name == 'PSO':
        ber = data['methods']['pso']['ber']
    elif name in ('LS', 'ISTA', 'Neural', 'DFT'):
        method_key = list(data['methods'].keys())[0]
        ber = data['methods'][method_key]['ber']
    else:  # Adaptive
        method_key = list(data['methods'].keys())[0]
        ber = data['methods'][method_key]['ber']
    plot_ber(ax, ebno, ber, name, COLORS[i], MARKERS[i])

ax.set_xlabel('Eb/N0 (dB)')
ax.set_ylabel('Bit Error Rate (BER)')
ax.set_title('Channel Estimator BER Comparison (QPSK, TR 38.901 UMi)')
ax.legend(loc='best')
ax.set_xlim(-1, 21)
ax.set_ylim(1e-5, 1)
save(fig, 'estimator_ber_vs_ebno.png')


# ── 2. Estimator Latency comparison ──────────────────────────────────────────
print("2. Estimator latency comparison...")
fig, ax = plt.subplots()
for i, (name, path) in enumerate(EST_SOURCES.items()):
    data = load_stage(path)
    ebno = data['ebno_db_range']
    if name == 'PSO':
        lat = data['methods']['pso']['latency_ms']
    else:
        method_key = list(data['methods'].keys())[0]
        lat = data['methods'][method_key]['latency_ms']
    ax.plot(ebno, lat, marker=MARKERS[i], color=COLORS[i], label=name,
            markersize=6, linewidth=1.8)

ax.set_xlabel('Eb/N0 (dB)')
ax.set_ylabel('Latency (ms)')
ax.set_title('Channel Estimator Latency Comparison')
ax.legend(loc='best')
save(fig, 'estimator_latency_vs_ebno.png')


# ── 3. Estimator Runtime bar chart ───────────────────────────────────────────
print("3. Estimator runtime comparison...")
fig, ax = plt.subplots(figsize=(10, 5))
runtimes = {}
for name, path in EST_SOURCES.items():
    data = load_stage(path)
    if name == 'PSO':
        rt = sum(data['methods']['pso']['runtime_sec'])
    else:
        method_key = list(data['methods'].keys())[0]
        rt = sum(data['methods'][method_key]['runtime_sec'])
    runtimes[name] = rt

names = list(runtimes.keys())
vals = list(runtimes.values())
bars = ax.barh(names, vals, color=COLORS[:len(names)])
ax.set_xlabel('Total Runtime (seconds)')
ax.set_title('Channel Estimator Total Runtime')
for bar, val in zip(bars, vals):
    label = f'{val:.0f}s' if val < 3600 else f'{val/3600:.1f}h'
    ax.text(bar.get_width() + max(vals)*0.01, bar.get_y() + bar.get_height()/2,
            label, va='center', fontsize=10)
ax.set_xlim(0, max(vals) * 1.15)
save(fig, 'estimator_runtime.png')


# ── 4. Modulation BER comparison ─────────────────────────────────────────────
print("4. Modulation BER comparison...")
fig, ax = plt.subplots()
for i, (name, path) in enumerate(MOD_SOURCES.items()):
    data = load_stage(path)
    ebno = data['ebno_db_range']
    ber = data['methods']['ls']['ber']
    plot_ber(ax, ebno, ber, name, COLORS[i], MARKERS[i])

ax.set_xlabel('Eb/N0 (dB)')
ax.set_ylabel('Bit Error Rate (BER)')
ax.set_title('Modulation Order Impact on BER (LS Estimator, TR 38.901 UMi)')
ax.legend(loc='best')
ax.set_xlim(-1, 21)
ax.set_ylim(1e-4, 1)
save(fig, 'modulation_ber_vs_ebno.png')


# ── 5. Modulation Latency comparison ─────────────────────────────────────────
print("5. Modulation latency comparison...")
fig, ax = plt.subplots()
for i, (name, path) in enumerate(MOD_SOURCES.items()):
    data = load_stage(path)
    ebno = data['ebno_db_range']
    lat = data['methods']['ls']['latency_ms']
    ax.plot(ebno, lat, marker=MARKERS[i], color=COLORS[i], label=name,
            markersize=6, linewidth=1.8)

ax.set_xlabel('Eb/N0 (dB)')
ax.set_ylabel('Latency (ms)')
ax.set_title('Modulation Order Impact on Latency')
ax.legend(loc='best')
save(fig, 'modulation_latency_vs_ebno.png')


# ── 6. Channel Model BER comparison ──────────────────────────────────────────
print("6. Channel model BER comparison...")
fig, ax = plt.subplots()
for i, (name, path) in enumerate(CH_SOURCES.items()):
    data = load_stage(path)
    ebno = data['ebno_db_range']
    ber = data['methods']['ls']['ber']
    plot_ber(ax, ebno, ber, name, COLORS[i], MARKERS[i])

ax.set_xlabel('Eb/N0 (dB)')
ax.set_ylabel('Bit Error Rate (BER)')
ax.set_title('Channel Model Impact on BER (LS Estimator, QPSK)')
ax.legend(loc='best')
ax.set_xlim(-1, 21)
ax.set_ylim(1e-6, 1)
save(fig, 'channel_model_ber_vs_ebno.png')


# ── 7. Factory Size BER comparison ───────────────────────────────────────────
print("7. Factory size BER comparison...")
fig, ax = plt.subplots()
for i, (name, path) in enumerate(SZ_SOURCES.items()):
    data = load_stage(path)
    ebno = data['ebno_db_range']
    ber = data['methods']['ls']['ber']
    plot_ber(ax, ebno, ber, name, COLORS[i], MARKERS[i])

ax.set_xlabel('Eb/N0 (dB)')
ax.set_ylabel('Bit Error Rate (BER)')
ax.set_title('Factory Size Impact on BER (LS Estimator, QPSK, TR 38.901 UMi)')
ax.legend(loc='best')
ax.set_xlim(-1, 21)
ax.set_ylim(1e-4, 1)
save(fig, 'factory_size_ber_vs_ebno.png')


# ── 8. JIDD-SCMA BER comparison (Run 1 vs Run 2) ────────────────────────────
print("8. JIDD-SCMA bug fix comparison...")
fig, ax = plt.subplots()
styles = [('--', 'x'), ('-', 'o')]
for i, (name, path) in enumerate(JIDD_SOURCES.items()):
    data = load_stage(path)
    ebno = data['ebno_db_range']
    ber = data['methods']['jidd_scma']['ber']
    ls, mk = styles[i]
    plot_ber(ax, ebno, ber, name, COLORS[i], mk, linestyle=ls)

ax.set_xlabel('Eb/N0 (dB)')
ax.set_ylabel('Bit Error Rate (BER)')
ax.set_title('JIDD-SCMA: Bug Fix Impact (Run 1 vs Run 2)')
ax.legend(loc='best')
ax.set_xlim(0, 21)
ax.set_ylim(1e-7, 1)
save(fig, 'jidd_ber_comparison.png')


# ── 9. Combined cross-system BER ─────────────────────────────────────────────
print("9. Combined cross-system BER comparison...")
fig, ax = plt.subplots()

# LS baseline
data = load_stage(EST_SOURCES['LS'])
plot_ber(ax, data['ebno_db_range'], data['methods'][list(data['methods'].keys())[0]]['ber'],
         'LS (baseline)', COLORS[0], 'o')

# Adaptive
data = load_stage(EST_SOURCES['Adaptive'])
plot_ber(ax, data['ebno_db_range'], data['methods'][list(data['methods'].keys())[0]]['ber'],
         'Adaptive (best estimator)', COLORS[2], '^')

# PSO
data = load_stage(EST_SOURCES['PSO'])
plot_ber(ax, data['ebno_db_range'], data['methods']['pso']['ber'],
         'PSO', COLORS[4], 'D')

# JIDD-SCMA Run 2
data = load_stage(JIDD_SOURCES['JIDD Run 2 (fixed)'])
plot_ber(ax, data['ebno_db_range'], data['methods']['jidd_scma']['ber'],
         'JIDD-SCMA (Polar+SCMA)', COLORS[3], 's')

ax.set_xlabel('Eb/N0 (dB)')
ax.set_ylabel('Bit Error Rate (BER)')
ax.set_title('Cross-System BER Comparison')
ax.legend(loc='best')
ax.set_xlim(-1, 21)
ax.set_ylim(1e-7, 1)
save(fig, 'combined_ber.png')


# ── 10. Overall runtime comparison ───────────────────────────────────────────
print("10. Overall runtime comparison...")
fig, ax = plt.subplots(figsize=(10, 5))

all_runtimes = {}
# Estimators
for name, path in EST_SOURCES.items():
    data = load_stage(path)
    if name == 'PSO':
        rt = sum(data['methods']['pso']['runtime_sec'])
    else:
        method_key = list(data['methods'].keys())[0]
        rt = sum(data['methods'][method_key]['runtime_sec'])
    all_runtimes[name] = rt

# JIDD runs
for name, path in JIDD_SOURCES.items():
    data = load_stage(path)
    rt = sum(data['methods']['jidd_scma']['runtime_sec'])
    all_runtimes[name] = rt

names = list(all_runtimes.keys())
vals = list(all_runtimes.values())
colors_bar = COLORS[:len(names)]
bars = ax.barh(names, vals, color=colors_bar)
ax.set_xlabel('Total Runtime (seconds)')
ax.set_title('Overall Runtime Comparison (All Methods)')
ax.set_xscale('log')
for bar, val in zip(bars, vals):
    label = f'{val:.0f}s' if val < 3600 else f'{val/3600:.1f}h'
    ax.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height()/2,
            label, va='center', fontsize=10)
save(fig, 'runtime_comparison.png')


print("\nAll plots generated successfully in reports/plots/")
