#!/usr/bin/env python3
"""Generate comprehensive comparison plots for the Factory6G simulation report.

Legend ordering: methods are sorted by BER performance (best first).
"""

import glob
import json
import os
import numpy as np
from scipy.special import erfc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")
PLOTS_DIR = os.path.join(BASE, "reports", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

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
COLORS = ['#2ca02c', '#9467bd', '#1f77b4', '#e377c2', '#ff7f0e', '#d62728', '#8c564b', '#7f7f7f']


def load_stage(path):
    with open(path) as f:
        return json.load(f)


def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved {name}")


def get_method_ber(data, method_key):
    return data['methods'][method_key]['ber']


def replace_zero_ber(ber_list):
    return [b if b > 0 else None for b in ber_list]


def plot_ber(ax, ebno, ber, label, color, marker, linestyle='-'):
    ber_clean = replace_zero_ber(ber)
    eb_plot = [e for e, b in zip(ebno, ber_clean) if b is not None]
    b_plot = [b for b in ber_clean if b is not None]
    ax.semilogy(eb_plot, b_plot, marker=marker, color=color, label=label,
                linestyle=linestyle, markersize=6, linewidth=1.8)


# ── Data sources ──────────────────────────────────────────────────────────────

# Estimators — ordered by BER performance (best first)
EST_SOURCES = [
    ('Adaptive', os.path.join(RESULTS, "20260318_094228_adaptive_umi_qpsk/estimators/stage_results_v2.json"), None),
    ('PSO',      os.path.join(RESULTS, "20260317_083505_ls_pso_umi_qpsk/estimators/stage_results_v2.json"), 'pso'),
    ('DFT',      os.path.join(RESULTS, "20260415_052116_dft_umi_qpsk_s/estimators/stage_results_v2.json"), None),
    ('LS',       os.path.join(RESULTS, "20260319_103327_ls_umi_qpsk_s/estimators/stage_results_v2.json"), None),
    ('Neural',   os.path.join(RESULTS, "20260319_110248_neural_umi_qpsk_s/estimators/stage_results_v2.json"), None),
    ('ISTA',     os.path.join(RESULTS, "20260319_102908_ista_umi_qpsk_s/estimators/stage_results_v2.json"), None),
]

MOD_SOURCES = [
    ('QPSK',   os.path.join(RESULTS, "20260318_172603_ls_umi_qpsk_16qam_64qam/low/estimators/stage_results_v2.json")),
    ('16-QAM', os.path.join(RESULTS, "20260318_172603_ls_umi_qpsk_16qam_64qam/mid/estimators/stage_results_v2.json")),
    ('64-QAM', os.path.join(RESULTS, "20260318_172603_ls_umi_qpsk_16qam_64qam/high/estimators/stage_results_v2.json")),
]

CH_SOURCES = [
    ('Rayleigh',       os.path.join(RESULTS, "20260318_200508_ls_rayleigh_rician_umi_qpsk/rayleigh/estimators/stage_results_v2.json")),
    ('Rician (K=1)',   os.path.join(RESULTS, "20260318_200508_ls_rayleigh_rician_umi_qpsk/rician/estimators/stage_results_v2.json")),
    ('TR 38.901 UMi',  os.path.join(RESULTS, "20260318_200508_ls_rayleigh_rician_umi_qpsk/tr38901/estimators/stage_results_v2.json")),
]

SZ_SOURCES = [
    ('Small (15x15 m)',  os.path.join(RESULTS, "20260319_072409_ls_umi_qpsk_s_m_l/s/estimators/stage_results_v2.json")),
    ('Medium (25x25 m)', os.path.join(RESULTS, "20260319_072409_ls_umi_qpsk_s_m_l/m/estimators/stage_results_v2.json")),
    ('Large (40x40 m)',  os.path.join(RESULTS, "20260319_072409_ls_umi_qpsk_s_m_l/l/estimators/stage_results_v2.json")),
]

JIDD_SOURCES = [
    ('JIDD Run 1 (buggy)', os.path.join(RESULTS, "20260320_083455_jidd_scma/jidd_scma/stage_results_v2.json")),
    ('JIDD Run 2 (fixed)', os.path.join(RESULTS, "20260320_171006_jidd_scma/jidd_scma/stage_results_v2.json")),
]


def _est_ber_and_meta(name, path, forced_key):
    """Load BER, latency, runtime for an estimator entry."""
    data = load_stage(path)
    key = forced_key or list(data['methods'].keys())[0]
    m = data['methods'][key]
    return data['ebno_db_range'], m['ber'], m['latency_ms'], sum(m['runtime_sec'])


# ── 1. Estimator BER comparison (ordered best→worst) ─────────────────────────
print("1. Estimator BER comparison...")
fig, ax = plt.subplots()
for i, (name, path, fkey) in enumerate(EST_SOURCES):
    ebno, ber, _, _ = _est_ber_and_meta(name, path, fkey)
    plot_ber(ax, ebno, ber, name, COLORS[i], MARKERS[i])
ax.set_xlabel('Eb/N0 (dB)')
ax.set_ylabel('Bit Error Rate (BER)')
ax.set_title('Channel Estimator BER Comparison (QPSK, TR 38.901 UMi)')
ax.legend(loc='upper right')
ax.set_xlim(-1, 21)
ax.set_ylim(1e-5, 1)
save(fig, 'estimator_ber_vs_ebno.png')

# ── 2. Estimator Latency ─────────────────────────────────────────────────────
print("2. Estimator latency comparison...")
fig, ax = plt.subplots()
for i, (name, path, fkey) in enumerate(EST_SOURCES):
    ebno, _, lat, _ = _est_ber_and_meta(name, path, fkey)
    ax.plot(ebno, lat, marker=MARKERS[i], color=COLORS[i], label=name,
            markersize=6, linewidth=1.8)
ax.set_xlabel('Eb/N0 (dB)')
ax.set_ylabel('Latency (ms)')
ax.set_title('Channel Estimator Latency Comparison')
ax.legend(loc='best')
save(fig, 'estimator_latency_vs_ebno.png')

# ── 3. Estimator Runtime bar chart (sorted by runtime) ───────────────────────
print("3. Estimator runtime comparison...")
fig, ax = plt.subplots(figsize=(10, 5))
rt_data = [(n, _est_ber_and_meta(n, p, fk)[3]) for n, p, fk in EST_SOURCES]
rt_data.sort(key=lambda x: x[1])
names_rt = [n for n, _ in rt_data]
vals_rt = [v for _, v in rt_data]
bars = ax.barh(names_rt, vals_rt, color=COLORS[:len(names_rt)])
ax.set_xlabel('Total Runtime (seconds)')
ax.set_title('Channel Estimator Total Runtime')
for bar, val in zip(bars, vals_rt):
    label = f'{val:.0f}s' if val < 3600 else f'{val/3600:.1f}h'
    ax.text(bar.get_width() + max(vals_rt)*0.01, bar.get_y() + bar.get_height()/2,
            label, va='center', fontsize=10)
ax.set_xlim(0, max(vals_rt) * 1.15)
save(fig, 'estimator_runtime.png')

# ── 4. Modulation BER (ordered best→worst) ───────────────────────────────────
print("4. Modulation BER comparison...")
fig, ax = plt.subplots()
for i, (name, path) in enumerate(MOD_SOURCES):
    data = load_stage(path)
    plot_ber(ax, data['ebno_db_range'], data['methods']['ls']['ber'], name, COLORS[i], MARKERS[i])
ax.set_xlabel('Eb/N0 (dB)')
ax.set_ylabel('Bit Error Rate (BER)')
ax.set_title('Modulation Order Impact on BER (LS Estimator, TR 38.901 UMi)')
ax.legend(loc='upper right')
ax.set_xlim(-1, 21)
ax.set_ylim(1e-4, 1)
save(fig, 'modulation_ber_vs_ebno.png')

# ── 5. Modulation Latency ────────────────────────────────────────────────────
print("5. Modulation latency comparison...")
fig, ax = plt.subplots()
for i, (name, path) in enumerate(MOD_SOURCES):
    data = load_stage(path)
    ax.plot(data['ebno_db_range'], data['methods']['ls']['latency_ms'],
            marker=MARKERS[i], color=COLORS[i], label=name, markersize=6, linewidth=1.8)
ax.set_xlabel('Eb/N0 (dB)')
ax.set_ylabel('Latency (ms)')
ax.set_title('Modulation Order Impact on Latency')
ax.legend(loc='best')
save(fig, 'modulation_latency_vs_ebno.png')

# ── 6. Channel Model BER (ordered best→worst) + theoretical curves ───────────
print("6. Channel model BER comparison (with theoretical curves)...")
fig, ax = plt.subplots()
for i, (name, path) in enumerate(CH_SOURCES):
    data = load_stage(path)
    plot_ber(ax, data['ebno_db_range'], data['methods']['ls']['ber'], name, COLORS[i], MARKERS[i])

# Theoretical BER curves (uncoded QPSK)
ebno_theory = np.linspace(0, 20, 200)
ebno_lin = 10 ** (ebno_theory / 10)
# AWGN: BER_QPSK = erfc(sqrt(Eb/N0)) / 2
ber_awgn = erfc(np.sqrt(ebno_lin)) / 2
# Rayleigh flat-fading (no diversity): BER_QPSK = 0.5 * (1 - sqrt(gamma / (1 + gamma)))
ber_rayleigh_theory = 0.5 * (1 - np.sqrt(ebno_lin / (1 + ebno_lin)))
ax.semilogy(ebno_theory, ber_awgn, '--', color='#555555', linewidth=1.2, label='Theory: AWGN (uncoded)')
ax.semilogy(ebno_theory, ber_rayleigh_theory, ':', color='#555555', linewidth=1.2, label='Theory: Rayleigh (uncoded)')

ax.set_xlabel('Eb/N0 (dB)')
ax.set_ylabel('Bit Error Rate (BER)')
ax.set_title('Channel Model Impact on BER (LS Estimator, QPSK)')
ax.legend(loc='upper right')
ax.set_xlim(-1, 21)
ax.set_ylim(1e-6, 1)
save(fig, 'channel_model_ber_vs_ebno.png')

# ── 7. Factory Size BER (ordered best→worst) ────────────────────────────────
print("7. Factory size BER comparison...")
fig, ax = plt.subplots()
for i, (name, path) in enumerate(SZ_SOURCES):
    data = load_stage(path)
    plot_ber(ax, data['ebno_db_range'], data['methods']['ls']['ber'], name, COLORS[i], MARKERS[i])
ax.set_xlabel('Eb/N0 (dB)')
ax.set_ylabel('Bit Error Rate (BER)')
ax.set_title('Factory Size Impact on BER (LS Estimator, QPSK, TR 38.901 UMi)')
ax.legend(loc='best')
ax.set_xlim(-1, 21)
ax.set_ylim(1e-4, 1)
save(fig, 'factory_size_ber_vs_ebno.png')

# ── 8. JIDD-SCMA BER comparison ─────────────────────────────────────────────
print("8. JIDD-SCMA bug fix comparison...")
fig, ax = plt.subplots()
styles = [('--', 'x'), ('-', 'o')]
for i, (name, path) in enumerate(JIDD_SOURCES):
    data = load_stage(path)
    ls, mk = styles[i]
    plot_ber(ax, data['ebno_db_range'], data['methods']['jidd_scma']['ber'],
             name, COLORS[i], mk, linestyle=ls)
ax.set_xlabel('Eb/N0 (dB)')
ax.set_ylabel('Bit Error Rate (BER)')
ax.set_title('JIDD-SCMA: Bug Fix Impact (Run 1 vs Run 2)')
ax.legend(loc='best')
ax.set_xlim(0, 21)
ax.set_ylim(1e-7, 1)
save(fig, 'jidd_ber_comparison.png')

# ── 9. Combined cross-system BER (ordered best→worst) ───────────────────────
print("9. Combined cross-system BER comparison...")
fig, ax = plt.subplots()

# JIDD-SCMA Run 2 (best at high SNR)
data = load_stage(JIDD_SOURCES[1][1])
plot_ber(ax, data['ebno_db_range'], data['methods']['jidd_scma']['ber'],
         'JIDD-SCMA (Polar+SCMA)', '#d62728', 's')

# Adaptive
ebno, ber, _, _ = _est_ber_and_meta(*EST_SOURCES[0])
plot_ber(ax, ebno, ber, 'Adaptive (best estimator)', '#2ca02c', '^')

# PSO
ebno, ber, _, _ = _est_ber_and_meta(*EST_SOURCES[1])
plot_ber(ax, ebno, ber, 'PSO', '#9467bd', 'D')

# LS baseline
ebno, ber, _, _ = _est_ber_and_meta(*EST_SOURCES[2])
plot_ber(ax, ebno, ber, 'LS (baseline)', '#1f77b4', 'o')

ax.set_xlabel('Eb/N0 (dB)')
ax.set_ylabel('Bit Error Rate (BER)')
ax.set_title('Cross-System BER Comparison')
ax.legend(loc='upper right')
ax.set_xlim(-1, 21)
ax.set_ylim(1e-7, 1)
save(fig, 'combined_ber.png')

# ── 10. Overall runtime comparison (sorted by runtime) ──────────────────────
print("10. Overall runtime comparison...")
fig, ax = plt.subplots(figsize=(10, 5))

all_rt = [(n, _est_ber_and_meta(n, p, fk)[3]) for n, p, fk in EST_SOURCES]
for name, path in JIDD_SOURCES:
    data = load_stage(path)
    all_rt.append((name, sum(data['methods']['jidd_scma']['runtime_sec'])))
all_rt.sort(key=lambda x: x[1])

names_all = [n for n, _ in all_rt]
vals_all = [v for _, v in all_rt]
bars = ax.barh(names_all, vals_all, color=COLORS[:len(names_all)])
ax.set_xlabel('Total Runtime (seconds)')
ax.set_title('Overall Runtime Comparison (All Methods)')
ax.set_xscale('log')
for bar, val in zip(bars, vals_all):
    label = f'{val:.0f}s' if val < 3600 else f'{val/3600:.1f}h'
    ax.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height()/2,
            label, va='center', fontsize=10)
save(fig, 'runtime_comparison.png')


# ── 11. Neural vs LS on Rayleigh (if data available) ─────────────────────────
# Auto-discover the most recent neural+ls run on rayleigh
neural_rayleigh_dirs = sorted(glob.glob(os.path.join(RESULTS, "*neural*ls*")))
if not neural_rayleigh_dirs:
    neural_rayleigh_dirs = sorted(glob.glob(os.path.join(RESULTS, "*ls*neural*")))

# Also try any run directory that has both neural and ls in its estimators stage
if not neural_rayleigh_dirs:
    for d in sorted(glob.glob(os.path.join(RESULTS, "*/"))):
        # Check all subdirectories for rayleigh
        rayleigh_stage = os.path.join(d, "rayleigh", "estimators", "stage_results_v2.json")
        low_stage = os.path.join(d, "low", "estimators", "stage_results_v2.json")
        direct_stage = os.path.join(d, "estimators", "stage_results_v2.json")
        for candidate in [rayleigh_stage, low_stage, direct_stage]:
            if os.path.exists(candidate):
                try:
                    data = load_stage(candidate)
                    if 'neural' in data['methods'] and 'ls' in data['methods']:
                        neural_rayleigh_dirs.append(d.rstrip('/'))
                except Exception:
                    pass

if neural_rayleigh_dirs:
    latest_run = neural_rayleigh_dirs[-1]
    print(f"\n11. Neural vs LS plots from {latest_run}...")

    # Look for modulation sub-runs (low/mid/high) or direct estimators/
    sub_dirs = {}
    for sub in ['low', 'mid', 'high']:
        stage_path = os.path.join(latest_run, sub, "estimators", "stage_results_v2.json")
        if os.path.exists(stage_path):
            sub_dirs[sub] = stage_path

    # Also check for rayleigh sub-dir
    ray_path = os.path.join(latest_run, "rayleigh", "estimators", "stage_results_v2.json")
    if os.path.exists(ray_path):
        sub_dirs['rayleigh'] = ray_path

    # Direct (no sub-dir)
    direct_path = os.path.join(latest_run, "estimators", "stage_results_v2.json")
    if os.path.exists(direct_path):
        sub_dirs['direct'] = direct_path

    # Plot Neural vs LS BER for each sub-run found
    mod_names = {'low': 'QPSK', 'mid': '16-QAM', 'high': '64-QAM',
                 'rayleigh': 'Rayleigh', 'direct': 'QPSK'}

    # (a) If we have modulation variants, make a multi-modulation comparison
    if all(k in sub_dirs for k in ['low', 'mid', 'high']):
        print("  Neural vs LS across modulations...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        for ax_idx, mod_key in enumerate(['low', 'mid', 'high']):
            ax = axes[ax_idx]
            data = load_stage(sub_dirs[mod_key])
            ebno = data['ebno_db_range']
            for j, method in enumerate(['neural', 'ls']):
                if method in data['methods']:
                    label = 'Neural (retrained)' if method == 'neural' else 'LS'
                    color = '#2ca02c' if method == 'neural' else '#1f77b4'
                    marker = '^' if method == 'neural' else 'o'
                    plot_ber(ax, ebno, data['methods'][method]['ber'], label, color, marker)
            ax.set_xlabel('Eb/N0 (dB)')
            if ax_idx == 0:
                ax.set_ylabel('Bit Error Rate (BER)')
            ax.set_title(f'{mod_names[mod_key]}')
            ax.legend(loc='upper right', fontsize=9)
            ax.set_xlim(-1, 21)
            ax.set_ylim(1e-6, 1)
        fig.suptitle('Neural vs LS: Modulation Impact (Rayleigh Channel)', fontsize=14, y=1.02)
        fig.tight_layout()
        save(fig, 'neural_vs_ls_modulation_ber.png')

        # Latency comparison
        fig, ax = plt.subplots()
        for mod_key, ls_style in [('low', '-'), ('mid', '--'), ('high', ':')]:
            data = load_stage(sub_dirs[mod_key])
            ebno = data['ebno_db_range']
            for method in ['neural', 'ls']:
                if method in data['methods']:
                    label = f'{"Neural" if method == "neural" else "LS"} ({mod_names[mod_key]})'
                    color = '#2ca02c' if method == 'neural' else '#1f77b4'
                    ax.plot(ebno, data['methods'][method]['latency_ms'],
                            linestyle=ls_style, color=color, label=label,
                            marker='o' if method == 'ls' else '^', markersize=5, linewidth=1.5)
        ax.set_xlabel('Eb/N0 (dB)')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Neural vs LS Latency Across Modulations')
        ax.legend(loc='best', fontsize=8)
        save(fig, 'neural_vs_ls_latency.png')

    # (b) Single-run BER plot (rayleigh or direct)
    for tag in ['rayleigh', 'direct']:
        if tag in sub_dirs:
            data = load_stage(sub_dirs[tag])
            ebno = data['ebno_db_range']
            if 'neural' in data['methods'] and 'ls' in data['methods']:
                fig, ax = plt.subplots()
                plot_ber(ax, ebno, data['methods']['neural']['ber'],
                         'Neural (retrained)', '#2ca02c', '^')
                plot_ber(ax, ebno, data['methods']['ls']['ber'],
                         'LS (baseline)', '#1f77b4', 'o')
                ax.set_xlabel('Eb/N0 (dB)')
                ax.set_ylabel('Bit Error Rate (BER)')
                ch_label = 'Rayleigh' if tag == 'rayleigh' else 'QPSK'
                ax.set_title(f'Neural vs LS BER ({ch_label} Channel)')
                ax.legend(loc='upper right')
                ax.set_xlim(-1, 21)
                ax.set_ylim(1e-7, 1)
                save(fig, f'neural_vs_ls_{tag}_ber.png')
else:
    print("\n11. No neural vs LS run found — skipping. Run simulation first.")


print("\nAll plots generated successfully in reports/plots/")
