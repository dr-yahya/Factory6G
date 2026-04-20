"""Generate comparison plots from existing simulation results for the report."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path("/app")
RESULTS = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "reports" / "plots"

ESTIMATOR_SOURCES = {
    "LS": RESULTS / "20260319_103327_ls_umi_qpsk_s" / "estimators" / "stage_results_v2.csv",
    "ISTA": RESULTS / "20260319_102908_ista_umi_qpsk_s" / "estimators" / "stage_results_v2.csv",
    "Neural": RESULTS / "20260319_110248_neural_umi_qpsk_s" / "estimators" / "stage_results_v2.csv",
    "DFT": RESULTS / "20260415_052116_dft_umi_qpsk_s" / "estimators" / "stage_results_v2.csv",
    "Adaptive": RESULTS / "20260318_094228_adaptive_umi_qpsk" / "estimators" / "stage_results_v2.csv",
}

JIDD_SOURCES = {
    "JIDD-SCMA Run 1 (buggy)": RESULTS / "20260320_083455_jidd_scma" / "jidd_scma" / "stage_results_v2.csv",
    "JIDD-SCMA Run 2 (fixed)": RESULTS / "20260320_171006_jidd_scma" / "jidd_scma" / "stage_results_v2.csv",
}

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
MARKERS = ["o", "s", "D", "^", "v", "x", "*"]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric = df[df["ebno_db"].notna() & (df["ebno_db"] != "")]
    numeric = numeric.copy()
    numeric["ebno_db"] = pd.to_numeric(numeric["ebno_db"], errors="coerce")
    numeric["value"] = pd.to_numeric(numeric["value"], errors="coerce")
    numeric = numeric.dropna(subset=["ebno_db", "value"])
    return numeric


def get_metric(df: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    subset = df[df["metric"] == metric].sort_values("ebno_db")
    return subset["ebno_db"].values, subset["value"].values


def get_runtime_total(path: Path) -> float:
    df = pd.read_csv(path)
    row = df[(df["metric"] == "runtime_total_sec")]
    return float(row["value"].iloc[0])


def plot_estimator_ber():
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (label, path) in enumerate(ESTIMATOR_SOURCES.items()):
        df = load_csv(path)
        ebno, ber = get_metric(df, "ber")
        mask = ber > 0
        ax.semilogy(ebno[mask], ber[mask], marker=MARKERS[i], color=COLORS[i],
                     label=label, linewidth=1.5, markersize=6)
    ax.set_xlabel("Eb/N0 (dB)", fontsize=12)
    ax.set_ylabel("BER", fontsize=12)
    ax.set_title("Channel Estimators: BER vs Eb/N0", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "estimator_ber_vs_ebno.png", dpi=300)
    plt.close(fig)
    print("  -> estimator_ber_vs_ebno.png")


def plot_jidd_comparison():
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (label, path) in enumerate(JIDD_SOURCES.items()):
        df = load_csv(path)
        ebno, ber = get_metric(df, "ber")
        mask = ber > 0
        ax.semilogy(ebno[mask], ber[mask], marker=MARKERS[i], color=COLORS[i],
                     label=label, linewidth=1.5, markersize=6)
    ax.annotate("BER rebounds to ~0.5\n(numerical issue)",
                xy=(13, 0.49), xytext=(8, 0.3),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=10, color="red", fontweight="bold")
    ax.set_xlabel("Eb/N0 (dB)", fontsize=12)
    ax.set_ylabel("BER", fontsize=12)
    ax.set_title("JIDD-SCMA: BER Before and After Bug Fix", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "jidd_ber_comparison.png", dpi=300)
    plt.close(fig)
    print("  -> jidd_ber_comparison.png")


def plot_combined_ber():
    fig, ax = plt.subplots(figsize=(9, 6))
    sources = {
        "Adaptive (best estimator)": ESTIMATOR_SOURCES["Adaptive"],
        "LS (baseline)": ESTIMATOR_SOURCES["LS"],
        "JIDD-SCMA (fixed)": JIDD_SOURCES["JIDD-SCMA Run 2 (fixed)"],
    }
    for i, (label, path) in enumerate(sources.items()):
        df = load_csv(path)
        ebno, ber = get_metric(df, "ber")
        mask = ber > 0
        ax.semilogy(ebno[mask], ber[mask], marker=MARKERS[i], color=COLORS[i],
                     label=label, linewidth=1.5, markersize=6)
    ax.set_xlabel("Eb/N0 (dB)", fontsize=12)
    ax.set_ylabel("BER", fontsize=12)
    ax.set_title("Cross-System BER Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "combined_ber.png", dpi=300)
    plt.close(fig)
    print("  -> combined_ber.png")


def plot_runtime_comparison():
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = []
    runtimes = []
    for label, path in ESTIMATOR_SOURCES.items():
        labels.append(label)
        runtimes.append(get_runtime_total(path))
    for label, path in JIDD_SOURCES.items():
        short = label.replace("JIDD-SCMA ", "JIDD\n")
        labels.append(short)
        runtimes.append(get_runtime_total(path))

    bars = ax.bar(range(len(labels)), runtimes, color=COLORS[:len(labels)], edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Total Runtime (seconds, log scale)", fontsize=12)
    ax.set_title("Total Runtime Comparison", fontsize=14)
    ax.grid(True, which="both", alpha=0.3, axis="y")
    for bar, val in zip(bars, runtimes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                f"{val:.0f}s", ha="center", va="bottom", fontsize=8, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "runtime_comparison.png", dpi=300)
    plt.close(fig)
    print("  -> runtime_comparison.png")


def plot_latency():
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (label, path) in enumerate(ESTIMATOR_SOURCES.items()):
        df = load_csv(path)
        ebno, latency = get_metric(df, "latency_ms")
        if len(latency) > 0:
            ax.plot(ebno, latency, marker=MARKERS[i], color=COLORS[i],
                    label=label, linewidth=1.5, markersize=6)
    ax.set_xlabel("Eb/N0 (dB)", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Channel Estimators: Latency vs Eb/N0", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "estimator_latency_vs_ebno.png", dpi=300)
    plt.close(fig)
    print("  -> estimator_latency_vs_ebno.png")


def plot_throughput():
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (label, path) in enumerate(ESTIMATOR_SOURCES.items()):
        df = load_csv(path)
        ebno, tp = get_metric(df, "throughput_bits_per_batch")
        if len(tp) > 0:
            ax.plot(ebno, tp / 1000, marker=MARKERS[i], color=COLORS[i],
                    label=label, linewidth=1.5, markersize=6)
    ax.set_xlabel("Eb/N0 (dB)", fontsize=12)
    ax.set_ylabel("Throughput (kbits/batch)", fontsize=12)
    ax.set_title("Channel Estimators: Throughput vs Eb/N0", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "estimator_throughput_vs_ebno.png", dpi=300)
    plt.close(fig)
    print("  -> estimator_throughput_vs_ebno.png")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating report plots...")
    plot_estimator_ber()
    plot_jidd_comparison()
    plot_combined_ber()
    plot_runtime_comparison()
    plot_latency()
    plot_throughput()
    print("Done! All plots saved to", PLOTS_DIR)


if __name__ == "__main__":
    main()
