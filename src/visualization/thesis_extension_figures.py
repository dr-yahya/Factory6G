"""Combined and summary thesis figures from stage_results_v2.json (multi-panel, Pareto, inventory)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.visualization.thesis_plot_style import (
    EBNO_XLABEL,
    FIG_CALLOUT_PT,
    FIG_TITLE_PT,
    THESIS_DPI,
    THESIS_FIGSIZE,
    apply_thesis_rcparams,
    method_color,
    method_marker,
    order_method_names,
    order_methods_dict,
    style_ebno_axis,
)
from src.visualization.thesis_summary_figures import (
    CANONICAL_SCHEDULERS,
    CHANNEL_LABELS,
    RM_LABELS,
    _ebno_index,
    _effective_ber,
    load_stage_json,
    plot_ch04_ber_heatmap,
)

SUNWAY_GREY = "#64748b"
POINT_STATUS_UPPER_BOUND_ONLY = "upper_bound_only"


def _coerce_float_array(values: list[Any]) -> np.ndarray:
    return np.asarray([float(v) for v in values], dtype=float)


def _point_status_array(metric_map: dict[str, list[Any]], *, fallback_len: int) -> np.ndarray:
    statuses = metric_map.get("point_status", [])
    if isinstance(statuses, list) and len(statuses) == fallback_len:
        return np.asarray([str(value) for value in statuses], dtype=object)
    return np.asarray(["resolved"] * fallback_len, dtype=object)


def _plot_ber_publication(
    *,
    plt: Any,
    methods: dict[str, dict[str, list[Any]]],
    ebno_range: list[float],
    title: str,
    output_path: Path,
    stage_hint: str | None = None,
) -> None:
    apply_thesis_rcparams(plt)
    fig, ax = plt.subplots(1, 1, figsize=THESIS_FIGSIZE)
    x = np.asarray(ebno_range, dtype=float)
    ordered = order_methods_dict(methods, stage_hint=stage_hint)
    for name, metric_map in ordered.items():
        if "ber" not in metric_map:
            continue
        ber = _coerce_float_array(metric_map["ber"])
        upper = _coerce_float_array(metric_map.get("ber_upper_confidence", metric_map["ber"]))
        statuses = _point_status_array(metric_map, fallback_len=ber.size)
        upper_mask = statuses == POINT_STATUS_UPPER_BOUND_ONLY
        valid_mask = ber > 0
        marker = method_marker(name, stage_hint=stage_hint)
        color = method_color(name, stage_hint=stage_hint)
        if np.any(valid_mask):
            ax.semilogy(
                x[valid_mask],
                ber[valid_mask],
                marker=marker,
                linewidth=1.8,
                color=color,
                label=name,
            )
        upper_show = upper_mask & ~valid_mask
        if np.any(upper_show):
            ax.semilogy(
                x[upper_show],
                upper[upper_show],
                marker=marker,
                markerfacecolor="none",
                linestyle="--",
                linewidth=1.5,
                color=color,
                label=name if not np.any(valid_mask) else "_nolegend_",
            )
    style_ebno_axis(ax, ylabel="BER", title=title)
    ax.legend()
    # Place outside axes so the note does not collide with y-tick labels.
    fig.text(
        0.5,
        0.01,
        "Dashed/open markers: 95% BER upper bound (zero observed errors)",
        ha="center",
        fontsize=FIG_CALLOUT_PT,
        alpha=0.85,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.savefig(output_path, dpi=THESIS_DPI)
    plt.close(fig)


def _plot_metric_vs_ebno(
    *,
    plt: Any,
    methods: dict[str, dict[str, list[Any]]],
    ebno_range: list[float],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    stage_hint: str | None = None,
) -> None:
    apply_thesis_rcparams(plt)
    fig, ax = plt.subplots(1, 1, figsize=THESIS_FIGSIZE)
    ordered = order_methods_dict(methods, stage_hint=stage_hint)
    for name, metric_map in ordered.items():
        if metric not in metric_map:
            continue
        values = _coerce_float_array(metric_map[metric])
        ax.plot(
            ebno_range,
            values,
            marker=method_marker(name, stage_hint=stage_hint),
            color=method_color(name, stage_hint=stage_hint),
            label=name,
            linewidth=1.8,
        )
    style_ebno_axis(ax, ylabel=ylabel, title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=THESIS_DPI)
    plt.close(fig)


def _filter_methods(
    methods: dict[str, dict[str, list[Any]]],
    methods_filter: Sequence[str] | None,
) -> dict[str, dict[str, list[Any]]]:
    if not methods_filter:
        return methods
    allowed = set(methods_filter)
    return {name: data for name, data in methods.items() if name in allowed}


def _plot_ber_on_axis(
    ax: Any,
    *,
    methods: dict[str, dict[str, list[Any]]],
    ebno_range: list[float],
    stage_hint: str | None,
    legend: bool = False,
) -> None:
    x = np.asarray(ebno_range, dtype=float)
    ordered = order_methods_dict(methods, stage_hint=stage_hint)
    for name, metric_map in ordered.items():
        if "ber" not in metric_map:
            continue
        ber = _coerce_float_array(metric_map["ber"])
        upper = _coerce_float_array(metric_map.get("ber_upper_confidence", metric_map["ber"]))
        statuses = metric_map.get("point_status", [])
        upper_mask = np.asarray(
            [str(s) == POINT_STATUS_UPPER_BOUND_ONLY for s in statuses],
            dtype=bool,
        ) if statuses else np.zeros_like(ber, dtype=bool)
        valid_mask = ber > 0
        marker = method_marker(name, stage_hint=stage_hint)
        color = method_color(name, stage_hint=stage_hint)
        label = name
        if np.any(valid_mask):
            ax.semilogy(
                x[valid_mask],
                ber[valid_mask],
                marker=marker,
                linewidth=1.8,
                color=color,
                label=label,
            )
        upper_show = upper_mask & ~valid_mask
        if np.any(upper_show):
            ax.semilogy(
                x[upper_show],
                upper[upper_show],
                marker=marker,
                markerfacecolor="none",
                linestyle="--",
                linewidth=1.5,
                color=color,
                label=label if not np.any(valid_mask) else "_nolegend_",
            )
    style_ebno_axis(ax, ylabel="BER")
    ax.grid(True, which="both", alpha=0.35)
    if legend:
        ax.legend(fontsize=FIG_CALLOUT_PT)


def _plot_metric_on_axis(
    ax: Any,
    *,
    methods: dict[str, dict[str, list[Any]]],
    ebno_range: list[float],
    metric: str,
    ylabel: str,
    stage_hint: str | None,
    legend: bool = False,
) -> None:
    ordered = order_methods_dict(methods, stage_hint=stage_hint)
    for name, metric_map in ordered.items():
        if metric not in metric_map:
            continue
        values = _coerce_float_array(metric_map[metric])
        ax.plot(
            ebno_range,
            values,
            marker=method_marker(name, stage_hint=stage_hint),
            color=method_color(name, stage_hint=stage_hint),
            label=name,
            linewidth=1.8,
        )
    style_ebno_axis(ax, ylabel=ylabel)
    if legend:
        ax.legend(fontsize=FIG_CALLOUT_PT, loc="best")


def plot_multi_panel_ber(
    *,
    panels: list[dict[str, str]],
    output_path: Path,
    supertitle: str | None = None,
    methods_filter: Sequence[str] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    apply_thesis_rcparams(plt)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 3.6), squeeze=False)
    axes_flat = axes.flatten()
    stage_hint = "estimators"

    for ax, panel in zip(axes_flat, panels):
        payload = load_stage_json(Path(panel["source"]))
        methods = _filter_methods(payload["methods"], methods_filter)
        _plot_ber_on_axis(
            ax,
            methods=methods,
            ebno_range=payload["ebno_db_range"],
            stage_hint=stage_hint,
            legend=(panel == panels[-1]),
        )
        ax.set_title(panel["title"], fontsize=FIG_TITLE_PT)

    if supertitle:
        fig.suptitle(supertitle, fontsize=FIG_TITLE_PT, y=1.02)
    fig.text(
        0.01,
        0.01,
        "dashed/open = 95% BER upper bound (zero observed errors)",
        fontsize=FIG_CALLOUT_PT,
        color=SUNWAY_GREY,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=THESIS_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_multi_panel_metric(
    *,
    panels: list[dict[str, str]],
    metric: str,
    ylabel: str,
    output_path: Path,
    supertitle: str | None = None,
    methods_filter: Sequence[str] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    apply_thesis_rcparams(plt)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 3.6), squeeze=False)
    axes_flat = axes.flatten()
    stage_hint = None
    if panels:
        first = load_stage_json(Path(panels[0]["source"]))
        stage_hint = first.get("stage")

    for ax, panel in zip(axes_flat, panels):
        payload = load_stage_json(Path(panel["source"]))
        methods = _filter_methods(payload["methods"], methods_filter)
        stage_hint = payload.get("stage")
        _plot_metric_on_axis(
            ax,
            methods=methods,
            ebno_range=payload["ebno_db_range"],
            metric=metric,
            ylabel=ylabel,
            stage_hint=stage_hint,
            legend=(panel == panels[-1]),
        )
        ax.set_title(panel["title"], fontsize=FIG_TITLE_PT)

    if supertitle:
        fig.suptitle(supertitle, fontsize=FIG_TITLE_PT, y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=THESIS_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_runtime_bar(
    *,
    payload: dict[str, Any],
    output_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    apply_thesis_rcparams(plt)
    runtime = payload.get("runtime_totals_sec", {})
    stage_hint = payload.get("stage")
    names = order_method_names(list(runtime.keys()), stage_hint=stage_hint)
    values = [float(runtime[n]) for n in names]
    colors = [method_color(n, stage_hint=stage_hint) for n in names]

    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE)
    ax.bar(names, values, color=colors)
    ax.set_ylabel("Total runtime (s)")
    ax.set_xlabel("Method")
    ax.set_title(title, fontsize=FIG_TITLE_PT)
    ax.grid(True, axis="y", alpha=0.35)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=THESIS_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_rm_pareto(
    *,
    run_b_dir: Path,
    ebno_db: float,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    apply_thesis_rcparams(plt)
    channels = ["rayleigh", "rician", "tr38901"]
    channel_markers = {"rayleigh": "o", "rician": "s", "tr38901": "D"}

    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE)
    for channel in channels:
        payload = load_stage_json(run_b_dir / channel / "resource_managers/stage_results_v2.json")
        idx = _ebno_index(payload, ebno_db)
        for scheduler in CANONICAL_SCHEDULERS:
            metric_map = payload["methods"][scheduler]
            ber, _status = _effective_ber(metric_map, idx)
            throughput = float(metric_map["throughput_bits_per_batch"][idx])
            ax.scatter(
                throughput,
                max(ber, 1e-12),
                color=method_color(scheduler, stage_hint="resource_managers"),
                marker=channel_markers[channel],
                s=55,
                edgecolors="black",
                linewidths=0.4,
                zorder=3,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Throughput (bits/batch)")
    ax.set_ylabel("BER")
    ax.set_title(
        rf"Throughput--reliability trade-off at $E_b/N_0 = {ebno_db:.0f}\,\mathrm{{dB}}$ (Run B)",
        fontsize=FIG_TITLE_PT,
    )
    ax.grid(True, which="both", alpha=0.35)

    channel_handles = [
        Line2D(
            [0],
            [0],
            marker=channel_markers[ch],
            color="0.2",
            linestyle="None",
            markersize=7,
            label=CHANNEL_LABELS[ch],
        )
        for ch in channels
    ]
    scheduler_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=method_color(sched, stage_hint="resource_managers"),
            linestyle="None",
            markersize=7,
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=RM_LABELS[sched],
        )
        for sched in CANONICAL_SCHEDULERS
    ]
    # Compact single legend below axes (column-friendly).
    all_handles = channel_handles + scheduler_handles
    ax.legend(
        handles=all_handles,
        fontsize=FIG_CALLOUT_PT,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=True,
        borderaxespad=0.0,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=THESIS_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def export_single_json_sweeps(
    *,
    json_path: Path,
    output_prefix: str,
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    payload = load_stage_json(json_path)
    stage = payload["stage"]
    ebno_range = payload["ebno_db_range"]
    methods = payload["methods"]

    _plot_ber_publication(
        plt=plt,
        methods=methods,
        ebno_range=ebno_range,
        title="BER vs Eb/No",
        output_path=out_dir / f"{output_prefix}_ber_vs_ebno.png",
        stage_hint=stage,
    )
    _plot_metric_vs_ebno(
        plt=plt,
        methods=methods,
        ebno_range=ebno_range,
        metric="throughput_bits_per_batch",
        ylabel="Throughput (bits/batch)",
        title="Throughput vs Eb/No",
        output_path=out_dir / f"{output_prefix}_throughput_vs_ebno.png",
        stage_hint=stage,
    )


def export_single_json_ber(
    *,
    json_path: Path,
    output_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    payload = load_stage_json(json_path)
    _plot_ber_publication(
        plt=plt,
        methods=payload["methods"],
        ebno_range=payload["ebno_db_range"],
        title=title,
        output_path=output_path,
        stage_hint=payload.get("stage"),
    )


def _format_ber_latex(ber: float, *, upper_bound: bool = False) -> str:
    if ber <= 0:
        body = "0"
    else:
        exponent = int(math.floor(math.log10(ber)))
        mantissa = ber / (10**exponent)
        body = f"{mantissa:.2f}\\times10^{{{exponent}}}"
    if upper_bound:
        return rf"$\leq {body}$"
    return f"${body}$"


def write_exploratory_anchor_table(
    *,
    rows: list[dict[str, str]],
    ebno_db: float,
    output_path: Path,
    note: str,
) -> None:
    lines = [
        "% Auto-generated from thesis/figure_manifest.json — do not edit by hand.",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Exploratory single-method channel-estimation campaigns at "
        f"$E_b/N_0 = {ebno_db:.0f}$\\,dB on TR~38.901 UMi QPSK (cross-run; not P1-shared).}}",
        "  \\label{tab:ch04-ce-exploratory}",
        "  \\small",
        "  \\begin{tabular}{@{}lccc@{}}",
        "    \\toprule",
        "    \\textbf{Estimator} & \\textbf{BER} & \\textbf{Status} & \\textbf{Run ID} \\\\",
        "    \\midrule",
    ]
    for row in rows:
        payload = load_stage_json(Path(row["source"]))
        method = row["method"]
        metric_map = payload["methods"][method]
        idx = _ebno_index(payload, ebno_db)
        ber, status = _effective_ber(metric_map, idx)
        status_tex = "Resolved" if status != POINT_STATUS_UPPER_BOUND_ONLY else "Upper-bound only"
        ber_tex = _format_ber_latex(ber, upper_bound=(status == POINT_STATUS_UPPER_BOUND_ONLY))
        run_id = str(payload.get("run_id", Path(row["source"]).parts[1][:15])).replace("_", "\\_")
        lines.append(f"    {row['label']} & {ber_tex} & {status_tex} & \\texttt{{{run_id}}} \\\\")
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "  \\vspace{0.35em}",
            "  \\footnotesize",
            f"  {note}",
            "\\end{table}",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


def scan_results_inventory(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.is_dir() or not run_dir.name[0].isdigit():
            continue
        jsons = list(run_dir.rglob("stage_results_v2.json"))
        if not jsons:
            rows.append(
                {
                    "run_dir": run_dir.name,
                    "status": "incomplete",
                    "stage_jsons": 0,
                    "detail": "no stage_results_v2.json",
                }
            )
            continue
        for jpath in sorted(jsons):
            rel = str(jpath.relative_to(run_dir))
            try:
                payload = load_stage_json(jpath)
            except json.JSONDecodeError:
                rows.append({"run_dir": run_dir.name, "status": "bad_json", "json": rel})
                continue
            methods = list(payload.get("methods", {}).keys())
            rows.append(
                {
                    "run_dir": run_dir.name,
                    "status": "complete",
                    "json": rel,
                    "stage": payload.get("stage"),
                    "run_id": payload.get("run_id"),
                    "methods": ", ".join(methods),
                    "n_ebno": len(payload.get("ebno_db_range", [])),
                }
            )
    return rows


def _tex_escape(text: str) -> str:
    return text.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def _tex_path(text: str) -> str:
    """Raw path text for \\tabpath / \\seqsplit (underscores stay literal)."""
    return text.replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


def write_run_inventory_tables(
    *,
    inventory: list[dict[str, Any]],
    output_path: Path,
    cleanup: dict[str, Any],
) -> None:
    lines = [
        "% Auto-generated from results/ scan — do not edit by hand.",
        "\\section{Local run directory inventory}",
        "\\label{app:run-inventory}",
        "",
        "Table~\\ref{tab:app-run-inventory} lists every timestamped directory under",
        "\\texttt{results/} at export time. Incomplete folders were removed after",
        "successful figure regeneration (Table~\\ref{tab:app-run-deletions}).",
        "",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Simulation run inventory (complete stage JSON artefacts).}",
        "  \\label{tab:app-run-inventory}",
        "  \\footnotesize",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\renewcommand{\\arraystretch}{1.15}",
        "  \\begin{tabularx}{\\linewidth}{@{}",
        "    >{\\raggedright\\arraybackslash}X",
        "    >{\\raggedright\\arraybackslash}p{1.55cm}",
        "    >{\\raggedright\\arraybackslash}p{2.35cm}",
        "    >{\\raggedright\\arraybackslash}X@{}}",
        "    \\toprule",
        "    \\textbf{Run directory} & \\textbf{Stage} & \\textbf{Methods} & \\textbf{JSON path} \\\\",
        "    \\midrule",
    ]
    for row in inventory:
        if row.get("status") != "complete":
            continue
        run_tex = _tex_path(row["run_dir"])
        methods_raw = row.get("methods", "")
        if len(methods_raw) > 42:
            methods_raw = methods_raw[:39] + "..."
        methods = _tex_path(methods_raw)
        json_path = _tex_path(row.get("json", ""))
        stage_tex = _tex_path(str(row.get("stage", "")))
        lines.append(
            f"    \\tabpath{{{run_tex}}} & \\tabpath{{{stage_tex}}} & "
            f"\\tabpath{{{methods}}} & \\tabpath{{{json_path}}} \\\\"
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabularx}",
            "\\end{table}",
            "",
            "\\begin{table}[htbp]",
            "  \\centering",
            "  \\caption{Run directories deleted after export (incomplete or superseded).}",
            "  \\label{tab:app-run-deletions}",
            "  \\footnotesize",
            "  \\setlength{\\tabcolsep}{4pt}",
            "  \\renewcommand{\\arraystretch}{1.15}",
            "  \\begin{tabularx}{\\linewidth}{@{}",
            "    >{\\raggedright\\arraybackslash}X",
            "    >{\\raggedright\\arraybackslash}X@{}}",
            "    \\toprule",
            "    \\textbf{Directory} & \\textbf{Reason} \\\\",
            "    \\midrule",
        ]
    )
    for path in cleanup.get("incomplete", []):
        name = _tex_path(Path(path).name)
        lines.append(f"    \\tabpath{{{name}}} & Incomplete (no stage JSON) \\\\")
    for item in cleanup.get("redundant", []):
        name = _tex_path(Path(item["path"]).name)
        reason = _tex_escape(item["reason"].replace("Run~B", "Run\\textasciitilde{}B"))
        lines.append(f"    \\tabpath{{{name}}} & {reason} \\\\")
    lines.extend(["    \\bottomrule", "  \\end{tabularx}", "\\end{table}", ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_path}")


__all__ = [
    "export_single_json_ber",
    "export_single_json_sweeps",
    "plot_ch04_ber_heatmap",
    "plot_multi_panel_ber",
    "plot_multi_panel_metric",
    "plot_rm_pareto",
    "plot_runtime_bar",
    "scan_results_inventory",
    "write_exploratory_anchor_table",
    "write_run_inventory_tables",
]
