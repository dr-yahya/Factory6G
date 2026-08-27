"""Matplotlib style for PhD thesis figures (Sunway / IEEE, reproducible exports)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

# Printed at \linewidth (~145 mm). Aug 2026: larger labels for Ch1 engagement figures
# (supervisor: figure text too small); wrap earlier so callouts stay readable.
TABLE_FONT_PT = 15
FIG_TITLE_PT = 16
FIG_CALLOUT_PT = 14
# Per-line character caps for matplotlib annotations (Figure label line breaking).
FIG_LABEL_MAX_CHARS = 22
FIG_NOTE_MAX_CHARS = 40

_SEMANTIC_BREAKS = (" · ", " → ", "; ", " — ", " – ", ", ")


def _wrap_paragraph(paragraph: str, *, max_chars: int) -> list[str]:
    paragraph = paragraph.strip()
    if len(paragraph) <= max_chars:
        return [paragraph]
    for sep in _SEMANTIC_BREAKS:
        if sep not in paragraph:
            continue
        parts = paragraph.split(sep)
        lines: list[str] = []
        current = parts[0].strip()
        for part in parts[1:]:
            candidate = f"{current}{sep}{part}" if current else part.strip()
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = part.strip()
        if current:
            lines.append(current)
        if len(lines) > 1:
            return lines
    words = paragraph.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines if lines else [paragraph]


def wrap_figure_text(text: str, *, max_chars: int = FIG_LABEL_MAX_CHARS) -> str:
    """Return text with \\n breaks for matplotlib (Figure label line breaking)."""
    lines_out: list[str] = []
    for paragraph in text.replace("<br>", "\n").split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        lines_out.extend(_wrap_paragraph(paragraph, max_chars=max_chars))
    return "\n".join(lines_out)


# Thesis body figures: slightly under linewidth (~5.7 in) so fonts stay near body size in print.
THESIS_FIGSIZE = (5.8, 3.8)
THESIS_DPI = 300
EBNO_XLABEL = r"$E_b/N_0$ (dB)"

# Canonical legend order (classical → learned). Unknown keys sort after, alphabetically.
ESTIMATOR_ORDER: tuple[str, ...] = (
    "ls",
    "dft",
    "lmmse",
    "adaptive",
    "neural",
    "pso",
)
RESOURCE_MANAGER_ORDER: tuple[str, ...] = (
    "static",
    "round_robin",
    "max_throughput",
    "pf",
    "wmmse",
    "queue_aware_drl",
    "drl",
    "ber_drl",
    "cnn",
)

# Colorblind-friendly palette aligned with thesis Excalidraw accents.
PALETTE: tuple[str, ...] = (
    "#1e3a5f",
    "#3b82f6",
    "#047857",
    "#c2410c",
    "#6d28d9",
    "#b45309",
    "#0e7490",
    "#be123c",
    "#4b5563",
    "#84cc16",
)

MARKERS: tuple[str, ...] = ("o", "s", "D", "^", "v", "P", "X", "*", "+", "h")


def _canonical_order(stage_hint: str | None) -> tuple[str, ...]:
    if stage_hint in {"estimators", "estimator"}:
        return ESTIMATOR_ORDER
    if stage_hint in {"resource_managers", "resource_manager"}:
        return RESOURCE_MANAGER_ORDER
    merged = list(ESTIMATOR_ORDER) + [m for m in RESOURCE_MANAGER_ORDER if m not in ESTIMATOR_ORDER]
    return tuple(merged)


def order_method_names(names: Sequence[str], *, stage_hint: str | None = None) -> list[str]:
    """Return method names in stable thesis legend order."""
    canonical = _canonical_order(stage_hint)
    rank = {name: index for index, name in enumerate(canonical)}
    return sorted(names, key=lambda name: (rank.get(name, len(canonical)), name))


def order_methods_dict(
    methods: Mapping[str, Any],
    *,
    stage_hint: str | None = None,
) -> dict[str, Any]:
    ordered_names = order_method_names(list(methods.keys()), stage_hint=stage_hint)
    return {name: methods[name] for name in ordered_names}


def method_style_index(name: str, *, stage_hint: str | None = None) -> int:
    canonical = _canonical_order(stage_hint)
    if name in canonical:
        return canonical.index(name)
    return len(canonical) + hash(name) % 8


def method_color(name: str, *, stage_hint: str | None = None) -> str:
    index = method_style_index(name, stage_hint=stage_hint)
    return PALETTE[index % len(PALETTE)]


def method_marker(name: str, *, stage_hint: str | None = None) -> str:
    index = method_style_index(name, stage_hint=stage_hint)
    return MARKERS[index % len(MARKERS)]


def apply_thesis_rcparams(plt: Any) -> None:
    """Apply table-anchored thesis typography (Times New Roman) and consistent line weights."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": TABLE_FONT_PT,
            "axes.labelsize": FIG_TITLE_PT,
            "axes.titlesize": FIG_TITLE_PT,
            "legend.fontsize": TABLE_FONT_PT,
            "xtick.labelsize": TABLE_FONT_PT,
            "ytick.labelsize": TABLE_FONT_PT,
            "lines.linewidth": 1.8,
            "lines.markersize": 6,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "-",
            "figure.dpi": THESIS_DPI,
            "savefig.dpi": THESIS_DPI,
            "savefig.bbox": "tight",
        }
    )


def style_ebno_axis(ax: Any, *, ylabel: str, title: str | None = None) -> None:
    ax.set_xlabel(EBNO_XLABEL)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.35)
