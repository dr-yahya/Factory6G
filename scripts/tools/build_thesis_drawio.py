#!/usr/bin/env python3
"""Generate thesis architecture figures as draw.io XML (corporate + Sunway navy titles).

Connector routing: orthogonal edges with explicit waypoints in open corridors so
lines never pass over box fills.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from xml.etree.ElementTree import Element, SubElement, tostring

FIGDIR = Path(__file__).resolve().parents[2] / "thesis" / "figures"
SUNWAY_NAVY = "#233369"

# Canonical figure style (thesis/CONTEXT.md → Table typography anchor, Figure typographic tiers)
FONT_FAMILY = "Times New Roman"
FONT_TITLE = 12
FONT_SUBTITLE = 11
FONT_BOX = 11  # matches LaTeX \small table cells in 12pt report
FONT_NOTE = 10
FONT_EDGE = 10
FONT_BADGE = 11

CANVAS_LANDSCAPE = (1200, 750)
CANVAS_PORTRAIT = (900, 1050)

# Per-line character caps (Figure label line breaking). See thesis/CONTEXT.md.
LABEL_MAX_CHARS_BOX = 28
LABEL_MAX_CHARS_NARROW = 22
LABEL_MAX_CHARS_NOTE = 52
LABEL_MAX_CHARS_SUBTITLE = 78

_SEMANTIC_BREAKS = (" · ", " → ", "; ", " — ", " – ", ", ")

PALETTE = {
    "primary": ("#e3f2fd", "#1565c0"),
    "success": ("#e8f5e9", "#2e7d32"),
    "warning": ("#fff9c4", "#f57c00"),
    "accent": ("#fff3e0", "#e65100"),
    "secondary": ("#f3e5f5", "#6a1b9a"),
    "neutral": ("#eceff1", "#455a64"),
}

EDGE = (
    "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
    "html=1;endArrow=classic;endFill=1;strokeColor=#455a64;"
)
EDGE_DASHED = EDGE + "dashed=1;"


def _cell_value(text: str) -> str:
    """Draw.io multiline labels use &#xa; — not <br> (html.escape breaks <br>)."""
    normalized = text.replace("<br>", "\n")
    return "&#xa;".join(html.escape(line) for line in normalized.split("\n"))


def _wrap_paragraph(paragraph: str, *, max_chars: int) -> list[str]:
    """Split one paragraph into lines ≤ max_chars, preferring semantic boundaries."""
    paragraph = paragraph.strip()
    if len(paragraph) <= max_chars:
        return [paragraph]

    for sep in _SEMANTIC_BREAKS:
        if sep not in paragraph:
            continue
        parts = paragraph.split(sep)
        lines: list[str] = []
        current = parts[0].strip()
        sep_stripped = sep.strip()
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
    lines: list[str] = []
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


def normalize_label(text: str, *, max_chars: int = LABEL_MAX_CHARS_BOX) -> str:
    """Ensure multi-line labels; never leave long single-line cram (Figure label line breaking)."""
    if not text.strip():
        return text
    lines_out: list[str] = []
    for paragraph in text.replace("<br>", "\n").split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        lines_out.extend(_wrap_paragraph(paragraph, max_chars=max_chars))
    return "<br>".join(lines_out)


def estimate_box_dims(
    label: str,
    *,
    font_pt: int = FONT_BOX,
    min_w: float = 72,
    min_h: float = 32,
    h_pad: float = 14,
    w_pad: float = 16,
    char_width_factor: float = 0.58,
    line_height_factor: float = 1.4,
) -> tuple[float, float]:
    """Minimum shape size so wrapped text fits at locked tier (Figure shape text fitting)."""
    lines = label.replace("<br>", "\n").split("\n")
    max_len = max((len(line) for line in lines), default=1)
    n_lines = max(len(lines), 1)
    w = max(min_w, max_len * font_pt * char_width_factor + 2 * w_pad)
    h = max(min_h, n_lines * font_pt * line_height_factor + 2 * h_pad)
    return w, h


class DrawioBuilder:
    def __init__(self, page_w: int = CANVAS_LANDSCAPE[0], page_h: int = CANVAS_LANDSCAPE[1]) -> None:
        self.page_w = page_w
        self.page_h = page_h
        self._id = 2
        self.cells: list[tuple[str, dict[str, Any]]] = []
        self._geom: dict[str, tuple[float, float, float, float]] = {}

    def _nid(self) -> str:
        self._id += 1
        return str(self._id)

    def title(self, x: float, y: float, w: float, text: str) -> str:
        tid = self._nid()
        style = (
            f"text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=top;"
            f"fontFamily={FONT_FAMILY};fontSize={FONT_TITLE};fontStyle=1;fontColor={SUNWAY_NAVY};"
        )
        self.cells.append((tid, {"value": _cell_value(text), "style": style, "vertex": "1", "x": x, "y": y, "w": w, "h": 32}))
        return tid

    def subtitle(self, x: float, y: float, w: float, text: str) -> str:
        tid = self._nid()
        text = normalize_label(text, max_chars=LABEL_MAX_CHARS_SUBTITLE)
        n_lines = max(1, len(text.split("<br>")))
        style = f"text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=top;fontFamily={FONT_FAMILY};fontSize={FONT_SUBTITLE};fontColor=#64748b;"
        self.cells.append(
            (tid, {"value": _cell_value(text), "style": style, "vertex": "1", "x": x, "y": y, "w": w, "h": n_lines * 14 + 8})
        )
        return tid

    def note(self, x: float, y: float, w: float, h: float, text: str, role: str = "neutral") -> str:
        tid = self._nid()
        text = normalize_label(text, max_chars=LABEL_MAX_CHARS_NOTE)
        fill, stroke = PALETTE[role]
        style = (
            f"whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
            f"fontFamily={FONT_FAMILY};fontSize={FONT_NOTE};rounded=0;dashed=1;"
        )
        auto_w, auto_h = estimate_box_dims(text, font_pt=FONT_NOTE, min_w=w, min_h=h)
        w, h = max(w, auto_w), max(h, auto_h)
        self._geom[tid] = (x, y, w, h)
        self.cells.append((tid, {"value": _cell_value(text), "style": style, "vertex": "1", "x": x, "y": y, "w": w, "h": h}))
        return tid

    def step_badge(self, x: float, y: float, n: int) -> str:
        cid = self._nid()
        style = (
            f"ellipse;whiteSpace=wrap;html=1;fillColor={PALETTE['accent'][0]};"
            f"strokeColor={PALETTE['accent'][1]};fontFamily={FONT_FAMILY};fontSize={FONT_BADGE};fontStyle=1;"
        )
        self.cells.append((cid, {"value": str(n), "style": style, "vertex": "1", "step_badge": True, "x": x, "y": y, "w": 28, "h": 28}))
        return cid

    def step_badge_near_tl(self, node_id: str, n: int, gap: float = 6) -> str:
        """Numbered circle just outside top-left corner—close, no overlap with node fill."""
        x, y, _w, _h = self._geom[node_id]
        badge_w, badge_h = 28, 28
        bx = x - badge_w - gap
        by = y - badge_h - gap
        return self.step_badge(bx, by, n)

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        role: str = "primary",
        dashed: bool = False,
        ellipse: bool = False,
        *,
        fixed_width: bool = False,
    ) -> str:
        cid = self._nid()
        max_chars = LABEL_MAX_CHARS_NARROW if fixed_width else LABEL_MAX_CHARS_BOX
        label = normalize_label(label, max_chars=max_chars)
        fill, stroke = PALETTE[role]
        auto_w, auto_h = estimate_box_dims(label, font_pt=FONT_BOX, min_w=w, min_h=h)
        if fixed_width:
            h = max(h, auto_h)
        else:
            w, h = max(w, auto_w), max(h, auto_h)
        parts = [
            "whiteSpace=wrap",
            "html=1",
            f"fillColor={fill}",
            f"strokeColor={stroke}",
            f"fontFamily={FONT_FAMILY}",
            f"fontSize={FONT_BOX}",
            "rounded=0",
        ]
        if dashed:
            parts.append("dashed=1")
        if ellipse:
            parts.insert(0, "ellipse")
        style = ";".join(parts)
        self._geom[cid] = (x, y, w, h)
        self.cells.append((cid, {"value": _cell_value(label), "style": style, "vertex": "1", "x": x, "y": y, "w": w, "h": h}))
        return cid

    def center_text(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        color: str = "#64748b",
        font_pt: int = FONT_NOTE,
    ) -> str:
        tid = self._nid()
        text = normalize_label(text, max_chars=LABEL_MAX_CHARS_NARROW)
        style = (
            f"text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;"
            f"fontFamily={FONT_FAMILY};fontSize={font_pt};fontColor={color};"
        )
        self.cells.append((tid, {"value": _cell_value(text), "style": style, "vertex": "1", "x": x, "y": y, "w": w, "h": h}))
        return tid

    def group_rect(self, x: float, y: float, w: float, h: float) -> str:
        gid = self._nid()
        style = f"rounded=0;dashed=1;strokeColor={PALETTE['secondary'][1]};fillColor=none;"
        self.cells.append((gid, {"value": "", "style": style, "vertex": "1", "x": x, "y": y, "w": w, "h": h}))
        return gid

    def _center(self, cid: str) -> tuple[float, float]:
        x, y, w, h = self._geom[cid]
        return x + w / 2, y + h / 2

    def bottom(self, node_id: str) -> float:
        x, y, w, h = self._geom[node_id]
        return y + h

    def row_label(self, x: float, y: float, w: float, h: float, text: str) -> str:
        """Solid Sunway-navy row label (research-framework left column)."""
        cid = self._nid()
        text = normalize_label(text, max_chars=LABEL_MAX_CHARS_NARROW)
        style = (
            f"whiteSpace=wrap;html=1;fillColor={SUNWAY_NAVY};strokeColor={SUNWAY_NAVY};"
            f"fontFamily={FONT_FAMILY};fontSize={FONT_BOX};fontColor=#ffffff;fontStyle=1;rounded=0;"
        )
        auto_w, auto_h = estimate_box_dims(text, font_pt=FONT_BOX, min_w=w, min_h=h)
        h = max(h, auto_h)
        # Keep column width fixed; wrap label text instead of widening into content area.
        self._geom[cid] = (x, y, w, h)
        self.cells.append((cid, {"value": _cell_value(text), "style": style, "vertex": "1", "x": x, "y": y, "w": w, "h": h}))
        return cid

    def edge(
        self,
        src: str,
        dst: str,
        dashed: bool = False,
        color: str | None = None,
        points: list[tuple[float, float]] | None = None,
    ) -> None:
        eid = self._nid()
        style = EDGE_DASHED if dashed else EDGE
        if color:
            style = style.replace("strokeColor=#455a64", f"strokeColor={color}")
        self.cells.append(
            (
                eid,
                {
                    "value": "",
                    "style": style,
                    "edge": "1",
                    "source": src,
                    "target": dst,
                    "points": points or [],
                },
            )
        )

    def route(
        self,
        src: str,
        dst: str,
        waypoints: list[tuple[float, float]],
        dashed: bool = False,
        color: str | None = None,
    ) -> None:
        self.edge(src, dst, dashed=dashed, color=color, points=waypoints)

    def to_xml(self) -> str:
        mxfile = Element("mxfile", {"host": "app.diagrams.net", "agent": "build_thesis_drawio", "version": "24.7.17"})
        diagram = SubElement(mxfile, "diagram", {"id": "page1", "name": "Page-1"})
        model = SubElement(
            diagram,
            "mxGraphModel",
            {
                "dx": "1200",
                "dy": "800",
                "grid": "1",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": str(self.page_w),
                "pageHeight": str(self.page_h),
                "math": "0",
                "shadow": "0",
            },
        )
        root = SubElement(model, "root")
        SubElement(root, "mxCell", {"id": "0"})
        SubElement(root, "mxCell", {"id": "1", "parent": "0"})

        edges = [c for c in self.cells if c[1].get("edge")]
        verts = [c for c in self.cells if c[1].get("vertex")]
        # Z-order: connectors → grouping frames → labelled nodes → step badges (badges above connectors).
        bg_verts = [c for c in verts if not c[1].get("value")]
        fg_nodes = [c for c in verts if c[1].get("value") and not c[1].get("step_badge")]
        step_badges = [c for c in verts if c[1].get("step_badge")]

        for cid, data in edges + bg_verts + fg_nodes + step_badges:
            attrs: dict[str, str] = {"id": cid, "parent": "1", "style": data["style"]}
            if "value" in data:
                attrs["value"] = data["value"]
            if data.get("vertex"):
                attrs["vertex"] = "1"
            if data.get("edge"):
                attrs["edge"] = "1"
                attrs["source"] = data["source"]
                attrs["target"] = data["target"]
            cell = SubElement(root, "mxCell", attrs)
            if "x" in data:
                SubElement(
                    cell,
                    "mxGeometry",
                    {"x": str(data["x"]), "y": str(data["y"]), "width": str(data["w"]), "height": str(data["h"]), "as": "geometry"},
                )
            elif data.get("edge"):
                geo = SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
                pts = data.get("points") or []
                if pts:
                    arr = SubElement(geo, "Array", {"as": "points"})
                    for px, py in pts:
                        SubElement(arr, "mxPoint", {"x": str(px), "y": str(py)})

        return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" + tostring(mxfile, encoding="unicode")


def build_integrated_pipeline() -> str:
    """Landscape canvas; horizontal two-row integrated PHY/MAC pipeline."""
    pw, ph = CANVAS_LANDSCAPE
    d = DrawioBuilder(page_w=pw, page_h=ph)
    mx = 48
    cw = pw - 2 * mx
    row_gap = 28
    col_gap = 18
    n_cols = 5
    bw = (cw - (n_cols - 1) * col_gap) // n_cols

    d.title(mx, 16, cw, "Integrated ML PHY/MAC pipeline")
    d.subtitle(
        mx,
        50,
        cw,
        "Joint integrated evaluation — estimation error propagates into MAC scheduling on shared factory geometry.",
    )

    y1 = 88
    xs = [mx + i * (bw + col_gap) for i in range(n_cols)]

    tx = d.box(xs[0], y1, bw, 72, "OFDM transmitter<br>5G LDPC · QAM · resource grid<br>pilots @ sym 2, 11", "accent")
    ch = d.box(xs[1], y1, bw, 72, "Factory channel model<br>Rayleigh · Rician · TR 38.901 UMi<br>profile s/m/l/apple", "primary")
    rx = d.box(xs[2], y1, bw, 72, "Receiver front-end<br>FFT · pilot extraction<br>per-UT batch tensors", "primary")
    ce = d.box(xs[3], y1, bw, 72, "Channel estimation<br>LS · DFT · LMMSE · adaptive · neural<br>pluggable receiver estimators", "secondary")
    phy = d.box(xs[4], y1, bw, 72, "PHY decode chain<br>LMMSE equalise · demap · LDPC<br>batch BER accounting", "primary")
    row1_h = max(d.bottom(n) for n in (tx, ch, rx, ce, phy))

    couple_y = row1_h + 12
    d.subtitle(xs[3], couple_y, 2 * bw + col_gap, "Integrated ML coupling (PHY estimate → MAC policy)")
    y2 = couple_y + 34

    mc = d.box(xs[0], y2, bw, 72, "Monte Carlo harness<br>Eb/N0 sweep 0–20 dB<br>shared channel context (P1)", "accent")
    met = d.box(xs[1], y2, bw, 72, "End-to-end metrics<br>BER · throughput · latency<br>power · runtime", "success")
    rd = d.box(xs[2], y2, bw, 72, "MAC resource directives<br>per-UT mask · power scale<br>MAC → PHY coupling", "neutral", dashed=True)
    cqs = d.box(xs[3], y2, bw, 72, "Channel-quality state<br>H_hat · err_var · Eb/N0<br>feedback to scheduler", "primary")
    rmm = d.box(xs[4], y2, bw, 72, "Resource manager<br>PF · WMMSE · queue-aware DRL<br>BER-aware DRL policies", "secondary")
    row2_h = max(d.bottom(n) for n in (mc, met, rd, cqs, rmm))

    d.group_rect(xs[3] - 10, y2 - 12, 2 * bw + col_gap + 20, row2_h - y2 + 22)

    for a, b in [(tx, ch), (ch, rx), (rx, ce), (ce, phy), (met, mc), (cqs, rmm), (rmm, rd)]:
        d.edge(a, b)

    margin_r = xs[4] + bw + 36
    margin_l = mx - 12
    d.route(phy, met, [(margin_r, d._center(phy)[1]), (margin_r, d._center(met)[1])], color=PALETTE["success"][1])
    d.edge(mc, tx, color=PALETTE["accent"][1])
    d.edge(ce, cqs, color=PALETTE["secondary"][1])
    d.route(rd, tx, [(margin_l, d._center(rd)[1]), (margin_l, d._center(tx)[1])], dashed=True, color=PALETTE["accent"][1])

    ny = row2_h + row_gap
    d.note(mx, ny, cw // 2 - 8, 56, "Fixed-order stages (P3): estimator benchmark → RM on estimated CSI<br>comparative metrics per operating point", "neutral")
    d.note(mx + cw // 2 + 8, ny, cw // 2 - 8, 48, "Decoupled baseline: CE curves without MAC impact (not primary narrative)", "warning")
    return d.to_xml()


def build_lr_taxonomy() -> str:
    """Portrait canvas; vertical top-down tree with CE/RM pillar branches."""
    pw, ph = CANVAS_PORTRAIT
    d = DrawioBuilder(page_w=pw, page_h=ph)
    mx = 48
    cw = pw - 2 * mx
    row_gap = 12
    col_gap = 24
    col_w = (cw - col_gap) // 2
    lx, rx = mx, mx + col_w + col_gap

    def bottom(node_id: str) -> float:
        x, y, w, h = d._geom[node_id]
        return y + h

    d.title(mx, 16, cw, "Smart-factory wireless literature map")
    d.subtitle(
        mx,
        50,
        cw,
        "Chapter 2 — classical vs ML families for CE and RM; decoupled vs integrated PHY–MAC evaluation.",
    )

    y = 88
    req = d.box(mx, y, cw, 52, "Industrial requirements<br>URLLC · latency · mobility · heterogeneous UTs", "accent")
    y = bottom(req) + row_gap

    root = d.box(
        mx,
        y,
        cw,
        60,
        "6G / B5G smart-factory wireless<br>industrial IoT · AGV/robotics · factory RF · network slicing",
        "primary",
    )
    y = bottom(root) + row_gap

    met = d.box(mx, y, cw, 52, "Evaluation metrics<br>BER · throughput · latency · power · fairness", "success")
    y = bottom(met) + row_gap

    dec = d.box(
        mx,
        y,
        cw,
        48,
        "Typical literature: decoupled CE/RM benchmarks<br>CE curves without MAC impact · perfect-CSI schedulers",
        "neutral",
        dashed=True,
    )
    y = bottom(dec) + row_gap + 8

    d.subtitle(lx, y, col_w, "Channel estimation branch")
    d.subtitle(rx, y, col_w, "Resource management branch")
    y += 30

    ce_cls = d.box(
        lx,
        y,
        col_w,
        54,
        "Classical channel estimation<br>LS · DFT · LMMSE · MMSE<br>pilot-aided OFDM · Sionna baselines",
        "primary",
    )
    rm_cls = d.box(
        rx,
        y,
        col_w,
        54,
        "Classical resource management<br>PF · WMMSE · max-throughput<br>static · round-robin",
        "primary",
    )
    y = max(bottom(ce_cls), bottom(rm_cls)) + row_gap

    ce_ml = d.box(
        lx,
        y,
        col_w,
        54,
        "ML channel estimation<br>neural CE · adaptive receivers<br>physics-informed / learned pilots",
        "secondary",
    )
    rm_ml = d.box(
        rx,
        y,
        col_w,
        54,
        "ML resource management<br>DRL · CNN schedulers<br>queue-aware · cross-layer RL",
        "secondary",
    )
    y = max(bottom(ce_ml), bottom(rm_ml)) + row_gap

    ce_ctx = d.box(lx, y, col_w, 48, "Factory channel context<br>Rayleigh · Rician · TR 38.901", "accent")
    rm_ctx = d.box(
        rx,
        y,
        col_w,
        48,
        "MAC objectives<br>scheduling · power control<br>URLLC QoS · multi-UT fairness",
        "warning",
    )
    y = max(bottom(ce_ctx), bottom(rm_ctx)) + row_gap + 16

    gap = d.box(
        mx,
        y,
        cw,
        72,
        "RESEARCH GAP (this thesis)<br>Integrated ML PHY–MAC evaluation on shared channel realisations<br>"
        "end-to-end under factory geometry · staged coupled benchmarking (P1–P3)",
        "secondary",
    )
    y = bottom(gap) + row_gap + 12

    leg_h = 34
    leg_w = 118
    leg_gap = 14
    d.box(mx, y, leg_w, leg_h, "Classical", "primary")
    d.box(mx + leg_w + leg_gap, y, leg_w, leg_h, "ML / DRL", "secondary")
    d.box(mx + 2 * (leg_w + leg_gap), y, leg_w + 24, leg_h, "Factory context", "accent")

    d.edge(req, root)
    d.edge(root, met)
    d.edge(met, dec)

    branch_y = d._geom[ce_cls][1] - 12
    dec_cx, _ = d._center(dec)
    ce_cx, _ = d._center(ce_cls)
    rm_cx, _ = d._center(rm_cls)
    d.route(dec, ce_cls, [(dec_cx, branch_y), (ce_cx, branch_y)])
    d.route(dec, rm_cls, [(dec_cx, branch_y), (rm_cx, branch_y)])

    for a, b in [(ce_cls, ce_ml), (ce_ml, ce_ctx), (rm_cls, rm_ml), (rm_ml, rm_ctx)]:
        d.edge(a, b)

    merge_y = d._geom[gap][1] - 14
    gap_cx, _ = d._center(gap)
    ctx_ce_x, _ = d._center(ce_ctx)
    ctx_rm_x, _ = d._center(rm_ctx)
    d.route(ce_ctx, gap, [(ctx_ce_x, merge_y), (gap_cx, merge_y)])
    d.route(rm_ctx, gap, [(ctx_rm_x, merge_y), (gap_cx, merge_y)])

    return d.to_xml()


def build_phy_stack() -> str:
    """Portrait canvas; vertical PHY layer stack top-to-bottom."""
    pw, ph = CANVAS_PORTRAIT
    d = DrawioBuilder(page_w=pw, page_h=ph)
    mx = 48
    cw = pw - 2 * mx
    row_gap = 12
    stack_w = 500
    bx = mx + (cw - stack_w) // 2

    def bottom(node_id: str) -> float:
        x, y, w, h = d._geom[node_id]
        return y + h

    d.title(mx, 16, cw, "Physical-layer stack")
    d.subtitle(mx, 50, cw, "LDPC–QAM–OFDM batch chain; BER metric propagates to Chapter 4 results.")

    pre_rx: list[tuple[str, str]] = [
        ("Info bits · per batch", "accent"),
        ("5G LDPC encode (n=648)", "primary"),
        ("QAM mapping · QPSK canonical", "primary"),
        ("OFDM resource grid · pilots @ sym 2,11 · FFT 128 · 4 UT", "primary"),
        ("Channel · Rayleigh/Rician · TR 38.901 UMi + AWGN", "primary"),
    ]
    post_rx: list[tuple[str, str]] = [
        ("Estimator plugin · LS/DFT/LMMSE/adaptive/neural", "secondary"),
        ("LMMSE equalise using H_hat", "neutral"),
        ("Demap + LDPC decode · CRC", "primary"),
        ("BER metrics · bit/block errors per batch", "success"),
    ]

    y = 88
    ids: list[str] = []
    for label, role in pre_rx:
        html_label = label.replace(" · ", "<br>")
        ids.append(d.box(bx, y, stack_w, 48, html_label, role))
        y = bottom(ids[-1]) + row_gap

    subtitle_y = y + 6
    d.subtitle(bx, subtitle_y, stack_w, "Receiver chain (estimate → equalise → decode → metrics)")
    y = subtitle_y + 30
    rx_top = y

    for label, role in post_rx:
        html_label = label.replace(" · ", "<br>")
        ids.append(d.box(bx, y, stack_w, 48, html_label, role))
        y = bottom(ids[-1]) + row_gap

    d.group_rect(bx - 12, rx_top - 10, stack_w + 24, bottom(ids[-1]) - rx_top + 18)

    grid = ids[3]
    grid_x, grid_y, grid_w, grid_h = d._geom[grid]
    rm_w = cw - (bx + stack_w + 20) - mx
    rm = d.box(
        bx + stack_w + 20,
        grid_y,
        max(rm_w, 160),
        grid_h,
        "MAC resource directives<br>mask · power scale<br>from scheduler stage",
        "warning",
        dashed=True,
    )

    for a, b in zip(ids[:-1], ids[1:]):
        d.edge(a, b)
    d.edge(grid, rm, dashed=True, color=PALETTE["warning"][1])

    d.note(mx, bottom(ids[-1]) + row_gap + 8, cw, 48, "Estimator plugins — identical channel realisation per method (P1)", "neutral")
    return d.to_xml()


def build_monte_carlo() -> str:
    """Portrait canvas; vertical trunk with horizontal parallel-stage branches."""
    pw, ph = CANVAS_PORTRAIT
    d = DrawioBuilder(page_w=pw, page_h=ph)
    mx = 48
    cw = pw - 2 * mx
    row_gap = 12
    col_gap = 16
    tw = min(300, cw - 80)
    tx = (pw - tw) // 2

    d.title(mx, 16, cw, "Monte Carlo orchestration flow")
    d.subtitle(mx, 50, cw, "Integrated evaluation design — shared channel context (P1); stopping policy gates each Eb/N0 point.")

    y = 88
    start = d.box(tx, y, tw, 46, "Monte Carlo entry<br>reproducible environment", "success", ellipse=True)
    d.step_badge_near_tl(start, 1)
    y = d.bottom(start) + row_gap

    cfg = d.box(tx, y, tw, 50, "Configuration and seeds<br>GPU/CPU · output directory", "neutral")
    d.step_badge_near_tl(cfg, 2)
    y = d.bottom(cfg) + row_gap

    ebn0 = d.box(tx - 12, y, tw + 24, 54, "Eb/N0 sweep 0…20 dB (step 2)<br>outer Monte Carlo loop", "accent")
    d.step_badge_near_tl(ebn0, 3)
    y = d.bottom(ebn0) + row_gap

    batch = d.box(tx - 8, y, tw + 16, 54, "Prepare shared channel context<br>channel · noise · source bits", "primary")
    d.step_badge_near_tl(batch, 4)
    y = d.bottom(batch) + row_gap + 16

    col_w = (cw - 2 * col_gap) // 3
    branch_y = y
    est = d.box(mx, branch_y, col_w, 62, "Estimator stage (fixed order)<br>all methods · same context<br>BER curves per method", "secondary")
    d.step_badge_near_tl(est, 5)
    rm = d.box(mx + col_w + col_gap, branch_y, col_w, 62, "RM stage<br>h_hat → directives<br>throughput · latency · power", "warning")
    d.step_badge_near_tl(rm, 6)
    stop = d.box(mx + 2 * (col_w + col_gap), branch_y, col_w, 62, "Stopping policy<br>min/max batches · target blocks<br>≥30 bit errors → resolved", "success")
    d.step_badge_near_tl(stop, 7)
    y = max(d.bottom(est), d.bottom(rm), d.bottom(stop)) + row_gap + 24

    out = d.box(tx - 8, y, tw + 16, 54, "Structured result artefacts<br>metrics tables · thesis figures", "neutral")
    d.step_badge_near_tl(out, 8)
    y = d.bottom(out) + row_gap

    end = d.box(tx + 12, y, tw - 24, 46, "Experiment complete<br>locked run IDs · Ch.4", "success", ellipse=True)

    d.edge(start, cfg)
    d.edge(cfg, ebn0)
    d.edge(ebn0, batch)

    batch_cx, _ = d._center(batch)
    split_y = branch_y - 12
    for node in (est, rm, stop):
        nx, _ = d._center(node)
        d.route(batch, node, [(batch_cx, split_y), (nx, split_y)])

    for node in (est, rm, stop):
        d.edge(node, out)

    d.edge(out, end, color=PALETTE["success"][1])

    bus_r = pw - 40
    ebn0_cy = d._center(ebn0)[1]
    stop_cy = d._center(stop)[1]
    d.route(stop, ebn0, [(bus_r, stop_cy), (bus_r, ebn0_cy)], dashed=True, color=PALETTE["accent"][1])

    ny = d.bottom(end) + row_gap + 8
    d.note(mx, ny, cw, 48, "Point status: resolved (≥30 bit errors) · upper_bound_only (dashed Ch.4 markers)", "neutral")
    d.note(mx, ny + 58, cw, 48, "Stopping defaults: batch 20 · min 10/max 20 batches · target 100 block errors · 1M bits (Appendix A)", "accent")
    return d.to_xml()


def build_factory_topology() -> str:
    """Landscape canvas; spatial factory floor grouping."""
    pw, ph = CANVAS_LANDSCAPE
    d = DrawioBuilder(page_w=pw, page_h=ph)
    mx = 48
    cw = pw - 2 * mx
    row_gap = 14
    col_gap = 24

    d.title(mx, 16, cw, "Factory deployment topology")
    d.subtitle(
        mx,
        50,
        cw,
        "Plan-view geometry — size presets scale room, machines, and UT count; couples to Rayleigh/Rician/TR 38.901 models.",
    )

    sidebar_w = 176
    right_w = 196
    room_x = mx + sidebar_w + col_gap
    room_y = 96
    room_w = pw - room_x - mx - right_w - col_gap
    room_h = 400
    pad = 20
    elem_gap = 16

    # --- Left sidebar: even vertical stack ---
    sy = room_y
    size_box = d.box(
        mx,
        sy,
        sidebar_w,
        48,
        "Size presets<br>s: 15×15 m · 5 machines · 4 UT<br>m: 25×25 m · 10 · 8 UT<br>l: 40×40 m · 20 · 16 UT<br>apple: 60×35 m · 22 · 8 UT",
        "neutral",
    )
    mat_box = d.box(
        mx,
        d.bottom(size_box) + row_gap,
        sidebar_w,
        48,
        "Materials (ray trace)<br>Metal (high σ) · Concrete (εr≈7)<br>blocking / scattering",
        "neutral",
    )
    d.box(
        mx,
        d.bottom(mat_box) + row_gap,
        sidebar_w,
        48,
        "Mobility<br>static (v=0) · AGV (v>0)",
        "accent",
    )

    # --- Factory hall ---
    d.group_rect(room_x, room_y, room_w, room_h)
    d.subtitle(
        room_x + 10,
        room_y + 8,
        room_w - 20,
        "Factory hall — profile s (15×15×5 m) · perception / communication layer",
    )

    bs_label = "Base station (BS)<br>8 antennas · 3.5 GHz<br>TR 38.901 antenna"
    ut_label = "UT 1<br>1 antenna · QPSK"
    mach_label = "Machine<br>metal σ"
    bs_w, bs_h = estimate_box_dims(normalize_label(bs_label))
    ut_w, ut_h = estimate_box_dims(normalize_label(ut_label))
    mach_w, mach_h = estimate_box_dims(normalize_label(mach_label))

    inner_left = room_x + pad
    inner_right = room_x + room_w - pad
    inner_top = room_y + 30
    uplink_y = room_y + room_h - 26
    inner_bottom = uplink_y - elem_gap

    bs_x = room_x + (room_w - bs_w) / 2
    bs_y = inner_top
    bs = d.box(bs_x, bs_y, bs_w, bs_h, bs_label, "primary")

    ut_y_top = d.bottom(bs) + elem_gap + 8
    ut_y_bot = inner_bottom - ut_h
    ut_x_left = inner_left
    ut_x_right = inner_right - ut_w

    ut_ids = [
        d.box(ut_x_left, ut_y_bot, ut_w, ut_h, "UT 1<br>1 antenna · QPSK", "success"),
        d.box(ut_x_right, ut_y_bot, ut_w, ut_h, "UT 2<br>1 antenna · QPSK", "success"),
        d.box(ut_x_left, ut_y_top, ut_w, ut_h, "UT 3<br>1 antenna · QPSK", "success"),
        d.box(ut_x_right, ut_y_top, ut_w, ut_h, "UT 4<br>1 antenna · QPSK", "success"),
    ]

    # Five machines in one evenly spaced row between the UT tiers.
    mach_zone_left = ut_x_left + ut_w + elem_gap
    mach_zone_right = ut_x_right - elem_gap
    mach_y = ut_y_top + ut_h + max(elem_gap, (ut_y_bot - (ut_y_top + ut_h) - mach_h) / 2)

    def _even_row_x(n: int, left: float, right: float, box_w: float, gap: float) -> list[float]:
        avail = right - left
        if n == 1:
            return [left + (avail - box_w) / 2]
        total = n * box_w + (n - 1) * gap
        if total > avail:
            gap = max(6.0, (avail - n * box_w) / (n - 1))
            total = n * box_w + (n - 1) * gap
        start = left + max(0.0, (avail - total) / 2)
        return [start + i * (box_w + gap) for i in range(n)]

    machine_xs = _even_row_x(5, mach_zone_left, mach_zone_right, mach_w, elem_gap)
    for mx_m in machine_xs:
        d.box(mx_m, mach_y, mach_w, mach_h, mach_label, "accent")

    d.subtitle(
        room_x + pad,
        uplink_y,
        room_w - 2 * pad,
        "Uplink: 4 UT → BS · Kronecker pilots · num_ut | FFT(128)",
    )

    # --- Right column: ray tracing + channel model ---
    chan_x = room_x + room_w + col_gap
    ray = d.box(
        chan_x,
        room_y,
        right_w,
        48,
        "Ray tracing (optional)<br>geometry-driven CIR<br>factory scene model",
        "secondary",
    )
    chan = d.box(
        chan_x,
        d.bottom(ray) + row_gap,
        right_w,
        48,
        "Channel model<br>Rayleigh / Rician<br>TR 38.901 UMi · AWGN",
        "primary",
    )
    d.edge(ray, chan, color=PALETTE["secondary"][1])

    # --- Connectors in open corridors (right of hall) ---
    margin_r = room_x + room_w + elem_gap
    bs_cx, bs_cy = d._center(bs)
    for ut in ut_ids:
        ut_cx, ut_cy = d._center(ut)
        d.route(ut, bs, [(margin_r, ut_cy), (margin_r, bs_cy)], dashed=True, color=PALETTE["primary"][1])
    chan_cy = d._center(chan)[1]
    d.route(bs, chan, [(margin_r, bs_cy), (margin_r, chan_cy)], color=PALETTE["primary"][1])

    # --- Application layer below hall ---
    app_label = "Application layer<br>scheduling · QoS"
    app_w, app_h = estimate_box_dims(normalize_label(app_label))
    app = d.box(room_x + (room_w - app_w) / 2, room_y + room_h + row_gap, app_w, app_h, app_label, "warning")
    app_cy = d._center(app)[1]
    d.route(
        app,
        chan,
        [(chan_x - elem_gap, app_cy), (chan_x - elem_gap, chan_cy)],
        dashed=True,
        color=PALETTE["warning"][1],
    )

    ny = d.bottom(app) + row_gap
    note1_text = normalize_label(
        "Canonical Ch4: profile s · QPSK · Rayleigh (estimator + RM). RM sweep adds Rician + TR 38.901 UMi.",
        max_chars=LABEL_MAX_CHARS_NOTE,
    )
    note1 = d.note(room_x, ny, room_w, 40, note1_text, "neutral")
    note2_text = normalize_label(
        "Geometry and antenna parameters recorded per locked experiment (Appendix A).",
        max_chars=LABEL_MAX_CHARS_NOTE,
    )
    d.note(mx, d.bottom(note1) + row_gap, cw, 32, note2_text, "neutral")
    return d.to_xml()


def build_estimator_comparison() -> str:
    """Landscape canvas; columnar comparison of five canonical estimators (no step badges)."""
    pw, ph = CANVAS_LANDSCAPE
    d = DrawioBuilder(page_w=pw, page_h=ph)
    mx = 48
    cw = pw - 2 * mx
    row_gap = 8
    col_gap = 8
    label_w = 112
    n_cols = 5
    col_w = (cw - label_w - col_gap - n_cols * col_gap) // n_cols

    d.title(mx, 16, cw, "Channel estimator comparison")
    d.subtitle(
        mx,
        50,
        cw,
        "Ch.3 conceptual map — five canonical estimators on shared channel realisations (P1); BER curves in Ch.4.",
    )

    hx = mx + label_w + col_gap
    y = 88

    headers: list[tuple[str, str]] = [
        ("LS", "primary"),
        ("DFT", "primary"),
        ("LMMSE", "primary"),
        ("Adaptive", "secondary"),
        ("Neural", "secondary"),
    ]

    header_ids = [d.box(mx, y, label_w, 48, "Dimension", "neutral", fixed_width=True)]
    for i, (name, role) in enumerate(headers):
        header_ids.append(d.box(hx + i * (col_w + col_gap), y, col_w, 48, name, role, fixed_width=True))
    y = max(d.bottom(n) for n in header_ids) + row_gap

    rows: list[tuple[str, list[str]]] = [
        (
            "Input",
            [
                "Pilot obs. Y,<br>noise no",
                "h_ls, err_var<br>from LS",
                "h_ls, err_var<br>from LS",
                "h_ls + SNR<br>quality proxy",
                "h_ls + normalised<br>Eb/N0 map",
            ],
        ),
        (
            "Mechanism",
            [
                "Least-squares<br>on pilots",
                "Delay-domain<br>tap truncation",
                "Freq-domain<br>LMMSE shrinkage",
                "Branch: DFT /<br>blend / LMMSE",
                "Conv2D residual<br>Δ on h_ls",
            ],
        ),
        (
            "Assumption",
            [
                "AWGN on<br>pilot REs",
                "Sparse CIR<br>(≤ CP length)",
                "Exp. freq.<br>correlation R_freq",
                "Fixed quality<br>thresholds",
                "Train/eval same<br>PHY statistics",
            ],
        ),
        (
            "Training",
            [
                "None",
                "None",
                "None<br>(R_freq hyperparam)",
                "None<br>(rule-based branches)",
                "Offline · synthetic<br>factory data",
            ],
        ),
        (
            "Inference cost",
            [
                "Low ·<br>O(pilot REs)",
                "Medium ·<br>FFT per batch",
                "Medium · FFT +<br>eig. cache",
                "Medium–high ·<br>up to 3 branches",
                "High · GPU<br>forward pass",
            ],
        ),
        (
            "PHY placement",
            [
                "Estimator<br>slot",
                "Same receiver<br>chain",
                "Same receiver<br>chain",
                "Same receiver<br>chain",
                "Same → LMMSE<br>equaliser",
            ],
        ),
    ]

    for row_label, cells in rows:
        row_ids = [d.box(mx, y, label_w, 48, row_label, "neutral", fixed_width=True)]
        for i, text in enumerate(cells):
            row_ids.append(d.box(hx + i * (col_w + col_gap), y, col_w, 48, text, "neutral", fixed_width=True))
        y = max(d.bottom(n) for n in row_ids) + row_gap

    d.note(
        mx,
        y + 10,
        cw,
        40,
        "All estimators evaluated on identical shared channel contexts per batch (P1) · PSO omitted from canonical Ch.4 run.",
        "accent",
    )
    return d.to_xml()


def build_rm_comparison() -> str:
    """Landscape canvas; columnar comparison of eight canonical resource managers (no step badges)."""
    pw, ph = CANVAS_LANDSCAPE
    d = DrawioBuilder(page_w=pw, page_h=ph)
    mx = 48
    cw = pw - 2 * mx
    row_gap = 6
    col_gap = 5
    label_w = 88
    n_cols = 8
    col_w = (cw - label_w - col_gap - (n_cols - 1) * col_gap) // n_cols

    d.title(mx, 16, cw, "Resource manager comparison")
    d.subtitle(
        mx,
        50,
        cw,
        "Ch.3 conceptual map — eight canonical schedulers on shared channel-quality feedback; Ch.4 plots hold BER/throughput curves.",
    )

    hx = mx + label_w + col_gap
    y = 88

    headers: list[tuple[str, str]] = [
        ("Static", "primary"),
        ("Round-robin", "primary"),
        ("Max-T", "primary"),
        ("PF", "primary"),
        ("WMMSE", "primary"),
        ("Q-aware", "secondary"),
        ("DRL", "secondary"),
        ("BER-DRL", "secondary"),
    ]

    header_ids = [d.box(mx, y, label_w, 44, "Dimension", "neutral", fixed_width=True)]
    for i, (name, role) in enumerate(headers):
        header_ids.append(d.box(hx + i * (col_w + col_gap), y, col_w, 44, name, role, fixed_width=True))
    y = max(d.bottom(n) for n in header_ids) + row_gap

    rows: list[tuple[str, list[str]]] = [
        (
            "State input",
            [
                "Config mask/<br>power only",
                "Batch index<br>(no CSI)",
                "h_hat,<br>Eb/N0",
                "h_hat, Eb/N0 +<br>PF history",
                "h_hat<br>Gram matrix",
                "h_hat + synthetic<br>queue",
                "h_hat, err_var,<br>Eb/N0",
                "h_hat, err_var,<br>Eb/N0",
            ],
        ),
        (
            "Policy type",
            [
                "Fixed<br>directives",
                "Cyclic UT<br>rotation",
                "Greedy rate<br>maximisation",
                "Proportional-fair<br>metric",
                "Weighted MMSE<br>power ctrl",
                "Lyapunov drift +<br>penalty",
                "Learned<br>RL policy",
                "BER-weighted<br>RL policy",
            ],
        ),
        (
            "Objective",
            [
                "Baseline<br>schedule",
                "Fair<br>time-sharing",
                "Sum-rate /<br>throughput",
                "Throughput–<br>fairness trade-off",
                "Interference-<br>aware SINR",
                "Queue stability<br>+ rate",
                "Throughput ·<br>latency",
                "Reliability-<br>first BER",
            ],
        ),
        (
            "Training",
            [
                "None",
                "None",
                "None",
                "None<br>(history state)",
                "None<br>(iterative WMMSE)",
                "None<br>(virtual queues)",
                "Offline RL<br>checkpoint",
                "Offline RL ·<br>BER reward",
            ],
        ),
        (
            "Inference cost",
            [
                "O(1)",
                "O(1)",
                "O(num_ut)",
                "O(num_ut)",
                "O(num_ut²)<br>WMMSE iters",
                "O(num_ut)<br>queue update",
                "Policy network<br>forward",
                "Policy network<br>forward",
            ],
        ),
        (
            "MAC output",
            [
                "Resource<br>directives",
                "mask · power<br>· pilots",
                "Same scheduler<br>slot",
                "Same scheduler<br>slot",
                "Same scheduler<br>slot",
                "Same scheduler<br>slot",
                "Same → PHY<br>TX chain",
                "Same → PHY<br>TX chain",
            ],
        ),
    ]

    for row_label, cells in rows:
        row_ids = [d.box(mx, y, label_w, 44, row_label, "neutral", fixed_width=True)]
        for i, text in enumerate(cells):
            row_ids.append(d.box(hx + i * (col_w + col_gap), y, col_w, 44, text, "neutral", fixed_width=True))
        y = max(d.bottom(n) for n in row_ids) + row_gap

    d.note(
        mx,
        y + 8,
        cw,
        40,
        "Canonical Ch.4 RM sweep: eight schedulers · shared estimated CSI feedback (P2) · CNN supervisor omitted.",
        "accent",
    )
    return d.to_xml()


def build_drl_rm_architecture() -> str:
    """Landscape canvas; dual-column DRL/BER-DRL inference loop with shared offline callout."""
    pw, ph = CANVAS_LANDSCAPE
    d = DrawioBuilder(page_w=pw, page_h=ph)
    mx = 48
    cw = pw - 2 * mx
    row_gap = 14
    gap = 28
    group_pad = 8

    d.title(mx, 16, cw, "Learned RM inference loop")
    d.subtitle(mx, 50, cw, "DRL and BER-DRL actors — shared state, distinct checkpoints")

    # Left column + actor/output row share top band; actors equal size (fixed_width)
    lw = 196
    box_h = 72
    actor_w = 212
    actor_h = 96
    actor_y = 92

    fb = d.box(
        mx,
        actor_y,
        lw,
        box_h,
        "Estimated CSI feedback (P2)<br>H_hat · err_var · Eb/N0<br>shared channel context",
        "primary",
        fixed_width=True,
    )
    state = d.box(
        mx,
        d.bottom(fb) + row_gap,
        lw,
        box_h,
        "Policy state assembly<br>channel energy · Eb/N0 · fairness debt<br>z-score normalisation",
        "primary",
        fixed_width=True,
    )

    group_x = mx + lw + gap
    drl_x = group_x + group_pad
    ber_x = drl_x + actor_w + gap
    group_right = ber_x + actor_w + group_pad
    out_x = ber_x + actor_w + gap

    group_w = group_right - group_x
    d.subtitle(group_x + group_pad, actor_y - group_pad - 22, group_w - 2 * group_pad, "Parallel learned scheduler actors")
    d.group_rect(group_x, actor_y - group_pad, group_w, actor_h + 2 * group_pad)

    drl = d.box(
        drl_x,
        actor_y,
        actor_w,
        actor_h,
        "DRL actor (inference)<br>pretrained policy network<br>throughput · latency reward",
        "secondary",
        fixed_width=True,
    )
    ber = d.box(
        ber_x,
        actor_y,
        actor_w,
        actor_h,
        "BER-DRL actor (inference)<br>pretrained policy network<br>BER-aware reward",
        "secondary",
        fixed_width=True,
    )
    out = d.box(
        out_x,
        actor_y,
        actor_w,
        actor_h,
        "MAC resource directives<br>active-UT mask · per-UT power<br>→ PHY transmitter chain",
        "accent",
        fixed_width=True,
    )

    d.edge(fb, state)
    branch_y = actor_y + actor_h / 2
    state_rx = mx + lw
    drl_cx, drl_cy = d._center(drl)
    ber_cx, ber_cy = d._center(ber)
    d.route(state, drl, [(state_rx + 6, branch_y), (drl_cx, branch_y)])
    d.route(state, ber, [(state_rx + 6, branch_y), (ber_cx, branch_y)])

    merge_x = out_x - gap // 2
    out_cy = d._center(out)[1]
    d.route(drl, out, [(merge_x, drl_cy), (merge_x, out_cy)])
    d.route(ber, out, [(merge_x, ber_cy), (merge_x, out_cy)], color=PALETTE["secondary"][1])

    content_bottom = max(d.bottom(state), actor_y + actor_h + group_pad)
    ny = content_bottom + row_gap + 16
    d.note(
        mx,
        ny,
        cw,
        48,
        "Offline (dashed): actor trained on link-level RM dataset · shared inference loop · "
        "distinct throughput vs BER-aware reward objectives",
        "neutral",
    )
    return d.to_xml()


def build_lr_eval_coupling() -> str:
    """Landscape canvas; three-column literature evaluation-coupling patterns (Ch2 §2.7)."""
    pw, ph = 1200, 820
    d = DrawioBuilder(page_w=pw, page_h=ph)
    mx = 48
    cw = pw - 2 * mx
    col_gap = 72
    n_cols = 3
    col_w = (cw - (n_cols - 1) * col_gap) // n_cols
    row_gap = 22
    inner_pad = 16
    inner_gap = 12
    title_row_h = 34
    before_flow_gap = 18
    box_h = 54

    d.title(mx, 16, cw, "Literature evaluation coupling patterns")
    d.subtitle(
        mx,
        52,
        cw,
        "How prior CE and RM studies are typically benchmarked",
    )

    columns: list[dict[str, Any]] = [
        {
            "title": "PHY-only CE studies",
            "subtitle": "Neural / classical estimator surveys (§2.4)",
            "flow": [
                ("Channel<br>pilots", "neutral"),
                ("CE<br>algorithm", "primary"),
                ("BER /<br>MSE", "success"),
            ],
            "footer": "No MAC masks or scheduling feedback",
            "footer_role": "primary",
        },
        {
            "title": "MAC-only RM studies",
            "subtitle": "Heuristic and DRL scheduling (§2.5)",
            "flow": [
                ("Channel<br>model", "neutral"),
                ("Scheduler<br>(perfect CSI)", "warning"),
                ("Throughput /<br>fairness", "accent"),
            ],
            "footer": "No h_hat error propagation",
            "footer_role": "warning",
        },
        {
            "title": "Integrated (sparse)",
            "subtitle": "Few factory-grounded joint reports",
            "flow": [
                ("Shared<br>context", "primary"),
                ("CE → RM<br>stages", "secondary"),
                ("BER +<br>throughput", "success"),
            ],
            "footer": "Shared realisations + estimated CSI",
            "footer_role": "secondary",
            "sparse": True,
        },
    ]

    panel_y = 98
    col_x = [mx + i * (col_w + col_gap) for i in range(n_cols)]
    inner_w = col_w - 2 * inner_pad
    bw = (inner_w - 2 * inner_gap) // 3

    panel_bottom = panel_y
    flow_ids: list[list[str]] = []

    for i, col in enumerate(columns):
        cx = col_x[i]
        ty = panel_y + inner_pad
        d.title(cx + inner_pad, ty, inner_w, col["title"])
        sty = ty + title_row_h
        sub_norm = normalize_label(col["subtitle"], max_chars=LABEL_MAX_CHARS_SUBTITLE)
        sub_lines = max(1, len(sub_norm.split("<br>")))
        sub_h = sub_lines * 14 + 8
        d.subtitle(cx + inner_pad, sty, inner_w, col["subtitle"])

        fy = sty + sub_h + before_flow_gap
        boxes: list[str] = []
        for j, (label, role) in enumerate(col["flow"]):
            bx = cx + inner_pad + j * (bw + inner_gap)
            bid = d.box(bx, fy, bw, box_h, label, role, fixed_width=True)
            boxes.append(bid)
            if j > 0:
                d.edge(boxes[j - 1], bid)
        flow_ids.append(boxes)

        ny = d.bottom(boxes[-1]) + row_gap
        nid = d.note(
            cx + inner_pad,
            ny,
            inner_w,
            40,
            col["footer"],
            col["footer_role"],
        )
        panel_bottom = max(panel_bottom, d.bottom(nid) + inner_pad)

    for i in range(n_cols):
        d.group_rect(col_x[i], panel_y, col_w, panel_bottom - panel_y)

    _, flow_cy = d._center(flow_ids[0][1])
    gap1_cx = col_x[0] + col_w + col_gap / 2
    gap2_cx = col_x[1] + col_w + col_gap / 2
    d.center_text(gap1_cx - 28, flow_cy - 18, 56, 36, "broken<br>link", color="#be123c")
    d.center_text(gap2_cx - 20, flow_cy - 12, 40, 24, "rare")

    footer_y = panel_bottom + row_gap + 10
    d.note(
        mx,
        footer_y,
        cw,
        44,
        "Integrated end-to-end evaluation under shared factory geometry remains underrepresented "
        "(cf. taxonomy gap)",
        "neutral",
    )
    return d.to_xml()


def build_decoupled_vs_integrated() -> str:
    """Landscape canvas; decoupled literature (left) vs staged integrated design (right)."""
    pw, ph = CANVAS_LANDSCAPE
    d = DrawioBuilder(page_w=pw, page_h=ph)
    mx = 48
    cw = pw - 2 * mx
    col_gap = 24
    panel_w = (cw - col_gap) // 2
    lx, rx = mx, mx + panel_w + col_gap
    row_gap = 10

    d.title(mx, 16, cw, "Decoupled vs integrated evaluation")
    d.subtitle(
        mx,
        50,
        cw,
        "Literature PHY/MAC splits versus staged coupled benchmarking (P1–P3)",
    )

    ly = 88
    lh = 280
    d.group_rect(lx, ly, panel_w, lh)
    d.subtitle(lx + 10, ly + 8, panel_w - 20, "Typical decoupled evaluation")
    d.group_rect(rx, ly, panel_w, lh)
    d.subtitle(rx + 10, ly + 8, panel_w - 20, "Integrated evaluation design (this thesis)")

    inner_gap = 14
    inner_mx = lx + 16
    inner_w = panel_w - 32
    bw = (inner_w - 2 * inner_gap) // 3
    y1 = ly + 40

    ce_ch = d.box(inner_mx, y1, bw, 52, "Channel<br>draw", "neutral", dashed=True, fixed_width=True)
    ce_est = d.box(inner_mx + bw + inner_gap, y1, bw, 52, "Estimator<br>comparison", "primary", fixed_width=True)
    ce_met = d.box(inner_mx + 2 * (bw + inner_gap), y1, bw, 52, "BER / MSE<br>PHY-only", "success", fixed_width=True)
    d.edge(ce_ch, ce_est)
    d.edge(ce_est, ce_met)
    ce_note = d.note(inner_mx, d.bottom(ce_ch) + 6, inner_w, 28, "No RM / MAC impact", "warning")

    y2 = d.bottom(ce_note) + row_gap + 8
    rm_ch = d.box(inner_mx, y2, bw, 52, "Channel<br>model", "neutral", dashed=True, fixed_width=True)
    rm_sched = d.box(inner_mx + bw + inner_gap, y2, bw, 52, "Scheduler<br>perfect CSI / rate tables", "warning", fixed_width=True)
    rm_out = d.box(inner_mx + 2 * (bw + inner_gap), y2, bw, 52, "Sum-rate /<br>throughput", "accent", fixed_width=True)
    d.edge(rm_ch, rm_sched, dashed=True)
    d.edge(rm_sched, rm_out, dashed=True)
    d.note(inner_mx, d.bottom(rm_out) + 8, inner_w, 32, "No h_hat error propagation (broken cross-layer link)", "neutral")

    # Integrated pipeline: two rows of three (six boxes cannot fit one row at tier-11 widths).
    ix = rx + 16
    igap = 12
    row_bw = (inner_w - 2 * igap) // 3
    iy = ly + 52
    ch = d.box(ix, iy, row_bw, 58, "Shared channel<br>draw (P1)", "primary", fixed_width=True)
    ce = d.box(ix + row_bw + igap, iy, row_bw, 58, "CE stage", "secondary", fixed_width=True)
    fb = d.box(ix + 2 * (row_bw + igap), iy, row_bw, 58, "h_hat + err_var<br>(P2)", "primary", fixed_width=True)
    d.edge(ch, ce)
    d.edge(ce, fb)

    iy2 = d.bottom(ch) + 14
    rm = d.box(ix, iy2, row_bw, 58, "RM stage", "secondary", fixed_width=True)
    rd = d.box(ix + row_bw + igap, iy2, row_bw, 58, "MAC resource<br>directives", "accent", dashed=True, fixed_width=True)
    met = d.box(ix + 2 * (row_bw + igap), iy2, row_bw, 58, "BER / throughput<br>(P3)", "success", fixed_width=True)
    d.edge(fb, rm)
    d.edge(rm, rd)
    d.edge(rd, met)

    ny = ly + lh + row_gap + 12
    d.note(
        mx,
        ny,
        cw,
        44,
        "P4 factory scenarios (Rayleigh / Rician / TR 38.901) · P5 end-to-end BER / throughput / latency · "
        "full PHY/MAC chain → Figure integrated pipeline",
        "neutral",
    )
    return d.to_xml()


def build_research_framework() -> str:
    """Landscape research-framework grid: problems → objectives → methodology → outcomes."""
    pw, ph = CANVAS_LANDSCAPE
    d = DrawioBuilder(page_w=pw, page_h=ph)
    mx = 32
    label_w = 108
    gap = 10
    content_x = mx + label_w + gap
    content_w = pw - content_x - mx
    col_gap = 12
    col_w = (content_w - 2 * col_gap) // 3
    row_gap = 10
    arrow_color = "#c62828"

    def col_x(i: int) -> float:
        return content_x + i * (col_w + col_gap)

    def down_arrow(src: str, dst: str) -> None:
        d.edge(src, dst, color=arrow_color)

    def row_h(nodes: list[str], y0: float) -> float:
        return max(d.bottom(n) for n in nodes) - y0

    def label_row(y0: float, nodes: list[str], text: str) -> None:
        d.row_label(mx, y0, label_w, row_h(nodes, y0), text)

    # --- Row 1: Title (full width) ---
    y = 24
    title = d.box(
        content_x,
        y,
        content_w,
        36,
        normalize_label(
            "Integrated Machine Learning for Channel Estimation and "
            "Resource Management in 6G Smart-Factory Wireless Systems",
            max_chars=LABEL_MAX_CHARS_NARROW,
        ),
        "primary",
        fixed_width=True,
    )
    label_row(y, [title], "Title")

    # --- Row 2: Problems ---
    y = d.bottom(title) + row_gap
    problem_labels = [
        "Industrial halls impose multipath, metallic scattering, and mobility "
        "(AGVs, cobots) that stress classical channel models.",
        "Literature often optimises PHY channel estimation and MAC scheduling "
        "in isolation—without propagating estimation error.",
        "Decoupled benchmarking cannot establish whether learned PHY and MAC "
        "policies compose into URLLC-grade factory reliability.",
    ]
    probs = [
        d.box(col_x(i), y, col_w, 36, normalize_label(text, max_chars=LABEL_MAX_CHARS_NARROW), "primary", fixed_width=True)
        for i, text in enumerate(problem_labels)
    ]
    label_row(y, probs, "Problems")
    for p in probs:
        down_arrow(title, p)

    # --- Row 3: Objectives ---
    y = max(d.bottom(p) for p in probs) + row_gap
    objective_labels = [
        "RQ1: Compare learned and classical estimators on BER, confidence, and "
        "runtime across factory channel models.",
        "RQ2: Compare ML/DRL and classical schedulers on throughput, latency, and "
        "fairness under identical estimated CSI.",
        "RQ3: Interpret integrated end-to-end outcomes under methodology "
        "principles P1–P5 versus decoupled PHY-only or MAC-only readings.",
    ]
    objs = [
        d.box(col_x(i), y, col_w, 36, normalize_label(text, max_chars=LABEL_MAX_CHARS_NARROW), "secondary", fixed_width=True)
        for i, text in enumerate(objective_labels)
    ]
    label_row(y, objs, "Objectives")
    for p, o in zip(probs, objs):
        down_arrow(p, o)

    # --- Row 4: Methodology (nested sub-steps; group height follows content) ---
    y = max(d.bottom(o) for o in objs) + row_gap
    inner_pad = 8
    sub_gap = 6
    inner_w = col_w - 2 * inner_pad
    meth_steps = [
        [
            "Survey decoupled CE/RM evaluation and factory wireless requirements.",
            "Anchor scenarios: Rayleigh, Rician, TR 38.901 UMi (P4).",
            "Model NLOS and mobility via stochastic profiles and ray-tracing propagation.",
        ],
        [
            "Integrated evaluation methodology (P1–P5) for staged PHY–MAC coupling.",
            "CE stage: classical and neural estimators on shared channel draws (P1).",
            "RM stage: ML/DRL schedulers on estimated CSI and error variance (P2–P3).",
        ],
        [
            "Monte Carlo link simulation (Sionna) with Eb/N0 sweeps.",
            "Benchmark against LS, PF, WMMSE, and static baselines.",
            "Report BER, throughput, and latency on locked canonical runs.",
        ],
    ]
    meth_tops: list[str] = []
    meth_bottoms: list[str] = []
    meth_cols: list[str] = []

    for i, steps in enumerate(meth_steps):
        gx = col_x(i)
        sub_ids: list[str] = []
        sy = y + inner_pad
        for step in steps:
            sub_id = d.box(
                gx + inner_pad,
                sy,
                inner_w,
                32,
                normalize_label(step, max_chars=LABEL_MAX_CHARS_NARROW),
                "neutral",
                fixed_width=True,
            )
            sub_ids.append(sub_id)
            sy = d.bottom(sub_id) + sub_gap
        group_h = d.bottom(sub_ids[-1]) + inner_pad - y
        d.group_rect(gx, y, col_w, group_h)
        meth_tops.append(sub_ids[0])
        meth_bottoms.append(sub_ids[-1])
        meth_cols.append(sub_ids[-1])

    label_row(y, meth_cols, "Methodology")
    for o, top_id in zip(objs, meth_tops):
        down_arrow(o, top_id)

    # --- Row 5: Outcome ---
    y = max(d.bottom(n) for n in meth_cols) + row_gap
    outcome_labels = [
        "Neural estimation delivers 160–415× lower BER than LS at low Eb/N0 on "
        "shared channel contexts (RQ1).",
        "BER-aware DRL achieves ~25× lower BER than static scheduling on Rician "
        "links at 0 dB (RQ2).",
        "TR 38.901 UMi shows ~10× BER gain over static at matched throughput; "
        "integrated readings expose MAC-dominated trade-offs (RQ3).",
    ]
    results = [
        d.box(col_x(i), y, col_w, 36, normalize_label(text, max_chars=LABEL_MAX_CHARS_NARROW), "success", fixed_width=True)
        for i, text in enumerate(outcome_labels)
    ]
    label_row(y, results, "Outcome")
    for mb, r in zip(meth_bottoms, results):
        down_arrow(mb, r)

    # --- Row 6: Significance (full width) ---
    y = max(d.bottom(r) for r in results) + row_gap
    sig = d.box(
        content_x,
        y,
        content_w,
        36,
        normalize_label(
            "An integrated ML evaluation framework for robust, efficient, and scalable "
            "6G wireless in Industry 5.0 smart factories.",
            max_chars=LABEL_MAX_CHARS_NARROW,
        ),
        "accent",
        fixed_width=True,
    )
    label_row(y, [sig], "Significance Impact")
    for r in results:
        down_arrow(r, sig)

    d.page_h = max(ph, int(d.bottom(sig)) + 24)
    return d.to_xml()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate thesis architecture figures as draw.io XML.")
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="STEM",
        help="Write only these figure stems (e.g. fig_lr_taxonomy), without .drawio suffix.",
    )
    args = parser.parse_args()

    BUILDERS = {
        "fig_integrated_pipeline.drawio": build_integrated_pipeline,
        "fig_lr_taxonomy.drawio": build_lr_taxonomy,
        "fig_phy_stack.drawio": build_phy_stack,
        "fig_monte_carlo_flow.drawio": build_monte_carlo,
        "fig_factory_topology.drawio": build_factory_topology,
        "fig_estimator_comparison.drawio": build_estimator_comparison,
        "fig_rm_comparison.drawio": build_rm_comparison,
        "fig_drl_rm_architecture.drawio": build_drl_rm_architecture,
        "fig_decoupled_vs_integrated.drawio": build_decoupled_vs_integrated,
        "fig_lr_eval_coupling.drawio": build_lr_eval_coupling,
        "fig_research_framework.drawio": build_research_framework,
    }

    if args.only:
        wanted = {f"{stem}.drawio" if not stem.endswith(".drawio") else stem for stem in args.only}
        missing = wanted - set(BUILDERS)
        if missing:
            raise SystemExit(f"Unknown figure stem(s): {', '.join(sorted(missing))}")
        outputs = {name: BUILDERS[name]() for name in wanted}
    else:
        outputs = {name: fn() for name, fn in BUILDERS.items()}

    for name, xml in outputs.items():
        path = FIGDIR / name
        path.write_text(xml, encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
