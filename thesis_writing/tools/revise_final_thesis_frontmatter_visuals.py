#!/usr/bin/env python3
"""Revise the final thesis DOCX without python-docx/lxml dependencies.

The project Docker image intentionally lacks Word/document packages, so this
script edits the DOCX package directly with stdlib XML plus Pillow.
"""

from __future__ import annotations

import copy
import re
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

import complete_from_v3 as base


ROOT = Path("/app")
THESIS_DIR = ROOT / "thesis_writing"
SRC_DOCX = THESIS_DIR / "Factory6G-final-thesis.docx"
OUT_DOCX = THESIS_DIR / "Factory6G-final-thesis-revised.docx"
TMP_DIR = THESIS_DIR / ".revision_docx"
GENERATED_DIR = THESIS_DIR / "generated_revision"
AUDIT_MD = THESIS_DIR / "Factory6G-final-thesis-source-audit.md"

FOUNDATION_FILES = {
    "framework": GENERATED_DIR / "factory6g_framework_foundation.png",
    "workflow": GENERATED_DIR / "factory6g_evidence_workflow_foundation.png",
    "ber_drl": GENERATED_DIR / "ber_drl_inference_loop_foundation.png",
}

EXCALIDRAW_RENDERS = {
    "framework": GENERATED_DIR / "excalidraw" / "factory6g_framework.png",
    "workflow": GENERATED_DIR / "excalidraw" / "evidence_workflow.png",
    "ber_drl": GENERATED_DIR / "excalidraw" / "ber_drl_loop.png",
}

DIAGRAM_FILES = {
    "framework": GENERATED_DIR / "factory6g_framework_diagram.png",
    "workflow": GENERATED_DIR / "factory6g_evidence_workflow.png",
    "ber_drl": GENERATED_DIR / "ber_drl_inference_loop.png",
}

PUBLICATIONS = [
    (
        "CCTV armed robbery detection with YOLOv8",
        "AIP Conference Proceedings 3367(1), 020006, 2025.",
        "https://pubs.aip.org/aip/acp/article-abstract/3367/1/020006/3367869/CCTV-armed-robbery-detection-with-YOLOv8",
    ),
    (
        "URLLC for 6G Enabled Industry 5.0: A Taxonomy of Architectures, Cross Layer Techniques, and Time Critical Applications",
        "arXiv preprint arXiv:2510.08080, 2025.",
        "https://arxiv.org/pdf/2510.08080",
    ),
    (
        "Review and enhancement of VoIP security: Identifying vulnerabilities and proposing integrated solutions",
        "Journal of Telecommunications and the Digital Economy 12(4), 109-136, 2024.",
        "https://www.researchgate.net/profile/Athirah-Mohd-Ramly/publication/387551333_Review_and_Enhancement_of_VoIP_Security_Identifying_Vulnerabilities_and_Proposing_Integrated_Solutions/links/67754747e74ca64e1f40257d/Review-and-Enhancement-of-VoIP-Security-Identifying-Vulnerabilities-and-Proposing-Integrated-Solutions.pdf",
    ),
    (
        "Ethics and its role in the future of AI development",
        "Proceedings of the 1st International Conference on Frontier of Digital, AIP Conference Proceedings 2808(1), 040003, 2023.",
        "https://pubs.aip.org/aip/acp/article-abstract/2808/1/040003/2891836/Ethics-and-its-role-in-the-future-of-AI?redirectedFrom=PDF",
    ),
    (
        "Artificial 3d printed Robotic Arm controlled by Brain Waves",
        "Najah National University repository, 2019.",
        "https://repository.najah.edu/items/ba2fdb62-1a2d-4cd5-9a07-c1e31302a608",
    ),
    (
        "Leat",
        "Najah National University repository, 2019.",
        "https://repository.najah.edu/handle/20.500.11888/15500",
    ),
]

W = base.W
R = base.R
REL = base.REL
CT = base.CT
NS = base.NS
XML = "http://www.w3.org/XML/1998/namespace"
WP = base.WP
PIC = base.PIC
MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"

PORTRAIT_W = "11906"
PORTRAIT_H = "16838"
LANDSCAPE_W = "16838"
LANDSCAPE_H = "11906"
MARGIN = "1440"
BODY_WIDTH_PORTRAIT = 9360
BODY_WIDTH_LANDSCAPE = 13958


FIGURE_SOURCES = [
    "system_design/topology_3d_factory.png",
    "system_design/topology_mobility.png",
    "system_design/phy_mac_topology.png",
    "thesis_writing/generated_revision/factory6g_framework_diagram.png",
    "thesis_writing/generated_revision/factory6g_evidence_workflow.png",
    "thesis_writing/generated_revision/ber_drl_inference_loop.png",
    "reports/plots/estimator_ber_vs_ebno.png",
    "reports/plots/estimator_latency_vs_ebno.png",
    "reports/plots/estimator_throughput_vs_ebno.png",
    "reports/plots/estimator_runtime.png",
    "reports/plots/neural_vs_ls_direct_ber.png",
    "reports/plots/neural_vs_ls_latency.png",
    "reports/plots/channel_model_ber_vs_ebno.png",
    "reports/plots/modulation_ber_vs_ebno.png",
    "reports/plots/modulation_latency_vs_ebno.png",
    "reports/plots/factory_size_ber_vs_ebno.png",
    "results/20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s/overview/resource_managers/ber_vs_ebno.png",
    "results/20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s/overview/resource_managers/latency_vs_ebno.png",
    "results/20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s/overview/resource_managers/throughput_vs_ebno.png",
    "results/20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s/rayleigh/resource_managers/ber_vs_ebno.png",
    "results/20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s/rician/resource_managers/ber_vs_ebno.png",
    "results/20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s/tr38901/resource_managers/ber_vs_ebno.png",
    "results/20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s/resource_manager_channel_comparison_synthetic_methods/ber_drl_ber_vs_ebno.png",
    "reports/plots/jidd_ber_comparison.png",
    "reports/plots/combined_ber.png",
    "reports/plots/runtime_comparison.png",
]


def qn(tag: str) -> str:
    return base.qn(W, tag)


def rel_qn(tag: str) -> str:
    return base.qn(REL, tag)


def ct_qn(tag: str) -> str:
    return base.qn(CT, tag)


def paragraph_text(p: ET.Element) -> str:
    return base.paragraph_text(p)


def p_style(p: ET.Element) -> str:
    return base.p_style(p)


def children(body: ET.Element) -> list[ET.Element]:
    return list(body)


def is_p(el: ET.Element) -> bool:
    return el.tag == qn("p")


def is_tbl(el: ET.Element) -> bool:
    return el.tag == qn("tbl")


def ensure_ppr(p: ET.Element) -> ET.Element:
    ppr = p.find("./w:pPr", NS)
    if ppr is None:
        ppr = base.w_el("pPr")
        p.insert(0, ppr)
    return ppr


def clear_paragraph_runs(p: ET.Element) -> None:
    for child in list(p):
        if child.tag != qn("pPr"):
            p.remove(child)


def set_paragraph_text(p: ET.Element, text: str, *, bold: bool = False, italic: bool = False) -> None:
    clear_paragraph_runs(p)
    r = base.sub(p, "r")
    if bold or italic:
        rpr = base.sub(r, "rPr")
        if bold:
            base.sub(rpr, "b")
        if italic:
            base.sub(rpr, "i")
    t = base.sub(r, "t", {base.qn(XML, "space"): "preserve"})
    t.text = text


def make_body_p(text: str, *, italic: bool = False) -> ET.Element:
    p = base.make_p(text)
    if italic:
        r = p.find("./w:r", NS)
        if r is not None:
            rpr = r.find("./w:rPr", NS)
            if rpr is None:
                rpr = base.w_el("rPr")
                r.insert(0, rpr)
            base.sub(rpr, "i")
    return p


def make_page_break_p() -> ET.Element:
    p = base.w_el("p")
    r = base.sub(p, "r")
    base.sub(r, "br", {qn("type"): "page"})
    return p


def make_list_entry(text: str, bookmark: str, level: int = 0) -> ET.Element:
    p = base.w_el("p")
    ppr = base.sub(p, "pPr")
    base.sub(
        ppr,
        "tabs",
    ).append(
        base.w_el(
            "tab",
            {
                qn("val"): "right",
                qn("leader"): "dot",
                qn("pos"): str(BODY_WIDTH_PORTRAIT),
            },
        )
    )
    if level:
        base.sub(ppr, "ind", {qn("left"): str(level * 360)})
    base.sub(ppr, "spacing", {qn("after"): "40", qn("line"): "240", qn("lineRule"): "auto"})

    r = base.sub(p, "r")
    t = base.sub(r, "t", {base.qn(XML, "space"): "preserve"})
    t.text = text
    base.sub(p, "r").append(base.w_el("tab"))
    add_pageref_field(p, bookmark)
    return p


def add_pageref_field(p: ET.Element, bookmark: str) -> None:
    r_begin = base.sub(p, "r")
    base.sub(r_begin, "fldChar", {qn("fldCharType"): "begin", qn("dirty"): "true"})
    r_instr = base.sub(p, "r")
    instr = base.sub(r_instr, "instrText", {base.qn(XML, "space"): "preserve"})
    instr.text = f" PAGEREF {bookmark} \\h "
    r_sep = base.sub(p, "r")
    base.sub(r_sep, "fldChar", {qn("fldCharType"): "separate"})
    r_res = base.sub(p, "r")
    t = base.sub(r_res, "t")
    t.text = "0"
    r_end = base.sub(p, "r")
    base.sub(r_end, "fldChar", {qn("fldCharType"): "end"})


def add_external_hyperlink(p: ET.Element, rels_root: ET.Element, text: str, url: str) -> None:
    rid = base.next_rel_id(rels_root)
    rels_root.append(
        ET.Element(
            rel_qn("Relationship"),
            {
                "Id": rid,
                "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
                "Target": url,
                "TargetMode": "External",
            },
        )
    )
    hyperlink = ET.SubElement(p, qn("hyperlink"), {base.qn(R, "id"): rid})
    r = ET.SubElement(hyperlink, qn("r"))
    rpr = ET.SubElement(r, qn("rPr"))
    ET.SubElement(rpr, qn("color"), {qn("val"): "0563C1"})
    ET.SubElement(rpr, qn("u"), {qn("val"): "single"})
    t = ET.SubElement(r, qn("t"))
    t.text = text


def make_publication_p(title: str, venue: str, url: str, rels_root: ET.Element) -> ET.Element:
    p = base.w_el("p")
    ppr = base.sub(p, "pPr")
    base.sub(ppr, "spacing", {qn("after"): "140", qn("line"): "276", qn("lineRule"): "auto"})
    base.sub(ppr, "ind", {qn("left"): "360"})

    title_run = base.sub(p, "r")
    title_rpr = base.sub(title_run, "rPr")
    base.sub(title_rpr, "b")
    title_text = base.sub(title_run, "t", {base.qn(XML, "space"): "preserve"})
    title_text.text = f"{title}. "

    venue_run = base.sub(p, "r")
    venue_text = base.sub(venue_run, "t", {base.qn(XML, "space"): "preserve"})
    venue_text.text = f"{venue} "

    add_external_hyperlink(p, rels_root, "[Link]", url)
    return p


def find_direct_index(body: ET.Element, text: str) -> int:
    for idx, el in enumerate(children(body)):
        if is_p(el) and paragraph_text(el) == text:
            return idx
    raise ValueError(f"Could not find paragraph: {text}")


def find_direct_index_contains(body: ET.Element, needle: str) -> int:
    for idx, el in enumerate(children(body)):
        if is_p(el) and needle in paragraph_text(el):
            return idx
    raise ValueError(f"Could not find paragraph containing: {needle}")


def replace_between_headings(body: ET.Element, start_heading: str, end_heading: str, new_elems: list[ET.Element]) -> None:
    elems = children(body)
    start = find_direct_index(body, start_heading)
    end = find_direct_index(body, end_heading)
    body[:] = elems[: start + 1] + new_elems + elems[end:]


def next_table_after(body: ET.Element, caption_idx: int) -> int:
    elems = children(body)
    for idx in range(caption_idx + 1, len(elems)):
        if is_tbl(elems[idx]):
            return idx
        if is_p(elems[idx]) and p_style(elems[idx]).startswith("Heading"):
            break
    raise ValueError("No table after caption")


def replace_table_after_caption(body: ET.Element, caption_needle: str, new_table: ET.Element) -> None:
    caption_idx = find_direct_index_contains(body, caption_needle)
    table_idx = next_table_after(body, caption_idx)
    elems = children(body)
    elems[table_idx] = new_table
    body[:] = elems


def insert_after_index(body: ET.Element, index: int, new_elems: list[ET.Element]) -> None:
    elems = children(body)
    body[:] = elems[: index + 1] + new_elems + elems[index + 1 :]


def paragraph_after_heading(body: ET.Element, heading_text: str) -> int:
    elems = children(body)
    start = find_direct_index(body, heading_text)
    for idx in range(start + 1, len(elems)):
        if is_p(elems[idx]) and paragraph_text(elems[idx]):
            return idx
    raise ValueError(f"No paragraph after {heading_text}")


def table_after_caption_text(body: ET.Element, caption_text: str) -> int:
    cap_idx = find_direct_index(body, caption_text)
    return next_table_after(body, cap_idx)


def max_bookmark_id(root: ET.Element) -> int:
    max_id = 0
    for bm in root.iter(qn("bookmarkStart")):
        raw = bm.get(qn("id"), "0")
        if raw.isdigit():
            max_id = max(max_id, int(raw))
    return max_id


def add_bookmark_to_paragraph(p: ET.Element, name: str, bm_id: int) -> None:
    # Remove an existing same-name bookmark in the paragraph if a previous run created one.
    for child in list(p):
        if child.tag in {qn("bookmarkStart"), qn("bookmarkEnd")} and child.get(qn("name")) == name:
            p.remove(child)
    start = base.w_el("bookmarkStart", {qn("id"): str(bm_id), qn("name"): name})
    end = base.w_el("bookmarkEnd", {qn("id"): str(bm_id)})
    insert_pos = 1 if p.find("./w:pPr", NS) is not None else 0
    p.insert(insert_pos, start)
    p.append(end)


def add_or_replace_section_break(p: ET.Element, *, landscape: bool) -> None:
    ppr = ensure_ppr(p)
    old = ppr.find("./w:sectPr", NS)
    if old is not None:
        ppr.remove(old)
    sect = base.sub(ppr, "sectPr")
    base.sub(sect, "type", {qn("val"): "nextPage"})
    if landscape:
        base.sub(sect, "pgSz", {qn("w"): LANDSCAPE_W, qn("h"): LANDSCAPE_H, qn("orient"): "landscape"})
    else:
        base.sub(sect, "pgSz", {qn("w"): PORTRAIT_W, qn("h"): PORTRAIT_H})
    base.sub(
        sect,
        "pgMar",
        {
            qn("top"): MARGIN,
            qn("right"): MARGIN,
            qn("bottom"): MARGIN,
            qn("left"): MARGIN,
            qn("header"): "720",
            qn("footer"): "720",
            qn("gutter"): "0",
        },
    )
    base.sub(sect, "cols", {qn("space"): "720"})
    base.sub(sect, "docGrid", {qn("linePitch"): "360"})


def set_body_section(body: ET.Element, *, landscape: bool) -> None:
    sect = body.find("w:sectPr", NS)
    if sect is None:
        sect = base.w_el("sectPr")
        body.append(sect)
    for child in list(sect):
        if child.tag in {qn("pgSz"), qn("pgMar"), qn("cols"), qn("docGrid")}:
            sect.remove(child)
    if landscape:
        base.sub(sect, "pgSz", {qn("w"): LANDSCAPE_W, qn("h"): LANDSCAPE_H, qn("orient"): "landscape"})
    else:
        base.sub(sect, "pgSz", {qn("w"): PORTRAIT_W, qn("h"): PORTRAIT_H})
    base.sub(
        sect,
        "pgMar",
        {
            qn("top"): MARGIN,
            qn("right"): MARGIN,
            qn("bottom"): MARGIN,
            qn("left"): MARGIN,
            qn("header"): "720",
            qn("footer"): "720",
            qn("gutter"): "0",
        },
    )
    base.sub(sect, "cols", {qn("space"): "720"})
    base.sub(sect, "docGrid", {qn("linePitch"): "360"})


def make_section_break_paragraph(*, landscape: bool) -> ET.Element:
    p = base.w_el("p")
    add_or_replace_section_break(p, landscape=landscape)
    return p


def end_section_before_heading(body: ET.Element, heading: str, *, previous_landscape: bool) -> None:
    elems = children(body)
    start = find_direct_index(body, heading)
    for idx in range(start - 1, -1, -1):
        if is_p(elems[idx]):
            add_or_replace_section_break(elems[idx], landscape=previous_landscape)
            body[:] = elems
            return
        if is_tbl(elems[idx]):
            elems.insert(start, make_section_break_paragraph(landscape=previous_landscape))
            body[:] = elems
            return
    raise ValueError(f"Could not insert section break before heading: {heading}")


def set_table_geometry(tbl: ET.Element, widths: list[int]) -> None:
    total = sum(widths)
    tbl_pr = tbl.find("./w:tblPr", NS)
    if tbl_pr is None:
        tbl_pr = base.w_el("tblPr")
        tbl.insert(0, tbl_pr)
    tbl_w = tbl_pr.find("./w:tblW", NS)
    if tbl_w is None:
        tbl_w = base.w_el("tblW")
        tbl_pr.insert(0, tbl_w)
    tbl_w.set(qn("w"), str(total))
    tbl_w.set(qn("type"), "dxa")
    layout = tbl_pr.find("./w:tblLayout", NS)
    if layout is None:
        layout = base.w_el("tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("type"), "fixed")

    grid = tbl.find("./w:tblGrid", NS)
    if grid is None:
        grid = base.w_el("tblGrid")
        tbl.insert(1, grid)
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        base.sub(grid, "gridCol", {qn("w"): str(width)})

    for row in tbl.findall("./w:tr", NS):
        cells = row.findall("./w:tc", NS)
        if not cells:
            continue
        for idx, cell in enumerate(cells):
            width = widths[min(idx, len(widths) - 1)]
            tc_pr = cell.find("./w:tcPr", NS)
            if tc_pr is None:
                tc_pr = base.w_el("tcPr")
                cell.insert(0, tc_pr)
            tc_w = tc_pr.find("./w:tcW", NS)
            if tc_w is None:
                tc_w = base.w_el("tcW")
                tc_pr.insert(0, tc_w)
            tc_w.set(qn("w"), str(width))
            tc_w.set(qn("type"), "dxa")


def widen_appendix_tables(body: ET.Element) -> None:
    elems = children(body)
    app_a = find_direct_index(body, "Appendix A: Reproducibility Commands")
    refs = find_direct_index(body, "References")
    for idx in range(app_a, refs):
        el = elems[idx]
        if not is_tbl(el):
            continue
        rows = el.findall("./w:tr", NS)
        first_cells = rows[0].findall("./w:tc", NS) if rows else []
        count = max(1, len(first_cells))
        if count == 1:
            widths = [BODY_WIDTH_LANDSCAPE]
        elif count == 2:
            text = paragraph_text(first_cells[0]) if first_cells else ""
            widths = [3800, BODY_WIDTH_LANDSCAPE - 3800] if any(token in text for token in ("Algorithm", "Stage")) else [7000, BODY_WIDTH_LANDSCAPE - 7000]
        elif count == 3:
            widths = [1900, 6200, BODY_WIDTH_LANDSCAPE - 8100]
        else:
            even = BODY_WIDTH_LANDSCAPE // count
            widths = [even] * count
            widths[-1] += BODY_WIDTH_LANDSCAPE - sum(widths)
        set_table_geometry(el, widths)


def set_update_fields(settings_path: Path) -> None:
    tree = ET.parse(settings_path)
    root = tree.getroot()
    remove_ignorable_attr(root)
    existing = root.find("w:updateFields", NS)
    if existing is None:
        existing = base.w_el("updateFields")
        root.append(existing)
    existing.set(qn("val"), "true")
    tree.write(settings_path, encoding="utf-8", xml_declaration=True)


def remove_ignorable_attr(root: ET.Element) -> None:
    for attr in list(root.attrib):
        if attr == base.qn(MC, "Ignorable"):
            del root.attrib[attr]


def renumber_drawing_ids(root: ET.Element) -> None:
    next_id = 1
    for doc_pr in root.iter(base.qn(WP, "docPr")):
        doc_pr.set("id", str(next_id))
        if not doc_pr.get("name"):
            doc_pr.set("name", f"Picture {next_id}")
        next_id += 1
    next_id = 1
    for cnv_pr in root.iter(base.qn(PIC, "cNvPr")):
        cnv_pr.set("id", str(next_id))
        next_id += 1


def ensure_png_content_type(tmp_dir: Path) -> None:
    path = tmp_dir / "[Content_Types].xml"
    tree = ET.parse(path)
    root = tree.getroot()
    if any(e.tag == ct_qn("Default") and e.get("Extension") == "png" for e in root):
        return
    root.append(ET.Element(ct_qn("Default"), {"Extension": "png", "ContentType": "image/png"}))
    ET.register_namespace("", CT)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def draw_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], fill: str, outline: str, text: str, font: ImageFont.ImageFont, *, align: str = "center") -> None:
    draw.rounded_rectangle(xy, radius=18, fill=fill, outline=outline, width=3)
    x1, y1, x2, y2 = xy
    words = text.replace("\n", " ").split()
    lines: list[str] = []
    line = ""
    max_width = x2 - x1 - 28
    for word in words:
        test = f"{line} {word}".strip()
        if draw.textbbox((0, 0), test, font=font)[2] <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    sample = draw.textbbox((0, 0), "Ag", font=font)
    line_h = (sample[3] - sample[1]) + 10
    y = y1 + (y2 - y1 - len(lines) * line_h) // 2
    for ln in lines:
        width = draw.textbbox((0, 0), ln, font=font)[2]
        x = x1 + 18 if align == "left" else x1 + (x2 - x1 - width) // 2
        draw.text((x, y), ln, fill="#17324d", font=font)
        y += line_h


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill: str = "#375a7f") -> None:
    draw.line([start, end], fill=fill, width=4)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex > sx else -1
        pts = [(ex, ey), (ex - 16 * direction, ey - 9), (ex - 16 * direction, ey + 9)]
    else:
        direction = 1 if ey > sy else -1
        pts = [(ex, ey), (ex - 9, ey - 16 * direction), (ex + 9, ey - 16 * direction)]
    draw.polygon(pts, fill=fill)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/local/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf" if bold else "/usr/local/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_foundation(kind: str) -> Image.Image:
    path = FOUNDATION_FILES[kind]
    if path.exists():
        return Image.open(path).convert("RGBA")
    return Image.new("RGBA", (1672, 941), (248, 251, 253, 255))


def wrap_text_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def labeled_panel(
    img: Image.Image,
    xy: tuple[int, int, int, int],
    title: str,
    body: str,
    *,
    accent: str,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
) -> None:
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    x1, y1, x2, y2 = xy
    sd.rounded_rectangle((x1 + 8, y1 + 10, x2 + 8, y2 + 10), radius=24, fill=(12, 33, 56, 34))
    sd.rounded_rectangle(xy, radius=24, fill=(255, 255, 255, 225), outline=accent, width=3)
    sd.rounded_rectangle((x1, y1, x2, y1 + 15), radius=12, fill=accent)
    img.alpha_composite(shadow)

    d = ImageDraw.Draw(img)
    pad = 24
    title_lines = wrap_text_to_width(d, title, title_font, x2 - x1 - pad * 2)
    body_lines = wrap_text_to_width(d, body, body_font, x2 - x1 - pad * 2)
    title_h = getattr(title_font, "size", 34) + 7
    body_h = getattr(body_font, "size", 25) + 6
    if body_lines:
        y = y1 + 28
    else:
        y = y1 + max(18, (y2 - y1 - len(title_lines) * title_h) // 2)
    for line in title_lines:
        d.text((x1 + pad, y), line, fill="#0b2d4d", font=title_font)
        y += title_h
    y += 5
    for line in body_lines:
        d.text((x1 + pad, y), line, fill="#29475f", font=body_font)
        y += body_h


def generate_diagrams() -> dict[str, Path]:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    if all(path.exists() for path in EXCALIDRAW_RENDERS.values()):
        for key, src in EXCALIDRAW_RENDERS.items():
            shutil.copyfile(src, DIAGRAM_FILES[key])
        return dict(DIAGRAM_FILES)

    title_font = load_font(32, bold=True)
    body_font = load_font(23)

    out: dict[str, Path] = {}

    img = load_foundation("framework")
    labeled_panel(img, (715, 125, 925, 260), "Factory context", "", accent="#0b4e78", title_font=title_font, body_font=body_font)
    labeled_panel(img, (715, 305, 925, 440), "Control variables", "", accent="#0b4e78", title_font=title_font, body_font=body_font)
    labeled_panel(img, (960, 140, 1185, 285), "PHY/MAC model", "", accent="#138a8a", title_font=title_font, body_font=body_font)
    labeled_panel(img, (960, 650, 1185, 805), "Policy space", "", accent="#138a8a", title_font=title_font, body_font=body_font)
    labeled_panel(img, (1215, 255, 1430, 405), "Evidence", "", accent="#b7832d", title_font=title_font, body_font=body_font)
    labeled_panel(img, (1445, 300, 1625, 600), "Bounded claims", "", accent="#0b3f75", title_font=title_font, body_font=body_font)
    path = DIAGRAM_FILES["framework"]
    img.convert("RGB").save(path)
    out["framework"] = path

    img = load_foundation("workflow")
    workflow_panels = [
        ((70, 300, 300, 475), "Design", "seed + grid", "#0d4776"),
        ((335, 300, 565, 475), "Context", "same channel", "#168a96"),
        ((600, 300, 830, 475), "Estimator", "receiver BER", "#168a96"),
        ((865, 300, 1095, 475), "PHY feedback", "state view", "#628a55"),
        ((1130, 300, 1360, 475), "Policy", "scheduler + DRL", "#b9862c"),
        ((1390, 300, 1618, 475), "Evidence", "audit trail", "#0d4776"),
    ]
    for xy, title, body, accent in workflow_panels:
        labeled_panel(img, xy, title, body, accent=accent, title_font=title_font, body_font=body_font)
    path = DIAGRAM_FILES["workflow"]
    img.convert("RGB").save(path)
    out["workflow"] = path

    img = load_foundation("ber_drl")
    labeled_panel(img, (200, 300, 550, 470), "State", "channel + mask", accent="#168a96", title_font=title_font, body_font=body_font)
    labeled_panel(img, (665, 65, 1015, 235), "Policy", "pi_theta reliability preference", accent="#0d4776", title_font=title_font, body_font=body_font)
    labeled_panel(img, (1110, 300, 1455, 470), "Directive", "users + power", accent="#6d3f91", title_font=title_font, body_font=body_font)
    labeled_panel(img, (965, 680, 1330, 845), "Outcome", "BER, latency, throughput", accent="#628a55", title_font=title_font, body_font=body_font)
    labeled_panel(img, (320, 680, 690, 845), "Reward", "baseline reading", accent="#b9862c", title_font=title_font, body_font=body_font)
    path = DIAGRAM_FILES["ber_drl"]
    img.convert("RGB").save(path)
    out["ber_drl"] = path
    return out


def make_figure_block(tmp_dir: Path, rels_root: ET.Element, image_path: Path, image_name: str, caption: str) -> list[ET.Element]:
    return [
        base.make_image_p(tmp_dir, rels_root, image_path, image_name, max_width_in=6.15),
        base.make_p(caption, "Caption", bold=True, italic=True),
    ]


def insert_chapter3_visuals(body: ET.Element, tmp_dir: Path, rels_root: ET.Element, diagrams: dict[str, Path]) -> None:
    intro_idx = paragraph_after_heading(body, "Chapter 3: System Model and Methodology")
    insert_after_index(
        body,
        intro_idx,
        [make_page_break_p()]
        + make_figure_block(
            tmp_dir,
            rels_root,
            diagrams["framework"],
            "factory6g_framework_diagram",
            "Figure 4. Conceptual cross-layer framework for reliability evidence in AI-assisted 6G smart-factory communication.",
        )
        + [
            make_body_p(
                "The framework should be read as a theoretical evidence model rather than as a software architecture. Factory context and controlled assumptions define the operating point; PHY estimation and MAC/resource-management policies instantiate the communication mechanisms; and the evidence layer constrains which claims can be made about reliability, latency, throughput, and learned control."
            )
        ],
    )

    alg1_table = table_after_caption_text(body, "Algorithm 1. Factory6G evidence-generation workflow")
    insert_after_index(
        body,
        alg1_table,
        make_figure_block(
            tmp_dir,
            rels_root,
            diagrams["workflow"],
            "factory6g_evidence_workflow",
            "Figure 5. Methodological evidence chain connecting experimental design, common channel context, estimator comparison, policy comparison, and reproducible interpretation.",
        )
        + [
            make_body_p(
                "The workflow complements Algorithm 1 by showing how the simulation protocol functions as a research design. Common random context, fixed Eb/N0 grids, and consistent stopping rules make the estimator and scheduler comparisons interpretable as controlled evidence rather than as unrelated computational runs."
            )
        ],
    )

    alg2_table = table_after_caption_text(body, "Algorithm 2. BER-oriented resource-manager inference")
    insert_after_index(
        body,
        alg2_table,
        [make_page_break_p()]
        + make_figure_block(
            tmp_dir,
            rels_root,
            diagrams["ber_drl"],
            "ber_drl_inference_loop",
            "Figure 6. Reliability-oriented DRL control loop interpreted as a bounded policy model inside the 6G smart-factory communication stack.",
        )
        + [
            make_body_p(
                "The loop frames BER-DRL as a policy-level hypothesis: a learned mapping from channel/state observations to scheduling and power actions can encode a reliability preference. The thesis does not treat this policy as universally optimal; it evaluates whether the learned preference remains credible when read against observed BER, upper-confidence bounds, latency, throughput, runtime, and deterministic baselines."
            )
        ],
    )


def add_results_interpretation(body: ET.Element) -> None:
    insert_after_index(
        body,
        find_direct_index_contains(body, "The estimator figures should not be interpreted as a single winner-takes-all ranking."),
        [
            make_body_p(
                "The estimator evidence separates three theoretical effects that are easy to collapse into a single ranking: error-rate reduction as Eb/N0 improves, latency introduced by receiver processing, and computational burden introduced by method complexity. A method is therefore meaningful only when the reliability gain remains defensible after these timing and feasibility constraints are considered."
            )
        ],
    )
    insert_after_index(
        body,
        find_direct_index_contains(body, "The implication is methodological: results should be reported by channel family"),
        [
            make_body_p(
                "The channel-model result also explains why the later MAC-layer ranking is reported separately for Rayleigh, Rician, and TR 38.901 UMi. A single averaged curve would hide whether a policy is robust or merely well matched to one propagation assumption."
            )
        ],
    )
    insert_after_index(
        body,
        find_direct_index_contains(body, "Figure 19. TR 38.901 UMi resource-manager BER curves"),
        [
            make_body_p(
                "The cross-channel resource-manager figures show that reliability is policy- and channel-dependent rather than a property of a scheduler alone. In Rayleigh, several policies reach the zero-observed-error group and must be compared using the BER upper-confidence bound. In Rician and UMi/TR 38.901, max-throughput remains the best observed method, while BER-DRL stays near the top but does not dominate the deterministic baseline."
            ),
            make_body_p(
                "The theoretical implication is therefore hybrid rather than purely learned. Deterministic methods provide transparent reference behaviour and strong baselines, whereas BER-DRL contributes a reliability-oriented learned policy whose value depends on reward design, channel family, and the evidence boundary used for comparison."
            ),
        ],
    )
    insert_after_index(
        body,
        find_direct_index_contains(body, "Figure 21. JIDD-SCMA BER comparison"),
        [
            make_body_p(
                "The JIDD-SCMA result is intentionally not used as a central ranking claim. Its value is methodological: it shows that joint detection and decoding can be incorporated into the Factory6G evidence chain, while the current non-monotonic behaviour signals the need for additional stability checks before strong conclusions are drawn."
            )
        ],
    )


def academicize_existing_language(body: ET.Element) -> None:
    replacements = {
        "Each major result is linked to a CSV, JSON, or PNG source in the audit note.": "Each major result is linked to a traceable evidence artifact and audit note.",
        "Factory6G is organized around a fixed simulation flow. A configuration file defines channel model, modulation, factory profile, Monte Carlo policy, enabled estimators, enabled resource managers, and output preferences. The runtime creates a timestamped output directory, prepares deterministic seeds, runs selected stages, writes JSON and CSV results, and generates plots for BER, raw BER, latency, throughput, power, and runtime.": "Factory6G is organized as a controlled evidence model for a smart-factory radio system. Each experiment fixes the propagation family, modulation, factory profile, Monte Carlo design, estimator set, and resource-management policy set so that reliability comparisons are made under comparable operating assumptions. The software implementation instantiates these choices, but the methodological object is the experiment design: a reproducible mapping from channel conditions and control policies to BER, latency, throughput, power, and runtime evidence.",
        "The system is intentionally modular. PHY components handle transmitter, channel, receiver, and estimator behaviour. MAC/resource-management components generate scheduling and power directives. Simulation orchestration controls shared batch contexts, stopping rules, output writing, and plot generation. This separation allows individual methods to be compared while retaining a common measurement protocol.": "The modularity reflects the cross-layer structure of the research problem. The PHY layer represents transmission, propagation, reception, and estimation; the MAC/resource-management layer represents scheduling and power-control decisions; and the evidence layer preserves the measurement protocol. This separation allows individual methods to be compared while retaining a common theoretical basis for interpretation.",
        "Input: config.json, selected channel family, estimator list, resource-manager list, Eb/N0 grid": "Input: experimental design, selected channel family, estimator set, resource-manager set, Eb/N0 grid",
        "8. Write stage_results_v2.csv/json, summary_v2.csv/json, plots, and logs.": "8. Preserve traceable evidence records, summary tables, figures, and logs for verification.",
        "Output: traceable result artifacts for thesis tables, figures, and interpretation.": "Output: traceable evidence artifacts for thesis tables, figures, and interpretation.",
        "Input: channel-energy tensor, active-user mask, trained policy checkpoint": "Input: channel-state representation, active-user mask, trained policy representation",
        "1. Normalize input features using checkpoint metadata.": "1. Normalize state features using learned-policy metadata.",
        "2. Run the policy network to produce scheduling logits and power outputs.": "2. Evaluate the learned policy to produce scheduling preference and power-control outputs.",
        "5. Return ResourceDirectives to the PHY/MAC evaluation stage.": "5. Return scheduling and power directives to the PHY/MAC evaluation stage.",
        "6. Record BER, BER upper confidence, throughput, latency, power, and runtime.": "6. Interpret BER, BER upper confidence, throughput, latency, power, and runtime as policy evidence.",
    }
    for p in body.iter(qn("p")):
        text = paragraph_text(p)
        if text in replacements:
            set_paragraph_text(p, replacements[text])


def replace_publication_list(body: ET.Element, rels_root: ET.Element) -> None:
    elems: list[ET.Element] = [
        make_body_p("Publications and research outputs supplied for the thesis submission record:"),
    ]
    elems.extend(make_publication_p(title, venue, url, rels_root) for title, venue, url in PUBLICATIONS)
    elems.append(make_page_break_p())
    replace_between_headings(body, "List of Publications", "Chapter 1: Introduction", elems)


def replace_declaration_page(body: ET.Element) -> None:
    elems = [
        make_body_p(
            "I declare that this thesis is my own original work except where due acknowledgement is made in the text, tables, figures, references, and appendices. The work has been prepared for the degree of Doctor of Philosophy in Computing at Sunway University and has not been submitted, in whole or in part, for another academic award."
        ),
        make_body_p(
            "The simulation evidence, source artifacts, figures, and methodological descriptions used in this thesis are reported as research outputs of the Factory6G study. Where software frameworks, datasets, publications, or third-party materials are used, they are cited and interpreted within the stated research boundary."
        ),
        make_body_p("Candidate: Yahya S. M. Khamayseh"),
        make_body_p("Degree: Doctor of Philosophy in Computing"),
        make_body_p("Thesis title: Optimizing Cross-Layer 6G Networks in Smart Factories Using Machine Learning: Challenges and Solutions"),
        make_body_p("Candidate signature: ______________________________    Date: __________________"),
        make_body_p("Supervisor acknowledgement: ________________________    Date: __________________"),
        make_page_break_p(),
    ]
    replace_between_headings(body, "Original Literary Work Declaration", "Abstract", elems)


def clean_algorithm_blocks(body: ET.Element) -> None:
    alg1_rows = [
        ["Part", "Pseudocode"],
        ["Input", "Experimental design D; channel family C; estimator set E; resource-manager set M; Eb/N0 grid G; Monte Carlo stopping rule R."],
        ["Output", "Traceable evidence bundle B containing metrics, figures, source paths, confidence notes, and interpretation boundaries."],
        ["1", "Define the smart-factory operating assumption by fixing C, G, modulation, factory profile, and reproducibility seed policy."],
        ["2", "For each operating point in G, instantiate a shared channel/noise context so candidate methods are compared under equivalent evidence."],
        ["3", "Evaluate each estimator in E and record reliability, latency, throughput, runtime, and estimator-side claim boundaries."],
        ["4", "Translate PHY outcomes into feedback variables that can be read by the resource-management layer."],
        ["5", "Evaluate each policy in M under the same evidence family, including deterministic baselines and learned policies."],
        ["6", "Compute BER, BER upper confidence, latency, throughput, power, and runtime for each method and channel family."],
        ["7", "Interpret zero-observed-error cases through confidence bounds rather than treating them as zero true error probability."],
        ["8", "Archive B so that every thesis table, figure, and claim can be traced to its source artifact."],
    ]
    replace_table_after_caption(
        body,
        "Algorithm 1. Factory6G evidence-generation workflow",
        base.make_table(alg1_rows, [1700, BODY_WIDTH_PORTRAIT - 1700]),
    )

    alg2_rows = [
        ["Part", "Pseudocode"],
        ["Input", "State representation s_t; active-user mask u_t; trained policy representation pi_theta; scheduling and power constraints Omega."],
        ["Output", "Resource directive a_t specifying active users and bounded power allocation for the next PHY/MAC evaluation step."],
        ["1", "Receive s_t as a theoretical state summary of channel quality, user activity, and reliability-relevant feedback."],
        ["2", "Normalize and validate state features using the learned-policy metadata and the active-user mask u_t."],
        ["3", "Apply Omega so that unavailable users, invalid power values, and infeasible scheduling actions are excluded."],
        ["4", "Evaluate pi_theta(s_t) to obtain scheduling preference and power-control signals."],
        ["5", "Construct a_t by selecting valid users and normalizing the power directive within the configured simulation bounds."],
        ["6", "Return a_t to the PHY/MAC evidence loop and observe BER, BER upper confidence, latency, throughput, power, and runtime."],
        ["7", "Read the learned decision against deterministic baselines; treat superiority as channel- and evidence-bound, not universal."],
    ]
    replace_table_after_caption(
        body,
        "Algorithm 2. BER-oriented resource-manager inference",
        base.make_table(alg2_rows, [1700, BODY_WIDTH_PORTRAIT - 1700]),
    )

    alg3_rows = [
        ["Stage", "Docker-first reproducibility sequence"],
        ["Prepare", "Build or refresh the simulation container before regenerating thesis evidence."],
        ["Run evidence", "Execute configured Factory6G experiments through Docker Compose with explicit channel, modulation, factory-size, and policy selections."],
        ["Regenerate reports", "Run report-generation utilities through the container so plots and summary tables are derived from stored artifacts."],
        ["Validate", "Run the focused policy-pipeline tests in the same container before treating regenerated outputs as thesis evidence."],
        ["Record", "Preserve commands, configuration, result paths, and generated figures in the audit trail."],
    ]
    replace_table_after_caption(
        body,
        "Algorithm 3. Representative Docker commands",
        base.make_table(alg3_rows, [3200, BODY_WIDTH_LANDSCAPE - 3200]),
    )
    for p in body.iter(qn("p")):
        if paragraph_text(p) == "Algorithm 3. Representative Docker commands":
            set_paragraph_text(p, "Algorithm 3. Docker-first reproducibility sequence", bold=True, italic=True)


def renumber_captions(body: ET.Element) -> tuple[list[tuple[str, str, ET.Element]], list[tuple[str, str, ET.Element]]]:
    figures: list[tuple[str, str, ET.Element]] = []
    tables: list[tuple[str, str, ET.Element]] = []
    fig_no = 0
    tbl_no = 0
    for el in children(body):
        if not is_p(el) or p_style(el) != "Caption":
            continue
        text = paragraph_text(el)
        fig_match = re.match(r"^Figure\s+\d+\.\s*(.+)$", text)
        table_match = re.match(r"^Table\s+\d+\.\s*(.+)$", text)
        if fig_match:
            fig_no += 1
            caption = fig_match.group(1)
            new_text = f"Figure {fig_no}. {caption}"
            set_paragraph_text(el, new_text, bold=True, italic=True)
            figures.append((f"Figure {fig_no}", caption, el))
        elif table_match:
            tbl_no += 1
            caption = table_match.group(1)
            new_text = f"Table {tbl_no}. {caption}"
            set_paragraph_text(el, new_text, bold=True, italic=True)
            tables.append((f"Table {tbl_no}", caption, el))
    return figures, tables


def collect_captions(body: ET.Element, prefix: str) -> list[tuple[str, str, ET.Element]]:
    out = []
    for el in children(body):
        if not is_p(el) or p_style(el) != "Caption":
            continue
        text = paragraph_text(el)
        match = re.match(rf"^({prefix}\s+\d+)\.\s*(.+)$", text)
        if match:
            out.append((match.group(1), match.group(2), el))
    return out


def rebuild_appendix_tables(body: ET.Element, figures: list[tuple[str, str, ET.Element]]) -> None:
    source_rows = [
        ["Path", "Use"],
        ["system_design/topology_3d_factory.png", "Embedded as Figure 1"],
        ["system_design/topology_mobility.png", "Embedded as Figure 2"],
        ["system_design/phy_mac_topology.png", "Embedded as Figure 3"],
        ["thesis_writing/generated_revision/factory6g_framework_diagram.png", "Embedded as Figure 4"],
        ["thesis_writing/generated_revision/factory6g_evidence_workflow.png", "Embedded as Figure 5"],
        ["thesis_writing/generated_revision/ber_drl_inference_loop.png", "Embedded as Figure 6"],
        ["thesis_writing/generated_revision/excalidraw/factory6g_framework.excalidraw", "Editable Excalidraw source for Figure 4"],
        ["thesis_writing/generated_revision/excalidraw/evidence_workflow.excalidraw", "Editable Excalidraw source for Figure 5"],
        ["thesis_writing/generated_revision/excalidraw/ber_drl_loop.excalidraw", "Editable Excalidraw source for Figure 6"],
        ["data/README.md", "Dataset schema and factory profile source"],
        ["config/factory_size_profiles.json", "Factory profile definitions"],
        [
            "results/20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s/summary_v2.csv",
            "May 23 cross-channel resource-manager run summary",
        ],
        [
            "results/20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s/summary_v2.json",
            "May 23 cross-channel resource-manager run metadata",
        ],
        [
            "results/20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s/rayleigh/resource_managers/stage_results_v2.json",
            "Primary rayleigh resource-manager stage metrics",
        ],
        [
            "results/20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s/rician/resource_managers/stage_results_v2.json",
            "Primary rician resource-manager stage metrics",
        ],
        [
            "results/20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s/tr38901/resource_managers/stage_results_v2.json",
            "Primary tr38901 resource-manager stage metrics",
        ],
        ["reports/weekly/2026-05-23/resource_manager_ber_comparison.csv", "Primary May 23 cross-channel resource-manager summary table"],
    ]
    for idx, src in enumerate(FIGURE_SOURCES, start=1):
        if idx <= 6:
            continue
        source_rows.append([src, f"Embedded as Figure {idx}"])
    replace_table_after_caption(body, "Source artifacts reused in the thesis", base.make_table(source_rows, [7600, 6358]))

    figure_rows = [["Figure", "Caption", "Source path"]]
    for i, (label, caption, _) in enumerate(figures, start=1):
        source = FIGURE_SOURCES[i - 1] if i - 1 < len(FIGURE_SOURCES) else ""
        figure_rows.append([label, caption, source])
    replace_table_after_caption(body, "Embedded figure inventory", base.make_table(figure_rows, [1800, 6500, 5658]))


def rebuild_algorithm_inventory(body: ET.Element) -> None:
    rows = [
        ["Algorithm", "Caption"],
        ["Algorithm 1", "Factory6G evidence-generation workflow"],
        ["Algorithm 2", "BER-oriented resource-manager inference"],
        ["Algorithm 3", "Docker-first reproducibility sequence"],
    ]
    replace_table_after_caption(body, "Embedded algorithm inventory", base.make_table(rows, [2800, BODY_WIDTH_LANDSCAPE - 2800]))


def add_caption_and_heading_bookmarks(body: ET.Element, root: ET.Element) -> tuple[list[tuple[int, str, str]], list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    next_id = max_bookmark_id(root) + 1
    headings: list[tuple[int, str, str]] = []
    figures: list[tuple[str, str, str]] = []
    tables: list[tuple[str, str, str]] = []

    for el in children(body):
        if not is_p(el):
            continue
        text = paragraph_text(el)
        style = p_style(el)
        if style.startswith("Heading") and text:
            level_match = re.search(r"(\d+)$", style)
            level = int(level_match.group(1)) if level_match else 1
            if text == "Table of Contents":
                continue
            name = f"toc_h_{len(headings) + 1:03d}"
            add_bookmark_to_paragraph(el, name, next_id)
            next_id += 1
            headings.append((level, text, name))
        elif style == "Caption":
            fig_match = re.match(r"^(Figure\s+\d+)\.\s*(.+)$", text)
            tbl_match = re.match(r"^(Table\s+\d+)\.\s*(.+)$", text)
            if fig_match:
                name = f"fig_{len(figures) + 1:03d}"
                add_bookmark_to_paragraph(el, name, next_id)
                next_id += 1
                figures.append((fig_match.group(1), fig_match.group(2), name))
            elif tbl_match:
                name = f"tbl_{len(tables) + 1:03d}"
                add_bookmark_to_paragraph(el, name, next_id)
                next_id += 1
                tables.append((tbl_match.group(1), tbl_match.group(2), name))
    return headings, figures, tables


def fill_front_matter_lists(body: ET.Element, headings: list[tuple[int, str, str]], figures: list[tuple[str, str, str]], tables: list[tuple[str, str, str]]) -> None:
    toc_entries = [
        make_list_entry(text, bookmark, min(max(level - 1, 0), 2))
        for level, text, bookmark in headings
    ]
    figure_entries = [
        make_list_entry(f"{label}. {caption}", bookmark, 0)
        for label, caption, bookmark in figures
    ]
    table_entries = [
        make_list_entry(f"{label}. {caption}", bookmark, 0)
        for label, caption, bookmark in tables
    ]
    replace_between_headings(body, "Table of Contents", "List of Figures", toc_entries)
    replace_between_headings(body, "List of Figures", "List of Tables", figure_entries)
    replace_between_headings(body, "List of Tables", "List of Symbols and Abbreviations", table_entries)


def add_section_breaks(body: ET.Element) -> None:
    section_plan = [
        ("Appendix A: Reproducibility Commands", False),
        ("Appendix D: Result Interpretation Notes", True),
        ("Appendix E: Selected Configuration and Dataset Schema", False),
        ("Appendix F: Extended Methodological Commentary", True),
        ("References", False),
    ]
    for heading, previous_landscape in section_plan:
        end_section_before_heading(body, heading, previous_landscape=previous_landscape)
    set_body_section(body, landscape=False)


def update_audit(figures: list[tuple[str, str, ET.Element]]) -> None:
    if not AUDIT_MD.exists():
        return
    lines = AUDIT_MD.read_text().splitlines()
    out: list[str] = []
    in_artifacts = False
    for line in lines:
        if line == "## Reused Evidence Artifacts":
            in_artifacts = True
            out.append(line)
            out.extend(
                [
                    "- `system_design/topology_3d_factory.png` - Embedded as Figure 1",
                    "- `system_design/topology_mobility.png` - Embedded as Figure 2",
                    "- `system_design/phy_mac_topology.png` - Embedded as Figure 3",
                    "- `thesis_writing/generated_revision/factory6g_framework_diagram.png` - Embedded as Figure 4",
                    "- `thesis_writing/generated_revision/factory6g_evidence_workflow.png` - Embedded as Figure 5",
                    "- `thesis_writing/generated_revision/ber_drl_inference_loop.png` - Embedded as Figure 6",
                    "- `thesis_writing/generated_revision/excalidraw/factory6g_framework.excalidraw` - Editable Excalidraw source for Figure 4",
                    "- `thesis_writing/generated_revision/excalidraw/evidence_workflow.excalidraw` - Editable Excalidraw source for Figure 5",
                    "- `thesis_writing/generated_revision/excalidraw/ber_drl_loop.excalidraw` - Editable Excalidraw source for Figure 6",
                    "- `data/README.md` - Dataset schema and factory profile source",
                    "- `config/factory_size_profiles.json` - Factory profile definitions",
                    "- `results/20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s/summary_v2.csv` - May 23 cross-channel resource-manager run summary",
                    "- `results/20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s/summary_v2.json` - May 23 cross-channel resource-manager run metadata",
                    "- `results/20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s/rayleigh/resource_managers/stage_results_v2.json` - Primary rayleigh resource-manager stage metrics",
                    "- `results/20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s/rician/resource_managers/stage_results_v2.json` - Primary rician resource-manager stage metrics",
                    "- `results/20260523_173452_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_ber_drl_rayleigh_rician_umi_qpsk_s/tr38901/resource_managers/stage_results_v2.json` - Primary tr38901 resource-manager stage metrics",
                    "- `reports/weekly/2026-05-23/resource_manager_ber_comparison.csv` - Primary May 23 cross-channel resource-manager summary table",
                ]
            )
            for idx, source in enumerate(FIGURE_SOURCES[6:], start=7):
                out.append(f"- `{source}` - Embedded as Figure {idx}")
            out.append("- `thesis_writing/Factory6G-v3-Completed-From-Current-Progress.html` - Reference list extracted from the existing v3 thesis HTML")
            continue
        if in_artifacts:
            if line.startswith("## "):
                in_artifacts = False
                out.append("")
                out.append(line)
            continue
        if "Unresolved manual check: confirm the official Sunway declaration form" in line:
            out.append("- Administrative note: the revised document now contains an unsigned original-work declaration page and the supplied publication list; institution-specific signed-form checks remain outside the automated builder.")
        elif "update Word/LibreOffice fields for TOC/list of figures/list of tables" in line:
            out.append("- Front matter placeholders were replaced with Word-updatable table of contents, list of figures, and list of tables entries.")
        else:
            out.append(line)
    AUDIT_MD.write_text("\n".join(out) + "\n")


def write_docx(tmp_dir: Path, out_docx: Path) -> None:
    if out_docx.exists():
        out_docx.unlink()
    with zipfile.ZipFile(out_docx, "w", zipfile.ZIP_DEFLATED) as zout:
        for path in sorted(tmp_dir.rglob("*")):
            if path.is_file():
                zout.write(path, path.relative_to(tmp_dir).as_posix())


def main() -> None:
    if not SRC_DOCX.exists():
        raise SystemExit(f"Missing source DOCX: {SRC_DOCX}")
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True)
    with zipfile.ZipFile(SRC_DOCX) as zin:
        zin.extractall(TMP_DIR)

    diagrams = generate_diagrams()
    ensure_png_content_type(TMP_DIR)

    document_path = TMP_DIR / "word" / "document.xml"
    rels_path = TMP_DIR / "word" / "_rels" / "document.xml.rels"
    document_tree = ET.parse(document_path)
    rels_tree = ET.parse(rels_path)
    document_root = document_tree.getroot()
    rels_root = rels_tree.getroot()
    body = document_root.find("w:body", NS)
    if body is None:
        raise SystemExit("No w:body found")

    replace_between_headings(body, "Table of Contents", "List of Figures", [])
    replace_between_headings(body, "List of Figures", "List of Tables", [])
    replace_between_headings(body, "List of Tables", "List of Symbols and Abbreviations", [])

    replace_declaration_page(body)
    insert_chapter3_visuals(body, TMP_DIR, rels_root, diagrams)
    clean_algorithm_blocks(body)
    add_results_interpretation(body)
    academicize_existing_language(body)
    replace_publication_list(body, rels_root)

    figures, _ = renumber_captions(body)
    rebuild_appendix_tables(body, figures)
    rebuild_algorithm_inventory(body)
    figures, _ = renumber_captions(body)

    add_section_breaks(body)
    widen_appendix_tables(body)

    headings, figure_refs, table_refs = add_caption_and_heading_bookmarks(body, document_root)
    fill_front_matter_lists(body, headings, figure_refs, table_refs)
    renumber_drawing_ids(document_root)
    remove_ignorable_attr(document_root)

    document_tree.write(document_path, encoding="utf-8", xml_declaration=True)
    ET.register_namespace("", REL)
    rels_tree.write(rels_path, encoding="utf-8", xml_declaration=True)
    set_update_fields(TMP_DIR / "word" / "settings.xml")
    write_docx(TMP_DIR, OUT_DOCX)
    update_audit(figures)

    print(f"Wrote {OUT_DOCX}")
    print(f"Generated diagrams: {', '.join(str(p) for p in diagrams.values())}")
    print(f"Headings in TOC: {len(headings)}; figures: {len(figure_refs)}; tables: {len(table_refs)}")


if __name__ == "__main__":
    main()
