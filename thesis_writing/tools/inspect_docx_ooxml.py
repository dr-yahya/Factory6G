#!/usr/bin/env python3
"""Inspect a thesis DOCX at the OOXML level using only the Python stdlib."""

from __future__ import annotations

import argparse
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
W = NS["w"]


def qn(tag: str) -> str:
    prefix, local = tag.split(":", 1)
    return f"{{{NS[prefix]}}}{local}"


def text_of(el: ET.Element) -> str:
    return "".join(t.text or "" for t in el.findall(".//w:t", NS))


def style_of_paragraph(p: ET.Element) -> str:
    style = p.find("./w:pPr/w:pStyle", NS)
    return style.get(qn("w:val"), "") if style is not None else ""


def section_page_size(sect_pr: ET.Element | None) -> str:
    if sect_pr is None:
        return ""
    pg_sz = sect_pr.find("./w:pgSz", NS)
    pg_mar = sect_pr.find("./w:pgMar", NS)
    if pg_sz is None:
        return ""
    attrs = {
        "w": pg_sz.get(qn("w:w")),
        "h": pg_sz.get(qn("w:h")),
        "orient": pg_sz.get(qn("w:orient"), "portrait"),
    }
    if pg_mar is not None:
        attrs.update(
            {
                "left": pg_mar.get(qn("w:left")),
                "right": pg_mar.get(qn("w:right")),
                "top": pg_mar.get(qn("w:top")),
                "bottom": pg_mar.get(qn("w:bottom")),
            }
        )
    return " ".join(f"{k}={v}" for k, v in attrs.items() if v is not None)


def table_width(tbl: ET.Element) -> str:
    tbl_w = tbl.find("./w:tblPr/w:tblW", NS)
    grid_cols = [
        col.get(qn("w:w"), "")
        for col in tbl.findall("./w:tblGrid/w:gridCol", NS)
    ]
    rows = tbl.findall("./w:tr", NS)
    first_cells = rows[0].findall("./w:tc", NS) if rows else []
    cell_ws = []
    for cell in first_cells:
        tc_w = cell.find("./w:tcPr/w:tcW", NS)
        cell_ws.append(tc_w.get(qn("w:w"), "") if tc_w is not None else "")
    return (
        f"tblW={tbl_w.get(qn('w:w'), '') if tbl_w is not None else ''} "
        f"grid={','.join(grid_cols)} cells={','.join(cell_ws)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("docx", type=Path)
    parser.add_argument("--first", type=int, default=220)
    parser.add_argument("--around", action="append", default=[])
    parser.add_argument("--tables", action="store_true")
    parser.add_argument("--captions", action="store_true")
    parser.add_argument("--grep", action="append", default=[])
    args = parser.parse_args()

    with zipfile.ZipFile(args.docx) as zf:
        document_xml = zf.read("word/document.xml")

    root = ET.fromstring(document_xml)
    body = root.find("w:body", NS)
    if body is None:
        raise SystemExit("No document body found")

    body_items: list[tuple[int, str, str, str]] = []
    style_counts: Counter[str] = Counter()
    table_count = 0
    figure_paras = 0
    p_sect_count = 0

    for idx, child in enumerate(list(body)):
        tag = child.tag.split("}", 1)[-1]
        if tag == "p":
            text = text_of(child).strip()
            style = style_of_paragraph(child)
            style_counts[style or "(none)"] += 1
            if child.findall(".//w:drawing", NS) or child.findall(".//w:pict", NS):
                figure_paras += 1
            sect_pr = child.find("./w:pPr/w:sectPr", NS)
            if sect_pr is not None:
                p_sect_count += 1
                text = (text + " " if text else "") + f"[SECT {section_page_size(sect_pr)}]"
            if text:
                body_items.append((idx, "p", style, text.replace("\n", " ")[:260]))
        elif tag == "tbl":
            table_count += 1
            first_text = text_of(child).strip().replace("\n", " ")[:180]
            body_items.append((idx, "tbl", "", f"{table_width(child)} | {first_text}"))

    body_sect = body.find("w:sectPr", NS)
    print(
        "SUMMARY",
        f"body_items={len(list(body))}",
        f"visible_items={len(body_items)}",
        f"tables={table_count}",
        f"figure_paragraphs={figure_paras}",
        f"p_sections={p_sect_count}",
        f"body_section={section_page_size(body_sect)}",
    )
    print("STYLES")
    for style, count in style_counts.most_common(35):
        print(f"{count:4d} {style}")

    print("FIRST_ITEMS")
    for idx, kind, style, text in body_items[: args.first]:
        print(f"{idx:04d} {kind:<3} [{style}] {text}")

    for needle in args.around:
        print(f"AROUND {needle!r}")
        hits = [
            i
            for i, (_, _, _, text) in enumerate(body_items)
            if needle.lower() in text.lower()
        ]
        for hit in hits[:8]:
            for idx, kind, style, text in body_items[max(0, hit - 6) : hit + 12]:
                print(f"{idx:04d} {kind:<3} [{style}] {text}")
            print("---")

    if args.tables:
        print("TABLES")
        for idx, kind, style, text in body_items:
            if kind == "tbl":
                print(f"{idx:04d} {text}")

    if args.captions:
        print("CAPTIONS")
        for idx, kind, style, text in body_items:
            if style == "Caption" or text.startswith(("Figure ", "Table ", "Algorithm ")):
                print(f"{idx:04d} [{style}] {text}")

    for needle in args.grep:
        print(f"GREP {needle!r}")
        for idx, kind, style, text in body_items:
            if needle.lower() in text.lower():
                print(f"{idx:04d} {kind:<3} [{style}] {text}")


if __name__ == "__main__":
    main()
