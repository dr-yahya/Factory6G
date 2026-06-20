#!/usr/bin/env python3
"""Replace front-matter PAGEREF entries with rendered static page numbers.

The thesis builder first creates Word PAGEREF fields so LibreOffice/Word can
compute pagination. This script consumes the rendered front-matter text and
materializes the visible page numbers in the DOCX so users do not see stale
fallback values such as "0" before refreshing fields manually.
"""

from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS}
BODY_WIDTH_PORTRAIT = 9360


def qn(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def xml_qn(tag: str) -> str:
    return f"{{{XML_NS}}}{tag}"


def paragraph_text(p: ET.Element) -> str:
    return "".join(t.text or "" for t in p.findall(".//w:t", NS))


def is_p(el: ET.Element) -> bool:
    return el.tag == qn("p")


def clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.replace("\x0c", "").strip())


def section_lines(lines: list[str], start: str, end: str) -> list[str]:
    start_idx = None
    for idx, line in enumerate(lines):
        if clean_line(line) == start:
            start_idx = idx
            break
    if start_idx is None:
        raise ValueError(f"Could not find front-matter section: {start}")
    for idx in range(start_idx + 1, len(lines)):
        if clean_line(lines[idx]) == end:
            return lines[start_idx + 1 : idx]
    raise ValueError(f"Could not find front-matter section end: {end}")


def parse_rendered_entries(lines: list[str]) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    buffer: list[str] = []
    for raw in lines:
        line = clean_line(raw)
        if not line:
            continue
        buffer.append(line)
        if not re.search(r"\.{3,}\s*\d+\s*$", line):
            continue
        joined = " ".join(buffer)
        match = re.match(r"^(.*?)\.{3,}\s*(\d+)\s*$", joined)
        if not match:
            raise ValueError(f"Could not parse rendered front-matter entry: {joined}")
        label = re.sub(r"\s+", " ", match.group(1)).strip()
        entries.append((label, match.group(2)))
        buffer = []
    if buffer:
        raise ValueError(f"Unparsed front-matter lines remain: {' '.join(buffer)}")
    return entries


def parse_frontmatter_text(path: Path) -> dict[str, list[tuple[str, str]]]:
    lines = path.read_text(errors="replace").splitlines()
    return {
        "toc": parse_rendered_entries(section_lines(lines, "Table of Contents", "List of Figures")),
        "figures": parse_rendered_entries(section_lines(lines, "List of Figures", "List of Tables")),
        "tables": parse_rendered_entries(section_lines(lines, "List of Tables", "List of Symbols and Abbreviations")),
    }


def find_direct_index(body: ET.Element, text: str) -> int:
    for idx, el in enumerate(list(body)):
        if is_p(el) and paragraph_text(el) == text:
            return idx
    raise ValueError(f"Could not find paragraph: {text}")


def range_count(body: ET.Element, start: str, end: str) -> int:
    elems = list(body)
    start_idx = find_direct_index(body, start)
    end_idx = find_direct_index(body, end)
    return sum(1 for el in elems[start_idx + 1 : end_idx] if is_p(el))


def toc_level(label: str) -> int:
    if re.match(r"^\d+\.\d+\.\d+\b", label):
        return 2
    if re.match(r"^\d+\.\d+\b", label):
        return 1
    if re.match(r"^[A-Z]\.\d+\b", label):
        return 1
    return 0


def make_static_entry(text: str, page: str, *, level: int = 0) -> ET.Element:
    p = ET.Element(qn("p"))
    ppr = ET.SubElement(p, qn("pPr"))
    tabs = ET.SubElement(ppr, qn("tabs"))
    ET.SubElement(
        tabs,
        qn("tab"),
        {
            qn("val"): "right",
            qn("leader"): "dot",
            qn("pos"): str(BODY_WIDTH_PORTRAIT),
        },
    )
    if level:
        ET.SubElement(ppr, qn("ind"), {qn("left"): str(level * 360)})
    ET.SubElement(ppr, qn("spacing"), {qn("after"): "40", qn("line"): "240", qn("lineRule"): "auto"})

    r_text = ET.SubElement(p, qn("r"))
    t = ET.SubElement(r_text, qn("t"), {xml_qn("space"): "preserve"})
    t.text = text
    r_tab = ET.SubElement(p, qn("r"))
    ET.SubElement(r_tab, qn("tab"))
    r_page = ET.SubElement(p, qn("r"))
    t_page = ET.SubElement(r_page, qn("t"))
    t_page.text = page
    return p


def replace_between_headings(body: ET.Element, start: str, end: str, new_elems: list[ET.Element]) -> None:
    elems = list(body)
    start_idx = find_direct_index(body, start)
    end_idx = find_direct_index(body, end)
    body[:] = elems[: start_idx + 1] + new_elems + elems[end_idx:]


def materialize(document_xml: Path, rendered_entries: dict[str, list[tuple[str, str]]]) -> tuple[int, int, int]:
    tree = ET.parse(document_xml)
    root = tree.getroot()
    body = root.find("w:body", NS)
    if body is None:
        raise ValueError("No w:body found")

    expected = {
        "toc": range_count(body, "Table of Contents", "List of Figures"),
        "figures": range_count(body, "List of Figures", "List of Tables"),
        "tables": range_count(body, "List of Tables", "List of Symbols and Abbreviations"),
    }
    actual = {key: len(value) for key, value in rendered_entries.items()}
    if expected != actual:
        raise ValueError(f"Rendered front-matter counts do not match DOCX ranges: expected {expected}, actual {actual}")

    toc_entries = [
        make_static_entry(label, page, level=toc_level(label))
        for label, page in rendered_entries["toc"]
    ]
    figure_entries = [
        make_static_entry(label, page)
        for label, page in rendered_entries["figures"]
    ]
    table_entries = [
        make_static_entry(label, page)
        for label, page in rendered_entries["tables"]
    ]

    replace_between_headings(body, "Table of Contents", "List of Figures", toc_entries)
    replace_between_headings(body, "List of Figures", "List of Tables", figure_entries)
    replace_between_headings(body, "List of Tables", "List of Symbols and Abbreviations", table_entries)

    ET.register_namespace("w", W_NS)
    ET.register_namespace("xml", XML_NS)
    tree.write(document_xml, encoding="utf-8", xml_declaration=True)
    return len(toc_entries), len(figure_entries), len(table_entries)


def write_docx(tmp_dir: Path, out_docx: Path) -> None:
    target = out_docx
    if out_docx.exists():
        target = out_docx.with_suffix(out_docx.suffix + ".tmp")
        if target.exists():
            target.unlink()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zout:
        for path in sorted(tmp_dir.rglob("*")):
            if path.is_file():
                zout.write(path, path.relative_to(tmp_dir).as_posix())
    if target != out_docx:
        target.replace(out_docx)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docx", required=True, type=Path)
    parser.add_argument("--frontmatter-text", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if not args.docx.exists():
        raise SystemExit(f"Missing DOCX: {args.docx}")
    if not args.frontmatter_text.exists():
        raise SystemExit(f"Missing front-matter text: {args.frontmatter_text}")

    tmp_dir = args.out.parent / ".frontmatter_materialize_docx"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    with zipfile.ZipFile(args.docx) as zin:
        zin.extractall(tmp_dir)

    counts = materialize(tmp_dir / "word" / "document.xml", parse_frontmatter_text(args.frontmatter_text))
    write_docx(tmp_dir, args.out)
    print(f"Wrote {args.out}")
    print(f"Materialized TOC/list entries: TOC={counts[0]}, figures={counts[1]}, tables={counts[2]}")


if __name__ == "__main__":
    main()
