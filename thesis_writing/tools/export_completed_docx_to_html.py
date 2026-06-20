#!/usr/bin/env python3
from __future__ import annotations

import html
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path("/app")
IN_DOCX = ROOT / "thesis_writing" / "Factory6G-v3-Completed-From-Current-Progress.docx"
OUT_HTML = ROOT / "thesis_writing" / "Factory6G-v3-Completed-From-Current-Progress.html"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}


def qn(tag: str) -> str:
    return f"{{{W}}}{tag}"


def paragraph_text(p: ET.Element) -> str:
    return "".join(t.text or "" for t in p.iter(qn("t"))).strip()


def p_style(p: ET.Element) -> str:
    style = p.find("./w:pPr/w:pStyle", NS)
    return style.get(qn("val"), "") if style is not None else ""


def table_html(tbl: ET.Element) -> str:
    rows = []
    for tr in tbl.findall("./w:tr", NS):
        cells = []
        for tc in tr.findall("./w:tc", NS):
            parts = [paragraph_text(p) for p in tc.findall(".//w:p", NS)]
            cells.append("<td>" + "<br/>".join(html.escape(p) for p in parts if p) + "</td>")
        if cells:
            rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table>" + "".join(rows) + "</table>"


def main() -> int:
    with zipfile.ZipFile(IN_DOCX) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    body = root.find(qn("body"))
    if body is None:
        raise SystemExit("No document body")

    out = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'/>",
        "<style>",
        "@page { size: Letter; margin: 1in; }",
        "body { font-family: Calibri, Arial, sans-serif; font-size: 11pt; line-height: 1.35; color: #111; }",
        "h1 { color: #1f4e79; font-size: 18pt; margin-top: 22pt; page-break-before: always; }",
        "h1:first-of-type { page-break-before: auto; }",
        "h2 { color: #2f75b5; font-size: 14pt; margin-top: 15pt; }",
        "h3 { color: #1f4e79; font-size: 12pt; margin-top: 12pt; }",
        "p { text-align: justify; margin: 0 0 8pt 0; }",
        ".caption { text-align: center; color: #1f4e79; font-style: italic; margin-top: 8pt; }",
        "table { border-collapse: collapse; width: 100%; margin: 8pt 0 12pt 0; page-break-inside: avoid; }",
        "td { border: 1px solid #bfbfbf; padding: 5pt; vertical-align: top; font-size: 9.5pt; }",
        "tr:first-child td { background: #d9eaf7; font-weight: bold; }",
        "code { color: #333; }",
        "</style></head><body>",
    ]

    for child in list(body):
        if child.tag == qn("p"):
            text = paragraph_text(child)
            if not text:
                continue
            esc = html.escape(text).replace("`", "")
            style = p_style(child)
            if text.startswith("Figure source: "):
                rel_path = text.replace("Figure source:", "", 1).strip().strip("`.")
                image_path = ROOT / rel_path
                if image_path.exists():
                    out.append(
                        f"<p style='text-align:center'><img src='{html.escape(str(image_path))}' style='max-width:6.1in; max-height:4.3in;'/></p>"
                    )
                    out.append(f"<p class='caption'>Figure source: {html.escape(rel_path)}</p>")
                continue
            if style == "Heading1":
                out.append(f"<h1>{esc}</h1>")
            elif style == "Heading2":
                out.append(f"<h2>{esc}</h2>")
            elif style == "Heading3":
                out.append(f"<h3>{esc}</h3>")
            elif style == "Caption":
                out.append(f"<p class='caption'>{esc}</p>")
            else:
                out.append(f"<p>{esc}</p>")
        elif child.tag == qn("tbl"):
            out.append(table_html(child))

    out.append("</body></html>")
    OUT_HTML.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {OUT_HTML}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
