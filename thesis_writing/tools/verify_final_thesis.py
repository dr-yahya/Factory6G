from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
THESIS_DIR = ROOT / "thesis_writing"
DOCX = THESIS_DIR / "Factory6G-final-thesis.docx"
PDF = THESIS_DIR / "rendered_final/Factory6G-final-thesis.pdf"
AUDIT = THESIS_DIR / "Factory6G-final-thesis-source-audit.md"


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def text_from_docx_xml(xml: str) -> str:
    xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    return xml


def count_pdf_pages(path: Path) -> int:
    data = path.read_bytes()
    return len(re.findall(rb"/Type\s*/Page(?!s)", data))


def main() -> None:
    for path in (DOCX, AUDIT):
        if not path.exists():
            fail(f"missing required artifact: {path}")

    with zipfile.ZipFile(DOCX) as archive:
        names = archive.namelist()
        if "word/document.xml" not in names:
            fail("DOCX does not contain word/document.xml")
        media = [name for name in names if name.startswith("word/media/")]
        xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")

    text = text_from_docx_xml(xml)
    figure_count = len(re.findall(r"\bFigure\s+\d+\.", text))
    table_count = len(re.findall(r"\bTable\s+\d+\.", text))
    chapter_count = len(re.findall(r"Chapter\s+[1-6]:", text))

    if chapter_count < 6:
        fail(f"expected 6 thesis chapters, found {chapter_count}")
    if figure_count < 20:
        fail(f"expected at least 20 embedded figure captions, found {figure_count}")
    if table_count < 12:
        fail(f"expected at least 12 table captions, found {table_count}")
    if len(media) < figure_count:
        fail(f"expected DOCX media entries to cover figures; media={len(media)}, figures={figure_count}")

    audit_text = AUDIT.read_text(errors="ignore")
    if "MISSING figure requested but not found" in audit_text:
        fail("audit note reports a missing figure")
    if "BER-DRL is presented as competitive" not in audit_text:
        fail("audit note is missing BER-DRL claim boundary")
    if "No new fabricated DOI" not in audit_text:
        fail("audit note is missing citation fabrication boundary")

    pdf_pages = None
    if PDF.exists():
        pdf_pages = count_pdf_pages(PDF)
        if pdf_pages < 90:
            fail(f"rendered PDF looks too short: {pdf_pages} pages")
        if pdf_pages > 130:
            fail(f"rendered PDF exceeds intended range: {pdf_pages} pages")

    print("Factory6G final thesis verification")
    print(f"DOCX: {DOCX} ({DOCX.stat().st_size} bytes)")
    print(f"Figures: {figure_count}; tables: {table_count}; media files: {len(media)}")
    if pdf_pages is None:
        print("PDF: not present, render step still required")
    else:
        print(f"PDF: {PDF} ({pdf_pages} pages, {PDF.stat().st_size} bytes)")
    print(f"Audit: {AUDIT} ({AUDIT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
