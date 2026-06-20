from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[2]
THESIS_DIR = ROOT / "thesis_writing"
DOCX = THESIS_DIR / "Factory6G-final-thesis.docx"
RENDER_DIR = THESIS_DIR / "rendered_final"
PDF = RENDER_DIR / "Factory6G-final-thesis.pdf"


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def count_pdf_pages(path: Path) -> int:
    data = path.read_bytes()
    return len(re.findall(rb"/Type\s*/Page(?!s)", data))


def make_contact_sheets(page_paths: list[Path], pages_per_sheet: int = 54) -> None:
    for old in RENDER_DIR.glob("contact-sheet-*.png"):
        old.unlink()

    thumb_width = 230
    label_height = 18
    padding = 10
    columns = 6

    for sheet_index, start in enumerate(range(0, len(page_paths), pages_per_sheet), start=1):
        subset = page_paths[start : start + pages_per_sheet]
        thumbs: list[Image.Image] = []
        for path in subset:
            image = Image.open(path).convert("RGB")
            image.thumbnail((thumb_width, 340), Image.Resampling.LANCZOS)
            framed = ImageOps.expand(image, border=1, fill=(210, 210, 210))
            thumbs.append(framed)

        rows = (len(thumbs) + columns - 1) // columns
        cell_width = thumb_width + 2 * padding
        cell_height = 360 + label_height + 2 * padding
        sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
        draw = ImageDraw.Draw(sheet)

        for idx, thumb in enumerate(thumbs):
            row, col = divmod(idx, columns)
            x = col * cell_width + padding
            y = row * cell_height + padding
            sheet.paste(thumb, (x, y + label_height))
            label = f"p.{start + idx + 1:03d}"
            draw.text((x, y), label, fill=(60, 60, 60))

        out_path = RENDER_DIR / f"contact-sheet-{sheet_index}.png"
        sheet.save(out_path)
        print(f"Wrote {out_path}")


def main() -> None:
    if not DOCX.exists():
        raise SystemExit(f"Missing DOCX: {DOCX}")

    soffice = shutil.which("libreoffice") or shutil.which("soffice")
    pdftoppm = shutil.which("pdftoppm")
    if not soffice:
        raise SystemExit("LibreOffice/soffice is not available in this Docker image.")
    if not pdftoppm:
        raise SystemExit("pdftoppm is not available in this Docker image.")

    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    for path in [PDF, *RENDER_DIR.glob("page-*.png")]:
        path.unlink(missing_ok=True)

    run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(RENDER_DIR), str(DOCX)])
    if not PDF.exists():
        raise SystemExit(f"Expected rendered PDF was not created: {PDF}")

    page_count = count_pdf_pages(PDF)
    print(f"Rendered PDF pages: {page_count}")
    if not 90 <= page_count <= 130:
        raise SystemExit(f"Rendered page count outside expected range: {page_count}")

    prefix = RENDER_DIR / "page"
    run([pdftoppm, "-png", "-r", "80", str(PDF), str(prefix)])
    page_paths = sorted(RENDER_DIR.glob("page-*.png"))
    if len(page_paths) != page_count:
        raise SystemExit(f"Expected {page_count} rendered page images, found {len(page_paths)}")
    make_contact_sheets(page_paths)


if __name__ == "__main__":
    main()
