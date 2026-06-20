#!/usr/bin/env python3
"""Build contact sheets for rendered DOCX page PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("render_dir", type=Path)
    parser.add_argument("--pages-per-sheet", type=int, default=40)
    args = parser.parse_args()

    pages = sorted(args.render_dir.glob("page-*.png"))
    if not pages:
        raise SystemExit(f"No page PNGs found in {args.render_dir}")
    for old in args.render_dir.glob("contact-sheet-*.png"):
        old.unlink()

    thumb_w = 185
    thumb_h = 260
    label_h = 22
    pad = 10
    cols = 5
    cell_w = thumb_w + 2 * pad
    cell_h = thumb_h + label_h + 2 * pad

    for sheet_idx, start in enumerate(range(0, len(pages), args.pages_per_sheet), start=1):
        subset = pages[start : start + args.pages_per_sheet]
        rows = (len(subset) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
        draw = ImageDraw.Draw(sheet)
        for idx, path in enumerate(subset):
            image = Image.open(path).convert("RGB")
            image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            thumb = ImageOps.expand(image, border=1, fill=(190, 190, 190))
            row, col = divmod(idx, cols)
            x = col * cell_w + pad
            y = row * cell_h + pad
            draw.text((x, y), f"p.{start + idx + 1:03d}", fill=(50, 50, 50))
            sheet.paste(thumb, (x, y + label_h))
        out = args.render_dir / f"contact-sheet-{sheet_idx}.png"
        sheet.save(out)
        print(out)


if __name__ == "__main__":
    main()
