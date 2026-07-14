#!/usr/bin/env python3
"""Merge thesis/zotero_import/bucket_*.bib into thesis/references.bib (dedupe by cite key)."""

from __future__ import annotations

import re
from pathlib import Path

THESIS_DIR = Path(__file__).resolve().parents[2] / "thesis"
IMPORT_DIR = THESIS_DIR / "zotero_import"
OUTPUT = THESIS_DIR / "references.bib"


def split_entries(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for chunk in text.split("@"):
        chunk = chunk.strip()
        if not chunk:
            continue
        full = "@" + chunk
        if not full.rstrip().endswith("}"):
            continue
        key_match = re.match(r"\w+\{([^,]+),", chunk)
        if not key_match:
            continue
        entries[key_match.group(1).strip()] = full
    return entries


def main() -> None:
    merged: dict[str, str] = {}
    for path in sorted(IMPORT_DIR.glob("bucket_*.bib")):
        merged.update(split_entries(path.read_text(encoding="utf-8")))

    header = (
        "% Merged bibliography for Factory6G thesis.\n"
        "% Zotero: import bucket_*.bib into collections A–E, then auto-export here via Better BibTeX.\n"
        "% Regenerate: python scripts/tools/merge_thesis_bibliography.py\n\n"
    )
    body = "\n\n".join(merged.values()) + "\n"
    OUTPUT.write_text(header + body, encoding="utf-8")
    print(f"Wrote {len(merged)} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
