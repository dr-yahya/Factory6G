#!/usr/bin/env python3
"""Create a DOCX copy with Word field/bookmark markup removed."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import complete_from_v3 as base


W = base.W
NS = base.NS


def qn(tag: str) -> str:
    return base.qn(W, tag)


def strip_fields_and_bookmarks(root: ET.Element) -> None:
    for parent in root.iter():
        for child in list(parent):
            if child.tag in {qn("bookmarkStart"), qn("bookmarkEnd")}:
                parent.remove(child)
            elif child.tag == qn("r"):
                if child.find("./w:fldChar", NS) is not None or child.find("./w:instrText", NS) is not None:
                    parent.remove(child)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--tmp", type=Path, default=Path("/app/thesis_writing/.flatten_docx"))
    args = parser.parse_args()

    if args.tmp.exists():
        shutil.rmtree(args.tmp)
    args.tmp.mkdir(parents=True)
    with zipfile.ZipFile(args.src) as zin:
        zin.extractall(args.tmp)

    document_path = args.tmp / "word" / "document.xml"
    tree = ET.parse(document_path)
    root = tree.getroot()
    strip_fields_and_bookmarks(root)
    tree.write(document_path, encoding="utf-8", xml_declaration=True)

    if args.out.exists():
        args.out.unlink()
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zout:
        for path in sorted(args.tmp.rglob("*")):
            if path.is_file():
                zout.write(path, path.relative_to(args.tmp).as_posix())
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
