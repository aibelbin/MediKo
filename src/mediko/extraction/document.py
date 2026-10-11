"""Document input: embedded PDF text, plain text, and PaddleOCR fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mediko.extraction.pdf_text import extract_pdf_page_texts
from mediko.extraction.selection import pages_needing_ocr
from mediko.extraction.validate import read_txt, validate_input


def extract_pages(
    path: Path,
    *,
    max_upload_bytes: int,
    min_embedded_chars: int,
    ocr_engine: Any,
    render_dpi: int,
) -> list[dict[str, Any]]:
    kind = validate_input(path, max_upload_bytes=max_upload_bytes)
    if kind == "txt":
        return [
            {
                "page_index": 0,
                "text_source": "plain_text",
                "text": read_txt(path),
                "lines": [],
            }
        ]

    embedded = extract_pdf_page_texts(path)
    ocr_indexes = pages_needing_ocr(embedded, min_embedded_chars)
    recognized = {}
    if ocr_indexes:
        recognized = ocr_engine.recognize_pdf_pages(
            path,
            ocr_indexes,
            render_dpi=render_dpi,
        )

    pages: list[dict[str, Any]] = []
    for index, text in enumerate(embedded):
        if index in recognized:
            pages.append(recognized[index])
        else:
            pages.append(
                {
                    "page_index": index,
                    "text_source": "embedded_text",
                    "text": text,
                    "lines": [],
                }
            )
    return pages
