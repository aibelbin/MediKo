"""Choose which PDF pages lack enough embedded text to skip OCR."""

from __future__ import annotations


def pages_needing_ocr(page_texts: list[str], min_embedded_chars: int) -> list[int]:
    return [
        index
        for index, text in enumerate(page_texts)
        if len(text.strip()) < min_embedded_chars
    ]
