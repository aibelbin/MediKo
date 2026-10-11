"""Embedded text extraction for digital PDFs. Scanned pages return little or no text."""

from __future__ import annotations

from pathlib import Path

from mediko.extraction.validate import InputError


def extract_pdf_page_texts(path: Path) -> list[str]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(str(path))
    except PdfReadError as exc:
        raise InputError(f"The PDF could not be read: {exc}") from exc
    if reader.is_encrypted:
        raise InputError("This PDF is encrypted and cannot be read.")
    if not reader.pages:
        raise InputError("The PDF has no pages.")
    return [(page.extract_text() or "") for page in reader.pages]
