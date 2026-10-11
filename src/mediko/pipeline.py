"""Sequential report pipeline. Later stages stay explicitly unavailable."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from mediko.config import Settings, load_settings
from mediko.extraction.document import extract_pages
from mediko.extraction.ocr import PaddleOcrEngine
from mediko.normalization import normalize_pages
from mediko.schema import assemble_document

logger = logging.getLogger("mediko.pipeline")


def run_pipeline(
    path: Path,
    *,
    source_filename: str | None = None,
    settings: Settings | None = None,
    engine: PaddleOcrEngine | None = None,
) -> dict:
    settings = settings or load_settings()
    engine = engine or PaddleOcrEngine(lang=settings.ocr_lang, device=settings.ocr_device)
    pages = extract_pages(
        path,
        max_upload_bytes=settings.max_upload_bytes,
        min_embedded_chars=settings.min_embedded_chars,
        ocr_engine=engine,
        render_dpi=settings.render_dpi,
    )
    pages = normalize_pages(pages)
    ocr_used = any(page["text_source"] == "ocr" for page in pages)
    logger.info(
        "Extracted %s page(s) from %s; OCR used: %s; normalization changes: %s",
        len(pages),
        source_filename or path.name,
        ocr_used,
        sum(len(page["transformations"]) for page in pages),
    )
    return assemble_document(
        document_id=str(uuid.uuid4()),
        source_filename=source_filename or path.name,
        pages=pages,
        ocr_used=ocr_used,
        model_version=engine.model_version,
        ocr_device=settings.ocr_device,
    )
