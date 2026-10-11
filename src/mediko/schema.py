"""Versioned document result. Empty analysis fields mean the stage did not run."""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "0.1.0"

RESEARCH_NOTICE = (
    "This tool extracts information documented in the report. "
    "It does not diagnose conditions, recommend treatment, or support medical decisions."
)

_UNAVAILABLE_DETAIL = "Not implemented in this boilerplate."
_UNAVAILABLE_STAGES = ("normalization", "ner", "negation", "relations", "temporal")


def unavailable_components() -> dict[str, dict[str, str]]:
    return {
        name: {"status": "unavailable", "detail": _UNAVAILABLE_DETAIL}
        for name in _UNAVAILABLE_STAGES
    }


def assemble_document(
    *,
    document_id: str,
    source_filename: str,
    pages: list[dict[str, Any]],
    ocr_used: bool,
    model_version: str | None,
    ocr_device: str | None,
) -> dict[str, Any]:
    """Build the public JSON document from pages that were actually extracted."""
    text = "\n\n".join(page["text"] for page in pages)
    ocr_page_indexes = [
        page["page_index"] for page in pages if page["text_source"] == "ocr"
    ]
    embedded_page_indexes = [
        page["page_index"]
        for page in pages
        if page["text_source"] == "embedded_text"
    ]
    metadata: dict[str, Any] = {
        "ocr_used": ocr_used,
        "normalization_applied": False,
        "model_version": model_version if ocr_used else None,
        "ocr_device": ocr_device if ocr_used else None,
        "ocr_page_indexes": ocr_page_indexes,
        "embedded_page_indexes": embedded_page_indexes,
        "page_separator": "\n\n",
    }
    if ocr_used:
        metadata["ocr_confidence"] = (
            "Per-line model_confidence values are PaddleOCR recognition scores, "
            "not calibrated probabilities of clinical correctness."
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "document_id": document_id,
        "source_filename": source_filename,
        "notice": RESEARCH_NOTICE,
        "text": text,
        "pages": pages,
        "entities": [],
        "relations": [],
        "timeline": [],
        "components": unavailable_components(),
        "processing_metadata": metadata,
    }
