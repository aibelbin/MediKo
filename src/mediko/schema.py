"""Versioned document result. Empty analysis fields mean the stage did not run."""

from __future__ import annotations

from typing import Any

from mediko.normalization import NORMALIZATION_DETAIL

SCHEMA_VERSION = "0.2.0"

RESEARCH_NOTICE = (
    "This tool extracts information documented in the report. "
    "It does not diagnose conditions, recommend treatment, or support medical decisions."
)

_UNAVAILABLE_DETAIL = "Not implemented in this boilerplate."
_UNAVAILABLE_STAGES = ("ner", "negation", "relations", "temporal")


def unavailable_components() -> dict[str, dict[str, str]]:
    components = {
        "normalization": {
            "status": "unavailable",
            "detail": _UNAVAILABLE_DETAIL,
        }
    }
    components.update(
        {
            name: {"status": "unavailable", "detail": _UNAVAILABLE_DETAIL}
            for name in _UNAVAILABLE_STAGES
        }
    )
    return components


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
    normalized_values = [page.get("normalized_text") for page in pages]
    normalization_applied = bool(pages) and all(
        value is not None for value in normalized_values
    )
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
        "normalization_applied": normalization_applied,
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
    components = unavailable_components()
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "document_id": document_id,
        "source_filename": source_filename,
        "notice": RESEARCH_NOTICE,
        "text": text,
        "pages": pages,
        "entities": [],
        "relations": [],
        "timeline": [],
        "components": components,
        "processing_metadata": metadata,
    }
    if normalization_applied:
        components["normalization"] = {
            "status": "applied",
            "detail": NORMALIZATION_DETAIL,
        }
        document["normalized_text"] = "\n\n".join(normalized_values)
    return document
