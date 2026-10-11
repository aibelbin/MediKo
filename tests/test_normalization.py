"""Normalization keeps the source text and records every edit."""

from __future__ import annotations

from mediko.normalization import normalize_pages, normalize_text
from mediko.schema import assemble_document


def test_cleanup_is_recorded_and_reversible_by_inspection() -> None:
    outcome = normalize_text("Persist-\nent  ﬁnding\r\nwas\u00a0present.\n\n\n")
    assert outcome["text"] == "Persistent finding\nwas present.\n\n"
    rules = [edit["rule"] for edit in outcome["transformations"]]
    assert "linebreak_hyphen" in rules
    assert "ligature" in rules
    assert "newlines" in rules
    assert "unicode_space" in rules
    assert "internal_space" in rules
    assert "blank_lines" in rules


def test_clinical_abbreviations_and_real_hyphens_stay() -> None:
    text = "History of MS. Give co-\namoxiclav. See follow-\nup."
    assert normalize_text(text)["text"] == text
    assert normalize_text(text)["transformations"] == []


def test_second_pass_makes_no_further_changes() -> None:
    once = normalize_text("medi-\ncation\t\tdose")
    twice = normalize_text(once["text"])
    assert once["text"] == "medication dose"
    assert twice["transformations"] == []
    assert twice["text"] == once["text"]


def test_page_keeps_original_text() -> None:
    pages = normalize_pages(
        [
            {
                "page_index": 0,
                "text_source": "plain_text",
                "text": "fever  today",
                "lines": [],
            }
        ]
    )
    assert pages[0]["text"] == "fever  today"
    assert pages[0]["normalized_text"] == "fever today"
    assert pages[0]["transformations"][0]["rule"] == "internal_space"
    assert pages[0]["transformations"][0]["page_index"] == 0


def test_document_marks_normalization_applied() -> None:
    pages = normalize_pages(
        [
            {
                "page_index": 0,
                "text_source": "plain_text",
                "text": "Patient has persistent fever.",
                "lines": [],
            }
        ]
    )
    document = assemble_document(
        document_id="doc-1",
        source_filename="note.txt",
        pages=pages,
        ocr_used=False,
        model_version=None,
        ocr_device=None,
    )
    assert document["text"] == "Patient has persistent fever."
    assert document["normalized_text"] == document["text"]
    assert document["components"]["normalization"]["status"] == "applied"
    assert document["processing_metadata"]["normalization_applied"] is True
    assert document["components"]["ner"]["status"] == "unavailable"
