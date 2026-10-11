"""Sync this repo to Fox and build the container there."""

from __future__ import annotations

import json

import pytest

from mediko.extraction.ocr import OcrError, assemble_lines, parse_ocr_result
from mediko.extraction.selection import pages_needing_ocr
from mediko.extraction.validate import InputError, validate_input
from mediko.schema import SCHEMA_VERSION, assemble_document


def test_short_pdf_page_is_selected_for_ocr() -> None:
    assert pages_needing_ocr(["  \n", "a" * 40, "short"], 40) == [0, 2]


def test_line_offsets_follow_visual_reading_order() -> None:
    lines, text = assemble_lines(
        {
            "rec_texts": ["second", "first"],
            "rec_scores": [0.5, 0.9],
            "rec_boxes": [[0, 20, 10, 30], [0, 0, 10, 10]],
        }
    )
    assert text == "first\nsecond"
    assert lines[0]["text"] == "first"
    assert lines[0]["start_char"] == 0
    assert lines[0]["end_char"] == 5
    assert lines[0]["model_confidence"] == 0.9
    assert lines[1]["start_char"] == 6
    assert lines[1]["box"][0] == [0.0, 20.0]


def test_parse_unwraps_paddleocr_res_envelope() -> None:
    payload = parse_ocr_result(
        {"res": {"rec_texts": ["fever"], "rec_scores": [0.91]}}
    )
    lines, text = assemble_lines(payload)
    assert text == "fever"
    assert lines[0]["model_confidence"] == 0.91
    assert lines[0]["box"] == []


def test_missing_recognition_text_is_an_error() -> None:
    with pytest.raises(OcrError):
        parse_ocr_result({"res": {"rec_scores": [0.2]}})


def test_mismatched_score_count_is_an_error() -> None:
    with pytest.raises(OcrError):
        assemble_lines({"rec_texts": ["a", "b"], "rec_scores": [0.4]})


def test_missing_scores_stay_null() -> None:
    lines, _text = assemble_lines({"rec_texts": ["a"]})
    assert lines[0]["model_confidence"] is None


def test_rejects_unsupported_file_type(tmp_path) -> None:
    path = tmp_path / "report.doc"
    path.write_bytes(b"not-a-report")
    with pytest.raises(InputError):
        validate_input(path, max_upload_bytes=1000)


def test_document_json_does_not_invent_analysis() -> None:
    document = assemble_document(
        document_id="doc-1",
        source_filename="note.txt",
        pages=[
            {
                "page_index": 0,
                "text_source": "plain_text",
                "text": "Patient has persistent fever.",
                "lines": [],
            }
        ],
        ocr_used=False,
        model_version=None,
        ocr_device="cpu",
    )
    assert document["schema_version"] == SCHEMA_VERSION
    assert document["entities"] == []
    assert document["relations"] == []
    assert document["timeline"] == []
    assert document["components"]["ner"]["status"] == "unavailable"
    assert document["processing_metadata"]["ocr_used"] is False
    assert document["processing_metadata"]["model_version"] is None
    json.dumps(document)
