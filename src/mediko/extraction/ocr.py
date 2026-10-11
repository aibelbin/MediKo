"""PaddleOCR engine.

Model weights load on the first scanned page. Recognition scores are model
confidence values, not calibrated clinical probabilities.

A page is OCR'd only when embedded text is below the configured character
threshold. A digital page that also contains a scanned figure is kept as
embedded text; this boilerplate does not separate mixed layouts.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger("mediko.ocr")


class OcrError(RuntimeError):
    """PaddleOCR could not produce a line result for a page."""


class PaddleOcrEngine:
    def __init__(self, *, lang: str, device: str) -> None:
        self.lang = lang
        self.device = device
        self.model_version: str | None = None
        self._ocr: Any = None
        self._lock = threading.Lock()

    def recognize_pdf_pages(
        self,
        pdf_path: Path,
        page_indexes: list[int],
        *,
        render_dpi: int,
    ) -> dict[int, dict[str, Any]]:
        if render_dpi <= 0:
            raise OcrError("render_dpi must be greater than zero.")
        self._ensure_loaded()
        import pypdfium2 as pdfium

        scale = render_dpi / 72
        recognized: dict[int, dict[str, Any]] = {}
        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            page_count = len(pdf)
            for index in page_indexes:
                if index < 0 or index >= page_count:
                    raise OcrError(f"Page {index} is outside this PDF.")
                logger.info("Running PaddleOCR on page %s", index + 1)
                image = _render_page(pdf, index, scale)
                recognized[index] = self._recognize_array(image, index)
        finally:
            _close(pdf)
        return recognized

    def _ensure_loaded(self) -> None:
        if self._ocr is not None:
            return
        with self._lock:
            if self._ocr is None:
                self._ocr = _build_paddle_ocr(self.lang, self.device)
                self.model_version = _paddleocr_version()

    def _recognize_array(self, image: Any, page_index: int) -> dict[str, Any]:
        predict = getattr(self._ocr, "predict", None)
        if predict is None:
            raise OcrError(
                "This PaddleOCR build has no predict() method. Install PaddleOCR 3.x."
            )
        with self._lock:
            raw = predict(image)
        results = list(raw)
        if len(results) != 1:
            raise OcrError(
                f"PaddleOCR returned {len(results)} results for page {page_index + 1}."
            )
        lines, text = assemble_lines(parse_ocr_result(results[0]))
        return {
            "page_index": page_index,
            "text_source": "ocr",
            "text": text,
            "lines": lines,
        }


def parse_ocr_result(result: Any) -> dict[str, Any]:
    """Unwrap a PaddleOCR 3.x result down to the mapping that holds rec_texts."""
    data: Any = result
    if not isinstance(data, Mapping) and hasattr(result, "json"):
        data = result.json
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as exc:
            raise OcrError("PaddleOCR returned JSON that could not be parsed.") from exc
    if isinstance(data, Mapping) and "rec_texts" not in data:
        inner = data.get("res")
        if isinstance(inner, Mapping):
            data = inner
    if not isinstance(data, Mapping) or "rec_texts" not in data:
        raise OcrError("PaddleOCR returned a result without rec_texts.")
    return dict(data)


def assemble_lines(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Turn parallel PaddleOCR arrays into reading-order lines and page text."""
    texts = [_text(item) for item in _sequence(payload.get("rec_texts"))]
    count = len(texts)
    scores = _aligned(_sequence(payload.get("rec_scores")), count, "rec_scores")
    polys = _aligned(_sequence(payload.get("rec_polys")), count, "rec_polys")
    boxes = _aligned(_sequence(payload.get("rec_boxes")), count, "rec_boxes")

    pending: list[tuple[float, float, int, str, float | None, list[list[float]]]] = []
    for index, text in enumerate(texts):
        points = _box_points(polys[index], boxes[index])
        top, left = _top_left(points)
        pending.append((top, left, index, text, _score(scores[index]), points))
    pending.sort(key=lambda item: (item[0], item[1], item[2]))

    lines: list[dict[str, Any]] = []
    parts: list[str] = []
    cursor = 0
    for _, _, _, text, score, points in pending:
        start = cursor
        end = start + len(text)
        lines.append(
            {
                "text": text,
                "model_confidence": score,
                "box": points,
                "start_char": start,
                "end_char": end,
            }
        )
        parts.append(text)
        cursor = end + 1
    return lines, "\n".join(parts)


def _build_paddle_ocr(lang: str, device: str) -> Any:
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise OcrError(
            "PaddleOCR is not installed. Run this pipeline in the MediKo container on Fox."
        ) from exc

    attempts = [
        {
            "lang": lang,
            "device": device,
            "engine": "paddle",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        },
        {
            "lang": lang,
            "device": device,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        },
        {
            "lang": lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        },
        {
            "engine": "paddle",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        },
    ]
    last_type_error: TypeError | None = None
    for kwargs in attempts:
        try:
            return PaddleOCR(**kwargs)
        except TypeError as exc:
            last_type_error = exc
            logger.info("PaddleOCR constructor rejected %s", sorted(kwargs))
            continue
    raise OcrError(
        "PaddleOCR 3.x could not be constructed with the supported arguments."
    ) from last_type_error


def _paddleocr_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return f"paddleocr=={version('paddleocr')}"
    except PackageNotFoundError:
        return "paddleocr"


def _render_page(pdf: Any, index: int, scale: float) -> Any:
    import numpy as np

    page = pdf[index]
    bitmap = page.render(scale=scale)
    try:
        image = bitmap.to_pil().convert("RGB")
    finally:
        _close(bitmap)
        _close(page)
    return np.ascontiguousarray(np.asarray(image))


def _close(obj: Any) -> None:
    close = getattr(obj, "close", None)
    if callable(close):
        close()


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _aligned(items: list[Any], count: int, field: str) -> list[Any]:
    if not items:
        return [None] * count
    if len(items) != count:
        raise OcrError(
            f"PaddleOCR {field} has {len(items)} entries for {count} text lines."
        )
    return items


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _box_points(poly: Any, box: Any) -> list[list[float]]:
    raw = poly if poly is not None else box
    if raw is None:
        return []
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if (
        isinstance(raw, (list, tuple))
        and len(raw) == 4
        and all(_is_number(item) for item in raw)
    ):
        x1, y1, x2, y2 = (float(item) for item in raw)
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    points: list[list[float]] = []
    for point in raw:
        if hasattr(point, "tolist"):
            point = point.tolist()
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return []
        if not _is_number(point[0]) or not _is_number(point[1]):
            return []
        points.append([float(point[0]), float(point[1])])
    return points


def _top_left(points: list[list[float]]) -> tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    return (min(point[1] for point in points), min(point[0] for point in points))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
