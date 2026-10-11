"""File type and size checks for uploaded reports."""

from __future__ import annotations

from pathlib import Path


class InputError(ValueError):
    """The file cannot enter the pipeline."""


def validate_input(path: Path, *, max_upload_bytes: int) -> str:
    """Return ``pdf`` or ``txt`` when the file is an allowed report."""
    if not path.is_file():
        raise InputError(f"File not found: {path}")
    size = path.stat().st_size
    if size == 0:
        raise InputError("The file is empty.")
    if size > max_upload_bytes:
        raise InputError(
            f"The file is {size} bytes, above the limit of {max_upload_bytes} bytes."
        )
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".txt":
        return "txt"
    raise InputError("Only PDF and TXT files are supported.")


def read_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise InputError("TXT files must be UTF-8.") from exc
