"""Conservative text cleanup after extraction.

The original page text stays in place. Clinical abbreviations are not expanded,
because a short form such as "MS" can mean more than one thing. Line-break
hyphenation is joined only when both sides look like pieces of a longer word.
"""

from __future__ import annotations

import re
from typing import Any, Callable

NORMALIZATION_DETAIL = (
    "Whitespace, line breaks, ligatures, curly quotes, and conservative "
    "line-break hyphenation were normalized. Clinical abbreviations were "
    "left unchanged. Original page text is preserved."
)

_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}
_QUOTES = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
}

_NEWLINES = re.compile(r"\r\n|\r")
_UNICODE_SPACES = re.compile(
    "[\u00a0\u2000-\u200a\u202f\u205f\u3000]"
)
_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\ufeff\u00ad]")
_LIGATURE_PATTERN = re.compile("|".join(map(re.escape, _LIGATURES)))
_QUOTE_PATTERN = re.compile("|".join(map(re.escape, _QUOTES)))
_TABS = re.compile(r"\t")
_INTERNAL_SPACE = re.compile(r"[ \t]{2,}")
_TRAILING_SPACE = re.compile(r"[ \t]+(?=\n|$)")
_LEADING_SPACE = re.compile(r"(?m)^[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_LINEBREAK_HYPHEN = re.compile(r"[A-Za-z]{3,}-\n[a-z]{3,}")


def normalize_text(text: str) -> dict[str, Any]:
    """Return cleaned text and the edits that produced it."""
    cleaned = text
    transformations: list[dict[str, Any]] = []
    for rule, pattern, replace in _RULES:
        cleaned, edits = _replace(cleaned, pattern, replace, rule)
        transformations.extend(edits)
    return {
        "text": cleaned,
        "transformations": transformations,
    }


def normalize_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach normalized text to each page without changing its extracted text."""
    normalized_pages: list[dict[str, Any]] = []
    for page in pages:
        outcome = normalize_text(page["text"])
        updated = dict(page)
        updated["normalized_text"] = outcome["text"]
        updated["transformations"] = [
            {**edit, "page_index": page["page_index"]}
            for edit in outcome["transformations"]
        ]
        normalized_pages.append(updated)
    return normalized_pages


def _replace(
    text: str,
    pattern: re.Pattern[str],
    replace: Callable[[re.Match[str]], str],
    rule: str,
) -> tuple[str, list[dict[str, Any]]]:
    edits: list[dict[str, Any]] = []
    pieces: list[str] = []
    cursor = 0
    for match in pattern.finditer(text):
        original = match.group(0)
        replacement = replace(match)
        pieces.append(text[cursor : match.start()])
        pieces.append(replacement)
        if replacement != original:
            edits.append(
                {
                    "rule": rule,
                    "start_char": match.start(),
                    "end_char": match.end(),
                    "original": original,
                    "replacement": replacement,
                }
            )
        cursor = match.end()
    pieces.append(text[cursor:])
    return "".join(pieces), edits


def _literal_map(table: dict[str, str]) -> Callable[[re.Match[str]], str]:
    def replace(match: re.Match[str]) -> str:
        return table[match.group(0)]

    return replace


_RULES: list[tuple[str, re.Pattern[str], Callable[[re.Match[str]], str]]] = [
    ("newlines", _NEWLINES, lambda _match: "\n"),
    ("unicode_space", _UNICODE_SPACES, lambda _match: " "),
    ("zero_width", _ZERO_WIDTH, lambda _match: ""),
    ("ligature", _LIGATURE_PATTERN, _literal_map(_LIGATURES)),
    ("curly_quote", _QUOTE_PATTERN, _literal_map(_QUOTES)),
    ("tab", _TABS, lambda _match: " "),
    ("internal_space", _INTERNAL_SPACE, lambda _match: " "),
    ("trailing_space", _TRAILING_SPACE, lambda _match: ""),
    ("leading_space", _LEADING_SPACE, lambda _match: ""),
    ("blank_lines", _BLANK_LINES, lambda _match: "\n\n"),
    ("linebreak_hyphen", _LINEBREAK_HYPHEN, lambda match: match.group(0).replace("-\n", "")),
]
