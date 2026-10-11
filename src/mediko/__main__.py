"""Command-line entry point for a single PDF or TXT report."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from mediko.extraction.ocr import OcrError
from mediko.extraction.validate import InputError
from mediko.pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract text from a medical report PDF or TXT file."
    )
    parser.add_argument("path", type=Path, help="PDF or UTF-8 TXT report")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write the JSON result to this path. Prints JSON when omitted.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        result = run_pipeline(args.path)
    except (InputError, OcrError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    payload = json.dumps(result, indent=2)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
