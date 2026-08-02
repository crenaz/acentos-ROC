#!/usr/bin/env python3
"""
Measure character error rate (CER) of the pipeline against a manual transcription.

Tesseract's average confidence is its own self-assessment, not accuracy -- a model
can be confidently wrong, and at small margins confidence cannot separate two
configurations. CER against a hand-typed transcription is the metric that actually
settles whether a change helped.

Transcription convention (Cayman corpus):

    <base>/Raw-Photos-Of-Cayman-Job-Listings/<Month><Day>/IMG_1594.JPEG
    <base>/text-of-IMG_1594.md

so the transcription for an image is `text-of-<stem>.md`, two directories up from
the image. Override with --truth for anything that does not follow that layout.

Usage:
    uv run python scripts/evaluate_cer.py path/to/IMG_1594.JPEG
    uv run python scripts/evaluate_cer.py IMG_1594.JPEG --truth notes.md --psm 3 4 6
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from acentos_ocr.ocr.tesseract_wrapper import TesseractWrapper
from acentos_ocr.utils.image_io import load_image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import build_default_pipeline, resolve_tessdata_dir  # noqa: E402


def default_truth_path(image: Path) -> Path:
    """Apply the corpus convention: <base>/text-of-<stem>.md, two levels up."""
    return image.resolve().parents[2] / f"text-of-{image.stem}.md"


def normalise(text: str) -> str:
    """
    Strip markdown scaffolding and collapse whitespace.

    The transcriptions are written as Markdown, the OCR output is plain text, so
    comparing them raw would score formatting rather than recognition.
    """
    text = re.sub(r"^#+\s*", "", text, flags=re.M)       # headings
    text = re.sub(r"^[.·*-]\s*", "", text, flags=re.M)  # bullets, incl. U+00B7
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b))
            )
        previous = current
    return previous[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="Image to evaluate.")
    parser.add_argument("--truth", type=Path, default=None,
                        help="Transcription file (default: the corpus convention).")
    parser.add_argument("--lang", default="eng", help="Tesseract language code.")
    parser.add_argument("--psm", type=int, nargs="+", default=[3],
                        help="One or more page segmentation modes to compare.")
    parser.add_argument("--oem", type=int, default=3, choices=(1, 3))
    parser.add_argument("--tessdata-dir", default=None)
    args = parser.parse_args()

    truth_path = args.truth or default_truth_path(args.image)
    if not truth_path.is_file():
        parser.error(f"No transcription found at {truth_path}. Pass --truth explicitly.")

    truth = normalise(truth_path.read_text(encoding="utf-8"))
    image = load_image(args.image)
    ocr = TesseractWrapper(
        lang=args.lang,
        tessdata_dir=resolve_tessdata_dir(args.tessdata_dir),
    )

    print(f"image:  {args.image}")
    print(f"truth:  {truth_path}  ({len(truth)} chars normalised)\n")
    print(f"  {'psm':>4} {'CER':>8} {'confidence':>12}")
    print("  " + "-" * 26)

    for psm in args.psm:
        processed = build_default_pipeline().run(image)
        result = ocr.process_image(processed, custom_config=f"--oem {args.oem} --psm {psm}")
        cer = levenshtein(truth, normalise(result.text)) / len(truth)
        print(f"  {psm:>4} {cer:>7.1%} {result.confidence:>11.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
