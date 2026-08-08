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
import sys
from pathlib import Path

from acentos_ocr.config import resolve_tessdata_dir
from acentos_ocr.core.pipelines import build_default_pipeline
from acentos_ocr.eval.metrics import cer, normalise, word_miss_rate
from acentos_ocr.ocr.tesseract_wrapper import TesseractWrapper
from acentos_ocr.utils.image_io import load_image


def default_truth_path(image: Path) -> Path:
    """Apply the corpus convention: <base>/text-of-<stem>.md, two levels up."""
    return image.resolve().parents[2] / f"text-of-{image.stem}.md"


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
    parser.add_argument("--deskew", action=argparse.BooleanOptionalAction, default=True,
                        help="Match the default pipeline's deskew setting (on).")
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
    print(f"  {'psm':>4} {'CER':>8} {'word miss':>11} {'confidence':>12}")
    print("  " + "-" * 38)

    processed = build_default_pipeline(deskew=args.deskew).run(image)
    for psm in args.psm:
        result = ocr.process_image(processed, custom_config=f"--oem {args.oem} --psm {psm}")
        hypothesis = normalise(result.text)
        print(f"  {psm:>4} {cer(truth, hypothesis):>7.1%} "
              f"{word_miss_rate(truth, hypothesis):>10.1%} {result.confidence:>11.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
