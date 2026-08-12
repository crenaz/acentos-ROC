# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 crenaz

from __future__ import annotations

import argparse
from pathlib import Path

from acentos_ocr.config import resolve_tessdata_dir
from acentos_ocr.core.pipelines import build_default_pipeline, build_pipeline
from acentos_ocr.filters.registry import describe_filters
from acentos_ocr.ocr.tesseract_wrapper import TesseractWrapper
from acentos_ocr.utils.image_io import load_image
from acentos_ocr.utils.text_io import TEXT_SUFFIX, resolve_text_paths, save_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OCR on an image using the preprocessing pipeline.",
        # The filter table lives in the epilog because only description and epilog
        # escape argparse's re-wrapping; as help text its columns collapse.
        epilog="filters available to --pipeline:\n" + describe_filters(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "images",
        type=str,
        nargs="+",
        metavar="IMAGE",
        help="Path to the input image. Several may be given; each is OCR'd in turn.",
    )
    parser.add_argument(
        "--save-text",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            f"Write each picture's OCR text to DIR/<stem>{TEXT_SUFFIX} as UTF-8, "
            "creating DIR if needed. Note this is not the corpus convention: a "
            "hand-made transcription of IMG_1594.JPEG is text-of-IMG_1594.md, and "
            "that namespace is ground truth, so nothing here will ever write it."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging and save intermediate images.",
    )
    parser.add_argument(
        "--out-debug-dir",
        type=str,
        default="debug",
        help="Directory to save debug images when --debug is enabled.",
    )
    parser.add_argument(
        "--tesseract-cmd",
        type=str,
        default=None,
        help="Optional path to the tesseract binary (e.g. /usr/bin/tesseract).",
    )
    parser.add_argument(
        "--deskew",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Correct page skew before OCR. On by default: worth 5.1 points of "
            "character error rate across the corpus. Pass --no-deskew to disable "
            "it, which helps on the minority of pages that are already straight."
        ),
    )
    parser.add_argument(
        "--reading-order",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Rebuild reading order from word geometry when the page has columns. "
            "On by default, worth 18.9%% -> 12.0%% character error rate across the "
            "corpus. Costs no extra OCR, and single-column pages are left alone."
        ),
    )
    parser.add_argument(
        "--pipeline",
        nargs="+",
        metavar="FILTER",
        default=None,
        help=(
            "Compose the preprocessing stack explicitly, e.g. --pipeline "
            "grayscale clahe blur:ksize=3. Each entry is a filter name with "
            "optional key=value arguments; see the list below. Overrides --deskew."
        ),
    )
    parser.add_argument(
        "--tessdata-dir",
        type=str,
        default=None,
        help=(
            "Directory containing .traineddata files. Defaults to the project-local "
            "tessdata/ if populated (see scripts/fetch_tessdata.sh), otherwise falls "
            "back to Tesseract's system-wide lookup."
        ),
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=3,
        help=(
            "Tesseract page segmentation mode (PSM). Defaults to 3 (fully "
            # Literal percent signs must be doubled: argparse runs help text
            # through %-formatting, and "8.9% c" parses as a %c conversion.
            "automatic). Across the 15-image Cayman corpus, psm 3 measured "
            "18.9%% character error rate against psm 6's 22.3%%."
        ),
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="spa+eng",
        help=(
            "Tesseract language code (e.g. 'eng', 'spa', or 'spa+eng'). "
            "Defaults to 'spa+eng', which measured better than either alone "
            "on both sample images. Requires the tesseract-ocr-spa package."
        ),
    )
    parser.add_argument(
        "--oem",
        type=int,
        default=3,
        choices=(1, 3),
        help=(
            "Tesseract OCR Engine Mode (1=LSTM only, 3=default). "
            "Modes 0 and 2 require the legacy engine, which Tesseract 5 "
            "no longer ships."
        ),
    )

    args = parser.parse_args()

    images = [Path(path) for path in args.images]
    debug_dir = Path(args.out_debug_dir) if args.debug else None
    if args.debug:
        print(f"Pipeline steps (images written to {debug_dir}/):")

    # Both the names and the destination are settled before any OCR runs. A clash
    # discovered twelve pages in would already have overwritten a page's text, and
    # an unwritable directory is worth knowing about before a minute of Tesseract.
    text_paths: list[Path | None] = [None] * len(images)
    if args.save_text:
        try:
            text_paths = resolve_text_paths(images, args.save_text)
            Path(args.save_text).mkdir(parents=True, exist_ok=True)
        except (ValueError, OSError) as error:
            parser.error(str(error))

    if args.pipeline:
        try:
            pipeline = build_pipeline(args.pipeline, debug=args.debug, debug_dir=debug_dir)
        except ValueError as error:
            parser.error(str(error))   # a typo in a spec is user error, not a crash
    else:
        pipeline = build_default_pipeline(
            debug=args.debug, debug_dir=debug_dir, deskew=args.deskew
        )

    tessdata_dir = resolve_tessdata_dir(args.tessdata_dir)
    if args.debug:
        source = "system" if tessdata_dir is None else str(tessdata_dir)
        print(f"Using tessdata: {source}")

    config = f"--oem {args.oem} --psm {args.psm}"
    ocr = TesseractWrapper(
        tesseract_cmd=args.tesseract_cmd,
        lang=args.lang,
        tessdata_dir=tessdata_dir,
        reading_order=args.reading_order,
    )

    confidences: list[float] = []
    failures: list[Path] = []
    for position, (img_path, text_path) in enumerate(zip(images, text_paths), start=1):
        if len(images) > 1:
            print(f"\n=== [{position}/{len(images)}] {img_path.name} ===")

        try:
            image = load_image(img_path)
            processed = pipeline.run(image, name=img_path.stem)
            result = ocr.process_image(processed, custom_config=config)
        except Exception as error:
            # A batch must not lose the pages it already transcribed to one
            # unreadable file or one Tesseract failure. Report it, carry on, and
            # exit non-zero below so the run is not mistaken for a clean one.
            print(f"FAILED {img_path}: {error}")
            failures.append(img_path)
            continue

        print("=== OCR TEXT ===")
        print(result.text)
        print("\n=== SUMMARY ===")
        print(f"Average confidence: {result.confidence:.2f}%")
        print(f"Config used: {result.config_used}")
        if result.confidence < 70:
            print("Tip: If quality is low, try --psm 3 or --psm 4, or --oem 1 to force the LSTM engine.")

        confidences.append(result.confidence)
        if text_path is not None:
            save_text(text_path, result.text)
            print(f"Wrote {text_path}")

    if len(images) > 1:
        mean = sum(confidences) / len(confidences) if confidences else 0.0
        print("\n=== BATCH ===")
        print(
            f"{len(confidences)} of {len(images)} images transcribed, "
            f"mean confidence {mean:.2f}%"
        )
        for failed in failures:
            print(f"  failed: {failed}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

