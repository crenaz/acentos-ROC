# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 crenaz

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from acentos_ocr.config import resolve_tessdata_dir
from acentos_ocr.core.pipelines import build_default_pipeline, build_pipeline
from acentos_ocr.filters.registry import describe_filters
from acentos_ocr.ocr.tesseract_wrapper import TesseractWrapper
from acentos_ocr.utils.image_io import load_image
from acentos_ocr.utils.pdf_io import (
    MIN_TEXT_LAYER_CHARS,
    extract_text_layer,
    is_pdf,
    page_count,
    render_page,
)
from acentos_ocr.utils.text_io import TEXT_SUFFIX, resolve_text_paths, save_text


@dataclass(frozen=True)
class WorkItem:
    """
    One unit of work: a page to be OCR'd, or text a PDF already carried.

    An image is one item; a PDF is one item per page. Expanding inputs into these
    before anything runs is what lets the rest of main() stay unaware of PDFs --
    output naming, clash detection, per-item failure handling and the batch summary
    all operate on items and never ask where the pixels came from.
    """

    label: str                                  # what to print for this item
    name: Path                                  # drives --save-text naming
    source: Path                                # the file the user actually named
    load: Callable[[], Any] | None = None       # renders/reads the image, lazily
    text: str | None = None                     # set when a text layer supplied it


def expand_inputs(
    paths: Sequence[Path], *, pdf_dpi: int | None, force_ocr: bool
) -> list[WorkItem]:
    """
    Flatten the command line into per-page work items.

    PDF pages that carry a real text layer are taken as-is: those characters are
    exact, and rasterising them so Tesseract can guess at them again would only
    introduce errors. `force_ocr` overrides that, which matters when a file has a
    bad text layer left behind by some other tool.

    Page images are produced by the `load` callable rather than up front, so a long
    document is rendered a page at a time instead of all at once.
    """
    items: list[WorkItem] = []
    for path in paths:
        if not is_pdf(path):
            items.append(
                WorkItem(
                    label=path.name,
                    name=path,
                    source=path,
                    load=partial(load_image, path),
                )
            )
            continue

        texts = [""] * page_count(path) if force_ocr else extract_text_layer(path)
        digits = max(2, len(str(len(texts))))
        for number, text in enumerate(texts, start=1):
            # A virtual path, never written. It exists so page items get distinct
            # stems and flow through resolve_text_paths like any other input.
            virtual = path.with_name(f"{path.stem}-p{number:0{digits}d}{path.suffix}")
            label = f"{path.name} page {number}"
            if len(text) >= MIN_TEXT_LAYER_CHARS:
                items.append(
                    WorkItem(label=label, name=virtual, source=path, text=text)
                )
            else:
                items.append(
                    WorkItem(
                        label=label,
                        name=virtual,
                        source=path,
                        load=partial(render_page, path, number, pdf_dpi),
                    )
                )

    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OCR on an image using the preprocessing pipeline.",
        # The filter table lives in the epilog because only description and epilog
        # escape argparse's re-wrapping; as help text its columns collapse.
        epilog="filters available to --pipeline:\n" + describe_filters(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        type=str,
        nargs="+",
        metavar="INPUT",
        help=(
            "Path to an image or a PDF. Several may be given; each is processed in "
            "turn, and a PDF contributes one page at a time."
        ),
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
        "--pdf-dpi",
        type=int,
        default=None,
        metavar="DPI",
        help=(
            "Resolution to render PDF pages at. Defaults to each page's own native "
            "resolution, which is what a scan actually contains -- rendering above "
            "it interpolates, costing time and gaining no detail. Phone scan-to-PDF "
            "makes this especially worth leaving alone: it sizes the page box to the "
            "pixels at 72 DPI, so an ordinary sheet can claim to be 28 inches wide."
        ),
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help=(
            "OCR every PDF page even when it already carries a text layer. By "
            "default that text is used as-is, because it is exact and OCR could "
            "only introduce errors. Use this when another tool has left a bad text "
            "layer behind."
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

    try:
        items = expand_inputs(
            [Path(path) for path in args.inputs],
            pdf_dpi=args.pdf_dpi,
            force_ocr=args.force_ocr,
        )
    except Exception as error:
        # An unreadable or encrypted PDF is user error, not a crash, and it is
        # worth failing on before any of the batch has run.
        parser.error(f"could not read input: {error}")

    debug_dir = Path(args.out_debug_dir) if args.debug else None
    if args.debug:
        print(f"Pipeline steps (images written to {debug_dir}/):")

    # Both the names and the destination are settled before any OCR runs. A clash
    # discovered twelve pages in would already have overwritten a page's text, and
    # an unwritable directory is worth knowing about before a minute of Tesseract.
    text_paths: list[Path | None] = [None] * len(items)
    if args.save_text:
        try:
            text_paths = resolve_text_paths([item.name for item in items], args.save_text)
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
    failures: list[str] = []
    for position, (item, text_path) in enumerate(zip(items, text_paths), start=1):
        if len(items) > 1:
            print(f"\n=== [{position}/{len(items)}] {item.label} ===")

        try:
            if item.text is not None:
                text, summary = item.text, "text layer (no OCR)"
            else:
                image = item.load()
                processed = pipeline.run(image, name=item.name.stem)
                result = ocr.process_image(processed, custom_config=config)
                text, summary = result.text, result.config_used
                confidences.append(result.confidence)
        except Exception as error:
            # A batch must not lose the pages it already transcribed to one
            # unreadable file or one Tesseract failure. Report it, carry on, and
            # exit non-zero below so the run is not mistaken for a clean one.
            print(f"FAILED {item.label}: {error}")
            failures.append(item.label)
            continue

        print("=== OCR TEXT ===")
        print(text)
        print("\n=== SUMMARY ===")
        if item.text is None:
            print(f"Average confidence: {confidences[-1]:.2f}%")
        else:
            print("Source: embedded text layer, extracted exactly")
        print(f"Config used: {summary}")
        if item.text is None and confidences[-1] < 70:
            print("Tip: If quality is low, try --psm 3 or --psm 4, or --oem 1 to force the LSTM engine.")

        if text_path is not None:
            save_text(text_path, text)
            print(f"Wrote {text_path}")

    if len(items) > 1:
        transcribed = len(items) - len(failures)
        print("\n=== BATCH ===")
        summary = f"{transcribed} of {len(items)} pages transcribed"
        if confidences:
            mean = sum(confidences) / len(confidences)
            summary += f", mean confidence {mean:.2f}% over {len(confidences)} OCR'd"
        print(summary)
        for failed in failures:
            print(f"  failed: {failed}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

