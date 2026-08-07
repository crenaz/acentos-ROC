from __future__ import annotations

import argparse
from pathlib import Path

from acentos_ocr.config import resolve_tessdata_dir
from acentos_ocr.core.pipelines import build_default_pipeline
from acentos_ocr.ocr.tesseract_wrapper import TesseractWrapper
from acentos_ocr.utils.image_io import load_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR on an image using the preprocessing pipeline.")
    parser.add_argument("image", type=str, help="Path to the input image.")
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
        action="store_true",
        help=(
            "Correct page skew before OCR. Worth it for handheld photos of tilted "
            "pages; measurably harmful on pages that are already straight."
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
            "automatic). On the Cayman clipping sample, psm 3 measured 8.9% "
            "character error rate against psm 6's 16.2%."
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

    img_path = Path(args.image)
    image = load_image(img_path)

    debug_dir = Path(args.out_debug_dir) if args.debug else None
    if args.debug:
        print(f"Pipeline steps (images written to {debug_dir}/):")

    pipeline = build_default_pipeline(
        debug=args.debug, debug_dir=debug_dir, deskew=args.deskew
    )
    processed = pipeline.run(image, name=img_path.stem)

    tessdata_dir = resolve_tessdata_dir(args.tessdata_dir)
    if args.debug:
        source = "system" if tessdata_dir is None else str(tessdata_dir)
        print(f"Using tessdata: {source}")

    config = f"--oem {args.oem} --psm {args.psm}"
    ocr = TesseractWrapper(
        tesseract_cmd=args.tesseract_cmd,
        lang=args.lang,
        tessdata_dir=tessdata_dir,
    )
    result = ocr.process_image(processed, custom_config=config)

    print("=== OCR TEXT ===")
    print(result.text)
    print("\n=== SUMMARY ===")
    print(f"Average confidence: {result.confidence:.2f}%")
    print(f"Config used: {result.config_used}")
    if result.confidence < 70:
        print("Tip: If quality is low, try --psm 3 or --psm 4, or --oem 1 to force the LSTM engine.")


if __name__ == "__main__":
    main()

