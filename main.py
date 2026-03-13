from __future__ import annotations

import argparse
from pathlib import Path

from src.core.processor import PreprocessingPipeline
from src.filters.gaussian_blur import GaussianBlurFilter
from src.filters.grayscale import GrayscaleFilter
from src.filters.morphology import MorphologyFilter
from src.filters.threshold import AdaptiveThresholdFilter
from src.ocr.tesseract_wrapper import TesseractWrapper
from src.utils.image_io import load_image, save_image


def build_default_pipeline(debug: bool = False) -> PreprocessingPipeline:
    pipeline = PreprocessingPipeline(debug=debug)
    pipeline.add_filter(GrayscaleFilter())
    pipeline.add_filter(GaussianBlurFilter(ksize=5))
    pipeline.add_filter(AdaptiveThresholdFilter(block_size=15, constant=5))
    pipeline.add_filter(MorphologyFilter(op="close", kernel_size=2))
    return pipeline


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
        "--psm",
        type=int,
        default=6,
        help="Tesseract page segmentation mode (PSM).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="eng",
        help="Tesseract language code (e.g. 'eng', 'spa', or 'spa+eng').",
    )
    parser.add_argument(
        "--oem",
        type=int,
        default=3,
        choices=(0, 1, 2, 3),
        help="Tesseract OCR Engine Mode (0=legacy, 1=LSTM, 2=legacy+LSTM, 3=default).",
    )

    args = parser.parse_args()

    img_path = Path(args.image)
    image = load_image(img_path)

    pipeline = build_default_pipeline(debug=args.debug)
    processed = pipeline.run(image)

    if args.debug:
        save_image(Path(args.out_debug_dir) / f"{img_path.stem}_processed.png", processed)

    config = f"--oem {args.oem} --psm {args.psm}"
    ocr = TesseractWrapper(tesseract_cmd=args.tesseract_cmd, lang=args.lang)
    result = ocr.process_image(processed, custom_config=config)

    print("=== OCR TEXT ===")
    print(result.text)
    print("\n=== SUMMARY ===")
    print(f"Average confidence: {result.confidence:.2f}%")
    print(f"Config used: {result.config_used}")
    if result.confidence < 70:
        print("Tip: If quality is low, try --psm 3 or --psm 4, or --oem 1 for legacy engine.")


if __name__ == "__main__":
    main()

