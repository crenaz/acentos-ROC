from __future__ import annotations

from pathlib import Path

from ..filters.deskew import DeskewFilter
from ..filters.gaussian_blur import GaussianBlurFilter
from ..filters.grayscale import GrayscaleFilter
from .processor import PreprocessingPipeline


def build_default_pipeline(
    debug: bool = False,
    debug_dir: str | Path | None = None,
    deskew: bool = False,
) -> PreprocessingPipeline:
    """
    Grayscale plus a light blur. Tesseract 5's LSTM engine binarises internally and
    does it better than a hand-tuned threshold, so the pipeline deliberately stops
    short of that -- see the README baselines.

    Deskew is opt-in: it rescues a genuinely skewed page (45.3% -> 7.3% CER on a
    sample rotated 4 degrees) but is not free on one that is already straight.

    Lives in the package rather than in main.py so that the evaluation harness --
    which builds pipelines inside worker processes -- can import it without
    reaching back into the CLI entry point.
    """
    pipeline = PreprocessingPipeline(debug=debug, debug_dir=debug_dir)
    pipeline.add_filter(GrayscaleFilter())
    if deskew:
        pipeline.add_filter(DeskewFilter())
    pipeline.add_filter(GaussianBlurFilter(ksize=3))
    return pipeline
