from __future__ import annotations

from pathlib import Path

from ..filters.deskew import DeskewFilter
from ..filters.gaussian_blur import GaussianBlurFilter
from ..filters.grayscale import GrayscaleFilter
from .processor import PreprocessingPipeline


def build_default_pipeline(
    debug: bool = False,
    debug_dir: str | Path | None = None,
    deskew: bool = True,
) -> PreprocessingPipeline:
    """
    Grayscale, deskew, and a light blur. Tesseract 5's LSTM engine binarises
    internally and does it better than a hand-tuned threshold, so the pipeline
    deliberately stops short of that -- see the README baselines.

    Deskew is on by default as of 2026-08-07, worth 5.1 points of character error
    rate across the 15-image corpus (24.0% -> 18.9% at psm 3). It was opt-in
    before that on the strength of two samples that disagreed; the corpus settled
    it. Four images improve substantially, two regress modestly, and on the
    remaining eight the estimator finds no confident angle and declines to act --
    which is what keeps the downside small.

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
