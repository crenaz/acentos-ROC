from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..filters.registry import build_filter
from .processor import PreprocessingPipeline

#: The default stack, written in the same spec language `--pipeline` accepts, so
#: the documented default and the thing that actually runs cannot drift apart.
DEFAULT_SPEC = ("grayscale", "deskew", "blur:ksize=3")

#: The same stack with deskew removed, for `--no-deskew`.
NO_DESKEW_SPEC = ("grayscale", "blur:ksize=3")


def build_pipeline(
    specs: Iterable[str],
    debug: bool = False,
    debug_dir: str | Path | None = None,
) -> PreprocessingPipeline:
    """
    Build a pipeline from filter specs such as `("grayscale", "blur:ksize=3")`.

    This is what makes the Strategy pattern pay for itself. Every filter in the
    project was reachable only by editing `main.py` until now, so comparing two
    preprocessing stacks meant changing source between runs -- which is precisely
    the experiment the architecture was chosen to make easy. With specs, a stack is
    a string, and `scripts/evaluate_corpus.py` can sweep several against the corpus
    in one command.
    """
    pipeline = PreprocessingPipeline(debug=debug, debug_dir=debug_dir)
    for spec in specs:
        pipeline.add_filter(build_filter(spec))
    return pipeline


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
    return build_pipeline(
        DEFAULT_SPEC if deskew else NO_DESKEW_SPEC,
        debug=debug,
        debug_dir=debug_dir,
    )
