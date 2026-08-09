#!/usr/bin/env python3
"""
Evaluate the pipeline across a whole transcribed corpus.

`evaluate_cer.py` answers "how did this one image do?". This answers "did the
change help?", which needs every image, because single samples disagree: the two
transcriptions that existed before this script came along pointed opposite ways
on whether --deskew is worth enabling by default.

Two metrics are reported side by side, and the gap between them is the point:

    CER          order-sensitive -- what you actually get out of the pipeline.
    word miss    order-insensitive -- what the recogniser managed to read.

A high CER with a low word miss rate means the words were recognised and then
emitted in the wrong order, which is a page-segmentation failure and needs a
completely different fix from a blurry photo.

Corpus layout: an image `IMG_1594.JPEG` anywhere under the root is paired with
`text-of-IMG_1594.md`, also anywhere under the root.

Usage:
    export ACENTOS_CORPUS="/path/to/Cayman Job Clippings"
    uv run python scripts/evaluate_corpus.py
    uv run python scripts/evaluate_corpus.py --psm 3 4 6 --deskew both --per-image
    uv run python scripts/evaluate_corpus.py \
        --pipeline 'grayscale deskew blur:ksize=3' \
        --pipeline 'grayscale deskew clahe blur:ksize=3'
    uv run python scripts/evaluate_corpus.py --json results/baseline.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2

from acentos_ocr.config import resolve_tessdata_dir
from acentos_ocr.core.pipelines import (
    DEFAULT_SPEC,
    NO_DESKEW_SPEC,
    build_pipeline,
)
from acentos_ocr.filters.registry import describe_filters
from acentos_ocr.eval import CORPUS_ENV_VAR, Sample, default_root, discover
from acentos_ocr.eval.metrics import edit_counts, normalise, word_miss_counts
from acentos_ocr.layout.reading_order import reorder
from acentos_ocr.ocr.tesseract_wrapper import TesseractWrapper
from acentos_ocr.utils.image_io import load_image

#: A sample whose CER exceeds its word miss rate by more than this is being hurt
#: mainly by reading order rather than by recognition. Chosen to sit well clear of
#: the few points of slack that punctuation and spacing differences always produce.
ORDER_DAMAGE_THRESHOLD = 0.10


@dataclass(frozen=True)
class Config:
    psm: int
    pipeline: str
    reading_order: bool = False
    scale: float = 1.0

    @property
    def label(self) -> str:
        return (
            f"psm {self.psm} [{self.pipeline}]"
            + (" +order" if self.reading_order else "")
            + ("" if self.scale == 1.0 else f" x{self.scale:g}")
        )


def _evaluate(job: dict) -> list[dict]:
    """
    Run one image through one pipeline, scoring every requested PSM.

    The unit of work is (image, pipeline) rather than (image, pipeline, psm) so that
    loading and preprocessing a 16 MP photo is paid once and amortised over the
    PSM sweep. Runs in a worker process, so everything it needs arrives as plain
    data.
    """
    truth = normalise(Path(job["truth"]).read_text(encoding="utf-8"))
    image = load_image(job["image"])
    processed = build_pipeline(job["pipeline"].split()).run(image)

    ocr = TesseractWrapper(
        lang=job["lang"], tessdata_dir=job["tessdata_dir"], reading_order=False
    )

    rows = []
    for scale in job["scales"]:
        scaled = processed if scale == 1.0 else cv2.resize(
            processed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        for psm in job["psms"]:
            started = time.perf_counter()
            result = ocr.process_image(
                scaled, custom_config=f"--oem {job['oem']} --psm {psm}"
            )
            elapsed = time.perf_counter() - started

            # Geometric reordering reuses the word boxes this pass already
            # returned, so both orderings are scored from one OCR run.
            candidates = {False: result.text}
            if job["reading_order"]:
                candidates[True] = reorder(result.metadata) or result.text

            for reading_order, text in candidates.items():
                hypothesis = normalise(text)
                distance, truth_chars = edit_counts(truth, hypothesis)
                missed, truth_words = word_miss_counts(truth, hypothesis)

                rows.append(
                    {
                        "stem": Path(job["image"]).stem,
                        "psm": psm,
                        "pipeline": job["pipeline"],
                        "reading_order": reading_order,
                        "scale": scale,
                        "distance": distance,
                        "truth_chars": truth_chars,
                        "missed": missed,
                        "truth_words": truth_words,
                        "confidence": result.confidence,
                        "seconds": elapsed,
                    }
                )
    return rows


def _select(rows: list[dict], config: Config) -> list[dict]:
    return [
        r for r in rows
        if r["psm"] == config.psm
        and r["pipeline"] == config.pipeline
        and r["reading_order"] == config.reading_order
        and r["scale"] == config.scale
    ]


def _rate(rows: list[dict], numerator: str, denominator: str) -> float:
    total = sum(row[denominator] for row in rows)
    return sum(row[numerator] for row in rows) / total if total else 0.0


def _print_summary(rows: list[dict], configs: list[Config]) -> None:
    # Pipeline specs make labels long and variable; size the column to fit.
    width = max([len(c.label) for c in configs] + [len("configuration")])
    print(f"\n  {'configuration':<{width}} {'CER':>8} {'word miss':>11} "
          f"{'confidence':>12} {'time':>8}")
    print("  " + "-" * (width + 42))

    ranked = sorted(
        configs,
        key=lambda c: _rate(_select(rows, c), "distance", "truth_chars"),
    )
    for config in ranked:
        subset = _select(rows, config)
        if not subset:
            continue
        confidence = sum(r["confidence"] for r in subset) / len(subset)
        print(
            f"  {config.label:<{width}} "
            f"{_rate(subset, 'distance', 'truth_chars'):>7.1%} "
            f"{_rate(subset, 'missed', 'truth_words'):>10.1%} "
            f"{confidence:>11.1f}% "
            f"{sum(r['seconds'] for r in subset):>7.0f}s"
        )


def _print_per_image(rows: list[dict], configs: list[Config], stems: list[str]) -> None:
    print("\n  Per-image CER (* marks the best configuration for that image)\n")
    # Long pipeline labels would make a wide matrix unreadable, so the columns are
    # numbered and the key printed above them.
    for index, config in enumerate(configs, start=1):
        print(f"    [{index}] {config.label}")
    width = max(10, len(str(len(configs))) + 8)
    header = "\n  " + f"{'image':<12}" + "".join(
        f"{f'[{i}]':>{width}}" for i in range(1, len(configs) + 1))
    print(header)
    print("  " + "-" * (len(header) - 3))

    for stem in stems:
        cells = {}
        for config in configs:
            match = [r for r in _select(rows, config) if r["stem"] == stem]
            cells[config] = _rate(match, "distance", "truth_chars") if match else None

        scored = {c: v for c, v in cells.items() if v is not None}
        best = min(scored, key=scored.get) if scored else None
        line = f"  {stem:<12}"
        for config in configs:
            value = cells[config]
            text = "-" if value is None else f"{value:.1%}{'*' if config is best else ''}"
            line += f"{text:>{width}}"
        print(line)


def _print_order_damage(rows: list[dict], stems: list[str]) -> None:
    """Flag samples whose best result is limited by reading order, not recognition."""
    flagged = []
    for stem in stems:
        subset = [r for r in rows if r["stem"] == stem]
        if not subset:
            continue
        best = min(subset, key=lambda r: r["distance"] / max(1, r["truth_chars"]))
        cer = best["distance"] / max(1, best["truth_chars"])
        miss = best["missed"] / max(1, best["truth_words"])
        if cer - miss > ORDER_DAMAGE_THRESHOLD:
            flagged.append((stem, cer, miss, best))

    if not flagged:
        print("\n  Reading order: no sample is dominated by ordering damage.")
        return

    print("\n  Reading-order damage -- words recognised but emitted out of sequence.")
    print("  These need better page segmentation, not better image preprocessing.\n")
    width = max([len(Config(f[3]["psm"], f[3]["pipeline"], f[3]["reading_order"],
                            f[3]["scale"]).label) for f in flagged] + [11])
    print(f"  {'image':<12} {'best config':<{width}} {'CER':>8} {'word miss':>11} {'gap':>8}")
    print("  " + "-" * (width + 45))
    for stem, cer, miss, best in sorted(flagged, key=lambda f: f[2] - f[1]):
        label = Config(best["psm"], best["pipeline"], best["reading_order"],
                       best["scale"]).label
        print(f"  {stem:<12} {label:<{width}} {cer:>7.1%} {miss:>10.1%} {cer - miss:>7.1%}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="filters available to --pipeline:\n" + describe_filters(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=None,
                        help=f"Corpus root (default: ${CORPUS_ENV_VAR}).")
    parser.add_argument("--psm", type=int, nargs="+", default=[3, 4, 6],
                        help="Page segmentation modes to compare.")
    parser.add_argument("--deskew", choices=("off", "on", "both"), default="both",
                        help="Sweep the default stack with and without deskew. "
                             "Shorthand for the two --pipeline specs it expands to; "
                             "ignored when --pipeline is given.")
    parser.add_argument("--pipeline", action="append", metavar="SPEC", default=None,
                        help="A preprocessing stack to evaluate, as a quoted list of "
                             "filter specs, e.g. --pipeline 'grayscale deskew "
                             "blur:ksize=3'. Repeat to A/B several stacks in one run. "
                             "Replaces the --deskew sweep.")
    parser.add_argument("--scale", type=float, nargs="+", default=[1.0],
                        help="Upscale factors to sweep. Tesseract struggles below "
                             "roughly 30px of text height.")
    parser.add_argument("--reading-order", choices=("off", "on", "both"), default="both",
                        help="Whether to sweep geometric reading-order reconstruction. "
                             "Costs no extra OCR: it reuses the word boxes.")
    parser.add_argument("--lang", default="eng",
                        help="Tesseract language code. The Cayman corpus is English.")
    parser.add_argument("--oem", type=int, default=3, choices=(1, 3))
    parser.add_argument("--tessdata-dir", default=None)
    parser.add_argument("--jobs", type=int, default=4,
                        help="Worker processes. Each holds a full-size image, so "
                             "raising this on a memory-tight machine will thrash.")
    parser.add_argument("--per-image", action="store_true",
                        help="Print the full per-image CER matrix.")
    parser.add_argument("--json", type=Path, default=None,
                        help="Write raw per-row results here for later comparison.")
    args = parser.parse_args()

    root = args.root or default_root()
    if root is None:
        parser.error(
            f"No corpus root. Pass --root or set {CORPUS_ENV_VAR}, e.g.\n"
            f"    export {CORPUS_ENV_VAR}='/path/to/Cayman Job Clippings'"
        )

    samples, unmatched = discover(root)
    if not samples:
        parser.error(f"No image/transcription pairs found under {root}")

    choice = {"off": [False], "on": [True], "both": [False, True]}
    if args.pipeline:
        pipelines = args.pipeline
    else:
        pipelines = [" ".join(spec) for spec in
                     {"off": [NO_DESKEW_SPEC], "on": [DEFAULT_SPEC],
                      "both": [NO_DESKEW_SPEC, DEFAULT_SPEC]}[args.deskew]]
    try:
        for spec in pipelines:
            build_pipeline(spec.split())
    except ValueError as error:
        parser.error(str(error))
    orders = choice[args.reading_order]
    configs = [
        Config(psm, pipeline, order, scale)
        for pipeline in pipelines
        for order in orders
        for scale in args.scale
        for psm in args.psm
    ]
    tessdata_dir = resolve_tessdata_dir(args.tessdata_dir)

    print(f"corpus:   {root}")
    print(f"samples:  {len(samples)}")
    if unmatched:
        print(f"untranscribed ({len(unmatched)}): "
              f"{', '.join(p.name for p in unmatched)}")
    print(f"tessdata: {tessdata_dir or 'system'}   lang: {args.lang}")
    print(f"running:  {len(samples) * len(configs)} OCR passes "
          f"on {args.jobs} workers\n")

    jobs = [
        {
            "image": str(sample.image),
            "truth": str(sample.truth),
            "pipeline": pipeline,
            "psms": args.psm,
            "scales": args.scale,
            "reading_order": True in orders,
            "lang": args.lang,
            "oem": args.oem,
            "tessdata_dir": str(tessdata_dir) if tessdata_dir else None,
        }
        for sample in samples
        for pipeline in pipelines
    ]

    rows: list[dict] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_evaluate, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                rows.extend(future.result())
            except Exception as error:  # one bad image must not lose the whole run
                print(f"  [{done}/{len(jobs)}] FAILED {Path(job['image']).stem}: {error}")
                continue
            print(f"  [{done}/{len(jobs)}] {Path(job['image']).stem}"
                  f"  [{job['pipeline']}]")

    stems = [sample.stem for sample in samples]
    _print_summary(rows, configs)
    _print_order_damage(rows, stems)
    if args.per_image:
        _print_per_image(rows, configs, stems)
    print(f"\n  wall clock: {time.perf_counter() - started:.0f}s")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"  wrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
