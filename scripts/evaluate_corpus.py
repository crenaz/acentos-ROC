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

from acentos_ocr.config import resolve_tessdata_dir
from acentos_ocr.core.pipelines import build_default_pipeline
from acentos_ocr.eval import CORPUS_ENV_VAR, Sample, default_root, discover
from acentos_ocr.eval.metrics import edit_counts, normalise, word_miss_counts
from acentos_ocr.ocr.tesseract_wrapper import TesseractWrapper
from acentos_ocr.utils.image_io import load_image

#: A sample whose CER exceeds its word miss rate by more than this is being hurt
#: mainly by reading order rather than by recognition. Chosen to sit well clear of
#: the few points of slack that punctuation and spacing differences always produce.
ORDER_DAMAGE_THRESHOLD = 0.10


@dataclass(frozen=True)
class Config:
    psm: int
    deskew: bool

    @property
    def label(self) -> str:
        return f"psm {self.psm}" + (" +deskew" if self.deskew else "")


def _evaluate(job: dict) -> list[dict]:
    """
    Run one image through one pipeline, scoring every requested PSM.

    The unit of work is (image, deskew) rather than (image, deskew, psm) so that
    loading and preprocessing a 16 MP photo is paid once and amortised over the
    PSM sweep. Runs in a worker process, so everything it needs arrives as plain
    data.
    """
    truth = normalise(Path(job["truth"]).read_text(encoding="utf-8"))
    image = load_image(job["image"])
    processed = build_default_pipeline(deskew=job["deskew"]).run(image)

    ocr = TesseractWrapper(lang=job["lang"], tessdata_dir=job["tessdata_dir"])

    rows = []
    for psm in job["psms"]:
        started = time.perf_counter()
        result = ocr.process_image(
            processed, custom_config=f"--oem {job['oem']} --psm {psm}"
        )
        elapsed = time.perf_counter() - started

        hypothesis = normalise(result.text)
        distance, truth_chars = edit_counts(truth, hypothesis)
        missed, truth_words = word_miss_counts(truth, hypothesis)

        rows.append(
            {
                "stem": Path(job["image"]).stem,
                "psm": psm,
                "deskew": job["deskew"],
                "distance": distance,
                "truth_chars": truth_chars,
                "missed": missed,
                "truth_words": truth_words,
                "confidence": result.confidence,
                "seconds": elapsed,
            }
        )
    return rows


def _rate(rows: list[dict], numerator: str, denominator: str) -> float:
    total = sum(row[denominator] for row in rows)
    return sum(row[numerator] for row in rows) / total if total else 0.0


def _print_summary(rows: list[dict], configs: list[Config]) -> None:
    print(f"\n  {'configuration':<16} {'CER':>8} {'word miss':>11} "
          f"{'confidence':>12} {'time':>8}")
    print("  " + "-" * 58)

    ranked = sorted(
        configs,
        key=lambda c: _rate(
            [r for r in rows if r["psm"] == c.psm and r["deskew"] == c.deskew],
            "distance", "truth_chars",
        ),
    )
    for config in ranked:
        subset = [r for r in rows if r["psm"] == config.psm and r["deskew"] == config.deskew]
        if not subset:
            continue
        confidence = sum(r["confidence"] for r in subset) / len(subset)
        print(
            f"  {config.label:<16} "
            f"{_rate(subset, 'distance', 'truth_chars'):>7.1%} "
            f"{_rate(subset, 'missed', 'truth_words'):>10.1%} "
            f"{confidence:>11.1f}% "
            f"{sum(r['seconds'] for r in subset):>7.0f}s"
        )


def _print_per_image(rows: list[dict], configs: list[Config], stems: list[str]) -> None:
    print("\n  Per-image CER (* marks the best configuration for that image)\n")
    header = "  " + f"{'image':<12}" + "".join(f"{c.label:>16}" for c in configs)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for stem in stems:
        cells = {}
        for config in configs:
            match = [
                r for r in rows
                if r["stem"] == stem and r["psm"] == config.psm and r["deskew"] == config.deskew
            ]
            cells[config] = _rate(match, "distance", "truth_chars") if match else None

        scored = {c: v for c, v in cells.items() if v is not None}
        best = min(scored, key=scored.get) if scored else None
        line = f"  {stem:<12}"
        for config in configs:
            value = cells[config]
            text = "-" if value is None else f"{value:.1%}{'*' if config is best else ''}"
            line += f"{text:>16}"
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
    print(f"  {'image':<12} {'best config':<16} {'CER':>8} {'word miss':>11} {'gap':>8}")
    print("  " + "-" * 58)
    for stem, cer, miss, best in sorted(flagged, key=lambda f: f[2] - f[1]):
        label = Config(best["psm"], best["deskew"]).label
        print(f"  {stem:<12} {label:<16} {cer:>7.1%} {miss:>10.1%} {cer - miss:>7.1%}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=None,
                        help=f"Corpus root (default: ${CORPUS_ENV_VAR}).")
    parser.add_argument("--psm", type=int, nargs="+", default=[3, 4, 6],
                        help="Page segmentation modes to compare.")
    parser.add_argument("--deskew", choices=("off", "on", "both"), default="both",
                        help="Whether to sweep the deskew filter.")
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

    deskews = {"off": [False], "on": [True], "both": [False, True]}[args.deskew]
    configs = [Config(psm, deskew) for deskew in deskews for psm in args.psm]
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
            "deskew": deskew,
            "psms": args.psm,
            "lang": args.lang,
            "oem": args.oem,
            "tessdata_dir": str(tessdata_dir) if tessdata_dir else None,
        }
        for sample in samples
        for deskew in deskews
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
                  f"{' +deskew' if job['deskew'] else ''}")

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
