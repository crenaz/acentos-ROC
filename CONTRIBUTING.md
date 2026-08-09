# Contributing to acentos-ROC

Thanks for your interest. Bug reports, failing images, and transcriptions are all
welcome, and so are patches.

## Licensing of contributions — please read before opening a pull request

The project is released under **AGPL-3.0-only**, and the copyright is currently
held by a single author. That matters: a sole copyright holder can also offer the
same code under separate commercial terms to anyone who needs to build on it
without AGPL obligations. If patches arrive under AGPL alone, each contributor
holds copyright in their part, and every future commercial licence would require
tracking down and getting permission from all of them. In practice that ends the
option permanently.

So, by opening a pull request you confirm that:

1. You wrote the contribution, or otherwise have the right to submit it.
2. You license it under **AGPL-3.0-only**, like the rest of the project.
3. You **also grant the project owner a perpetual, worldwide, irrevocable,
   royalty-free right to relicense your contribution under other terms**,
   including proprietary or commercial ones.

You keep the copyright in your work. Point 3 grants an additional permission on
top of the AGPL; it does not take your rights away, and it does not stop you using
your own contribution however you like elsewhere.

If you would rather not grant point 3, that is completely reasonable — please open
an issue describing the change instead of a pull request, and it can be
reimplemented independently.

Add yourself to the SPDX header of any file you create:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) <year> <you>
```

## What is most useful

**Transcriptions.** Every accuracy claim in this project is measured against
hand-typed ground truth, and there is currently no Spanish in the corpus at all
despite Spanish accented text being the project's original target. A transcribed
page is worth more than most patches.

Convention: an image `IMG_1594.JPEG` anywhere under the corpus root pairs with
`text-of-IMG_1594.md`, also anywhere under the root. Transcribe what is printed;
if you annotate a graphic, keep the annotation distinguishable (see how `LOGO:`
labels are handled in `src/acentos_ocr/eval/metrics.py`).

**Photographs that fail.** Especially anything with perspective or curvature —
that is the known weak point, documented as finding #13.

## Before submitting a patch

```bash
uv sync
uv run pytest -q
```

If the change could affect OCR output — any filter, the pipeline, the wrapper, or
the layout code — **measure it against the corpus**, and put the numbers in the
pull request:

```bash
export ACENTOS_CORPUS="/path/to/corpus"
uv run python scripts/evaluate_corpus.py \
    --pipeline 'grayscale deskew blur:ksize=3' \
    --pipeline '<your stack>'
```

Two conventions that the project takes seriously, both learned the hard way:

- **A single image is not evidence.** Upscaling improved one sample by 9 points
  and made the corpus worse. Two samples once disagreed about whether `--deskew`
  helped, and were only settled by fifteen.
- **Tesseract's confidence is not accuracy.** It has risen while character error
  rate got worse. Use `scripts/evaluate_corpus.py`, not the confidence figure.

New filters need registering in `src/acentos_ocr/filters/registry.py` and a
behavioural test in `tests/test_filters.py`; see README section 8.
